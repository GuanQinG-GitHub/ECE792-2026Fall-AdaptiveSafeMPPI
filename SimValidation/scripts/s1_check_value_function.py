"""Stage 1b: drop the robot at random collision-free poses in the forest, run the adaptive SF pipeline
(local occupancy -> SDF -> VAE latent -> value network) and inspect the zero-level set of the value function.

    python scripts/s1_check_value_function.py --world worlds/forest_seed0.json --n-stats 100 --device mps

Outputs in results/stage1/:
  vf_detail.png     per-pose panels (occupancy, SDF / VAE reconstruction, V slice at ego heading, V=0 for 4 headings)
  vf_world.png      zero-level sets mapped back into the world frame
  vf_stats.png      sanity statistics over n-stats poses
  vf_metrics.json   numbers behind the plots
No DP ground truth exists for these windows, so the checks are necessary conditions only:
  * V(x, y, th) <= SDF_true(x, y) - r   (the value at t = 0 can never exceed the current clearance)
  * "certain false-safe" = V > 0 at a state whose true clearance is already <= r
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon as MplPoly

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from simenv import ROBOT_RADIUS as R, World, sensing as S  # noqa: E402
from s1_generate_forest import draw_world  # noqa: E402

am = S.am
TH = am.theta_axis()
K0 = int(np.argmin(np.abs(TH)))                # theta = 0 slice = ego heading
KS = [int(np.argmin(np.abs(((TH - a) + np.pi) % (2 * np.pi) - np.pi))) for a in (0, np.pi / 2, np.pi, -np.pi / 2)]
EXT = [-2.5, 2.5, -2.5, 2.5]
AX = S.AX


def sample_poses(world, rng, n_random, n_challenge):
    """Random collision-free poses (clearance > r + 5 cm) and 'challenge' poses close to an obstacle, facing it."""
    poses, tags = [], []
    while len(poses) < n_random:
        p = np.r_[rng.uniform(3, world.size - 3, 2), rng.uniform(-np.pi, np.pi)]
        if world.sdf(p[None, :2])[0] > R + 0.05:
            poses.append(p); tags.append("random")
    while len(poses) < n_random + n_challenge:
        q = rng.uniform(3, world.size - 3, 2)
        d = world.sdf(q[None])[0]
        if not (R + 0.08 < d < R + 0.45):
            continue
        h = 1e-3
        g = np.array([world.sdf((q + [h, 0])[None])[0] - world.sdf((q - [h, 0])[None])[0],
                      world.sdf((q + [0, h])[None])[0] - world.sdf((q - [0, h])[None])[0]])
        psi = np.arctan2(-g[1], -g[0]) + rng.uniform(-0.4, 0.4)       # roughly facing the nearest obstacle
        poses.append(np.r_[q, psi]); tags.append("challenge")
    return np.array(poses), tags


def run_pose(world, pose, vae, net):
    t0 = time.time()
    occ = S.local_occupancy(world, pose)
    sdf_in = S.sdf_from_occupancy(occ)
    t_sense = time.time() - t0
    t1 = time.time()
    z = am.encode_sdf(vae, sdf_in)
    V = am.value_on_grid(net, z)
    t_vf = time.time() - t1
    sdf_dec = am.decode_latent(vae, z)[0].cpu().numpy()
    v_ego, g_ego = am.value_and_gradient(net, z, [[0.0, 0.0, 0.0]])
    sdf_true = S.exact_window_sdf(world, pose)
    return dict(occ=occ, sdf_in=sdf_in, sdf_true=sdf_true, sdf_dec=sdf_dec, V=V, z=z,
                v_ego=float(v_ego[0]), g_ego=g_ego[0], t_sense=t_sense, t_vf=t_vf)


def pose_metrics(r):
    V, st = r["V"], r["sdf_true"][..., None]
    bound = np.maximum(st - R, -0.25)                # V can never exceed SDF - r at t = 0 (labels clipped at -0.25)
    over = V - bound
    valid = st[..., 0] > 0.0                         # cells outside obstacles
    edge = np.abs(r["sdf_true"] - R) < 0.05
    clip = lambda a: np.clip(a, -1, 2)
    return dict(
        v_ego=r["v_ego"],
        safe_frac_theta0=float((V[:, :, K0] > 0).mean()),
        safe_frac_any_theta=float((V.max(2) > 0).mean()),
        safe_frac_all_states=float((V > 0).mean()),
        certain_false_safe_frac=float(((V > 0) & (st <= R)).mean()),
        overshoot_max_m=float(over.max()),
        overshoot_max_where_safe_m=float(over[V > 0].max()) if (V > 0).any() else 0.0,
        overshoot_frac_gt_2cm_where_safe=float((over[V > 0] > 0.02).mean()) if (V > 0).any() else 0.0,
        certain_false_safe_max_V_m=float(V[(V > 0) & (st <= R)].max()) if ((V > 0) & (st <= R)).any() else 0.0,
        overshoot_frac_gt_2cm=float((over > 0.02).mean()),
        vae_boundary_err_cm=float(100 * np.abs(clip(r["sdf_dec"]) - clip(r["sdf_true"]))[edge].mean()) if edge.any() else None,
        vae_false_free_frac=float(((r["sdf_dec"] > R) & (r["sdf_true"] <= R)).mean()),
        vae_false_occ_frac=float(((r["sdf_dec"] <= R) & (r["sdf_true"] > R))[valid].mean()),
        occ_sdf_vs_exact_mean_cm=float(100 * np.abs(clip(r["sdf_in"]) - clip(r["sdf_true"]))[valid].mean()),
        t_sense_s=r["t_sense"], t_vf_s=r["t_vf"])


def detail_figure(world, poses, tags, runs, path):
    n = len(runs)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.3 * n), constrained_layout=True)
    hc = ["#d6249f", "#1f77b4", "#2ca02c", "#ff7f0e"]
    for i, (p, tag, r) in enumerate(zip(poses, tags, runs)):
        a = axes[i, 0]
        a.imshow(r["occ"].T, origin="lower", extent=EXT, cmap=ListedColormap(["#fff", "#4a4a4a"]), vmin=0, vmax=1,
                 interpolation="nearest")
        a.contour(AX, AX, r["sdf_true"].T, levels=[R], colors="#999", linewidths=0.8, linestyles="--")
        a.plot(0, 0, "o", ms=6, mfc="#ffd400", mec="k")
        a.annotate("", xy=(0.5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.3))
        a.set_title(f"#{i} ({tag})  world pose ({p[0]:.1f}, {p[1]:.1f}, {np.rad2deg(p[2]):.0f}°)\n"
                    f"occupancy, dashed: true SDF = r", fontsize=10)
        a = axes[i, 1]
        im = a.imshow(np.clip(r["sdf_in"], -1, 2).T, origin="lower", extent=EXT, cmap="RdBu", vmin=-2, vmax=2)
        a.contour(AX, AX, r["sdf_true"].T, levels=[R], colors="k", linewidths=1.2)
        a.contour(AX, AX, r["sdf_dec"].T, levels=[R], colors="#d6249f", linewidths=1.2, linestyles="--")
        fig.colorbar(im, ax=a, shrink=0.8)
        a.set_title("VAE input SDF (m)\nblack: true SDF = r   magenta dashed: decoded SDF = r", fontsize=10)
        a = axes[i, 2]
        Vs = r["V"][:, :, K0]
        im = a.imshow(Vs.T, origin="lower", extent=EXT, cmap="RdBu", vmin=-0.6, vmax=0.6)
        a.contour(AX, AX, r["sdf_true"].T, levels=[0], colors="#d62728", linewidths=0.9)
        a.contour(AX, AX, r["sdf_true"].T, levels=[R], colors="#666", linewidths=0.7, linestyles="--")
        if (Vs > 0).any() and (Vs <= 0).any():
            a.contour(AX, AX, Vs.T, levels=[0], colors="#d6249f", linewidths=1.6)
        a.plot(0, 0, "o", ms=6, mfc="#ffd400", mec="k")
        a.annotate("", xy=(0.5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.3))
        fig.colorbar(im, ax=a, shrink=0.8)
        a.set_title(f"V(x, y, θ = ego heading) (m),  V(ego) = {r['v_ego']:+.3f}\n"
                    "magenta: V = 0   red: obstacle   grey dashed: SDF = r", fontsize=10)
        a = axes[i, 3]
        anyv = r["V"].max(2) > 0
        a.imshow(np.where(anyv, 1.0, np.nan).T, origin="lower", extent=EXT, cmap=ListedColormap(["#cfe8cf"]),
                 interpolation="nearest")
        a.contourf(AX, AX, r["sdf_true"].T, levels=[-10, 0], colors=["#4a4a4a"])
        for k, c in zip(KS, hc):
            Vk = r["V"][:, :, k]
            if (Vk > 0).any() and (Vk <= 0).any():
                a.contour(AX, AX, Vk.T, levels=[0], colors=c, linewidths=1.3)
            a.annotate("", xy=(0.35 * np.cos(TH[k]) + 1.9, 0.35 * np.sin(TH[k]) - 1.9), xytext=(1.9, -1.9),
                       arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
        a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5); a.set_aspect("equal")
        a.set_title("V = 0 for headings 0°, 90°, 180°, −90° (arrows, ego frame)\n"
                    "green fill: safe for some heading (max_θ V > 0)", fontsize=10)
        for a in axes[i]:
            a.set_xlabel("x_ego (m)"); a.set_ylabel("y_ego (m)")
    fig.savefig(path, dpi=72)
    plt.close(fig)


def world_figure(world, poses, runs, path, zoom_ids):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.6), constrained_layout=True)
    a = axes[0]
    a.set_facecolor("#f4f1e8")
    draw_world(a, world)
    v = np.array([r["v_ego"] for r in runs])
    sc = a.scatter(poses[:, 0], poses[:, 1], c=v, cmap="RdBu", vmin=-0.4, vmax=0.4, s=28, ec="k", lw=0.5, zorder=5)
    a.quiver(poses[:, 0], poses[:, 1], np.cos(poses[:, 2]), np.sin(poses[:, 2]), scale=40, width=0.002, zorder=6)
    fig.colorbar(sc, ax=a, shrink=0.75, label="V at the ego state (m)")
    a.set_title(f"{len(poses)} sampled robot poses coloured by V(ego)  (blue: safe, red: unsafe)", fontsize=11)
    lo = poses[zoom_ids, :2].min(0) - 3.5
    hi = poses[zoom_ids, :2].max(0) + 3.5
    a.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec="#d6249f", lw=1.6))
    a = axes[1]
    a.set_facecolor("#f4f1e8")
    draw_world(a, world)
    cols = plt.cm.tab10(np.arange(len(zoom_ids)) % 10)
    X, Y = np.meshgrid(AX, AX, indexing="ij")
    for c, i in zip(cols, zoom_ids):
        p, r = poses[i], runs[i]
        sq = S.ego_to_world(p, np.array([[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]]))
        a.add_patch(MplPoly(sq, closed=True, fill=False, ec=c, lw=1.0, ls="--"))
        W = S.ego_to_world(p, np.stack([X, Y], -1))
        Vs = r["V"][:, :, K0]
        a.contourf(W[..., 0], W[..., 1], Vs, levels=[0, 10], colors=[c], alpha=0.28)
        if (Vs > 0).any() and (Vs <= 0).any():
            a.contour(W[..., 0], W[..., 1], Vs, levels=[0], colors=[c], linewidths=1.4)
        a.plot(*p[:2], "o", ms=7, mfc=c, mec="k", zorder=6)
        a.annotate("", xy=p[:2] + 0.8 * np.array([np.cos(p[2]), np.sin(p[2])]), xytext=p[:2],
                   arrowprops=dict(arrowstyle="->", color="k", lw=1.4), zorder=7)
    a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1])
    a.set_title("Zoom: each robot's 5 m window (dashed) and estimated safe set {V > 0} at the robot's current heading,\n"
                "mapped to the world frame", fontsize=11)
    for a in axes:
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    fig.savefig(path, dpi=80)
    plt.close(fig)


def stats_figure(poses, tags, runs, mets, path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    v = np.array([m["v_ego"] for m in mets])
    clr = np.array([r["sdf_true"][49:51, 49:51].mean() - R for r in runs])  # clearance at the ego (centre cells)
    ch = np.array([t == "challenge" for t in tags])
    a = axes[0]
    a.hist([v[~ch], v[ch]], bins=30, stacked=True, color=["#1f77b4", "#ff7f0e"], label=["random", "challenge"])
    a.axvline(0, color="k", lw=1)
    a.set_xlabel("V(ego) (m)"); a.set_ylabel("poses"); a.legend()
    a.set_title(f"Value at the ego state; {100 * (v > 0).mean():.0f}% of poses safe", fontsize=10)
    a = axes[1]
    a.scatter(clr[~ch], v[~ch], s=14, label="random"); a.scatter(clr[ch], v[ch], s=14, label="challenge")
    lim = [min(clr.min(), v.min()) - 0.05, max(clr.max(), v.max()) + 0.05]
    a.plot(lim, lim, "k--", lw=1, label="V = SDF − r (upper bound)")
    a.axhline(0, color="#888", lw=0.7)
    a.set_xlabel("true clearance at ego, SDF − r (m)"); a.set_ylabel("V(ego) (m)"); a.legend(fontsize=8)
    a.set_title("V(ego) must lie below the dashed line", fontsize=10)
    a = axes[2]
    a.hist([m["safe_frac_theta0"] * 100 for m in mets], bins=25, alpha=0.7, label="θ = ego heading")
    a.hist([m["safe_frac_any_theta"] * 100 for m in mets], bins=25, alpha=0.7, label="some θ")
    a.set_xlabel("% of window cells with V > 0"); a.legend(fontsize=8)
    a.set_title("Size of the estimated safe set per window", fontsize=10)
    a = axes[3]
    a.hist([[m["overshoot_max_m"] * 100 for m in mets], [m["overshoot_max_where_safe_m"] * 100 for m in mets]],
           bins=30, color=["#c5b0d5", "#9467bd"], label=["all states", "states with V > 0"])
    a.axvline(0, color="k", lw=1)
    a.legend(fontsize=8)
    a.set_xlabel("max over window of V − max(SDF_true − r, −0.25)  (cm)")
    a.set_title("Worst violation of the t = 0 upper bound per window\n(> 0 means over-estimation somewhere)", fontsize=10)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default=os.path.join(ROOT, "worlds", "forest_seed0.json"))
    ap.add_argument("--n-stats", type=int, default=100)
    ap.add_argument("--n-detail", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "stage1")
    os.makedirs(out, exist_ok=True)
    world = World.load(a.world)
    vae, net = am.load_vae(device=a.device), am.load_value_net(device=a.device)
    rng = np.random.default_rng(a.seed)
    n_ch = int(round(0.3 * a.n_stats))
    poses, tags = sample_poses(world, rng, a.n_stats - n_ch, n_ch)
    runs, mets = [], []
    for p in poses:
        r = run_pose(world, p, vae, net)
        m = pose_metrics(r)
        m["ego_clearance_minus_r"] = float(world.sdf(p[None, :2])[0] - R)
        runs.append(r); mets.append(m)

    # detail: the first few random poses + the first few challenge poses
    nr = a.n_detail - a.n_detail // 3
    det = list(range(nr)) + list(range(a.n_stats - n_ch, a.n_stats - n_ch + a.n_detail // 3))
    detail_figure(world, poses[det], [tags[i] for i in det], [runs[i] for i in det], os.path.join(out, "vf_detail.png"))
    # zoom: a cluster of poses near the first detail pose
    d = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    zoom = list(np.argsort(d)[:6])
    world_figure(world, poses, runs, os.path.join(out, "vf_world.png"), zoom)
    stats_figure(poses, tags, runs, mets, os.path.join(out, "vf_stats.png"))

    agg = lambda k, sel=slice(None): float(np.mean([m[k] for m in np.array(mets, dtype=object)[sel]]))
    ch = np.array([t == "challenge" for t in tags])
    summary = dict(
        world=os.path.relpath(a.world, ROOT), device=a.device, n_poses=len(poses), n_challenge=int(ch.sum()),
        frac_ego_safe_random=float(np.mean([m["v_ego"] > 0 for m, c in zip(mets, ch) if not c])),
        frac_ego_safe_challenge=float(np.mean([m["v_ego"] > 0 for m, c in zip(mets, ch) if c])),
        mean_safe_frac_theta0=agg("safe_frac_theta0"), mean_safe_frac_any_theta=agg("safe_frac_any_theta"),
        mean_certain_false_safe_frac=agg("certain_false_safe_frac"),
        max_certain_false_safe_frac=float(max(m["certain_false_safe_frac"] for m in mets)),
        overshoot_max_m_over_all=float(max(m["overshoot_max_m"] for m in mets)),
        mean_overshoot_frac_gt_2cm=agg("overshoot_frac_gt_2cm"),
        overshoot_max_where_safe_m_over_all=float(max(m["overshoot_max_where_safe_m"] for m in mets)),
        mean_overshoot_frac_gt_2cm_where_safe=agg("overshoot_frac_gt_2cm_where_safe"),
        margin_to_remove_certain_false_safe_m=float(max(m["certain_false_safe_max_V_m"] for m in mets)),
        ego_bound_violations=int(sum(m["v_ego"] > m["ego_clearance_minus_r"] + 0.01 for m in mets)),
        mean_vae_boundary_err_cm=float(np.nanmean([m["vae_boundary_err_cm"] or np.nan for m in mets])),
        mean_vae_false_free_frac=agg("vae_false_free_frac"), mean_vae_false_occ_frac=agg("vae_false_occ_frac"),
        mean_occ_sdf_vs_exact_cm=agg("occ_sdf_vs_exact_mean_cm"),
        mean_t_sense_s=agg("t_sense_s"), mean_t_vf_full_grid_s=float(np.mean([m["t_vf_s"] for m in mets[1:]])),
    )
    print(json.dumps(summary, indent=1))
    with open(os.path.join(out, "vf_metrics.json"), "w") as f:
        json.dump(dict(summary=summary, detail_ids=[int(i) for i in det], zoom_ids=[int(i) for i in zoom],
                       poses=poses.tolist(), tags=tags, per_pose=mets), f, indent=1)


if __name__ == "__main__":
    main()
