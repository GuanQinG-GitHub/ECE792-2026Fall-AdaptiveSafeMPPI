"""Example: estimate the safety value function of the five provided environments and compare it with dynamic programming.

    python run_example.py                       # CPU, heading 0 deg, writes output/
    python run_example.py --heading-deg 90 --device cuda:0

For every environment in examples/:
  1. the SDF window is encoded to a 64-number latent with the VAE,
  2. the value network is evaluated on the whole 100 x 100 x 36 grid from that latent,
  3. the result is compared with the two DP references stored with the example
       value_dp          DP on the environment's own SDF            (end to end: includes the VAE error)
       value_dp_decoded  DP on the SDF decoded from the latent      (what the network was trained to reproduce)
Outputs: output/comparison_heading_XXXdeg.png, output/metrics.json, and a table on the terminal.
"""
import argparse
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import adaptivesf_models as am

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_CLIP = -0.25        # DP values are clipped below at this level (m)


def metrics(pred, ref):
    """Agreement between a predicted and a DP value grid. Distances in cm, rates in % of all states."""
    valid = ref > LABEL_CLIP + 1e-3
    edge = np.abs(ref) < 0.05
    err = pred - ref
    return dict(value_error_cm=float(100 * np.abs(err[valid]).mean()),
                boundary_error_cm=float(100 * np.abs(err[edge]).mean()),
                false_safe_pct=float(100 * ((pred > 0) & (ref <= 0)).mean()),
                false_unsafe_pct=float(100 * ((pred <= 0) & (ref > 0)).mean()),
                safe_pct_reference=float(100 * (ref > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--heading-deg", type=float, default=0.0, help="heading of the slice that is plotted")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vae = am.load_vae(device=a.device)
    net = am.load_value_net(device=a.device)
    files = sorted(glob.glob(os.path.join(HERE, "examples", "env_*.npz")))
    theta = am.theta_axis()
    k = int(np.argmin(np.abs((np.rad2deg(theta) - a.heading_deg + 180) % 360 - 180)))
    ext = [-2.5, 2.5, -2.5, 2.5]
    ax_m = am.window_axis()

    fig, axes = plt.subplots(len(files), 5, figsize=(21, 4.1 * len(files)), constrained_layout=True)
    results = {}
    print(f"{'environment':16s} {'reference':28s} {'value err':>10s} {'boundary err':>13s} {'false-safe':>11s} {'false-unsafe':>13s}")
    for row, path in enumerate(files):
        ex = np.load(path)
        name = os.path.basename(path)[:-4]
        t0 = time.time()
        z = am.encode_sdf(vae, ex["sdf"])                     # [1, 64]
        V = am.value_on_grid(net, z)                          # [100, 100, 36], metres
        dt = time.time() - t0
        v0, g0 = am.value_and_gradient(net, z, [[0.0, 0.0, 0.0]])       # value and gradient at the ego state
        results[name] = dict(kind=str(ex["kind"]), seconds=dt, value_at_ego=float(v0[0]), gradient_at_ego=g0[0].tolist(),
                             vs_dp=metrics(V, ex["value_dp"]), vs_dp_decoded=metrics(V, ex["value_dp_decoded"]))
        for key, lab in (("vs_dp", "DP on the true SDF"), ("vs_dp_decoded", "DP on the decoded SDF")):
            m = results[name][key]
            print(f"{name + ' (' + str(ex['kind']) + ')':16s} {lab:28s} {m['value_error_cm']:8.2f} cm {m['boundary_error_cm']:10.2f} cm "
                  f"{m['false_safe_pct']:9.2f} % {m['false_unsafe_pct']:11.2f} %")

        sdf, dp, nn = ex["sdf"], ex["value_dp"][:, :, k], V[:, :, k]
        # arrays are indexed [ix, iy]: transpose so that x is horizontal and y points up
        p = axes[row, 0]
        p.imshow(ex["occupancy"].T, origin="lower", extent=ext, cmap=ListedColormap(["#ffffff", "#4a4a4a"]), vmin=0, vmax=1, interpolation="nearest")
        p.contour(ax_m, ax_m, sdf.T, levels=[0], colors="#d62728", linewidths=1.3)
        p.plot(0, 0, "o", ms=5, color="#111", mfc="#ffd400")
        p.annotate("", xy=(0.45, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#111", lw=1.2))
        p.set_title(f"{name} ({ex['kind']}): occupancy grid\nred: SDF = 0", fontsize=10)
        p = axes[row, 1]
        im_s = p.imshow(sdf.T, origin="lower", extent=ext, cmap="RdBu", vmin=-2.5, vmax=2.5)
        p.contour(ax_m, ax_m, sdf.T, levels=[0], colors="#d62728", linewidths=1.3)
        p.set_title("SDF (m)", fontsize=10)
        fig.colorbar(im_s, ax=p, shrink=0.85)
        for col, (img, title) in ((2, (dp, "DP value (reference)")), (3, (nn, "network value (estimate)"))):
            p = axes[row, col]
            im_v = p.imshow(img.T, origin="lower", extent=ext, cmap="RdBu", vmin=-1, vmax=1)
            p.contour(ax_m, ax_m, sdf.T, levels=[0], colors="#d62728", linewidths=0.8)
            p.contour(ax_m, ax_m, dp.T, levels=[0], colors="k", linewidths=1.3)
            if col == 3:
                p.contour(ax_m, ax_m, nn.T, levels=[0], colors="#d6249f", linewidths=1.3)
            p.set_title(f"{title}, heading {np.rad2deg(theta[k]):.0f}°", fontsize=10)
            fig.colorbar(im_v, ax=p, shrink=0.85)
        p = axes[row, 4]
        im_e = p.imshow((nn - dp).T, origin="lower", extent=ext, cmap="PuOr", vmin=-0.15, vmax=0.15)
        p.contour(ax_m, ax_m, dp.T, levels=[0], colors="k", linewidths=0.9)
        p.set_title("network − DP (m)", fontsize=10)
        fig.colorbar(im_e, ax=p, shrink=0.85)
        for p in axes[row]:
            p.set_xlabel("x (m)")
            p.set_ylabel("y (m)")
    fig.suptitle("Estimated value function against dynamic programming on the true SDF.  "
                 "red: obstacle surface (SDF = 0)   black: DP safe-set boundary (V = 0)   magenta: network safe-set boundary", fontsize=12)
    fig_path = os.path.join(a.out, f"comparison_heading_{int(round(np.rad2deg(theta[k]))) % 360:03d}deg.png")
    fig.savefig(fig_path, dpi=70)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    mean = lambda key, q: np.mean([r[key][q] for r in results.values()])
    print(f"\nmean over the {len(files)} examples, against DP on the true SDF:    value error {mean('vs_dp', 'value_error_cm'):.2f} cm, "
          f"boundary error {mean('vs_dp', 'boundary_error_cm'):.2f} cm, false-safe {mean('vs_dp', 'false_safe_pct'):.2f} %")
    print(f"mean over the {len(files)} examples, against DP on the decoded SDF: value error {mean('vs_dp_decoded', 'value_error_cm'):.2f} cm, "
          f"boundary error {mean('vs_dp_decoded', 'boundary_error_cm'):.2f} cm, false-safe {mean('vs_dp_decoded', 'false_safe_pct'):.2f} %")
    print(f"time per environment (encode + full grid of 360,000 states) on {a.device}: {np.mean([r['seconds'] for r in results.values()]):.2f} s")
    print("wrote", fig_path, "and", os.path.join(a.out, "metrics.json"))


if __name__ == "__main__":
    main()
