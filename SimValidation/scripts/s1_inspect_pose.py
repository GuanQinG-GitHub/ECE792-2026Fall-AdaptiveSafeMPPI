"""Inspect one pose of the stage-1 value-function check in detail.

    python scripts/s1_inspect_pose.py --pose-id 20
Shows, for the heading slice where the 'certain false-safe' value is largest: true vs decoded SDF = r,
the V = 0 contour, and the states with V > 0 but true clearance <= r. Writes results/stage1/inspect_pose{id}.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from simenv import ROBOT_RADIUS as R, World, sensing as S  # noqa: E402
from s1_check_value_function import AX, EXT, TH, run_pose  # noqa: E402

am = S.am


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-id", type=int, required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, "results", "stage1", "vf_metrics.json")))
    world = World.load(os.path.join(ROOT, d["summary"]["world"]))
    pose = np.array(d["poses"][a.pose_id])
    r = run_pose(world, pose, am.load_vae(device=a.device), am.load_value_net(device=a.device))
    V, st = r["V"], r["sdf_true"]
    fs = (V > 0) & (st[..., None] <= R)
    k = int(np.argmax(np.where(fs, V, -1).max((0, 1)))) if fs.any() else int(np.argmin(np.abs(TH)))
    Vk = V[:, :, k]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for a_, (img, ttl, cm, lim) in zip(axes[:2], ((r["sdf_true"], "true SDF (m)", "RdBu", 2),
                                                  (Vk, f"V at θ = {np.rad2deg(TH[k]):.0f}° (ego frame)", "RdBu", 0.6))):
        im = a_.imshow(np.clip(img, -lim, lim).T, origin="lower", extent=EXT, cmap=cm, vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=a_, shrink=0.8)
        a_.set_title(ttl)
    for a_ in axes:
        a_.contour(AX, AX, st.T, levels=[R], colors="k", linewidths=1.2)
        a_.contour(AX, AX, r["sdf_dec"].T, levels=[R], colors="#ff7f0e", linewidths=1.0, linestyles="--")
        if (Vk > 0).any() and (Vk <= 0).any():
            a_.contour(AX, AX, Vk.T, levels=[0], colors="#d6249f", linewidths=1.3)
        a_.set_xlabel("x_ego (m)"); a_.set_ylabel("y_ego (m)")
    ix, iy = np.nonzero(fs[:, :, k])
    sc = axes[2].scatter(AX[ix], AX[iy], c=Vk[ix, iy] * 100, cmap="viridis", s=14, zorder=5)
    fig.colorbar(sc, ax=axes[2], shrink=0.8, label="V at false-safe state (cm)")
    axes[2].set_xlim(-2.5, 2.5); axes[2].set_ylim(-2.5, 2.5); axes[2].set_aspect("equal")
    axes[2].set_title(f"states with V > 0 but true SDF ≤ r (this slice: {fs[:, :, k].sum()} cells)")
    fig.suptitle(f"pose #{a.pose_id} (x={pose[0]:.2f} m, y={pose[1]:.2f} m, ψ={np.rad2deg(pose[2]):.0f}°)   black: true SDF = r   orange dashed: decoded SDF = r   "
                 f"magenta: V = 0", fontsize=11)
    path = os.path.join(ROOT, "results", "stage1", f"inspect_pose{a.pose_id}.png")
    fig.savefig(path, dpi=80)
    print("wrote", path)


if __name__ == "__main__":
    main()
