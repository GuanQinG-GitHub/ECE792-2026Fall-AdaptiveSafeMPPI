"""Stage 1a: generate the 50 m x 50 m static forest world and plot it.

    python scripts/s1_generate_forest.py --seed 0
Writes worlds/forest_seed{seed}.json and results/stage1/forest_seed{seed}_map.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon as MplPoly, Circle as MplCircle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from simenv import ROBOT_RADIUS, generate_forest  # noqa: E402

COLORS = dict(trunk="#5b4636", big_trunk="#3e2f23", rock="#7d7d7d", bush="#4f7a3a", log="#8a6a45", box="#5c5c70",
              ellipse="#8a6a45", polygon="#555")


def draw_world(ax, world, lw=0.0):
    for s in world.shapes:
        if s.kind == "circle":
            col = COLORS["big_trunk"] if s.r > 0.35 else COLORS["trunk"]
            ax.add_patch(MplCircle(s.c, s.r, fc=col, ec="none"))
        else:
            ax.add_patch(MplPoly(s.V, closed=True, fc=COLORS.get(s.label, "#555"), ec="none"))
    ax.set_xlim(0, world.size)
    ax.set_ylim(0, world.size)
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=None, help="override number of obstacles")
    a = ap.parse_args()
    kw = {} if a.n is None else dict(n_obstacles=a.n)
    world = generate_forest(a.seed, **kw)
    os.makedirs(os.path.join(ROOT, "worlds"), exist_ok=True)
    out = os.path.join(ROOT, "results", "stage1")
    os.makedirs(out, exist_ok=True)
    wpath = os.path.join(ROOT, "worlds", f"forest_seed{a.seed}.json")
    world.save(wpath)
    m = world.meta
    print(json.dumps({k: v for k, v in m.items() if k != "params"}, indent=1))

    sdf, gax = world.global_sdf(0.05)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.3), constrained_layout=True)
    ax = axes[0]
    ax.set_facecolor("#f4f1e8")
    draw_world(ax, world)
    ax.plot(*world.start, "o", ms=11, mfc="#2ca02c", mec="k", zorder=5)
    ax.plot(*world.goal, "*", ms=18, mfc="#ffd400", mec="k", zorder=5)
    ax.add_patch(plt.Rectangle(world.start - 2.5, 5, 5, fill=False, ec="#1f77b4", lw=1.6, ls="--"))
    ax.text(world.start[0] - 2.4, world.start[1] + 2.7, "5 m sensing window", color="#1f77b4", fontsize=9)
    ax.set_title(f"Forest world (seed {a.seed}): {m['n_obstacles']} obstacles, "
                 f"{100 * m['occupied_fraction']:.1f}% occupied", fontsize=12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=COLORS[k], mec="none", ms=9, label=f"{k} ({m['counts'][k]})")
               for k in m["counts"]]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="#2ca02c", mec="k", ms=9, label="start"),
                plt.Line2D([], [], marker="*", ls="", mfc="#ffd400", mec="k", ms=13, label="goal")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    ax = axes[1]
    free = sdf > ROBOT_RADIUS
    im = ax.imshow(np.clip(sdf, -0.5, 2.0).T, origin="lower", extent=[0, world.size, 0, world.size], cmap="viridis")
    ax.contour(gax, gax, sdf.T, levels=[ROBOT_RADIUS], colors="w", linewidths=0.5)
    ax.imshow(np.ma.masked_where(free, ~free).T, origin="lower", extent=[0, world.size, 0, world.size],
              cmap=ListedColormap(["#d62728"]), alpha=0.35, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="SDF to nearest obstacle (m), clipped")
    ax.set_title(f"Global SDF; red tint = SDF ≤ r = {ROBOT_RADIUS} m (robot centre in collision)\n"
                 f"start→goal connected for disc robot: {m['connected_start_goal']}", fontsize=11)
    ax.set_xlabel("x (m)")
    fig.savefig(os.path.join(out, f"forest_seed{a.seed}_map.png"), dpi=110)

    # free-space clearance statistics
    c = sdf[sdf > 0]
    gap = {f"p{q}": float(np.percentile(c, q)) for q in (10, 50, 90)}
    stats = dict(meta={k: v for k, v in m.items()}, clearance_of_free_cells_m=gap,
                 frac_free_with_clearance_gt_r=float((sdf > ROBOT_RADIUS).mean()))
    with open(os.path.join(out, f"forest_seed{a.seed}_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print("wrote", wpath)


if __name__ == "__main__":
    main()
