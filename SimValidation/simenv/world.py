"""Static 2D world (forest) for the adaptive-safety-filter simulation.

World frame: x right, y up, metres, map = [0, L] x [0, L]. Outside the map there are no obstacles.
Obstacles are analytic shapes (see shapes.py) so the true SDF is exact at any query point.
"""
import json

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from .shapes import Circle, make_box, make_convex_blob, make_ellipse, make_star_blob, shape_from_dict

ROBOT_RADIUS = 0.25     # m, fixed by the trained value function


class World:
    def __init__(self, shapes, size=50.0, start=None, goal=None, meta=None):
        self.shapes = list(shapes)
        self.size = float(size)
        self.start = None if start is None else np.asarray(start, float)
        self.goal = None if goal is None else np.asarray(goal, float)
        self.meta = meta or {}
        self._bc = np.array([s.bounding for s in self.shapes]).reshape(-1, 3)   # cx, cy, R

    # ---------------- geometry queries ----------------
    def nearby(self, center, radius):
        """Indices of shapes whose bounding disc intersects the disc (center, radius)."""
        d = np.linalg.norm(self._bc[:, :2] - np.asarray(center)[None], axis=1)
        return np.nonzero(d - self._bc[:, 2] < radius)[0]

    def sdf(self, P, cull_center=None, cull_radius=None, far=10.0):
        """Exact signed distance (m) to the nearest obstacle at world points P [N, 2]. Capped at `far`."""
        P = np.atleast_2d(np.asarray(P, float))
        if cull_center is None:
            lo, hi = P.min(0), P.max(0)
            cull_center, cull_radius = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2 + far
        out = np.full(len(P), far)
        for i in self.nearby(cull_center, cull_radius):
            out = np.minimum(out, self.shapes[i].sdf(P))
        return out

    def occupied(self, P, **kw):
        return self.sdf(P, **kw) <= 0.0

    def rasterize(self, res=0.05, pad=0.0):
        """Global occupancy grid occ[ix, iy] with cell centres at (i + 0.5) * res - pad."""
        n = int(round((self.size + 2 * pad) / res))
        ax = (np.arange(n) + 0.5) * res - pad
        occ = np.zeros((n, n), bool)
        for s in self.shapes:
            cx, cy, R = s.bounding
            i0, i1 = np.searchsorted(ax, [cx - R - res, cx + R + res])
            j0, j1 = np.searchsorted(ax, [cy - R - res, cy + R + res])
            if i1 <= i0 or j1 <= j0:
                continue
            X, Y = np.meshgrid(ax[i0:i1], ax[j0:j1], indexing="ij")
            occ[i0:i1, j0:j1] |= (s.sdf(np.stack([X.ravel(), Y.ravel()], 1)) <= 0).reshape(X.shape)
        return occ, ax

    def global_sdf(self, res=0.05):
        """Signed distance on the global raster (EDT, for plots / path checks; use sdf() for exact values)."""
        occ, ax = self.rasterize(res)
        d_out = distance_transform_edt(~occ) * res
        d_in = distance_transform_edt(occ) * res
        return np.where(occ, -d_in, d_out), ax

    # ---------------- persistence ----------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(size=self.size, start=None if self.start is None else self.start.tolist(),
                           goal=None if self.goal is None else self.goal.tolist(), meta=self.meta,
                           shapes=[s.to_dict() for s in self.shapes]), f)

    @staticmethod
    def load(path):
        with open(path) as f:
            d = json.load(f)
        return World([shape_from_dict(s) for s in d["shapes"]], d["size"], d["start"], d["goal"], d["meta"])


# ---------------- forest generator ----------------

DEFAULT_FOREST = dict(
    size=50.0,
    n_obstacles=900,          # attempts are capped; actual count reported in meta
    min_gap=0.15,             # m, minimum free gap between bounding discs (so shapes never merge)
    clear_radius=2.0,         # m, obstacle-free disc around start and goal
    start=(3.0, 3.0),
    goal=(47.0, 47.0),
    # mixture weights of obstacle types
    mix=dict(trunk=0.50, big_trunk=0.12, rock=0.14, bush=0.10, log=0.08, box=0.06),
)


def _sample_shape(kind, c, rng):
    if kind == "trunk":
        return Circle(c[0], c[1], rng.uniform(0.12, 0.35))
    if kind == "big_trunk":
        return Circle(c[0], c[1], rng.uniform(0.35, 0.75))
    if kind == "rock":
        return make_convex_blob(c, rng.uniform(0.35, 0.9), rng)
    if kind == "bush":
        return make_star_blob(c, rng.uniform(0.5, 1.1), rng)
    if kind == "log":                                    # fallen log: long thin box or ellipse
        L, w, a = rng.uniform(1.5, 3.5), rng.uniform(0.2, 0.4), rng.uniform(0, np.pi)
        return make_box(c, L, w, a) if rng.random() < 0.5 else make_ellipse(c, L / 2, w / 2, a)
    if kind == "box":
        return make_box(c, rng.uniform(0.4, 1.4), rng.uniform(0.4, 1.4), rng.uniform(0, np.pi))
    raise ValueError(kind)


def generate_forest(seed=0, **overrides):
    p = {**DEFAULT_FOREST, **overrides}
    rng = np.random.default_rng(seed)
    kinds, w = zip(*p["mix"].items())
    w = np.array(w) / np.sum(w)
    L = p["size"]
    shapes, bc, counts = [], [], {k: 0 for k in kinds}
    keep_out = [np.array(p["start"]), np.array(p["goal"])]
    attempts = 0
    while len(shapes) < p["n_obstacles"] and attempts < 40 * p["n_obstacles"]:
        attempts += 1
        k = kinds[rng.choice(len(kinds), p=w)]
        c = rng.uniform(0, L, 2)
        s = _sample_shape(k, c, rng)
        cx, cy, R = s.bounding
        if any(np.linalg.norm(np.array([cx, cy]) - q) < R + p["clear_radius"] for q in keep_out):
            continue
        if bc:
            B = np.array(bc)
            if np.any(np.linalg.norm(B[:, :2] - [cx, cy], axis=1) < B[:, 2] + R + p["min_gap"]):
                continue
        shapes.append(s)
        bc.append((cx, cy, R))
        counts[k] += 1
    meta = dict(generator="forest", seed=seed, params={k: v for k, v in p.items()}, counts=counts,
                n_obstacles=len(shapes))
    world = World(shapes, L, p["start"], p["goal"], meta)
    # statistics: occupied fraction and start->goal connectivity for the disc robot
    sdf, ax = world.global_sdf(0.05)
    world.meta["occupied_fraction"] = float((sdf <= 0).mean())
    world.meta["connected_start_goal"] = bool(path_exists(sdf, ax, p["start"], p["goal"], ROBOT_RADIUS + 0.05))
    return world


def path_exists(sdf, ax, a, b, clearance):
    free = sdf > clearance
    lab, _ = label(free)
    ia = np.argmin(np.abs(ax - a[0])), np.argmin(np.abs(ax - a[1]))
    ib = np.argmin(np.abs(ax - b[0])), np.argmin(np.abs(ax - b[1]))
    return lab[ia] != 0 and lab[ia] == lab[ib]
