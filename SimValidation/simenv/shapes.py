"""Analytic 2D obstacle shapes with exact signed distance (positive outside, negative inside).

Every shape exposes
    sdf(P)       P: [N, 2] world points (m) -> [N] signed distance (m)
    bounding     (cx, cy, R): a disc that contains the shape (used for culling and spacing)
    to_dict()    JSON-serialisable description (for saving worlds)
"""
import numpy as np


class Circle:
    kind = "circle"

    def __init__(self, cx, cy, r):
        self.c = np.array([cx, cy], float)
        self.r = float(r)
        self.bounding = (self.c[0], self.c[1], self.r)

    def sdf(self, P):
        return np.linalg.norm(P - self.c, axis=1) - self.r

    def to_dict(self):
        return dict(kind=self.kind, cx=float(self.c[0]), cy=float(self.c[1]), r=self.r)


class Polygon:
    """Simple polygon (convex or not), vertices [M, 2] in order. Exact SDF via edge distances + crossing test."""
    kind = "polygon"

    def __init__(self, vertices, label="polygon"):
        self.V = np.asarray(vertices, float)
        self.label = label
        c = self.V.mean(0)
        self.bounding = (c[0], c[1], float(np.linalg.norm(self.V - c, axis=1).max()))

    def sdf(self, P):
        A = self.V
        B = np.roll(self.V, -1, axis=0)
        E = B - A                                   # [M, 2]
        W = P[:, None, :] - A[None]                 # [N, M, 2]
        t = np.clip((W * E).sum(-1) / (E * E).sum(-1), 0.0, 1.0)
        D = W - t[..., None] * E
        dist = np.sqrt((D * D).sum(-1).min(1))
        # crossing-number inside test
        py = P[:, 1:2]
        ay, by = A[None, :, 1], B[None, :, 1]
        cond = (ay > py) != (by > py)
        xint = A[None, :, 0] + (py - ay) * (B[None, :, 0] - A[None, :, 0]) / np.where(by - ay == 0, 1e-12, by - ay)
        inside = (cond & (P[:, 0:1] < xint)).sum(1) % 2 == 1
        return np.where(inside, -dist, dist)

    def to_dict(self):
        return dict(kind=self.kind, label=self.label, vertices=self.V.tolist())


def shape_from_dict(d):
    if d["kind"] == "circle":
        return Circle(d["cx"], d["cy"], d["r"])
    return Polygon(d["vertices"], d.get("label", "polygon"))


# ---------- random shape factories (all centred at c) ----------

def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def make_box(c, w, h, ang):
    V = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    return Polygon(V @ rot(ang).T + c, "box")


def make_ellipse(c, a, b, ang, n=28):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = np.stack([a * np.cos(t), b * np.sin(t)], 1)
    return Polygon(V @ rot(ang).T + c, "ellipse")


def make_convex_blob(c, R, rng, n_pts=9):
    """Convex hull of random points in a disc of radius R (rock-like)."""
    from scipy.spatial import ConvexHull
    ang = rng.uniform(0, 2 * np.pi, n_pts)
    rad = R * np.sqrt(rng.uniform(0.35, 1.0, n_pts))
    P = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    return Polygon(P[ConvexHull(P).vertices] + c, "rock")


def make_star_blob(c, R, rng, n=11):
    """Non-convex star-shaped polygon (bush / root cluster)."""
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = R * rng.uniform(0.45, 1.0, n)
    return Polygon(np.stack([r * np.cos(t), r * np.sin(t)], 1) + c, "bush")
