"""Local 5 m x 5 m ego-frame sensing, in the conventions of AdaptiveSF/README.md section 1.1.

Window: 100 x 100 cells of 0.05 m, centred on the robot, +x along the robot heading, arrays indexed [ix, iy].
Everything outside the window counts as an obstacle (closed border).
"""
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "AdaptiveSF"))
import adaptivesf_models as am  # noqa: E402

N, CELL = am.N_GRID, am.CELL_M
AX = am.window_axis()                     # -2.475 ... 2.475
HALF = N * CELL / 2                       # 2.5 m


def ego_to_world(pose, P):
    """pose = (x, y, psi) in world; P [..., 2] ego-frame points -> world points."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack([pose[0] + c * P[..., 0] - s * P[..., 1], pose[1] + s * P[..., 0] + c * P[..., 1]], -1)


def world_to_ego(pose, P):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = np.asarray(P) - np.asarray(pose[:2])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def window_world_points(pose):
    X, Y = np.meshgrid(AX, AX, indexing="ij")
    return ego_to_world(pose, np.stack([X, Y], -1)).reshape(-1, 2)


def local_occupancy(world, pose, mode="crop"):
    """Occupancy grid occ[ix, iy] of the ego window (True = occupied).

    mode="crop": perfect perception, the true occupancy of every cell centre (current default).
    (A LiDAR ray-casting mode with occluded cells marked occupied/unknown is a planned option.)
    """
    if mode != "crop":
        raise NotImplementedError(mode)
    P = window_world_points(pose)
    return world.occupied(P, cull_center=pose[:2], cull_radius=HALF * np.sqrt(2) + 0.1).reshape(N, N)


def sdf_from_occupancy(occ, cell=CELL):
    """README 2.3: SDF in metres with the closed window border (outside = occupied)."""
    o = np.pad(occ, 1, constant_values=True)
    d_out = distance_transform_edt(~o) * cell
    d_in = distance_transform_edt(o) * cell
    return np.where(o, -(d_in - cell / 2), d_out - cell / 2)[1:-1, 1:-1].astype(np.float32)


def exact_window_sdf(world, pose):
    """Exact SDF on the window cell centres: min(distance to obstacles, distance to the window border)."""
    P = window_world_points(pose)
    d_obs = world.sdf(P, cull_center=pose[:2], cull_radius=HALF * np.sqrt(2) + 3.0, far=10.0).reshape(N, N)
    X, Y = np.meshgrid(AX, AX, indexing="ij")
    d_border = HALF - np.maximum(np.abs(X), np.abs(Y))
    return np.minimum(d_obs, d_border).astype(np.float32)
