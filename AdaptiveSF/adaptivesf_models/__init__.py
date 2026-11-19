"""Loading and using the AdaptiveSF deliverable-1 models: the SDF VAE (latent 64) and the latent-conditioned value network."""
import math
import os

import numpy as np
import torch

from .sdf_vae import SdfVae
from .value_net import ValueNet

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
SDF_CLIP = (-1.0, 2.0)      # the VAE was trained on SDFs clipped to this range (metres)
N_GRID = 100                # cells per side of the window
CELL_M = 0.05               # cell size (m)


def window_axis():
    """Cell-centre coordinates (m) along x and y: -2.475 ... +2.475."""
    return (np.arange(N_GRID) + 0.5) * CELL_M - N_GRID * CELL_M / 2


def theta_axis(n_theta=36):
    """Headings (rad) of the DP grid: -pi + k * 2 pi / n_theta."""
    return -math.pi + np.arange(n_theta) * (2 * math.pi / n_theta)


def load_vae(path=None, device="cpu"):
    ck = torch.load(path or os.path.join(_MODELS, "sdf_vae_latent64_penalty.pt"), map_location=device)
    vae = SdfVae(**ck["config"]).to(device)
    vae.load_state_dict(ck["state_dict"])
    return vae.eval()


def load_value_net(path=None, device="cpu"):
    ck = torch.load(path or os.path.join(_MODELS, "value_net_16000env.pt"), map_location=device)
    net = ValueNet(**ck["config"]).to(device)
    net.load_state_dict(ck["state_dict"])
    return net.eval()


@torch.no_grad()
def encode_sdf(vae, sdf):
    """SDF window(s) -> latent mean(s).

    sdf: array or tensor [100, 100] or [B, 100, 100], metres, indexed [ix, iy], positive in free space, with the
    closed window border included (see README). Returns a tensor [B, 64] on the model's device.
    """
    dev = next(vae.parameters()).device
    x = torch.as_tensor(np.asarray(sdf), dtype=torch.float32, device=dev)
    if x.ndim == 2:
        x = x[None]
    return vae.encode(x.clamp(*SDF_CLIP)[:, None])


@torch.no_grad()
def decode_latent(vae, z):
    """Latent(s) [B, 64] -> decoded SDF window(s) [B, 100, 100] in metres (clipped to the trained range)."""
    return vae.decode(z)[:, 0].clamp(*SDF_CLIP)


@torch.no_grad()
def value_on_grid(net, z, n_theta=36, chunk=500_000):
    """Value function of one environment on the 100 x 100 x n_theta grid.

    z: tensor [64] or [1, 64]. Returns a numpy array V[ix, iy, itheta] in metres; V > 0 means safe.
    """
    dev = next(net.parameters()).device
    ax = torch.tensor(window_axis(), dtype=torch.float32, device=dev)
    th = torch.tensor(theta_axis(n_theta), dtype=torch.float32, device=dev)
    X, Y, T = torch.meshgrid(ax, ax, th, indexing="ij")
    states = torch.stack([X, Y, T], -1).reshape(-1, 3)
    z = z.reshape(1, -1).to(dev)
    out = torch.cat([net(states[c:c + chunk], z.expand(len(states[c:c + chunk]), -1)) for c in range(0, len(states), chunk)])
    return out.view(N_GRID, N_GRID, n_theta).cpu().numpy()


def value_and_gradient(net, z, states):
    """Value and its gradient at arbitrary states.

    states: array or tensor [N, 3] = (x, y, theta) in the window frame. z: [64] or [1, 64].
    Returns (V [N], dV/d(x, y, theta) [N, 3]) as numpy arrays.
    """
    dev = next(net.parameters()).device
    s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=dev).clone().requires_grad_(True)
    v = net(s, z.reshape(1, -1).to(dev).expand(len(s), -1))
    g = torch.autograd.grad(v.sum(), s)[0]
    return v.detach().cpu().numpy(), g.cpu().numpy()
