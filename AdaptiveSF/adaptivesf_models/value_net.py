"""Latent-conditioned value network V(x, y, theta, z): SIREN MLP with the latent entering at low frequency."""
import math

import torch
from torch import nn


class Siren(nn.Module):
    """Fully connected network with sin(w0 * .) activations and SIREN initialisation; linear output."""

    def __init__(self, n_in, hidden=512, layers=3, w0=30.0):
        super().__init__()
        self.w0 = w0
        dims = [n_in] + [hidden] * (layers + 1) + [1]
        self.lin = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        with torch.no_grad():
            for i, m in enumerate(self.lin):
                n = m.weight.shape[1]
                bound = 1 / n if i == 0 else math.sqrt(6 / n) / w0
                m.weight.uniform_(-bound, bound)

    def forward(self, x):
        for m in self.lin[:-1]:
            x = torch.sin(self.w0 * m(x))
        return self.lin[-1](x).squeeze(-1)


class ValueNet(nn.Module):
    """Safety value in metres.

    state : [N, 3] = (x [m], y [m], theta [rad]) in the window frame (origin = ego at the sensing instant, +x = its heading)
    z     : [N, n_latent] latent mean of the SDF window (one row per state; repeat the same latent for one environment)

    The normalisation constants (latent mean/std over the training environments, value mean/std) are buffers and are
    restored from the checkpoint.
    """

    def __init__(self, n_latent=64, hidden=512, layers=3, half_window_m=2.5, z_scale=1 / 30):
        super().__init__()
        self.register_buffer("z_mean", torch.zeros(n_latent))
        self.register_buffer("z_std", torch.ones(n_latent))
        self.register_buffer("v_mean", torch.tensor(0.0))
        self.register_buffer("v_std", torch.tensor(1.0))
        self.half, self.z_scale = half_window_m, z_scale
        self.net = Siren(4 + n_latent, hidden, layers)

    def forward(self, state, z):
        sf = torch.cat([state[:, :2] / self.half, torch.cos(state[:, 2:3]), torch.sin(state[:, 2:3])], dim=1)
        zf = (z - self.z_mean) / self.z_std * self.z_scale
        return self.v_mean + self.v_std * self.net(torch.cat([sf, zf], dim=1))
