"""Convolutional VAE for 100 x 100 local SDF windows (encoder used to obtain the 64-number latent).

Architecture after the range-image VAE of ntnu-arl/sdf-nmpc: ResNet encoder -> AdaptiveAvgPool(2, 2) -> linear
mean / log-variance heads; linear -> 4 residual transposed-conv blocks -> bilinear resize to the image size.
"""
import torch
from torch import nn


def _bn(c, use):
    return nn.BatchNorm2d(c) if use else nn.Identity()


class ResBlock(nn.Module):
    def __init__(self, c_in, stride, batchnorm=True, dropout=0.0):
        super().__init__()
        c_out = c_in * stride
        self.layers = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=not batchnorm), _bn(c_out, batchnorm), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=not batchnorm), _bn(c_out, batchnorm))
        self.shortcut = nn.Identity() if stride == 1 else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, 0, bias=not batchnorm), _bn(c_out, batchnorm))
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        return self.drop(torch.relu(self.layers(x) + self.shortcut(x)))


class ResBlockDeconv(nn.Module):
    def __init__(self, c_in, stride, batchnorm=True, dropout=0.0):
        super().__init__()
        c_out, op = c_in // stride, stride - 1
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, 3, stride, 1, output_padding=op, bias=not batchnorm),
            _bn(c_out, batchnorm), nn.ReLU(),
            nn.ConvTranspose2d(c_out, c_out, 3, 1, 1, bias=not batchnorm), _bn(c_out, batchnorm))
        self.shortcut = nn.Identity() if stride == 1 else nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, 1, stride, 0, output_padding=op, bias=not batchnorm),
            _bn(c_out, batchnorm))
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        return self.drop(torch.relu(self.layers(x) + self.shortcut(x)))


class Encoder(nn.Module):
    def __init__(self, latent, dropout=0.1, batchnorm=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3), nn.ELU(), nn.MaxPool2d(3, 2, 1),
            ResBlock(64, 2, batchnorm, dropout), ResBlock(128, 2, batchnorm, dropout),
            ResBlock(256, 2, batchnorm, dropout), ResBlock(512, 1, batchnorm, 0.0),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Dropout2d(dropout) if dropout else nn.Identity(), nn.Flatten())
        self.mean = nn.Linear(512 * 4, latent)
        self.logvar = nn.Linear(512 * 4, latent)

    def forward(self, x):
        h = self.features(x)
        return self.mean(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent, out_hw=(100, 100), dropout=0.1, batchnorm=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 512 * 7 * 7), nn.ELU(), nn.Unflatten(1, (512, 7, 7)),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            ResBlockDeconv(512, 2, batchnorm, dropout), ResBlockDeconv(256, 2, batchnorm, dropout),
            ResBlockDeconv(128, 2, batchnorm, dropout), ResBlockDeconv(64, 2, batchnorm, dropout),
            nn.ConvTranspose2d(32, 1, 5, 1, 2),
            nn.Upsample(size=out_hw, mode="bilinear", align_corners=False))

    def forward(self, z):
        return self.net(z)


class SdfVae(nn.Module):
    def __init__(self, latent=64, out_hw=(100, 100), dropout=0.1, batchnorm=True):
        super().__init__()
        self.latent = latent
        self.encoder = Encoder(latent, dropout, batchnorm)
        self.decoder = Decoder(latent, out_hw, dropout, batchnorm)

    def encode(self, x):
        """Latent mean for SDF windows x of shape [B, 1, 100, 100] (metres, already clipped to the SDF range)."""
        return self.encoder(x)[0]

    def decode(self, z):
        """Decoded SDF windows [B, 1, 100, 100] in metres for latents z of shape [B, latent]."""
        return self.decoder(z)
