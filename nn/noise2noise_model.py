"""4-level U-Net for single-channel Noise2Noise denoising.

Input/output: (B, 1, H, W) float32 tensors.
~7 M parameters — enough to learn camera noise structure, small enough to
train in minutes on a consumer GPU.

Reference:
    Lehtinen et al., "Noise2Noise: Learning Image Restoration without Clean Data",
    ICML 2018. https://arxiv.org/abs/1803.04189
"""

from typing import Optional

import torch
import torch.nn as nn


def _norm_layer(kind: str, ch: int) -> Optional[nn.Module]:
    """Normalisation layer by name, or None for a plain conv stack.

    Default is none, which is what the N2N paper's network uses. BatchNorm is
    a poor fit here for two reasons specific to this task: it bakes the
    training set's activation statistics into running means, so any shift in
    input noise level at inference lands harder than it otherwise would; and
    this is a regression where the absolute output level carries the
    photometry, not a classification where only the argmax matters. Kept
    switchable rather than deleted so the comparison can be run.
    """
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        return nn.GroupNorm(min(8, ch), ch)
    return None


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "none"):
        super().__init__()
        layers: list[nn.Module] = []
        for src in (in_ch, out_ch):
            # Without a norm layer the conv needs its bias back — it was
            # disabled only because BatchNorm's shift made it redundant.
            n = _norm_layer(norm, out_ch)
            layers.append(nn.Conv2d(src, out_ch, 3, padding=1, bias=n is None))
            if n is not None:
                layers.append(n)
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, features: tuple[int, ...] = (32, 64, 128, 256),
                 norm: str = "none"):
        super().__init__()
        self.norm = norm
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = 1
        for f in features:
            self.encoders.append(_ConvBlock(in_ch, f, norm))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f

        self.bottleneck = _ConvBlock(features[-1], features[-1] * 2, norm)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev = list(reversed(features))
        in_ch = features[-1] * 2
        for f in rev:
            self.upconvs.append(nn.ConvTranspose2d(in_ch, f, 2, stride=2))
            self.decoders.append(_ConvBlock(f * 2, f, norm))
            in_ch = f

        self.head = nn.Conv2d(features[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            # Handle odd spatial dimensions from max-pool
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)
