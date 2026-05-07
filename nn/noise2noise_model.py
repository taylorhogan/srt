"""4-level U-Net for single-channel Noise2Noise denoising.

Input/output: (B, 1, H, W) float32 tensors.
~7 M parameters — enough to learn camera noise structure, small enough to
train in minutes on a consumer GPU.

Reference:
    Lehtinen et al., "Noise2Noise: Learning Image Restoration without Clean Data",
    ICML 2018. https://arxiv.org/abs/1803.04189
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, features: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = 1
        for f in features:
            self.encoders.append(_ConvBlock(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f

        self.bottleneck = _ConvBlock(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev = list(reversed(features))
        in_ch = features[-1] * 2
        for f in rev:
            self.upconvs.append(nn.ConvTranspose2d(in_ch, f, 2, stride=2))
            self.decoders.append(_ConvBlock(f * 2, f))
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
