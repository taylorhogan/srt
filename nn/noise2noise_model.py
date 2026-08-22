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
    """4-level U-Net. `residual` selects what the head's output means.

    Input and output are both in asinh space, t = asinh(x / sigma_sky), and
    inference inverts with x = sigma * sinh(t). That inversion is the whole
    problem: d(x)/d(t) = cosh(t), which reaches ~6200 at the bright end of a
    real frame, so any error in t at a bright star is amplified by that factor
    in ADU. Measured, the brightest flux bin came out at 0.0023 of raw.

    Three modes, in the order they were tried (docs/N2N_LAB_MANUAL.md):

    "none"    head output IS the prediction. The original. Bright-end error is
              amplified by cosh(t) with nothing opposing it.

    "asinh"   prediction is t + f(t). Tried on the reasoning that a value which
              passes through is never reconstructed, so nothing amplifies.
              That reasoning is wrong and this measured *worse* than "none"
              (brightest 0.0645 vs 0.2490): the correction is added before the
              sinh, so sensitivity is still cosh(t). f = -2 at a bright star is
              already sinh(7)/sinh(9) = 0.27.

    "linear"  prediction is asinh(sinh(t) + g), i.e. the correction g is added
              in *linear* units of sky sigma, not in asinh space. Default.

    Why "linear" is different in kind rather than degree: the ADU error is
    sigma * dg, with no exponential factor anywhere, so it cannot be amplified.
    A bright star at 5000 sigma is moved 0.06% by a correction as large as
    g = 3, while the same g is a 3-sigma change to the sky — which is the
    asymmetry the task actually needs. sinh(t) recovers the linear frame from
    the network's own input exactly, so this needs no change to the dataset,
    and the loss stays in asinh space where it is not dominated by bright
    pixels.

    The head is zero-initialised in both residual modes, so training starts
    from exact identity: at step 0 the frame passes through untouched and
    photometry is preserved by construction.
    """

    def __init__(self, features: tuple[int, ...] = (32, 64, 128, 256),
                 norm: str = "none", residual: str = "linear",
                 in_ch: int = 1, out_ch: Optional[int] = None):
        """`in_ch` > 1 denoises several filters jointly.

        The default of 1 is the historical single-filter model and every
        checkpoint written before 2026-08-22 assumes it; `load_state_dict` is
        strict, so a mismatch fails loudly rather than silently reinterpreting
        weights.

        Why more than one channel is worth having: with in_ch=1 each filter is
        denoised in complete isolation, so per-channel retention differences
        cannot be anything but independent — which is exactly what produced the
        measured colour casts (lab manual steps 21, and the S-II case on
        NGC 6888). A joint model can at least represent "this structure appears
        in every channel, so it is real", which a single-channel one cannot.

        `out_ch` defaults to `in_ch`: N2N predicts its input's clean counterpart,
        so the shapes match by construction.
        """
        super().__init__()
        if residual not in ("none", "asinh", "linear"):
            raise ValueError(f"residual must be none/asinh/linear, got {residual!r}")
        if in_ch < 1:
            raise ValueError(f"in_ch must be >= 1, got {in_ch}")
        self.norm = norm
        self.residual = residual
        self.in_ch = in_ch
        self.out_ch = in_ch if out_ch is None else out_ch
        if self.residual != "none" and self.out_ch != in_ch:
            # The residual paths add the head's output to the input, so the two
            # must have the same channel count.
            raise ValueError(
                f"residual={residual!r} needs out_ch == in_ch "
                f"({self.out_ch} != {in_ch})")
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = self.in_ch
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

        self.head = nn.Conv2d(features[0], self.out_ch, 1)
        if residual != "none":
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
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

        out = self.head(x)
        if self.residual == "asinh":
            return identity + out
        if self.residual == "linear":
            # Correction applied in linear sky-sigma units, then returned to
            # asinh space so the loss sees the same units as the target.
            return torch.asinh(torch.sinh(identity) + out)
        return out
