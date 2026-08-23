"""Multi-scale L2: spend the loss where the measured headroom is.

Plain L2 on asinh pixels is dominated by whatever band holds the most residual
power. On noise-dominated stacks that is the fine scales — which the Wiener
analysis (lab manual step 32) showed are already at the optimal bound — while
the 16-256 px bands, where the scene is signal-dominated and the model passes
only 0.72-0.93 of a 0.95-0.999 bound, contribute little to the loss and so are
weakly optimised. The result is a model optimal at the hard scales and
over-regularised at the easy ones.

This loss decomposes the *residual* (pred - target) into a Laplacian pyramid
and weights each octave by the headroom measured in step 32:

    band (px)   ~2-4  4-8  8-16  16-32  32-64  64-128  128-256  >256
    weight       1.0  1.0  1.25   1.7    1.7     1.4      1.4    1.0

(then normalised so the weights average 1, keeping the loss magnitude — and
therefore the working lr and grad-clip settings — comparable to plain L2).

Note the weighting is deliberately mild at 4-16 px even though the headroom
there is ~zero: down-weighting a band to nothing would free the model to dump
arbitrary error into it, and the bound says the *current* behaviour there is
already right. The aim is to raise the price of mid-band error, not to license
fine-band error.

Decomposing the residual rather than the two images separately is both cheaper
(one pyramid, not two) and exactly equivalent for a quadratic loss, since the
pyramid is linear.
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


class MultiScaleL2(torch.nn.Module):
    """L2 on a Laplacian pyramid of the residual, one octave per level.

    Level j is the band between smoothing scales 2^j and 2^(j+1) px, built with
    avg-pool / bilinear-upsample (a box-Laplacian pyramid — the exact kernel
    shape is irrelevant for a *weighting*, only the octave split matters). The
    final low-pass residual carries everything coarser than the last level.
    """

    # step-32 headroom mapped onto octaves; see module docstring
    DEFAULT_WEIGHTS = (1.0, 1.0, 1.25, 1.7, 1.7, 1.4, 1.4, 1.0)

    def __init__(self, weights: Optional[Sequence[float]] = None):
        super().__init__()
        w = torch.tensor(weights or self.DEFAULT_WEIGHTS, dtype=torch.float32)
        w = w * (len(w) / w.sum())          # mean 1: loss scale matches plain L2
        self.register_buffer("weights", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r = pred - target
        total = torch.zeros((), device=r.device, dtype=r.dtype)
        levels = len(self.weights) - 1
        for j in range(levels):
            if min(r.shape[-2:]) < 4:
                # patch exhausted early (tiny inputs in tests): dump the rest
                # into this level rather than pooling a 2-px map
                break
            down = F.avg_pool2d(r, 2)
            up = F.interpolate(down, size=r.shape[-2:], mode="bilinear",
                               align_corners=False)
            band = r - up
            total = total + self.weights[j] * band.pow(2).mean()
            r = down
        total = total + self.weights[min(levels, len(self.weights) - 1)] * r.pow(2).mean()
        return total
