# N2N improvement plan — 2026-08-23

Drawn from `N2N_LAB_MANUAL.md` steps 18–31. The manual's own conclusion frames
the problem: **model choice currently makes ~10% of the decision; the shared
band limit makes the other 90%** (step 31). So this plan spends almost nothing
on training-set variations — six of those failed in two days — and aims at the
three things the evidence says are actually wrong: the band limit, the tiling
artefact, and the cost of running experiments at all.

## What "better" means, measurably

Primary metrics, in order:

1. **Transfer function by scale** (`n2n_fractal_injection.py`) — the only
   measurement with ground truth. Target: raise T(k) in the 8–128 px bands
   without losing the >256 px bands. Current (p512, 3 seeds): 0.59–0.64 at
   8–16 px, 0.70–0.80 at 32–64, 0.325–0.335 at 4–8.
2. **Extended retention + cross-channel spread** (`n2n_extended_check.py`,
   paired via `n2n_compare_paired.py`) on held-out ngc6888 **and** ic1396 — two
   fields, so a win on one field alone is not a result.
3. **Photometry** must not regress: flux within 0.03 of unity (the seed floor),
   sources ≥95%.

Non-negotiable protocol (step 28): every A/B has ≥2 seeds per arm, or is
compared paired against the existing 3-seed baselines
(`n2n_depth_var_full_ts*`, p256; `n2n_depth_ctx_full*_p512`, p512 — all sh2-92
deep, so new interventions on that config need no baseline retraining). Effects
must clear the floor: ~0.02 retention at 1–2σ, ~0.012 at 4–8σ, ~0.03 flux.
Never judged by loss, best-epoch, or previews.

## Phase 0 — analysis only, no training (~2 h). Decides whether Phase 2 exists.

**0a. Wiener bound on the transfer function.** Nobody has checked whether the
measured T(k) is *already near optimal*. For the injected field the signal
spectrum S(k) is known exactly (band-limited k⁻³ at 1σ RMS) and the noise
spectrum N(k) is measurable from star-free sky of the same stack. The
Wiener-optimal transfer is T*(k) = S(k)/(S(k)+N(k)).

- If measured T(k) ≈ T*(k) in the fine bands, **the band limit is optimal
  estimation, not a defect** — no loss or architecture change can beat it at
  this SNR, Phase 2 is cancelled, and the honest answer for fine nebular
  texture is integration time. That would close the manual's biggest open
  question for the cost of one afternoon of arithmetic.
- If measured T(k) sits well below T*(k), the gap is the headroom Phase 2 is
  allowed to chase, band by band.

**0b. Tiling artefact mechanism test** (open question, step 20): denoise one
region at two tile offsets and difference. Confirms the overlap blend as the
mechanism and gives a before-number for the Phase 1 fix.

**0c. Marginal-vs-absolute gap probe** (step 29's open gap: real ic1396 reads
0.384 where T(k) predicts better). Inject the fractal field at 0.25σ and 2σ
amplitudes; if recovered fraction depends on amplitude, the operator's response
is state-dependent and injection numbers need a caveat everywhere they are
quoted. Pure computation on existing models.

## Phase 1 — cheap certain wins (~1 day total, mostly unattended)

Ordered by value per hour; none depends on Phase 0.

**1a. Epochs 20 vs 60** — one training run, judged on retention not loss. Both
logged runs plateau by epoch 7–12; confirming at 20 makes every later
experiment ~3× cheaper, which compounds through the rest of this plan. Then set
`epochs: 20` in config with the evidence comment.

**1b. Fix the tiled-inference blend.** Replace the Tukey overlap blend with
reflection-padded tiles + centre crop (each output pixel computed once, with
full context, no blend disagreement). Measure: artefact power at 64/512 px
before vs after, retention unchanged. This improves **every image the
observatory publishes**, including the ones already liked, and needs no
retraining.

**1c. Coverage-aware patch sampling.** `N2NDataset` currently samples patch
origins blind to the stacker's median-fill regions (open question, step 20).
Mask them. Small, clean, removes a known contamination from all future
training.

**1d. Fix pooled-path validation leakage** (holdout `dso`, not `dso|filter`).
Makes every future val number honest; three lines.

**1e. Port TensorBoard logging into the remaining trainers** so no future run
is blind — the epoch-plateau finding was invisible for a week purely for lack
of curves.

## Phase 2 — attack the band limit (0a ran 2026-08-23: headroom confirmed, retargeted)

Phase 0a's verdict (manual step 32): the 4-8 px band is **at or above** the
Wiener bound — closed, photons only — and 8-16 px is at the bound for p512 on
O-III. The real defect is **16-256 px**, where the scene is signal-dominated
(S/N 17-870, bound 0.95-0.999) and the model passes 0.72-0.93: headroom +0.07
to +0.27, largest at 16-64 px, on both channels measured. So Phase 2's
objective is the mid band, its loss weights are the measured per-band headroom,
and any fine-scale claim must first beat the bound in step 32.

The two levers step 29 names, neither ever tried, in this order:

**2a. Fine-scale-weighted loss.** Plain L2 on asinh pixels is dominated by
large-scale agreement; fine scales are mostly noise, so the loss barely pays
for keeping them. Replace with multi-scale L2 (Laplacian-pyramid or FFT-band
weighting, weights ∝ the Phase-0 headroom per band). One config, 2–3 seeds on
sh2-92 deep, judged by T(k) against the six existing baseline seeds. Cost ~4 h
at 20 epochs.

**2b. Fifth U-Net level** (receptive field ~140 → ~280 px). Patch 512 moved
exactly the bands near the current receptive field (step 30), which is the
strongest hint architecture sets the mid-band ceiling. Same protocol. Only run
if 2a leaves mid-band headroom unclaimed — the two overlap, and the loss change
is cheaper than the architecture change.

Explicitly **not** in this phase, with the step that killed each: training
target (27, 31), scene count (22), channel count (31), pooling scheme (25, 31),
source bias (28), self-training as production (31 — it bounds all of them).

## Phase 3 — productionise (only what survived)

- Retrain both `DOMAIN_MODELS` at patch 512 + whatever Phase 2 kept, 20 epochs,
  2 seeds. Promote **only if** paired-better than current production above the
  seed floor on both held-out fields; otherwise keep `pooledNB`, which six
  challengers have failed to beat.
- Re-run the routine path end-to-end on ngc6888 (HSO) and one broadband target;
  eyeball both, because every instrument here has been fooled at least once.
- Update manual (step 32+), runbook, and the "Current state" header the same
  day — it went nine days stale once already.

## Stop conditions

- Phase 0a says near-optimal → stop model work entirely; write it up as the
  closing result; the lever is photons.
- Phase 2 moves T(k) by less than the seed spread of the p512 baselines → stop;
  the architecture keeps its limit and the manual says so.
- Any intervention that wins on one test field and loses on the other is a
  no-promote, full stop.
