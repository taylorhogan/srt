# Noise2Noise on the Spark — runbook

Everything N2N runs **only on the Spark** (`spark-3129`). `subs_dir` is defined
for that host alone in `configs/config_public.py`, and the Windows observatory
PC has no torch in its `.venv`. Nothing here can be tested on the observatory
machine.

Paths on the Spark: repo `/home/taylor/Documents/srt`, frames
`/home/taylor/Desktop/Targets`.

The 06:00 cron (`scripts/spark_morning_search.bsh`) does the rsync from the
observatory and then the transit search. **N2N is not wired into it** — this is
a manual chain.

---

## 0. Get the code and check the GPU

```bash
cd /home/taylor/Documents/srt && git pull
source .venv/bin/activate
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

## 1. Smoke test before the real run

The 2026-08-11 repairs changed the dataset path and **have never been executed
anywhere** — there is no torch on the machine they were written on. Prove the
pipeline runs before spending 130 epochs on it.

There is no `--epochs` flag. The value comes from `cfg["nn"]["epochs"]` in
`configs/config_public.py`. Lower it to ~5, run step 2, confirm it completes and
writes a checkpoint, then put it back.

## 2. Train

```bash
python scripts/n2n_train.py <filter> <seconds>     # e.g. R 300
```

Needs at least 20 frames spread over at least 2 DSOs that have ≥2 frames each.

**Write down the line it prints:**

```
Held-out DSOs for validation: <names> (N/M frames); training on K
```

Step 4 is meaningless without it.

Any existing checkpoint for this filter/exposure is deleted first, and older
ones will fail to load anyway — `load_state_dict` is strict, and all three of
the fixes below changed the model or its inputs. That failure is intended.

## 3. Denoise

```bash
python scripts/n2n_denoise.py <filter> <seconds>
```

Output goes to a flat `denoised/{filter}_{seconds}s/` containing every DSO,
keyed on bare frame names. Checked 2026-08-11: 1189 LIGHT frames, zero name
collisions — but it is a real hazard if NINA's naming ever restarts per target.

## 4. Evaluate — on a held-out DSO from step 2

```bash
python scripts/n2n_evaluate.py <dso> <filter> <seconds>
```

Measuring on a target the model trained on answers a question nobody asked.
Plots land in `local/n2n_eval/`.

### Reading the result

Two measurements, because they fail in different ways.

**Convergence.** `decay_ratio` says how far the curve's tail sits above what
*independent* noise predicts. ~1.0 is textbook 1/√N.

| outcome | meaning |
| --- | --- |
| denoised decay_ratio ≈ raw's, RMSE lower | genuinely bought frames |
| denoised decay_ratio ≫ raw's | bias shared across frames — stacking will never remove it |
| curve flattens to a floor raw does not have | same, and worse |

The reason this is the right test: every frame passes through the same network
with the same learned prior, so whatever it gets wrong it gets wrong the same
way on all of them. Independent noise averages down as √N; a shared bias
survives stacking untouched.

**Photometry.** Aperture flux, raw stack vs denoised, binned by brightness.
Sources are detected on the **raw** stack only — detecting separately would
compare different star lists and hide a source the denoiser erased.

Any bin below ~0.97 is a systematic brightness error. The predicted failure is
specific and faint-end only: the training loss is L1, whose optimum is the
conditional *median*, and for a source near the detection limit the posterior
mass sits at background — so an L1-trained net is structurally inclined to erase
faint sources while leaving bright ones alone.

The instrument was checked against synthetic ground truth: unbiased input reads
0.9996–1.0016 in every bin; 20% faint suppression reads bright 0.9999, mid
0.804, faint 0.800.

---

## What changed on 2026-08-11 (commit 91b43b9)

Three bugs, all of which invalidate older checkpoints:

1. **Registration used one global reference** — the first frame of the first
   DSO — for every frame of every target. A translation-only fit capped at
   200 px cannot align targets degrees apart. Frames of every DSO but the first
   were either silently unregistered or shifted by a spurious median, and
   nothing in the output distinguished the two. Now per-DSO, matching training.
2. **Train/inference normalisation had drifted apart again.** `c617047` added
   background subtraction to inference; training still normalised the raw frame.
   Measured scale error 1.0–1.7×, and *field-dependent* (m13 R 1.73×, ngc5907 L
   1.08×, sh2-92 Ha 1.03×, bubble Ha 0.98×). Both paths now share
   `denoiser.subtract_background()` and `denoiser.normalise()`.
3. **BatchNorm removed** (`UNet(norm=...)`, default `"none"`, still switchable).
   The N2N paper's network has none; this is a regression where the absolute
   output level carries the photometry; and BN bakes training activation
   statistics into running means, which amplifies exactly the shift in (2).

## Standing rule

Denoised frames are a **display** product until the gate above passes.
Calibrated linear frames stay the science product for the transit work, the
colour–magnitude diagram, and the transient search. A denoiser that makes an
image prettier and one that preserves photometry are different things.

## Cost to expect

Dataset construction now runs one `sep.Background` per frame and holds
normalised copies alongside the originals — roughly 2× frame memory. Fine on
128 GB, but visible.
