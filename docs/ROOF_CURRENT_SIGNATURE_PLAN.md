# Roof Current-Signature Anomaly Detection — Working Plan

_Last updated: 2026-06-22. Paused (raining — no hardware testing). Resume tomorrow._

## Goal
Watch the **current signature** of the roof motor each time the roof moves,
compare it against known-good signatures, and flag anomalies (jam, ice,
failing motor, early stop) — without running a full imaging session to do it.

## The hardware (confirmed)
- Device at **192.168.87.46** is a **Shelly 3EM Gen 3** (`S3EM-002CXCEU`, app `EMG3`)
  — **NOT a Gen 1**. So it uses the Gen 2/3 **RPC API**, not the Gen 1
  `/status` + `/relay/0` REST that the roof/dehumidifier relays use.
- Running in **EM1 (independent-channel) mode**: channels **0 and 1** exist
  (`id=2` returns 500). Each channel has its **own voltage** measurement → good
  for split-phase. 50 A clamps (`ct_type "50AEMG3"`).
- Read endpoint: `GET http://192.168.87.46/rpc/EM1.GetStatus?id=<channel>`
  → `{voltage, current, act_power, aprt_power, pf, freq}`.
- Sustained polling measured at **~29 Hz** (~32 ms/read) — plenty for a dense
  trace across the ~45 s roof travel.

## How the roof moves (context)
`cmd_processing/super_user_commands.py:toggle_roof()` — powers the "Roof motor"
Kasa plug, waits 10 s, fires the Shelly relay (momentary; direction depends on
current position), `sleep(45)` for travel, powers the motor off. The move is
**command-triggered with a known window**, so capture is aligned to the command
— no need to detect motion from the current itself.

## Chosen method (decided)
1. **Capture** at ~20 Hz across the travel window (triggered by the roof command).
2. **Represent** as a `(t, current, power, voltage)` trace + scalar features.
3. **Compare** — Tier A scalar envelope now; Tier B curve-shape (DTW) later.
4. **Library** of known-good runs, separate per direction (open vs close differ).
5. **Action** — log/alert on anomaly now; real-time stall cutoff is a later phase.

Scalar features (in `extract_features`): baseline_w, peak_w/peak_a,
**running_w/running_a** (median while moving), **move_duration_s**,
**energy_ws** (∫P·dt), **returned_to_baseline**. Anomaly = any feature outside
the good runs' `mean ± 3σ` (15% relative fallback when variance ≈ 0), plus a
"did not return to baseline" check. Reasons are reported in plain language.

## What's already built (committed in 887ca5e)
- `configs/config_public.py` — added `current_monitor_url` (192.168.87.46) and
  `current_monitor_channel` (0) to the `hardware` block.
- `hardware_control/utl_shelly.py` — `read_current_monitor(channel=None)`:
  Gen 2/3 RPC reader, returns the status dict or None.
- `scripts/power_monitor.py` — live power, **prints only when it changes**
  (`--channel`, `--interval`, `--threshold`). Already tested, works.
- `sentry/roof_current_signature.py` — capture / features / save / load /
  compare + background-capture helpers + CLI
  (`capture | features | compare | label | list`). Validated on synthetic
  traces (normal passes, struggling-motor trips with correct reasons).
- `cmd_processing/super_user_commands.py` — `toggle_roof` now auto-captures
  (best-effort daemon thread, can't disrupt the roof sequence) and gained an
  optional `capture_direction` label arg.
- `scripts/cycle_roof.py` — **lightweight single roof cycle + capture**, gated
  on the **visual** parked check (`vision_safety` via `get_status_with_lights()`)
  so it works without the mount powered/connected; refuses if scope not parked
  (`--force` overrides). Does NOT start NINA/imaging.
- `.gitignore` — ignores `sentry/roof_signatures/` (runtime data).

## TOMORROW — do this
1. **Move the CT clamp** from the mount to the **roof-motor feed** (single 120 V
   conductor). Confirm with: `.venv/bin/python scripts/power_monitor.py` — you
   should see the draw jump when the motor runs.
2. **Cycle the roof** a few times (scope parked!):
   ```
   .venv/bin/python scripts/cycle_roof.py --direction open
   .venv/bin/python scripts/cycle_roof.py --direction close
   ```
   Each banks an `unlabeled` signature and prints its features.
3. **Inspect** each trace:
   ```
   .venv/bin/python sentry/roof_current_signature.py features <file>
   ```
4. **Label the clean ones good** (need ≥2 per direction for the envelope):
   ```
   .venv/bin/python sentry/roof_current_signature.py label <file> --good
   ```
5. After ≥2 good per direction, anomaly detection is **live** in `toggle_roof`
   (logs a warning) and via `... roof_current_signature.py compare <file>`.

## Open decisions (pick up tomorrow)
- **Direction labeling:** captured as `unknown` by default (the relay just
  toggles; `toggle_roof` doesn't know open vs close). `cycle_roof.py
  --direction` labels it manually. Option: auto-infer from `vision_safety`
  roof state before/after. **Decide:** manual label vs vision inference.
- **Commit:** DONE — committed in `887ca5e` (epochs already reverted to 130,
  NN params unchanged). Not yet pushed.
- **`sentry/power_classify.py`:** old standalone prototype (hardcoded IP, 5 s
  poll, matplotlib/CSV). Superseded by `roof_current_signature.py`. **Decide:**
  keep or delete.

## Later / phase 2
- **Tier B curve comparison** via DTW once a few real good/bad traces exist.
- **Real-time stall cutoff:** if running current exceeds a ceiling for >N s
  during a move, immediately cut the "Roof motor" Kasa plug (active protection).
- **Pushover alert** on anomaly (`utils/pushover.py`), and/or mark observatory
  unsafe.

## Git state (as of 2026-06-22)
All the work above is committed in **`887ca5e`** (`epochs` reverted to 130, NN
params unchanged). **Not yet pushed** — last pushed is `2ba083d`. Working tree
is clean.

## Unrelated loose end
- The **500-epoch n2n R 300 retrain** is still running in the background
  (was at epoch ~230/500). It was a one-off experiment; the committed default
  is 130. Best-checkpoint saving means the saved `n2n_R_300s.pt` is safe
  regardless. Config default is back to 130 (committed); the running process
  keeps its own loaded value (500) until it finishes.
