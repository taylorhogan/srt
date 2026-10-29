# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**SRT (Social Robotic Telescope)** — an autonomous observatory controller for "Iris". Users request deep-sky object (DSO) images via a self-hosted web chat interface (accessible over Tailscale); the system optimizes nightly imaging, controls hardware, monitors safety, and posts results. Mastodon mirroring is available as an optional feature. The goal is fully unattended operation.

## Setup

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtualenv (Python 3.13+ required, per pyproject.toml).
# Use uv's default .venv path — `uv pip` auto-discovers it; a differently
# named venv will be silently ignored by uv and cause import errors.
uv venv --python 3.13
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Create private config (required before running anything)
cp configs/config_blank_private.py configs/config_private.py
# Then fill in API keys/tokens in config_private.py
```

## Running

```bash
# Start both Social Server and Scheduler together
python end_points/start_srt.py

# Or run individually:
python end_points/scheduler_server.py   # Nightly state machine
python cmd_processing/social_server.py  # Web chat server

# Observatory startup/shutdown sequences
python end_points/start.py   # Power on mount, lights off
python end_points/end.py     # Park scope, close roof, dehumidifier on

# Safety check before imaging
python end_points/goforimagecheck.py

# Audio anomaly detection (run in observatory)
python sentry/audio_classify.py
```

## Architecture

The system has two long-running processes launched by `end_points/start_srt.py`:

1. **Social Server** (`cmd_processing/social_server.py`) — Runs a FastAPI/WebSocket web chat server on port 8095. Parses commands (`image`, `best`, `tonight`, `status`, `calendar`, `latest`, etc.) and dispatches them. Only responds to users listed in `Super Users` config. Posts status/images back to the web chat (and optionally mirrors to Mastodon via `mastodon_mirror` config flag).

2. **Scheduler Server** (`end_points/scheduler_server.py`) — State machine that runs daily: waits for noon → checks best DSO for tonight → waits for pre-sunset → generates NINA sequence → triggers imaging. States: `WAITING_FOR_NOON → NOON_CHECK → WAITING_FOR_PRE_SUNSET → PRE_SUNSET_CHECK → IMAGING`. With `conductor.drives_checks` on (ADR 0015, built 2026-10-16, off until the owner flips it) the conductor's clock makes the noon and sunset decisions (`iris/conductor/clock.py`, journaled as `NIGHT_PLAN`, served by `GET /v1/plan`) and the scheduler only carries them out (`driven_cycle`): it never ranks to decide, and no plan from the conductor means no night.

They communicate via **MQTT** (`paho-mqtt`). Admin push notifications go via **Pushover** (`utils/pushover.py`, rate-limited to 6/min).

## Hardware Safety Rules

These rules are absolute and must be enforced in any code that moves observatory hardware:

- **Never move the scope (mount) unless the roof is confirmed open.** Moving the mount while the roof is closed risks a collision that could destroy the telescope.
- **Never move the roof unless the scope is confirmed parked.** Toggling the roof motor while the mount is tracking or slewing risks the scope intersecting the roof travel path.

**The one exception (operator decision 2026-09-17): stop! may park a scope whose roof it cannot see.** At 00:33 that night the unparked scope hid the scope tag, the roof tag and the gold star; stop! refused to park and left the mount *tracking* under an unconfirmed roof. `super_user_commands.blind_park_refusal` replaces "confirmed open now" with "confirmed open earlier and provably not moved since": mount powered, PWI4 says unparked, and it was connected and TRACKING at a plausible altitude (tracking implies the main sequence's FindHome ran; slewing alone does not, since FindHome slews; the park never connects or homes); frames returned and none read the roof tag at shut; a gating read confirmed OPEN within 16 h and nothing powered or fired the roof after it (`sentry/roof_evidence.py`); roof motor plug reads off; no roof move in progress. Tracking is stopped first either way. Only stop! uses this, and the roof still closes only on a fresh vision read after the park. Limit switches were dropped by the operator on 2026-09-25 ("I feel confident that the cameras are working"): the north camera, which sees the roof tag at both positions regardless of where the scope is, is the second roof sense, and the evidence file remains the blind-park witness for the rare read where neither camera answers.

Before writing any code that calls `toggle_roof()`, `pwi4.mount_park()`, `pwi4.mount_goto()`, or any other hardware-moving function, verify the precondition is met — either via `vision_safety.visual_status()` or `pwi4_utils.get_is_parked()` — and abort with a logged warning if it is not.

Since 2026-09-25 the conductor decides both invariants (ADR 0014): every roof fire asks through `_ask_conductor` / `iris.client.request_roof_move` (binding, `conductor.roof_authority`), and every mount move or mount power-on asks through `_ask_mount` / `iris.client.request_mount` (`MOUNT_MOVE_REQUESTED` / `MOUNT_POWER_REQUESTED`; binding since 2026-10-09, `conductor.mount_authority`). An unreachable conductor refuses, with two designed exceptions that must happen even if the brain died: end.py's roof close and stop!'s park (`_ask_mount(..., emergency=True)`); a conductor that answers no is obeyed even there. `stop!` posts `ESTOP_REQUESTED` first, and only `resolve!` releases the hold. New code that moves or powers the mount must ask; the table in `iris/core/machine.py` is where the rule lives.

### Key Subsystems

- **`cmd_processing/jobs.py`** — Job registry behind the web chat's cards. Long commands run in their own OS process via `jobs.spawn_process`; the child binds the job id so its `/api/post` messages route to that card. Those children hold one end of a parent-death pipe and exit the moment the server does — `update` restarts via `os._exit`, so without it a stack outlives the restart, keeps saturating the disk, and posts to a card that no longer exists. Posts carrying an unknown job id are labelled `[orphan <id>]` in the Observatory feed rather than blending in.

- **`configs/config.py`** — Merges `PublicConfig` + `PrivateConfig`. Every module calls `config.data()` to get a flat dict. Private credentials live in `configs/config_private.py` (gitignored).

- **`iris_astronomy/`** — Astronomy logic: DSO visibility windows, air mass, best imaging night, weather (Open-Meteo API, no key needed; since 2026-09-30 cloud cover comes from the NWS grid, Open-Meteo per-hour fallback, `cfg["weather"]["cloud_source"]`; an hour is too cloudy above `cfg["weather"]["max_cloud_pct"]` (40), chosen by `scripts/forecast_score.py` against the sky camera), sunrise/sunset via `astral`.

- **`control/instructions.py`** — JSON-backed queue of DSO image requests (`my_instructions.json`). Sorted by status → priority → hours above horizon. Each instruction has: `dso`, `requestor`, `status` (waiting/in process/completed), `above_horizon`, `air_mass`, `best` (best date).

- **`hardware_control/`** — TP-Link Kasa smart plug control (`kasa_utils.py` wraps `kasa_local/`), Shelly HTTP relay control, PWI4 mount control (`pwi4_client.py` + `pwi4_utils.py`). Device discovery via Kasa UDP.

- **`sentry/`** — Safety systems. `vision_safety.visual_status()` is the single vision entry point for every mount/roof decision; since 2026-09-07 it answers from the inside Kasa camera via `kasa_state.py`: AprilTag id 0 on the scope against the recorded park pose (`local/scope_marker_parked.json`) and id 1 on the roof panel against the shut reference (`local/roof_marker_shut.json`), with the scope tag as witness that the camera can see. Each gating read drives the camera to the reference pan/tilt pose (cloud round trip; unreachable = unknown = refuse), switches the inside light on, waits for the camera to leave greyscale IR, and takes 3 frames: UNSAFE in any frame is not parked; SHUT needs one decoded roof tag at its position; OPEN (since 2026-09-11) needs every frame open-like (roof tag gone, scope tag seen, aperture not objecting) AND the gold star on the wall seen in at least one frame at its recorded position (`local/roof_star_open.json` + template, recorded by `scripts/roof_star_check.py --set-open`; the star is hidden by the roof structure when shut). A tag that fails to decode looks like a roof that is not there, and a false open is the collision case, so absence alone never means open. The scope-top webcam and its USB mic were retired that day; the template ladder in `vision_safety.py` is kept only to replay archived exposure sets. The camera has ONE stream: roof-move audio (`kasa_audio.py`, library `sentry/roof_audio_kasa/good/{open,close}/`, 8 kHz, never mixed with the old 44.1 kHz `roof_audio/`) holds it for ~50 s per move and the vision grab waits for it. Label leftovers with the webchat `audio <open|close> <good|bad>` command.

- **Roof state is decided by the north camera (since 2026-09-24).** `sentry/north_roof.py` reads AprilTag id 2 on the moving south truss from "Iris North" (192.168.87.78, DHCP-reserved; home pose (102, −9), driven there before every read) and compares its corners with two recorded positions in `local/north_roof_marker.json`: SHUT (72 px, low in frame) and OPEN (351 px, high), 650 px apart. Both states are positive decodes; absence is only ever *unknown*; a part-way roof lands between the references and is *unknown*. It needs no inside light (decodes in pure IR) and is read FIRST in `visual_status()`, before the light is touched, because Iris North takes ~9 s to swap IR/colour and a frame inside the swap decodes nothing. `north_roof.decide()` combines it with Iris cam: north decides, Iris cam (tag 1 at shut; the gold star for open) may veto — disagreement is unknown — and is the fallback when the north camera cannot answer. `cfg["camera safety"]["north_decides"]` False reverts to Iris cam deciding. The picture the operator sees (`status`, every Pushover push of `scope_view`) is the north frame that decided, tag and verdict drawn on; Iris cam's own picture is `cam_view`. Every north frame is archived under `local/north_archive/<date>/`. The gold-star description in the `sentry/` bullet above is now the fallback path, not the decider.

- **`nina_gen/nina_sequence_gen.py`** — Generates N.I.N.A imaging sequences by recursively patching a JSON template with target name and RA/Dec coordinates. Since 2026-09-12 it also writes a `MoveFocuserByTemperature` (absolute: slope × T + intercept) before every SmartExposure block, per filter, from `fits_processing/focus_model.py` — a fit of N.I.N.A's own autofocus reports refit at every generation (`local/focus_model.json`). The seed centres the autofocus sweep and is what N.I.N.A restores when a run fails; `cfg["nina"]["focus_seed"] = False` turns it off.

- **`stacking/`** — `stacker.py` stacks LIGHT frames: bias/dark/flat calibration
  (masters cached as `.npy` in scratch, keyed on the calibration frames' mtimes),
  a shared quality gate, astroalign registration, and a tiled sigma-clip combine
  that streams frames to disk so full resolution does not need the whole cube in
  RAM. `color_process.py` backs the `process <dso> <recipe>` command (LRGB / HALRGB /
  HOO / SHO; HALRGB adds the Hα *excess* — Hα minus its measured continuum
  share of R — into R and L, never the raw Hα): it stacks each filter against **one shared reference** so the channels
  land on the same pixels, then stretches them on a **shared** brightness scale so
  the ratio between channels survives into the image. Both of those are easy to
  get wrong in ways that still produce a plausible-looking picture.
  `deblur=N` deconvolves before the stretch and is ON by default (`deblur=40`;
  `deblur=0` turns it off). Since the afternoon of 2026-10-21 the default
  method is `learned` (`stacking/deblur.deblur_channels` -> `nn/deblur_model`):
  the luminance goes through the flux-moving network trained on re-blurred
  stacks of this telescope, by the field's own kernel, and the colour planes
  stay as stacked (per-plane deconvolution made coloured noise specks and a
  cyan star core); measured on the holdouts it gives stars 7.5 -> ~4 px with
  no rings and the galaxy improved at every scale
  (`docs/LEARNED_DEBLUR_PLAN.md`); chosen by the owner by eye on ngc6946 and
  ngc1961. Since 2026-10-23 the checkpoint is the kernel-core model
  `scatter_both_taper24_s0`: the first model removed the measured kernel's
  whole far wing from every bright star and left the halo beyond its 145 px
  stamp, so each bright star sat in a dark ring inside its own halo (worst
  on narrowband); the model is now trained to remove only the kernel inside
  2-4 FWHM (`nn/deblur_data.taper_kernel`, `KERNEL_TAPER`, stamped on the
  checkpoint, applied at inference), which took the 32-60 px wing removal
  from 63 % to 7 % and the moat with it, star flux within 12 px from 1.15x to
  1.09x, and keeps 85 % of the galaxy's 6-12 px sharpening (a 1.5-3 taper,
  the default for one night, kept 65 %); promoted by the owner by eye on
  ngc1961, trunk and ic1396. A set with no
  luminance (HOO/SHO) gets a synthetic one (`stacking/deblur.synthetic_luminance`:
  the palette's luminance weights on the mapped planes), sharpens that, and
  the composite is lit by the ratio of the sharpened L to its unsharpened
  self (`NO_L_RULE="synthetic"`, `--no-l starless` is the old rule, written
  as `deblurred`); the planes themselves are never put through the network:
  the nebula sweep of 2026-10-21 (ic1396, squid, ngc7380, trunk) did that and
  it sharpened each plane's own noise into specks, broke bright rims into
  beads and gave saturated stars mottled halos. Without torch or the
  checkpoint it falls back to starless with a warning. `method=starless` (the default 2026-10-19 to 10-21) is
  `stacking/starless.py`: every channel's stars are taken off
  (`deblur.star_layer`), the luminance is Richardson-Lucy'd with the field's
  own kernel on what is left, and the lifted-off star layer is put back as
  imaged (`stars=imaged`, default since the evening of 2026-10-19; `stars=points`
  renders them back as 5 px points at each channel's own flux, colour from
  matched apertures against L). Nothing RL sees is a point source, so the
  stars cannot ring, halo or moat; peaks broader than the kernel at 4-6 px
  (nuclei, HII knots) are left to RL, except that a BRIGHT one (2000 ADU
  above the floor) goes back as imaged after RL, because a close double
  reads as a knot and RL rings it (NGC 1961's 4 px pair). `method=rl` is the earlier
  whole-luminance RL in `stacking/deblur.py`: RL rings around every star
  unless the sky is out, the floor follows the galaxy without clipping, and a
  dark clamp holds — the module doc has the measurements — and on a denoised
  stack every bright star still paid with some artefact, which is why the
  stars are no longer deconvolved at all. 40 iterations is the working count
  (stars converge by 15, galaxies need 40; split-half test says the added
  detail is real). On a denoised stack the order is stack → denoise → deblur:
  the denoiser erodes faint stars' wings and spikes, so the denoised stack has
  no single PSF; the spikes are fitted off star by star and RL runs with the
  azimuthal kernel. Deblurring the raw stack and denoising after was measured
  worse (2026-10-17). The morning render (`scripts/n2n_lrgb_render.py routine`)
  writes three composites per recipe — raw, denoised, and since the afternoon
  of 2026-10-21 `learned` (the denoised set with its luminance through the
  learned deconvolver, `--learned-planes L`; checkpoint
  `local/deconv/models/learned_default.pt`); the starless `deblurred` one is
  an optional extra (`--deblur N`) — on one stretch pinned
  to the raw set; a field a deconvolution refuses still gets the others.
  Only ONE of them is pushed (since 2026-10-28, owner's choice): the learned
  composite, or the next class down when a field refused the deconvolution
  (`PUSH_TAGS`); `--push-all` sends the triple again. A narrowband target
  gets SHO with `--gain hubble` when S-II has 12 frames and HOO beside it
  (both since 2026-10-29, owner's choice; the day before it was one
  palette), HOO alone otherwise; HSO is gone
  (`spark_morning_render.RECIPE_RULES`, `PALETTE_GAIN`). Known weak case: a bright star on a smooth galaxy's envelope
  (coloured rings where the per-channel masks meet the fill, NGC 7317).

  **The morning render cuts every channel to the common footprint (since
  2026-10-28).** The stacker crops a lone stack to where 80 % of its frames
  landed, but not one built onto a shared reference (the box differs per
  filter); outside the footprint the stack is a flat fill at the GLOBAL sky
  level, which at a vignetted edge sits above the real sky beside it, and a
  palette gain made that strip a red bar along the top of every IC 410
  render. `stacker.stack` now reports `coverage_box` either way, the render
  records it per filter in the meta, and `crop_common` cuts every channel to
  the intersection before anything is composed. A stack cached before this
  has no box and is not cut on its account; `--force` rebuilds it. The
  stack's record (box, accepted count, frame lists) is written beside the
  npy as `<dso>_<filter>_raw.json` and read from there on a cache hit
  (`stack_record`, since 2026-10-29), falling back to the newest entry
  naming the file in any recipe's meta: it belongs to the stack, not the
  recipe, and the HOO after the morning's SHO rebuild, reading only its own
  day-old meta, rendered uncropped.

  Frames are **levelled to a common sky before combining** and the level is
  restored afterwards. This is not cosmetic: sky varies 3x across nights (moon,
  transparency), and combining at native levels turns that spread into per-pixel
  noise, because sigma-clip rejects a different subset at every pixel and those
  frames sit at different levels. Measured on ngc5907 G: 8x the photon limit at
  39 frames, worse than a single frame, and it got worse with more frames — so
  it hides in any small test.

  **A palette can be toned on purpose: `gain=` (since 2026-10-28).** The
  shared scale makes a true-ratio SHO of an HII region green, because Ha is
  3-5x S-II and O-III (IC 410: S-II 0.2-0.3 of Ha, O-III 0.3-0.5). `gain=hubble`
  on `process` (`--gain hubble` on the morning render) scales every palette
  filter so its nebula level matches the strongest's, which is the classic
  each-channel-on-its-own-range Hubble palette; `gain=Ha:0.4/O-III:1.5` is
  explicit. The level is the 90th percentile of 32 px block medians
  (`nebula_level`): a pixel percentile is the stars, which are the same in
  every narrowband filter (first try on IC 410 measured 1.00/1.02/1.04 and
  changed nothing), and the top half percent of blocks is the brightest
  knot, not the nebula (IC 410: S-II x2.4 and green tadpoles at p99.5,
  x3.6 and gold at p90). Factors are capped at 6. Applied in `_prepare` on
  the subtracted planes; the shared white is taken BEFORE the gain, because
  the white is the stars and a scaled-up filter would carry it up and darken
  the picture (white 124 -> 266 ADU on IC 410), and the sky-anchored black
  offset is sized by the PRE-gain sigmas (`sky_anchored_blacks(gains=)`),
  because a filter scaled up carries its noise up and the offset sized to
  that put the sky at 30 % grey (offset 1.3 -> 4.8 ADU). Measured on the raw set and
  pinned on the denoised and learned composites (`info["palette_gain"]`,
  `shared_from_raw`), printed in the options line. HOO/SHO/HSO only: on
  LRGB/HALRGB it refuses. The stars do not take the gain and come out
  NEUTRAL (`PALETTE_STAR_MODE`): scaling R and B made every star magenta,
  and putting each plane's own star light back kept them magenta, because
  a star is fainter through the Ha filter (its own absorption line) than
  through S-II and O-III and Ha is SHO's green. So the stars are found once
  on the raw set (`star_mask` on the planes' sum, 3 sigma, a peak/flux
  compactness cut so a knot is never a star, discs sized by the star's peak
  so a star on nebula does not inherit the knot's isophote) and the discs
  pinned on the denoised and learned sets (a bright star found again on the
  quiet denoised plane outgrew the size limit, fell out of the mask, and
  the gain painted its halo magenta); inside a disc the fill under the
  star (`smooth_fill`, block medians with the mask excluded, interpolated
  from the neighbours) takes the gain and the star's light above it,
  averaged over the planes, goes back the same in every channel. The
  FAINT stars are found on the denoised planes (`join_quiet_stars`,
  `_prepare(palette_quiet=)`, the morning render passes its denoised set):
  at 3 sigma of the raw sum Melotte 15 gave 2614 stars and every fainter
  one took the x4 S-II gain and came out as a red speck over the whole
  field ("many red stars", 2026-10-28); the denoised sum, a tenth of the
  noise, gives 7360. Joined to the raw list, not swapped for it (8-14 of
  the raw's 60 brightest fall out of the quiet pass), and the quiet discs
  are sized from the RAW floor (`star_mask(peak_floor=)`), or every star
  reads a decade brighter and the discs cover 40 % of the field (10 %
  joined). A BRIGHT star's disc grows a halo term past 2.5 dex
  (`radius_from_dex`, 146 px at 4.2 dex, cap 160, skirt capped at 24 px):
  the brightest star's halo reached 108 px at 1 ADU and its 56 px disc
  left a magenta ring. Its diffraction spikes get four arms
  (`spike_axis`, `spike_arms`, `paint_arms`): the axis is measured on the
  field, an azimuthal profile 0.5-1.5 r around the biggest stars folded
  modulo 90 (67 deg on Melotte 15; the spider's angle on the sensor, so
  never a constant), the length grows with the star (40 + 100 (d-2.5)^2
  px, cap 400), and a field whose folded profile has no peak gets none.
  Outside the mask a spike took the gain and was red ("difraction spike
  are red", 2026-10-28). The arms are pinned on the series with the discs
  (`info["palette_arms"]`).

  **The sky-anchored black is one offset for every channel (since 2026-10-25).**
  The morning render and the movie put black at each channel's sky median
  minus `--black-sigma` (0.5) times the LARGEST channel's noise sigma
  (`color_process.sky_anchored_blacks`), so the sky lands at one grey level in
  every channel. With each channel offset by its own sigma the shallowest
  channel's sky sat highest: m1 SHO on 5 S-II subs against 10 Ha and 10 O-III
  put the S-II sky 1.33 ADU above black against 1.03 and 0.95, and the 2.5x
  saturation turned that into a brown sky (RGB 54/32/25). A balanced set is
  unchanged.

  **Calibration matches the lights' camera gain (since 2026-10-24).** The
  QHY600M writes GAIN and OFFSET into every frame; the lights' majority
  (GAIN, OFFSET) is the stack's gain (`stacker.select_gain`), lights at any
  other gain are rejected with the quality-gate rejects, and bias, darks and
  flats at another gain are dropped (`stacker.match_calibration`), loudly,
  with the stack running without a term it cannot match rather than with
  the wrong one. A frame with no GAIN header (the 2024 archive) matches
  anything. This exists because narrowband is moving to gain 26 (the
  sensor's high-conversion-gain mode) while broadband may stay at 0:
  `cfg["nina"]["gain"]` is `{filter: [gain, offset]}` and the sequence
  generator writes it into every exposure block it names for that filter
  (`nina_sequence_gen.apply_filter_gain`, -1/-1 for every other block,
  verified on write). The flats run from their own sequence on the
  observatory PC (`scripts/nina_flats.bat` -> `flat_sequence.json`, Trained
  Flat Exposure per filter, exposure from the profile's trained table keyed
  by filter and gain), so a gain change there means the same gain in those
  instructions and the Flat Wizard re-trained at it.

  Flats live per session as `<flat_root>/<date>/FLAT` with every filter mixed
  together, so they are grouped by FITS `FILTER` header rather than by directory
  (`flats_by_filter`). They accumulate at ~1.2 GB per filter per night and are
  highly redundant — masters from 2026-07-14 and 2026-07-31 agreed to 0.2–0.3%
  RMS — so the `purge` command deletes all but the newest
  `cfg["disk"]["flat_keep_sets"]` (3) sets per filter (`purge keep N` overrides;
  the stacker combines every flat on disk with no date matching, so a single
  kept set would have no fallback if that night's flats were bad). It is
  dry-run unless given `go`, and is the only command that deletes data outright
  (`bad` renames). Irreversible: older lights lose the option of epoch-matched
  flats, which matters only if the optics were disturbed between runs.
  `purge bad [go]` does the same for frames `bad` rejected, once marked more
  than `cfg["disk"]["bad_keep_days"]` (30) ago; `bad` stamps the mark time on
  the renamed file. Cached masters unused for 30 days are deleted whenever a
  new one is built (`stacker.prune_master_cache`; a cache hit refreshes the
  mtime). Raw lights are never pruned: `utils/disk_space` makes the morning
  check warn below 100 GB free and `image!!` refuse below 25 GB.

  Every picture product (stack/process/channel/sweep JPEGs) is written with
  **sky parity** through `stacker.sky_parity()`: the array as stored, row 0 at
  the top. On this rig the FITS-convention view is a mirror of the sky
  (positive CD determinant on a solved light), and until 2026-09-12 every
  JPEG was flipped into it. The rule is fixed by the optical train, not read
  from a WCS; the FITS products keep the stored orientation.

  Finished products from `stack` and `process` go to
  `<image_dir>/Iris/<dso>/`, beside that target's lights and calibration frames
  rather than into scratch. Safe because every path that collects light frames
  requires the parent directory to be named `LIGHT`, so FITS written under
  `Iris/` cannot be picked up as data.

- **`fits_processing/`** — FITS to JPEG conversion, FWHM analysis, header editing.

- **`kasa_local/`** — Local copy of python-kasa library (used instead of the pip package for local modifications).

### Config Structure

Config keys commonly used across modules:
- `cfg["location"]` — lat/lon, timezone, city, file paths for instructions/image grid
- `cfg["camera safety"]` — camera image paths, template paths, parked/open/closed positions and tolerances
- `cfg["nina"]` — NINA image directory, sequence template/output paths
- `cfg["globals"]` — runtime objects (mastodon instance, mqtt client, logger)
- `cfg["pushover"]` — token/user for push notifications
- `cfg["web_chat"]` — web chat server settings (port, host, mastodon_mirror, max_history)
- `cfg["mastodon"]` — access token, API base URL (only used when mastodon_mirror is True)

### Path Convention

Every module that can be run directly adds the project root to `sys.path` with this pattern:
```python
if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
```

### Data Files (gitignored)

- `local/my_instructions.json` — DSO imaging queue (path from `cfg["location"]["instructions"]`; the copy at the repo root is stale)
- `my_calendar.json` — per-day imaging history (state + DSO)
- `iris.log` — application log
- `safety.txt` — written by end sequence to record observatory state
- `sentry/roof_audio/{unlabeled,good,bad}/{open,close}/` — roof-move audio spectrograms/WAVs; `good/` is the anomaly-detection reference library
