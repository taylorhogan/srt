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

2. **Scheduler Server** (`end_points/scheduler_server.py`) — State machine that runs daily: waits for noon → checks best DSO for tonight → waits for pre-sunset → generates NINA sequence → triggers imaging. States: `WAITING_FOR_NOON → NOON_CHECK → WAITING_FOR_PRE_SUNSET → PRE_SUNSET_CHECK → IMAGING`.

They communicate via **MQTT** (`paho-mqtt`). Admin push notifications go via **Pushover** (`utils/pushover.py`, rate-limited to 6/min).

## Hardware Safety Rules

These rules are absolute and must be enforced in any code that moves observatory hardware:

- **Never move the scope (mount) unless the roof is confirmed open.** Moving the mount while the roof is closed risks a collision that could destroy the telescope.
- **Never move the roof unless the scope is confirmed parked.** Toggling the roof motor while the mount is tracking or slewing risks the scope intersecting the roof travel path.

Before writing any code that calls `toggle_roof()`, `pwi4.mount_park()`, `pwi4.mount_goto()`, or any other hardware-moving function, verify the precondition is met — either via `vision_safety.visual_status()` or `pwi4_utils.get_is_parked()` — and abort with a logged warning if it is not.

### Key Subsystems

- **`configs/config.py`** — Merges `PublicConfig` + `PrivateConfig`. Every module calls `config.data()` to get a flat dict. Private credentials live in `configs/config_private.py` (gitignored).

- **`iris_astronomy/`** — Astronomy logic: DSO visibility windows, air mass, best imaging night, weather (Open-Meteo API, no key needed), sunrise/sunset via `astral`.

- **`control/instructions.py`** — JSON-backed queue of DSO image requests (`my_instructions.json`). Sorted by status → priority → hours above horizon. Each instruction has: `dso`, `requestor`, `status` (waiting/in process/completed), `above_horizon`, `air_mass`, `best` (best date).

- **`hardware_control/`** — TP-Link Kasa smart plug control (`kasa_utils.py` wraps `kasa_local/`), Shelly HTTP relay control, PWI4 mount control (`pwi4_client.py` + `pwi4_utils.py`). Device discovery via Kasa UDP.

- **`sentry/`** — Safety systems: `vision_safety.py` uses OpenCV template matching on an indoor camera snapshot to detect scope parked/roof open/closed. `audio_classify.py` records roof-move audio, generates mel spectrograms, and classifies each move via MSE against the known-good library in `sentry/roof_audio/good/{open,close}/` (built by labeling captures with the webchat `audio <open|close> <good|bad>` command).

- **`nina_gen/nina_sequence_gen.py`** — Generates N.I.N.A imaging sequences by recursively patching a JSON template with target name and RA/Dec coordinates.

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

- `my_instructions.json` — DSO imaging queue
- `my_calendar.json` — per-day imaging history (state + DSO)
- `iris.log` — application log
- `safety.txt` — written by end sequence to record observatory state
- `sentry/roof_audio/{unlabeled,good,bad}/{open,close}/` — roof-move audio spectrograms/WAVs; `good/` is the anomaly-detection reference library
