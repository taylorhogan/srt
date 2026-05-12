# Spec: `sentry/audio_mind` — Observatory Audio Intelligence

## Hardware

| Component | Recommendation | Notes |
|---|---|---|
| Device | Beelink Mini S12 Pro (Intel N100, 16GB) | x86, fanless-ish, ~10W, no ARM pip pain |
| Microphone | Blue Snowball iCE (USB) | Plug-and-play on Linux, cardioid, good for mechanical sounds |
| Storage | USB thumb drive (128GB+) mounted at `/mnt/audio` | Clips accumulate; prune configurable |

---

## Architecture

Three new files, one config section, no changes to existing code:

```
sentry/
  audio_mind.py       — main process: capture, classify, notify
  audio_catalog.py    — category store: load/save, centroid math, reclassify
  audio_server.py     — tiny FastAPI HTTP server serving clips/ directory
  clips/              — YYYY-MM-DD/HH-MM-SS.wav (prunable)
  categories.json     — persistent category + centroid store
```

---

## Layer 1 — Capture

- **Ring buffer**: 1 second of pre-roll audio always in memory.
- **Onset**: RMS crosses `rms_threshold` → start capturing. Post to web chat immediately: `[14:33] Sound detected — listening...`
- **Offset**: RMS stays below threshold for `silence_timeout` seconds (default 3s, per-category once labeled) → clip is complete.
- Clip = `pre_roll (1s) + active audio + post_roll (1s)`, capped at `max_clip_seconds` (default 30s) to handle sustained sounds like fans.
- Clip saved to `clips/YYYY-MM-DD/HH-MM-SS.wav`.
- **Onset/offset are separate events** — onset posts immediately, offset posts the categorized result.

---

## Layer 2 — Feature Extraction

MFCC-based embedding via librosa (already installed, fast enough on x86):
- 40 MFCC coefficients → compute mean and std across time → 80-dimensional feature vector.
- L2-normalize before storing or comparing.
- Sufficient to distinguish different fans (distinct motor frequency), roof motor, camera cooling hum, birds, planes, roomba.

*(Can swap in `panns_inference` CNN14 later for richer semantic embeddings if needed — the catalog interface is the same.)*

---

## Layer 3 — Auto-Categorization

On each completed clip:
1. Extract embedding.
2. Compute cosine similarity against every category centroid.
3. **If best match ≥ `similarity_threshold` (default 0.80):** assign clip to that category, update centroid (rolling mean of all member embeddings).
4. **If no match:** create `unknown_NNN` (auto-incremented), this clip as first member.
5. Post to web chat (see Layer 5).
6. Publish MQTT event.

**Reclassification cascade:** When user moves clip X from `cat_A` → `cat_B`:
- Remove X's embedding from `cat_A` centroid, add to `cat_B` centroid.
- Re-evaluate every remaining clip in `cat_A` against all centroids (one pass, no recursion).
- Auto-move any clip that now matches a different category better.
- Post summary: `Reclassified 1 clip; 2 others in cat_A re-evaluated, 1 moved to fan_intake.`

---

## Layer 4 — State & Alerting

Each category has an optional `state_label` (e.g., `"roof_moving"`, `"fan_exhaust"`, `"bird"`, `"plane"`).

**MQTT publish on every detection:**
```json
topic: srt/audio/state
payload: {"label": "fan_exhaust", "category": "cat_003", "confidence": 0.91, "clip": "clips/2026-05-12/143301.wav", "onset": true}
```
Separate message for offset with `"onset": false`.

**Pushover alert** if `state_label` is in config `alert_if_unexpected` list AND the scheduler did not initiate an action that would explain it (e.g., `roof_moving` heard when scheduler isn't in `IMAGING` transition state).

---

## Layer 5 — Web Chat Integration

**On onset** (immediate):
```
[14:33] Sound detected — listening...
```

**On offset** (with playback):
```
[14:33] fan_exhaust (0.94) — <audio controls src="http://pi:8096/clips/..."> [reclassify]
```
or for unknown:
```
[14:33] New sound: unknown_004 — <audio controls src="..."> [name this]
```

Sent via existing `post_message(text=..., html=...)` — zero changes to message bus or chat.html. Audio element works because the `html` field is rendered as innerHTML.

The Pi runs `audio_server.py` (5-line FastAPI static mount) on port 8096. Pi's IP stored in config as `audio_server_url`.

---

## Layer 6 — Web Chat Commands

| Command | Action |
|---|---|
| `audio status` | List all categories with last-heard timestamp |
| `audio categories` | List all categories with clip count, label, silence timeout |
| `audio name <cat_id> <label>` | Assign state label to a category |
| `audio reclassify <clip_id> <cat_id>` | Move clip, cascade re-evaluate |
| `audio new <clip_id>` | Promote clip to a new category |
| `audio merge <cat1> <cat2>` | Merge two categories, recompute centroid |
| `audio silence <cat_id> <seconds>` | Set per-category silence timeout |
| `audio threshold <value>` | Adjust RMS onset threshold live |
| `audio prune <days>` | Delete clips older than N days |

`audio status` output example:
```
fan_exhaust (cat_003): last heard 4 min ago
fan_intake (cat_007): last heard 2h 14m ago
roof_moving (cat_001): last heard 3 days ago
bird (cat_005): last heard 18 min ago
unknown_012: last heard 1h ago
[8 categories total]
```

---

## Config Addition (`configs/config_public.py`)

```python
"audio": {
    "enabled": True,
    "rms_threshold": 0.06,
    "default_silence_timeout": 3.0,
    "similarity_threshold": 0.80,
    "pre_roll_seconds": 1.0,
    "max_clip_seconds": 30.0,
    "clips_dir": "/mnt/audio/clips/",
    "categories_file": "sentry/categories.json",
    "audio_server_url": "http://iris-audio:8096",
    "audio_server_port": 8096,
    "alert_if_unexpected": ["roof_moving"],
    "prune_after_days": 30,
}
```

---

## Deployment

- `audio_mind.py` and `audio_server.py` run as two systemd units on the Beelink at boot.
- MQTT connects to Mosquitto on the NVIDIA box; retries with backoff if broker is offline at startup.
- Web chat posts go via HTTP POST to `http://nvidia-box:8095/api/post` (same path the scheduler uses).
- Thumb drive mounted at `/mnt/audio`, clips and categories written there.

---

## Out of Scope (for now)

- Training a classifier from labeled categories (after enough labeled clips accumulate).
- Real-time streaming audio to the web chat (clips only, not live feed).
- Multi-microphone support.
