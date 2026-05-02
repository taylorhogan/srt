# Iris Observatory — Social Robotic Telescope

Iris is a fully autonomous deep-sky observatory. Users request objects through a chat interface; the system plans the night, controls the hardware, monitors safety, and delivers calibrated images — all without anyone in the observatory.

---

## What it does

**Chat-driven requests**
Users submit deep-sky object (DSO) requests through a web chat interface accessible over Tailscale. The system queues, prioritises, and schedules them automatically.

**Nightly planning**
Each day at noon the scheduler scores every queued target by visibility window, air mass, and priority, then selects the best object for that night.

**Hardware control**
Controls the roof motor, mount, focuser, and smart plugs. Sequences power-on, slew, autofocus, imaging, flats, park, and roof-close without human intervention.

**Safety monitoring**
Computer vision checks the indoor camera before every hardware move. If the scope isn't parked the roof won't move; if the roof isn't open the mount won't slew.

**Live frame analysis**
Every incoming FITS frame is analysed for FWHM, eccentricity, star count, and sky brightness. Results stream to the chat ticker in real time.

**Image delivery**
Finished frames are converted to JPEG and posted back to the chat. Per-session stats plots, convergence curves, and sky heatmaps are generated on demand.

---

## A night in the life

1. **Noon check** — The scheduler wakes up, recalculates visibility for every queued DSO, picks tonight's best target, and generates a N.I.N.A imaging sequence for it.

2. **Pre-sunset checks** — Weather forecast, safety file, and camera snapshot are verified. A Sonos announcement warns anyone in the observatory that the roof is about to open.

3. **Roof opens, prelude runs** — Vision safety confirms the scope is parked before toggling the roof motor. N.I.N.A runs a prelude sequence: cool camera, slew, plate-solve, and autofocus.

4. **Main imaging sequence** — N.I.N.A captures light frames across all configured filters. The frame watcher analyses each sub as it arrives and streams quality metrics to the chat.

5. **Flats and shutdown** — At the end of the imaging window, flat frames are captured for calibration. The mount parks, the roof closes, the dehumidifier turns on, and a summary is posted to chat.

---

## Chat commands

| Command | Description |
|---|---|
| `image <dso>` | Queue a DSO for imaging |
| `best` | Best target for tonight |
| `tonight` | Full tonight's plan |
| `status` | Observatory status |
| `latest` | Most recent image |
| `stats` | Per-frame quality plots |
| `active` | All DSOs with frame counts |
| `calendar` | Imaging history |
| `schedule` | Generate tonight's sequence |
| `sequence <dso>` | Generate sequence for DSO |
| `log` | Recent log entries |
| `safe! / stop!` | Mark safe or abort |

---

## Optical quality analysis

The `optics` command runs a full per-star analysis on any LIGHT frame and posts diagnostic plots and a scalar scorecard to the chat. Each star is fitted with a 2-D Gaussian to extract its position, FWHM, eccentricity, and major-axis angle.

**FWHM & field uniformity**
Median full-width at half-maximum across all detected stars, in arcseconds. A grid heatmap shows whether sharpness degrades toward the edges — a sign of field curvature or focus tilt.

**Eccentricity**
How elongated each star is (0 = perfect circle, 1 = line). High eccentricity across the whole field indicates poor collimation, mirror shift, or tracking error.

**Coma score**
Pearson correlation between eccentricity and distance from the image centre. A high score means stars get progressively more elongated toward the edges — the classic signature of coma.

**Collimation score**
Measures whether elongated stars point radially outward from the centre (coma/collimation) or in a consistent direction across the frame (tilt or flexure). Uses the mean cos² of the angle between each star's elongation axis and its radial direction.

**Tilt score**
Fits a plane to the FWHM values across the sensor. The gradient magnitude, normalised by image diagonal and median FWHM, reveals focus-plane tilt — one side sharp, the other soft.

**FWHM vs radius plot**
Scatter plot of FWHM against distance from the image centre. Flat = good optics. A rising slope indicates field curvature; a U-shape suggests the best focus is off-axis.

---

## Safety rules

> **The mount will never move unless the roof is confirmed open. The roof will never move unless the scope is confirmed parked.**
>
> Every hardware action is gated by a computer-vision check of the observatory camera. If the camera image is stale or ambiguous, the action is aborted and a push notification is sent.

---

## Audio anomaly detection

A microphone in the observatory runs a continuous listen loop. Any sound above the RMS threshold — a motor stall, a mechanical bang, an unexpected relay click — triggers a 10-second capture, which is converted to a mel spectrogram and compared against a library of known-good reference sounds.

**Continuous listening**
Audio is sampled at 44 100 Hz in 1 024-sample chunks. The RMS level of each chunk is checked against a configurable threshold. Below threshold the system is silent; above it a capture begins immediately.

**Mel spectrogram fingerprint**
Each 10-second capture is rendered as a 128-band mel spectrogram image using a fixed figure size, DPI, and colour map so all images are pixel-comparable regardless of when they were recorded.

**Library matching**
The new spectrogram is compared to every PNG in the reference library using mean-squared error. The closest match and its similarity score are identified. The library is built by saving spectrograms of known sounds — roof motor running, normal ambient noise, etc.

**Push notification on new sounds**
A Pushover notification is sent whenever a new distinct sound is detected — i.e. the best-match label differs from the previous event. Repeated instances of the same sound are suppressed to avoid alert fatigue. The spectrogram image is attached to the notification.

**Hardware covered**
The microphone is positioned to hear the roof motor, the mount drive, and the observatory door. Unusual sounds during an imaging run — a motor stall mid-travel, a mechanical impact, or unexpected silence where motor noise should be — are flagged in real time.

**Detected archive**
Every triggered capture is saved as both a PNG spectrogram and a WAV file in `detected_spectrograms/`, creating a full audit trail of every acoustic event the observatory experienced during a run.

---

## Key subsystems

| Module | Role |
|---|---|
| `scheduler_server.py` | State machine: noon check → pre-sunset → imaging → shutdown |
| `social_server.py` | FastAPI/WebSocket chat server on port 8095 |
| `nina_sequence_gen.py` | Patches N.I.N.A JSON templates with target coords and filter plans |
| `vision_safety.py` | OpenCV template matching — parked / roof open / roof closed |
| `frame_watcher.py` | Background thread: FWHM, eccentricity, sky brightness per frame |
| `kasa_utils` / `pwi4` | TP-Link Kasa smart plugs, Shelly relay, PlaneWave mount |
| `astro_dso_visibility.py` | Visibility windows, air mass, weather via Open-Meteo |
| `pushover.py` | Rate-limited admin push notifications via Pushover |

---

*Iris Observatory · Social Robotic Telescope · fully unattended operation*
