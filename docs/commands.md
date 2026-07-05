# Iris Commands

All commands are sent via the web chat interface at `http://<tailscale-ip>:8095/`.
Type `help` (or `?`) in the chat for this list, and `help <command>` for details
on one command. The single source of truth is `cmd_processing/help_registry.py` —
when adding a command, add its registry entry and update this file.

## General Commands

Available to all users in the Super Users list.

| Command | Description | Example |
|---------|-------------|---------|
| `help` / `?` | Show command list, or detailed help for one command | `help snr` |
| `tonight` | Show tonight's best DSO with weather and sky chart | `tonight` |
| `best <dso>` | Report when the named DSO is best imaged (rises, air mass, hours) | `best m 13` |
| `bestradec <ra> <dec>` | Best night for an explicit RA/Dec; with a name, queues it for imaging | `bestradec wr134 20:10:14 +36:10:35` |
| `image <dso>` | Add a DSO to the imaging queue | `image m 31` |
| `db` | Display the current imaging queue | `db` |
| `version` | Post the running SRT version string | `version` |
| `status` | Post observatory status (roof, mount, scheduler, weather) | `status` |
| `latest` | Post the most recently captured FITS as an annotated JPEG | `latest` |
| `schedule` | Generate a NINA sequence for tonight's best object | `schedule` |
| `calendar` | Post the per-day imaging history calendar image | `calendar` |
| `show <dso>` | Fetch and post a sky-survey preview of a DSO | `show ngc 891` |
| `speedtest` | Run an internet speed test and post results | `speedtest` |
| `history [n]` | Show recent command history | `history 10` |

## Super User Commands

Restricted to super users. These control observatory hardware and the imaging pipeline.

### Safety & Hardware

| Command | Description | Example |
|---------|-------------|---------|
| `safe!` | Mark conditions as safe for imaging (writes USER SAFE to safety.txt) | `safe!` |
| `stop!` | Emergency stop: kill NINA, park the scope, close the roof, shut down | `stop!` |
| `roof!!` | Move or report the roof (`!!` = hardware moves). `open`/`close` are vision-safety-checked (scope must be parked); the relay only toggles, so direction follows current position. Append `force` to skip the parked check (DANGEROUS) | `roof!! status`, `roof!! open`, `roof!! close force` |
| `audio` | List unlabeled roof-move audio captures, or label the latest (or named) one good/bad — also files the matching motor-current signature from the same move under the same verdict | `audio`, `audio open good`, `audio close bad` |
| `announce <speaker> <text>` | Say text on a Sonos speaker | `announce Observatory roof closing in one minute` |

### Imaging

| Command | Description | Example |
|---------|-------------|---------|
| `image!! <dso>` | Start a full imaging run for a DSO right now | `image!! m 31` |
| `doflats` | Run a flat-frame capture sequence in the background | `doflats` |
| `sequence <dso>` | Generate a NINA sequence file for the named DSO | `sequence m 31` |
| `mode <auto\|manual>` | Set the scheduler to auto or manual mode | `mode auto` |

### Frame Analysis & Science

| Command | Description | Example |
|---------|-------------|---------|
| `stats [dso] [all]` | Per-frame FWHM, eccentricity, sky-brightness, and star-count graph for the latest session (`all` for full history) | `stats m31 all` |
| `snr [dso]` | Stack-convergence (RMSE vs frame count) curves per filter | `snr m31` |
| `transit <dso> <filter>` | Search saved subs for transit-like dips on every star in the field | `transit m31 L` |
| `transient <dso> <filter>` | Difference the newest night against prior nights to find new sources (supernova candidates). Best on a galaxy. Alias: `diff` | `diff ngc5907 L` |
| `hr <dso> [blue] [red]` | Gaia-calibrated colour–magnitude (H–R) diagram from two filters. Best on a star cluster | `hr m13 B R` |
| `optics [dso\|*] [n]` | Optical-quality diagnostic plots for a FITS frame | `optics m31 3` |
| `drift [dso\|*]` | ZScale difference images: first-k-frames stack vs golden (L filter) | `drift m31` |
| `stack [dso] [filter]` | Stack all LIGHT frames of a DSO (per filter) and post each as JPEG | `stack m31 ha` |
| `bad [dso] [go]` | Flag (and with `go`, rename) LIGHT frames that fail per-filter median quality thresholds | `bad m31 go` |
| `dab [dso] [go]` | Restore frames previously flagged by `bad` back to active | `dab m31 go` |
| `active` | Per-DSO tiles: a date×filter grid of how many subs were taken | `active` |

### Database (Imaging Queue)

| Command | Description | Example |
|---------|-------------|---------|
| `dbr` | Rehash the imaging queue and regenerate the instructions table | `dbr` |
| `dbd <id>` | Delete an entry from the imaging queue by ID | `dbd 17` |
| `dbc <id>` | Mark an imaging queue entry as completed | `dbc 17` |
| `dbb` | Rebuild the imaging queue from scratch (rehash + recreate table) | `dbb` |
| `prioritize [dso]` | Give a DSO top scheduling priority, or reset all to equal priority | `prioritize ngc 7331` |

### System

| Command | Description | Example |
|---------|-------------|---------|
| `log [n]` | Post the last N lines of iris.log | `log 50` |
| `update` | Pull latest code from git and restart the server | `update` |
| `todo [text]` | Show or append items to the project todo list | `todo investigate Ha gradient on M31` |
