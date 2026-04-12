# Iris Commands

All commands are sent via the web chat interface at `http://<tailscale-ip>:8095/`.

## Standard Commands

Available to all users in the Super Users list.

| Command | Description | Example |
|---------|-------------|---------|
| `tonight` | Show tonight's best imaging target with altitude plot, sky chart, weather forecast, and DSS2 preview | `tonight` |
| `best <dso>` | Look up the best night to image a DSO — reports hours above horizon, date, and air mass. Automatically queues it if above horizon for 3+ hours | `best m 31` |
| `image <dso>` | Add a DSO to the imaging queue | `image ngc 6888` |
| `status` | Show observatory state: parked/roof position (with camera snapshot), scheduler state, mode, safety, and imaging state | `status` |
| `latest` | Post the most recent FITS frame as a JPEG with FWHM/eccentricity stats, heatmaps, distance plot, and elongation angle map | `latest` |
| `schedule` | Generate a NINA imaging sequence for tonight's best object | `schedule` |
| `calendar` | Post this month's imaging calendar | `calendar` |
| `show <dso>` | Fetch and post a DSS2 Red survey image of a DSO | `show ngc 891` |
| `version` | Show current software version and observatory state | `version` |
| `help` | List all available commands | `help` |

## Super User Commands

Restricted to super users. These control observatory hardware and the imaging pipeline.

### Safety

| Command | Description | Example |
|---------|-------------|---------|
| `safe!` | Mark conditions as safe for imaging. Required before any imaging run can start | `safe!` |
| `stop!` | Mark conditions as unsafe — aborts any in-progress imaging run at the next safety gate | `stop!` |

### Imaging

| Command | Description | Example |
|---------|-------------|---------|
| `image!!` | Start a full imaging run (safety checks, roof open, NINA prelude, main imaging). Append `1`, `2`, or `3` to select mode: 1/2 = full run with different NINA sequences, 3 = home and park only | `image!! 1` |
| `doflats` | Run a standalone flats sequence — powers on mount, launches NINA flats, waits for completion, powers off mount | `doflats` |
| `stats` | Post per-frame FWHM and eccentricity graph (colour-coded by filter) for last night's frames. Use `stats full` to analyse all frames for the same DSO | `stats` or `stats full` |

### Mode

| Command | Description | Example |
|---------|-------------|---------|
| `mode <auto\|manual>` | Set scheduler mode. `auto` lets the scheduler trigger imaging runs automatically; `manual` requires explicit `image!!` commands | `mode auto` |

### Database (Imaging Queue)

| Command | Description | Example |
|---------|-------------|---------|
| `dbr` | Rehash the imaging queue and regenerate the instructions table | `dbr` |
| `dbd <id>` | Delete an entry from the imaging queue by ID | `dbd 12` |
| `dbc <id>` | Mark an imaging queue entry as completed | `dbc 1` |
| `dbb` | Rehash and rebuild the entire imaging queue from scratch | `dbb` |
| `prioritize <dso>` | Give a DSO top scheduling priority. With no argument, resets all priorities | `prioritize m 31` |

### Sequence Generation

| Command | Description | Example |
|---------|-------------|---------|
| `sequence <dso>` | Generate a NINA imaging sequence for a specific DSO with RA/Dec and filter plan | `sequence m 31` |

### Announcements

| Command | Description | Example |
|---------|-------------|---------|
| `announce <speaker> <text>` | Say text on a Sonos speaker | `announce bedroom the roof is opening` |
