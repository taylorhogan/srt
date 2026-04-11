# Iris Commands

All commands are sent as Mastodon mentions to Iris.

## Standard Commands

Available to all users in the Super Users list.

| Command | Description | Example |
|---------|-------------|---------|
| `tonight` | Show tonight's best imaging target with altitude plot, sky chart, weather forecast, and DSS2 preview | `@iris tonight` |
| `best <dso>` | Look up the best night to image a DSO — reports hours above horizon, date, and air mass. Automatically queues it if above horizon for 3+ hours | `@iris best m 31` |
| `image <dso>` | Add a DSO to the imaging queue | `@iris image ngc 6888` |
| `status` | Show observatory state: parked/roof position (with camera snapshot), scheduler state, mode, safety, and imaging state | `@iris status` |
| `latest` | Post the most recent FITS frame as a JPEG with FWHM/eccentricity stats, heatmaps, distance plot, and elongation angle map | `@iris latest` |
| `schedule` | Generate a NINA imaging sequence for tonight's best object | `@iris schedule` |
| `calendar` | Post this month's imaging calendar | `@iris calendar` |
| `show <dso>` | Fetch and post a DSS2 Red survey image of a DSO | `@iris show ngc 891` |
| `version` | Show current software version and observatory state | `@iris version` |
| `help` | List all available commands | `@iris help` |

## Super User Commands

Restricted to super users. These control observatory hardware and the imaging pipeline.

### Safety

| Command | Description | Example |
|---------|-------------|---------|
| `safe!` | Mark conditions as safe for imaging. Required before any imaging run can start | `@iris safe!` |
| `stop!` | Mark conditions as unsafe — aborts any in-progress imaging run at the next safety gate | `@iris stop!` |

### Imaging

| Command | Description | Example |
|---------|-------------|---------|
| `image!!` | Start a full imaging run (safety checks, roof open, NINA prelude, main imaging). Append `1`, `2`, or `3` to select mode: 1/2 = full run with different NINA sequences, 3 = home and park only | `@iris image!! 1` |
| `doflats` | Run a standalone flats sequence — powers on mount, launches NINA flats, waits for completion, powers off mount | `@iris doflats` |
| `stats` | Post per-frame FWHM and eccentricity graph (colour-coded by filter) for last night's frames. Use `stats full` to analyse all frames for the same DSO | `@iris stats` or `@iris stats full` |

### Mode

| Command | Description | Example |
|---------|-------------|---------|
| `mode <auto\|manual>` | Set scheduler mode. `auto` lets the scheduler trigger imaging runs automatically; `manual` requires explicit `image!!` commands | `@iris mode auto` |

### Database (Imaging Queue)

| Command | Description | Example |
|---------|-------------|---------|
| `dbr` | Rehash the imaging queue and regenerate the instructions table | `@iris dbr` |
| `dbd <id>` | Delete an entry from the imaging queue by ID | `@iris dbd 12` |
| `dbc <id>` | Mark an imaging queue entry as completed | `@iris dbc 1` |
| `dbb` | Rehash and rebuild the entire imaging queue from scratch | `@iris dbb` |
| `prioritize <dso>` | Give a DSO top scheduling priority. With no argument, resets all priorities | `@iris prioritize m 31` |

### Sequence Generation

| Command | Description | Example |
|---------|-------------|---------|
| `sequence <dso>` | Generate a NINA imaging sequence for a specific DSO with RA/Dec and filter plan | `@iris sequence m 31` |

### Announcements

| Command | Description | Example |
|---------|-------------|---------|
| `announce <speaker> <text>` | Say text on a Sonos speaker | `@iris announce bedroom the roof is opening` |
