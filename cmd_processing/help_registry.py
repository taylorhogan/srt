"""Structured help for every chat command.

Single source of truth for what each command does, the arguments it accepts,
and how to invoke it. Read by the `help` and `help <command>` handlers.

Each entry:
    category : "general" | "super"
    summary  : one-line description
    usage    : list of "<cmd> <args>" forms
    examples : (optional) list of concrete invocations
"""

from typing import Optional


_ALIASES: dict[str, str] = {
    "?": "help",
    "diff": "transient",
}


HELP: dict[str, dict] = {
    # ------------------------------------------------------------------ general
    "help": {
        "category": "general",
        "summary": "Show command list, or detailed help for one command.",
        "usage": ["help", "help <command>", "?", "? <command>"],
        "examples": ["help", "help snr", "? transit"],
    },
    "tonight": {
        "category": "general",
        "summary": "Show tonight's best DSO with weather and sky chart.",
        "usage": ["tonight"],
    },
    "best": {
        "category": "general",
        "summary": "Report when the named DSO is best imaged (rises, air mass, hours).",
        "usage": ["best <dso>"],
        "examples": ["best m 13", "best ngc 891"],
    },
    "bestradec": {
        "category": "general",
        "summary": "Best night for an explicit RA/Dec; with a name, queues it for imaging.",
        "usage": ["bestradec <ra> <dec>", "bestradec <name> <ra> <dec>"],
        "examples": ["bestradec 12:30:49 +12:23:28", "bestradec wr134 20:10:14 +36:10:35"],
    },
    "image": {
        "category": "general",
        "summary": "Add a DSO to the imaging queue.",
        "usage": ["image <dso>"],
        "examples": ["image m 31", "image ngc 7331"],
    },
    "db": {
        "category": "general",
        "summary": "Display the current imaging queue.",
        "usage": ["db"],
    },
    "version": {
        "category": "general",
        "summary": "Post the running SRT version string.",
        "usage": ["version"],
    },
    "status": {
        "category": "general",
        "summary": "Post observatory status (roof, mount, scheduler, weather).",
        "usage": ["status"],
    },
    "latest": {
        "category": "general",
        "summary": "Post the most recently captured FITS as an annotated JPEG.",
        "usage": ["latest"],
    },
    "schedule": {
        "category": "general",
        "summary": "Generate a NINA sequence for tonight's best object.",
        "usage": ["schedule"],
    },
    "calendar": {
        "category": "general",
        "summary": "Post the per-day imaging history calendar image.",
        "usage": ["calendar"],
    },
    "show": {
        "category": "general",
        "summary": "Fetch and post a sky-survey preview of a DSO.",
        "usage": ["show <dso>"],
        "examples": ["show ngc 891"],
    },
    "speedtest": {
        "category": "general",
        "summary": "Run an internet speed test and post results.",
        "usage": ["speedtest"],
    },
    "history": {
        "category": "general",
        "summary": "Show recent command history (defaults to last few entries).",
        "usage": ["history", "history <n>"],
        "examples": ["history", "history 10"],
    },

    # --------------------------------------------------------------- super-user
    "image!!": {
        "category": "super",
        "summary": "Start a full imaging run for a DSO right now (super-user image).",
        "usage": ["image!! <dso>"],
        "examples": ["image!! m 31"],
    },
    "roof!!": {
        "category": "super",
        "summary": "Move or report the roof (!! = hardware moves). Movement now "
                   "requires safe! first, plus no imaging run and mount power off; "
                   "open/close are also vision-safety-checked (scope must be parked). "
                   "The relay only toggles, so direction follows current position. "
                   "Append 'force' to skip only the vision check (DANGEROUS). "
                   "status is read-only and needs no safe!.",
        "usage": [
            "roof!! status",
            "roof!! open",
            "roof!! close",
            "roof!! toggle",
            "roof!! open|close|toggle force",
        ],
        "examples": ["roof!! status", "roof!! open", "roof!! close", "roof!! toggle", "roof!! close force"],
    },
    "stop!": {
        "category": "super",
        "summary": "Emergency stop: kill NINA, park the scope, close the roof, shut down.",
        "usage": ["stop!"],
    },
    "safe!": {
        "category": "super",
        "summary": "Mark conditions as safe for imaging (writes USER SAFE to safety.txt).",
        "usage": ["safe!"],
    },
    "announce": {
        "category": "super",
        "summary": "Say text on a Sonos speaker in the observatory.",
        "usage": ["announce <speaker_name> <text...>"],
        "examples": ["announce Observatory roof closing in one minute"],
    },
    "sequence": {
        "category": "super",
        "summary": "Generate a NINA sequence file for the named DSO.",
        "usage": ["sequence <dso>"],
        "examples": ["sequence m 31"],
    },
    "mode": {
        "category": "super",
        "summary": "Set the scheduler to auto or manual mode.",
        "usage": ["mode auto", "mode manual"],
    },
    "prioritize": {
        "category": "super",
        "summary": "Give a DSO top scheduling priority, or reset all to equal priority.",
        "usage": ["prioritize", "prioritize <dso>"],
        "examples": ["prioritize", "prioritize ngc 7331"],
    },
    "filters": {
        "category": "super",
        "summary": "Set which filters a DSO is shot in, and how many exposures each.",
        "usage": ["filters <dso>", "filters <dso> <FILTER>=<count> ...",
                  "filters <dso> clear"],
        "examples": ["filters bubble O-III=40 Ha=10", "filters bubble",
                     "filters bubble clear"],
        "notes": ["Counts are taken literally, not scaled to the hours available.",
                  "Filters: L, R, G, B, S-II, O-III, Ha. Up to 3 per night.",
                  "Without a plan the sequence splits by object type as before: "
                  "nebulae equally across Ha/O-III/S-II, everything else L+RGB."],
    },
    "doflats": {
        "category": "super",
        "summary": "Run a flat-frame capture sequence in the background.",
        "usage": ["doflats"],
    },
    "todo": {
        "category": "super",
        "summary": "Show or append items to the project todo list.",
        "usage": ["todo", "todo <text...>"],
        "examples": ["todo", "todo investigate Ha gradient on M31"],
    },
    "active": {
        "category": "super",
        "summary": "Per-DSO tiles: a date×filter grid of how many subs were taken.",
        "usage": ["active"],
    },
    "stats": {
        "category": "super",
        "summary": "Post per-frame FWHM, eccentricity, sky-brightness, and star-count "
                   "graph for the latest session (add 'all' for full history, "
                   "'resky' to refresh cached sky values, 'rebuild' to re-analyse "
                   "every frame).",
        "usage": ["stats", "stats <dso>", "stats <dso> all",
                  "stats <dso> resky", "stats <dso> rebuild"],
        "examples": ["stats", "stats m31", "stats m31 all",
                     "stats m31 all resky", "stats m31 rebuild"],
    },
    "snr": {
        "category": "super",
        "summary": "Post stack-convergence (RMSE vs frame count) curves per filter.",
        "usage": ["snr", "snr <dso>"],
        "examples": ["snr", "snr m31"],
    },
    "transit": {
        "category": "super",
        "summary": "Search saved subs for transit-like dips on every star in the field.",
        "usage": ["transit <dso> <filter>"],
        "examples": ["transit m31 L", "transit ngc7331 Ha"],
    },
    "transient": {
        "category": "super",
        "summary": "Difference the newest night against prior nights to find new "
                   "sources (supernova candidates). Best on a galaxy. Alias: diff.",
        "usage": ["transient <dso> <filter>", "diff <dso> <filter>"],
        "examples": ["transient m101 r", "diff ngc5907 L"],
    },
    "hr": {
        "category": "super",
        "summary": "Build a Gaia-calibrated colour–magnitude (H–R) diagram from two "
                   "filters. Best on a star cluster.",
        "usage": ["hr <dso>", "hr <dso> <bluefilter> <redfilter>"],
        "examples": ["hr m13", "hr m13 B R"],
    },
    "log": {
        "category": "super",
        "summary": "Post the last N lines of iris.log.",
        "usage": ["log", "log <n>"],
        "examples": ["log 50"],
    },
    "ninalog": {
        "category": "super",
        "summary": "Post the last N lines of N.I.N.A's own log (default 5).",
        "usage": ["ninalog", "ninalog <n>"],
        "examples": ["ninalog", "ninalog 20"],
        "notes": ["Separate from `log`, which is iris.log.",
                  "This is the only place that says which sequence instruction "
                  "is running right now — look for 'Starting Category: ... "
                  "Item: ...' — without adding script calls to the sequence.",
                  "Newest log file by modification time, since N.I.N.A opens a "
                  "new one per process. Capped at 200 lines."],
    },
    "update": {
        "category": "super",
        "summary": "Pull latest code from git and restart the server.",
        "usage": ["update"],
    },
    "optics": {
        "category": "super",
        "summary": "Post optical-quality diagnostic plots for a FITS frame.",
        "usage": [
            "optics",
            "optics <dso>",
            "optics * <n>",
            "optics <dso> <n>",
        ],
        "examples": ["optics", "optics m31", "optics m31 3", "optics * 12"],
    },
    "drift": {
        "category": "super",
        "summary": "Post ZScale difference images: first-k-frames stack vs golden (L filter).",
        "usage": ["drift", "drift <dso>", "drift *"],
        "examples": ["drift", "drift m31"],
    },
    "skysolve": {
        "category": "super",
        "summary": "Re-solve the all-sky camera after it has been moved.",
        "usage": ["skysolve", "skysolve check"],
        "examples": ["skysolve", "skysolve check"],
        "notes": ["Needs a clear night with real stars -- a blind solve has "
                  "nothing to match against otherwise, and the frame is "
                  "rejected if the negative-image control says it is cloud or "
                  "rain.",
                  "The camera is normally solved once and left alone. Run this "
                  "when it has been physically disturbed: taken down, knocked, "
                  "or refocused.",
                  "Only saves if the new solve matches at least as many stars "
                  "as the stored one, and keeps the old file alongside. "
                  "`skysolve check` verifies and changes nothing.",
                  "Fixes the compass headings on the published frame, the "
                  "measured-region outline, and limiting magnitude."],
    },
    "purge": {
        "category": "super",
        "summary": ("Delete superseded flat frames, keeping the newest set per "
                    "filter. Dry run unless you add 'go'. Frees ~4 GB per "
                    "redundant night; irreversible, so older lights lose the "
                    "option of epoch-matched flats."),
        "usage": ["purge", "purge go"],
        "examples": ["purge", "purge go"],
    },
    "process": {
        "category": "super",
        "summary": ("Stack a DSO's filters and combine into a colour image at full "
                    "resolution. LRGB (L as luminance), HOO (Ha->R, O-III->G/B) or "
                    "SHO (S-II->R, Ha->G, O-III->B). All filters register to one "
                    "shared reference so the channels align. Add 'noflat' to skip "
                    "flat correction, which is worth trying when the only flats "
                    "available were shot in a different epoch. Display options: "
                    "black= white= soft= mesh= scnr= nobg scale=. scnr=0..1 is "
                    "average-neutral green removal, min(g,(r+b)/2) — good on "
                    "LRGB, leave off for HOO. Add 'reuse' to "
                    "re-render the cached channels in seconds instead of "
                    "re-stacking; give an option a comma list (black=45,55,65) "
                    "to sweep it and get a labelled contact sheet to pick from. "
                    "'auto' sweeps the standard black/white/soft grid (27 "
                    "variants, ~1 min) without enumerating it, leaving any axis "
                    "you set explicitly alone. "
                    "Saves the colour image, plus each stacked channel as linear "
                    "FITS (with WCS) and as a mono JPEG, to <image_dir>/Iris/<dso>/."),
        "usage": ["process <dso> <recipe>", "process <dso> <recipe> noflat",
                  "process <dso> <recipe> reuse black=50 mesh=6"],
        "examples": ["process sh2-92 hoo", "process abell2151 lrgb noflat",
                     "process abell2151 lrgb reuse black=50",
                     "process sh2-92 hoo reuse soft=0.01 nobg",
                     "process abell2151 lrgb reuse black=45,55,65,75",
                     "process ngc5907 lrgb reuse scnr=0,0.5,1",
                     "process sh2-92 hoo auto",
                     "process sh2-92 hoo auto black=55"],
    },
    "seen": {
        "category": "super",
        "summary": ("Chart of catalogue stars actually seen per night, one row "
                    "per night, newest at the top. The y-axis is the percent of "
                    "stars brighter than mag 5 in the sky camera's field that "
                    "were detected. Yellow shading marks periods the rain detector's "
                    "signal sat above its onset threshold; red where an alert "
                    "was actually sent. Grey dots are samples whose star "
                    "counts the pipeline itself distrusts (rain/cloud frames) — "
                    "so a starless night reads as weather (yellow/red) rather "
                    "than a camera fault. "
                    "History reaches back to 2026-08-08 via sky_log.jsonl."),
        "usage": ["seen", "seen <days>"],
        "examples": ["seen", "seen 7", "seen 14"],
    },
    "publish": {
        "category": "super",
        "summary": ("Re-render one variant from a `process <dso> <recipe> auto` "
                    "sweep at full resolution. The number is the one printed on "
                    "the panel of the contact sheet — that number exists to be "
                    "typed back here. The sweep itself renders binned 4x for "
                    "speed, so this is a genuine re-render from the cached "
                    "channels at native resolution, not an upscale of a panel. "
                    "The optional crop keeps the centre of the frame: 0.75 and "
                    "75 both mean the central 75% of width and height; omit it "
                    "for the full frame. Settings come from that sweep's own "
                    "record, so a sweep that pinned an axis (auto black=55) "
                    "republishes correctly. Writes a lossless PNG and a "
                    "downscaled JPG to <image_dir>/Iris/<dso>/. Needs a sweep on "
                    "record first — run `process <dso> <recipe> auto`."),
        "usage": ["publish <dso> <id>", "publish <dso> <id> <crop>"],
        "examples": ["publish ic1396 19", "publish ic1396 19 .75",
                     "publish sh2-92 7 75", "publish bubble 19 .75"],
    },
    "stack": {
        "category": "super",
        "summary": "Stack all LIGHT frames of a DSO (per filter) and post each as JPEG.",
        "usage": ["stack", "stack <dso>", "stack <dso> <filter>"],
        "examples": ["stack", "stack m31", "stack m31 ha"],
    },
    "bad": {
        "category": "super",
        "summary": "Flag (and optionally rename) LIGHT frames that fail per-filter median quality thresholds.",
        "usage": [
            "bad",
            "bad <dso>",
            "bad <dso> go",
            "bad go",
        ],
        "examples": ["bad m31", "bad m31 go"],
    },
    "dab": {
        "category": "super",
        "summary": "Restore frames previously flagged by `bad` back to active (reverse of `bad`).",
        "usage": [
            "dab",
            "dab <dso>",
            "dab <dso> go",
            "dab go",
        ],
        "examples": ["dab m31", "dab m31 go"],
    },
    "dbb": {
        "category": "super",
        "summary": "Rebuild the imaging queue from scratch (rehash + recreate table).",
        "usage": ["dbb"],
    },
    "dbr": {
        "category": "super",
        "summary": "Rehash the imaging queue and regenerate the instructions table.",
        "usage": ["dbr"],
    },
    "dbd": {
        "category": "super",
        "summary": "Delete an entry from the imaging queue by ID.",
        "usage": ["dbd <id>"],
        "examples": ["dbd 17"],
    },
    "dbc": {
        "category": "super",
        "summary": "Mark an imaging queue entry as completed by ID.",
        "usage": ["dbc <id>"],
        "examples": ["dbc 17"],
    },
    "live": {
        "category": "super",
        "summary": "Post TWO no-light sky views from the scope-top webcam (same "
                   "camera as vision safety): a low-gain long-exposure pass for "
                   "STARS and a high-gain pass for SKYGLOW / clouds. Lights stay "
                   "off; safe to run while imaging. An optional frame count "
                   "averages that many frames per pass to cut noise (default 1).",
        "usage": ["live [frames]"],
        "examples": ["live", "live 8"],
    },
    "audio": {
        "category": "super",
        "summary": "List unlabeled roof-move audio captures, or label the latest "
                   "(or named) one good/bad — also files the matching motor-current "
                   "signature from the same move under the same verdict.",
        "usage": ["audio", "audio <open|close> <good|bad>", "audio <open|close> <good|bad> <name>"],
        "examples": ["audio", "audio open good", "audio close bad 2026-07-03T18-21-04_close"],
    },
}


def _resolve(name: str) -> Optional[str]:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    return key if key in HELP else None


def get(name: str) -> Optional[dict]:
    """Return the help entry for a command name (resolving aliases). None if unknown."""
    key = _resolve(name)
    return HELP[key] if key else None


def format_command(name: str) -> str:
    """Format detailed help for one command."""
    entry = get(name)
    if entry is None:
        return (
            f"No help for '{name}'. Type 'help' for the list of commands."
        )
    key = _resolve(name) or name
    lines = [f"{key} — {entry['summary']}"]
    if entry.get("usage"):
        lines.append("Usage:")
        for u in entry["usage"]:
            lines.append(f"  {u}")
    if entry.get("examples"):
        lines.append("Examples:")
        for e in entry["examples"]:
            lines.append(f"  {e}")
    return "\n".join(lines)


def format_list(include_super: bool) -> str:
    """Format the full command list (general always; super-user if include_super)."""
    width = max(len(n) for n in HELP)
    gen = sorted((n, e) for n, e in HELP.items() if e["category"] == "general")
    sup = sorted((n, e) for n, e in HELP.items() if e["category"] == "super")
    lines = ["Commands (type 'help <name>' for details):", "", "General:"]
    for n, e in gen:
        lines.append(f"  {n.ljust(width)}  {e['summary']}")
    if include_super:
        lines.append("")
        lines.append("Super-user:")
        for n, e in sup:
            lines.append(f"  {n.ljust(width)}  {e['summary']}")
    return "\n".join(lines)
