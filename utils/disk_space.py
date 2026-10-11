"""Free disk space on the observatory PC: warn early, refuse a night that would fill it.

Raw frames land in cfg["nina"]["image_dir"], on C: with everything else, and
nothing prunes them. On 2026-10-11 the drive had 377 GB free of 930 GB, with
60-90 GB written a month since June (308 GB of it raw light frames). A disk
that fills mid-night fails N.I.N.A partway through a sequence, so (owner,
2026-10-11):

  * the 09:00 morning check warns below cfg["disk"]["warn_free_gb"] (100 GB)
    and reports a problem below refuse_free_gb;
  * image!! refuses to start below cfg["disk"]["refuse_free_gb"] (25 GB:
    about one long winter night of 122 MB frames plus the flats).

A drive that cannot be measured does not block a night: free space is a
capacity question, not a safety one, and the morning check reports it.
"""
import os
import shutil

WARN_FREE_GB = 100.0
REFUSE_FREE_GB = 25.0
BAD_KEEP_DAYS = 30.0      # `purge bad` deletes rejected frames marked longer ago than this
_GB = 1e9


def thresholds(cfg) -> tuple:
    """(warn_gb, refuse_gb) from cfg["disk"], with the defaults above."""
    d = (cfg or {}).get("disk", {}) or {}
    return (float(d.get("warn_free_gb", WARN_FREE_GB)),
            float(d.get("refuse_free_gb", REFUSE_FREE_GB)))


def data_path(cfg) -> str:
    """The directory whose drive matters: where N.I.N.A writes the frames.

    Walks up to the nearest existing ancestor (disk_usage needs a real path);
    falls back to this repository's drive."""
    p = ((cfg or {}).get("nina", {}) or {}).get("image_dir") or ""
    p = os.path.abspath(p) if p else ""
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    if p and os.path.exists(p):
        return p
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def usage(path) -> dict:
    """{"path", "free_gb", "total_gb"}, or None when the drive cannot be read."""
    try:
        u = shutil.disk_usage(path)
    except OSError:
        return None
    return {"path": str(path), "free_gb": round(u.free / _GB, 1),
            "total_gb": round(u.total / _GB, 1)}


def morning_status(u, warn_gb: float, refuse_gb: float) -> tuple:
    """(problem, warning) for the morning check, either None. Pure."""
    if u is None:
        return None, "could not read the free space on the image drive"
    free = u["free_gb"]
    if free < refuse_gb:
        return ("disk nearly full: %.0f GB free on the image drive -- image!! refuses "
                "below %.0f GB; free space before tonight" % (free, refuse_gb)), None
    if free < warn_gb:
        return None, ("disk getting full: %.0f GB free of %.0f GB (warns below %.0f GB, "
                      "image!! refuses below %.0f GB)" % (free, u["total_gb"], warn_gb, refuse_gb))
    return None, None


def image_refusal(u, refuse_gb: float):
    """Why image!! must not start, or None. Pure. Unreadable = do not block."""
    if u is None or u["free_gb"] >= refuse_gb:
        return None
    return ("only %.0f GB free on the image drive, and a night needs %.0f GB -- "
            "free space, then run image!! again" % (u["free_gb"], refuse_gb))


def bad_keep_days(cfg) -> float:
    return float(((cfg or {}).get("disk", {}) or {}).get("bad_keep_days", BAD_KEEP_DAYS))


def stale_bad_frames(image_dir, keep_days: float, now: float = None) -> list:
    """[(path, bytes)] of rejected light frames marked more than *keep_days* ago.

    A rejected frame is `<frame>.fits.bad` directly under a LIGHT directory
    (what `bad` renames to, and what `dab` restores from). `bad` stamps the
    rename time on the file since 2026-10-11, so the age is the time since it
    was marked; a frame marked before then still carries its capture time.
    """
    import time
    from pathlib import Path
    cutoff = (time.time() if now is None else now) - float(keep_days) * 86400.0
    out = []
    for p in sorted(Path(image_dir).rglob("*.fits.bad")):
        if p.parent.name.upper() != "LIGHT":
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime < cutoff:
            out.append((p, st.st_size))
    return out


def check_image_run(cfg):
    """image_refusal for the configured drive, read now."""
    _warn, refuse = thresholds(cfg)
    return image_refusal(usage(data_path(cfg)), refuse)
