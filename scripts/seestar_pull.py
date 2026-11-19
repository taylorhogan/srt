"""Pull the pictures off a ZWO Seestar on the house network.

    python scripts/seestar_pull.py [--list] [--thumbs] [--since DAYS] [--dry-run]
                                   [--host NAME] [--dest DIR]

The Seestar in station mode registers itself as seestar.local and serves one
SMB share, "EMMC Images", to a guest with no password; under it MyWorks/ holds
one folder per target (and Scenery_photo for the daytime shots) with the
stacked JPEG, its _thn thumbnail, and the FITS the app was told to keep. This
mirrors MyWorks into <image_dir>/seestar on this machine: a file is copied when
it is missing locally or its size differs, never deleted here when it goes
from the device, and written to a temporary name and renamed so a copy that
dies mid-way leaves nothing half-written. Thumbnails stay on the device unless
--thumbs. The device serves one control client at a time but the share is
independent of that, so this runs while the app is connected.

Nothing is written to the Seestar. cfg["seestar"] has the host, share, root
and destination; the flags override it.
"""
import argparse
import os
import socket
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

THUMB_SUFFIX = "_thn"
DEFAULTS = {"host": "seestar.local", "share": "EMMC Images", "root": "MyWorks",
            "dest": "seestar", "thumbnails": False}


def is_thumbnail(name: str) -> bool:
    """The app writes <name>_thn.jpg beside every picture."""
    stem, _ext = os.path.splitext(name)
    return stem.lower().endswith(THUMB_SUFFIX)


def local_path(dest: Path, relpath: str) -> Path:
    """A share path (backslashes) under the destination folder."""
    return dest.joinpath(*[p for p in relpath.replace("\\", "/").split("/") if p])


def plan(remote, local_sizes, thumbnails=False, since=None):
    """Which remote files to copy.

    remote: iterable of (relpath, size, mtime as a datetime); local_sizes: {relpath: size}
    for what is already here. since: a datetime; older files are left alone.
    Returns the (relpath, size, mtime) tuples to copy, in the order given.
    """
    out = []
    for relpath, size, mtime in remote:
        name = relpath.replace("\\", "/").rsplit("/", 1)[-1]
        if not thumbnails and is_thumbnail(name):
            continue
        if since is not None and mtime is not None and mtime < since:
            continue
        if local_sizes.get(relpath) == size:
            continue
        out.append((relpath, size, mtime))
    return out


def local_inventory(dest: Path, relpaths):
    """{relpath: size} for the files of the remote listing that exist here."""
    sizes = {}
    for rel in relpaths:
        p = local_path(dest, rel)
        if p.is_file():
            sizes[rel] = p.stat().st_size
    return sizes


# ----------------------------------------------------------------- the device
def resolve(host: str) -> str:
    try:
        return socket.gethostbyname(host)
    except socket.gaierror as exc:
        raise SystemExit("cannot resolve %s (is the Seestar on the house wifi and awake?): %s" % (host, exc))


def connect(ip: str):
    """A guest SMB session; the Seestar neither signs nor encrypts."""
    import smbclient  # noqa: WPS433  (optional dependency, only needed to talk to the device)
    from smbclient import ClientConfig
    ClientConfig(username="guest", password="", require_secure_negotiate=False)
    smbclient.register_session(ip, username="guest", password="", require_signing=False,
                               encrypt=False, connection_timeout=10)
    return smbclient


def walk_share(smb, ip: str, share: str, root: str):
    """Every file under root as (relpath, size, mtime); relpath uses the share's backslashes."""
    base = r"\\%s\%s" % (ip, share)
    top = base + ("\\" + root if root else "")
    stack = [top]
    while stack:
        folder = stack.pop()
        for entry in smb.scandir(folder):
            full = folder + "\\" + entry.name
            if entry.is_dir():
                stack.append(full)
                continue
            st = entry.stat()
            rel = full[len(top) + 1:]
            yield rel, st.st_size, datetime.fromtimestamp(st.st_mtime)


def copy_file(smb, ip: str, share: str, root: str, relpath: str, target: Path, chunk=1 << 20):
    src = r"\\%s\%s" % (ip, share) + ("\\" + root if root else "") + "\\" + relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    with smb.open_file(src, mode="rb") as fh, open(tmp, "wb") as out:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            out.write(block)
    os.replace(tmp, target)


def default_dest(cfg) -> Path:
    image_dir = (cfg.get("nina") or {}).get("image_dir") or str(Path.home() / "Desktop" / "Targets")
    return Path(image_dir) / (cfg.get("seestar") or {}).get("dest", DEFAULTS["dest"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", help="the Seestar's name or address (cfg seestar.host)")
    ap.add_argument("--dest", help="folder to mirror into (default <image_dir>/seestar)")
    ap.add_argument("--list", action="store_true", help="list the share, copy nothing")
    ap.add_argument("--thumbs", action="store_true", help="copy the _thn thumbnails too")
    ap.add_argument("--since", type=float, metavar="DAYS", help="only files newer than this many days")
    ap.add_argument("--dry-run", action="store_true", help="say what would be copied")
    args = ap.parse_args(argv)

    from configs import config
    cfg = config.data()
    scfg = dict(DEFAULTS, **(cfg.get("seestar") or {}))
    host = args.host or scfg["host"]
    dest = Path(args.dest) if args.dest else default_dest(cfg)
    ip = resolve(host)
    print("Seestar %s at %s, share '%s'\\%s -> %s" % (host, ip, scfg["share"], scfg["root"], dest))

    smb = connect(ip)
    t0 = time.time()
    remote = list(walk_share(smb, ip, scfg["share"], scfg["root"]))
    print("%d files on the device (%.1f MB), listed in %.1f s"
          % (len(remote), sum(r[1] for r in remote) / 1e6, time.time() - t0))
    if args.list:
        for rel, size, mtime in sorted(remote):
            print("  %10.1f MB  %s  %s" % (size / 1e6, mtime.strftime("%Y-%m-%d %H:%M"), rel))
        return 0

    since = datetime.now() - timedelta(days=args.since) if args.since else None
    todo = plan(remote, local_inventory(dest, [r[0] for r in remote]),
                thumbnails=args.thumbs or bool(scfg.get("thumbnails")), since=since)
    if not todo:
        print("nothing new; %s is current" % dest)
        return 0
    print("%d to copy (%.1f MB)%s" % (len(todo), sum(t[1] for t in todo) / 1e6, " -- dry run" if args.dry_run else ""))
    failed = 0
    for rel, size, _mtime in todo:
        target = local_path(dest, rel)
        if args.dry_run:
            print("  would copy %s (%.1f MB)" % (rel, size / 1e6))
            continue
        t1 = time.time()
        try:
            copy_file(smb, ip, scfg["share"], scfg["root"], rel, target)
        except Exception as exc:  # one bad file does not stop the rest
            failed += 1
            print("  FAILED %s: %s" % (rel, exc))
            continue
        print("  %s  %.1f MB in %.1f s" % (rel, size / 1e6, time.time() - t1))
    print("done: %d copied, %d failed -> %s" % (len(todo) - failed, failed, dest))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
