"""iris-conductor entrypoint (Phase 1: shadow mode).

Runs two things: the shadow watcher (a thread polling the legacy state every
few seconds) and the HTTP API on :8096. Read-only towards the observatory;
its only writes are journal files under local/journal/.

    python -m iris.conductor.main
"""
import logging
import os
import sys
import threading
from pathlib import Path

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from iris.conductor.shadow import ShadowConductor
from iris.conductor.targets import derive_registry
from iris.core.journal import Journal

_logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8096


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s conductor %(levelname)s %(message)s")

    try:
        from configs import config
        cfg = config.data().get("conductor", {})
    except Exception:
        cfg = {}
    port = int(cfg.get("port", DEFAULT_PORT))

    # Single-instance check BEFORE anything starts writing. Two conductors
    # would interleave duplicate events into the same journal, which corrupts
    # the record the whole design trusts. Claiming the API port first makes
    # the port the lock: a second instance exits here, before its watcher
    # thread has journaled anything.
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("0.0.0.0", port))
    except OSError:
        print(f"conductor: port {port} already in use — another instance is "
              "running; exiting without touching the journal")
        return 0
    finally:
        probe.close()

    journal = Journal(REPO_ROOT / "local" / "journal")
    conductor = ShadowConductor(REPO_ROOT, journal)
    journal.append("note", "CONDUCTOR_STARTED", "conductor",
                   data={"mode": "shadow", "state": conductor.state})

    watcher = threading.Thread(target=conductor.run_forever, daemon=True,
                               name="shadow-watcher")
    watcher.start()

    from iris.core.api import build_app
    import uvicorn
    app = build_app(conductor, journal, derive_registry)
    # log_level warning: the 5-min publishers will poll /v1/state forever and
    # an access-log line per poll is pure noise.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
