"""Run the Target-registry audit against the live system.

    python apps/registry_audit.py [--post]

Prefers the RUNNING conductor's answer (GET :8096/v1/targets) so a stale
process shows up as mismatches; falls back to deriving in-process when the
conductor is down (and says so in the report header). Posts to the webchat
only with --post. Exits 0 always — like shadow_report, a check that dies
silently is worse than no check.
"""
import json
import os
import sys
import urllib.request

if __package__ is None or __package__ == "":
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from iris.conductor import audit as A


def _live_registry():
    # --derive audits the CURRENT code's derivation instead of the running
    # conductor's answer — the way to verify a derivation fix before the
    # conductor has restarted onto it.
    if "--derive" not in sys.argv:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8096/v1/targets",
                                        timeout=5) as r:
                return json.load(r), "live conductor :8096"
        except Exception:
            pass
    from iris.conductor.targets import derive_registry
    return derive_registry(), ("derived in-process" if "--derive" in sys.argv
                               else "derived in-process; conductor unreachable")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from configs import config
    from fits_processing.convergence import is_dso_done, load_convergence
    from iris.conductor.targets import _has_frames

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    queue_path = os.path.join(root, config.data()["location"]["instructions"])
    with open(queue_path, encoding="utf-8") as fh:
        queue = json.load(fh)

    registry, source = _live_registry()
    result = A.audit_registry(queue, registry, is_dso_done, _has_frames)
    findings = A.find_inconsistencies(queue, load_convergence(), is_dso_done)
    report = A.render(result, findings, source)
    print(report)
    if "--post" in sys.argv:
        try:
            from cmd_processing import social_server
            social_server.post_social_message(report)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
