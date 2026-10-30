"""post_event.py -- post one event or note to the conductor from the command line.

    python scripts/post_event.py RENDER_DONE --source spark --kind note --b64 <base64 JSON>
    python scripts/post_event.py REOPENED --data '{"dso": "m1"}'

Runs on the observatory PC, where the conductor's API answers on loopback
only. The Spark reaches it over ssh (scripts/spark_morning_render.post_pc_event)
with the data as base64 JSON, because the ssh lands in cmd.exe and nothing
with quotes survives the trip (morning_trigger._ps). D1 of
docs/AUTOMATION_PLAN.md (2026-10-30): the Spark reports render outcomes to
the journal and never writes a fact into the PC's records itself.
"""
import argparse
import base64
import json
import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def parse_data(b64: str | None, raw: str | None) -> dict:
    """The event's data from --b64 (base64 of UTF-8 JSON) or --data (JSON)."""
    if b64:
        return json.loads(base64.b64decode(b64).decode("utf-8"))
    if raw:
        return json.loads(raw)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event")
    ap.add_argument("--source", default="spark")
    ap.add_argument("--kind", default="note", choices=("note", "event"))
    ap.add_argument("--b64", default=None, help="base64 of the data JSON")
    ap.add_argument("--data", default=None, help="the data JSON")
    ap.add_argument("--retire", action="store_true",
                    help="after a PUBLISH_DONE: set the queue row of data['dso'] to completed "
                         "(instructions.set_status_db), which the registry reads as RETIRED")
    args = ap.parse_args()
    try:
        data = parse_data(args.b64, args.data)
    except (ValueError, UnicodeDecodeError) as exc:
        print(f"bad data: {exc}", file=sys.stderr)
        return 2
    from iris import client
    reply = client.post_event(args.event, args.source, data, kind=args.kind)
    if reply is None:
        print("conductor unreachable", file=sys.stderr)
        return 1
    if args.retire and data.get("dso"):
        try:
            from control import instructions
            ok = instructions.set_status_db(str(data["dso"]), "completed",
                                            why="published %s" % (data.get("id") or ""), who="publish")
            print(json.dumps({"retired": bool(ok)}))
        except Exception as exc:  # noqa: BLE001
            print(f"retire failed: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(reply))
    return 0


if __name__ == "__main__":
    sys.exit(main())
