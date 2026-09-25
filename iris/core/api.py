"""The conductor's HTTP surface — five endpoints, deliberately no more.

GET  /v1/state              current machine state + context
POST /v1/events             offer an event (Phase 1: journaled; notes rendered;
                            machine-driving external events arrive with
                            authority in later phases)
GET  /v1/journal?since=SEQ  page of entries after SEQ
GET  /v1/journal/stream     SSE of new entries (poll-backed; honest and simple)
GET  /v1/targets[/dso]      the Target registry snapshot

There is intentionally NO "request transition" endpoint: operator commands,
sensor reports and cooperative capture signals are all just events, and the
machine decides. Refusals are journaled with the guard's reason.
"""
import asyncio
import json
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class EventIn(BaseModel):
    event: str
    source: str = "unknown"
    kind: str = "event"           # "event" drives the machine; "note" annotates
    data: dict = {}
    # A live sensor read posted with a roof request (iris/client.py
    # evidence_from_vision): parked_vision / parked_kasa / roof as Tri names.
    evidence: Optional[dict] = None


def _entry_dict(e):
    d = asdict(e)
    d["from"] = d.pop("from_state")
    d["to"] = d.pop("to_state")
    return d


def build_app(conductor, journal, registry_fn):
    """App factory. `conductor` needs .state/.slots/.evidence and .offer();
    `registry_fn` returns the Target registry dict."""
    app = FastAPI(title="iris-conductor", version="0.1-shadow")

    @app.get("/v1/state")
    def state():
        ev = conductor._current_evidence()
        authority = bool(getattr(conductor, "roof_authority", False))
        return {
            "state": conductor.state,
            "seq": journal.head(),
            # True until the conductor owns everything (Phase 3); the roof
            # is the first thing it decides (Phase 2), reported separately.
            "shadow": not authority,
            "authority": {"roof": authority,
                          "mount": bool(getattr(conductor, "mount_authority", False))},
            "context": {
                "slots_remaining": conductor.slots,
                "safety": "armed" if ev.safety_armed else "cleared",
                "mode": "auto" if ev.mode_auto else "manual",
                "roof": ev.roof.value,
                "parked_vision": ev.parked_vision.value,
                "nina_alive": ev.nina_alive,
            },
        }

    @app.post("/v1/events")
    def post_event(body: EventIn):
        if body.kind == "note":
            e = journal.append("note", body.event, body.source, data=body.data)
            return {"accepted": True, "seq": e.seq, "state": conductor.state}
        before = conductor.state
        v = conductor.offer(body.event, body.source, body.data, evidence=body.evidence)
        # accepted == the machine moved. A guard refusal and a missing row are
        # both refusals to a caller asking for the roof; the reason says which.
        return {"accepted": v.accepted, "kind": v.kind,
                "guard": v.guard, "would_refuse": v.would_refuse,
                "seq": v.seq or journal.head(), "state": v.state, "was": before,
                "authority": bool(getattr(conductor, "roof_authority", False)),
                "mount_authority": bool(getattr(conductor, "mount_authority", False))}

    @app.get("/v1/journal")
    def journal_page(since: int = 0, limit: int = 500):
        entries = journal.entries_since(since, limit=min(limit, 2000))
        return {"entries": [_entry_dict(e) for e in entries],
                "head": journal.head()}

    @app.get("/v1/journal/stream")
    async def stream():
        async def gen():
            seq = journal.head()
            while True:
                entries = journal.entries_since(seq)
                for e in entries:
                    seq = e.seq
                    yield "data: %s\n\n" % json.dumps(_entry_dict(e))
                await asyncio.sleep(2.0)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/v1/targets")
    def targets():
        return registry_fn()

    @app.get("/v1/targets/{dso}")
    def target(dso: str):
        reg = registry_fn()
        if dso not in reg:
            raise HTTPException(404, f"unknown target {dso!r}")
        return reg[dso]

    return app
