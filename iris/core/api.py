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

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class EventIn(BaseModel):
    event: str
    source: str = "unknown"
    kind: str = "event"           # "event" drives the machine; "note" annotates
    data: dict = {}


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
        return {
            "state": conductor.state,
            "seq": journal.head(),
            "shadow": True,        # dropped when authority arrives (Phase 3)
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
        conductor.offer(body.event, body.source, body.data)
        after = conductor.state
        # In shadow mode a rejected event is visible as an unchanged state
        # with a journaled rejection; report faithfully.
        last = journal.entries_since(journal.head() - 1)
        rejected = bool(last and last[-1].kind == "rejected")
        return {"accepted": not rejected, "seq": journal.head(),
                "state": after, "was": before}

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
