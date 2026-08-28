# ADR 0003: Append-only journal + five-endpoint HTTP API

Status: accepted 2026-08-28

The journal (JSONL per day, monotonic seq, fsync per line) is the system
of record; current state is derived by replay plus a reconciliation pass
against fresh sensors. The API is five endpoints (/v1/state, /v1/events,
/v1/journal, /v1/journal/stream, /v1/targets) and there is deliberately no
"request transition" endpoint: everything is an event offered to the
machine, and REFUSALS ARE JOURNALED with the guard's reason -- the audit
trail of every time a guard saved the roof comes free.
Files-as-interfaces (imaging.txt, scheduler_state.json, safety.txt,
mode.txt) and MQTT retire in Phase 3.
