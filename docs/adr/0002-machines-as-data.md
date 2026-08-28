# ADR 0002: Two state machines, defined as data

Status: accepted 2026-08-28

One authoritative Night machine (the observatory's night) and one Target
machine (a DSO's life). Each is a declarative transition table; a ~60-line
pure interpreter steps them; the diagrams in docs/ are GENERATED from the
tables and CI fails when stale. No FSM framework and no TLA+: the guards
are pure functions of an enumerable SensorSnapshot, so sweeping the entire
snapshot space in tests IS the model check (1,944 worlds, every one
asserted against Invariant A).
Consequence: any behaviour change is a table diff, reviewable in one hunk,
and exhaustively tested by construction.
