# ADR 0005: Strangler fig, gated by shadow nights

Status: accepted 2026-08-28

The observatory must never lose a night to the migration. Each phase
ships beside the legacy system; authority transfers only after the new
code has SHADOWED real nights cleanly (shadow_report verdict CLEAN), and
the first hardware act under new authority is a supervised daytime drill.
Replay of recorded history (tests/replay/) extends the shadow discipline
backwards: five months of real nights validate every table change in CI.
Already vindicated twice: the replay caught a one-event-two-meanings bug
(2026-05-18's NINA restart) and the simulated nights forced the PARKING
state.
