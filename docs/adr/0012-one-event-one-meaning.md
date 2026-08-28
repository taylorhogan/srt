# ADR 0012: Event names carry one meaning

Status: accepted 2026-08-28

Learned from the 2026-05-18 replay: NINA_SLOT_DONE briefly meant
"sequence launched" in one state and "sequence finished" in another,
disambiguated only by machine state. A NINA restart desynced the state by
one and the meanings crossed, stranding the night. Rule: if two rows need
the same event to mean different things, it is two events (SLOT_STARTED
vs NINA_SLOT_DONE). State must never be the codebook for an event's
meaning.
