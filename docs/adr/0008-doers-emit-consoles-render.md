# ADR 0008: Doers emit journal notes; consoles render

Status: accepted 2026-08-28

post_social_message's 320 call sites are why the legacy layers are
circular: everything that does anything must also talk in chat. The
replacement direction: services append kind:"note" journal entries; the
chat and the website subscribe and render. Dependency arrows all point
inward and the cycle class dies. Migration is a same-signature shim, then
a mechanical rename (Phase 5).
