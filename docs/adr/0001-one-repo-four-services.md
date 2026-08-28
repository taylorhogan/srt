# ADR 0001: One repository, four services

Status: accepted 2026-08-28

The four systems (observatory+safety, webchat, publishing, n2n/GPU) run as
separate processes with a defined contract, but live in ONE repository with
one deploy. Separate repos would multiply the update surface of an
unattended system by four; process separation gives the isolation that
matters (a chat crash cannot take the night down) without it.
Consequence: package boundaries inside the repo are the real interfaces
and must be enforced (imports point inward; iris/ never imports the
legacy command layer).
