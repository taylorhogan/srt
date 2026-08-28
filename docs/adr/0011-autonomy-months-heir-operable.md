# ADR 0011: Autonomy target: months unattended, heir-operable

Status: accepted 2026-08-28

Revised DOWN from fully-autonomous-until-hardware-death, on reflection.
Self-healing is bounded to what half-exists (supervisor restarts, journal
replay recovery, disk purge, automatic re-solve); everything else a human
must eventually do -- pay the domain, replace a camera, collimate -- is a
RUNBOOK in docs/SUCCESSION.md, written so an heir can follow it. The
saved effort goes to tests and documents, which is what actually
lengthens the system's life.
