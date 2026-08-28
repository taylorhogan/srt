"""iris.conductor — the observatory's brain (Phase 1: read-only shadow).

In shadow mode the conductor OWNS NOTHING and COMMANDS NOTHING: it watches
the legacy state files and the log, synthesizes events, and drives the Night
machine + journal in parallel with reality. Its output is judged each morning
by apps/shadow_report.py; authority arrives phase by phase (roof in Phase 2,
the night in Phase 3) only after clean shadow nights.
"""
