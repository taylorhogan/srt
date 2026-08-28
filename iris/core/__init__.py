"""iris.core — the machines, their guards, the journal, and the API.

The machines are DATA (transition tables in machine.py / target_machine.py).
The diagrams in docs/ are generated from these tables; CI fails if they drift.
Everything here is deliberately stdlib-only except api.py, so the tables,
guards and journal are testable on a bare CI runner.
"""
