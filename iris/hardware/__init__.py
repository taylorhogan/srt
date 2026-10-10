"""The hardware layer of the new spine: the only code that drives an actuator.

docs/ARCHITECTURE_PLAN.md, Phase 2b: one module per actuator, and a CI test
(tests/test_roof_single_path.py) that nothing else in the repo reaches the
roof's motor plug or relay directly.
"""
