"""The roof limit switches' entire truth table, pinned.

The complementary-pair design (sheet RLS-1) exists for its failure rows: a
broken switch must read as a FAULT, never as a roof position. These tests
enumerate all 16 input combinations so no wiring state has an unconsidered
meaning, then pin the not-configured/unreachable behaviour -- absence of a
measurement is never a position either.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hardware_control.roof_limit_switches import (
    CLOSED, FAULT, IN_TRANSIT, NOT_CONFIGURED, OPEN, UNREACHABLE, decode, read)


def test_the_four_healthy_states():
    assert decode(True, False, False, True).state == CLOSED
    assert decode(False, True, True, False).state == OPEN
    assert decode(False, True, False, True).state == IN_TRANSIT
    # both at limit: pairs individually healthy, physically impossible
    r = decode(True, False, True, False)
    assert r.state == FAULT and "impossible" in r.detail


def test_cut_wire_reads_as_fault_not_position():
    # The exact failure of the old install: a dead switch. Pair reads 00.
    r = decode(False, False, False, True)
    assert r.state == FAULT
    assert "CLOSED-limit" in r.detail and "cut wire" in r.detail


def test_short_reads_as_fault_not_position():
    r = decode(False, True, True, True)
    assert r.state == FAULT
    assert "OPEN-limit" in r.detail and "short" in r.detail


def test_every_combination_has_exactly_one_meaning():
    """Exhaustive: 16 combinations -> 4 healthy states + 12 faults, and a
    position is only ever claimed when BOTH pairs are healthy."""
    states = {}
    for i in range(16):
        bits = tuple(bool(i >> b & 1) for b in range(4))
        r = decode(*bits)
        states[bits] = r.state
        pair_ok = lambda no, nc: no != nc
        if r.state in (OPEN, CLOSED, IN_TRANSIT):
            assert pair_ok(bits[0], bits[1]) and pair_ok(bits[2], bits[3])
        else:
            assert r.detail
    assert sum(1 for s in states.values() if s == FAULT) == 12
    assert sorted(s for s in states.values() if s != FAULT) == [
        CLOSED, IN_TRANSIT, OPEN]


def test_unconfigured_and_unreachable_are_not_positions():
    assert read(cfg={"hardware": {}}).state == NOT_CONFIGURED
    r = read(cfg={"hardware": {"roof_limit_shelly_ip": "192.0.2.1"}},
             timeout_s=0.2)          # TEST-NET address: guaranteed unreachable
    assert r.state == UNREACHABLE
