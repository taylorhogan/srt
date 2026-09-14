"""Kasa discovery is one best-effort UDP broadcast; a pass can miss a plug
(2026-09-14: the flats map lacked 'Telescope mount', the end sequence's map
an hour earlier had it). The map is the union of passes, keeps going while an
expected name is missing, and kasa_do re-discovers a missing key once before
declaring it absent."""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ku = pytest.importorskip("hardware_control.kasa_utils")


def _passes(monkeypatch, results):
    calls = []

    async def fake_once():
        calls.append(1)
        return dict(results[min(len(calls) - 1, len(results) - 1)])
    monkeypatch.setattr(ku, "_discover_once", fake_once)
    return calls


def test_map_is_the_union_of_two_passes(monkeypatch):
    calls = _passes(monkeypatch, [{"Roof motor": "1.1"}, {"Telescope mount": "1.2"}])
    m = asyncio.run(ku.make_discovery_map())
    assert m == {"Roof motor": "1.1", "Telescope mount": "1.2"}
    assert len(calls) == 2


def test_expected_name_keeps_discovery_going_up_to_the_cap(monkeypatch):
    calls = _passes(monkeypatch, [{"Roof motor": "1.1"}, {"Roof motor": "1.1"},
                                  {"Telescope mount": "1.2"}])
    m = asyncio.run(ku.make_discovery_map(expect=("Telescope mount",)))
    assert m["Telescope mount"] == "1.2" and len(calls) == 3


def test_expected_name_found_early_stops_after_the_normal_passes(monkeypatch):
    calls = _passes(monkeypatch, [{"Telescope mount": "1.2"}, {"Roof motor": "1.1"}, {}])
    asyncio.run(ku.make_discovery_map(expect=("Telescope mount",)))
    assert len(calls) == ku.DISCOVERY_PASSES


def test_missing_name_gives_up_after_the_cap(monkeypatch):
    calls = _passes(monkeypatch, [{}])
    m = asyncio.run(ku.make_discovery_map(expect=("Telescope mount",)))
    assert "Telescope mount" not in m and len(calls) == ku.DISCOVERY_PASSES_EXPECT


def test_kasa_do_rediscovers_a_missing_key_and_updates_the_map(monkeypatch):
    async def fake_map(expect=(), passes=2):
        return {"Telescope mount": "1.2"}
    monkeypatch.setattr(ku, "make_discovery_map", fake_map)
    monkeypatch.setattr(ku, "legacy_relay", lambda ip, state=None, timeout=2.0, retries=1: 0)
    dev_map = {"Roof motor": "1.1"}                     # stale: mount missing
    out = asyncio.run(ku.kasa_do(dev_map, {"Telescope mount": "off"}))
    assert out == {"Telescope mount": True}
    assert dev_map["Telescope mount"] == "1.2"           # caller's map healed


def test_kasa_do_still_refuses_when_rediscovery_fails(monkeypatch):
    async def fake_map(expect=(), passes=2):
        return {}
    monkeypatch.setattr(ku, "make_discovery_map", fake_map)
    monkeypatch.setattr(ku, "legacy_relay", lambda *a, **k: pytest.fail("must not switch"))
    out = asyncio.run(ku.kasa_do({}, {"Telescope mount": "off"}))
    assert out == {"Telescope mount": False}
