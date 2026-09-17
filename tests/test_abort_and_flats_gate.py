"""stop! must reach a run in another process, and flats must refuse an open roof.

2026-09-17 00:32: the auto night ran in the scheduler process, stop! in the web
chat's. The abort was a threading.Event, so the run never saw it; it read the
imaging state NONE that stop! wrote as "main phase complete", powered the mount
on and launched flats with the roof open (only a warning). These pin both fixes.

Needs the private config to import the command module; skipped on a bare CI
runner like the other hardware-adjacent tests.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

suc = pytest.importorskip("cmd_processing.super_user_commands")


@pytest.fixture
def flag(tmp_path, monkeypatch):
    path = tmp_path / "emergency_abort.flag"
    monkeypatch.setattr(suc, "ABORT_FLAG_PATH", str(path))
    suc._abort_event.clear()
    yield path
    suc._abort_event.clear()


def test_abort_is_seen_by_another_process(flag):
    suc.request_abort()
    assert flag.exists()
    suc._abort_event.clear()          # what a different process's Event looks like
    assert suc.is_aborting()


def test_clear_abort_removes_the_flag(flag):
    suc.request_abort()
    suc.clear_abort()
    assert not flag.exists()
    assert not suc.is_aborting()
    suc.clear_abort()                 # idempotent with no flag present


class _Recorder:
    def __init__(self):
        self.kasa = []
        self.posts = []
        self.pushes = []


@pytest.fixture
def flats_env(flag, monkeypatch):
    rec = _Recorder()

    async def fake_map(*a, **k):
        return {"Telescope mount": "1", "Iris inside light": "2"}

    async def fake_do(dev_map, inst):
        rec.kasa.append(dict(inst))
        return {k: True for k in inst}

    monkeypatch.setattr(suc.ku, "make_discovery_map", fake_map)
    monkeypatch.setattr(suc.ku, "kasa_do", fake_do)
    monkeypatch.setattr(suc.social_server, "post_social_message", rec.posts.append)
    monkeypatch.setattr(suc.pushover, "push_message", lambda *a, **k: rec.pushes.append(a))
    monkeypatch.setattr(suc, "set_imaging_state",
                        lambda s: pytest.fail("flats must not reach IN_FLATS"))
    monkeypatch.setattr(suc.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("flats must not launch NINA"))
    return rec


@pytest.mark.parametrize("parked, closed", [(True, False), (False, True), (False, False)])
def test_flats_refuse_unless_parked_and_closed(flats_env, monkeypatch, parked, closed):
    monkeypatch.setattr(suc, "get_status_with_lights", lambda: (parked, closed, not closed, None))
    assert suc.do_flats() is False
    assert {"Telescope mount": "on"} not in flats_env.kasa
    assert any("Flats REFUSED" in p for p in flats_env.posts)
    assert flats_env.pushes


def test_flats_refuse_during_an_abort_without_touching_hardware(flats_env, monkeypatch):
    monkeypatch.setattr(suc, "get_status_with_lights",
                        lambda: pytest.fail("no vision read during an abort"))
    suc.request_abort()
    assert suc.do_flats() is False
    assert flats_env.kasa == []
