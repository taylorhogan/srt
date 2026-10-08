"""scripts/live_trigger: one render at a time, stale locks cleared, bursts debounced."""
import os
import time

import pytest

from scripts import live_trigger as lt


def test_lock_is_exclusive(tmp_path):
    lock = tmp_path / "r.lock"
    with lt.render_lock(lock):
        assert lock.exists()
        with pytest.raises(lt.RenderBusy):
            with lt.render_lock(lock):
                pass
    assert not lock.exists()


def test_stale_lock_is_taken_over(tmp_path):
    lock = tmp_path / "r.lock"
    lock.write_text("dead")
    old = time.time() - 1000
    os.utime(lock, (old, old))
    with lt.render_lock(lock, stale_s=240):
        assert lock.read_text() == str(os.getpid())
    assert not lock.exists()


def test_fresh_lock_is_respected(tmp_path):
    lock = tmp_path / "r.lock"
    lock.write_text("alive")
    with pytest.raises(lt.RenderBusy):
        with lt.render_lock(lock, stale_s=240):
            pass
    assert lock.exists()      # not ours to remove


def test_trigger_debounces_a_burst(monkeypatch, tmp_path):
    spawned = []
    monkeypatch.setattr(lt.subprocess, "Popen", lambda *a, **k: spawned.append(a[0]))
    monkeypatch.setattr(lt, "LOG", tmp_path / "t.log")
    lt._last["t"] = 0.0
    assert lt.trigger("frame a", debounce_s=20) is True
    assert lt.trigger("frame b", debounce_s=20) is False
    assert len(spawned) == 1 and "--reason" in spawned[0] and "frame a" in spawned[0]


def test_trigger_never_raises(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("no python")
    monkeypatch.setattr(lt.subprocess, "Popen", boom)
    monkeypatch.setattr(lt, "LOG", tmp_path / "t.log")
    lt._last["t"] = 0.0
    assert lt.trigger("state IN_MAIN", debounce_s=0) is False
    assert "trigger failed" in (tmp_path / "t.log").read_text()
