"""The dawn close retry (owner, 2026-10-20).

When the end sequence cannot confirm the park -- for instance with the internet
down, which blinds both roof cameras' cloud-only pan/tilt -- the roof used to
stay open with nothing ever trying again. Now the same gated check repeats
every 10 minutes. What must hold:

- the loop stops on "closed" or "stop", gives up after its budget, and tells
  the phone each hour it is still open;
- a retry never pulses the relay unless vision positively reads the roof OPEN:
  the relay toggles, so a pulse at a roof someone already closed would open it.
"""
import pytest

end = pytest.importorskip("end_points.end")


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_the_loop_retries_until_the_roof_closes():
    c, notes, answers = Clock(), [], iter(["waiting", "waiting", "closed"])
    r = end.dawn_retry_loop(lambda: next(answers), c.sleep, c.now,
                            lambda m, phone: notes.append(m), interval_s=600)
    assert r == "closed" and c.t == 1200 and notes == []


def test_the_loop_tells_the_phone_hourly_and_gives_up_after_its_budget():
    c, notes = Clock(), []
    r = end.dawn_retry_loop(lambda: "waiting", c.sleep, c.now,
                            lambda m, phone: notes.append((m, phone)),
                            interval_s=600, max_s=3 * 3600, notify_every_s=3600)
    assert r == "gave_up"
    hourly = [m for m, _ in notes if "still OPEN" in m]
    assert len(hourly) == 2 and all(p for _, p in notes)
    assert "gave up" in notes[-1][0] and "roof!! close force" in notes[-1][0]


def test_someone_else_operating_stops_the_retry():
    c = Clock()
    assert end.dawn_retry_loop(lambda: "stop", c.sleep, c.now, lambda m, p: None) == "stop"


@pytest.fixture
def world(monkeypatch):
    """Vision, conductor and relay stubbed; records whether the relay fired."""
    fired = []
    monkeypatch.setattr(end.pushover, "push_message", lambda *a, **k: True)
    monkeypatch.setattr(end.social_server, "post_social_message", lambda *a, **k: None)
    monkeypatch.setattr(end.super_user_commands, "announce_roof_movement", lambda *a, **k: None)
    monkeypatch.setattr(end.super_user_commands, "toggle_roof", lambda *a, **k: fired.append(k))
    monkeypatch.setattr(end.super_user_commands, "confirm_roof_state", lambda *a, **k: True)
    monkeypatch.setattr(end.conductor, "request_roof_move", lambda *a, **k: (True, "", {"kind": "ok"}))
    monkeypatch.setattr(end.conductor, "report", lambda *a, **k: None)
    monkeypatch.setattr(end.ku, "kasa_do", lambda *a, **k: _done({}))
    monkeypatch.setattr(end.utl_shelly, "set_dehumidifier", lambda *a, **k: None)
    return fired


async def _done(v):
    return v


def _vision(monkeypatch, parked, closed, is_open):
    monkeypatch.setattr(end.vision_safety, "visual_status", lambda *a, **k: (parked, closed, is_open, None))


def test_a_retry_never_pulses_a_roof_that_reads_closed(world, monkeypatch):
    _vision(monkeypatch, parked=True, closed=True, is_open=False)
    assert end._close_if_parked({}, require_open=True) == (True, False)
    assert world == []                      # the relay toggles: a pulse here would OPEN it


def test_a_retry_never_pulses_a_roof_it_cannot_read(world, monkeypatch):
    _vision(monkeypatch, parked=True, closed=False, is_open=False)
    assert end._close_if_parked({}, require_open=True) == (True, False)
    assert world == []


def test_a_retry_closes_a_parked_scope_under_an_open_roof(world, monkeypatch):
    _vision(monkeypatch, parked=True, closed=False, is_open=True)
    assert end._close_if_parked({}, require_open=True) == (True, True)
    assert world == [{"capture_direction": "close"}]


def test_not_parked_moves_nothing(world, monkeypatch):
    _vision(monkeypatch, parked=False, closed=False, is_open=True)
    assert end._close_if_parked({}, require_open=True) == (False, False)
    assert world == []


def test_the_end_sequence_itself_is_unchanged(world, monkeypatch):
    """require_open is the retry's extra condition only: the first attempt
    closes on a confirmed park exactly as before."""
    _vision(monkeypatch, parked=True, closed=False, is_open=False)
    assert end._close_if_parked({}) == (True, True)
    assert world == [{"capture_direction": "close"}]
