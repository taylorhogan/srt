"""Test-wide guards.

The suite runs against a checkout that can reach the live observatory, so a
test that forgets to stub an outbound call does not fail — it succeeds, and
the side effect lands on the real instrument.

That is not hypothetical. `test_stop_sequence_reports_a_crash_instead_of_dying
_silently` stubbed the chat and Pushover but not the conductor, so every run of
the suite posted a real ESTOP_FAILED into the observatory's safety journal:
40 of them between 2026-09-25 and 2026-10-08, each carrying that test's fake
"ConnectionRefusedError: PWI4 refused" and each indistinguishable, to anyone
reading the journal, from a genuine failed emergency stop.

So the conductor is stubbed for every test by default. A test that wants to
assert on what was posted stubs it again itself, which still works: the last
monkeypatch wins.
"""
import pytest


@pytest.fixture(autouse=True)
def _no_live_conductor(monkeypatch):
    """Block outbound conductor posts for the whole suite."""
    try:
        import iris.client as _ic
    except Exception:          # noqa: BLE001 -- iris not importable in this env
        return
    monkeypatch.setattr(_ic, "post_event", lambda event, *a, **k: None, raising=False)
    monkeypatch.setattr(_ic, "state", lambda *a, **k: None, raising=False)


@pytest.fixture(autouse=True)
def _drives_checks_from_the_test_not_the_machine(monkeypatch):
    """A ShadowConductor built without drives_checks= reads the live config,
    so the observatory's own flag (True since 2026-10-16) decided whether the
    shadow-planning tests passed there while CI, with the config stubbed, saw
    False. Default to off; tests of the driven path pass drives_checks=True."""
    try:
        import iris.conductor.shadow as _shadow
    except Exception:          # noqa: BLE001
        return
    monkeypatch.setattr(_shadow, "_read_drives_checks", lambda: False, raising=False)
