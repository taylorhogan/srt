"""The grid's SIMBAD setup and the generator's fallback cannot hang on a
connect the OS will not abandon (CDS outage of 2026-11-02)."""
import sys
import time
import types

import pytest

pytest.importorskip("astropy")
from iris_astronomy import astro_dso_visibility as av            # noqa: E402
from iris_astronomy.astro_dso_visibility import _get_simbad     # noqa: E402  the real one, before conftest stubs it
from nina_gen import nina_sequence_gen as g                      # noqa: E402


def _fresh(monkeypatch):
    monkeypatch.setattr(av, "_simbad_ready", None)
    monkeypatch.setattr(av, "_simbad_retry_after", 0.0)
    monkeypatch.setattr(av, "_SIMBAD_TIMEOUT_S", 0.2)


def test_a_setup_that_never_answers_is_an_outage_with_a_cooldown(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setattr(av._simbad, "add_votable_fields", lambda *a, **k: time.sleep(3))
    t0 = time.monotonic()
    assert _get_simbad() is None
    assert time.monotonic() - t0 < 2.0
    assert av._simbad_retry_after > time.monotonic()        # not asked again per target


def test_a_setup_that_answers_is_used(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setattr(av._simbad, "add_votable_fields", lambda *a, **k: None)
    assert _get_simbad() is av._simbad


def test_the_generator_asks_simbad_once_through_the_grid(monkeypatch):
    """A grid "?" used to send the generator to a second, unbounded client."""
    monkeypatch.setattr(g, "_queued_object_type", lambda name: None)
    monkeypatch.setitem(sys.modules, "iris_astronomy.astro_dso_visibility",
                        types.SimpleNamespace(get_dso_type=lambda name: "?"))
    monkeypatch.setitem(sys.modules, "astroquery.simbad",
                        types.SimpleNamespace(Simbad=lambda: pytest.fail("a second SIMBAD lookup")))
    assert g._classify_object_type("slot one") == "unknown"
