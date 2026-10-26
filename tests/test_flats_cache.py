"""flats_by_filter's cache follows the disk (2026-10-24).

The chat server lives for days. Its cached flat list outlived a purge (the
morning analysis rebuilt it mid-purge), so every master flat after named a
deleted file and stacks fell back to raw frames.
"""
import pytest

fits = pytest.importorskip("astropy.io.fits")
np = pytest.importorskip("numpy")
from stacking import stacker  # noqa: E402


def _flat(path, filt):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(np.zeros((4, 4), np.uint16))
    hdu.header["FILTER"] = filt
    hdu.writeto(path)


@pytest.fixture(autouse=True)
def _clean_cache():
    stacker._FLATS_BY_FILTER_CACHE.clear()
    stacker._FLATS_SIGNATURE.clear()
    yield
    stacker._FLATS_BY_FILTER_CACHE.clear()
    stacker._FLATS_SIGNATURE.clear()


def test_a_purged_session_drops_out_of_the_cache(tmp_path):
    old, new = tmp_path / "2026-10-19" / "FLAT" / "a.fits", tmp_path / "2026-10-23" / "FLAT" / "b.fits"
    _flat(old, "L")
    _flat(new, "L")
    assert len(stacker.flats_by_filter(tmp_path)["L"]) == 2
    old.unlink()
    old.parent.rmdir()
    old.parent.parent.rmdir()
    assert stacker.flats_by_filter(tmp_path)["L"] == [new]


def test_a_new_session_of_flats_is_seen(tmp_path):
    _flat(tmp_path / "2026-10-22" / "FLAT" / "a.fits", "L")
    assert "R" not in stacker.flats_by_filter(tmp_path)
    _flat(tmp_path / "2026-10-23" / "FLAT" / "b.fits", "R")
    assert len(stacker.flats_by_filter(tmp_path)["R"]) == 1


def test_an_unchanged_disk_is_served_from_the_cache(tmp_path, monkeypatch):
    _flat(tmp_path / "2026-10-22" / "FLAT" / "a.fits", "L")
    first = stacker.flats_by_filter(tmp_path)
    monkeypatch.setattr(stacker.fits, "getheader", lambda *_a, **_k: pytest.fail("re-read headers"))
    assert stacker.flats_by_filter(tmp_path) is first


def test_a_master_uses_only_the_newest_sessions():
    # 2026-10-26: the Spark's sync never deletes, so its masters averaged every
    # flat set ever taken.
    from pathlib import Path
    paths = [Path("/f/%s/FLAT/L_%d.fits" % (d, i)) for d in ("2026-10-19", "2026-10-25", "2026-10-22",
                                                              "2026-10-24", "2026-07-14") for i in range(2)]
    got = stacker.newest_flat_sets(paths, 3)
    assert sorted({p.parent.parent.name for p in got}) == ["2026-10-22", "2026-10-24", "2026-10-25"]
    assert len(got) == 6
    assert stacker.newest_flat_sets(paths, 0) == paths            # 0 = no limit
    assert stacker.newest_flat_sets([], 3) == []


def test_the_purge_still_sees_every_set(tmp_path):
    # flats_by_filter itself is never trimmed: `purge` reads it to find OLD sets.
    for d in ("2026-10-19", "2026-10-22", "2026-10-23", "2026-10-24"):
        _flat(tmp_path / d / "FLAT" / "a.fits", "L")
    assert len(stacker.flats_by_filter(tmp_path)["L"]) == 4
