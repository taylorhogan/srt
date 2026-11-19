"""scripts/seestar_pull: what gets copied off the Seestar's share (no device, no network)."""
from datetime import datetime, timedelta
from pathlib import Path

from scripts import seestar_pull as sp


def _remote():
    t = datetime(2026, 11, 19, 10, 0)
    return [
        (r"Scenery_photo\2026-11-19-095113-Scenery.jpg", 1_800_000, t),
        (r"Scenery_photo\2026-11-19-095113-Scenery_thn.jpg", 60_000, t),
        (r"M 31\Stacked_120_M 31_10.0s_IRCUT_20261118-213000.fit", 48_000_000, t - timedelta(days=1)),
        (r"M 31\Stacked_120_M 31_10.0s_IRCUT_20261118-213000.jpg", 2_000_000, t - timedelta(days=1)),
        (r"Old\2026-01-01-Old.jpg", 1_000_000, t - timedelta(days=300)),
    ]


def test_thumbnails_stay_on_the_device_unless_asked():
    assert sp.is_thumbnail("2026-11-19-095113-Scenery_thn.jpg")
    assert not sp.is_thumbnail("2026-11-19-095113-Scenery.jpg")
    names = [r[0] for r in sp.plan(_remote(), {})]
    assert not any("_thn" in n for n in names)
    assert len(names) == 4
    with_thumbs = [r[0] for r in sp.plan(_remote(), {}, thumbnails=True)]
    assert len(with_thumbs) == 5


def test_a_file_already_here_at_the_same_size_is_skipped_and_a_different_size_is_copied_again():
    remote = _remote()
    have = {remote[0][0]: remote[0][1],          # same size: current
            remote[2][0]: remote[2][1] - 1000}   # short: a copy that died, or a new version
    names = [r[0] for r in sp.plan(remote, have)]
    assert remote[0][0] not in names
    assert remote[2][0] in names


def test_since_leaves_old_pictures_alone():
    since = datetime(2026, 11, 19, 10, 0) - timedelta(days=7)
    names = [r[0] for r in sp.plan(_remote(), {}, since=since)]
    assert not any(n.startswith("Old") for n in names)
    assert any(n.startswith("M 31") for n in names)


def test_share_paths_become_folders_here(tmp_path: Path):
    p = sp.local_path(tmp_path, r"M 31\Stacked_120_M 31_10.0s_IRCUT_20261118-213000.fit")
    assert p == tmp_path / "M 31" / "Stacked_120_M 31_10.0s_IRCUT_20261118-213000.fit"
    (tmp_path / "M 31").mkdir()
    p.write_bytes(b"x" * 10)
    inv = sp.local_inventory(tmp_path, [r"M 31\Stacked_120_M 31_10.0s_IRCUT_20261118-213000.fit", r"Nope\none.jpg"])
    assert inv == {r"M 31\Stacked_120_M 31_10.0s_IRCUT_20261118-213000.fit": 10}


def test_default_destination_sits_under_this_machines_image_dir():
    cfg = {"nina": {"image_dir": "/home/taylor/Desktop/Targets"}, "seestar": {"dest": "seestar"}}
    assert sp.default_dest(cfg) == Path("/home/taylor/Desktop/Targets/seestar")
