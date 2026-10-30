"""C4 of docs/AUTOMATION_PLAN.md (2026-10-30): `image <dso>` stores the object type on the row.

Needs the web chat server's import stack: venv only.
"""
import pytest

ss = pytest.importorskip("cmd_processing.social_server")


def _world(monkeypatch, letter):
    posted, added = [], []

    class Obj:
        class coord:
            class ra:
                deg = 83.63308
            class dec:
                deg = 22.01446
    monkeypatch.setattr(ss.astro_dso_visibility, "is_a_dso_object", lambda n: Obj())
    monkeypatch.setattr(ss.astro_dso_visibility, "get_dso_type", lambda n: letter)
    monkeypatch.setattr(ss.instructions, "add_dso_object_instruction",
                        lambda *a, **k: added.append((a, k)) or True)
    monkeypatch.setattr(ss, "post_social_message", lambda m: posted.append(m))
    return posted, added


def test_image_stores_the_type_and_says_so(monkeypatch):
    posted, added = _world(monkeypatch, "N")
    ss.image_cmd(["", "image", "m", "1"], 1, None, "taylor")
    (args, kw), = added
    assert args[0] == "m 1" and kw["obj_type"] == "nebula"
    assert kw["ra_deg"] == 83.63308 and kw["dec_deg"] == 22.01446
    assert "nebula: narrowband" in posted[-1]
    posted, added = _world(monkeypatch, "G")
    ss.image_cmd(["", "image", "ngc1961"], 1, None, "taylor")
    assert added[0][1]["obj_type"] == "galaxy" and "galaxy: L+RGB" in posted[-1]


def test_an_unknown_or_cluster_type_is_not_stored(monkeypatch):
    for letter in ("?", "C"):
        posted, added = _world(monkeypatch, letter)
        ss.image_cmd(["", "image", "m13"], 1, None, "taylor")
        assert added[0][1]["obj_type"] is None and "type unknown" in posted[-1]


def test_a_failing_type_lookup_still_queues(monkeypatch):
    posted, added = _world(monkeypatch, "N")

    def boom(n):
        raise RuntimeError("SIMBAD down")
    monkeypatch.setattr(ss.astro_dso_visibility, "get_dso_type", boom)
    ss.image_cmd(["", "image", "m1"], 1, None, "taylor")
    assert added and added[0][1]["obj_type"] is None and "added to the queue" in posted[-1]


def test_a_catalogue_name_gets_its_kind_when_the_classifier_cannot_say(monkeypatch):
    posted, added = _world(monkeypatch, "?")
    ss.image_cmd(["", "image", "ic1795"], 1, None, "taylor")             # in configs/candidates.json
    assert added[0][1]["obj_type"] == "nebula" and "nebula: narrowband" in posted[-1]
    posted, added = _world(monkeypatch, "?")
    ss.image_cmd(["", "image", "ngc99999"], 1, None, "taylor")
    assert added[0][1]["obj_type"] is None
