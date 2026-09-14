"""control.imaged_tonight: which targets got lights tonight, from disk."""
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.imaged_tonight import dsos_imaged_on, night_of  # noqa: E402


def test_night_of_is_the_evening_date_until_noon():
    assert night_of(datetime(2026, 9, 14, 4, 36)) == date(2026, 9, 13)   # after midnight
    assert night_of(datetime(2026, 9, 13, 23, 0)) == date(2026, 9, 13)   # evening
    assert night_of(datetime(2026, 9, 14, 12, 1)) == date(2026, 9, 14)   # next noon


def test_dsos_with_lights_under_the_night_only(tmp_path):
    def light(dso, night, n, ext=".fits"):
        d = tmp_path / dso / "cdk17" / night / "LIGHT"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"f{i}{ext}").write_bytes(b"x")
    light("ngc7380", "2026-09-13", 5)
    light("m33", "2026-09-12", 9)                 # an earlier night: not tonight
    light("bubble", "2026-09-13", 0)              # LIGHT folder, no frames
    (tmp_path / "Iris" / "ngc7380").mkdir(parents=True)   # rendered products tree
    (tmp_path / "ngc7380" / "cdk17" / "2026-09-13" / "LIGHT" / "notes.csv").write_text("x")
    assert dsos_imaged_on(tmp_path, date(2026, 9, 13)) == ["ngc7380"]
    assert dsos_imaged_on(tmp_path, date(2026, 9, 12)) == ["m33"]
    assert dsos_imaged_on(tmp_path / "missing", date(2026, 9, 13)) == []
