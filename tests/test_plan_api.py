"""GET /v1/plan (iris/core/api.py, ADR 0015): the night's plan as the scheduler reads it."""
import sys
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from iris.core.api import build_app  # noqa: E402
from iris.core.journal import Journal  # noqa: E402


class _Conductor:
    state, slots, drives_checks = "ARMED", 1, True


def test_plan_serves_the_latest_record_per_check(tmp_path):
    j = Journal(tmp_path)
    today = date.today().isoformat()
    j.append("note", "NIGHT_PLAN", "clock", data={"date": today, "check": "noon", "outcome": "bad"})
    j.append("note", "NIGHT_PLAN", "clock", data={"date": today, "check": "noon", "outcome": "good",
                                                  "plan": {"best": "ngc7790"}})
    client = TestClient(build_app(_Conductor(), j, lambda: {}))
    r = client.get("/v1/plan").json()
    assert r["date"] == today and r["drives_checks"] is True and r["state"] == "ARMED"
    assert r["noon"]["outcome"] == "good" and r["noon"]["seq"] == 2 and r["sunset"] is None
    assert client.get("/v1/plan?date=2026-01-01").json()["noon"] is None
    assert client.get("/v1/plan?date=junk").status_code == 400
