"""`!!` in the web chat: the server's last command, and a chat page that is revalidated."""
import pytest

from cmd_processing import cmd_history


def test_last_is_the_newest_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(cmd_history, "_history_path", lambda: tmp_path / "h.json")
    assert cmd_history.last() is None
    cmd_history.record("status")
    cmd_history.record("filters melotte15 Ha=1 O-III=2 S-II=2")
    assert cmd_history.last() == "filters melotte15 Ha=1 O-III=2 S-II=2"


def test_chat_page_is_revalidated_not_reused(tmp_path):
    # The web stack is not on the CI runner (pytest only): skip there.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from cmd_processing.web_server import _RevalidatedStatic
    (tmp_path / "chat.html").write_text("<p>hi</p>")
    (tmp_path / "icon.svg").write_text("<svg/>")
    app = FastAPI()
    app.mount("/static", _RevalidatedStatic(directory=str(tmp_path)), name="static")
    c = TestClient(app)
    assert c.get("/static/chat.html").headers["cache-control"] == "no-cache"
    assert "cache-control" not in c.get("/static/icon.svg").headers
