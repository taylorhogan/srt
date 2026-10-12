"""Who may talk to the chat and the conductor (2026-10-12).

Neither has a login. With the "remote Any" firewall rules deleted, the house
LAN could still reach the chat through Windows' python.exe program rules, so
the peer address is checked inside the app.
"""
import asyncio

import pytest

from utils.client_allowlist import LOOPBACK, TAILNET, ClientAllowlist, allowed, describe, parse

CHAT = parse(LOOPBACK + TAILNET)
CONDUCTOR = parse(LOOPBACK)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "::ffff:127.0.0.1",
                                  "100.95.7.19", "100.91.17.119", "fd7a:115c:a1e0::1"])
def test_this_pc_and_the_tailnet_are_served(host):
    assert allowed(host, CHAT)


@pytest.mark.parametrize("host", ["192.168.87.78", "::ffff:192.168.87.20", "10.0.0.5",
                                  "8.8.8.8", "testclient", None, ""])
def test_the_house_lan_and_anything_else_is_refused(host):
    assert not allowed(host, CHAT)


def test_the_conductor_serves_this_pc_only():
    assert allowed("127.0.0.1", CONDUCTOR)
    assert not allowed("100.95.7.19", CONDUCTOR)          # not even the tailnet


def test_the_refusal_names_who_is_served():
    assert describe(CONDUCTOR) == "this PC only"
    assert describe(CHAT) == "this PC and the tailnet only"
    assert describe(parse(["10.0.0.0/8"])) == "only 10.0.0.0/8"


def _drive(scope):
    called, sent = [], []

    async def inner(scope, receive, send):
        called.append(scope["type"])

    async def receive():
        return {"type": "http.request"}

    async def send(msg):
        sent.append(msg)
    asyncio.run(ClientAllowlist(inner, LOOPBACK + TAILNET, name="test")(scope, receive, send))
    return called, sent


def test_middleware_refuses_http_and_websocket_from_the_lan():
    called, sent = _drive({"type": "http", "client": ("192.168.87.20", 5555), "path": "/api/post"})
    assert called == [] and sent[0]["status"] == 403
    called, sent = _drive({"type": "websocket", "client": ("192.168.87.20", 5555), "path": "/ws"})
    assert called == [] and sent == [{"type": "websocket.close", "code": 1008}]


def test_middleware_passes_loopback_tailnet_and_lifespan():
    assert _drive({"type": "http", "client": ("127.0.0.1", 1), "path": "/"})[0] == ["http"]
    assert _drive({"type": "websocket", "client": ("100.95.7.19", 1), "path": "/ws"})[0] == ["websocket"]
    assert _drive({"type": "lifespan"})[0] == ["lifespan"]


def test_the_conductor_app_is_wired():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from iris.core.api import build_app
    app = build_app(object(), object(), lambda: {}, allowed_networks=list(LOOPBACK))
    assert TestClient(app).get("/v1/state").status_code == 403    # TestClient's peer is "testclient"


def test_the_chat_app_is_wired():
    ws = pytest.importorskip("cmd_processing.web_server")
    assert any(m.cls is ClientAllowlist for m in ws.app.user_middleware)
