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


# The Core2 desk display (owner-approved 2026-10-25): one exact path, GET only.
CORE2 = "192.168.87.54"


def _drive_dash(scope):
    called, sent = [], []

    async def inner(scope, receive, send):
        called.append(scope["type"])

    async def receive():
        return {"type": "http.request"}

    async def send(msg):
        sent.append(msg)
    mw = ClientAllowlist(inner, LOOPBACK + TAILNET, name="test",
                         path_networks={"/api/dash": [CORE2 + "/32"]})
    asyncio.run(mw(scope, receive, send))
    return called, sent


def test_the_core2_may_get_the_dashboard():
    called, _ = _drive_dash({"type": "http", "method": "GET", "client": (CORE2, 1), "path": "/api/dash"})
    assert called == ["http"]


@pytest.mark.parametrize("scope", [
    {"type": "http", "method": "GET", "client": (CORE2, 1), "path": "/api/post"},
    {"type": "http", "method": "GET", "client": (CORE2, 1), "path": "/"},
    {"type": "http", "method": "GET", "client": (CORE2, 1), "path": "/api/dash/"},
    {"type": "http", "method": "GET", "client": (CORE2, 1), "path": "/api/dashx"},
    {"type": "http", "method": "POST", "client": (CORE2, 1), "path": "/api/dash"},
    {"type": "http", "client": (CORE2, 1), "path": "/api/dash"},                    # no method
    {"type": "websocket", "client": (CORE2, 1), "path": "/api/dash"},
    {"type": "websocket", "client": (CORE2, 1), "path": "/ws"},
    {"type": "http", "method": "GET", "client": ("192.168.87.55", 1), "path": "/api/dash"},
    {"type": "http", "method": "GET", "client": ("192.168.87.78", 1), "path": "/api/dash"},
])
def test_everything_else_from_the_lan_is_still_refused(scope):
    called, sent = _drive_dash(scope)
    assert called == []
    assert sent[0].get("status") == 403 or sent[0] == {"type": "websocket.close", "code": 1008}


def test_without_path_networks_the_dashboard_is_refused_to_the_lan():
    called, sent = _drive({"type": "http", "method": "GET", "client": (CORE2, 1), "path": "/api/dash"})
    assert called == [] and sent[0]["status"] == 403
