"""Who may talk to the web chat (8095) and the conductor API (8096): an address check inside the app.

Neither has a login, and both listen on every interface. The 2026-10-09 audit
found the chat open to the whole house LAN -- phones, guests, the
cloud-connected Kasa cameras and plugs could run stop!, roof!!, image!!,
update -- and on 2026-10-12, with the two "remote Any" firewall rules for 8095
deleted, it still was: Windows also carries program rules that let python.exe
accept any port from any address on the Public profile the LAN sits in.

So the rule lives here, where no firewall rule can widen it: a request or
websocket from outside the allowlist is refused (HTTP 403, websocket closed
1008) before any route sees it. The binding is unchanged on purpose: Windows
does not treat 127.0.0.1:port and 0.0.0.0:port as a conflict (the 2026-09-07
shadowed-8095 incident), and start_srt / service_check reason about ports.

    chat       loopback (scheduler, NINA scripts, live_skymap ...) + the tailnet
               (the phone, the Spark posting to /api/post)
    conductor  loopback only: every caller is on this PC

One narrow exception (owner-approved 2026-10-25): *path_networks* serves a
single exact path, GET only, to extra addresses -- the read-only /api/dash for
the Core2 desk display (192.168.87.54, DHCP-pinned), which cannot join the
tailnet. Never a websocket, never another method, never another path: from
those addresses everything else is refused exactly as before.
"""
import ipaddress
import logging
import time

_logger = logging.getLogger(__name__)

LOOPBACK = ("127.0.0.0/8", "::1/128")
TAILNET = ("100.64.0.0/10", "fd7a:115c:a1e0::/48")      # Tailscale's IPv4 CGNAT + IPv6 ranges
_LOG_EVERY_S = 600.0


def parse(networks) -> tuple:
    return tuple(ipaddress.ip_network(n, strict=False) for n in networks)


def allowed(host, nets) -> bool:
    """True when *host* (the peer address) is inside one of *nets*. Pure.

    Anything that is not an IP address -- a unix socket, a test client's
    placeholder name -- is refused."""
    try:
        ip = ipaddress.ip_address(str(host))
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return any(ip.version == n.version and ip in n for n in nets)


def describe(nets) -> str:
    """Who *nets* serves, for the refusal log line. Pure."""
    s, loop, tail = set(nets), set(parse(LOOPBACK)), set(parse(TAILNET))
    if s == loop:
        return "this PC only"
    if s == loop | tail:
        return "this PC and the tailnet only"
    return "only " + ", ".join(str(n) for n in nets)


class ClientAllowlist:
    """ASGI middleware: refuse HTTP and websocket peers outside *networks*."""

    def __init__(self, app, networks, name: str = "server", path_networks=None):
        self.app = app
        self.nets = parse(networks)
        self.name = name
        self.serves = describe(self.nets)
        self.path_nets = {path: parse(nets) for path, nets in (path_networks or {}).items() if nets}
        self._logged = {}

    def _path_allows(self, host, scope) -> bool:
        """True only for an HTTP GET of an exact *path_networks* path from one of its addresses."""
        nets = self.path_nets.get(scope.get("path"))
        return (bool(nets) and scope.get("type") == "http"
                and scope.get("method") == "GET" and allowed(host, nets))

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            host = (scope.get("client") or (None, None))[0]
            if not allowed(host, self.nets) and not self._path_allows(host, scope):
                self._log(host, scope)
                if scope["type"] == "http":
                    await send({"type": "http.response.start", "status": 403,
                                "headers": [(b"content-type", b"text/plain")]})
                    await send({"type": "http.response.body", "body": b"forbidden"})
                else:
                    await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)

    def _log(self, host, scope):
        now = time.time()
        if now - self._logged.get(host, 0.0) >= _LOG_EVERY_S:
            self._logged[host] = now
            _logger.warning("%s: refused %s %s from %s (serves %s)",
                            self.name, scope.get("type"), scope.get("path"), host, self.serves)
