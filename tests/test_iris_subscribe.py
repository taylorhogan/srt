"""scripts/host/iris_subscribe: the web host's subscription service, stdlib only.

The HTTP layer is exercised through the real handler on a loopback port with
a fake mailer, so the GET-does-not-unsubscribe rule and the always-the-same
answer on subscribe are tested as a client would see them.
"""
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from scripts.host import iris_subscribe as s


class FakeMailer:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send(self, messages):
        if self.fail:
            raise RuntimeError("provider down")
        self.sent.extend(messages)


@pytest.fixture
def store(tmp_path):
    return s.Store(str(tmp_path / "s.db"))


# --- validation and limits ---------------------------------------------------

def test_email_validation():
    assert s.valid_email("  Taylor@Example.ORG ") == "taylor@example.org"
    assert s.valid_email("nope") == "" and s.valid_email("a@b") == "" and s.valid_email("") == ""
    assert s.valid_email("x" * 250 + "@a.bc") == ""


def test_rate_limiter_per_ip_and_daily():
    r = s.RateLimiter(per_ip=(2, 3600), daily=(3, 86400))
    assert r.allow_request("1.1.1.1", now=0) and r.allow_request("1.1.1.1", now=1)
    assert not r.allow_request("1.1.1.1", now=2)
    assert r.allow_request("2.2.2.2", now=2)               # other client unaffected
    assert r.allow_request("1.1.1.1", now=3700)            # window passed
    assert [r.allow_mail(now=10 + i) for i in range(4)] == [True, True, True, False]
    assert r.allow_mail(now=10 + 90000)


# --- store transitions -------------------------------------------------------

def test_subscribe_confirm_unsubscribe_resubscribe(store):
    t = store.subscribe("a@b.co", now=1000.0)
    assert t and store.counts() == {"pending": 1}
    assert store.subscribe("a@b.co", now=2000.0) is None        # mailed within 24 h: no second mail
    t2 = store.subscribe("a@b.co", now=1000.0 + 90000)
    assert t2 and t2 != t                                       # a day later: fresh token
    assert store.confirm("wrong") is None and store.confirm(t) is None   # old token dead
    assert store.confirm(t2) == "a@b.co" and store.counts() == {"confirmed": 1}
    assert store.confirm(t2) is None                            # one-shot
    assert store.subscribe("a@b.co") is None                    # confirmed: nothing to do
    (email, unsub), = store.confirmed()
    assert email == "a@b.co" and store.by_unsub_token(unsub) == "a@b.co"
    assert store.unsubscribe("bad") is None
    assert store.unsubscribe(unsub) == "a@b.co" and store.counts() == {"unsubscribed": 1}
    assert store.confirmed() == []
    assert store.subscribe("a@b.co", now=5e5)                   # starts over as pending
    assert store.counts() == {"pending": 1}


def test_pending_purge(store):
    store.subscribe("old@b.co", now=0.0)
    store.subscribe("new@b.co", now=10 * 86400.0)
    assert store.purge_pending(days=7, now=10 * 86400.0) == 1
    assert [r["email"] for r in store.rows()] == ["new@b.co"]


def test_sent_log(store):
    assert store.sent_ids() == set()
    store.mark_sent("m33", 0, 12); store.mark_sent("m33", 1, 3)
    assert store.sent_ids() == set() and store.sent_batches("m33") == {0, 1}   # not complete yet
    store.mark_complete("m33", 15)
    assert store.sent_ids() == {"m33"} and store.sent_batches("m33") == {0, 1}
    store.forget_sent("m33")
    assert store.sent_ids() == set()


# --- announcements -----------------------------------------------------------

ARTS = [{"id": "m33", "kind": "image", "title": "M33", "date": None, "summary": "A galaxy.",
         "image_url": "https://irisscience.org/images/m33.jpg", "url": "https://irisscience.org/#m33"},
        {"id": "quintet", "kind": "note", "title": "Quintet", "date": "2026-10-06", "summary": "Redshift.",
         "image_url": None, "url": "https://irisscience.org/#quintet"}]


def _confirmed(store, *emails):
    for e in emails:
        store.confirm(store.subscribe(e))


def test_plan_announcements_is_pure():
    assert [a["id"] for a in s.plan_announcements(ARTS, set())] == ["m33", "quintet"]
    assert [a["id"] for a in s.plan_announcements(ARTS, {"m33"})] == ["quintet"]
    assert [a["id"] for a in s.plan_announcements(ARTS, {"m33", "quintet"}, again="m33")] == ["m33"]
    assert [a["id"] for a in s.plan_announcements(ARTS, set(), only="quintet")] == ["quintet"]


def test_announce_skips_sent_and_records_each_batch(store, monkeypatch):
    monkeypatch.setattr(s, "BATCH", 2)
    _confirmed(store, "a@b.co", "b@b.co", "c@b.co")
    store.mark_complete("m33", 0)                                 # already announced (seeded)
    m = FakeMailer(); out = io.StringIO()
    assert s.announce(store, m, ARTS, out=out) == 3
    assert {x["to"][0] for x in m.sent} == {"a@b.co", "b@b.co", "c@b.co"}
    assert all(x["subject"] == "New note: Quintet" for x in m.sent)
    assert store.sent_batches("quintet") == {0, 1} and "m33" in store.sent_ids()
    hdr = m.sent[0]["headers"]
    assert hdr["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click" and "/api/unsubscribe?t=" in hdr["List-Unsubscribe"]
    assert "Unsubscribe" in m.sent[0]["text"] and "irisscience.org/#quintet" in m.sent[0]["text"]
    # a second run has nothing to do
    assert s.announce(store, m, ARTS, out=io.StringIO()) == 0 and len(m.sent) == 3


def test_announce_failed_batch_is_not_recorded_and_resumes(store, monkeypatch):
    monkeypatch.setattr(s, "BATCH", 1)
    _confirmed(store, "a@b.co", "b@b.co")
    calls = {"n": 0}

    class Flaky:
        def __init__(self): self.sent = []
        def send(self, msgs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("provider hiccup")
            self.sent.extend(msgs)
    f = Flaky()
    with pytest.raises(RuntimeError):
        s.announce(store, f, ARTS[:1], out=io.StringIO())
    assert store.sent_batches("m33") == {0}                       # first batch recorded, second not
    assert s.announce(store, f, ARTS[:1], out=io.StringIO()) == 1  # resumes with batch 1 only
    assert store.sent_batches("m33") == {0, 1} and len(f.sent) == 2 and store.sent_ids() == {"m33"}


def test_dry_run_and_test_recipient_send_nothing_to_the_list(store):
    _confirmed(store, "a@b.co", "b@b.co")
    m = FakeMailer(); out = io.StringIO()
    assert s.announce(store, m, ARTS, dry_run=True, out=out) == 0 and m.sent == []
    assert "Subject: New image: M33" in out.getvalue() and "confirmed subscribers: 2" in out.getvalue()
    assert s.announce(store, m, ARTS, to="me@b.co", out=io.StringIO()) == 2
    assert all(x["to"] == ["me@b.co"] for x in m.sent) and store.sent_ids() == set()


def test_again_resends_one_article(store):
    _confirmed(store, "a@b.co")
    store.mark_complete("quintet", 1)
    m = FakeMailer()
    assert s.announce(store, m, ARTS, again="quintet", out=io.StringIO()) == 1
    assert m.sent[0]["subject"] == "New note: Quintet" and store.sent_batches("quintet") == {0}


def test_messages_escape_html_and_omit_missing_image():
    art = dict(ARTS[1], title='A <b>"title"</b>', summary="x & y")
    msg = s.announcement_message(art, "a@b.co", "TOK")
    assert "&lt;b&gt;&quot;title&quot;&lt;/b&gt;" in msg["html"] and "<img" not in msg["html"]
    assert "x &amp; y" in msg["html"] and 'A <b>"title"</b>' in msg["text"]
    assert "<img" in s.announcement_message(ARTS[0], "a@b.co", "TOK")["html"]
    c = s.confirmation_message("a@b.co", "TOK")
    assert "/api/confirm?t=TOK" in c["text"] and c["to"] == ["a@b.co"]


def test_read_env(tmp_path):
    p = tmp_path / "e"; p.write_text('# c\nRESEND_API_KEY="re_123"\nX=1\n')
    assert s.read_env(str(p)) == {"RESEND_API_KEY": "re_123", "X": "1"}
    assert s.read_env(str(tmp_path / "missing")) == {}


# --- the HTTP service, as a client sees it -----------------------------------

@pytest.fixture
def service(store):
    mailer = FakeMailer()
    limiter = s.RateLimiter(per_ip=(3, 3600), daily=(2, 86400))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), s.make_handler(store, limiter, mailer))
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    yield "http://127.0.0.1:%d" % srv.server_address[1], store, mailer
    srv.shutdown()


def _call(url, method="GET", body=None, headers=None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.headers, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read().decode()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def test_http_subscribe_confirm_unsubscribe_flow(service):
    base, store, mailer = service
    urllib.request.install_opener(urllib.request.build_opener(NoRedirect))
    j = lambda d: json.dumps(d).encode()
    code, _, body = _call(base + "/api/subscribe", "POST", j({"email": "a@b.co", "hp": ""}),
                          {"Content-Type": "application/json", "Cf-Connecting-IP": "9.9.9.9"})
    assert code == 200 and "Check your inbox" in body and len(mailer.sent) == 1
    token = mailer.sent[0]["text"].split("t=")[1].split()[0]
    # same answer for a repeat, but no second mail
    code, _, body2 = _call(base + "/api/subscribe", "POST", j({"email": "a@b.co", "hp": ""}),
                           {"Content-Type": "application/json", "Cf-Connecting-IP": "9.9.9.9"})
    assert code == 200 and body2 == body and len(mailer.sent) == 1
    # honeypot and junk are 400
    assert _call(base + "/api/subscribe", "POST", j({"email": "b@b.co", "hp": "bot"}))[0] == 400
    assert _call(base + "/api/subscribe", "POST", j({"email": "junk"}))[0] == 400
    assert _call(base + "/api/subscribe", "POST", b"{")[0] == 400
    # confirm: one-shot redirect to the site
    code, h, _ = _call(base + "/api/confirm?t=" + token)
    assert code == 303 and h["Location"].endswith("/#subscribed") and store.counts() == {"confirmed": 1}
    assert _call(base + "/api/confirm?t=" + token)[0] == 400
    (_, unsub), = store.confirmed()
    # GET unsubscribe shows a button and changes nothing
    code, _, page = _call(base + "/api/unsubscribe?t=" + unsub)
    assert code == 200 and "Yes, unsubscribe" in page and "a@b.co" in page and store.counts() == {"confirmed": 1}
    assert _call(base + "/api/unsubscribe?t=nope")[0] == 400
    # POST (button or one-click) does
    code, h, _ = _call(base + "/api/unsubscribe?t=" + unsub, "POST", b"List-Unsubscribe=One-Click",
                       {"Content-Type": "application/x-www-form-urlencoded"})
    assert code == 303 and h["Location"].endswith("/#unsubscribed") and store.counts() == {"unsubscribed": 1}
    assert _call(base + "/api/health")[0] == 200 and _call(base + "/api/nothing")[0] == 404


def test_http_limits_give_the_same_answer_and_no_mail(service):
    base, store, mailer = service
    j = lambda d: json.dumps(d).encode()
    for i in range(5):
        code, _, body = _call(base + "/api/subscribe", "POST", j({"email": "u%d@b.co" % i}),
                              {"Content-Type": "application/json", "Cf-Connecting-IP": "7.7.7.7"})
        assert code == 200 and "Check your inbox" in body
    # 3 requests/IP allowed, 2 mails/day: two mails went out, three rows exist, the rest silently dropped
    assert len(mailer.sent) == 2 and store.counts() == {"pending": 3}


def test_http_provider_failure_is_a_503(store):
    mailer = FakeMailer(fail=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), s.make_handler(store, s.RateLimiter(), mailer))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        code, _, body = _call("http://127.0.0.1:%d/api/subscribe" % srv.server_address[1], "POST",
                              json.dumps({"email": "a@b.co"}).encode(), {"Content-Type": "application/json"})
        assert code == 503 and "try again later" in body
    finally:
        srv.shutdown()
