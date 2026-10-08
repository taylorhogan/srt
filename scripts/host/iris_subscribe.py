#!/usr/bin/env python3
"""irisscience.org subscriptions: sign up on the site, get a mail per new note or image.

Runs on the web host (installed as /usr/local/bin/iris-subscribe by
scripts/host/install_subscribe.sh), stdlib only, Python 3.12.

    iris-subscribe serve                      the HTTP service (systemd), 127.0.0.1:8097
    iris-subscribe count | list               who is on the list
    iris-subscribe seed      < articles.json  mark everything on the site as already sent
    iris-subscribe announce  < articles.json  mail every confirmed subscriber about new ids
         [--dry-run] [--again SLUG] [--only SLUG] [--to EMAIL]
    iris-subscribe test EMAIL                 one announcement-shaped test mail

Caddy proxies /api/* here. Endpoints:
    POST /api/subscribe      {"email": ..., "hp": ""}   -> confirmation mail (double opt-in)
    GET  /api/confirm?t=     one-shot token            -> confirmed, redirect /#subscribed
    GET  /api/unsubscribe?t= shows a button; the GET changes nothing because mail
                             scanners prefetch links
    POST /api/unsubscribe?t= (the button, or RFC 8058 one-click)  -> redirect /#unsubscribed

Abuse: no captcha is possible (the site's CSP allows no third-party script), so
the subscribe endpoint has a honeypot field, 5 requests an hour per client IP
(Cf-Connecting-IP), one confirmation mail per address per 24 h and 20 a day in
all -- the real risk is being used to send confirmation mails to strangers. The
answer is always "check your inbox", whether the address is new, known, or over
a limit, so nothing can be learned about who is subscribed.

Tokens are random (secrets.token_urlsafe) and stored, compared in constant
time; the confirm token is cleared when used, the unsubscribe token lives as
long as the row. Tokens travel in GET URLs and so appear in Caddy's access log;
acceptable for a hobby list. The Resend API key is the only secret, read from
the env file (service: EnvironmentFile; CLI: parsed here).

Sending: Resend's batch endpoint, up to 100 messages per call. A `sent` row is
written per (article, batch) only on a 2xx, so a rerun after a failure resumes
at the failed batch and never re-mails a batch that went out.
"""
import argparse
import hmac
import html
import json
import os
import re
import secrets
import sqlite3
import sys
import time
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

SITE = "https://irisscience.org"
FROM = "Iris Lab Notes <notes@irisscience.org>"
REPLY_TO = None                      # set to the owner's address if replies should go somewhere
DB_PATH = "/var/lib/iris-subscribe/subscribers.db"
ENV_FILE = "/etc/iris-mail.env"
PORT = 8097
RESEND_URL = "https://api.resend.com/emails"
BATCH = 100
DAILY_CAP = 100                      # Resend free tier, confirmations included
PENDING_DAYS = 7
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def valid_email(s) -> str:
    """Lower-cased address if it looks like one, else ''."""
    s = (s or "").strip().lower()
    return s if len(s) <= 254 and EMAIL_RE.match(s) else ""


# --------------------------------------------------------------------------- store

class Store:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS subscribers (
        email TEXT PRIMARY KEY, status TEXT NOT NULL,
        confirm_token TEXT, unsub_token TEXT NOT NULL,
        created_at TEXT NOT NULL, confirm_sent_at TEXT,
        confirmed_at TEXT, unsubscribed_at TEXT);
    CREATE TABLE IF NOT EXISTS sent (
        article_id TEXT NOT NULL, batch_no INTEGER NOT NULL,
        sent_at TEXT NOT NULL, recipients INTEGER NOT NULL,
        PRIMARY KEY (article_id, batch_no));
    """

    def __init__(self, path: str = DB_PATH):
        self.db = sqlite3.connect(path, timeout=5, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(self.SCHEMA)

    # -- subscribers
    def subscribe(self, email: str, now: float = None):
        """-> confirm token to mail, or None (already confirmed / mailed within 24 h).
        Re-subscribing after an unsubscribe starts over as pending."""
        now = now if now is not None else time.time()
        stamp = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")
        row = self.db.execute("SELECT * FROM subscribers WHERE email=?", (email,)).fetchone()
        if row and row["status"] == "confirmed":
            return None
        if row and row["status"] == "pending" and row["confirm_sent_at"]:
            last = datetime.fromisoformat(row["confirm_sent_at"]).timestamp()
            if now - last < 86400:
                return None
        token = secrets.token_urlsafe(32)
        if row:
            self.db.execute("UPDATE subscribers SET status='pending', confirm_token=?, confirm_sent_at=?, "
                            "unsubscribed_at=NULL WHERE email=?", (token, stamp, email))
        else:
            self.db.execute("INSERT INTO subscribers VALUES (?,?,?,?,?,?,NULL,NULL)",
                            (email, "pending", token, secrets.token_urlsafe(32), stamp, stamp))
        self.db.commit()
        return token

    def confirm(self, token: str):
        """-> email confirmed by this one-shot token, or None."""
        row = self._by_token("confirm_token", token)
        if row is None:
            return None
        self.db.execute("UPDATE subscribers SET status='confirmed', confirm_token=NULL, confirmed_at=? "
                        "WHERE email=?", (now_iso(), row["email"]))
        self.db.commit()
        return row["email"]

    def by_unsub_token(self, token: str):
        row = self._by_token("unsub_token", token)
        return row["email"] if row else None

    def unsubscribe(self, token: str):
        row = self._by_token("unsub_token", token)
        if row is None:
            return None
        self.db.execute("UPDATE subscribers SET status='unsubscribed', confirm_token=NULL, unsubscribed_at=? "
                        "WHERE email=?", (now_iso(), row["email"]))
        self.db.commit()
        return row["email"]

    def _by_token(self, column: str, token: str):
        if not token or len(token) > 128:
            return None
        for row in self.db.execute("SELECT * FROM subscribers WHERE %s IS NOT NULL" % column):
            if hmac.compare_digest(row[column], token):
                return row
        return None

    def confirmed(self) -> list:
        """[(email, unsub_token), ...]"""
        return [(r["email"], r["unsub_token"]) for r in
                self.db.execute("SELECT email, unsub_token FROM subscribers WHERE status='confirmed' ORDER BY email")]

    def counts(self) -> dict:
        return {r["status"]: r["n"] for r in
                self.db.execute("SELECT status, COUNT(*) n FROM subscribers GROUP BY status")}

    def rows(self) -> list:
        return [dict(r) for r in self.db.execute("SELECT * FROM subscribers ORDER BY created_at")]

    def purge_pending(self, days: int = PENDING_DAYS, now: float = None) -> int:
        now = now if now is not None else time.time()
        cutoff = datetime.fromtimestamp(now - days * 86400, timezone.utc).isoformat(timespec="seconds")
        cur = self.db.execute("DELETE FROM subscribers WHERE status='pending' AND created_at < ?", (cutoff,))
        self.db.commit()
        return cur.rowcount

    # -- sent log. One row per batch that went out (batch_no >= 0) and a
    # closing row (batch_no = -1) once every batch has: only the closing row
    # makes an article "announced", so a send that failed part way is resumed
    # on the next run, skipping the batches that succeeded.
    def sent_ids(self) -> set:
        return {r["article_id"] for r in self.db.execute("SELECT article_id FROM sent WHERE batch_no=-1")}

    def sent_batches(self, article_id: str) -> set:
        return {r["batch_no"] for r in self.db.execute(
            "SELECT batch_no FROM sent WHERE article_id=? AND batch_no>=0", (article_id,))}

    def mark_sent(self, article_id: str, batch_no: int, recipients: int) -> None:
        self.db.execute("INSERT OR REPLACE INTO sent VALUES (?,?,?,?)", (article_id, batch_no, now_iso(), recipients))
        self.db.commit()

    def mark_complete(self, article_id: str, recipients: int) -> None:
        self.mark_sent(article_id, -1, recipients)

    def forget_sent(self, article_id: str) -> None:
        self.db.execute("DELETE FROM sent WHERE article_id=?", (article_id,))
        self.db.commit()


# --------------------------------------------------------------------------- rate limit

class RateLimiter:
    """In memory; a restart forgets, which is fine for these limits."""

    def __init__(self, per_ip=(5, 3600), daily=(20, 86400)):
        self.per_ip_n, self.per_ip_s = per_ip
        self.daily_n, self.daily_s = daily
        self.ip = defaultdict(deque)
        self.mails = deque()

    def allow_request(self, ip: str, now: float = None) -> bool:
        now = now if now is not None else time.time()
        q = self.ip[ip]
        while q and now - q[0] > self.per_ip_s:
            q.popleft()
        if len(q) >= self.per_ip_n:
            return False
        q.append(now)
        return True

    def allow_mail(self, now: float = None) -> bool:
        now = now if now is not None else time.time()
        while self.mails and now - self.mails[0] > self.daily_s:
            self.mails.popleft()
        if len(self.mails) >= self.daily_n:
            return False
        self.mails.append(now)
        return True


# --------------------------------------------------------------------------- mail

def read_env(path: str = ENV_FILE) -> dict:
    out = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"')
    except OSError:
        pass
    return out


class Resend:
    def __init__(self, api_key: str, url: str = RESEND_URL, opener=None):
        self.key = api_key
        self.url = url
        self.opener = opener or urllib.request.urlopen

    def send(self, messages: list) -> None:
        """One message -> POST /emails; several -> POST /emails/batch. Raises on failure."""
        if not messages:
            return
        single = len(messages) == 1
        req = urllib.request.Request(
            self.url if single else self.url + "/batch",
            data=json.dumps(messages[0] if single else messages).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json",
                     "Idempotency-Key": secrets.token_hex(16)},
            method="POST")
        with self.opener(req, timeout=30) as r:
            if not 200 <= r.status < 300:
                raise RuntimeError("resend %s" % r.status)


def _footer(unsub_url: str) -> tuple:
    text = ("\n\n--\nIris Lab Notes, an automated observatory's notebook: %s\n"
            "You get this because you subscribed on the site. Unsubscribe in one click: %s\n" % (SITE, unsub_url))
    htm = ('<p style="color:#8b9bb4;font-size:0.85em;margin-top:2em">Iris Lab Notes, an automated '
           'observatory\'s notebook: <a href="%s" style="color:#8b9bb4">%s</a><br>You get this because you '
           'subscribed on the site. <a href="%s" style="color:#8b9bb4">Unsubscribe</a> in one click.</p>'
           % (SITE, SITE, html.escape(unsub_url)))
    return text, htm


def confirmation_message(email: str, token: str) -> dict:
    url = "%s/api/confirm?t=%s" % (SITE, token)
    text = ("Confirm your subscription to Iris Lab Notes\n\nSomeone, probably you, asked for an email "
            "whenever the observatory publishes a new note or image. If that was you, confirm here:\n\n%s\n\n"
            "If it was not, ignore this and nothing more will be sent.\n" % url)
    htm = ('<p>Someone, probably you, asked for an email whenever the observatory publishes a new note or '
           'image.</p><p><a href="%s" style="display:inline-block;padding:0.6em 1.1em;background:#3e64ff;'
           'color:#fff;text-decoration:none;border-radius:4px">Confirm my subscription</a></p>'
           '<p style="color:#8b9bb4;font-size:0.85em">If it was not you, ignore this and nothing more will be sent.'
           '</p>' % html.escape(url))
    return {"from": FROM, "to": [email], "subject": "Confirm your subscription to Iris Lab Notes",
            "text": text, "html": htm}


def announcement_message(article: dict, email: str, unsub_token: str) -> dict:
    kind = "New image" if article.get("kind") == "image" else "New note"
    title = article.get("title") or article.get("id")
    unsub = "%s/api/unsubscribe?t=%s" % (SITE, unsub_token)
    date = (" (%s)" % article["date"]) if article.get("date") else ""
    text = "%s: %s%s\n\n%s\n\nRead it on the site: %s" % (kind, title, date, article.get("summary", ""), article["url"])
    img = ('<p><a href="%s"><img src="%s" alt="" style="max-width:100%%;border-radius:4px"></a></p>'
           % (html.escape(article["url"]), html.escape(article["image_url"]))) if article.get("image_url") else ""
    htm = ('<p style="color:#8b9bb4;text-transform:uppercase;letter-spacing:1px;font-size:0.8em">%s%s</p>'
           '<h2 style="margin:0.2em 0 0.6em"><a href="%s" style="color:#3e64ff;text-decoration:none">%s</a></h2>%s'
           '<p>%s</p><p><a href="%s">Read it on the site</a></p>'
           % (kind, html.escape(date), html.escape(article["url"]), html.escape(title), img,
              html.escape(article.get("summary", "")), html.escape(article["url"])))
    ft, fh = _footer(unsub)
    msg = {"from": FROM, "to": [email], "subject": "%s: %s" % (kind, title),
           "text": text + ft, "html": htm + fh,
           "headers": {"List-Unsubscribe": "<%s>" % unsub,
                       "List-Unsubscribe-Post": "List-Unsubscribe=One-Click"}}
    if REPLY_TO:
        msg["reply_to"] = REPLY_TO
    return msg


def plan_announcements(articles: list, sent_ids: set, again=None, only=None) -> list:
    """Which articles to announce: not yet sent, or the one asked for again. Pure."""
    if again:
        return [a for a in articles if a["id"] == again]
    todo = [a for a in articles if a["id"] not in sent_ids]
    if only:
        todo = [a for a in todo if a["id"] == only]
    return todo


def announce(store: Store, mailer, articles: list, dry_run=False, again=None, only=None, to=None, out=sys.stdout) -> int:
    """Mail every confirmed subscriber about each new article. Returns messages sent."""
    recipients = store.confirmed()
    if to:
        recipients = [(to, next((t for e, t in recipients if e == to), "test"))]
    todo = plan_announcements(articles, store.sent_ids(), again=again, only=only)
    print("confirmed subscribers: %d%s" % (len(store.confirmed()), "  (test: sending only to %s)" % to if to else ""), file=out)
    if not todo:
        print("nothing new to announce", file=out)
        return 0
    total = len(todo) * len(recipients)
    if total > DAILY_CAP:
        print("WARNING: %d messages exceeds the provider's %d/day cap; some will fail" % (total, DAILY_CAP), file=out)
    sent = 0
    for a in todo:
        print("\n== %s  [%s]  -> %d recipient(s)" % (a["id"], a.get("kind"), len(recipients)), file=out)
        preview = announcement_message(a, "preview@example.org", "TOKEN")
        print("Subject: " + preview["subject"], file=out)
        print(preview["text"].split("\n\n--\n")[0], file=out)
        if dry_run:
            continue
        if again and not to:
            store.forget_sent(a["id"])
        done = store.sent_batches(a["id"]) if not to else set()
        for n in range(0, len(recipients), BATCH):
            batch_no = n // BATCH
            if batch_no in done:
                print("batch %d already sent, skipped" % batch_no, file=out)
                continue
            msgs = [announcement_message(a, e, t) for e, t in recipients[n:n + BATCH]]
            mailer.send(msgs)
            sent += len(msgs)
            if not to:
                store.mark_sent(a["id"], batch_no, len(msgs))
            print("batch %d: sent %d" % (batch_no, len(msgs)), file=out)
        if not to:
            store.mark_complete(a["id"], len(recipients))
    return sent


# --------------------------------------------------------------------------- http

PAGE = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Iris Lab Notes</title><body style="font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;'
        'background:#0b0d17;color:#e0e6ed;max-width:36em;margin:4em auto;padding:0 1em">%s</body>')


def make_handler(store: Store, limiter: RateLimiter, mailer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "iris-subscribe/1"

        def log_message(self, fmt, *args):          # one line per request, no client IP
            sys.stdout.write("%s %s\n" % (self.command, self.path.split("?")[0])); sys.stdout.flush()

        def _send(self, code, body=b"", ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode("utf-8"))

        def _redirect(self, where):
            self._send(303, b"", "text/plain", {"Location": where})

        def _query(self):
            u = urlparse(self.path)
            return u.path, {k: v[0] for k, v in parse_qs(u.query).items()}

        def _body(self, limit=4096) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(min(n, limit)) if n > 0 else b""

        def do_GET(self):
            path, q = self._query()
            if path == "/api/confirm":
                if store.confirm(q.get("t", "")):
                    return self._redirect(SITE + "/#subscribed")
                return self._send(400, (PAGE % "<p>That confirmation link is no longer valid. "
                                        "Subscribe again on the site to get a fresh one.</p>").encode(), "text/html")
            if path == "/api/unsubscribe":
                email = store.by_unsub_token(q.get("t", ""))
                if not email:
                    return self._send(400, (PAGE % "<p>That unsubscribe link is not valid.</p>").encode(), "text/html")
                form = ('<p>Unsubscribe <b>%s</b> from Iris Lab Notes?</p>'
                        '<form method="post" action="/api/unsubscribe?t=%s"><button style="padding:0.6em 1.1em;'
                        'background:#3e64ff;color:#fff;border:0;border-radius:4px;font-size:1em">Yes, unsubscribe</button>'
                        '</form>' % (html.escape(email), html.escape(q.get("t", ""))))
                return self._send(200, (PAGE % form).encode(), "text/html")
            if path == "/api/health":
                return self._json(200, {"ok": True})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            path, q = self._query()
            if path == "/api/unsubscribe":
                self._body()                                   # form or List-Unsubscribe=One-Click; either way
                if store.unsubscribe(q.get("t", "")):
                    return self._redirect(SITE + "/#unsubscribed")
                return self._json(400, {"error": "invalid token"})
            if path == "/api/subscribe":
                try:
                    data = json.loads(self._body().decode("utf-8") or "{}")
                except ValueError:
                    return self._json(400, {"error": "bad json"})
                email = valid_email(data.get("email"))
                if not email or (data.get("hp") or "").strip():
                    return self._json(400, {"error": "that does not look like an email address"})
                ip = self.headers.get("Cf-Connecting-IP") or self.client_address[0]
                answer = {"ok": True, "message": "Check your inbox for a confirmation email."}
                if not limiter.allow_request(ip):
                    return self._json(200, answer)             # same answer: nothing to learn
                try:
                    store.purge_pending()
                    token = store.subscribe(email)
                    if token and limiter.allow_mail():
                        mailer.send([confirmation_message(email, token)])
                except Exception as exc:  # noqa: BLE001
                    sys.stdout.write("subscribe failed: %s\n" % exc); sys.stdout.flush()
                    return self._json(503, {"error": "could not send the confirmation; try again later"})
                return self._json(200, answer)
            self._json(404, {"error": "not found"})

    return Handler


def serve(port: int = PORT, db_path: str = DB_PATH, env_file: str = ENV_FILE) -> None:
    key = os.environ.get("RESEND_API_KEY") or read_env(env_file).get("RESEND_API_KEY", "")
    store = Store(db_path)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, RateLimiter(), Resend(key)))
    sys.stdout.write("iris-subscribe listening on 127.0.0.1:%d\n" % port); sys.stdout.flush()
    srv.serve_forever()


# --------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="iris-subscribe")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--env", default=ENV_FILE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve").add_argument("--port", type=int, default=PORT)
    sub.add_parser("count"); sub.add_parser("list"); sub.add_parser("seed")
    sub.add_parser("test").add_argument("email")
    an = sub.add_parser("announce")
    an.add_argument("--dry-run", action="store_true"); an.add_argument("--again"); an.add_argument("--only"); an.add_argument("--to")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        serve(a.port, a.db, a.env); return 0
    store = Store(a.db)
    if a.cmd == "count":
        c = store.counts(); print("confirmed %d, pending %d, unsubscribed %d, articles announced %d" % (
            c.get("confirmed", 0), c.get("pending", 0), c.get("unsubscribed", 0), len(store.sent_ids()))); return 0
    if a.cmd == "list":
        for r in store.rows():
            print("%-12s %s  since %s" % (r["status"], r["email"], r["created_at"][:10]))
        return 0
    if a.cmd == "seed":
        ids = [x["id"] for x in json.load(sys.stdin)["articles"]]
        new = [i for i in ids if i not in store.sent_ids()]
        for i in new:
            store.mark_complete(i, 0)
        print("seeded %d article(s) as already announced (%d total)" % (len(new), len(store.sent_ids()))); return 0
    key = os.environ.get("RESEND_API_KEY") or read_env(a.env).get("RESEND_API_KEY", "")
    mailer = Resend(key)
    if a.cmd == "test":
        art = {"id": "test", "kind": "note", "title": "A test from Iris Lab Notes", "date": now_iso()[:10],
               "summary": "If you can read this, announcements will reach you.", "image_url": None, "url": SITE}
        mailer.send([announcement_message(art, valid_email(a.email), "TEST")]); print("sent to", a.email); return 0
    if a.cmd == "announce":
        arts = json.load(sys.stdin)["articles"]
        n = announce(store, mailer, arts, dry_run=a.dry_run, again=a.again, only=a.only, to=a.to)
        print("\n%s%d message(s)" % ("would send " if a.dry_run else "sent ", n if not a.dry_run else
              len(plan_announcements(arts, store.sent_ids(), a.again, a.only)) * (1 if a.to else len(store.confirmed()))))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
