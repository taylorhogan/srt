#!/bin/bash
# Install the irisscience.org subscription service on the web host `iris`.
#
#   RESEND_API_KEY=re_xxx bash scripts/host/install_subscribe.sh
#
# Creates the irismail user, writes /etc/iris-mail.env (the only secret),
# installs /usr/local/bin/iris-subscribe and its systemd unit, adds the
# /api/* route to Caddy (backup kept), seeds the sent-log with everything
# already on the site so nothing old is ever announced, and smoke-tests.
# Re-runnable: every step is idempotent.
set -e
cd "$(dirname "$0")/../.."
if [ -z "$RESEND_API_KEY" ] && ! ssh -i ~/.ssh/id_ed25519_iris taylor@100.91.17.119 'test -s /etc/iris-mail.env'; then
    echo "RESEND_API_KEY=... must be set the first time"; exit 2
fi
K=~/.ssh/id_ed25519_iris; H=taylor@100.91.17.119
./.venv/Scripts/python.exe scripts/site_articles.py "$HOME/Documents/development/taylorhogan.github.io/index.html" > /tmp/iris_articles.json
scp -q -i $K scripts/host/iris_subscribe.py scripts/host/iris-subscribe.service scripts/host/caddy_api_snippet.txt /tmp/iris_articles.json $H:/tmp/
ssh -i $K $H "RESEND_API_KEY='$RESEND_API_KEY' bash -s" <<'EOF'
set -e
id -u irismail >/dev/null 2>&1 || sudo useradd --system --home /var/lib/iris-subscribe --shell /usr/sbin/nologin irismail
if [ -n "$RESEND_API_KEY" ]; then
    printf 'RESEND_API_KEY=%s\n' "$RESEND_API_KEY" | sudo tee /etc/iris-mail.env >/dev/null
fi
sudo chown root:irismail /etc/iris-mail.env; sudo chmod 640 /etc/iris-mail.env
sudo install -m 755 /tmp/iris_subscribe.py /usr/local/bin/iris-subscribe
sudo install -m 644 /tmp/iris-subscribe.service /etc/systemd/system/iris-subscribe.service
sudo install -d -o irismail -g irismail -m 750 /var/lib/iris-subscribe
if ! grep -q 'handle /api/\*' /etc/caddy/Caddyfile; then
    sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak-subscribe
    sudo python3 - <<PY
p="/etc/caddy/Caddyfile"; t=open(p).read(); s=open("/tmp/caddy_api_snippet.txt").read()
a="\tencode gzip zstd\n"; assert t.count(a)==1
open(p,"w").write(t.replace(a, a+s))
PY
    sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1 | tail -1
    sudo chown caddy:caddy /var/log/caddy/access.log 2>/dev/null || true
    sudo systemctl reload caddy
fi
sudo systemctl daemon-reload
sudo systemctl enable --now iris-subscribe
sudo systemctl restart iris-subscribe
sleep 2
sudo -u irismail iris-subscribe seed < /tmp/iris_articles.json
echo "--- service:"; systemctl is-active iris-subscribe
echo "--- health via caddy:"; curl -s http://127.0.0.1:8080/api/health; echo
echo "--- empty subscribe must be a 400:"; curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8080/api/subscribe -d '{}'
echo "--- count:"; sudo -u irismail iris-subscribe count
rm -f /tmp/iris_subscribe.py /tmp/iris-subscribe.service /tmp/caddy_api_snippet.txt /tmp/iris_articles.json
EOF
echo "--- public:"; curl -s "https://irisscience.org/api/health"; echo
