"""scripts/live_watchdog.diagnose: what the stale-feed push tells the owner."""
from scripts.live_watchdog import diagnose

SCP_255 = ('File "scripts/live_push.py", line 65, in push\n'
           '    subprocess.run(["scp", ...\n'
           "subprocess.CalledProcessError: Command '['scp', ...]' returned non-zero exit status 255.")


def test_tailscale_logged_out_names_the_fix():
    msg = diagnose(SCP_255, "NeedsLogin")
    assert "Tailscale on iris-pc is logged out" in msg
    assert "tray icon" in msg and "key expiry" in msg


def test_upload_failure_with_tailscale_running_points_at_the_server():
    assert "cannot upload it to the web server" in diagnose(SCP_255, "Running")


def test_generator_error_quotes_its_last_line():
    msg = diagnose("Traceback...\nValueError: no sky solution", "Running")
    assert "chart generator on iris-pc is failing: ValueError: no sky solution" in msg


def test_nothing_visible():
    assert "not visible from iris-pc" in diagnose("", None)
