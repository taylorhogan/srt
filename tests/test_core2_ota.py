"""The Core2's over-the-air update, server side (2026-10-26).

The device computes HMAC-SHA256 over the image as it downloads and boots it
only on a match; these pin the server's half of that contract.
"""
import hashlib
import hmac
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmd_processing import dash  # noqa: E402
from scripts import core2_publish as cp  # noqa: E402


def test_the_signature_is_plain_hmac_sha256_hex():
    key, data = bytes(range(32)), b"firmware bytes"
    assert cp.firmware_hmac(key, data) == hmac.new(key, data, hashlib.sha256).hexdigest()
    assert len(cp.firmware_hmac(key, data)) == 64           # what the sketch checks for


def test_the_key_is_made_once_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "FW_DIR", tmp_path / "fw")
    monkeypatch.setattr(cp, "KEY_BIN", tmp_path / "fw" / "ota_key.bin")
    monkeypatch.setattr(cp, "KEY_H", tmp_path / "ota_key.h")
    k1 = cp.ensure_key()
    k2 = cp.ensure_key()
    assert k1 == k2 and len(k1) == 32
    header = (tmp_path / "ota_key.h").read_text()
    assert "OTA_KEY[32]" in header and header.count("0x") == 32


def test_the_reply_advertises_what_is_published(tmp_path):
    m = tmp_path / "manifest.json"
    assert dash.firmware_info(m) is None                    # nothing published: no "fw"
    m.write_text(json.dumps({"v": "abc1234-1026T0815", "size": 1262171, "hmac": "ab" * 32,
                             "built": "2026-10-26T08:15:00"}))
    assert dash.firmware_info(m) == {"v": "abc1234-1026T0815", "size": 1262171, "hmac": "ab" * 32}
    b = tmp_path / "firmware.bin"
    assert dash.firmware(b) is None
    b.write_bytes(b"\x01\x02")
    assert dash.firmware(b) == b"\x01\x02"
