"""
probe_kasa_camera.py
Read-only probe of the Kasa cameras on the LAN: which protocol family each one
belongs to, which local ports answer, and whether we can pull a still frame
without the TP-Link cloud.

Nothing here writes to a camera. Every call is a read.

Two different camera families answer to "Kasa", and they need opposite code:

  * IOT.IPCAMERA (KC-series: KC410S, KC420WS, ...) - the legacy Kasa cloud cam.
    No RTSP, no ONVIF. Local access is an HTTPS MJPEG/H.264 stream on port
    19443 with HTTP Basic auth using the *TP-Link cloud account*, behind a
    self-signed certificate. python-kasa cannot open these at all.
  * SMART/Tapo cameras - RTSP on 554 and ONVIF on 2020, authenticated with the
    app's separate "Camera Account". python-kasa exposes both URLs.

The unauthenticated UDP discovery payload names the family outright, so this
script leads with that and only then tries the protocol that fits.

Usage:
    python probe_kasa_camera.py                        # find and classify all
    python probe_kasa_camera.py --host 192.168.87.70   # probe one
    python probe_kasa_camera.py --host 192.168.87.70 --snapshot sky.jpg

Credentials:
    --username/--password  TP-Link cloud account. Legacy KC cameras use this
                           for the 19443 stream (env KASA_USERNAME/KASA_PASSWORD)
    --camera-user/--camera-pass
                           Tapo "Camera Account" from the app, used for RTSP
                           and ONVIF (env KASA_CAM_USER/KASA_CAM_PASS).
                           Falls back to the cloud account if unset.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import requests
import urllib3
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from kasa import Discover

# The KC cameras present a self-signed cert; suppress the per-request warning.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class _LegacyTLSAdapter(HTTPAdapter):
    """TLS settings a 2019-era camera will actually complete a handshake with.

    The KC firmware offers TLS 1.2 but only with AES256-GCM-SHA384 under a
    static-RSA key exchange. OpenSSL 3 excludes static RSA at its default
    security level (no forward secrecy), so a stock requests call dies with
    SSLV3_ALERT_HANDSHAKE_FAILURE before any HTTP happens. SECLEVEL=1 permits
    it again. The cert is self-signed, so verification is off regardless -
    this is a device on the LAN, not a public endpoint.
    """

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def legacy_tls_session():
    session = requests.Session()
    session.mount("https://", _LegacyTLSAdapter())
    return session

# Local ports worth knowing about, and what an open one implies.
PORTS = {
    80: "http",
    443: "https",
    554: "RTSP (Tapo/SmartCam)",
    2020: "ONVIF (Tapo/SmartCam)",
    9999: "Kasa IOT control",
    19443: "Kasa local HTTPS stream (KC-series)",
}

# Legacy KC local stream. The firmware ignores video=mjpeg and always sends
# H.264 regardless, so there is no point asking for anything else.
KC_STREAM_PATHS = [
    "/https/stream/mixed?video=h264&audio=g711",
    "/https/stream/mixed?video=h264&audio=g711&resolution=hd",
]

ONVIF_NS = {
    "device": "http://www.onvif.org/ver10/device/wsdl",
    "media": "http://www.onvif.org/ver10/media/wsdl",
    "media2": "http://www.onvif.org/ver20/media/wsdl",
    "imaging": "http://www.onvif.org/ver20/imaging/wsdl",
    "events": "http://www.onvif.org/ver10/events/wsdl",
    "ptz": "http://www.onvif.org/ver20/ptz/wsdl",
    "analytics": "http://www.onvif.org/ver20/analytics/wsdl",
}

ENVELOPE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
    "{header}<s:Body>{body}</s:Body></s:Envelope>"
)


def _mask(url):
    """Hide the password in a scheme://user:pass@host URL before printing it."""
    if not url or "@" not in url or "//" not in url:
        return url
    scheme, rest = url.split("//", 1)
    creds, host = rest.rsplit("@", 1)
    return f"{scheme}//{creds.split(':', 1)[0]}:***@{host}"


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _find(elem, name):
    if elem is None:
        return None
    for node in elem.iter():
        if _local(node.tag) == name:
            return node
    return None


def _text(elem, name, default=None):
    node = _find(elem, name)
    return node.text if node is not None and node.text is not None else default


# ------------------------------------------------------------------- discovery

async def discover_cameras(host, timeout):
    """Classify every Kasa device from the raw, unauthenticated UDP payload.

    Deliberately does not call ``Device.update()``: cameras refuse it (the
    KC-series is not a supported python-kasa device at all), but the discovery
    payload already carries model, alias, resolution and power state.
    """
    raw = {}

    def on_raw(item):
        raw[item["meta"]["ip"]] = item["discovery_response"]

    print("--- discovery ---")
    try:
        if host:
            await Discover.discover_single(host, on_discovered_raw=on_raw,
                                           discovery_timeout=int(timeout))
        else:
            await Discover.discover(on_discovered_raw=on_raw,
                                    discovery_timeout=int(timeout))
    except Exception as e:
        # Cameras raise on the connect attempt that follows discovery; the raw
        # callback has already fired by then, so keep whatever we collected.
        if not raw:
            print(f"  discovery FAILED - {type(e).__name__}: {e}")
            return []
        print(f"  (device connect raised {type(e).__name__}; using raw payloads)")

    cameras = []
    for ip, info in sorted(raw.items()):
        sysinfo = (info.get("system", {}) or {}).get("get_sysinfo", {}) or {}
        sysinfo = sysinfo.get("system", sysinfo)
        result = info.get("result") or {}
        dtype = sysinfo.get("type") or sysinfo.get("mic_type") or result.get("device_type", "")
        model = sysinfo.get("model") or result.get("device_model", "?")
        alias = sysinfo.get("alias") or result.get("alias") or "?"
        if "CAMERA" not in str(dtype).upper() and "IPCAMERA" not in str(dtype).upper():
            continue

        legacy = str(dtype).upper().startswith("IOT")
        cameras.append({"ip": ip, "model": model, "alias": alias,
                        "type": dtype, "legacy": legacy, "sysinfo": sysinfo})
        print(f"  {ip:<16} {model:<14} {dtype:<16} {alias!r}")
        for key, label in (("resolution", "resolution"),
                           ("camera_switch", "camera on"),
                           ("led_status", "status LED"),
                           ("sw_ver", "firmware"),
                           ("rssi", "wifi rssi")):
            if key in sysinfo:
                print(f"    {label:<12} {sysinfo[key]}")

    if not cameras:
        print("  no cameras in the discovery response")
    return cameras


# ------------------------------------------------------------------ port sweep

def scan_ports(host, timeout=1.5):
    """Which local services the camera actually exposes."""
    print(f"\n--- open ports on {host} ---")
    open_ports = set()
    for port, label in sorted(PORTS.items()):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            state = "OPEN  " if sock.connect_ex((host, port)) == 0 else "closed"
        except OSError:
            state = "closed"
        finally:
            sock.close()
        if state.strip() == "OPEN":
            open_ports.add(port)
        print(f"  {port:<6} {state}  {label}")
    return open_ports


# ------------------------------------------------- legacy KC local HTTPS stream

def probe_kc_stream(host, user, password, snapshot_path=None, timeout=10,
                    dump_stream=0):
    """The only local path off a KC-series camera: HTTPS stream on 19443."""
    print("\n--- Kasa local stream (port 19443) ---")
    if not user:
        print("  no TP-Link credentials given; the stream requires HTTP Basic auth")
        print("  with your cloud account (the same email/password as the app)")
        return None

    session = legacy_tls_session()

    # Firmware differs on whether the Basic password is the cloud password or
    # its MD5 hex; try the plain one first and report which was accepted.
    md5_pw = hashlib.md5((password or "").encode()).hexdigest()
    credentials = [("cloud password", password or ""),
                   ("md5(cloud password)", md5_pw)]

    saw_ok = False
    for path in KC_STREAM_PATHS:
        url = f"https://{host}:19443{path}"
        for label, secret in credentials:
            try:
                # An endless multipart body - stream it and read only what we
                # need, never resp.content. Separate connect/read timeouts so a
                # silent stream surfaces quickly instead of hanging.
                resp = session.get(url, auth=HTTPBasicAuth(user, secret),
                                   verify=False, stream=True,
                                   timeout=(timeout, 8))
            except requests.RequestException as e:
                print(f"  {path}  FAILED - {type(e).__name__}: {str(e)[:120]}")
                break

            status = resp.status_code
            ctype = resp.headers.get("Content-Type", "?")
            if status == 401:
                resp.close()
                continue
            if status >= 400:
                resp.close()
                print(f"  {path}  HTTP {status}")
                break

            saw_ok = True
            print(f"  {path}")
            print(f"    HTTP {status}, Content-Type: {ctype}, auth: {label}")
            if not snapshot_path:
                resp.close()
                return url
            got_frame = grab_frame(resp, snapshot_path, ctype=ctype,
                                   dump_stream=dump_stream)
            resp.close()
            if got_frame:
                return url
            # A 200 that yields no decodable frame is not a working endpoint;
            # keep trying the remaining variants.
            print("    -> no frame from this variant, trying the next")
            break

    if saw_ok:
        print("  the stream authenticates and delivers data, but no variant")
        print("  yielded a decodable frame - see the part listing above")
    else:
        print("  every variant returned 401.")
        print("  The realm is smarthome@tp-link.com, so these are TP-Link cloud")
        print("  credentials - check the email/password by signing out and back")
        print("  into the Kasa app. Note some firmware requires the account to be")
        print("  the camera's *owner*, not a shared/guest account.")
    return None


def _boundary_of(ctype):
    """Pull the boundary token out of a multipart Content-Type header."""
    for field in (ctype or "").split(";"):
        field = field.strip()
        if field.lower().startswith("boundary="):
            return field.split("=", 1)[1].strip().strip('"')
    return None


def _split_parts(buf, delimiter):
    """Yield (headers, body) for each complete multipart part in buf.

    Bodies are trimmed to the declared Content-Length: the raw segment also
    carries the trailing CRLF before the next boundary, and a decoder handed
    those extra bytes will complain.
    """
    for segment in bytes(buf).split(delimiter)[1:]:
        head, sep, body = segment.partition(b"\r\n\r\n")
        if not sep:
            head, sep, body = segment.partition(b"\n\n")
        if not sep:
            continue
        headers = {}
        for line in head.decode("latin-1", "replace").splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()
        declared = headers.get("content-length")
        if declared and declared.isdigit():
            length = int(declared)
            if len(body) < length:
                continue  # truncated tail of the buffer
            body = body[:length]
        yield headers, body


def grab_frame(resp, out_path, ctype=None, dump_stream=0,
               read_seconds=20, max_bytes=8 * 1024 * 1024):
    """Pull one still out of an already-open stream response.

    The KC stream is multipart, interleaving H.264 video parts with G.711
    audio. Reassembling the video parts in order gives a plain Annex-B
    elementary stream that any decoder will open - the first part after a
    connect carries SPS+PPS+IDR, so a single keyframe decodes standalone.

    Buffering locally rather than handing the URL to ffmpeg/OpenCV is
    deliberate: their TLS stacks hit the same handshake failure this script
    works around at the requests layer.
    """
    import time

    boundary = _boundary_of(ctype)
    if not boundary:
        print("    no multipart boundary in Content-Type; cannot parse")
        return False
    delimiter = b"--" + boundary.encode()

    buf = bytearray()
    deadline = time.monotonic() + read_seconds
    try:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if chunk:
                buf.extend(chunk)
            if len(buf) >= max_bytes or time.monotonic() > deadline:
                break
            # One keyframe plus a following boundary is all a decoder needs.
            if buf.count(delimiter) >= 3:
                break
    except requests.RequestException as e:
        print(f"    stream read stopped - {type(e).__name__}: {str(e)[:100]}")
    if not buf:
        print("    stream produced no bytes")
        return False

    if dump_stream:
        raw_path = os.path.splitext(out_path)[0] + ".stream"
        with open(raw_path, "wb") as fh:
            fh.write(buf[:dump_stream])
        print(f"    raw stream ({min(dump_stream, len(buf))} bytes) -> {raw_path}")

    video = bytearray()
    kinds = []
    for headers, body in _split_parts(buf, delimiter):
        kind = headers.get("content-type", "?")
        kinds.append(f"{kind}({headers.get('x-frametype', '-')})")
        if "h264" in kind:
            video.extend(body)
    print(f"    read {len(buf)} bytes; parts: {', '.join(kinds) or 'none'}")
    if not video:
        print("    no H.264 parts found")
        return False

    if not video.startswith(b"\x00\x00\x00\x01") and not video.startswith(b"\x00\x00\x01"):
        print(f"    unexpected payload start: {bytes(video[:16]).hex(' ')}")
        return False

    stream_path = os.path.splitext(out_path)[0] + ".h264"
    with open(stream_path, "wb") as fh:
        fh.write(video)
    print(f"    {len(video)} bytes of H.264 -> {stream_path}")
    return _decode_h264(stream_path, out_path)


def _decode_h264(stream_path, out_path, timeout=30):
    """Turn a raw Annex-B file into a still, OpenCV first then ffmpeg."""
    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        cap = cv2.VideoCapture(stream_path)
        frame = None
        if cap.isOpened():
            ok, candidate = cap.read()
            if ok:
                frame = candidate
        cap.release()
        if frame is not None:
            cv2.imwrite(out_path, frame)
            print(f"    decoded {frame.shape[1]}x{frame.shape[0]} -> {out_path}")
            return True
        print("    OpenCV could not decode the H.264 stream")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("    ffmpeg not on PATH; cannot try a second decoder")
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "h264", "-i", stream_path,
           "-frames:v", "1", "-q:v", "2", out_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        print(f"    ffmpeg timed out after {timeout}s")
        return False
    if proc.returncode == 0 and os.path.exists(out_path):
        print(f"    ffmpeg decoded a frame -> {out_path}")
        return True
    print(f"    ffmpeg failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}")
    return False


# ------------------------------------------------------------------ ONVIF probe

def _ws_security_header(user, password):
    """WS-Security UsernameToken: Base64(SHA1(nonce + created + password))."""
    if not user:
        return ""
    nonce = secrets.token_bytes(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = hashlib.sha1(nonce + created.encode() + (password or "").encode()).digest()
    wsse = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
    wsu = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
    pwd_type = ("http://docs.oasis-open.org/wss/2004/01/"
                "oasis-200401-wss-username-token-profile-1.0#PasswordDigest")
    enc_type = ("http://docs.oasis-open.org/wss/2004/01/"
                "oasis-200401-wss-soap-message-security-1.0#Base64Binary")
    return (
        f'<s:Header><Security s:mustUnderstand="1" xmlns="{wsse}">'
        f"<UsernameToken><Username>{user}</Username>"
        f'<Password Type="{pwd_type}">{base64.b64encode(digest).decode()}</Password>'
        f'<Nonce EncodingType="{enc_type}">{base64.b64encode(nonce).decode()}</Nonce>'
        f'<Created xmlns="{wsu}">{created}</Created>'
        "</UsernameToken></Security></s:Header>"
    )


def _soap(url, body, user, password, timeout=10):
    """POST one SOAP body. Returns (root_element, error_string)."""
    payload = ENVELOPE.format(header=_ws_security_header(user, password), body=body)
    headers = {"Content-Type": "application/soap+xml; charset=utf-8"}
    try:
        resp = requests.post(url, data=payload.encode(), headers=headers,
                             timeout=timeout)
        # Some firmwares want HTTP Digest rather than the WS-Security header.
        if resp.status_code == 401 and user:
            plain = ENVELOPE.format(header="", body=body)
            resp = requests.post(url, data=plain.encode(), headers=headers,
                                 auth=HTTPDigestAuth(user, password or ""),
                                 timeout=timeout)
    except requests.RequestException as e:
        return None, f"{type(e).__name__}: {e}"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        return None, f"HTTP {resp.status_code}, unparseable response ({e})"

    fault = _find(root, "Fault")
    if fault is not None:
        reason = _text(fault, "Text") or _text(fault, "Value") or "unspecified"
        return None, f"SOAP Fault: {reason.strip()}"
    if resp.status_code >= 400:
        return None, f"HTTP {resp.status_code}"
    return root, None


def probe_onvif(host, port, user, password, snapshot_path=None, timeout=10):
    device_url = f"http://{host}:{port}/onvif/device_service"
    print(f"\n--- ONVIF probe: {device_url} ---")

    root, err = _soap(device_url,
                      f'<GetDeviceInformation xmlns="{ONVIF_NS["device"]}"/>',
                      user, password, timeout)
    if err:
        print(f"  GetDeviceInformation FAILED - {err}")
        return
    for field in ("Manufacturer", "Model", "FirmwareVersion", "SerialNumber"):
        print(f"  {field:<16} {_text(root, field)}")

    services = {}
    root, err = _soap(device_url,
                      f'<GetServices xmlns="{ONVIF_NS["device"]}">'
                      "<IncludeCapability>false</IncludeCapability></GetServices>",
                      user, password, timeout)
    if err:
        print(f"  GetServices FAILED - {err}")
    else:
        print("\n  Services advertised:")
        for svc in root.iter():
            if _local(svc.tag) != "Service":
                continue
            ns, addr = _text(svc, "Namespace", ""), _text(svc, "XAddr", "")
            services[ns] = addr
            known = next((k for k, v in ONVIF_NS.items() if v == ns), None)
            print(f"    {known or ns:<10} {addr}")

    media_url = services.get(ONVIF_NS["media"]) or f"http://{host}:{port}/onvif/media_service"
    imaging_url = services.get(ONVIF_NS["imaging"])

    profile_token = None
    root, err = _soap(media_url, f'<GetProfiles xmlns="{ONVIF_NS["media"]}"/>',
                      user, password, timeout)
    if err:
        print(f"\n  GetProfiles FAILED - {err}")
    else:
        print("\n  Media profiles:")
        for prof in root.iter():
            if _local(prof.tag) != "Profiles":
                continue
            token = prof.get("token")
            print(f"    {token:<20} {_text(prof, 'Encoding')} "
                  f"{_text(prof, 'Width')}x{_text(prof, 'Height')} @ "
                  f"{_text(prof, 'FrameRateLimit')}fps")
            profile_token = profile_token or token

    if profile_token:
        root, err = _soap(media_url,
                          f'<GetSnapshotUri xmlns="{ONVIF_NS["media"]}">'
                          f"<ProfileToken>{profile_token}</ProfileToken>"
                          "</GetSnapshotUri>", user, password, timeout)
        snap_uri = None if err else _text(root, "Uri")
        print(f"\n  Snapshot URI : {snap_uri or f'unavailable ({err})'}")
        if snapshot_path and snap_uri:
            _fetch_snapshot(snap_uri, user, password, snapshot_path, timeout)

    _probe_imaging(media_url, imaging_url, user, password, timeout)


def _fetch_snapshot(uri, user, password, out_path, timeout):
    try:
        resp = requests.get(uri, timeout=timeout)
        if resp.status_code == 401:
            resp = requests.get(uri, auth=HTTPDigestAuth(user, password or ""),
                                timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  snapshot fetch FAILED - {type(e).__name__}: {e}")
        return
    with open(out_path, "wb") as fh:
        fh.write(resp.content)
    print(f"  snapshot     : wrote {len(resp.content)} bytes to {out_path}")


def _probe_imaging(media_url, imaging_url, user, password, timeout):
    """Whether exposure and IR-cut can be pinned - see report_exposure_verdict."""
    print("\n--- ONVIF Imaging service (exposure / IR-cut) ---")
    if not imaging_url:
        print("  NOT advertised.")
        report_exposure_verdict(manual=False, ir_lock=False)
        return

    root, err = _soap(media_url, f'<GetVideoSources xmlns="{ONVIF_NS["media"]}"/>',
                      user, password, timeout)
    source_token = None
    if not err:
        for src in root.iter():
            if _local(src.tag) == "VideoSources" and src.get("token"):
                source_token = src.get("token")
                break
    if not source_token:
        print(f"  no VideoSource token ({err or 'none returned'})")
        return

    root, err = _soap(imaging_url,
                      f'<GetImagingSettings xmlns="{ONVIF_NS["imaging"]}">'
                      f"<VideoSourceToken>{source_token}</VideoSourceToken>"
                      "</GetImagingSettings>", user, password, timeout)
    if err:
        print(f"  GetImagingSettings FAILED - {err}")
    else:
        print("  Current settings:")
        for name in ("Brightness", "Contrast", "ColorSaturation", "Sharpness",
                     "IrCutFilter"):
            val = _text(root, name)
            if val is not None:
                print(f"    {name:<18} {val}")
        exposure = _find(root, "Exposure")
        if exposure is not None:
            for name in ("Mode", "Priority", "ExposureTime", "Gain",
                         "MinExposureTime", "MaxExposureTime"):
                val = _text(exposure, name)
                if val is not None:
                    print(f"    Exposure.{name:<9} {val}")

    root, err = _soap(imaging_url,
                      f'<GetOptions xmlns="{ONVIF_NS["imaging"]}">'
                      f"<VideoSourceToken>{source_token}</VideoSourceToken>"
                      "</GetOptions>", user, password, timeout)
    if err:
        print(f"  GetOptions FAILED - {err}")
        return

    ir_modes = [n.text for n in root.iter() if _local(n.tag) == "IrCutFilterModes"]
    exposure_opts = _find(root, "Exposure")
    exp_modes = []
    if exposure_opts is not None:
        exp_modes = [n.text for n in exposure_opts.iter() if _local(n.tag) == "Mode"]
        et = _find(exposure_opts, "ExposureTime")
        if et is not None:
            print(f"    ExposureTime range {_text(et, 'Min')} .. {_text(et, 'Max')} us")
    print(f"    IrCutFilterModes   {ir_modes or '(none)'}")
    print(f"    Exposure modes     {exp_modes or '(none)'}")

    report_exposure_verdict(
        manual=any(m and m.upper() == "MANUAL" for m in exp_modes),
        ir_lock=any(m and m.upper() in ("ON", "OFF") for m in ir_modes),
    )


def report_exposure_verdict(manual, ir_lock):
    """What the exposure answer means for sky measurement."""
    print("\n  VERDICT:")
    if manual:
        print("    Manual exposure available -> sky brightness can be a real")
        print("    photometric quantity. Pin ExposureTime and Gain and mean")
        print("    luminance tracks skyglow directly.")
    else:
        print("    No manual exposure -> auto-exposure renormalises every frame,")
        print("    so absolute brightness does not track the sky. Use")
        print("    scale-invariant metrics (texture, gradient statistics, star")
        print("    counts), or keep a fixed terrestrial object in frame as a")
        print("    brightness anchor.")
    if ir_lock:
        print("    IR-cut filter can be pinned -> lock it so the day/night flip")
        print("    does not masquerade as a weather event.")
    else:
        print("    IR-cut filter cannot be pinned -> expect a discontinuity in")
        print("    every trend at the day/night transition; detect and flag it.")


# ------------------------------------------------------------------------ main

async def main(args):
    cam_user = args.camera_user or args.username
    cam_pass = args.camera_pass or args.password

    cameras = await discover_cameras(args.host, args.timeout)
    if not cameras and args.host:
        # Named a host that discovery did not classify - probe it anyway.
        cameras = [{"ip": args.host, "model": "?", "alias": "?",
                    "type": "?", "legacy": False, "sysinfo": {}}]

    for cam in cameras:
        host = cam["ip"]
        print(f"\n{'=' * 70}\n{cam['alias']}  ({cam['model']} at {host})\n{'=' * 70}")
        open_ports = scan_ports(host)

        if cam["legacy"] or 19443 in open_ports:
            print("\n  Family: legacy Kasa IOT camera - no RTSP, no ONVIF.")
            probe_kc_stream(host, args.username, args.password,
                            snapshot_path=args.snapshot, timeout=args.timeout,
                            dump_stream=args.dump_stream)
            print("\n  Exposure control: there is no ONVIF Imaging service to")
            print("  ask, and the stream endpoint takes no exposure parameters.")
            print("  Port 9999 (Kasa IOT JSON) is open and its camera command")
            print("  set is undocumented - unexplored, and the only remaining")
            print("  place a local exposure/IR-cut control could hide.")
            report_exposure_verdict(manual=False, ir_lock=False)

        if 2020 in open_ports or 554 in open_ports:
            print("\n  Family: Tapo/SmartCam - RTSP and ONVIF available.")
            print(f"  RTSP HD: rtsp://<camera-account>@{host}:554/stream1")
            print(f"  RTSP SD: rtsp://<camera-account>@{host}:554/stream2")
            probe_onvif(host, args.onvif_port, cam_user, cam_pass,
                        snapshot_path=args.snapshot, timeout=args.timeout)

        if args.dump_sysinfo and cam["sysinfo"]:
            print("\n  Full discovery sysinfo:")
            print("   " + json.dumps(cam["sysinfo"], indent=2).replace("\n", "\n   "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="camera IP; omit to sweep the LAN")
    ap.add_argument("--username", default=os.environ.get("KASA_USERNAME"),
                    help="TP-Link cloud account (env KASA_USERNAME)")
    ap.add_argument("--password", default=os.environ.get("KASA_PASSWORD"),
                    help="TP-Link cloud password (env KASA_PASSWORD)")
    ap.add_argument("--camera-user", default=os.environ.get("KASA_CAM_USER"),
                    help="Tapo camera-account user (env KASA_CAM_USER)")
    ap.add_argument("--camera-pass", default=os.environ.get("KASA_CAM_PASS"),
                    help="Tapo camera-account password (env KASA_CAM_PASS)")
    ap.add_argument("--onvif-port", type=int, default=2020)
    ap.add_argument("--snapshot", help="write one still frame to this path")
    ap.add_argument("--dump-sysinfo", action="store_true",
                    help="print the full discovery payload per camera")
    ap.add_argument("--dump-stream", type=int, default=0, metavar="N",
                    help="save the first N raw stream bytes next to --snapshot, "
                         "for identifying the payload format")
    ap.add_argument("--timeout", type=float, default=10.0)
    asyncio.run(main(ap.parse_args()))
