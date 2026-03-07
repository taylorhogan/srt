import os
import socket
import tempfile  # used for gettempdir()
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

import soco
from gtts import gTTS


def _get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _serve_file(path: str, port: int, ready_event: threading.Event, stop_event: threading.Event):
    directory = os.path.dirname(path)
    filename = os.path.basename(path)

    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mp3": "audio/mpeg"}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format, *args):
            print(f"HTTP: {format % args}")

    server = HTTPServer(("", port), Handler)
    server.timeout = 1
    ready_event.set()
    while not stop_event.is_set():
        server.handle_request()
    server.server_close()


def sonos_say(text: str, speaker_name: str, volume: int = 50):
    devices = list(soco.discover() or [])
    if not devices:
        raise RuntimeError("No Sonos devices found on the network")

    speaker = next((d for d in devices if d.player_name.lower() == speaker_name.lower()), None)
    if speaker is None:
        available = [d.player_name for d in devices]
        raise RuntimeError(f"Speaker '{speaker_name}' not found. Available: {available}")

    # play_uri must be called on the group coordinator
    speaker = speaker.group.coordinator

    tmp_path = os.path.join(tempfile.gettempdir(), "sonos_tts.mp3")

    try:
        gTTS(text=text, lang="en").save(tmp_path)
        file_size = os.path.getsize(tmp_path)
        print(f"DEBUG: mp3 path={tmp_path}, size={file_size} bytes")
        if file_size == 0:
            raise RuntimeError("gTTS generated an empty file — check internet access")

        port = 54321
        local_ip = _get_local_ip()
        url = f"http://{local_ip}:{port}/sonos_tts.mp3"

        ready = threading.Event()
        stop = threading.Event()
        server_thread = threading.Thread(
            target=_serve_file, args=(tmp_path, port, ready, stop), daemon=True
        )
        server_thread.start()
        ready.wait(timeout=5)

        import urllib.request
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/sonos_tts.mp3", timeout=3)
            print(f"DEBUG: HTTP self-test PASSED — server is accepting connections on port {port}")
        except Exception as e:
            print(f"DEBUG: HTTP self-test FAILED — {e}")

        print(f"DEBUG: coordinator={speaker.player_name} ip={speaker.ip_address}, local_ip={local_ip}, url={url}")
        print(f"DEBUG: volume={speaker.volume}, mute={speaker.mute}")
        previous_volume = speaker.volume
        speaker.volume = volume
        speaker.play_uri(url)

        # Wait for playback to finish
        for _ in range(60):
            time.sleep(1)
            info = speaker.get_current_transport_info()
            state = info["current_transport_state"]
            print(f"DEBUG: transport state={state}")
            if state not in ("PLAYING", "TRANSITIONING"):
                break

        speaker.volume = previous_volume
        stop.set()

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    sonos_say("Hello from the observatory. The telescope is ready for tonight.", "Office", volume=20)
