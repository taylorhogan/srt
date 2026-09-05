"""Labelled STILL PICTURES of the roof, open and shut, collected for free.

This is the training and validation set the indoor camera needs before it can
take the roof job off the scope-top webcam. It is pictures, not sound: the
audio library next door (kasa_audio.py) judges whether a move SOUNDED healthy,
which is advisory and gates nothing. This one answers the question the roof
gate actually asks, "is the roof open or shut right now", which today only the
webcam can answer and which it answers worst in bright sun.

Two things make this cheap enough to run on every move forever.

FIRST, the pictures cost no extra camera traffic. The Kasa camera hands over
ONE multipart stream carrying audio and video interleaved, and kasa_audio
already pulls that stream for 50 seconds around every roof move and throws the
video away. So the frames here are decoded from bytes that were already on the
wire. Opening a second connection to grab a still would risk colliding with
the audio capture for no benefit.

SECOND, the labels are free and they are correct. Nothing has to guess what
the roof was doing, and no operator has to write it down: toggle_roof knows
which direction it just commanded, so a frame from before the move shows the
roof in the state it was leaving and a frame from after shows the state it
arrived in. An `open` move is therefore a shut picture followed by an open
one, and a `close` move is the reverse. That is two labelled examples per
move, at roughly twelve moves a fortnight, across whatever weather and light
the season happens to supply.

The labelling assumption, stated so it can be checked rather than trusted: the
capture starts BEFORE the relay fires and runs well past the ~12 s of travel,
so the first frame precedes the move and the last follows it. Every frame is
written with a sidecar recording the direction, the phase, the frame's time
within the capture and the wall clock, so a mislabelled pair can be found and
corrected later instead of quietly poisoning the set. If the capture window
ever moves relative to the relay, that sidecar is what shows it.

Nothing reads this yet. Like the audio library it is a shadow observer: it
must never slow, block or break a roof move, so every failure here is caught
and logged and the roof flow never learns it happened.
"""
import json
import os
import tempfile
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "roof_frames_kasa")

# What the roof is showing BEFORE and AFTER a move of each direction. This is
# the whole labelling rule, kept as data so it reads as the obvious statement
# it is rather than as a pair of conditionals.
PHASE_STATE = {
    "open":  {"before": "shut", "after": "open"},
    "close": {"before": "open", "after": "shut"},
}

# The camera streams 1080p-class H.264 at roughly this rate. Used only to put a
# time on each decoded frame for the sidecar; nothing depends on it being
# exact, and a wrong value shifts a label's timestamp, never its state.
FPS_HINT = 15.0


def _log(msg):
    try:
        from utils import utils
        utils.set_logger().info(msg)
    except Exception:           # noqa: BLE001
        pass


def save_move_frames(video_bytes, direction, root=ROOT, fps_hint=FPS_HINT):
    """Write the first and last frame of a roof-move capture, labelled.

    Returns a list of the files written, empty on any failure. Never raises:
    the caller is the roof flow's own observer thread.
    """
    states = PHASE_STATE.get(direction)
    if not states:
        _log("kasa roof frames: unknown direction %r; not filed" % (direction,))
        return []
    if not video_bytes:
        _log("kasa roof frames: no video in the capture")
        return []

    try:
        import cv2
    except ImportError:
        _log("kasa roof frames: OpenCV unavailable")
        return []

    tmp = None
    try:
        # OpenCV needs a file, not a buffer, for a raw elementary stream.
        fd, tmp = tempfile.mkstemp(suffix=".h264")
        with os.fdopen(fd, "wb") as fh:
            fh.write(video_bytes)

        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            _log("kasa roof frames: could not decode the capture")
            return []
        first = last = None
        index = last_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if first is None:
                first = frame.copy()
            last, last_index = frame, index
            index += 1
        cap.release()
        if first is None:
            _log("kasa roof frames: capture decoded to zero frames")
            return []
        last = last.copy()

        stamp = datetime.now().strftime("%Y%m%dT%H-%M-%S")
        written = []
        for phase, frame, idx in (("before", first, 0), ("after", last, last_index)):
            state = states[phase]
            dest = os.path.join(root, state)
            os.makedirs(dest, exist_ok=True)
            base = os.path.join(dest, "%s_%s_%s" % (stamp, direction, phase))
            cv2.imwrite(base + ".jpg", frame)
            with open(base + ".json", "w") as fh:
                json.dump({"direction": direction, "phase": phase,
                           "roof_state": state,
                           "frame_index": idx,
                           "t_in_capture_s": round(idx / fps_hint, 1),
                           "when": datetime.now().astimezone()
                                           .isoformat(timespec="seconds"),
                           "label_basis": "inferred from the commanded roof "
                                          "direction; see module docstring"},
                          fh, indent=2)
            written.append(base + ".jpg")
        _log("kasa roof frames %s: filed %s before -> %s after (%d frames "
             "decoded)" % (direction, states["before"], states["after"], index))
        return written
    except Exception as e:              # noqa: BLE001 - shadow observer
        _log("kasa roof frames failed: %r" % (e,))
        return []
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def counts(root=ROOT):
    """{state: n} for the pictures filed so far. For the webchat/status."""
    out = {}
    for state in ("open", "shut"):
        d = os.path.join(root, state)
        try:
            out[state] = len([f for f in os.listdir(d) if f.endswith(".jpg")])
        except OSError:
            out[state] = 0
    return out
