#!/usr/bin/env python3
"""Live focus meter for a ZWO ASI camera. Prints a score per frame while you turn.

The score is the high-frequency power in the frame ABOVE what photon and read
noise alone would produce. A defocused frame has none: its only structure is
noise, so subtracting the noise variance drives the score to zero. Turning the
ring toward focus makes real edges appear and the score climbs.

Deliberately not a plain Laplacian variance, which is the usual recipe: on a
badly defocused frame that statistic is dominated by pixel noise and reads a few
hundred whether the lens is close to focus or nowhere near it. It cannot tell
you which way to turn, which is the only thing this needs to do.

Exposure and gain are held fixed so the score is comparable across the sweep,
and the score is normalised by mean level so a brightness change from vignetting
or a cloud does not masquerade as focus. The score is NOT comparable between
sweeps taken at different exposure or gain -- it reads several times higher on a
dim frame -- so a sweep is only ever read against itself.

Every frame is also shown with the score burnt into it, so the ring can be
turned while watching a picture rather than a console. Three ways out, because
the obvious one does not work: most Windows image viewers open a file once and
never re-stat it, so overwriting the JPEG in place leaves them showing the first
frame forever. So the default is a live OpenCV window, which really does update;
the JPEG is still written for anything that polls, and an HTML page is written
beside it that reloads the JPEG in a browser every 250 ms.

Usage:
    python scripts/asi_focus.py [seconds] [--exposure US] [--gain G]
                                [--camera SUBSTR] [--out PATH] [--no-preview]
                                [--no-window]

Press q in the window to stop early.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

SDK = r"C:\Program Files\N.I.N.A. - Nighttime Imaging 'N' Astronomy\External\x64\ASI\ASICamera2.dll"
DEFAULT_MATCH = "120"   # --camera picks which; only that one is opened
DEFAULT_OUT = Path(os.path.expanduser("~")) / "Desktop" / "asi_focus.jpg"
WINDOW = "ASI focus  (q to stop)"
FULL = 65535.0          # RAW16 full scale
CLIP_WARN = 1.0         # percent of clipped pixels the score survives


def clipped(a):
    """Percent of the frame pinned at full scale.

    Worth its own readout because clipping does not just lose highlights, it
    inverts the meter: a saturated region is perfectly flat, so it contributes
    no high-frequency power and DRAGS THE SCORE DOWN. Measured on a lit cabinet
    at gain 100, 20 ms put the median pixel at full scale with 56% of the frame
    clipped, and the score then tracked frame brightness at r = -0.95 -- every
    apparent focus peak was really a moment when less of the frame happened to
    be saturated. The failure is invisible in the picture, because the preview
    is percentile-stretched and looks correctly exposed either way.
    """
    return 100.0 * float(np.mean(a >= 0.98 * FULL))


def sharpness(a):
    """High-frequency power above the noise floor, normalised by mean level."""
    from scipy import ndimage
    a = a.astype(np.float32)
    hp = a - ndimage.gaussian_filter(a, 4.0)
    # Noise estimated from the pixel-to-pixel difference, which is pure noise on
    # a defocused frame and stays a fair estimate once detail appears.
    noise = 1.4826 * np.median(np.abs(a - ndimage.median_filter(a, 3)))
    excess = max(hp.var() - noise * noise, 0.0)
    return float(np.sqrt(excess) / max(np.mean(a), 1.0) * 1000.0), float(np.mean(a))


def _to_rgb(raw, prop):
    """Debayer if this is a colour camera, else pass the mono frame through.

    Without this the raw Bayer mosaic of a colour camera displays as a fine
    checkerboard, which reads as detail and would be mistaken for focus.
    """
    if not prop.get("IsColorCam"):
        return raw
    import cv2
    pattern = {0: cv2.COLOR_BAYER_RG2RGB, 1: cv2.COLOR_BAYER_BG2RGB,
               2: cv2.COLOR_BAYER_GR2RGB, 3: cv2.COLOR_BAYER_GB2RGB}
    return cv2.cvtColor(raw, pattern.get(prop.get("BayerPattern", 0),
                                         cv2.COLOR_BAYER_RG2RGB))


def _font(size):
    from PIL import ImageFont
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_preview(raw, prop, score, best, elapsed, clip=0.0):
    """Return the frame as an image with the score burnt in."""
    from PIL import Image, ImageDraw

    img = _to_rgb(raw, prop).astype(np.float32)
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.9)
    b = np.clip((img - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)
    frame = Image.fromarray(b).convert("RGB")
    w, h = frame.size

    pad = max(8, w // 100)
    big, small = _font(max(28, w // 22)), _font(max(15, w // 44))
    # Near the best seen so far is the only thing the eye needs off this image;
    # the absolute number means nothing on its own.
    frac = score / best if best > 0 else 0.0
    peak = frac > 0.985
    box_h = pad * 2 + big.size + 14
    # The readout gets its own strip above the picture rather than being painted
    # over it. Focus is judged on the corners and edges as much as the middle,
    # and a banner across the top would cover exactly that.
    im = Image.new("RGB", (w, h + box_h), (0, 0, 0))
    im.paste(frame, (0, box_h))

    d = ImageDraw.Draw(im)
    d.text((pad, pad), "%.1f" % score, font=big,
           fill=(120, 255, 120) if peak else (255, 255, 255))
    d.text((pad + big.size * 4, pad + 6),
           "best %.1f   %.0f%%   t=%.0fs%s"
           % (best, 100 * frac, elapsed, "   PEAK" if peak else ""),
           font=small, fill=(120, 255, 120) if peak else (190, 190, 190))
    if clip > CLIP_WARN:
        d.text((pad + big.size * 4, pad + 6 + small.size + 4),
               "CLIPPED %.0f%% - shorten the exposure, score is unreliable"
               % clip, font=small, fill=(255, 110, 110))
    # Bar scaled to the best seen so far: turning past the peak makes it retreat,
    # which is the signal for "you have gone too far".
    bar_y = box_h - pad - 14
    d.rectangle([pad, bar_y, w - pad, bar_y + 14], outline=(90, 90, 90))
    d.rectangle([pad, bar_y, pad + int((w - 2 * pad) * min(frac, 1.0)),
                 bar_y + 14], fill=(120, 255, 120) if peak else (90, 160, 230))
    return im


def save_preview(im, path):
    """Write the annotated frame to path. Returns False if it could not.

    Written to a temporary file and renamed, so a viewer polling the path never
    catches a half-written JPEG. On Windows the rename can lose to a viewer that
    holds the file open; that is a dropped frame, not an error worth stopping a
    sweep for, so it is swallowed and reported by the return value.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.jpg")
    try:
        im.save(tmp, quality=85)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def write_viewer_page(path):
    """Write an HTML page beside the JPEG that reloads it on a timer.

    The cache-busting query is the whole point: without it the browser serves
    the first frame from cache for the rest of the sweep, which is the same
    failure a desktop image viewer has.
    """
    page = Path(path).with_suffix(".html")
    page.write_text(
        "<!doctype html><title>ASI focus</title>"
        "<body style='margin:0;background:#000;text-align:center'>"
        "<img id=f style='max-width:100%%;max-height:100vh'>"
        "<script>setInterval(function(){"
        "document.getElementById('f').src='%s?'+Date.now();},250);</script>"
        % Path(path).name, encoding="utf-8")
    return page


def main():
    import zwoasi as asi
    secs = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 60.0
    exp = int(sys.argv[sys.argv.index('--exposure') + 1]) if '--exposure' in sys.argv else 20000
    gain = int(sys.argv[sys.argv.index('--gain') + 1]) if '--gain' in sys.argv else 100
    match = sys.argv[sys.argv.index('--camera') + 1] if '--camera' in sys.argv else DEFAULT_MATCH
    out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else DEFAULT_OUT
    if '--no-preview' in sys.argv:
        out = None
    window = '--no-window' not in sys.argv

    asi.init(SDK)
    names = asi.list_cameras()
    idx = [i for i, n in enumerate(names) if match in n]
    if not idx:
        print("no camera matching %r; found %s" % (match, names))
        return 2
    cam = asi.Camera(idx[0])
    try:
        cam.set_image_type(asi.ASI_IMG_RAW16)
        p = cam.get_camera_property()
        cam.set_roi(0, 0, p['MaxWidth'], p['MaxHeight'])
        cam.set_control_value(asi.ASI_EXPOSURE, exp)
        cam.set_control_value(asi.ASI_GAIN, gain)
        print("%s  %.3fs  gain %d   turn the ring slowly and steadily"
              % (names[idx[0]], exp / 1e6, gain), flush=True)
        if out is not None:
            print("preview: %s" % out, flush=True)
            print("browser: %s   (updates on its own; a desktop image viewer "
                  "will not)" % write_viewer_page(out), flush=True)
        if window:
            import cv2
            # WINDOW_NORMAL so the window can be dragged bigger than the screen
            # scaling allows: the frame behind it is full resolution, and the
            # corners are where defocus shows first.
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1100, 800)
            print("live window open; press q to stop", flush=True)
        print("%8s %9s %7s %6s  %s"
              % ("t(s)", "sharp", "level", "clip%", "score"), flush=True)
        t0 = time.time()
        best = (-1, 0.0)
        skipped = 0
        warned = False
        while time.time() - t0 < secs:
            a = cam.capture()
            s, lvl = sharpness(a)
            clip = clipped(a)
            t = time.time() - t0
            if s > best[0]:
                best = (s, t)
            # Scaled to the best seen so far, so the bar stays informative
            # whatever the absolute numbers turn out to be on a given lens.
            bar = "#" * int(60 * s / max(best[0], 1e-6))
            print("%8.1f %9.2f %7.0f %6.2f  %s" % (t, s, lvl, clip, bar),
                  flush=True)
            # Said once, loudly, and early: a sweep run over a clipped frame is
            # wasted, and there is nothing in the picture or the score to say so.
            if clip > CLIP_WARN and not warned:
                warned = True
                print("\n  *** %.0f%% of the frame is CLIPPED at %.1f ms. The "
                      "score is unreliable -- it\n      falls when more of the "
                      "frame saturates, so it will track brightness, not\n"
                      "      focus. Restart with --exposure %d and re-check.\n"
                      % (clip, exp / 1000.0, max(int(exp / 8), 50)), flush=True)
            if out is not None or window:
                im = render_preview(a, p, s, best[0], t, clip)
                if out is not None and not save_preview(im, out):
                    skipped += 1
                if window:
                    cv2.imshow(WINDOW, np.asarray(im)[:, :, ::-1])
                    # Also pumps the window's event loop; without it the window
                    # never paints and Windows reports it as not responding.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        if window:
            cv2.destroyWindow(WINDOW)
        print("\nbest %.2f at t=%.1fs" % best, flush=True)
        if skipped:
            print("%d preview frame(s) could not be written (viewer holding "
                  "the file open)" % skipped, flush=True)
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
