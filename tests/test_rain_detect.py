"""The rain ladder's moon compensation.

The 2026-08-25..29 waxing week produced one marginal false "predict" per night
at 1.0-1.4% central-sky motion -- full-moon glare, not weather -- while every
real event in rain_log.jsonl reads 10-99%. The fix raises the predict
threshold with lunar illumination; these tests pin the ladder's behaviour
around that boost without needing a camera, astropy, or the config files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentry.rain_detect import evaluate, moon_onset_boost

CFG = {"sky camera": {}, "location": {}}


def _run(pcts, boost):
    """Feed samples 60 s apart through a fresh state; return alerts raised."""
    st = {"samples": [], "last_predict": 0, "last_detect": 0}
    alerts = []
    for i, p in enumerate(pcts):
        st, alert = evaluate(p, night=True, sun_alt=-30.0, now=1_000_000 + 60 * i,
                             cfg=CFG, state=st, onset_boost=boost)
        if alert:
            alerts.append(alert)
    return alerts


def test_moon_glare_run_no_longer_predicts():
    # Last night's exact failure: sustained ~1-3% under a 99% moon.
    assert _run([1.3, 1.1, 1.06, 1.4], boost=3.0) == []
    # The same run on a dark night is a genuine onset ramp and must still fire.
    assert _run([1.3, 1.1, 1.06], boost=0.0) == ["predict"]


def test_real_ramp_clears_the_moon_raised_threshold():
    # Storm onsets ramp 1% -> 10% in ~30 min; by 5% they clear even the
    # full-moon threshold (1.0 + 3.0) with margin.
    assert _run([4.5, 5.0, 6.0], boost=3.0) == ["predict"]


def test_detect_threshold_is_never_moon_adjusted():
    assert _run([12.0, 15.0, 20.0], boost=3.0) == ["detect"]


def test_boost_fails_toward_sensitivity():
    # No astropy (CI) or an unusable location must yield 0.0 -- a false alert
    # on a moonlit night, never a missed storm on a dark one.
    assert moon_onset_boost(cfg=CFG) == 0.0
