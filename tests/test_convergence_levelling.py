"""The convergence curve levels every frame to one sky before its subsets are
combined (2026-10-31): IC 410 Ha's three nights sat at 6, 30 and 16 ADU and the
unlevelled curve read a flat 100 % residual."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("astropy")

from stacking import stacker  # noqa: E402


def test_level_to_common_sky_puts_every_frame_at_the_mean_level():
    rng = np.random.default_rng(0)
    frames = [rng.normal(lv, 1.0, (40, 40)).astype(np.float32) for lv in (6.0, 30.0, 16.0)]
    out, common, levels = stacker.level_to_common_sky(frames)
    assert abs(common - np.mean(levels)) < 1e-6
    assert [round(lv) for lv in levels] == [6, 30, 16]
    for f in out:
        assert abs(float(np.median(f)) - common) < 0.3
    # The frames' own structure is untouched: only an offset moved.
    assert np.allclose(out[1] - frames[1], common - levels[1], atol=1e-5)


def test_level_to_common_sky_moves_a_native_crop_with_its_frame_and_keeps_nans():
    small = np.full((20, 20), 10.0, np.float32)
    small[0, :] = np.nan
    crop = np.full((8, 8), 500.0, np.float32)
    other = np.full((20, 20), 30.0, np.float32)
    out, common, levels = stacker.level_to_common_sky([(small, crop), other])
    assert common == 20.0 and levels == [10.0, 30.0]
    s, c = out[0]
    assert np.isnan(s[0, 0]) and float(s[5, 5]) == 20.0
    assert float(c[0, 0]) == 510.0                     # the crop moved by the same +10
    assert float(out[1][0, 0]) == 20.0


def test_subset_residual_is_noise_once_the_nights_are_levelled():
    """Three 'nights' of noisy sky at different levels: the subset-vs-all RMSE
    is dominated by the level mix unlevelled and by noise alone levelled."""
    rng = np.random.default_rng(1)
    frames = [rng.normal(lv, 2.0, (64, 64)).astype(np.float32)
              for lv in [6.0] * 6 + [30.0] * 6 + [16.0] * 6]
    w = np.ones(len(frames)) / len(frames)

    def rmse(fr):
        arr = np.stack(fr)
        golden = stacker._combine_tile(arr, stacker.StackMethod.SIGMA_CLIP_FWHM, w, 3.0)
        idx = rng.choice(len(fr), 6, replace=False)
        sub = stacker._combine_tile(arr[idx], stacker.StackMethod.SIGMA_CLIP_FWHM,
                                    np.ones(6) / 6, 3.0)
        return float(np.sqrt(np.mean((sub - golden) ** 2)))
    levelled, _, _ = stacker.level_to_common_sky(frames)
    assert rmse(frames) > 3 * rmse(levelled)
    assert rmse(levelled) < 1.5                        # ~ sigma * sqrt(1/6 - 1/18)
