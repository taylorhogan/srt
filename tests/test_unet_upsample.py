"""The UNet's decoder choice, and that checkpoints of either kind load.

The transposed-convolution decoder leaves a 4 px / 2 px checkerboard on
smooth narrowband emission (lab manual step 39); the bilinear decoder cannot.
Old checkpoints have no stamp, so the loader reads the decoder off the state
dict's keys -- a wrong guess fails strict loading, which is how it used to
fail; `UNet.matching` must never guess.
"""
import pytest

torch = pytest.importorskip("torch")

from nn.noise2noise_model import UNet  # noqa: E402


def _state(upsample):
    return UNet(features=(4, 8), upsample=upsample).state_dict()


def test_default_is_the_historical_transpose_decoder():
    m = UNet(features=(4, 8))
    assert m.upsample == "transpose"
    assert isinstance(m.upconvs[0], torch.nn.ConvTranspose2d)


def test_bilinear_decoder_has_no_transposed_convolution():
    m = UNet(features=(4, 8), upsample="bilinear")
    assert not any(isinstance(mod, torch.nn.ConvTranspose2d) for mod in m.modules())
    assert any(isinstance(mod, torch.nn.Upsample) for mod in m.modules())


@pytest.mark.parametrize("upsample", UNet.UPSAMPLES)
def test_forward_keeps_the_frame_shape_including_odd_sizes(upsample):
    m = UNet(features=(4, 8), upsample=upsample).eval()
    x = torch.randn(1, 1, 37, 46)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape


def test_bilinear_starts_from_identity_like_transpose():
    # The zero-initialised head makes training start from exact pass-through;
    # the decoder change must not break that.
    m = UNet(features=(4, 8), upsample="bilinear").eval()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(m(x), x)


@pytest.mark.parametrize("upsample", UNet.UPSAMPLES)
def test_upsample_is_read_off_the_state_dict(upsample):
    assert UNet.upsample_of(_state(upsample)) == upsample


@pytest.mark.parametrize("upsample", UNet.UPSAMPLES)
def test_matching_builds_and_loads_either_kind(upsample):
    sd = _state(upsample)
    m = UNet.matching(sd, features=(4, 8))
    assert m.upsample == upsample
    for k, v in m.state_dict().items():
        assert torch.equal(v, sd[k])


def test_wrong_decoder_still_fails_loudly_without_matching():
    with pytest.raises(RuntimeError):
        UNet(features=(4, 8), upsample="bilinear").load_state_dict(_state("transpose"))


def test_unknown_upsample_is_refused():
    with pytest.raises(ValueError):
        UNet(features=(4, 8), upsample="nearest")
