"""Tests for image degradation operators (shape/dtype contract, purity, effect)."""

from __future__ import annotations

import numpy as np
import pytest

from grey_resolve.degradation import (
    DEGRADATIONS,
    brightness,
    downsample,
    gaussian_blur,
    off_angle,
)


def _image(seed: int = 1234, h: int = 32, w: int = 48) -> np.ndarray:
    """Deterministic non-constant uint8 test image with shape (h, w, 3)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


OPERATORS_AND_ARGS = [
    (gaussian_blur, 1.5),
    (brightness, 0.2),
    (downsample, 0.5),
    (off_angle, 30.0),
]


@pytest.mark.parametrize("op,arg", OPERATORS_AND_ARGS)
def test_preserves_shape_and_dtype(op, arg):
    img = _image()
    out = op(img, arg)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == img.shape


@pytest.mark.parametrize("op,arg", OPERATORS_AND_ARGS)
def test_does_not_mutate_input(op, arg):
    img = _image()
    original = img.copy()
    op(img, arg)
    assert np.array_equal(img, original)


@pytest.mark.parametrize("op,arg", OPERATORS_AND_ARGS)
def test_deterministic(op, arg):
    img = _image()
    assert np.array_equal(op(img, arg), op(img, arg))


def test_gaussian_blur_changes_image():
    img = _image()
    out = gaussian_blur(img, 2.0)
    assert float(np.mean(np.abs(out.astype(np.float64) - img))) > 0.0


def test_gaussian_blur_preserves_constant_image():
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    assert np.array_equal(gaussian_blur(img, 1.0), img)


def test_gaussian_blur_sigma_zero_is_identity():
    img = _image()
    assert np.array_equal(gaussian_blur(img, 0.0), img)


def test_gaussian_blur_rejects_bad_sigma():
    img = _image()
    with pytest.raises(ValueError):
        gaussian_blur(img, -0.5)
    with pytest.raises(ValueError):
        gaussian_blur(img, float("nan"))


def test_brightness_moves_mean_correctly():
    img = _image()
    mean0 = float(img.mean())
    up = brightness(img, 0.2)
    down = brightness(img, -0.2)
    assert float(up.mean()) > mean0
    assert float(down.mean()) < mean0
    # exact per-pixel semantics
    expected = np.clip(img.astype(np.int32) + round(0.2 * 255), 0, 255).astype(np.uint8)
    assert np.array_equal(up, expected)


def test_brightness_clips_at_bounds():
    img = _image()
    assert np.array_equal(brightness(img, 1.0), np.full_like(img, 255))
    assert np.array_equal(brightness(img, -1.0), np.zeros_like(img))


def test_brightness_rejects_out_of_range_delta():
    img = _image()
    with pytest.raises(ValueError):
        brightness(img, 1.5)
    with pytest.raises(ValueError):
        brightness(img, -1.01)


def test_downsample_changes_image():
    img = _image()
    out = downsample(img, 0.5)
    assert float(np.mean(np.abs(out.astype(np.float64) - img))) > 0.0


def test_downsample_scale_one_is_identity():
    img = _image()
    assert np.array_equal(downsample(img, 1.0), img)


def test_downsample_rejects_bad_scale():
    img = _image()
    for bad in (0.0, -0.5, 1.5, float("inf")):
        with pytest.raises(ValueError):
            downsample(img, bad)


def test_off_angle_zero_is_identity():
    img = _image()
    assert np.array_equal(off_angle(img, 0.0), img)


def test_off_angle_changes_image_and_fills_black():
    img = np.full((32, 32, 3), 200, dtype=np.uint8)
    out = off_angle(img, 45.0)
    assert not np.array_equal(out, img)
    # oblique warp maps some pixels outside the source -> black fill
    assert np.any(out == 0)


def test_off_angle_rejects_bad_angle():
    img = _image()
    for bad in (90.0, -90.0, 120.0, float("nan")):
        with pytest.raises(ValueError):
            off_angle(img, bad)


def test_off_angle_sign_controls_shrunk_side():
    img = np.full((32, 32, 3), 200, dtype=np.uint8)
    right = off_angle(img, 45.0)
    left = off_angle(img, -45.0)
    assert not np.array_equal(right, left)
    mid = img.shape[1] // 2

    def black_by_half(out: np.ndarray) -> tuple[int, int]:
        black = np.all(out == 0, axis=2)  # fill pixels only (source is constant 200)
        return int(black[:, :mid].sum()), int(black[:, mid:].sum())

    r_left, r_right = black_by_half(right)
    l_left, l_right = black_by_half(left)
    # positive angle foreshortens the right edge, negative angle the left
    assert r_right > 0 and r_right > r_left
    assert l_left > 0 and l_left > l_right


@pytest.mark.parametrize("op,arg", OPERATORS_AND_ARGS)
def test_invalid_images_rejected(op, arg):
    with pytest.raises(ValueError):
        op(_image().astype(np.float32), arg)  # wrong dtype
    with pytest.raises(ValueError):
        op(np.zeros((16, 16), dtype=np.uint8), arg)  # missing channels
    with pytest.raises(ValueError):
        op(np.zeros((16, 16, 4), dtype=np.uint8), arg)  # wrong channel count
    with pytest.raises(TypeError):
        op([[0]], arg)  # not an ndarray


def test_registry_contents():
    assert set(DEGRADATIONS) == {"gaussian_blur", "brightness", "downsample", "off_angle"}
    img = _image()
    assert np.array_equal(DEGRADATIONS["gaussian_blur"](img, 1.0), gaussian_blur(img, 1.0))
    assert np.array_equal(DEGRADATIONS["brightness"](img, 0.1), brightness(img, 0.1))
    assert np.array_equal(DEGRADATIONS["downsample"](img, 0.5), downsample(img, 0.5))
    assert np.array_equal(DEGRADATIONS["off_angle"](img, 10.0), off_angle(img, 10.0))
