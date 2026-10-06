"""Pure-function image degradation operators for benchmark sweeps.

Every operator takes a numpy ``uint8`` image of shape ``(H, W, 3)`` and returns a
new ``uint8`` image of the same shape. Inputs are never mutated. All operators are
deterministic (no randomness), so benchmarks are reproducible without seeds.

Interpolation is implemented with ``scipy.ndimage`` (bicubic spline sampling) --
no OpenCV dependency.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

__all__ = ["gaussian_blur", "brightness", "downsample", "off_angle", "DEGRADATIONS"]


def _check_image(image: np.ndarray) -> np.ndarray:
    """Validate the image contract; return the input unchanged (not a copy)."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.dtype != np.uint8:
        raise ValueError("image dtype must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be non-empty")
    return image


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Round and clip a float array to a new uint8 image."""
    return np.clip(np.rint(arr), 0.0, 255.0).astype(np.uint8)


def _resize(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bicubic-resize to ``(out_h, out_w, 3)`` float64 (no clipping yet)."""
    h, w = image.shape[:2]
    # Map output pixel centers into source continuous coordinates.
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * (w / out_w) - 0.5
    ys = np.clip(ys, 0.0, h - 1.0)
    xs = np.clip(xs, 0.0, w - 1.0)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    out = np.empty((out_h, out_w, 3), dtype=np.float64)
    for c in range(3):
        out[..., c] = ndimage.map_coordinates(
            image[..., c].astype(np.float64),
            [grid_y, grid_x],
            order=3,
            mode="nearest",
        )
    return out


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur ``image`` with an isotropic Gaussian of standard deviation ``sigma``.

    Args:
        image: uint8 array of shape (H, W, 3).
        sigma: Gaussian standard deviation in pixels, ``sigma >= 0``.
            ``sigma == 0`` returns an unchanged copy.

    Returns:
        New uint8 array of the same shape; the input is not mutated.
    """
    _check_image(image)
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be a finite float >= 0")
    if sigma == 0.0:
        return image.copy()
    blurred = ndimage.gaussian_filter(
        image.astype(np.float64), sigma=(sigma, sigma, 0.0)
    )
    return _to_uint8(blurred)


def brightness(image: np.ndarray, delta: float) -> np.ndarray:
    """Shift image brightness by ``delta`` as a fraction of the full range.

    Each pixel becomes ``clip(pixel + delta * 255, 0, 255)``.

    Args:
        image: uint8 array of shape (H, W, 3).
        delta: brightness offset in ``[-1, 1]``; positive brightens.

    Returns:
        New uint8 array of the same shape; the input is not mutated.
    """
    _check_image(image)
    delta = float(delta)
    if not math.isfinite(delta) or not -1.0 <= delta <= 1.0:
        raise ValueError("delta must be a finite float in [-1, 1]")
    if delta == 0.0:
        return image.copy()
    shifted = image.astype(np.float64) + delta * 255.0
    return _to_uint8(shifted)


def downsample(image: np.ndarray, scale: float) -> np.ndarray:
    """Simulate resolution loss: down-sample by ``scale``, then up-sample back.

    Both resampling steps use bicubic interpolation. The output has the same
    shape as the input, so degraded images stay comparable pixel-for-pixel.

    Args:
        image: uint8 array of shape (H, W, 3).
        scale: resolution retention factor in ``(0, 1]``. ``scale == 1``
            returns an unchanged copy.

    Returns:
        New uint8 array of the same shape; the input is not mutated.
    """
    _check_image(image)
    scale = float(scale)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("scale must be a finite float in (0, 1]")
    if scale == 1.0:
        return image.copy()
    h, w = image.shape[:2]
    small_h = max(1, int(round(h * scale)))
    small_w = max(1, int(round(w * scale)))
    down = _resize(image, small_h, small_w)
    up = _resize(_to_uint8(down), h, w)
    return _to_uint8(up)


def off_angle(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Simulate an oblique (off-angle) view of a fronto-parallel plane.

    Applies a projective (homography) warp that foreshortens one side of the
    image, as when the camera yaws away from the subject. The shrink factor is
    ``cos(angle_deg)``; positive angles shrink the right edge, negative angles
    the left edge. Pixels mapped outside the source are filled with black.

    Args:
        image: uint8 array of shape (H, W, 3).
        angle_deg: view angle in degrees, ``|angle_deg| < 90``.
            ``angle_deg == 0`` returns an unchanged copy.

    Returns:
        New uint8 array of the same shape; the input is not mutated.
    """
    _check_image(image)
    angle = float(angle_deg)
    if not math.isfinite(angle) or abs(angle) >= 90.0:
        raise ValueError("angle_deg must be a finite float with |angle_deg| < 90")
    if angle == 0.0:
        return image.copy()

    h, w = image.shape[:2]
    # Continuous-coordinate corners of the source quad.
    src = np.array(
        [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]
    )
    shrink = math.cos(math.radians(abs(angle)))
    inset = h * (1.0 - shrink) / 2.0
    if angle > 0.0:  # right edge foreshortens
        dst = np.array(
            [[0.0, 0.0], [float(w), inset], [float(w), float(h) - inset], [0.0, float(h)]]
        )
    else:  # left edge foreshortens
        dst = np.array(
            [[0.0, inset], [float(w), 0.0], [float(w), float(h)], [0.0, float(h) - inset]]
        )

    # Homography H mapping source -> destination, then invert for inverse warping.
    a_rows = []
    b_rows = []
    for (x, y), (u, v) in zip(src, dst):
        a_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        a_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        b_rows.append(u)
        b_rows.append(v)
    h8 = np.linalg.solve(np.array(a_rows), np.array(b_rows))
    hom = np.append(h8, 1.0).reshape(3, 3)
    hom_inv = np.linalg.inv(hom)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    denom = hom_inv[2, 0] * xx + hom_inv[2, 1] * yy + hom_inv[2, 2]
    map_x = (hom_inv[0, 0] * xx + hom_inv[0, 1] * yy + hom_inv[0, 2]) / denom
    map_y = (hom_inv[1, 0] * xx + hom_inv[1, 1] * yy + hom_inv[1, 2]) / denom

    valid = (map_x >= 0.0) & (map_x <= w - 1.0) & (map_y >= 0.0) & (map_y <= h - 1.0)
    out = np.empty((h, w, 3), dtype=np.float64)
    for c in range(3):
        out[..., c] = ndimage.map_coordinates(
            image[..., c].astype(np.float64),
            [map_y, map_x],
            order=3,
            mode="nearest",
        )
    out[~valid] = 0.0
    return _to_uint8(out)


DEGRADATIONS: dict[str, object] = {
    "gaussian_blur": gaussian_blur,
    "brightness": brightness,
    "downsample": downsample,
    "off_angle": off_angle,
}
"""Registry name -> operator, so benchmarks can sweep degradations by name.

Operators keep their own signatures (``sigma`` / ``delta`` / ``scale`` /
``angle_deg``); callers supply the strength parameter when sweeping.
"""
