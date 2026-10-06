"""Face image quality (FIQA) adapters.

Phase 1 quality gate decision (recorded 2026-10-06 in PLAN.md): ship a
dependency-free heuristic assessor now so the pipeline runs end-to-end, and
add the MagFace feature-magnitude assessor when its weights are fetched
(Apache-2.0 code, silent weight terms -- see docs/BACKBONE_LICENSES.md).
AdaFace feature-norm remains the zero-cost fallback once the AdaFace path
exists.

The heuristic is deliberately simple and labeled FIQA-lite: it scores sharpness
(Laplacian variance), exposure (mean brightness), and usable face size. It is a
stand-in for measured FIQA, not a claim of perceptual quality.
"""

from __future__ import annotations

import numpy as np

from grey_resolve.types import FaceObservation, QualityAssessment


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float64)
    return image.astype(np.float64).mean(axis=2)


class HeuristicQualityAssessor:
    """FIQA-lite: sharpness + exposure + face-size heuristic in [0, 1]."""

    method = "heuristic-fiqa-lite-v1"

    def __init__(self, min_face_px: int = 40, blur_ref: float = 500.0):
        self._min_face_px = min_face_px
        self._blur_ref = blur_ref

    def assess(self, image: "np.ndarray", observation: FaceObservation) -> QualityAssessment:
        x1, y1, x2, y2 = (int(round(v)) for v in observation.bbox)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        if x2 <= x1 or y2 <= y1:
            return QualityAssessment(score=0.0, method=self.method)

        crop = _to_gray(image[y1:y2, x1:x2])
        lap = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        padded = np.pad(crop, 1, mode="edge")
        conv = sum(
            padded[i:i + crop.shape[0], j:j + crop.shape[1]] * lap[i, j]
            for i in range(3) for j in range(3)
        )
        sharpness = float(conv.var())
        sharpness_score = min(sharpness / self._blur_ref, 1.0)

        mean_brightness = float(crop.mean()) / 255.0
        exposure_score = float(np.clip(1.0 - abs(mean_brightness - 0.5) * 2.0, 0.0, 1.0))

        size_px = min(x2 - x1, y2 - y1)
        size_score = float(np.clip(size_px / (self._min_face_px * 2.0), 0.0, 1.0))

        score = 0.5 * sharpness_score + 0.25 * exposure_score + 0.25 * size_score
        return QualityAssessment(score=float(np.clip(score, 0.0, 1.0)), method=self.method)
