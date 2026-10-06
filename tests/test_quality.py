"""Tests for the heuristic FIQA-lite assessor (no model weights required)."""

from __future__ import annotations

import numpy as np

from grey_resolve.detection.base import QualityGate, TrustDecision
from grey_resolve.detection.quality import HeuristicQualityAssessor
from grey_resolve.config import ThresholdProfile
from grey_resolve.types import FaceObservation

PROFILE = ThresholdProfile(
    name="test", quality_min=0.6, ambiguous_low=0.65, ambiguous_high=0.75, alpha=0.7, beta=0.3
)


def _obs(size: int = 200) -> FaceObservation:
    return FaceObservation(bbox=(0.0, 0.0, float(size), float(size)), detection_score=0.99)


def _sharp_image(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _flat_image(size: int = 256) -> np.ndarray:
    return np.full((size, size, 3), 128, dtype=np.uint8)


def test_assessor_scores_are_in_unit_range():
    assessor = HeuristicQualityAssessor()
    for img in (_sharp_image(), _flat_image()):
        result = assessor.assess(img, _obs())
        assert 0.0 <= result.score <= 1.0
        assert result.method == "heuristic-fiqa-lite-v1"


def test_flat_blurry_image_scores_lower_than_noise():
    assessor = HeuristicQualityAssessor()
    flat = assessor.assess(_flat_image(), _obs()).score
    sharp = assessor.assess(_sharp_image(), _obs()).score
    assert flat < sharp


def test_degenerate_bbox_scores_zero():
    assessor = HeuristicQualityAssessor()
    obs = FaceObservation(bbox=(10.0, 10.0, 5.0, 5.0), detection_score=0.5)
    assert assessor.assess(_sharp_image(), obs).score == 0.0


def test_quality_gate_thresholds():
    gate = QualityGate(profile=PROFILE)
    from grey_resolve.types import QualityAssessment

    assert gate.decide(QualityAssessment(score=0.9, method="m")) is TrustDecision.FULL
    assert gate.decide(QualityAssessment(score=0.5, method="m")) is TrustDecision.LOW
    assert gate.decide(QualityAssessment(score=0.1, method="m")) is TrustDecision.REJECT
