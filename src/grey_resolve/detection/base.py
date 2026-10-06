"""Adapter seams for face detection and FIQA quality gating.

Concrete detectors (SCRFD via InsightFace, MTCNN-class) and quality assessors
(MagFace feature magnitude primary, AdaFace feature norm fallback -- see
docs/BACKBONE_LICENSES.md) implement these protocols. Indexing and API code
depend only on these interfaces, so backbones stay swappable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from grey_resolve.config import ThresholdProfile
from grey_resolve.types import FaceObservation, QualityAssessment


@runtime_checkable
class FaceDetector(Protocol):
    """Detects face bounding boxes in one image (already decoded to a pixel array)."""

    def detect(self, image: "object") -> list[FaceObservation]:
        """Return observations sorted by detection score, descending."""
        ...


@runtime_checkable
class QualityAssessor(Protocol):
    """Computes a face image quality score in [0, 1]; higher is better."""

    method: str

    def assess(self, image: "object", observation: FaceObservation) -> QualityAssessment:
        ...


class TrustDecision(Enum):
    """Outcome of the quality gate for one face observation."""

    FULL = "full"        # index with full weight
    LOW = "low"          # index, flagged low-trust
    REJECT = "reject"    # do not index


@dataclass(frozen=True)
class QualityGate:
    """Applies a ThresholdProfile's quality_min to FIQA results.

    Port note: replaces IronClad's raw MTCNN-confidence gate (0.99 in
    design_config.json) with an explicit, profile-driven FIQA threshold.
    """

    profile: ThresholdProfile

    def decide(self, assessment: QualityAssessment) -> TrustDecision:
        if assessment.score >= self.profile.quality_min:
            return TrustDecision.FULL
        if assessment.score >= self.profile.quality_min * 0.5:
            return TrustDecision.LOW
        return TrustDecision.REJECT
