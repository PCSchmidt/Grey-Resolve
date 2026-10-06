"""Core data contracts shared across pipeline stages.

All score conventions follow ARCHITECTURE.md:
- face score (cosine) in [-1, 1]
- context score in [0, 1]
- fused score in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

EMBEDDING_DIM = 512


@dataclass(frozen=True)
class ContextMetadata:
    """Lightweight contextual payload attached to a media item."""

    timestamp: datetime | None = None
    geo_cluster: str | None = None
    source_platform: str | None = None
    text_entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class MediaItem:
    """One ingested media unit: an image reference plus optional context."""

    media_id: str
    image_ref: str
    context: ContextMetadata = field(default_factory=ContextMetadata)
    synthetic: bool = False


@dataclass(frozen=True)
class QualityAssessment:
    """Face image quality (FIQA) result. score in [0, 1], higher is better."""

    score: float
    method: str


@dataclass(frozen=True)
class FaceObservation:
    """One detected face within a media item."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    detection_score: float
    quality: QualityAssessment | None = None
    landmarks: np.ndarray | None = None  # optional 5-point landmarks [5, 2] for alignment


@dataclass(frozen=True)
class SearchResult:
    """Ranked candidate returned by search: scores, not identity claims."""

    media_id: str
    face_score: float
    context_score: float | None = None
    fused_score: float | None = None
    rank: int = 0


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    """Return a unit-norm float32 copy of ``vector`` (cosine-ready)."""
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vec / norm
