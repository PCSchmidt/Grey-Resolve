"""Adapter seam for face embedding extraction.

Primary backbone: InsightFace buffalo_l (w600k_r50 ArcFace, 512-d), weights
downloaded at runtime and never committed (see docs/BACKBONE_LICENSES.md).
Fallback: AdaFace IR-50 WebFace4M (note its BGR input convention).

Preprocessing contract (differs from IronClad's 160px / ImageNet-norm):
- align/crop to the backbone's expected geometry (112x112 for ArcFace)
- normalize per the backbone's documented mean/std
- return a unit-norm float32 vector of length EMBEDDING_DIM
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from grey_resolve.types import EMBEDDING_DIM, FaceObservation, normalize_embedding

__all__ = ["EMBEDDING_DIM", "EmbeddingExtractor"]


@runtime_checkable
class EmbeddingExtractor(Protocol):
    """Extracts unit-norm face embeddings. Implementations own their preprocessing."""

    @property
    def dimension(self) -> int: ...

    @property
    def model_id(self) -> str:
        """Stable identifier (e.g. 'insightface/buffalo_l/w600k_r50') for cache keys."""
        ...

    def extract(self, image: "object", observation: FaceObservation) -> np.ndarray:
        """Return a unit-norm float32 vector of length ``self.dimension``."""
        ...
