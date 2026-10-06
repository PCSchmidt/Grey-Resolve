"""Linear fusion scorer: P = alpha * S_face + beta * S_context.

Implements the Phase 2 fusion formula from ARCHITECTURE.md and
grey_resolve.fusion.base: the fused score is the threshold profile's weighted
sum of the face cosine and the context compatibility, clipped to [0, 1]. In
the profile's ambiguous band the context decides near-ties; use
:meth:`LinearFusionScorer.in_ambiguous_band` to detect that regime.
"""

from __future__ import annotations

import math
from datetime import datetime

from grey_resolve.config import ThresholdProfile
from grey_resolve.types import ContextMetadata

__all__ = [
    "ENTITY_WEIGHT",
    "GEO_WEIGHT",
    "TIME_HALF_LIFE_DAYS",
    "TIME_WEIGHT",
    "LinearFusionScorer",
]

# Context sub-weights (sum to 1.0). See context_score for the combination rule.
GEO_WEIGHT = 0.4
TIME_WEIGHT = 0.3
ENTITY_WEIGHT = 0.3
TIME_HALF_LIFE_DAYS = 30.0


class LinearFusionScorer:
    """``P = alpha * S_face + beta * S_context`` with profile weights.

    ``context_score(query, candidate)`` is in [0, 1] and combines three
    compatibility signals with fixed weights (summing to 1):

    - **geo** (weight 0.4): 1.0 when both payloads carry the same
      ``geo_cluster``, 0.0 when they differ.
    - **temporal** (weight 0.3): exponential decay of the timestamp gap,
      ``0.5 ** (days / 30)`` (1.0 for identical timestamps).
    - **entities** (weight 0.3): Jaccard overlap of the ``text_entities`` sets
      (|A n B| / |A u B|).

    Signals missing from either payload are excluded and the remaining weights
    are renormalized; when no signal is shared the context score is 0.0. The
    score is symmetric in (query, candidate).

    ``fuse`` evaluates ``alpha * face_score + beta * context_score`` and clips
    the result to [0, 1]; it is non-decreasing in ``face_score``.
    """

    def __init__(self, profile: ThresholdProfile) -> None:
        """Bind the scorer to one operating profile.

        Raises ValueError when alpha/beta are negative or both zero, or when
        the ambiguous band is not a valid 0 <= low <= high <= 1 interval.
        """
        alpha = float(profile.alpha)
        beta = float(profile.beta)
        if not (math.isfinite(alpha) and math.isfinite(beta)):
            raise ValueError("alpha and beta must be finite")
        if alpha < 0.0 or beta < 0.0 or (alpha == 0.0 and beta == 0.0):
            raise ValueError("alpha and beta must be >= 0 and not both zero")
        low = float(profile.ambiguous_low)
        high = float(profile.ambiguous_high)
        if not (math.isfinite(low) and math.isfinite(high)) or not 0.0 <= low <= high <= 1.0:
            raise ValueError("ambiguous band must satisfy 0 <= low <= high <= 1")
        self._profile = profile

    @property
    def profile(self) -> ThresholdProfile:
        """The bound operating profile."""
        return self._profile

    @property
    def alpha(self) -> float:
        """Face weight from the profile."""
        return float(self._profile.alpha)

    @property
    def beta(self) -> float:
        """Context weight from the profile."""
        return float(self._profile.beta)

    def context_score(self, query: ContextMetadata, candidate: ContextMetadata) -> float:
        """Compatibility of two context payloads in [0, 1].

        Weighted combination of the geo / temporal / entity signals documented
        on the class, renormalized over the signals both payloads carry.
        """
        parts: list[tuple[float, float]] = []
        if query.geo_cluster is not None and candidate.geo_cluster is not None:
            geo = 1.0 if query.geo_cluster == candidate.geo_cluster else 0.0
            parts.append((GEO_WEIGHT, geo))
        if query.timestamp is not None and candidate.timestamp is not None:
            dt = _abs_days(query.timestamp, candidate.timestamp)
            parts.append((TIME_WEIGHT, 0.5 ** (dt / TIME_HALF_LIFE_DAYS)))
        if query.text_entities and candidate.text_entities:
            a, b = set(query.text_entities), set(candidate.text_entities)
            union = a | b
            parts.append((ENTITY_WEIGHT, len(a & b) / len(union) if union else 0.0))
        total = sum(w for w, _ in parts)
        if total == 0.0:
            return 0.0
        return min(1.0, max(0.0, sum(w * v for w, v in parts) / total))

    def fuse(
        self,
        face_score: float,
        query: ContextMetadata,
        candidate: ContextMetadata,
    ) -> float:
        """Return ``clip(alpha * face_score + beta * context_score)`` in [0, 1].

        Raises ValueError when ``face_score`` is not finite.
        """
        s_face = float(face_score)
        if not math.isfinite(s_face):
            raise ValueError("face_score must be finite")
        s_ctx = self.context_score(query, candidate)
        return min(1.0, max(0.0, self.alpha * s_face + self.beta * s_ctx))

    def in_ambiguous_band(self, face_score: float) -> bool:
        """True when ``face_score`` lies in the profile's ambiguous band [low, high].

        Raises ValueError when ``face_score`` is not finite.
        """
        s = float(face_score)
        if not math.isfinite(s):
            raise ValueError("face_score must be finite")
        return self._profile.ambiguous_low <= s <= self._profile.ambiguous_high


def _abs_days(a: datetime, b: datetime) -> float:
    """Absolute gap between two timestamps in days."""
    return abs((a - b).total_seconds()) / 86400.0
