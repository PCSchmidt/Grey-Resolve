"""Adapter seam for the Phase 2 fusion scorer.

P(identity) = alpha * S_face + beta * S_context, calibrated on the synthetic
scenario dataset (docs/SYNTHETIC_SCENARIO_SPEC.md). In the ambiguous band
(profile.ambiguous_low .. ambiguous_high) context decides near-ties; outside it
the face score dominates. Honest ablation (face-only vs fused) is required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from grey_resolve.types import ContextMetadata


@runtime_checkable
class FusionScorer(Protocol):
    """Combines face similarity with context compatibility into one score in [0, 1]."""

    def context_score(self, query: ContextMetadata, candidate: ContextMetadata) -> float:
        """Compatibility of two context payloads in [0, 1]."""
        ...

    def fuse(self, face_score: float, query: ContextMetadata, candidate: ContextMetadata) -> float:
        """Return the fused score in [0, 1]."""
        ...
