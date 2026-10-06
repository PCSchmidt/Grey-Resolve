"""Pydantic response models for the Grey-Resolve REST service.

Score conventions follow ARCHITECTURE.md: face score (cosine) in [-1, 1],
context score in [0, 1], fused score in [0, 1]. All service outputs are
ranked similarity candidates, not identity assertions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "IngestedFace",
    "IngestResponse",
    "SearchCandidate",
    "SearchResponse",
    "HealthResponse",
]


class IngestedFace(BaseModel):
    """Outcome for one detected face within an ingested media item.

    ``trust`` mirrors the quality-gate decision (``TrustDecision``): ``full``
    indexes with full weight, ``low`` indexes flagged low-trust, ``reject``
    is reported but never indexed (``vector_id`` is None).
    """

    vector_id: int | None = Field(
        ..., description="Stable vector id in the index, or None when the face was rejected."
    )
    trust: Literal["full", "low", "reject"] = Field(
        ..., description="Quality-gate outcome for this face."
    )
    quality_score: float = Field(
        ..., description="FIQA face-quality score in [0, 1]; higher is better."
    )


class IngestResponse(BaseModel):
    """Ingest outcome for one uploaded media item.

    Per-face records say where each detected face landed and how much it is
    trusted. All service outputs are ranked similarity candidates, not
    identity assertions.
    """

    media_id: str = Field(..., description="Media id the faces were indexed under.")
    faces: list[IngestedFace] = Field(
        ..., description="One record per detected face, in detection order."
    )


class SearchCandidate(BaseModel):
    """One ranked candidate returned by search: scores, not identity claims."""

    media_id: str = Field(..., description="Media id of the candidate, from the metadata sidecar.")
    face_score: float = Field(..., description="Cosine face similarity in [-1, 1].")
    context_score: float | None = Field(
        default=None, description="Context compatibility in [0, 1], or None without fusion."
    )
    fused_score: float | None = Field(
        default=None, description="Fused score in [0, 1], or None without fusion."
    )
    rank: int = Field(..., description="1-based rank position (best candidate first).")


class SearchResponse(BaseModel):
    """Ranked search results for one query image.

    Candidates are ranked similarity results with scores; interpreting them
    as identity claims is out of scope and unsupported. All service outputs
    are ranked similarity candidates, not identity assertions.
    """

    candidates: list[SearchCandidate] = Field(
        ..., description="Candidates sorted best-first; empty when no faces or no index hits."
    )


class HealthResponse(BaseModel):
    """Service health and configuration snapshot.

    Reports the model, index size, and active threshold profile so callers can
    verify what is serving. All service outputs are ranked similarity
    candidates, not identity assertions.
    """

    version: str = Field(..., description="grey_resolve package version.")
    model_id: str = Field(..., description="Embedding backbone identifier.")
    index_size: int = Field(..., description="Number of vectors currently indexed.")
    profile: str = Field(..., description="Name of the active threshold profile.")
