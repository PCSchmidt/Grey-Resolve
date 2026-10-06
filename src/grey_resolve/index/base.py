"""Adapter seams for vector search and metadata storage.

Phase 1: FAISS HNSW (cosine) + SQLite metadata sidecar. The vector index stays
pure -- all payloads (timestamp, geo, source, text entities, quality) live in
the sidecar keyed by integer vector id. Keep brute-force exact search as the
oracle for recall validation (ported from IronClad BruteForce).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from grey_resolve.types import ContextMetadata


@runtime_checkable
class VectorIndex(Protocol):
    """Append-only ANN index over unit-norm embeddings (cosine / inner product)."""

    @property
    def size(self) -> int: ...

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add vectors [N, dim] with integer ids [N]. Ids must be unique."""
        ...

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return (id, cosine) pairs sorted by score descending."""
        ...

    def save(self, directory: str) -> None: ...

    def load(self, directory: str) -> None: ...


@runtime_checkable
class MetadataStore(Protocol):
    """Per-vector payloads; must stay consistent with the index (see ARCHITECTURE.md)."""

    def put(self, vector_id: int, media_id: str, context: ContextMetadata, quality_score: float) -> None: ...

    def get(self, vector_id: int) -> dict | None: ...

    def get_many(self, vector_ids: list[int]) -> dict[int, dict]: ...
