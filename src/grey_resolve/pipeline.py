"""Ingest and search orchestration over the adapter seams.

Wire-up per ARCHITECTURE.md:
- ingest: detect -> FIQA gate -> embed -> index + metadata sidecar
- search: detect best face -> embed -> ANN search -> context metadata -> optional fusion

The pipeline depends only on the Protocol seams in detection/, embeddings/,
index/, and fusion/, so every model choice stays swappable. Responses are
ranked candidates with scores; candidate ranking is not access authorization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grey_resolve.config import ThresholdProfile
from grey_resolve.detection.base import QualityAssessor, QualityGate, TrustDecision
from grey_resolve.index.base import MetadataStore, VectorIndex
from grey_resolve.types import ContextMetadata, FaceObservation, MediaItem, SearchResult

_AMBIGUOUS_EPS = 1e-9


@dataclass(frozen=True)
class IngestResult:
    """One ingested face: where it landed and how much it is trusted."""

    vector_id: int | None
    trust: TrustDecision
    quality_score: float


class Pipeline:
    """End-to-end ingest + search over swappable adapters."""

    def __init__(
        self,
        detector,
        assessor: QualityAssessor,
        extractor,
        index: VectorIndex,
        store: MetadataStore,
        profile: ThresholdProfile,
        fusion=None,
        start_id: int = 0,
    ):
        self._detector = detector
        self._assessor = assessor
        self._extractor = extractor
        self._index = index
        self._store = store
        self._gate = QualityGate(profile=profile)
        self._profile = profile
        self._fusion = fusion
        self._next_id = start_id

    # -- introspection (used by /health) ---------------------------------

    @property
    def model_id(self) -> str:
        return str(getattr(self._extractor, "model_id", "unknown"))

    @property
    def index_size(self) -> int:
        return int(self._index.size)

    @property
    def profile_name(self) -> str:
        return self._profile.name

    # -- ingest ---------------------------------------------------------

    def ingest(self, media: MediaItem, image_bgr: np.ndarray) -> list[IngestResult]:
        """Detect all faces in the image and ingest the ones that pass the gate.

        REJECTed faces are reported but never indexed. LOW-trust faces are
        indexed and flagged (quality score in the sidecar marks the trust).
        """
        results: list[IngestResult] = []
        for observation in self._detector.detect(image_bgr):
            assessment = self._assess(image_bgr, observation)
            trust = self._gate.decide(assessment)
            if trust is TrustDecision.REJECT:
                results.append(IngestResult(vector_id=None, trust=trust, quality_score=assessment.score))
                continue
            embedding = self._extractor.extract(image_bgr, observation)
            vector_id = self._next_id
            self._next_id += 1
            self._index.add(embedding[None, :], np.array([vector_id], dtype=np.int64))
            self._store.put(vector_id, media.media_id, media.context, quality_score=assessment.score)
            results.append(IngestResult(vector_id=vector_id, trust=trust, quality_score=assessment.score))
        return results

    def _assess(self, image_bgr: np.ndarray, observation: FaceObservation):
        if observation.quality is not None:
            return observation.quality
        return self._assessor.assess(image_bgr, observation)

    # -- search ---------------------------------------------------------

    def search(self, image_bgr: np.ndarray, k: int = 10, query_context=None) -> list[SearchResult]:
        """Rank indexed candidates for the most confident detected face.

        Returns ranked candidates with scores -- similarity results, never
        identity assertions. Without a fusion scorer, rank by face score only.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        observations = self._detector.detect(image_bgr)
        if not observations:
            return []
        best = observations[0]
        query_embedding = self._extractor.extract(image_bgr, best)
        hits = self._index.search(query_embedding, k)
        if not hits:
            return []
        payloads = self._store.get_many([vid for vid, _ in hits])

        results: list[SearchResult] = []
        for vector_id, face_score in hits:
            payload = payloads.get(vector_id, {})
            context = _context_from_payload(payload)
            fused = None
            context_score = None
            if self._fusion is not None and query_context is not None and context is not None:
                context_score = float(self._fusion.context_score(query_context, context))
                fused = float(self._fusion.fuse(face_score, query_context, context))
            results.append(
                SearchResult(
                    media_id=str(payload.get("media_id", "")),
                    face_score=float(face_score),
                    context_score=context_score,
                    fused_score=fused,
                )
            )
        key = (lambda r: r.fused_score) if (self._fusion is not None and query_context is not None) else (lambda r: r.face_score)
        results.sort(key=key, reverse=True)
        return [
            SearchResult(
                media_id=r.media_id,
                face_score=r.face_score,
                context_score=r.context_score,
                fused_score=r.fused_score,
                rank=i,
            )
            for i, r in enumerate(results, start=1)
        ]


def _context_from_payload(payload: dict) -> ContextMetadata | None:
    """Rebuild a ContextMetadata from a flat sidecar payload dict.

    SqliteMetadataStore returns flat keys (timestamp, geo_cluster,
    source_platform, text_entities); see index/sqlite_store.py.
    """
    if not payload:
        return None
    return ContextMetadata(
        timestamp=payload.get("timestamp"),
        geo_cluster=payload.get("geo_cluster"),
        source_platform=payload.get("source_platform"),
        text_entities=tuple(payload.get("text_entities") or ()),
    )
