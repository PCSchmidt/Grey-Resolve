"""Pipeline wiring tests over fake adapters + the real index/sidecar.

No model weights required: the detector/assessor/extractor are deterministic
fakes, while FaissHNSWIndex/BruteForceIndex and SqliteMetadataStore are real.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from grey_resolve.config import ThresholdProfile
from grey_resolve.detection.base import TrustDecision
from grey_resolve.index.bruteforce import BruteForceIndex
from grey_resolve.index.sqlite_store import SqliteMetadataStore
from grey_resolve.pipeline import Pipeline
from grey_resolve.types import ContextMetadata, FaceObservation, MediaItem

PROFILE = ThresholdProfile(
    name="test", quality_min=0.5, ambiguous_low=0.65, ambiguous_high=0.75, alpha=0.7, beta=0.3
)


class FakeDetector:
    def __init__(self, n_faces: int = 1, quality: float | None = None):
        self._n = n_faces
        self._quality = quality

    def detect(self, image):
        from grey_resolve.types import QualityAssessment

        out = []
        for i in range(self._n):
            q = QualityAssessment(score=self._quality, method="fake") if self._quality is not None else None
            out.append(
                FaceObservation(
                    bbox=(0.0, 0.0, 50.0, 50.0),
                    detection_score=0.9 - 0.1 * i,
                    quality=q,
                )
            )
        return out


class FakeAssessor:
    method = "fake-assessor"

    def assess(self, image, observation):
        from grey_resolve.types import QualityAssessment

        return QualityAssessment(score=0.9, method=self.method)


class UnitExtractor:
    """Returns unit-norm vectors seeded by observation bbox (exact for oracle tests)."""

    dimension = 512
    model_id = "fake/unit"

    def extract(self, image, observation):
        rng = np.random.default_rng(int(observation.bbox[2]))
        v = rng.normal(size=512).astype(np.float32)
        return v / np.linalg.norm(v)


class FakeFusion:
    def context_score(self, query, candidate):
        return 1.0 if query.geo_cluster == candidate.geo_cluster else 0.0

    def fuse(self, face_score, query, candidate):
        return 0.7 * face_score + 0.3 * self.context_score(query, candidate)


def _media(idx: int, geo: str = "geo-a") -> MediaItem:
    return MediaItem(
        media_id=f"m{idx}",
        image_ref=f"images/{idx}.png",
        context=ContextMetadata(
            timestamp=datetime(2024, 1 + (idx % 12), 1, tzinfo=UTC),
            geo_cluster=geo,
            source_platform="platform-a",
            text_entities=("ent-a",),
        ),
    )


def _pipeline(**kwargs) -> Pipeline:
    index = kwargs.pop("index", BruteForceIndex())
    store = kwargs.pop("store", SqliteMetadataStore())
    defaults = dict(
        detector=FakeDetector(),
        assessor=FakeAssessor(),
        extractor=UnitExtractor(),
        index=index,
        store=store,
        profile=PROFILE,
    )
    defaults.update(kwargs)
    return Pipeline(**defaults)


def test_ingest_indexes_passing_faces():
    pipe = _pipeline(detector=FakeDetector(n_faces=2))
    results = pipe.ingest(_media(1), np.zeros((64, 64, 3), dtype=np.uint8))
    assert [r.vector_id for r in results] == [0, 1]
    assert all(r.trust is TrustDecision.FULL for r in results)
    assert pipe._index.size == 2


def test_ingest_rejects_low_quality_faces():
    pipe = _pipeline(detector=FakeDetector(n_faces=1, quality=0.1))
    results = pipe.ingest(_media(1), np.zeros((64, 64, 3), dtype=np.uint8))
    assert results[0].trust is TrustDecision.REJECT
    assert results[0].vector_id is None
    assert pipe._index.size == 0


def test_search_ranks_by_face_score_without_fusion():
    pipe = _pipeline(detector=FakeDetector(n_faces=1))
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    pipe.ingest(_media(1), img)
    pipe.ingest(_media(2), img)
    results = pipe.search(img, k=2)
    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]
    assert results[0].face_score >= results[1].face_score
    assert all(r.fused_score is None for r in results)
    assert {r.media_id for r in results} == {"m1", "m2"}


def test_search_with_fusion_uses_fused_scores():
    pipe = _pipeline(detector=FakeDetector(n_faces=1), fusion=FakeFusion())
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    pipe.ingest(_media(1, geo="geo-a"), img)
    pipe.ingest(_media(2, geo="geo-b"), img)
    query_ctx = ContextMetadata(geo_cluster="geo-a")
    results = pipe.search(img, k=2, query_context=query_ctx)
    assert results[0].media_id == "m1"  # matching geo boosts the fused score
    assert results[0].fused_score is not None
    assert results[0].fused_score >= results[1].fused_score


def test_search_empty_index_returns_empty():
    pipe = _pipeline()
    assert pipe.search(np.zeros((64, 64, 3), dtype=np.uint8), k=5) == []


def test_search_no_faces_returns_empty():
    pipe = _pipeline(detector=FakeDetector(n_faces=0))
    assert pipe.search(np.zeros((64, 64, 3), dtype=np.uint8), k=5) == []


def test_search_rejects_bad_k():
    pipe = _pipeline()
    with pytest.raises(ValueError):
        pipe.search(np.zeros((64, 64, 3), dtype=np.uint8), k=0)
