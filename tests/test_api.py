"""REST service tests over fake adapters + the real index/sidecar.

Mirrors the fakes in tests/test_pipeline.py (FakeDetector / FakeAssessor /
UnitExtractor / FakeFusion) while BruteForceIndex and SqliteMetadataStore are
the real Phase 1 implementations. Deterministic: fixed seeds, no weights, no
network, no score tolerances beyond exact fake outputs.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from grey_resolve import __version__
from grey_resolve.api.app import create_app
from grey_resolve.config import ThresholdProfile
from grey_resolve.index.bruteforce import BruteForceIndex
from grey_resolve.index.sqlite_store import SqliteMetadataStore
from grey_resolve.pipeline import Pipeline
from grey_resolve.types import FaceObservation, QualityAssessment

PROFILE = ThresholdProfile(
    name="test", quality_min=0.5, ambiguous_low=0.65, ambiguous_high=0.75, alpha=0.7, beta=0.3
)


@pytest.fixture(autouse=True)
def _cross_thread_sqlite(monkeypatch):
    """Let the real SqliteMetadataStore serve FastAPI worker threads.

    FastAPI runs sync handlers on worker threads while SqliteMetadataStore
    opens its sqlite3 connection during setup on the test thread; stdlib
    sqlite3 refuses cross-thread use unless ``check_same_thread=False``.
    The store itself is used unmodified. Integrator note: production wiring
    should open the sidecar connection with ``check_same_thread=False`` (the
    service serializes pipeline calls behind one lock).
    """
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        kwargs.setdefault("check_same_thread", False)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)


class FakeDetector:
    def __init__(self, n_faces: int = 1, quality: float | None = None):
        self._n = n_faces
        self._quality = quality

    def detect(self, image):
        out = []
        for i in range(self._n):
            q = QualityAssessment(score=self._quality, method="fake") if self._quality is not None else None
            out.append(FaceObservation(bbox=(0.0, 0.0, 50.0, 50.0), detection_score=0.9 - 0.1 * i, quality=q))
        return out


class FakeAssessor:
    method = "fake-assessor"

    def assess(self, image, observation):
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


def _png_bytes() -> bytes:
    """One deterministic 64x64 RGB PNG for uploads."""
    ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _upload() -> dict:
    """Multipart file field for POST /ingest and POST /search."""
    return {"image": ("frame.png", _png_bytes(), "image/png")}


def _pipeline(**kwargs) -> tuple[Pipeline, SqliteMetadataStore]:
    store = kwargs.pop("store", SqliteMetadataStore())
    index = kwargs.pop("index", BruteForceIndex())
    defaults = dict(
        detector=FakeDetector(),
        assessor=FakeAssessor(),
        extractor=UnitExtractor(),
        index=index,
        store=store,
        profile=PROFILE,
    )
    defaults.update(kwargs)
    return Pipeline(**defaults), store


def _client(pipeline: Pipeline) -> TestClient:
    return TestClient(create_app(pipeline, profile_name="test"))


def _ingest(client: TestClient, media_id: str, data: dict | None = None) -> dict:
    payload = {"media_id": media_id}
    if data:
        payload.update(data)
    response = client.post("/ingest", files=_upload(), data=payload)
    assert response.status_code == 200, response.text
    return response.json()


# -- ingest ---------------------------------------------------------------


def test_ingest_returns_expected_json():
    pipe, _ = _pipeline()
    client = _client(pipe)
    body = _ingest(client, "m1")
    assert body == {
        "media_id": "m1",
        "faces": [{"vector_id": 0, "trust": "full", "quality_score": 0.9}],
    }
    assert pipe._index.size == 1


def test_ingest_reports_rejected_faces_without_indexing():
    pipe, _ = _pipeline(detector=FakeDetector(n_faces=1, quality=0.1))
    client = _client(pipe)
    body = _ingest(client, "m1")
    assert body["faces"] == [{"vector_id": None, "trust": "reject", "quality_score": 0.1}]
    assert pipe._index.size == 0


def test_ingest_metadata_round_trips_into_sidecar():
    pipe, store = _pipeline()
    client = _client(pipe)
    _ingest(
        client,
        "m1",
        data={
            "timestamp": "2024-03-01T12:00:00+00:00",
            "geo_cluster": "geo-a",
            "source_platform": "platform-a",
            "text_entities": '["ent-a", "ent-b"]',
        },
    )
    payload = store.get(0)
    assert payload is not None
    assert payload["media_id"] == "m1"
    assert payload["timestamp"] == datetime(2024, 3, 1, 12, 0, tzinfo=UTC)
    assert payload["geo_cluster"] == "geo-a"
    assert payload["source_platform"] == "platform-a"
    assert payload["text_entities"] == ("ent-a", "ent-b")
    assert payload["quality_score"] == 0.9


# -- search ---------------------------------------------------------------


def test_search_ranks_and_returns_scores():
    pipe, _ = _pipeline()
    client = _client(pipe)
    _ingest(client, "m1")
    _ingest(client, "m2")
    response = client.post("/search", files=_upload(), params={"k": 2})
    assert response.status_code == 200, response.text
    candidates = response.json()["candidates"]
    assert [c["rank"] for c in candidates] == [1, 2]
    assert {c["media_id"] for c in candidates} == {"m1", "m2"}
    scores = [c["face_score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    # no fusion scorer wired: context/fused stay None
    assert all(c["context_score"] is None and c["fused_score"] is None for c in candidates)


def test_search_with_fusion_uses_query_metadata():
    pipe, _ = _pipeline(fusion=FakeFusion())
    client = _client(pipe)
    _ingest(client, "m1", data={"geo_cluster": "geo-a"})
    _ingest(client, "m2", data={"geo_cluster": "geo-b"})
    response = client.post(
        "/search",
        files=_upload(),
        params={"k": 2},
        data={"geo_cluster": "geo-a"},
    )
    assert response.status_code == 200, response.text
    candidates = response.json()["candidates"]
    assert candidates[0]["media_id"] == "m1"  # matching geo boosts the fused score
    assert candidates[0]["context_score"] == 1.0
    assert candidates[0]["fused_score"] > candidates[1]["fused_score"]
    assert candidates[1]["context_score"] == 0.0


def test_search_on_empty_index_returns_empty_candidates():
    pipe, _ = _pipeline()
    response = _client(pipe).post("/search", files=_upload())
    assert response.status_code == 200, response.text
    assert response.json() == {"candidates": []}


# -- health ---------------------------------------------------------------


def test_health_reports_model_index_and_profile():
    pipe, _ = _pipeline()
    client = _client(pipe)
    _ingest(client, "m1")
    response = client.get("/health")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "version": __version__,
        "model_id": "fake/unit",
        "index_size": 1,
        "profile": "test",
    }


# -- validation -----------------------------------------------------------


def test_undecodable_image_bytes_return_400():
    pipe, _ = _pipeline()
    response = _client(pipe).post(
        "/ingest", files={"image": ("bad.png", b"not-an-image", "image/png")}
    )
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_missing_image_field_returns_422():
    pipe, _ = _pipeline()
    assert _client(pipe).post("/ingest", data={"media_id": "m1"}).status_code == 422
    assert _client(pipe).post("/search").status_code == 422


def test_bad_metadata_returns_422():
    pipe, _ = _pipeline()
    client = _client(pipe)
    bad_inputs = [
        {"text_entities": "not-json"},
        {"text_entities": '{"a": 1}'},
        {"text_entities": "[1, 2]"},
        {"timestamp": "yesterday"},
    ]
    for data in bad_inputs:
        response = client.post("/ingest", files=_upload(), data={"media_id": "m1", **data})
        assert response.status_code == 422, (data, response.text)
        response = client.post("/search", files=_upload(), data=data)
        assert response.status_code == 422, (data, response.text)


def test_k_must_be_positive():
    pipe, _ = _pipeline()
    response = _client(pipe).post("/search", files=_upload(), params={"k": 0})
    assert response.status_code == 422
    response = _client(pipe).post("/search", files=_upload(), params={"k": -3})
    assert response.status_code == 422


def test_k_above_100_is_capped():
    pipe, _ = _pipeline()
    client = _client(pipe)
    for i in range(105):
        _ingest(client, f"m{i}")
    response = client.post("/search", files=_upload(), params={"k": 250})
    assert response.status_code == 200, response.text
    assert len(response.json()["candidates"]) == 100
