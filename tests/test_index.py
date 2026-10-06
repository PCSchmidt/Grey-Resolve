"""Tests for the vector index (FAISS HNSW + brute force) and metadata sidecar."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from grey_resolve.index.bruteforce import BruteForceIndex
from grey_resolve.index.faiss_index import FaissHNSWIndex
from grey_resolve.index.sqlite_store import SqliteMetadataStore
from grey_resolve.types import EMBEDDING_DIM, ContextMetadata

SEED = 20240501


def _unit_vectors(rng: np.random.Generator, n: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic random unit-norm vectors [N, dim] (float32)."""
    raw = rng.standard_normal((n, dim), dtype=np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# (1) ANN recall vs exact oracle
# ---------------------------------------------------------------------------


def test_hnsw_recall_vs_bruteforce_at_least_0_9():
    """FAISS HNSW (ef_search=64) must reach >= 0.9 mean recall vs brute force."""
    rng = np.random.default_rng(SEED)
    n, k, n_queries = 2000, 10, 50
    vectors = _unit_vectors(rng, n)
    ids = np.arange(n, dtype=np.int64) + 500  # arbitrary non-contiguous offset
    queries = _unit_vectors(rng, n_queries)

    hnsw = FaissHNSWIndex(
        dim=EMBEDDING_DIM, m=16, ef_construction=40, ef_search=64
    )
    brute = BruteForceIndex(dim=EMBEDDING_DIM)
    hnsw.add(vectors, ids)
    brute.add(vectors, ids)
    assert hnsw.size == n
    assert brute.size == n

    recalls = []
    for query in queries:
        hnsw_ids = {vid for vid, _ in hnsw.search(query, k)}
        brute_ids = {vid for vid, _ in brute.search(query, k)}
        assert len(brute_ids) == k
        recalls.append(len(hnsw_ids & brute_ids) / k)
    assert float(np.mean(recalls)) >= 0.9


# ---------------------------------------------------------------------------
# (2) add/search/save/load round-trip
# ---------------------------------------------------------------------------


def test_faiss_add_search_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(SEED)
    vectors = _unit_vectors(rng, 24)
    ids = np.array([7, 42, 100, 3, 55, 900], dtype=np.int64)
    vectors = vectors[: len(ids)]
    query = _unit_vectors(rng, 1)[0]

    index = FaissHNSWIndex(dim=EMBEDDING_DIM, ef_search=64)
    index.add(vectors, ids)
    before = index.search(query, k=5)
    assert len(before) == 5
    assert [vid for vid, _ in before] == [vid for vid, _ in sorted(before, key=lambda p: -p[1])]

    index.save(str(tmp_path))
    restored = FaissHNSWIndex(dim=EMBEDDING_DIM)
    restored.load(str(tmp_path))
    assert restored.size == len(ids)
    after = restored.search(query, k=5)

    assert [vid for vid, _ in after] == [vid for vid, _ in before]
    assert [score for _, score in after] == pytest.approx(
        [score for _, score in before], abs=1e-6
    )


def test_bruteforce_add_search_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(SEED)
    vectors = _unit_vectors(rng, 24)
    ids = np.array([11, 22, 33], dtype=np.int64)
    vectors = vectors[: len(ids)]
    query = _unit_vectors(rng, 1)[0]

    index = BruteForceIndex(dim=EMBEDDING_DIM)
    index.add(vectors, ids)
    before = index.search(query, k=3)

    index.save(str(tmp_path))
    restored = BruteForceIndex(dim=EMBEDDING_DIM)
    restored.load(str(tmp_path))
    assert restored.size == len(ids)
    after = restored.search(query, k=3)

    assert [vid for vid, _ in after] == [vid for vid, _ in before]
    assert [score for _, score in after] == pytest.approx(
        [score for _, score in before], abs=1e-6
    )


# ---------------------------------------------------------------------------
# (3) add() validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_cls", [FaissHNSWIndex, BruteForceIndex])
def test_duplicate_id_raises(index_cls):
    rng = np.random.default_rng(SEED)
    index = index_cls(dim=EMBEDDING_DIM)
    vectors = _unit_vectors(rng, 3)
    index.add(vectors[:1], np.array([5], dtype=np.int64))
    with pytest.raises(ValueError, match="duplicate"):
        index.add(vectors[1:], np.array([5, 6], dtype=np.int64))
    with pytest.raises(ValueError, match="duplicate"):
        index.add(vectors[1:3], np.array([9, 9], dtype=np.int64))
    assert index.size == 1


@pytest.mark.parametrize("index_cls", [FaissHNSWIndex, BruteForceIndex])
def test_wrong_dim_raises(index_cls):
    rng = np.random.default_rng(SEED)
    index = index_cls(dim=EMBEDDING_DIM)
    with pytest.raises(ValueError, match="shape"):
        index.add(_unit_vectors(rng, 2, dim=32), np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="shape"):
        index.search(_unit_vectors(rng, 1, dim=32)[0], k=1)
    assert index.size == 0


# ---------------------------------------------------------------------------
# (4) empty search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_cls", [FaissHNSWIndex, BruteForceIndex])
def test_empty_search_returns_empty_list(index_cls):
    rng = np.random.default_rng(SEED)
    index = index_cls(dim=EMBEDDING_DIM)
    query = _unit_vectors(rng, 1)[0]
    assert index.search(query, k=10) == []
    index.add(_unit_vectors(rng, 4), np.arange(4, dtype=np.int64))
    assert index.search(query, k=0) == []


# ---------------------------------------------------------------------------
# (5)+(6) SQLite metadata store
# ---------------------------------------------------------------------------


def test_sqlite_put_get_roundtrip_full_and_empty_context():
    store = SqliteMetadataStore()
    full = ContextMetadata(
        timestamp=datetime(2024, 5, 1, 12, 30, 45, tzinfo=timezone.utc),
        geo_cluster="eu-west",
        source_platform="platform_a",
        text_entities=("alice", "berlin", "press-conference"),
    )
    store.put(1, "media-full", full, 0.87)
    row = store.get(1)
    assert row is not None
    assert row["media_id"] == "media-full"
    assert row["quality_score"] == pytest.approx(0.87)
    assert row["ingest_time"].tzinfo is not None
    # ContextMetadata round-trips through put/get.
    rebuilt = ContextMetadata(
        timestamp=row["timestamp"],
        geo_cluster=row["geo_cluster"],
        source_platform=row["source_platform"],
        text_entities=row["text_entities"],
    )
    assert rebuilt == full

    empty = ContextMetadata()
    store.put(2, "media-empty", empty, 0.1)
    row2 = store.get(2)
    assert row2 is not None
    assert row2["timestamp"] is None
    assert row2["geo_cluster"] is None
    assert row2["source_platform"] is None
    assert row2["text_entities"] == ()
    assert ContextMetadata(
        timestamp=row2["timestamp"],
        geo_cluster=row2["geo_cluster"],
        source_platform=row2["source_platform"],
        text_entities=row2["text_entities"],
    ) == empty
    store.close()


def test_sqlite_get_many_roundtrip():
    store = SqliteMetadataStore()
    contexts = {
        10: ContextMetadata(
            timestamp=datetime(2023, 1, 2, 3, 4, 5),
            geo_cluster="us-east",
            source_platform="platform_b",
            text_entities=("bob",),
        ),
        20: ContextMetadata(),
        30: ContextMetadata(geo_cluster="ap-south", text_entities=("x", "y", "z")),
    }
    for vid, ctx in contexts.items():
        store.put(vid, f"media-{vid}", ctx, float(vid) / 100.0)

    rows = store.get_many([30, 10, 999, 20])
    assert set(rows) == {10, 20, 30}  # 999 is missing and omitted
    assert list(rows) == [30, 10, 20]  # input order preserved
    for vid, ctx in contexts.items():
        row = rows[vid]
        assert row["media_id"] == f"media-{vid}"
        assert row["quality_score"] == pytest.approx(vid / 100.0)
        assert ContextMetadata(
            timestamp=row["timestamp"],
            geo_cluster=row["geo_cluster"],
            source_platform=row["source_platform"],
            text_entities=row["text_entities"],
        ) == ctx
    assert store.get_many([]) == {}
    store.close()


def test_sqlite_get_missing_id_returns_none():
    store = SqliteMetadataStore()
    assert store.get(12345) is None
    store.put(1, "media-1", ContextMetadata(), 0.5)
    assert store.get(2) is None
    assert store.get(1) is not None
    store.close()


def test_sqlite_close_and_reopen_persists(tmp_path):
    path = tmp_path / "metadata.db"
    with SqliteMetadataStore(path) as store:
        store.put(3, "media-3", ContextMetadata(geo_cluster="eu"), 0.9)
    with SqliteMetadataStore(path) as store:
        row = store.get(3)
        assert row is not None
        assert row["media_id"] == "media-3"
        assert row["geo_cluster"] == "eu"
