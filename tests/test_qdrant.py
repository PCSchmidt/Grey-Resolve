"""Tests for the Qdrant vector index (local mode, no server).

All tests run against ``QdrantClient(":memory:")`` local mode so the suite
needs no Qdrant daemon. Skipped cleanly when qdrant-client is not installed.
Correctness is checked against a brute-force numpy cosine oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from grey_resolve.index.qdrant_index import QdrantVectorIndex, point_id_for

SEED = 20240501
DIM = 32


def _unit_vectors(rng: np.random.Generator, n: int, dim: int = DIM) -> np.ndarray:
    """Deterministic random unit-norm vectors [N, dim] (float32)."""
    raw = rng.standard_normal((n, dim), dtype=np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def _oracle_top_k(vectors: np.ndarray, ids: np.ndarray, query: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Exact cosine top-k via numpy (vectors are unit-norm): (id, score) desc."""
    scores = vectors @ query
    order = np.argsort(-scores)[:k]
    return [(int(ids[i]), float(scores[i])) for i in order]


# ---------------------------------------------------------------------------
# (1) add/search correctness vs numpy oracle
# ---------------------------------------------------------------------------


def test_search_matches_numpy_oracle_top_k():
    rng = np.random.default_rng(SEED)
    n, k = 64, 5
    vectors = _unit_vectors(rng, n)
    ids = np.arange(n, dtype=np.int64) + 100  # arbitrary non-contiguous offset
    index = QdrantVectorIndex(dim=DIM)
    index.add(vectors, ids)
    assert index.size == n

    for query in _unit_vectors(rng, 10):
        expected = _oracle_top_k(vectors, ids, query, k)
        got = index.search(query, k)
        assert [vid for vid, _ in got] == [vid for vid, _ in expected]
        assert [score for _, score in got] == pytest.approx(
            [score for _, score in expected], abs=1e-5
        )
        assert [score for _, score in got] == sorted(
            (score for _, score in got), reverse=True
        )


def test_negative_and_sparse_ids_round_trip():
    """Signed application ids survive the uuid5 point-id mapping."""
    rng = np.random.default_rng(SEED)
    vectors = _unit_vectors(rng, 3)
    ids = np.array([-7, 0, 2**40], dtype=np.int64)
    index = QdrantVectorIndex(dim=DIM)
    index.add(vectors, ids)
    assert index.size == 3
    query = vectors[1]
    got = [vid for vid, _ in index.search(query, k=3)]
    assert got[0] == 0  # exact self-match on a unit vector
    assert set(got) == {-7, 0, 2**40}


# ---------------------------------------------------------------------------
# (2) unique-id enforcement
# ---------------------------------------------------------------------------


def test_duplicate_ids_raise():
    rng = np.random.default_rng(SEED)
    index = QdrantVectorIndex(dim=DIM)
    vectors = _unit_vectors(rng, 3)
    index.add(vectors[:1], np.array([5], dtype=np.int64))
    with pytest.raises(ValueError, match="duplicate"):
        index.add(vectors[1:], np.array([5, 6], dtype=np.int64))  # vs stored id
    with pytest.raises(ValueError, match="duplicate"):
        index.add(vectors[1:3], np.array([9, 9], dtype=np.int64))  # in-batch
    assert index.size == 1


# ---------------------------------------------------------------------------
# (3) save/load round trip in local mode
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_local_mode(tmp_path):
    rng = np.random.default_rng(SEED)
    vectors = _unit_vectors(rng, 24)
    ids = np.array([7, 42, 100, 3, 55, 900], dtype=np.int64)
    vectors = vectors[: len(ids)]
    query = _unit_vectors(rng, 1)[0]

    index = QdrantVectorIndex(dim=DIM)
    index.add(vectors, ids)
    before = index.search(query, k=5)
    assert len(before) == 5

    index.save(str(tmp_path))
    restored = QdrantVectorIndex(dim=DIM)  # fresh :memory: collection
    restored.load(str(tmp_path))
    assert restored.size == len(ids)
    after = restored.search(query, k=5)

    assert [vid for vid, _ in after] == [vid for vid, _ in before]
    assert [score for _, score in after] == pytest.approx(
        [score for _, score in before], abs=1e-6
    )
    # Restored index keeps enforcing uniqueness and accepts new ids.
    with pytest.raises(ValueError, match="duplicate"):
        restored.add(vectors[:1], np.array([7], dtype=np.int64))


def test_load_rejects_bad_snapshots(tmp_path):
    import json

    rng = np.random.default_rng(SEED)
    empty = QdrantVectorIndex(dim=DIM)
    with pytest.raises(ValueError, match="metadata stamp"):
        empty.load(str(tmp_path))  # empty directory

    # A valid snapshot, then corrupt one copy of it per failure mode.
    donor = QdrantVectorIndex(dim=DIM)
    donor.add(_unit_vectors(rng, 2), np.array([1, 2], dtype=np.int64))
    donor.save(str(tmp_path / "good"))

    meta = json.loads((tmp_path / "good" / "meta.json").read_text(encoding="utf-8"))
    (tmp_path / "good" / "meta.json").write_text(
        json.dumps({**meta, "kind": "faiss_hnsw"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not a QdrantVectorIndex snapshot"):
        QdrantVectorIndex(dim=DIM).load(str(tmp_path / "good"))
    (tmp_path / "good" / "meta.json").write_text(
        json.dumps({**meta, "format_version": 99}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="format version"):
        QdrantVectorIndex(dim=DIM).load(str(tmp_path / "good"))
    (tmp_path / "good" / "meta.json").write_text(
        json.dumps({**meta, "dim": DIM + 8}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="snapshot dim"):
        QdrantVectorIndex(dim=DIM).load(str(tmp_path / "good"))
    (tmp_path / "good" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Non-empty target refuses to load.
    busy = QdrantVectorIndex(dim=DIM)
    busy.add(_unit_vectors(rng, 1), np.array([9], dtype=np.int64))
    with pytest.raises(ValueError, match="non-empty"):
        busy.load(str(tmp_path / "good"))


# ---------------------------------------------------------------------------
# (4) empty search / k<=0
# ---------------------------------------------------------------------------


def test_empty_search_returns_empty_list():
    rng = np.random.default_rng(SEED)
    index = QdrantVectorIndex(dim=DIM)
    query = _unit_vectors(rng, 1)[0]
    assert index.search(query, k=10) == []
    index.add(_unit_vectors(rng, 4), np.arange(4, dtype=np.int64))
    assert index.search(query, k=0) == []


# ---------------------------------------------------------------------------
# (5) wrong-dim errors
# ---------------------------------------------------------------------------


def test_wrong_dim_raises():
    rng = np.random.default_rng(SEED)
    index = QdrantVectorIndex(dim=DIM)
    with pytest.raises(ValueError, match="shape"):
        index.add(_unit_vectors(rng, 2, dim=16), np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="shape"):
        index.search(_unit_vectors(rng, 1, dim=16)[0], k=1)
    assert index.size == 0


# ---------------------------------------------------------------------------
# (6) misc: validation + point-id mapping stability
# ---------------------------------------------------------------------------


def test_add_validates_ids_and_zero_vectors():
    rng = np.random.default_rng(SEED)
    index = QdrantVectorIndex(dim=DIM)
    with pytest.raises(ValueError, match="integers"):
        index.add(_unit_vectors(rng, 2), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="shape"):
        index.add(_unit_vectors(rng, 2), np.array([1], dtype=np.int64))
    with pytest.raises(ValueError, match="zero vector"):
        index.add(np.zeros((1, DIM), dtype=np.float32), np.array([1], dtype=np.int64))
    index.add(np.empty((0, DIM), dtype=np.float32), np.empty((0,), dtype=np.int64))
    assert index.size == 0


def test_point_id_mapping_is_deterministic():
    assert point_id_for(42) == point_id_for(42)
    assert point_id_for(42) != point_id_for(43)
    assert point_id_for(-1) != point_id_for(1)


def test_max_vector_id():
    rng = np.random.default_rng(SEED)
    index = QdrantVectorIndex(dim=DIM)
    assert index.max_vector_id() is None
    index.add(_unit_vectors(rng, 3), np.array([-5, 2, 9], dtype=np.int64))
    assert index.max_vector_id() == 9
