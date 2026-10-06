"""FAISS HNSW vector index with cosine scoring (inner product on unit-norm vectors).

Cosine similarity between unit-norm vectors equals their inner product, so the
index uses ``faiss.IndexHNSWFlat`` with ``METRIC_INNER_PRODUCT`` and normalizes
every stored and queried vector. Integer vector ids are kept in a Python-side
row<->id map persisted next to the FAISS file so ``search()`` returns stable
application ids rather than internal row numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from grey_resolve.types import EMBEDDING_DIM

_FORMAT_VERSION = 1
_INDEX_FILE = "index.faiss"
_IDS_FILE = "ids.json"
_META_FILE = "meta.json"


class FaissHNSWIndex:
    """Append-only ANN index over unit-norm embeddings (cosine via inner product).

    Hyperparameters follow ``configs/index_hnsw.yaml`` (m=16, ef_construction=40,
    ef_search=64). Vectors are L2-normalized on ``add()`` and ``search()`` so
    reported scores are true cosine similarities.
    """

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        m: int = 16,
        ef_construction: int = 40,
        ef_search: int = 64,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = int(dim)
        self._m = int(m)
        self._ef_construction = int(ef_construction)
        self._ef_search = int(ef_search)
        self._index = faiss.IndexHNSWFlat(self._dim, self._m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self._ef_construction
        self._index.hnsw.efSearch = self._ef_search
        self._ids: list[int] = []  # row -> vector id
        self._id_to_row: dict[int, int] = {}  # vector id -> row

    # -- properties ---------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        return len(self._ids)

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    # -- VectorIndex API ----------------------------------------------------

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add vectors [N, dim] with integer ids [N]. Ids must be unique.

        Vectors are L2-normalized internally (zero vectors are rejected).
        Raises ``ValueError`` on bad shapes, wrong dimension, non-integer or
        duplicate ids (within the batch or against already-stored ids).
        """
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError(f"vectors must have shape [N, {self._dim}], got {arr.shape}")
        id_arr = np.asarray(ids)
        if id_arr.ndim != 1 or id_arr.shape[0] != arr.shape[0]:
            raise ValueError(
                f"ids must have shape [{arr.shape[0]}], got {id_arr.shape}"
            )
        if not np.issubdtype(id_arr.dtype, np.integer):
            raise ValueError(f"ids must be integers, got dtype {id_arr.dtype}")
        batch_ids = [int(i) for i in id_arr.tolist()]
        seen: set[int] = set()
        for vid in batch_ids:
            if vid in seen or vid in self._id_to_row:
                raise ValueError(f"duplicate vector id: {vid}")
            seen.add(vid)
        if arr.shape[0] == 0:
            return
        normalized = _normalize_rows(arr)
        self._index.add(normalized)
        for vid in batch_ids:
            self._id_to_row[vid] = len(self._ids)
            self._ids.append(vid)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return (id, cosine) pairs sorted by score descending.

        Returns [] for an empty index or k <= 0. At most min(k, size) pairs are
        returned.
        """
        if self.size == 0 or k <= 0:
            return []
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self._dim:
            raise ValueError(f"vector must have shape ({self._dim},), got {arr.shape}")
        query = _normalize_rows(arr.reshape(1, -1))
        self._index.hnsw.efSearch = max(self._ef_search, int(k))
        scores, rows = self._index.search(query, min(int(k), self.size))
        pairs = [
            (self._ids[int(row)], float(score))
            for score, row in zip(scores[0], rows[0])
            if int(row) >= 0
        ]
        return sorted(pairs, key=lambda pair: -pair[1])

    def save(self, directory: str) -> None:
        """Persist the FAISS index, the id map, and a version/params stamp."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(out / _INDEX_FILE))
        (out / _IDS_FILE).write_text(json.dumps(self._ids), encoding="utf-8")
        meta = {
            "format_version": _FORMAT_VERSION,
            "kind": "faiss_hnsw",
            "dim": self._dim,
            "m": self._m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
            "size": self.size,
        }
        (out / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, directory: str) -> None:
        """Replace index state with a previously saved snapshot.

        Raises ``ValueError`` if the snapshot is missing, has an unsupported
        format version, or is inconsistent with the persisted id map. The
        current index must be empty.
        """
        if self.size > 0:
            raise ValueError("cannot load into a non-empty index")
        src = Path(directory)
        meta_path = src / _META_FILE
        if not meta_path.is_file():
            raise ValueError(f"no metadata stamp in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "faiss_hnsw":
            raise ValueError(f"not a FaissHNSWIndex snapshot: {meta.get('kind')!r}")
        if meta.get("format_version") != _FORMAT_VERSION:
            raise ValueError(f"unsupported format version: {meta.get('format_version')!r}")
        index = faiss.read_index(str(src / _INDEX_FILE))
        ids = [int(i) for i in json.loads((src / _IDS_FILE).read_text(encoding="utf-8"))]
        if index.ntotal != len(ids) or int(meta.get("size", -1)) != len(ids):
            raise ValueError("snapshot is inconsistent: index, id map, and meta disagree")
        if index.d != self._dim:
            raise ValueError(f"snapshot dim {index.d} does not match index dim {self._dim}")
        self._index = index
        self._dim = int(meta["dim"])
        self._m = int(meta["m"])
        self._ef_construction = int(meta["ef_construction"])
        self._ef_search = int(meta["ef_search"])
        self._index.hnsw.efConstruction = self._ef_construction
        self._index.hnsw.efSearch = self._ef_search
        self._ids = ids
        self._id_to_row = {vid: row for row, vid in enumerate(ids)}


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Return a float32 C-contiguous copy of [N, dim] with unit-norm rows."""
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("cannot normalize a zero vector")
    return np.ascontiguousarray(arr / norms[:, None], dtype=np.float32)
