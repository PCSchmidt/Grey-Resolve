"""Exact brute-force cosine index (numpy) used as the recall oracle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grey_resolve.types import EMBEDDING_DIM

_FORMAT_VERSION = 1
_VECTORS_FILE = "vectors.npz"
_META_FILE = "meta.json"


class BruteForceIndex:
    """Exact cosine top-k over a stored numpy matrix of unit-norm vectors.

    Mirrors the semantics of :class:`FaissHNSWIndex`: vectors are L2-normalized
    on ``add()`` and ``search()`` so scores are true cosine similarities.
    Exact by construction, so it serves as the recall oracle for ANN indexes.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = int(dim)
        self._matrix = np.zeros((0, self._dim), dtype=np.float32)
        self._ids: list[int] = []  # row -> vector id
        self._id_to_row: dict[int, int] = {}  # vector id -> row

    # -- properties ---------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of stored vectors."""
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
            raise ValueError(f"ids must have shape [{arr.shape[0]}], got {id_arr.shape}")
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
        self._matrix = np.concatenate([self._matrix, normalized], axis=0)
        for vid in batch_ids:
            self._id_to_row[vid] = len(self._ids)
            self._ids.append(vid)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return (id, cosine) pairs sorted by score descending.

        Returns [] for an empty index or k <= 0. At most min(k, size) pairs are
        returned. Exact cosine scores.
        """
        if self.size == 0 or k <= 0:
            return []
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self._dim:
            raise ValueError(f"vector must have shape ({self._dim},), got {arr.shape}")
        query = _normalize_rows(arr.reshape(1, -1))[0]
        scores = self._matrix @ query
        top = min(int(k), self.size)
        if top < self.size:
            candidate = np.argpartition(scores, -top)[-top:]
        else:
            candidate = np.arange(self.size)
        order = candidate[np.argsort(-scores[candidate], kind="stable")]
        return [(self._ids[int(row)], float(scores[int(row)])) for row in order]

    def save(self, directory: str) -> None:
        """Persist the vector matrix, the id map, and a version/params stamp."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / _VECTORS_FILE,
            matrix=self._matrix,
            ids=np.asarray(self._ids, dtype=np.int64),
        )
        meta = {
            "format_version": _FORMAT_VERSION,
            "kind": "bruteforce",
            "dim": self._dim,
            "size": self.size,
        }
        (out / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, directory: str) -> None:
        """Replace index state with a previously saved snapshot.

        Raises ``ValueError`` if the snapshot is missing, has an unsupported
        format version, or is inconsistent. The current index must be empty.
        """
        if self.size > 0:
            raise ValueError("cannot load into a non-empty index")
        src = Path(directory)
        meta_path = src / _META_FILE
        if not meta_path.is_file():
            raise ValueError(f"no metadata stamp in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != "bruteforce":
            raise ValueError(f"not a BruteForceIndex snapshot: {meta.get('kind')!r}")
        if meta.get("format_version") != _FORMAT_VERSION:
            raise ValueError(f"unsupported format version: {meta.get('format_version')!r}")
        with np.load(src / _VECTORS_FILE) as data:
            matrix = np.ascontiguousarray(data["matrix"], dtype=np.float32)
            ids = [int(i) for i in data["ids"].tolist()]
        if matrix.ndim != 2 or matrix.shape[1] != self._dim:
            raise ValueError(f"snapshot dim does not match index dim {self._dim}")
        if matrix.shape[0] != len(ids) or int(meta.get("size", -1)) != len(ids):
            raise ValueError("snapshot is inconsistent: matrix, id map, and meta disagree")
        self._matrix = matrix
        self._ids = ids
        self._id_to_row = {vid: row for row, vid in enumerate(ids)}


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Return a float32 C-contiguous copy of [N, dim] with unit-norm rows."""
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("cannot normalize a zero vector")
    return np.ascontiguousarray(arr / norms[:, None], dtype=np.float32)
