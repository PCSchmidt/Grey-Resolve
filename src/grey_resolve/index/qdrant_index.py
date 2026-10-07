"""Qdrant vector index (cosine, named vectors) over qdrant-client.

Migration target for :class:`grey_resolve.index.faiss_index.FaissHNSWIndex`
(ARCHITECTURE.md "Deployment (Phase 3)"): same ``VectorIndex`` contract, but
backed by a Qdrant collection instead of an in-process FAISS HNSW index.

Point-id mapping
----------------
Qdrant point ids must be unsigned 64-bit integers or UUIDs, while application
vector ids are arbitrary signed integers. Rather than an offset trick (which
breaks on negative or sparse ids) we map each vector id to a **deterministic
UUIDv5**: ``uuid5(_NAMESPACE, f"{_POINT_PREFIX}{vector_id}")``. The mapping is
stable across processes and restarts, so re-upserting the same vector id
targets the same point, and no id map file has to be kept in sync. The original
integer id is stored in the point payload (``vector_id``) for reverse
translation on ``search()``.

Modes
-----
- **Local mode** (default): ``QdrantClient(":memory:")`` or a path-backed local
  collection -- tests and single-process runs need no server.
- **Server mode**: pass an existing ``client``, or ``url``/``api_key``, to talk
  to a real Qdrant service (e.g. the ``qdrant`` service in docker-compose).

``save()``/``load()`` keep the FaissHNSWIndex snapshot semantics: ``save()``
writes a portable snapshot (meta stamp + point export) of the collection;
``load()`` restores it into an empty index. In server mode the Qdrant service
owns persistence (its volumes/snapshots), so ``save()`` there is an export and
a no-op for server-side persistence -- see ``save()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from grey_resolve.types import EMBEDDING_DIM

_FORMAT_VERSION = 1
_KIND = "qdrant"
_META_FILE = "meta.json"
_POINTS_FILE = "points.json"
_VECTOR_NAME = "embedding"  # named vector: room for extra vectors (e.g. context) later
_POINT_PREFIX = "grey-resolve/vector-id/"
_NAMESPACE = uuid5(NAMESPACE_URL, "grey-resolve/qdrant-point-ids")


def point_id_for(vector_id: int) -> str:
    """Deterministic UUIDv5 point id for an application vector id."""
    return str(uuid5(_NAMESPACE, f"{_POINT_PREFIX}{int(vector_id)}"))


class QdrantVectorIndex:
    """Append-only ANN index over unit-norm embeddings in a Qdrant collection.

    Implements ``VectorIndex``. Cosine distance on a named vector
    (``"embedding"`` by default); vectors are L2-normalized on ``add()`` and
    ``search()`` so reported scores are true cosine similarities (Qdrant's
    Cosine metric reports similarity as the score).

    ``ids`` are arbitrary signed integers; see the module docstring for the
    UUIDv5 point-id mapping. Ids must be unique -- duplicates raise
    ``ValueError`` (checked exactly against the collection, so shared
    server-mode collections are safe too).

    Constructor modes: default ``location=":memory:"`` runs embedded with no
    server; ``location="/path"`` persists a local collection at that path;
    ``url=``/``api_key=`` (or an injected ``client``) targets a Qdrant server.
    """

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        collection: str = "grey-resolve",
        *,
        client: QdrantClient | None = None,
        location: str = ":memory:",
        url: str | None = None,
        api_key: str | None = None,
        vector_name: str = _VECTOR_NAME,
        m: int = 16,
        ef_construct: int = 40,
        ef_search: int = 64,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if client is not None and url is not None:
            raise ValueError("pass either client or url, not both")
        self._dim = int(dim)
        self._collection = str(collection)
        self._vector_name = str(vector_name)
        self._m = int(m)
        self._ef_construct = int(ef_construct)
        self._ef_search = int(ef_search)
        if client is not None:
            self._client = client
            self._local = False  # external client: server owns persistence
        elif url is not None:
            self._client = QdrantClient(url=url, api_key=api_key)
            self._local = False
        else:
            if location == ":memory:":
                self._client = QdrantClient(":memory:")
            else:
                self._client = QdrantClient(path=location)
            self._local = True
        self._ensure_collection()

    # -- properties ---------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of indexed vectors (exact count via the client)."""
        return int(self._client.count(self._collection, exact=True).count)

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    @property
    def collection(self) -> str:
        """Qdrant collection name."""
        return self._collection

    @property
    def local(self) -> bool:
        """True when this index created its own local (memory/path) client."""
        return self._local

    # -- VectorIndex API ----------------------------------------------------

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add vectors [N, dim] with integer ids [N]. Ids must be unique.

        Vectors are L2-normalized internally (zero vectors are rejected).
        Raises ``ValueError`` on bad shapes, wrong dimension, non-integer or
        duplicate ids (within the batch or already in the collection). No-op
        for an empty batch.
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
            if vid in seen:
                raise ValueError(f"duplicate vector id: {vid}")
            seen.add(vid)
        if arr.shape[0] == 0:
            return
        normalized = _normalize_rows(arr)
        point_ids = [point_id_for(vid) for vid in batch_ids]
        existing = self._client.retrieve(
            self._collection, ids=point_ids, with_payload=["vector_id"]
        )
        if existing:
            found = [int(p.payload["vector_id"]) for p in existing if p.payload]
            raise ValueError(f"duplicate vector id: {found[0] if found else 'unknown'}")
        points = [
            models.PointStruct(
                id=pid,
                vector={self._vector_name: vec.tolist()},
                payload={"vector_id": vid},
            )
            for pid, vid, vec in zip(point_ids, batch_ids, normalized)
        ]
        self._client.upsert(self._collection, points=points, wait=True)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return (id, cosine) pairs sorted by score descending.

        Returns [] for an empty collection or k <= 0. At most min(k, size)
        pairs are returned. The query dimension is validated even for empty
        indexes.
        """
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self._dim:
            raise ValueError(f"vector must have shape ({self._dim},), got {arr.shape}")
        if k <= 0 or self.size == 0:
            return []
        query = _normalize_rows(arr.reshape(1, -1))[0]
        response = self._query(query, limit=min(int(k), self.size))
        pairs = [
            (int(point.payload["vector_id"]), float(point.score))
            for point in response
            if point.payload and "vector_id" in point.payload
        ]
        return sorted(pairs, key=lambda pair: -pair[1])

    def save(self, directory: str) -> None:
        """Persist the collection as a portable snapshot in ``directory``.

        Writes a version/params stamp (``meta.json``) plus every point's id and
        vector (``points.json``), matching the FaissHNSWIndex snapshot
        contract. Server mode: the Qdrant service is authoritative for
        persistence (volumes / Qdrant's own snapshot API); ``save()`` there is
        an export and a **no-op for server-side persistence**.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        points = self._export_points()
        meta = {
            "format_version": _FORMAT_VERSION,
            "kind": _KIND,
            "dim": self._dim,
            "collection": self._collection,
            "vector_name": self._vector_name,
            "m": self._m,
            "ef_construct": self._ef_construct,
            "ef_search": self._ef_search,
            "size": len(points),
        }
        (out / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out / _POINTS_FILE).write_text(json.dumps(points), encoding="utf-8")

    def load(self, directory: str) -> None:
        """Replace index state with a previously saved snapshot.

        Restores every snapshotted point into the (empty) collection. Raises
        ``ValueError`` if the snapshot is missing, has an unsupported format
        version, has a dim mismatch, or the current index is non-empty.
        """
        if self.size > 0:
            raise ValueError("cannot load into a non-empty index")
        src = Path(directory)
        meta_path = src / _META_FILE
        if not meta_path.is_file():
            raise ValueError(f"no metadata stamp in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("kind") != _KIND:
            raise ValueError(f"not a QdrantVectorIndex snapshot: {meta.get('kind')!r}")
        if meta.get("format_version") != _FORMAT_VERSION:
            raise ValueError(f"unsupported format version: {meta.get('format_version')!r}")
        if int(meta.get("dim", -1)) != self._dim:
            raise ValueError(f"snapshot dim {meta.get('dim')} does not match index dim {self._dim}")
        points = json.loads((src / _POINTS_FILE).read_text(encoding="utf-8"))
        if int(meta.get("size", -1)) != len(points):
            raise ValueError("snapshot is inconsistent: meta size disagrees with point export")
        structs = [
            models.PointStruct(
                id=point["id"],
                vector={self._vector_name: point["vector"]},
                payload={"vector_id": int(point["vector_id"])},
            )
            for point in points
        ]
        if structs:
            self._client.upsert(self._collection, points=structs, wait=True)

    def max_vector_id(self) -> int | None:
        """Highest application vector id in the collection (None when empty).

        Lets a service resume a sequential id allocator (e.g.
        ``grey_resolve.pipeline.Pipeline(start_id=...)``) after a restart.
        """
        best: int | None = None
        offset = None
        while True:
            records, offset = self._client.scroll(
                self._collection,
                limit=256,
                offset=offset,
                with_payload=["vector_id"],
                with_vectors=False,
            )
            for record in records:
                if record.payload and "vector_id" in record.payload:
                    vid = int(record.payload["vector_id"])
                    if best is None or vid > best:
                        best = vid
            if offset is None:
                return best

    # -- internals ----------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the collection if absent; verify dim/vector config if present."""
        if self._client.collection_exists(self._collection):
            info = self._client.get_collection(self._collection)
            params = info.config.params.vectors
            # Named vectors come back as a dict keyed by name; plain as a single config.
            if isinstance(params, dict):
                if self._vector_name not in params:
                    raise ValueError(
                        f"collection {self._collection!r} has no vector named {self._vector_name!r}"
                    )
                params = params[self._vector_name]
            if int(params.size) != self._dim:
                raise ValueError(
                    f"collection {self._collection!r} has dim {params.size}, expected {self._dim}"
                )
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                self._vector_name: models.VectorParams(
                    size=self._dim,
                    distance=models.Distance.COSINE,
                )
            },
            hnsw_config=models.HnswConfigDiff(m=self._m, ef_construct=self._ef_construct),
        )

    def _query(self, vector: np.ndarray, limit: int) -> list[Any]:
        """Query the named vector; fall back to search() for qdrant-client < 1.10.

        HNSW ef_search is raised to at least ``limit`` and only sent in server
        mode: local mode searches exactly and ignores ``search_params``.
        """
        extra: dict[str, Any] = {}
        if not self._local:
            extra["search_params"] = models.SearchParams(
                hnsw_ef=max(self._ef_search, limit)
            )
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                self._collection,
                query=vector.tolist(),
                using=self._vector_name,
                limit=limit,
                with_payload=["vector_id"],
                **extra,
            )
            return list(response.points)
        return list(
            self._client.search(  # pragma: no cover - old qdrant-client only
                self._collection,
                query_vector=(self._vector_name, vector.tolist()),
                limit=limit,
                with_payload=["vector_id"],
                **extra,
            )
        )

    def _export_points(self) -> list[dict]:
        """Scroll the whole collection into a JSON-friendly point list."""
        exported: list[dict] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                self._collection,
                limit=256,
                offset=offset,
                with_payload=["vector_id"],
                with_vectors=True,
            )
            for record in records:
                vectors = record.vector
                vector = vectors[self._vector_name] if isinstance(vectors, dict) else vectors
                exported.append(
                    {
                        "id": str(record.id),
                        "vector_id": int(record.payload["vector_id"]),
                        "vector": [float(x) for x in vector],
                    }
                )
            if offset is None:
                return exported


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Return a float32 C-contiguous copy of [N, dim] with unit-norm rows."""
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("cannot normalize a zero vector")
    return np.ascontiguousarray(arr / norms[:, None], dtype=np.float32)
