"""Container entrypoint: build the real Grey-Resolve pipeline, serve the REST API.

Wires the production adapters together (ScrfdDetector, HeuristicQualityAssessor,
InsightFaceArcFaceExtractor, FaissHNSWIndex or QdrantVectorIndex, and
SqliteMetadataStore) and hands the app to uvicorn via
``grey_resolve.api.app.create_app``. Configuration is environment-driven:

===============================  ==========================================================
``GREY_RESOLVE_INDEX``           ``faiss`` (default) or ``qdrant``
``GREY_RESOLVE_PROFILE``         threshold profile name (default ``broad_lead``)
``GREY_RESOLVE_STORAGE``         directory for the index snapshot + SQLite sidecar (default ``/data``)
``GREY_RESOLVE_INSIGHTFACE_ROOT`` weights root containing ``models/*.onnx`` (default ``~/.insightface``)
``GREY_RESOLVE_CONFIG_DIR``      directory with the YAML configs (default ``configs``)
``GREY_RESOLVE_QDRANT_URL``      Qdrant URL when ``GREY_RESOLVE_INDEX=qdrant`` (default ``http://qdrant:6333``)
``GREY_RESOLVE_QDRANT_API_KEY``  Qdrant API key (optional)
``GREY_RESOLVE_QDRANT_COLLECTION`` Qdrant collection name (default ``grey-resolve``)
``GREY_RESOLVE_HOST``            bind host (default ``0.0.0.0``)
``GREY_RESOLVE_PORT``            bind port (default ``8000``)
===============================  ==========================================================

Backbone weights are **never baked into the image** (non-commercial grant, see
docs/BACKBONE_LICENSES.md); mount them at runtime (docker-compose mounts
``~/.insightface``). This entrypoint verifies they exist before loading any
model and exits with a clear error otherwise.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from grey_resolve.types import EMBEDDING_DIM

DET_FILE = "det_10g.onnx"
REC_FILE = "w600k_r50.onnx"


def _require_weights(models_dir: Path) -> tuple[Path, Path]:
    """Return (det, rec) ONNX paths, or exit with a clear weights error."""
    det, rec = models_dir / DET_FILE, models_dir / REC_FILE
    missing = [p.name for p in (det, rec) if not p.is_file()]
    if missing:
        sys.stderr.write(
            "ERROR: backbone weights not found: "
            + ", ".join(missing)
            + f" (expected in {models_dir}).\n"
            "The image never contains weights (they are non-commercial; see\n"
            "docs/BACKBONE_LICENSES.md). Two options:\n"
            "  1. Mount the weights directory at runtime, e.g.\n"
            "     -v ~/.insightface:/home/app/.insightface\n"
            "     (docker-compose.yaml already does this).\n"
            "  2. Fetch them once on the host or in a throwaway container:\n"
            "     python scripts/fetch_backbone.py --root <weights-root>\n"
            "Then point GREY_RESOLVE_INSIGHTFACE_ROOT at the mounted root.\n"
        )
        raise SystemExit(1)
    return det, rec


def build_pipeline():
    """Construct the real end-to-end pipeline from environment configuration."""
    from grey_resolve.config import load_threshold_profiles
    from grey_resolve.detection.quality import HeuristicQualityAssessor
    from grey_resolve.detection.scrfd import ScrfdDetector
    from grey_resolve.embeddings.insightface_arcface import InsightFaceArcFaceExtractor
    from grey_resolve.index.sqlite_store import SqliteMetadataStore
    from grey_resolve.pipeline import Pipeline

    weights_root = Path(
        os.environ.get("GREY_RESOLVE_INSIGHTFACE_ROOT", str(Path.home() / ".insightface"))
    )
    det_model, rec_model = _require_weights(weights_root / "models")

    config_dir = Path(os.environ.get("GREY_RESOLVE_CONFIG_DIR", "configs"))
    profile_name = os.environ.get("GREY_RESOLVE_PROFILE", "broad_lead")
    profiles = load_threshold_profiles(config_dir / "threshold_profiles.yaml")
    if profile_name not in profiles:
        raise SystemExit(
            f"ERROR: unknown profile {profile_name!r}; "
            f"choose one of: {', '.join(sorted(profiles))}"
        )
    profile = profiles[profile_name]

    storage = Path(os.environ.get("GREY_RESOLVE_STORAGE", "/data"))
    storage.mkdir(parents=True, exist_ok=True)

    index_kind = os.environ.get("GREY_RESOLVE_INDEX", "faiss").strip().lower()
    if index_kind == "faiss":
        index, start_id = _build_faiss_index(config_dir, storage)
    elif index_kind == "qdrant":
        index, start_id = _build_qdrant_index()
    else:
        raise SystemExit(
            f"ERROR: GREY_RESOLVE_INDEX={index_kind!r} is not supported; use faiss or qdrant"
        )

    store = SqliteMetadataStore(storage / "metadata.db")
    pipeline = Pipeline(
        detector=ScrfdDetector(model_file=det_model),
        assessor=HeuristicQualityAssessor(),
        extractor=InsightFaceArcFaceExtractor(model_file=rec_model),
        index=index,
        store=store,
        profile=profile,
        fusion=None,
        start_id=start_id,
    )
    return pipeline, profile_name


def _build_faiss_index(config_dir: Path, storage: Path):
    """FAISS HNSW + SQLite path (default reference implementation)."""
    from grey_resolve.config import load_hnsw_config
    from grey_resolve.index.faiss_index import FaissHNSWIndex

    hnsw = load_hnsw_config(config_dir / "index_hnsw.yaml")
    index = FaissHNSWIndex(
        dim=EMBEDDING_DIM,
        m=hnsw.m,
        ef_construction=hnsw.ef_construction,
        ef_search=hnsw.ef_search,
    )
    snapshot_dir = storage / "index"
    if (snapshot_dir / "meta.json").is_file():
        index.load(str(snapshot_dir))
    # Pipeline ids are allocated sequentially from 0; resume past the snapshot.
    start_id = index.size
    ids_file = snapshot_dir / "ids.json"
    if ids_file.is_file():
        ids = [int(i) for i in json.loads(ids_file.read_text(encoding="utf-8"))]
        if ids:
            start_id = max(ids) + 1
    return index, start_id


def _build_qdrant_index():
    """Qdrant path: index in the qdrant service instead of in-process FAISS.

    Enabled with GREY_RESOLVE_INDEX=qdrant (see docker-compose.yaml). The
    SQLite metadata sidecar stays as-is; Qdrant payloads are a later step of
    the FAISS->Qdrant migration story.
    """
    from grey_resolve.index.qdrant_index import QdrantVectorIndex

    index = QdrantVectorIndex(
        dim=EMBEDDING_DIM,
        collection=os.environ.get("GREY_RESOLVE_QDRANT_COLLECTION", "grey-resolve"),
        url=os.environ.get("GREY_RESOLVE_QDRANT_URL", "http://qdrant:6333"),
        api_key=os.environ.get("GREY_RESOLVE_QDRANT_API_KEY") or None,
    )
    max_id = index.max_vector_id()
    return index, (max_id + 1) if max_id is not None else 0


def main() -> None:
    """Build the pipeline and serve it with uvicorn."""
    import uvicorn

    from grey_resolve.api.app import create_app

    pipeline, profile_name = build_pipeline()
    app = create_app(pipeline, profile_name=profile_name)
    host = os.environ.get("GREY_RESOLVE_HOST", "0.0.0.0")
    port = int(os.environ.get("GREY_RESOLVE_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
