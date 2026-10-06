"""FastAPI REST service wrapping :class:`grey_resolve.pipeline.Pipeline`.

Endpoints (ARCHITECTURE.md "API"): ``POST /ingest``, ``POST /search``,
``GET /health``. Semantic contract: responses are ranked similarity
candidates with scores -- never identity assertions. **Candidate ranking is
not access authorization** (noted in the app description too).

Dependency injection only: :func:`create_app` receives a fully constructed
pipeline and a profile name. No global state, no module-level model loading.

Serving (construct the app first, then hand it to uvicorn)::

    from grey_resolve.api.app import create_app
    import uvicorn

    app = create_app(pipeline, profile_name="broad-lead")
    uvicorn.run(app, host="0.0.0.0", port=8000)

The factory needs a constructed pipeline, so there is no import-string
entrypoint; a small launcher module builds the pipeline and calls
``uvicorn.run(create_app(pipeline, profile_name))``.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from grey_resolve import __version__
from grey_resolve.api.schemas import (
    HealthResponse,
    IngestedFace,
    IngestResponse,
    SearchCandidate,
    SearchResponse,
)
from grey_resolve.pipeline import Pipeline
from grey_resolve.types import ContextMetadata, MediaItem

_MAX_K = 100

_APP_DESCRIPTION = (
    "Multimodal entity-resolution research service over degraded media. "
    "Responses are ranked similarity candidates with scores; the API never "
    "asserts identity. Candidate ranking is not access authorization."
)


def create_app(pipeline: Pipeline, profile_name: str) -> FastAPI:
    """Build the Grey-Resolve REST app around ``pipeline``.

    ``pipeline`` is any :class:`~grey_resolve.pipeline.Pipeline` (or duck-typed
    equivalent with ``ingest`` / ``search``); adapters are injected, so no
    model is loaded here. ``profile_name`` is reported by ``GET /health``.

    Pipeline calls are serialized behind one lock: FastAPI runs sync handlers
    on worker threads, while the SQLite metadata sidecar keeps a single
    connection that must not run interleaved transactions.
    """
    pipeline_lock = threading.Lock()
    app = FastAPI(
        title="Grey-Resolve",
        description=_APP_DESCRIPTION,
        version=__version__,
    )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(
        image: UploadFile = File(..., description="Image bytes (PNG/JPEG/...), decoded to BGR pixels."),
        media_id: str | None = Form(None, description="Media id; defaults to the upload filename."),
        timestamp: str | None = Form(None, description="ISO 8601 timestamp (optional)."),
        geo_cluster: str | None = Form(None, description="Coarse geo label (optional)."),
        source_platform: str | None = Form(None, description="Source platform label (optional)."),
        text_entities: str | None = Form(
            None, description='JSON list of strings, e.g. ["entity-a", "entity-b"].'
        ),
    ) -> IngestResponse:
        """Detect, quality-gate, embed, and index all faces in the uploaded image.

        Returns one record per detected face (vector id, trust, quality score).
        Rejected faces are reported but never indexed.
        """
        image_bgr = _decode_image(image)
        context = _build_context(timestamp, geo_cluster, source_platform, text_entities)
        resolved_id = media_id or image.filename or f"upload-{uuid4().hex}"
        media = MediaItem(
            media_id=resolved_id,
            image_ref=image.filename or resolved_id,
            context=context if context is not None else ContextMetadata(),
        )
        with pipeline_lock:
            results = pipeline.ingest(media, image_bgr)
        return IngestResponse(
            media_id=resolved_id,
            faces=[
                IngestedFace(vector_id=r.vector_id, trust=r.trust.value, quality_score=r.quality_score)
                for r in results
            ],
        )

    @app.post("/search", response_model=SearchResponse)
    def search(
        image: UploadFile = File(..., description="Query image bytes, decoded to BGR pixels."),
        k: int = Query(10, ge=1, description=f"Maximum candidates (default 10, capped at {_MAX_K})."),
        timestamp: str | None = Form(None, description="Query-context ISO 8601 timestamp (optional)."),
        geo_cluster: str | None = Form(None, description="Query-context geo label (optional)."),
        source_platform: str | None = Form(None, description="Query-context source platform (optional)."),
        text_entities: str | None = Form(
            None, description='Query-context JSON list of strings, e.g. ["entity-a"].'
        ),
    ) -> SearchResponse:
        """Rank indexed candidates for the query image.

        Returns ranked similarity candidates with face, context, and fused
        scores -- not identity assertions. Context metadata is optional and
        only affects ranking when the pipeline has a fusion scorer.
        """
        image_bgr = _decode_image(image)
        query_context = _build_context(timestamp, geo_cluster, source_platform, text_entities)
        with pipeline_lock:
            results = pipeline.search(image_bgr, k=min(k, _MAX_K), query_context=query_context)
        return SearchResponse(
            candidates=[
                SearchCandidate(
                    media_id=r.media_id,
                    face_score=r.face_score,
                    context_score=r.context_score,
                    fused_score=r.fused_score,
                    rank=r.rank,
                )
                for r in results
            ]
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report service version, embedding model, index size, and profile."""
        with pipeline_lock:
            model_id, index_size = _model_id(pipeline), _index_size(pipeline)
        return HealthResponse(
            version=__version__,
            model_id=model_id,
            index_size=index_size,
            profile=profile_name,
        )

    return app


# -- helpers ---------------------------------------------------------------


def _decode_image(image: UploadFile) -> np.ndarray:
    """Decode uploaded bytes to a BGR uint8 array, or raise HTTP 400."""
    data = image.file.read()
    buffer = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode uploaded image; provide valid image bytes (e.g. PNG or JPEG).",
        )
    return image_bgr


def _build_context(
    timestamp: str | None,
    geo_cluster: str | None,
    source_platform: str | None,
    text_entities: str | None,
) -> ContextMetadata | None:
    """Parse optional form metadata into a ContextMetadata (None when absent).

    Raises HTTP 422 for an unparseable ISO 8601 timestamp or a ``text_entities``
    value that is not a JSON list of strings.
    """
    if timestamp is None and geo_cluster is None and source_platform is None and text_entities is None:
        return None
    parsed_timestamp: datetime | None = None
    if timestamp is not None:
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid timestamp {timestamp!r}; expected ISO 8601 (e.g. 2024-03-01T12:00:00Z).",
            ) from None
    entities: tuple[str, ...] = ()
    if text_entities is not None:
        try:
            loaded = json.loads(text_entities)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid text_entities {text_entities!r}; expected a JSON list of strings.",
            ) from None
        if not isinstance(loaded, list) or not all(isinstance(e, str) for e in loaded):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid text_entities {text_entities!r}; expected a JSON list of strings.",
            )
        entities = tuple(loaded)
    return ContextMetadata(
        timestamp=parsed_timestamp,
        geo_cluster=geo_cluster,
        source_platform=source_platform,
        text_entities=entities,
    )


def _model_id(pipeline: Pipeline) -> str:
    """Embedding backbone id from the injected pipeline (public accessor first)."""
    model_id = getattr(pipeline, "model_id", None)
    if model_id:
        return str(model_id)
    extractor = getattr(pipeline, "_extractor", None)
    return str(getattr(extractor, "model_id", None) or "unknown")


def _index_size(pipeline: Pipeline) -> int:
    """Vector count from the injected pipeline's index (public accessor first)."""
    size = getattr(pipeline, "index_size", None)
    if isinstance(size, int):
        return size
    index = getattr(pipeline, "_index", None)
    size = getattr(index, "size", None)
    return int(size) if isinstance(size, int) else 0
