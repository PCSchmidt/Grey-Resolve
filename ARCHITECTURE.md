# Grey-Resolve — Architecture

Research prototype for entity resolution under degraded media. Not for operational use.

## System context

Grey-Resolve ingests uncurated media items (images + lightweight metadata), extracts and
quality-gates face embeddings, indexes them for fast similarity search, and resolves query
media to candidate identities by fusing face similarity with contextual signals.

```
              Raw unstructured media (images + metadata)
                                   |
          +------------------------+------------------------+
          v                                             v
   Face detection & FIQA                        Metadata extraction
   (detector + quality score)                   (timestamp, geo, source,
          |                                      text entities)
          v                                             |
   Dense vector embedding                               |
   (ArcFace/AdaFace 512-d)                              |
          |                                             |
          v                                             v
   FAISS HNSW (cosine)  <---- SQLite metadata sidecar (payloads by vector id)
          |                                             |
          +------------------------+------------------------+
                                   v
                     Fusion scorer (Phase 2)
                 P(identity) = a*S_face + b*S_context
                                   |
                                   v
                    Candidate ranking / resolved cluster
```

## Components

### 1. Detection & quality gating (`src/grey_resolve/detection/`)
- Detect face bounding boxes (lightweight detector; MTCNN-class or SCRFD).
- Compute a face image quality score (FIQA method per `docs/BACKBONE_LICENSES.md` review).
- **Quality gate**: items below the profile threshold are flagged as low-trust rather than
  silently indexed with full weight. Operating profiles (e.g. "strict watch" vs "broad lead")
  live in `configs/threshold_profiles.yaml`.
- Port note: replaces IronClad's raw MTCNN-confidence gate (threshold 0.99 in
  `design_config.json`) with an explicit quality score.

### 2. Embedding extraction (`src/grey_resolve/embeddings/`)
- Angular-margin backbone (ArcFace/AdaFace class) producing 512-d unit-norm embeddings.
- The extractor is a thin adapter interface so the backbone can be swapped without touching
  indexing or API code (this is what makes the license-gated model choice low-risk).
- Batch extraction with deterministic preprocessing (align/crop/resize contract documented
  in code).

### 3. Index & metadata (`src/grey_resolve/index/`)
- **FAISS HNSW** index, cosine metric (equivalent: inner product on normalized vectors).
  Hyperparameters (`M`, `efConstruction`, `efSearch`) in `configs/index_hnsw.yaml`.
  Ported from IronClad's `FaissHNSW` (M=16, efConstruction=40 baseline).
- **SQLite metadata sidecar**: one row per vector id with `timestamp`, `geo_cluster`,
  `source_platform`, `text_entities`, `quality_score`, `ingest_time`. The vector index stays
  pure; all filtering/context lives in the sidecar.
- Persist/restore both together with a version stamp so index and metadata cannot drift apart.

### 4. Fusion scorer (`src/grey_resolve/fusion/`) — Phase 2
- `S_face`: cosine similarity from the ANN search.
- `S_context`: metadata compatibility (temporal proximity, geo cluster match, entity overlap).
- `P = alpha*S_face + beta*S_context` (`alpha`/`beta` per operating profile).
- Ambiguity is defined by the measured **face-score gap** (top-1 vs top-2), not a fixed
  cosine band — the originally planned 0.65-0.75 band is empty for ArcFace on this data.
- Evaluated variants (see `docs/RESULTS.md` findings 6-9): raw fused ranking (harmful in
  ties — extreme-value noise across many candidates), selective gap-gated fusion (still
  harmful), and the **top-2 evidence-only tiebreak** (context decides only the top-2 face
  candidates and only with non-absent context; safe, measured neutral). The context
  hypothesis itself is an honest negative on this data.

### 5. API (`src/grey_resolve/api/`)
- `POST /ingest` — media item (image + metadata) -> detection, quality gate, embed, index.
- `POST /search` — query media -> top-k candidates with face score, context score, fused score.
- `GET /health` — model/index/profile versions.
- Explicit semantic carried over from IronClad's `app.py`: **candidate ranking is not access
  authorization**. Responses are ranked candidates with scores, never identity assertions.

## Data contracts

- **Media item**: image bytes/URI + `timestamp` (ISO 8601, optional), `geo_cluster`
  (coarse label, optional), `source_platform` (label), `text_entities` (list of strings).
- **Vector id**: integer, stable across index rebuilds (stored in SQLite).
- **Scores**: cosine in [-1, 1] for `S_face`; `S_context` in [0, 1]; fused `P` in [0, 1].

## Evaluation pipeline (`benchmarks/`)

- Degradation simulator (`data/synthetic_noise/`): blur, brightness, downsampling, off-angle.
- `evaluate_roc.py`: sweeps decision thresholds -> FMR/FNMR + ROC per degradation type,
  gated vs ungated ablation.
- `latency_profiler.py` (Phase 3): p50/p95 ingest and search latency, index-size sweep.
- All benchmark outputs carry the scenario/config hash that produced them.

## Deployment (Phase 3)

- `docker/docker-compose.yaml`: inference API + Qdrant (metadata as payloads) as the
  "production-like" variant; the FAISS+SQLite path remains the default reference implementation.
- ONNX INT8 export of the embedding backbone: **cut from scope** (see PLAN.md Phase 3);
  recorded as future work.

## Design principles

1. **Adapter seams at every model choice** — backbone, detector, and FIQA are swappable.
2. **Honest evaluation** — synthetic scenario data for fusion; no claims about real-world
   accuracy from synthetic results.
3. **Data hygiene** — no non-commercial dataset in git; loaders require user-supplied local data.
4. **Prototype framing** — no watchlist/authorization semantics anywhere in the API surface.
