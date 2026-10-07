# Grey-Resolve — Docker deployment (Phase 3)

Two services (`docker-compose.yaml`):

- **grey-resolve-api** — the FastAPI inference service (`/ingest`, `/search`, `/health`)
  built from the multi-stage `Dockerfile` in this directory.
- **qdrant** — the official `qdrant/qdrant` vector database (port 6333), optional
  migration target for the vector index.

## Run

1. Fetch the backbone weights once on the host (or any machine that may hold them):

   ```bash
   python scripts/fetch_backbone.py --root ~/.insightface
   ```

2. Start the stack (build context is the repo root):

   ```bash
   docker compose -f docker/docker-compose.yaml up --build
   ```

3. Open `http://localhost:8000/health` (or `/docs` for the OpenAPI UI).

## Weights expectations

The image **never contains model weights**. The InsightFace `buffalo_l` weights
(`det_10g.onnx`, `w600k_r50.onnx`) carry a non-commercial grant
(`docs/BACKBONE_LICENSES.md`) and must not be redistributed, so they are not
fetched at build time and not copied into any layer. Instead:

- compose mounts `~/.insightface` read-only at `/home/app/.insightface`;
- the entrypoint (`docker/serve.py`) checks for both ONNX files at startup and
  exits with a clear error telling you to mount them or run `scripts/fetch_backbone.py`;
- `GREY_RESOLVE_INSIGHTFACE_ROOT` overrides the mount location.

## What is and is not in the image

In: project source (`src/`), `configs/`, the `fetch_backbone.py` script, and the
`[index,backbone,api,qdrant]` dependencies (FAISS, insightface/onnxruntime,
FastAPI/uvicorn, qdrant-client).
Never in: backbone weights, datasets, gallery data, or any user media. Runtime
data lives on volumes only (`/data` for the FAISS snapshot + SQLite sidecar,
`/qdrant/storage` inside the qdrant service).

## FAISS → Qdrant migration story

FAISS HNSW + SQLite remains the default reference index; `GREY_RESOLVE_INDEX=qdrant`
switches the vector index to `QdrantVectorIndex` pointing at the `qdrant` service
(same `VectorIndex` contract, same cosine scores). SQLite metadata stays a sidecar
for now; moving payloads into Qdrant is the next migration step. Snapshots of either
index are portable JSON exports (`save()`/`load()`), so migrating data is
export from one, load into the other.

## Environment variables (docker/serve.py)

| Variable | Default | Meaning |
|---|---|---|
| `GREY_RESOLVE_INDEX` | `faiss` | `faiss` or `qdrant` vector index |
| `GREY_RESOLVE_PROFILE` | `broad_lead` | threshold profile from `configs/threshold_profiles.yaml` |
| `GREY_RESOLVE_STORAGE` | `/data` | FAISS snapshot + SQLite sidecar directory |
| `GREY_RESOLVE_INSIGHTFACE_ROOT` | `~/.insightface` | weights root (`models/*.onnx`) |
| `GREY_RESOLVE_CONFIG_DIR` | `configs` | YAML config directory |
| `GREY_RESOLVE_QDRANT_URL` | `http://qdrant:6333` | Qdrant endpoint (`GREY_RESOLVE_INDEX=qdrant`) |
| `GREY_RESOLVE_QDRANT_API_KEY` | *(unset)* | Qdrant API key (optional) |
| `GREY_RESOLVE_QDRANT_COLLECTION` | `grey-resolve` | Qdrant collection name |
| `GREY_RESOLVE_HOST` / `GREY_RESOLVE_PORT` | `0.0.0.0` / `8000` | bind address |
