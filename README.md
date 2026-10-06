# Grey-Resolve

Multimodal entity resolution pipeline for degraded, uncurated media. Fuses face
embeddings with contextual metadata (temporal, geo, text entities) via vector search to
disambiguate identities across low-resolution, off-angle imagery. Includes face-quality
(FIQA) gating, HNSW indexing, and operational FMR/FNMR benchmarks.

> **Research prototype — not for operational use.** This project is a portfolio
> demonstration built on synthetic and public research data. It performs candidate
> *ranking* for media similarity research; it does not identify people, assert identities,
> or make access-control decisions. See [Ethics & limitations](#ethics--limitations).

## Why this exists

Intelligence and defense workflows increasingly operate on *grey-zone* data: uncurated,
low-resolution, off-angle media from open sources. Pure vector similarity degrades badly
under blur, poor lighting, and oblique angles, producing ambiguous candidate sets.
Grey-Resolve studies two engineering responses:

1. **Quality-aware retrieval** — a face image quality (FIQA) gate that flags low-trust
   inputs instead of indexing them with full confidence, and honest FMR/FNMR trade-off
   curves under controlled degradation.
2. **Contextual disambiguation** — a fusion scorer that breaks near-ties in face similarity
   using lightweight metadata (time, coarse geo, source, text entities) on a synthetic
   scenario dataset.

## Status

Early development. Current phase: **Phase 1 — core pipeline** (see [`PLAN.md`](PLAN.md)).

| Phase | Scope | Status |
|---|---|---|
| 1 | Backbone + FIQA gate + FAISS/HNSW + SQLite metadata + degraded-input ROC benchmarks | in progress |
| 2 | Synthetic scenario generator + contextual fusion scorer + ablation | planned |
| 3 | Docker Compose + Qdrant migration path + ONNX/INT8 + latency profiling | stretch |

## Repository structure

See [`PLAN.md`](PLAN.md) for the target layout and [`ARCHITECTURE.md`](ARCHITECTURE.md)
for component and data-flow design. Supporting docs in [`docs/`](docs/):

- [`docs/BACKBONE_LICENSES.md`](docs/BACKBONE_LICENSES.md) — license review gating the
  embedding-backbone choice.
- [`docs/PORT_INVENTORY.md`](docs/PORT_INVENTORY.md) — what is adapted from the earlier
  IronClad coursework and at what effort.
- [`docs/SYNTHETIC_SCENARIO_SPEC.md`](docs/SYNTHETIC_SCENARIO_SPEC.md) — Phase 2 synthetic
  scenario dataset spec.

## Getting started

```bash
uv venv .venv                             # Python 3.12
uv pip install -e ".[dev]"                # core deps + pytest/ruff
uv pip install -e ".[index,backbone,api]" # faiss, insightface/onnxruntime, fastapi
python scripts/fetch_backbone.py          # downloads non-commercial weights at runtime
.venv/Scripts/python -m pytest -q         # run the test suite
```

Weights are downloaded at runtime and never committed (see
`THIRD_PARTY_NOTICES.md`). Datasets with
non-commercial research terms (e.g. LFW, CASIA-WebFace, VGGFace2) are **never** included in this
repository; loaders expect user-supplied local data.

## Ethics & limitations

- **Synthetic and public data only.** No real persons, real incidents, or operational
  intelligence data are used in code, tests, or benchmarks. Phase 2 scenario data is
  entirely fabricated and clearly labeled as such.
- **Ranking, not identification.** The system returns scored similarity candidates.
  Interpreting those candidates as identity claims is out of scope and unsupported.
- **Known failure modes.** Face recognition degrades sharply on demographic subgroups,
  heavy occlusion, and extreme pose; quality gating reduces but does not remove this.
  Synthetic-set results do not transfer to real-world accuracy claims.
- **No surveillance tooling.** The repo deliberately excludes watchlist ingestion, real
  OSINT scraping, and gait/body re-identification.

## License

Code license to be confirmed before the first public release (see
`docs/BACKBONE_LICENSES.md` for the dependency-license review). Pretrained model weights
referenced by this project have their own terms and are not redistributed here.

## Acknowledgment

Adapted from coursework completed for JHU 705.603 Creating AI-Enabled Systems
(IronClad vector-search assignment). Course code informed the design; this repository is
an independent, re-scoped implementation.
