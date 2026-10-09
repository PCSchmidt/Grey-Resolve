# Grey-Resolve — Plan

**Multimodal entity resolution for degraded, uncurated media.**
Research prototype. Not for operational use.

This plan converts the completed IronClad coursework (face retrieval over an LFW-derived
gallery/probe split) into a defense-framed portfolio project: resolve an identity across
low-quality, off-angle, uncurated media by fusing face-embedding similarity with
contextual metadata.

---

## Goals

1. **Core retrieval pipeline (Phase 1)** — modern face backbone (ArcFace/AdaFace class),
   real face-quality (FIQA) gating, FAISS HNSW index with a metadata sidecar, clean REST API.
2. **Degraded-input evaluation (Phase 1)** — operational benchmarking: FMR/FNMR and ROC
   curves under blur, low resolution, brightness, and off-angle degradation. This is the
   "defense-grade" evaluation story and reuses the IronClad analysis harness.
3. **Contextual fusion (Phase 2)** — the differentiator: `P(identity) = alpha*S_face + beta*S_context`
   combining face similarity with temporal/geo/text signals on a **synthetic scenario dataset**.
4. **Deployment story (Phase 3, stretch)** — containerized services, vector-DB upgrade path,
   quantized inference, latency profiling.

## Non-goals (explicit cuts)

- No real-world OSINT scraping (no Telegram/WeChat/messaging ingestion).
- No gait / body-shape ReID — mentioned as future work only.
- No operational watchlist screening language in code, APIs, or docs. The technical framing
  is "entity resolution under degraded media"; the defense motivation lives in the README only.
- No real persons or real intelligence data anywhere in the repo or benchmarks.

## Phases

### Phase 1 — Core pipeline (high confidence; mostly a refactor of proven code)

| Work item | Notes |
|---|---|
| Backbones: ArcFace/AdaFace embedding extractor | Choice gated by license review, see `docs/BACKBONE_LICENSES.md` |
| FIQA gate replaces MTCNN-confidence gate | Score-based quality gating with a configurable threshold (`configs/threshold_profiles.yaml`) |
| FAISS HNSW (cosine) + SQLite metadata sidecar | Port of IronClad `FaissHNSW`; metadata (timestamp, geo cluster, source, text entities) in SQLite, not in the index |
| REST API (FastAPI recommended over Flask) | `/ingest`, `/search`, `/health`; explicit "ranking is not authorization" semantics carried over from IronClad |
| Degradation benchmark harness | Port `analysis/` probes; extend task4 CMC -> FMR/FNMR + ROC; record latency |

Acceptance criteria:
- Ingest + search runs end-to-end on a small public/synthetic gallery.
- ROC/FMR/FNMR curves produced for >= 4 degradation types, reproducible from one command.
- Quality gate demonstrably improves FMR at fixed FNMR vs. ungated baseline.
- No dataset with non-commercial terms is tracked in git.

### Phase 2 — Contextual fusion (the novel contribution)

- Synthetic scenario generator: gallery identities with fabricated timestamps, geo tags,
  platform sources, and text snippets (see `docs/SYNTHETIC_SCENARIO_SPEC.md`).
- Fusion scorer `P = alpha*S_face + beta*S_context`, calibrated on the synthetic scenario set.
- Ambiguity analysis: what happens when several faces score 0.65-0.75 and context breaks the tie.
- Honest evaluation: ablation (face-only vs fused) on the synthetic set only; no claims about
  real-world accuracy.

Acceptance criteria:
- Generator produces a versioned scenario dataset from one command with a fixed seed.
- Ablation table shows fusion benefit (or honestly reports it does not).
- Every metric traceable to a scenario config (reproducible).

### Phase 3 — Presentation (completed 2026-10-07)

- ~~Docker Compose: inference API + Qdrant~~ — **done**: `docker/` (API + Qdrant
  services, weights never baked into images) and `QdrantVectorIndex` with local-mode
  tests; `GREY_RESOLVE_INDEX=qdrant` switches the serve entrypoint.
- ~~Latency profiler~~ — **done**: `benchmarks/latency_profiler.py` (index-size sweep
  HNSW vs brute force + pipeline mode). HNSW p95 <= 1.6 ms at 50k vectors vs brute
  force ~29 ms.
- ~~Figures~~ — **done** (delivered instead of the cut item): `benchmarks/make_figures.py`
  renders ROC/EER/gate/fusion figures into `docs/figures/` from committed run artifacts;
  sanitized aggregate metrics in `docs/results/`.
- **Cut as planned:** ONNX INT8 export + accuracy delta (kept as future work in the
  README).

### Phase 4 — Demo UI (added 2026-10-07, completed)

- **Resolution Console** (`docs/index.html` + `docs/demo/`): a "forbidden cockpit"
  single-page demo on GitHub Pages — streaming synthetic contacts with trust badges,
  3D embedding manifold with tie-break resolution sequences, live degradation modes
  (the project's own operators), telemetry from sanitized real runs.
- Resolver shows confirmed / correct flip / wrong flip / absent outcomes with truthful
  fused bars; the feed queues ties so each verdict completes (see `docs/PAGES.md`).
- Constraints held: zero face imagery (procedural tiles only), persistent synthetic
  banner, no identification language, no build step (vanilla ESM + three.js CDN).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scope creep in multimodal features | Hard phase cuts above; gait/ReID dropped |
| Public framing of a face-matching tool | Research-prototype banner, ethics section, synthetic/public data only, no watchlist language |
| Backbone weight licensing | License review before any weights are referenced (`docs/BACKBONE_LICENSES.md`) |
| Dataset redistribution (course archive is LFW-derived, research-only) | Data never in git; loader expects user-supplied local data; benchmark CSVs checked for derived-data terms; fetch script + checksums only |

## Decisions (recorded 2026-10-06)

1. **Embedding backbone — InsightFace `buffalo_l` (w600k_r50 ArcFace, 512-d).** Chosen on
   weights-license clarity: the only candidate with an explicit signed non-commercial grant
   (`MODEL.LICENSE`). Runtime-downloaded weights; never committed. Fallback: AdaFace
   IR-50 WebFace4M (MIT code, best on low-quality benchmarks like IJB-S/TinyFace; note BGR
   input convention and unresolved weight terms). Full analysis: `docs/BACKBONE_LICENSES.md`.
2. **FIQA — MagFace feature magnitude (Apache-2.0 code) as primary**; AdaFace feature norm
   as zero-cost fallback if the backbone swap makes it free. CR-FIQA only as a cited
   benchmark baseline (no license on its code/weights — do not vendor).
3. **API framework: FastAPI** (typed request/response models, testable, auto OpenAPI docs).
   The IronClad Flask diff is acceptable because `app.py` is thin anyway per
   `docs/PORT_INVENTORY.md`.
4. **Metadata store: SQLite (Phase 1)**; revisit with Qdrant payloads in Phase 3.

## Open decisions

1. ~~Quality gate implementation~~ — **decided (2026-10-06):** ship the dependency-free
   heuristic FIQA-lite assessor (`HeuristicQualityAssessor`) behind the `QualityAssessor`
   seam now, so the pipeline runs end-to-end without extra weights; add the MagFace
   feature-magnitude assessor when its weights are fetched, and measure its extra
   forward-pass cost then. AdaFace feature-norm stays the zero-cost fallback.
2. ~~Code license~~ — **decided (2026-10-06): MIT.** Standard for portfolio repos,
   compatible with every dependency. Model-weight terms are unaffected (non-commercial
   grants travel with the weights, not with this code).

## Repo layout (as built)

```text
grey-resolve/
├── configs/                       # index_hnsw.yaml · threshold_profiles.yaml · scenario_v0.yaml
├── docs/
│   ├── RESULTS.md                 # findings 1-9 with numbers and caveats
│   ├── figures/ · results/        # generated figures · sanitized aggregate metrics
│   ├── index.html + demo/         # the Resolution Console (GitHub Pages)
│   ├── PAGES.md · BACKBONE_LICENSES.md · PORT_INVENTORY.md · SYNTHETIC_SCENARIO_SPEC.md
├── src/grey_resolve/
│   ├── detection/                 # SCRFD adapter + FIQA-lite quality gate
│   ├── embeddings/                # ArcFace extractor (runtime weights)
│   ├── index/                     # FAISS HNSW + SQLite sidecar + brute-force oracle + Qdrant
│   ├── fusion/                    # linear / selective / tiebreak scorers
│   ├── scenario/                  # synthetic persona + context generator
│   ├── evaluation/                # metrics · degradation sweeps · fusion ablation · latency
│   ├── plotting/                  # figure rendering
│   ├── pipeline.py                # ingest + search orchestration
│   └── api/                       # FastAPI service
├── benchmarks/                    # evaluate_roc · evaluate_fusion · latency_profiler · make_figures
├── docker/                        # Dockerfile · compose (API + Qdrant) · serve entrypoint
├── scripts/                       # fetch_backbone.py + demo/diagnostic tooling
├── tests/                         # 241 deterministic tests
├── README.md · LICENSE · THIRD_PARTY_NOTICES.md · pyproject.toml
```

## Related documents

- `ARCHITECTURE.md` — component and data-flow design.
- `docs/BACKBONE_LICENSES.md` — license review that gates the backbone choice.
- `docs/PORT_INVENTORY.md` — what ports from IronClad and at what effort.
- `docs/SYNTHETIC_SCENARIO_SPEC.md` — Phase 2 scenario generator spec.
