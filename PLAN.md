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

### Phase 3 — Stretch

- Docker Compose: inference API + Qdrant (migration path from FAISS justified in docs).
- ONNX INT8 export of the embedding model + accuracy delta report.
- Latency profiler: edge-vs-cloud lookup benchmarks.
- Cut list if time runs short: Qdrant migration and INT8 export; the README still tells the story.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scope creep in multimodal features | Hard phase cuts above; gait/ReID dropped |
| Public framing of a face-matching tool | Research-prototype banner, ethics section, synthetic/public data only, no watchlist language |
| Backbone weight licensing | License review before any weights are referenced (`docs/BACKBONE_LICENSES.md`) |
| Dataset redistribution (course archive is LFW-derived, research-only) | Data never in git; loader expects user-supplied local data; benchmark CSVs checked for derived-data terms; fetch script + checksums only |

## Open decisions

1. Embedding backbone (license review outcome) — gates Phase 1 start.
2. FIQA method: CR-FIQA vs MagFace quality score vs detector-confidence heuristic.
3. API framework: FastAPI (typed, testable) vs staying with Flask to minimize port diff.
4. Metadata store: SQLite (Phase 1) is the default; revisit with Qdrant payloads in Phase 3.

## Repo layout (target)

```text
grey-resolve/
├── configs/
│   ├── index_hnsw.yaml            # HNSW hyperparameters (M, efConstruction)
│   └── threshold_profiles.yaml    # FIQA + decision thresholds per operating profile
├── data/
│   ├── samples/                   # Public/representative test images only
│   └── synthetic_noise/           # Degradation simulation (blur, haze, low-res, off-angle)
├── docs/
│   ├── PLAN.md                    # (this file lives at repo root)
│   ├── ARCHITECTURE.md
│   ├── BACKBONE_LICENSES.md
│   ├── PORT_INVENTORY.md
│   └── SYNTHETIC_SCENARIO_SPEC.md
├── src/grey_resolve/
│   ├── detection/                 # Face detection + FIQA quality gating
│   ├── embeddings/                # Backbone feature extractor
│   ├── index/                     # FAISS wrapper + SQLite metadata sidecar
│   ├── fusion/                    # Multi-signal fusion scorer (Phase 2)
│   └── api/                       # REST service
├── benchmarks/
│   ├── evaluate_roc.py            # FMR/FNMR + ROC under degraded inputs
│   └── latency_profiler.py
├── docker/                        # Phase 3
├── README.md
└── pyproject.toml
```

## Related documents

- `ARCHITECTURE.md` — component and data-flow design.
- `docs/BACKBONE_LICENSES.md` — license review that gates the backbone choice.
- `docs/PORT_INVENTORY.md` — what ports from IronClad and at what effort.
- `docs/SYNTHETIC_SCENARIO_SPEC.md` — Phase 2 scenario generator spec.
