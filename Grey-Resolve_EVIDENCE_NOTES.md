# Grey-Resolve: Evidence Notes

Companion to `Grey-Resolve.md` (the Master Explainer). Questionnaire v2.8, FULL depth. Run date 2026-10-24. Repository: fresh clone of GitHub `main`, HEAD `c71e76b`, no `.env`. Paths are repository-relative. Confidence labels: **Confirmed**, **Ran it**, **Per the docs**, **My reading**, **Ask the author**, **Author says**.

---

## E1. Fact sheet (full)

| Fact | Value | Label |
| --- | --- | --- |
| Name, address, licence | Grey-Resolve; https://github.com/PCSchmidt/Grey-Resolve; MIT code (`LICENSE`); weights non-commercial (`THIRD_PARTY_NOTICES.md`, `docs/BACKBONE_LICENSES.md`) | Confirmed |
| GitHub description | "Multimodal entity resolution pipeline for grey-zone OSINT and battlefield intelligence. Fuses ArcFace biometric embeddings with contextual metadata (geo, temporal, NER) ... disambiguate Persons of Interest (POIs)..." (`gh api repos/PCSchmidt/Grey-Resolve`) | Confirmed |
| Kind, status | Research prototype: test bench + FastAPI service + static demo. PLAN.md marks Phase 3 "completed 2026-10-07" and Phase 4 "completed"; README treats Phases 1-2 as done | Confirmed |
| Size | 63 Python files. Lines (tracked files per folder, counted with `wc -l`): src 5,582; tests 3,992; benchmarks 1,759; scripts 372; docker 321. Demo JS 2,576 lines (506+367+254+939+510) | Ran it |
| Commits | 41 (`git log`); first `d4f6876` 2026-10-06 11:19 -0400; last `c71e76b` 2026-10-19 09:44; 40 code/doc commits 2026-10-06 to 2026-10-09 | Confirmed |
| Pack size | 14.86 MiB (hero GIF 9.4 MB of it) | Ran it |
| Tests | 241 collected; see E7 | Ran it |
| Lint | `ruff check` (0.17.2, default rules, no config beyond line length): 43 findings (28 auto-fixable), 17 in `src/`; `ruff format --check`: 36 files would be reformatted. No formatter is claimed by the project | Ran it |
| Runs from a clean copy | With the README's three install commands equivalent (extras index, backbone, api, figures, qdrant plus dev tools): 12 of 241 fail (API tests: "Form data requires python-multipart"). After `uv pip install python-multipart`: 241 pass. `scipy` is imported by `degradation/operators.py` but absent from `pyproject.toml`; it arrived via insightface | Ran it |
| Needs | Python >= 3.12; weights ~289 MB pack (`docs/BACKBONE_LICENSES.md`); a private gallery for the evidence runs; Docker optional | Confirmed |
| CI | No `.github/`. `gh api .../actions/workflows`: one workflow, `pages-build-deployment` (GitHub's automatic Pages build); 6 runs | Confirmed |
| Deployed | Pages from branch main, folder /docs (`gh api .../pages`: status built, source main /docs). URL returns 200; `index.html`, 5 JS files, CSS, GIF and a results JSON are byte-identical to the repo ignoring CR line endings; headless Chrome screenshot shows banner, feed, 3D manifold, resolver panel, degradation bench, telemetry ticker. The REST service and Docker stack are not deployed | Ran it |
| Users | GitHub: 0 stars, 0 forks, 0 open issues, no releases; 2 views / 2 unique in the 14 days to 2026-10-23 (`traffic/views`); security: secret scanning and push protection enabled, Dependabot updates disabled, `main` not protected | Confirmed |
| Author statements | ML engineer, 5.5 years; targets AI Engineer, Forward Deployed AI Engineer; very little code hand-written; defense-sector topics of interest | Author says |
| Headline claims | 38 (E3) | Confirmed |

---

## E2. Reading and running log

**Read:** README, PLAN, ARCHITECTURE, THIRD_PARTY_NOTICES, `docs/RESULTS.md`, `PAGES.md`, `BACKBONE_LICENSES.md` (head), `PORT_INVENTORY.md` (head), `SYNTHETIC_SCENARIO_SPEC.md` (key sections and first-commit version), `benchmarks/README.md`, `docker/README.md`; all of `src/grey_resolve/` except most of `plotting/`, `index/qdrant_index.py` (read `save`/`load` only) and `index/bruteforce.py`; `evaluation/experiment.py` and `fusion_experiment.py` (core functions fully; sweep helpers skimmed); `docker/serve.py`, `Dockerfile`, compose; `scripts/fetch_backbone.py`, `_fusion_runs_big.py`; selected tests; `docs/results/*.json` (parsed); demo `data.js` (head, fusion rule), `panels.js` (grep); full git log with stats and push events. The old `Grey-Resolve_MASTER_EXPLAINER.md` was used for leads only (section headings and Discrepancies list); each lead re-verified.

**Not read:** `plotting/curves.py`, `benchmarks/make_figures.py` and `latency_profiler.py` bodies, `manifold.js` and most demo JS, figures (PNG), `docs/PORT_INVENTORY.md` beyond the head.

**Commands run (selected)**
- Environment: `uv venv --python 3.12 venv` in the scratchpad (outside the clone); `uv pip install numpy pyyaml pytest ruff faiss-cpu fastapi uvicorn pydantic matplotlib qdrant-client opencv-python-headless onnxruntime insightface python-multipart coverage`. Run with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src` from the clone root, `-p no:cacheprovider`, `COVERAGE_FILE` in the scratchpad.
- `python -m pytest -p no:cacheprovider -q`: first run 12 failed / 229 passed (37.8 s); after python-multipart 241 passed (11.7 s, 14.2 s under coverage). `-rs` shows no skips.
- `ruff check --no-cache`, `ruff format --check --no-cache` (check-only).
- Probe scripts (scratchpad, write nothing in the clone): rule attacks, API attacks, restart/overwrite, FAISS-to-Qdrant load, real-model service build, context-discrimination probe, speed and recall probe, sample rows, metric toy examples, Wilson ranges.
- Read-only HTTP: GET of the live page and its files; GET of the three.js CDN file (status only); `gh api` calls listed in E1; headless Chrome (system install) loading the live page once with a 15 s virtual-time budget.
- Secret scan and history scan (E11).

**What touched the outside world:** the HTTP and `gh` reads above; headless Chrome loaded the live page (which itself fetches the three.js file from a public CDN). No paid service, no download of model weights (they were already present locally, in the user's home model folder), no Docker, no deploy, no push.

**Not run (and why):** Docker build and compose (Docker skipped under 0.2); `benchmarks/*.py` (need the private gallery; `latency_profiler.py` and `make_figures.py` write into committed `docs/` folders); `scripts/_capture_hero.py` (browser automation, writes GIF).

**Final git status of the clone:** `git status --short` printed nothing (no tracked changes, no untracked files) after every run, including the last one after coverage. Python bytecode and caches were not written (`PYTHONDONTWRITEBYTECODE`, `-p no:cacheprovider`).

---

## E3. Claims check

Levels: **L1** document matches committed result file; **L2** reproduced by re-running; **L3** independently recomputed. Where a claim has several levels, the lowest is reported. "Cannot re-run" = needs the private gallery.

| ID | Claim (where) | Evidence | Level | Verdict |
| --- | --- | --- | --- | --- |
| CL-01 | Live demo at the Pages URL (README top) | HTTP 200; files identical; headless Chrome render | L2 | Supported (first screen only) |
| CL-02 | Demo telemetry is "real metric telemetry" (README, PAGES.md) | `docs/demo/data.js:255-256` fetches `results/20261006T191330Z.json` and `results/20261007T111729Z.json`; ticker shows "TESTS 241 ... RUN 20261006T191330Z ... EER 0.000" | L1 | Supported |
| CL-03 | "241 green (pytest)" (README) | 241 pass only after installing undeclared `python-multipart`; 12 fail otherwise | L2 | Partly |
| CL-04 | Tests "deterministic, no weights needed" | 3 smoke tests skip without weights (`tests/test_backbone_smoke.py:22`); others use fakes; two full runs gave identical counts | L2 | Supported |
| CL-05 | Sweep: EER about 0 until severe; blur 8 EER 0.078; low-res 0.1 EER 0.046, FMR@FNMR1% 0.873 (README, RESULTS) | `docs/results/20261006T191330Z.json` sweep: 0.0782 / 0.0460 / 0.873; brightness 0.75 FMR 0.14948 (write-up says 0.150) | L1 (cannot re-run) | Supported (rounding slip in one cell) |
| CL-06 | 42/138 detection failures at brightness 0.75 | `n_scored` 96 of 138 | L1 | Supported |
| CL-07 | Gate: FNMR 0.028 to 0.020 on clean at 0.5 | JSON: full 0.02817 (6 of 213 genuine), gated 0.02010 (4 of 199) | L1/L3 counts | Supported |
| CL-08 | Gate lowers FNMR in nearly every condition; blur 3: 0.127 to 0.000 | JSON agrees; blur-3 gated rests on 12 probes (14 genuine pairs); brightness 0.5 on 5 probes | L1 | Partly (emphasis outruns n) |
| CL-09 | Gate "never materially worsens" | Omitted row: low-res 0.1 full 0.989, gated 1.000 on 6 probes; tilt 45: 0.047 to 0.049 (disclosed) | L1 | Partly |
| CL-10 | FMR = 0 in every cell (Finding 5) | All `fmr` fields 0.0 in `gated_vs_ungated` | L1 | Supported |
| CL-11 | Rotation: pilot (50 images) zero effect; 100 identities: 45 deg EER 0.014, FMR 0.256 | Pilot `20261006T175659Z.json` 0.0 at 45; main run 0.014 / 0.256 | L1 | Supported |
| CL-12 | Finding 3: "landmark re-alignment neutralizes much of 2D rotation" | `degradation/operators.py:132-198`: off_angle is a perspective squeeze (cos factor on one edge), not an in-plane rotation | Code read | Contradicted (explanation) |
| CL-13 | Raw fusion 0.500 to 0.167 hit@1 at gap <= 0.05 (n=18) | `20261007T125746Z.json` gap_le 0.05: 0.5 and 0.1667, n 18 (9/18, 3/18) | L1 | Supported |
| CL-14 | Selective gating reduces but does not remove damage (overall -0.057) | selective 0.849 vs 0.906 | L1 | Supported |
| CL-15 | Tiebreak safe (+/-0.02), neutral | Overall 0.906; slices 0.444/0.630/0.848; noise-sweep deltas +0.009 to -0.019 | L1 | Supported |
| CL-16 | Tiebreak "provably safe" (README:50, RESULTS.md:232) | No proof in repo; measured -1 of 18; demo shows scripted wrong flips; tests only on tiny constructed cases | Code read | Unsupported |
| CL-17 | n=4 positive did not replicate at n=18 (reversed) | `20261007T012503Z.json` (n=4: 0.5 to 0.75) vs `111729Z` (n=18: 0.5 to 0.167); different persona counts (30 vs 60) | L1 | Supported (not an exact repeat) |
| CL-18 | Only ~55% / 52% of queries carry clean context; clean context beats 89% of rival contexts (Findings 8, 9) | Config arithmetic 0.8 x 0.85 x 0.8 = 0.544 matches; my probe over the same generator: 91% of rivals beaten on average (34 queries with context on both sides); diagnostics script outputs not committed | L3-style probe | Supported (approximately) |
| CL-19 | Planned band 0.65-0.75 empty; cross-persona cosines top near 0.16; band resolved at 98th quantile | JSON config `near_tie_band` [0.1276, 1.0], quantile 0.98 | L1 | Supported |
| CL-20 | HNSW p95 <= 1.6 ms at 50k vs ~29 ms exact (README) | `latency-20261007T143030Z.json`: HNSW 50k p95 1.02/1.20/1.44 ms; exact 21.6/21.0/29.3 ms. My re-run (40 queries, k=10): HNSW p50 0.39 ms, exact 2.32 ms | L1; L2 direction only | Partly (accuracy unstated) |
| CL-21 | "recall 0.99 vs exact in our tests" (README glossary) | Test only asserts >= 0.9 (`tests/test_index.py:51-76`); my run on a same-style set (2,000 low-rank vectors): 0.996; 50,000 low-rank: 0.884; 50,000 random (the speed test's data): 0.125 | L2 | Partly |
| CL-22 | Docker Compose (API + Qdrant, weights via volume) | Files present; no build, no run | None | Not checked |
| CL-23 | "tested FAISS to Qdrant migration path" (README table); "snapshots of either index are portable JSON exports ... export from one, load into the other" (`docker/README.md:52`) | FAISS snapshot = `index.faiss`, `ids.json`, `meta.json`; Qdrant `load` rejects `kind != "qdrant"` (`qdrant_index.py:241-242`); I ran FAISS save then Qdrant load: ValueError. `tests/test_qdrant.py` tests Qdrant in memory only | L2 | Contradicted |
| CL-24 | "Persist/restore both together with a version stamp so index and metadata cannot drift" (ARCHITECTURE.md) | Only the index has a stamp; the service never calls `save` (grep); notes persist separately; restart overwrite reproduced (E4) | L2 | Contradicted |
| CL-25 | "All benchmark outputs carry the scenario/config hash" (ARCHITECTURE.md) | Fusion and latency JSONs carry hashes; ROC JSON `config` has none | L1 | Partly |
| CL-26 | No non-commercial dataset or weights tracked | `git ls-files` shows none; `.gitignore` covers `*.onnx`, `data/`; history scan found no weights | Confirmed | Supported |
| CL-27 | No face imagery in repo or UI | Tracked images are charts and the demo GIF; screenshot shows procedural tiles | Confirmed | Supported |
| CL-28 | GitHub description: "NER", "Persons of Interest", "battlefield intelligence" | No NER code anywhere (grep); README/PLAN say no watch-list language and no real OSINT | Confirmed | Contradicted |
| CL-29 | Phase 3 completed 2026-10-07; INT8 export cut | PLAN.md and commits agree | Confirmed | Supported |
| CL-30 | Findings "reproducible from committed scripts" | Commands saved (`scripts/_fusion_runs*.py`) and seed 42; the 263-photo gallery is outside the repo | Confirmed | Partly |
| CL-31 | Close-out run id `557189Z` (RESULTS.md:199, `data.js:29,155`, `panels.js:409-410`) | No such artefact; numbers match `20261007T125746Z.json` | L1 | Contradicted (label) |
| CL-32 | "Detection misses abort the run" (`benchmarks/README.md:75`) | `experiment.py` skip-and-count (`_safe_condition_result`, lines 388-425) | Code read | Contradicted (stale) |
| CL-33 | PLAN acceptance: "quality gate demonstrably improves FMR at fixed FNMR" | FMR is 0 everywhere; disclosed as Finding 5 | L1 | Unsupported (unmet, disclosed) |
| CL-34 | PLAN risk table: "fetch script + checksums only" | `scripts/fetch_backbone.py` has no checksum step | Code read | Contradicted |
| CL-35 | Context uses "time, coarse geo, source, text entities" (README) | `LinearFusionScorer.context_score` (`fusion/linear.py:87-107`) uses geo, time, entities only; `source_platform` unused | Code read | Partly |
| CL-36 | Figures regenerate from committed run artefacts (`make_figures.py`) | Not run (writes into `docs/`) | None | Not checked |
| CL-37 | Headless run reached all 8 verdicts in two feed cycles (PAGES.md) | I rendered once; did not run the long replay | None | Not checked |
| CL-38 | "FastAPI service ... over the real pipeline" | Tests use fakes; I built the real pipeline via `docker/serve.py:build_pipeline` with local weights and called `/health`, `/ingest`, `/search` in-process on a noise image: empty results | L2 (no real face) | Partly |

Totals: 17 supported, 9 partly, 7 contradicted, 2 unsupported, 3 not checked (38).

---

## E4. Rule attacks

Scripts in the scratchpad; each ran against the clone with `PYTHONPATH=src`. Output verbatim summary.

| Rule | Attack | Result |
| --- | --- | --- |
| Rejected face never stored | Gate scores 0.6 / 0.5999 / 0.3 / 0.2999 / 0.0 / NaN / 1.0 with `quality_min` 0.6 | full / low / low / reject / reject / reject / full. Held (`detection/base.py:56-61`) |
| Unique vector IDs | FAISS `add` with a duplicate in the batch, duplicate against stored, float ids, wrong dim, zero vector | All five raised `ValueError` with clear messages. Held |
| Index inputs | Search empty index; k=0; zero query; k larger than size | `[]`; `[]`; `ValueError`; returns all (2). Held |
| Blend weights valid | alpha = beta = 0; alpha < 0; reversed band; NaN face score | All `ValueError`. Held |
| Context symmetric and bounded | Both contexts empty | 0.0. Held |
| Missing context is no evidence | Tiebreak with absent query context | Held (test `tests/test_fusion_experiment.py:616`). **Hole:** candidate with no context: face 0.80 scores 0.56 vs face 0.75 with full match 0.825, so a swap follows (`fusion/linear.py:93-107` gives 0.0 when no signal is shared) |
| Time handling | Query date naive vs stored date aware (pipeline and via API with a fusion scorer attached) | `TypeError: can't subtract offset-naive and offset-aware datetimes`; API returns 500. **Hole** (only reachable with a scorer attached; `docker/serve.py:107` passes `fusion=None`) |
| Consistent storing | Store raises after the index add | `RuntimeError` propagates; index size 1, notes empty; next search returns the match with `media_id ''`; id counter consumed. **Hole** (`pipeline.py:91-92`: index before notes, no undo) |
| Numbering after restart | New empty index plus the same notes file, `start_id=0` (what `docker/serve.py:129-135` yields with no snapshot) | Notes row 0 changed from "photo-A" to "photo-C" after one ingest (upsert, `sqlite_store.py:38`). **Hole**; with the Qdrant path `start_id = max + 1` avoids it |
| Service never saves index | `grep "save("` in `src`, `docker`, `scripts`, `benchmarks` | No caller. Index lost on restart in FAISS mode |
| Same media twice | Ingest "same" twice | Two vectors, both returned at score 1.0; no de-duplication (neutral) |
| Metric inputs | All-genuine; NaN score; constant scores; perfect separation; zero vector; one row; target FNMR 1.5 | `ValueError`; `ValueError`; EER 0.5; EER 0.0; `ValueError`; `ValueError`; `ValueError`. Held |
| Damagers | Float image; sigma -1 or NaN; scale 0; angle 90; delta 2; empty image; mutation; determinism | All invalid inputs raised `ValueError`; input not mutated; same output twice. Held. Scale 0.01 on 20 by 20 gave a valid image |
| Quality formula | Flat grey 100 px face; box outside image; 10 px face; random noise; all white | 0.499; 0.0; 0.280; 0.999; 0.25. **Weakness:** noise rated almost perfect |
| API inputs | Empty upload; text file as image; k=100000; bad timestamp; 1 MB geo label | 400; 400; 200 (capped at 100 by `_MAX_K`); 422; 200 (no size limit) |
| Real models in the service | `build_pipeline()` with local weights, `/health`, `/ingest` and `/search` on a random-noise image | Health shows model `insightface/w600k_r50`, index 0, profile `broad_lead`; ingest `faces: []`; search `candidates: []`; `pipeline._fusion` is `None` |
| FAISS to Qdrant | FAISS `save` then Qdrant `load` | `ValueError: not a QdrantVectorIndex snapshot: 'faiss_hnsw'` |
| Gated score FNMR with few probes | Read gate rows | Brightness 0.5: 5 probes; low-res 0.1: 6 probes all failing (see E3 CL-09) |

Context-discrimination probe (mine, not the project's): generator with 60 personas and 2 items each, seed 42 (`configs/scenario_v0.yaml` overrides): 20 distinct (place, period) cells, exactly 3 personas per cell; 27 of 120 items without context; among 34 personas with context on both items, the true persona's context beat on average 91.2% of rivals strictly (0 ties); the earlier duel figures (59/31/10) mixed populations and are corrected in E18. One seed, not a project claim.

---

## E5. Code map (detail)

| Area | File and key lines | Role |
| --- | --- | --- |
| Data shapes | `types.py` (`EMBEDDING_DIM = 512`, `ContextMetadata`, `MediaItem`, `FaceObservation`, `SearchResult`) | Contracts |
| Profiles | `config.py:14-59`; `configs/threshold_profiles.yaml` (strict_surveillance: quality_min 0.60, alpha 0.7, beta 0.3; broad_lead: 0.35, 0.5, 0.5) | Settings |
| Ingest | `pipeline.py:75-94` (detect, assess, decide, embed, `index.add` line 91, `store.put` line 92) | Store a photo |
| Search | `pipeline.py:103-149` (best face line 114; fused sort key line 138) | Look up a photo |
| Quality formula | `detection/quality.py:37-61` (weights line 60) | FIQA-lite |
| Gate | `detection/base.py:46-61` | Three-way decision |
| Face finder | `detection/scrfd.py:21-57` (threshold 0.5, 640 by 640) | SCRFD wrapper |
| Fingerprint | `embeddings/insightface_arcface.py:29-79` (aligned 112 px; bbox fallback line 65-79) | ArcFace wrapper |
| HNSW | `index/faiss_index.py:26-117` (M 16, efC 40, efS 64; `efSearch = max(ef, k)` line 110) | Store and search |
| Exact | `index/bruteforce.py` | Oracle |
| Qdrant | `index/qdrant_index.py` (`_KIND = "qdrant"` line 45; save 202-227; load 229+) | Alternative store |
| Notes | `index/sqlite_store.py:54-154` (upsert line 33-46; `check_same_thread=False` line 72) | Notes table |
| Context score and blend | `fusion/linear.py:33-138` (weights 0.4/0.3/0.3, half-life 30 days lines 27-30; context_score 87-107; fuse 109-123) | Fusion |
| Ranking rules | `evaluation/fusion_experiment.py`: `_prepare` 308-372, `_rank_personas` 375-398, `_rank_selective` 401-417, `_has_context` 420-424, `_rank_tiebreak` 427-459, `run_ambiguity_ablation` 557-680, `run_context_noise_sweep` 708, `run_context_weight_sweep` 854 | Study |
| Degraded sweep | `evaluation/experiment.py`: `degrade_embeddings` 265, `probe_gallery_scores` 316-352, `_safe_condition_result` 388-425, `gated_vs_ungated` 503-614 | Study |
| Metrics | `evaluation/metrics.py`: `build_pairs` 66-96, `fmr_fnmr_at_threshold` 99-120, `roc_curve` 123-159, `eer` 162-181, `fmr_at_fnmr` 184-213 | Maths |
| Damagers | `degradation/operators.py:59-198` | Image damage |
| Scenario | `scenario/generator.py`: `generate_identities` 469-512, `fabricate_media` 535-679, `select_near_ties` 713-802, `write_scenario` 808-853 | Made-up people |
| Service | `api/app.py:55-153` (lock line 66; `_MAX_K` 46); `docker/serve.py:62-171` | Web doors |
| Drivers | `benchmarks/evaluate_roc.py` (gate threshold default 0.5, severity grid lines 44-49), `evaluate_fusion.py`, `latency_profiler.py` (random unit vectors lines 74-80), `make_figures.py` | Studies |

Entry points: `docker/serve.py` (service), `benchmarks/*.py`, `docs/index.html`. Dependency lists: `pyproject.toml` (lower bounds only; core: numpy, pyyaml).

---

## E6. Model call sites

| Where | Model | Purpose | Settings | Cost |
| --- | --- | --- | --- | --- |
| `detection/scrfd.py:44` (`detect_raw`), called from `pipeline.py:82,111` and `benchmarks/evaluate_roc.py`, `evaluate_fusion.py`, `latency_profiler.py` | SCRFD `det_10g.onnx` via insightface model zoo and onnxruntime, CPU (`ctx_id=-1`) | Find faces and 5 landmarks | `det_thresh` 0.5, input 640 by 640 | Free; ~45 ms per photo in a 2-sample smoke run (`latency-20261007T142956Z.json`) |
| `embeddings/insightface_arcface.py:58-63`, called from `pipeline.py:88,115` | ArcFace `w600k_r50.onnx`, CPU | 512-number fingerprint from a 112 px aligned crop | `norm_crop` alignment | Free |
| `detection/quality.py` | None (formula) | Quality mark | Weights 0.5/0.25/0.25; blur reference 500; min face 40 px | None |

No language model, no prompt, no paid API anywhere (`grep -ri "openai\|anthropic\|openrouter"` finds nothing in `src`, `docker`, `benchmarks`). Weights are fetched by `scripts/fetch_backbone.py` from the InsightFace release page, no checksum.

---

## E7. Tests and CI

**Counts (collected, run):** api 12, backbone smoke 3, config 4, degradation 31, experiment 34, fusion 18, fusion_experiment 36, index 13, latency 13, metrics 19, pipeline 7, plotting 12, qdrant 10, quality 4, scenario 25 = 241. Passing: 241 (after python-multipart). Skipped: 0 here because weights exist locally.

**What each kind touches outside the process:** none uses the network, Docker, a graphics card or a paid service. Qdrant tests use `QdrantClient(":memory:")`. Smoke tests load the real ONNX models on random noise (CPU).

**Coverage (my run, `coverage run --source=src`):** total 95% (2,293 statements, 122 missed). Lowest: `embeddings/insightface_arcface.py` 62%; `index/bruteforce.py` 87%; `index/qdrant_index.py` 87%; `index/faiss_index.py` 88%; `evaluation/experiment.py` 92%; `evaluation/metrics.py` 93%. The project does not measure coverage.

**Three spot-checked tests**
1. `tests/test_metrics.py:80` `test_fmr_fnmr_hand_computed_at_threshold`: scores [0.9, 0.8, 0.4, 0.3], genuine mask [T, F, T, F], threshold 0.5; asserts FMR 0.5 and FNMR 0.5 with the arithmetic in comments. Checks real behaviour.
2. `tests/test_fusion_experiment.py:616` `test_tiebreak_keeps_face_order_on_absent_context`: builds three items, query context empty, asserts the tiebreak block equals face-only and hit@1 is 1.0. Checks real behaviour (query side only).
3. `tests/test_quality.py:38` `test_flat_blurry_image_scores_lower_than_noise`: asserts a flat grey face scores below a random-noise face; this pins the formula's direction and encodes its flaw (noise rated as sharp). Also `tests/test_index.py:51` recall test uses 2,000 low-rank vectors, with an in-file note that uniform random 512-number vectors cap recall near 0.8 regardless of index quality (my run at 50,000 random vectors gave 0.125).

**No tests exist for:** a real face end to end, restart behaviour, partial store failure, time-zone mixing, Docker, demo JS, fairness, concurrency beyond the lock.

**CI:** none. `docs/PAGES.md` says "no build step, no CI". GitHub's automatic `pages-build-deployment` runs on each push to `main`.

**Lint:** E1.

---

## E8. Plan-versus-results and history checks

| Item | Evidence |
| --- | --- |
| Plan before results | `d4f6876` (2026-10-06 11:19 -0400) added PLAN, ARCHITECTURE, README and `SYNTHETIC_SCENARIO_SPEC.md`, including "Ablation table shows fusion benefit (or honestly reports it does not)" and "Reporting the negative region is required". First results commit `4e095b7` (12:43) and first numbers `c96bc47` (pushed 2026-10-06T21:02Z). This is a plan, not a protocol with fixed numbers; no hypothesis threshold was stated |
| Post-hoc changes | Near-tie band changed from 0.65-0.75 to a measured 98th-percentile band after it proved empty (`4c25d19`, Oct 6 18:56); selective blend (`a9a5d79`, pushed 2026-10-07T11:09Z) and tiebreak (`5d2a75f`, pushed 12:08Z) were added after Finding 8 (`e4b30af`, pushed 12:01Z). Tiebreak evaluated on the same photos and seed as the failure; no held-out set |
| Push times (GitHub events, within 90 days) | 34 push events, 2026-10-06T15:49Z to 2026-10-09T17:01Z; each commit pushed within minutes of its commit time; the Oct 19 explainer commit built by Pages at 13:46Z |
| Bulk commits | By inserted lines: `4eaae82` 10,981 (generated figures and result JSON), `dcf47da` 3,195 (demo), `5c27010` 1,880 (generator, fusion), `728e0fe` 1,552 (adapters, index modules: message "worker output, validation pending"), `6798e31` 1,498 (fusion ablation). Step-by-step history is recoverable |
| Commit credits by tool | 2 of 41 commits carry a trailer: `c71e76b` Claude Opus 5.5 (1M context); `4e29476` Claude Sonnet 5.5. No other tool credited. Author name on all commits: Chris Schmidt |
| Corrections visible in history | `51e5630` "fix tb_ranked_all ordering" 42 min after `5d2a75f`; `a3aa733` loader skips macOS side files; `8708ce4` untracks a run log (log deleted: `bench_run.log`; it held benchmark output and a local temp path); `c96bc47` revises the rotation finding; `676ea13` then `e4b30af` record and reverse Finding 7 |
| Agent instruction files | None tracked or present |

---

## E9. Gates and approvals

None. No `gates.yaml`, no checklist, no approval records, no evaluator agent. Approver identification: not applicable.

---

## E10. Environment

Author's environment (from log lines in the committed history and the old explainer): Windows, Python 3.12, insightface 2.1 (the code comments mention its "legacy model_store" being broken). No lock file; `pyproject.toml` has lower bounds only.

Installed here (Python 3.12.13): numpy 2.5.4, scipy 1.18.1, faiss-cpu 1.15.1, insightface 2.1, onnxruntime 1.31.0, opencv-python(-headless) 5.0.0.93, fastapi 0.162.0, pydantic 2.14.1, qdrant-client 1.20.0, matplotlib 3.11.2, pytest 9.1.1, ruff 0.17.2, httpx 0.28.1 (pulled by qdrant-client), python-multipart 0.0.32 (added by me). Result files record model file names and a settings hash (fusion and latency runs) but no package versions. Fresh install today: all 241 pass; one `StarletteDeprecationWarning` about httpx. Undeclared: `python-multipart`, `scipy` (and `httpx`, needed by the web test client).

---

## E11. Security and privacy scan

- **Secret scan, current tracked files:** a pattern set (AWS keys, GitHub tokens, `sk-` and `sk-or-` keys, Slack tokens, private-key headers, Google keys, and `api_key|secret|password|token` assigned a 12+ character value) run with `git grep`, excluding the old explainer: no matches. **History:** the same patterns over every commit (`git rev-list --all`): no matches. **Sensitive filenames** (`.env`, `.pem`, `.key`, `.onnx`, `.zip`, `.pt`) tracked: none. No dedicated tool (gitleaks) was installed, so this is a pattern scan, not exhaustive.
- **History oddity:** `bench_run.log` was committed in `0927786` and untracked in `8708ce4`; it holds benchmark output and a local temporary path. No secret.
- **Local paths:** `docs/PORT_INVENTORY.md` (line 3) and `scripts/_fusion_runs.py`, `_fusion_runs_big.py`, `_fusion_runs_verify.py` contain Windows paths with the author's computer username and course folder. Not secrets; a privacy nit in a public repository.
- **Outside services:** model weights from the InsightFace release page at run time (no checksum); three.js 0.160.0 from jsDelivr in the demo (`docs/index.html`); GitHub Pages. No API keys anywhere.
- **GitHub settings:** secret scanning enabled, push protection enabled, Dependabot security updates disabled, `main` not protected, no releases.
- **Service exposure:** no sign-in, no upload size limit, no rate limit, binds `0.0.0.0` by default (`docker/serve.py`), one global lock, OpenCV decodes uploads (`api/app.py:159-169`), face data stored unencrypted in SQLite, 1 MB metadata strings accepted.
- **Licensing:** InsightFace weights non-commercial; AdaFace and MagFace weights follow research-only data terms; CR-FIQA has no licence (documented, not vendored).
- **Supply chain:** lower-bound pins, no lock file, no hash checks, weights unpinned.

---

## E12. Shortcomings register

Severity: Critical / High / Medium / Low. "Verify" says how to check a fix.

**S-01 (High). Headline numbers rest on same-photo pairs, a tiny private set, one draw and no ranges.** Evidence: the first test has 213 genuine pairs, of which 137 are probe i against gallery i (the same file damaged versus clean); 62 of 100 people have one photo (138 images, 100 people; 213 = 137 + 2 x 38). `benchmarks/README.md` discloses same-item pairs; `docs/RESULTS.md` does not. For the fusion runs, a persona bound to a person with one photo would get two media items pointing at one file (`generator.py:661-662`, `group[(i + j) % len(group)]`); face-only scoring 1.000 on clean queries is consistent with that (My reading; Ask the author). Impact: every accuracy number is flattered and unrepeatable. Disclosed: partly. Option A: recompute using cross-photo pairs only (76 of 213) and add Wilson ranges; gains an honest headline, costs few pairs and wider ranges, may look worse, reversible. Option B: add a public multi-photo data set (licence work, new runs). Option C: caveat only. Recommend A, then B if reopened. Effort small; payoff high. Verify: report both rates side by side.

**S-02 (High). The service forgets its index and silently overwrites notes after a restart (FAISS mode).** Evidence: no caller of `save` (grep); `docker/serve.py:113-135` starts ids at `index.size`, 0 with no snapshot; upsert at `sqlite_store.py:33-46`; reproduced with stand-ins (E4). Impact: wrong photo IDs returned. Disclosed: no (`docker/README.md` implies persistence). Option A: save after each add or at shutdown, reload at start, add a restart test; costs a write per add, risk of a half-written snapshot (write to a temporary name then rename); reversible. Option B: use Qdrant mode only (adds a server). Option C: document "memory is lost". Recommend A. Small; high. Verify: ingest, restart, search returns the same ids and notes.

**S-03 (High). "241 green" is not repeatable from the README; no CI.** Evidence: 12 failures without `python-multipart`; `scipy` and `httpx` undeclared; no lock file; no workflow. Impact: a newcomer or reviewer sees red. Disclosed: no. Option A: declare dependencies, add a lock file and a small test workflow (an hour, some upkeep). Option B: README note only. Recommend A. Small; high. Verify: a fresh clone passes in the workflow.

**S-04 (Medium). Documents overstate or contradict code in several places** (E14). Highest: "provably safe" (CL-16), "tested FAISS to Qdrant migration" (CL-23), persistence claim (CL-24), description with NER and Persons of Interest (CL-28), wrong run id (CL-31). Impact: credibility. Option A: correct each (minutes). Option B: leave. Recommend A. Small; medium. Verify: a second reader finds no contradictions.

**S-05 (Medium). The quality formula is fooled by noise; "low trust" has no effect; the gate is not applied at search.** Evidence: noise scores 0.999 (E4); `pipeline.py:75-94` stores LOW with the mark only in notes; `search` ignores it (`pipeline.py:103-149`); `SearchResult` has no trust field. Impact: bad photos pass silently. Disclosed: the formula is called a stand-in (`quality.py` docstring); the flag's inertness is not. Option A: surface trust, gate queries, replace the formula with a learned score (needs weights and a licence check). Option B: drop the LOW level. Option C: document. Recommend B or C unless reopened. Medium; medium.

**S-06 (Medium). The gate's best-looking evidence rests on very few photos.** Brightness 0.5: 5 probes; blur 3: 12 probes; the low-resolution 0.1 row is omitted from the write-up (6 probes, all failing). Option A: show counts beside every gated cell and include omitted rows. Small; medium. Recommend A.

**S-07 (Medium). Context scoring rough edges:** a candidate with no context scores 0 (treated like disagreement); a naive/aware time mix raises `TypeError` (500 through the API when a scorer is attached); `source_platform` is stored but unused; weights 0.4/0.3/0.3, half-life 30 days and blend 0.7/0.3 are undocumented; the weight sweep is built but no result is committed (`context_weight_sweep` is null in `20261007T125746Z.json`). Option A: neutral-for-missing, normalise time zones, document, run the sweep. Option B: leave, since context did not help. Recommend A for the crash only. Small; low.

**S-08 (Medium). No sign-in, size limits, rate limits or logging on the service.** Option A: key, limits, logging. Option B: state local use only. Recommend B now, A before any hosting. Medium; medium.

**S-09 (Medium). A half-failed store leaves an orphan fingerprint** (index added before notes, no undo; `pipeline.py:91-92`). Option A: write notes first or undo on failure. Small; low.

**S-10 (Low). Speed measured on random vectors where recall is 0.125 at 50,000; accuracy not reported with speed.** Option A: report recall with each speed row on structured data. Small; medium.

**S-11 (Low). Write-up wording:** off-angle called "rotation" (CL-12); rounding 0.150 versus 0.14948; the "recall 0.99" figure is asserted nowhere (CL-21).

**S-12 (Low). Local paths with the author's username in public files and history** (E11). Option A: replace with placeholders (history keeps them). Small; low.

**S-13 (Low). Lint:** 43 findings under ruff 0.17 defaults, mostly import order and quoted annotations; no formatter configured. Small; low.

**S-14 (Low). The demo simplifies the real context rule** (equal-weight mean of three marks, `docs/demo/data.js:103-106`, versus 0.4/0.3/0.3 in `linear.py`) and scripts its outcomes. Documented as synthetic.

Corrections owed now (finished-project view): S-02 text, S-03 declarations, S-04 wording, S-12. Changes that would mean reopening: S-01 option B, S-05 option A, richer context.

---

## E13. Math and method detail

| Item | Detail |
| --- | --- |
| Cosine score | `score(a, b) = a.b / (|a||b|)`. Stored vectors are L2-normalised, so inner product equals cosine (`faiss_index.py:92,109`). Exact search is O(N d) per query; HNSW is roughly logarithmic in N for well-structured data, memory O(N (d + M)) |
| HNSW settings | M 16, efConstruction 40, efSearch 64 (raised to k when larger). Recall depends strongly on data structure (E3 CL-21) |
| FMR, FNMR | `FMR(t) = #{impostor >= t} / #impostor`; `FNMR(t) = #{genuine < t} / #genuine` (`metrics.py:99-120`) |
| ROC and EER | Thresholds are all unique scores plus one sentinel; EER by linear interpolation where `FMR - FNMR` changes sign (`metrics.py:162-181`); constant scores give 0.5 |
| FMR at FNMR 1% | Lowest-FMR point with FNMR <= target, linearly interpolated (`metrics.py:184-213`). With n genuine pairs the FNMR step is 1/n (1/175 = 0.0057 at low-res 0.1), so 1% is 1.75 pairs |
| Quality mark | `0.5 min(lapvar/500, 1) + 0.25 clip(1 - 2 abs(mean/255 - 0.5)) + 0.25 clip(min_side/80)`; lapvar is the variance of a 4-neighbour Laplacian over the face crop (`quality.py:37-61`) |
| Context score | Weighted mean of geo (1 or 0, weight 0.4), time (`0.5^(days/30)`, 0.3), Jaccard of name sets (0.3) over signals present on both sides; 0.0 if none (`linear.py:87-107`) |
| Blend | `clip(alpha s_face + beta s_ctx, 0, 1)`; two candidates swap when `gap < (beta/alpha)(ctx_2 - ctx_1)`; with 0.7/0.3 the factor is 0.4286, so a context difference above 0.233 can flip a gap of 0.1 |
| Selective and tiebreak | Selective: blend the whole ranking if top-1 minus top-2 face score <= gate, else face. Tiebreak: face order; if gap <= gate and the query context is non-empty (`_has_context`: timestamp, geo or entities; platform ignored), swap the top two iff blended second > blended first (`fusion_experiment.py:427-459`) |
| Persona ranking | Candidate persona face score is the max cosine over its other items; persona context is that best item's context; ties by ascending persona id (`_prepare`, `_rank_personas`) |
| hit@1, MRR | `hit@1 = mean[rank = 1]`; `MRR = mean(1/rank)` |
| Wilson 95% range | centre `(p + z^2/2n)/(1 + z^2/n)`, half-width `z sqrt(p(1-p)/n + z^2/4n^2)/(1 + z^2/n)`, z = 1.96. Values: 9/18 0.29-0.71; 3/18 0.06-0.39; 8/18 0.25-0.66; 2/4 0.15-0.85; 3/4 0.30-0.95; 96/106 0.83-0.95; 82/106 0.69-0.84. Assumes independence; queries share personas, so true ranges are wider |
| Extreme-value toy | `P(at least one of 59 rivals high) = 1 - 0.95^59 = 0.9515` for an assumed 5% chance each; illustration only |
| Degradations | Blur: Gaussian filter, sigma per image axis (0 across colour); brightness: `clip(p + delta x 255)`; downsample: bicubic down then up; tilt: homography with edge shrink `cos(angle)`, inset `h(1 - cos)/2` (200 px: 29.3 px), black fill |
| References | Deng et al. 2019 (ArcFace); Malkov and Yashunin 2018 (HNSW, the method FAISS implements); Wilson 1927; Jaccard 1901 (the last three named from general knowledge, not cited by the repo) |

---

## E14. Discrepancies (docs versus code)

| ID | Docs say | Code or files say |
| --- | --- | --- |
| D-01 | Tiebreak "provably safe" (README:50; RESULTS.md:232) | No proof; -1 of 18 measured; wrong flips shown in the demo |
| D-02 | "Tested FAISS to Qdrant migration path"; snapshots portable between indexes (`docker/README.md:52`) | Cross-load rejected (`qdrant_index.py:241-242`); formats differ |
| D-03 | Index and notes persisted together with a stamp (ARCHITECTURE.md) | Service never saves; only the index has a stamp |
| D-04 | GitHub description: NER, Persons of Interest, battlefield intelligence | No NER code; README and PLAN say no watch-list language; a config profile is named `strict_surveillance` |
| D-05 | Close-out run `557189Z` (RESULTS.md:199, `data.js:29,155`, `panels.js:409-410`) | The artefact is `20261007T125746Z.json` |
| D-06 | "Detection misses abort the run" (`benchmarks/README.md:75`) | Skip-and-count (`experiment.py:388-425`) |
| D-07 | "Gate improves FMR at fixed FNMR" (PLAN criteria) | FMR 0 everywhere; disclosed |
| D-08 | "fetch script + checksums" (PLAN risks) | No checksum step |
| D-09 | Context uses source (README) | Source unused |
| D-10 | "2D rotation" explanation (RESULTS Finding 3) | The operator is a perspective squeeze |
| D-11 | "All outputs carry config hash" (ARCHITECTURE) | ROC results do not |
| D-12 | "recall 0.99" (README glossary) | Test asserts >= 0.9 |
| D-13 | Ambiguity as a cosine band (ARCHITECTURE; profile fields `ambiguous_low/high`) | Gap-based gate; band measured empty; band fields only validated by `LinearFusionScorer` |
| D-14 | `create_app(..., profile_name="broad-lead")` (docstring, `api/app.py`) | The profile key is `broad_lead` |
| D-15 | RESULTS brightness 0.75 FMR 0.150 | JSON 0.14948 |
| D-16 | Spec: `generate_scenarios.py`, `images/` output, held-out queries | Library plus persona-level queries |
| D-17 | Demo: "real top-2 tiebreak rule" with a mean of three context marks | Real scorer weights 0.4/0.3/0.3 |
| D-18 | RESULTS: extreme conditions "reported as None" | Low-res 0.1 reports 1.000 on 6 probes and is omitted from the table |

---

## E15. Production readiness detail

| Trait | Mark | Evidence |
| --- | --- | --- |
| Clean-start run | Partly | README install steps; undeclared packages; no lock; gallery private (E1) |
| Checks block bad change | No | No CI, no hooks |
| Quality measured against a baseline | Partly | Face-only baseline in every ablation; no ranges; same-photo pairs |
| Safe when things go wrong | Partly | 400/422 validation; counted detection failures; clear exit when weights are absent (`serve.py`, `_require_weights`); no timeouts, size limits or undo (E4) |
| Human review | No | Output is advisory |
| Versioned prompts, models, settings | Partly | YAML settings; scenario hash; model files unpinned; no prompts |
| Logs and monitoring | No | No logging code (grep); `/health` only; Docker HEALTHCHECK polls `/health` |
| Data and secrets | Partly | No secrets; weights and photos kept out; no sign-in; unencrypted face data |

Placement: **Works locally.** Full gap list: packaging (multi-stage Dockerfile, non-root user, HEALTHCHECK; unbuilt by me), hosting (none), sign-in (none), monitoring (none), backups (notes volume yes, index no), rollback (none), model versioning (none), cost limits (not applicable), security review (none), runbook (none), compliance (licence review only). Mirrors of real practice: baseline-first measurement, counted failures, a design change after a measured failure, a licence review, weights excluded from the image, percentile timing. Reusable assets: evaluation harness, licence comparison, scenario generator, tests.

---

## E16. Review dispositions

| Pass | Status |
| --- | --- |
| Rewrite under questionnaire v3.0 (2026-10-25) | The Explainer was rewritten as a continuous narrative (12 chapters, plain titles, at most two small tables per chapter, four diagrams, "In short," endings, acronyms spelled out at first use, a long models chapter covering SCRFD, ArcFace, FaceNet, AdaFace, MagFace, MTCNN). The earlier outline-style draft is kept unedited as `Grey-Resolve.v2draft.md`. E1-E15 still apply. Evidence map: models chapter uses E6 and E4 (real-model run on a noise picture: first values -0.0572, -0.1017, -0.0506, 0.0622, 0.0364, -0.0508; length 1.0000; range -0.147 to 0.118); hand-made scoring chapter uses E13 and E4; results chapter uses E3 and E12 |
| P4.5 style check (earlier draft) | Run. Fixes: code locations and tool detail moved out of the narrative (kept only in card last lines and the back-matter list); terms defined before first use and a key-terms box added; each caveat said once with back-references; words such as "pipeline" and "latency" replaced; length trimmed (about 15,400 words excluding table pipes; about 16,500 by a plain `wc -w`); worked examples checked by hand or code (quality 0.5875; context 0.7347 and 0.3381; blend 0.6404 and 0.5354; flip threshold 0.170; FMR/FNMR toy tables; Wilson ranges; 0.9515; brightness clipping above 64; tilt 141 px) |
| Self-read as a smart 15-year-old | Done once. Plain wording for "extreme-value noise" in the body; FMR at FNMR explained in the key terms and Math 2; Section 5.2 cards marked skippable |
| Diagrams | The three Mermaid diagrams (parts of the service, the study, the timeline) rendered with `@mermaid-js/mermaid-cli` using the system Chrome, no errors |
| Plain-language check by a fresh reader (Pass 3) | Not yet run (mandatory next step) |
| Accuracy review by a different model (Pass 4) | Not yet run |
| Author pass | Pending (Author Questions) |

---

## E17. Completeness and backlog register

**Promised versus built**

| Item | Source | Status | Evidence |
| --- | --- | --- | --- |
| Detect, gate, fingerprint, index, search | PLAN Phase 1 | Done and tested (fakes; smoke tests on real models) | `pipeline.py`; tests |
| FAISS HNSW and SQLite sidecar | PLAN | Done and tested | `index/`; `tests/test_index.py` (13) |
| FastAPI `/ingest` `/search` `/health` | PLAN | Done and tested with fakes | `api/app.py`; `tests/test_api.py` |
| Degraded ROC/FMR/FNMR for 4 or more damages | PLAN criteria | Done | `docs/results/20261006T191330Z.json` |
| Gate improves FMR at fixed FNMR | PLAN criteria | Unmet, disclosed | FMR 0 everywhere |
| Scenario generator from one command, fixed seed | PLAN Phase 2 | Partly: library plus `evaluate_fusion.py`; no `generate_scenarios.py` | `scenario/generator.py` |
| Fusion scorer calibrated on the synthetic set | PLAN Phase 2 | Partly: scorer done, calibration not run | `context_weight_sweep` null |
| Ablation shows benefit "or honestly reports it does not" | PLAN criteria | Done (negative) | Findings 6-9 |
| Every metric traceable to a scenario config | PLAN criteria | Mostly | hash in fusion and latency JSON, not ROC JSON |
| Docker Compose API and Qdrant | PLAN Phase 3 | Written, not checked | `docker/` |
| Latency profiler | PLAN Phase 3 | Done; full-chain timing is 2-sample smoke | `latency-*.json` |
| Figures from run artefacts | PLAN Phase 3 | Done (not re-run) | `docs/figures/` |
| ONNX INT8 export | PLAN Phase 3 | Dropped on purpose | PLAN |
| Resolution Console demo | PLAN Phase 4 | Done, public | E1 |
| MagFace quality assessor | PLAN open decision | Planned only | `quality.py` docstring |
| AdaFace fallback | PLAN decisions | Planned only | `BACKBONE_LICENSES.md` |
| Notes inside Qdrant | `docker/README.md` | Planned only | no code |
| Hard pose (yaw/pitch) test | RESULTS Finding 3 | Planned (future work) | no code |
| Confidence intervals | RESULTS limitations | Not built | no code |
| Index persistence in the service | ARCHITECTURE | Stub-like: `save` and `load` exist, never called | grep |
| Context in the service | ARCHITECTURE `/search` | Not wired (`fusion=None`) | `serve.py:107` |
| Held-out query set; `images/` output | SPEC | Not built | spec versus code |

**Backlog (full)**

| Group | Item | Effort / payoff | Depends on | Trade-off |
| --- | --- | --- | --- | --- |
| Finish | Build the FAISS-to-Qdrant data move, or delete the claim | S (delete) or M / M | none | delete is free; build adds a converter |
| Finish | Run and commit the weight sweep | S / M | private gallery | may confirm failure |
| Finish | Wire persistence into the service (S-02) | S / H | none | write cost per add |
| Finish | Wire an optional context blend into the service | S / L | S-07 fixes | changes behaviour; context did not help |
| Fix | Declare dependencies, add a lock file (S-03) | S / H | none | upkeep |
| Fix | Correct overstated text and stale notes (S-04, E14) | S / M | none | none |
| Fix | Time-zone crash, orphan fingerprint, missing-context handling (S-07, S-09) | S / L | none | could change published numbers if applied to experiments |
| Fix | Local paths in public files (S-12) | S / L | none | history keeps old paths |
| Improve | Cross-photo rates with ranges (S-01) | S / H | gallery | fewer pairs |
| Improve | Gate counts beside each cell (S-06) | S / M | none | none |
| Improve | Surface trust, gate queries, better quality score (S-05) | M / M | weights, licence | complexity |
| Improve | Recall beside speed (S-10); logging | S / M | none | none |
| Add | CI on every change | S / H | S-03 | upkeep |
| Add | Private hosting, sign-in, limits, runbook (S-08) | M / H | S-02 | cost; licence limits public hosting |
| Add | Real-pose tests; richer context features | L / M | new data | open-ended |

**Milestone plan.** M1 "Repeatable and honest" (S-04, S-03, S-02, CI): exit test is that a stranger follows the README on a new machine and sees 241 passes in the workflow, the restart test passes, and no claim in E3 remains contradicted. M2 "Trustworthy numbers" (S-01, S-06, S-10): exit test is the headline table recomputed on cross-photo pairs with ranges and counts. M3 "Watched service" (S-08, hosting, logging, runbook): exit test is a private deployment that answers `/health`, logs a failed request, and can be restored from a clean start using the runbook. Judgment: polish and freeze after M1 and M2; reopen only for new context features or real multi-photo data.


---

## E18. Revision notes, diagram sources and image provenance

**Correction to the context probe (E4, CL-18).** The earlier one-on-one duel figures (true wins 59.2%, loses 31.0%, ties 9.8%) mixed populations: the duel included queries whose true persona's photo carried no context (scored 0, so it lost). A cleaner rerun (seed 42 plus ten other seeds, all 59 rivals per query, three groups) gives: both photos carry context, the true persona wins 91.2% (seed 42; 2,006 pairs) and 90.1% (seeds 1-10; 21,476 pairs), loses 8.8% and 9.9%, ties 0; query photo blank: all ties; only the true persona's photo blank: loses 69.5% (seed 42) and 78.7% (seeds 1-10), ties 30.5% and 21.3%. Pairs with at least one blank side: 43% (seed 42). Over all pairs (seed 42): win 51.7%, lose 25.8%, tie 22.5%. The guide uses the corrected statement. The project's own diagnosis (Finding 9) says the signal against real face rivals is weaker than against random rivals; not re-checked.

**Guide revision after the fresh-reader check (2026-10-25).** Prior version kept as `Grey-Resolve.v3draft1.md`. Moved here from the guide: edge-case table detail (E4), shortcoming cards (E12), full backlog and milestone plan (E17), claims list (E3), code map (E5), the 38-claim counts' per-claim verdicts, and the mermaid timeline.

**Images (in `images/`).** `grey-resolve-damage-types.png`: a drawn cartoon face (no real photo) damaged with the project's own operators at blur 8, low resolution 0.1, brightness 0.75, tilt 45. `grey-resolve-face-finder.png`: drawn illustration with hand-placed box and five points, not real model output. `grey-resolve-damage-results.png`: bars from `docs/results/20261006T191330Z.json` (sweep: eer and fmr_at_fnmr_0.01). `grey-resolve-blend-results.png`: bars from `docs/results/20261007T125746Z.json` (ablation: overall and gap_le 0.05 hit_at_1 for face_only, fused, tiebreak). Diagram PNGs were rendered from the sources below with mermaid-cli and the system Chrome, scale 2.

Diagram source, grey-resolve-parts.png:

```
flowchart TD
  P["Photo plus optional place, time, names"] --> F["Face finder"]
  F --> Q["Quality checker - hand-made formula"]
  Q --> G["Gatekeeper: full, low, or reject"]
  G --> E["Fingerprint maker"]
  E --> S["Fingerprint store"]
  G --> N["Notes database"]
  S --> R["Ranker - face score, optional blend"]
  N --> R
  R --> O["Ranked list of photo IDs with scores"]
  M["Better quality score - planned only"] -.-> Q
```

Diagram source, grey-resolve-study.png:

```
flowchart LR
  D["Damager - 4 kinds"] --> B["Measurer - mistakes and top-one hits"]
  G2["Made-up people generator"] --> B
  M2["The same face models"] --> B
  B --> J["Saved result files"]
  J --> C["Charts"]
  J --> W["Demo page - replays saved numbers"]
```

Diagram source, grey-resolve-learned-vs-rules.png:

```
flowchart LR
  A["Photo"] --> B["Face finder - learned"]
  B --> C["Straighten the face - fixed rule"]
  C --> D["Fingerprint maker - learned"]
  D --> E["Face score - fixed rule"]
  B --> F["Quality mark - hand-made formula"]
  F --> G["Gate - fixed rule"]
  E --> H["Optional blend with context - fixed rule"]
  H --> I["Ranked list for a person to judge"]
```


---

## E19. Origin evidence: the IronClad course project (read-only, 2026-10-25)

Read from the author's local course folder (JHU 705.603 Creating AI-Enabled Systems, Module 8 Vector Search Systems; the assignment repository for the fall 2026 offering). Files read: `README.md`, `IMPLEMENTATION_ASSIGNMENT.md`, `CASE_ANALYSIS.md` (summary, sections 1 and 10 and the table of contents); folder listing of `ironclad/` (app, `design_config.json`, `modules/extraction`, `modules/retrieval`, `utils/metrics.py`), `analysis/` (five notebooks `task1` to `task5`, `results`, `figures`), `test/` (unit tests and grade runner). Nothing was copied out and nothing was written there; no photos or data were opened.

Facts used in the guide:
- A door-access scenario: a camera probe photo is turned into a vector and searched against a gallery of enrolled employees (CASE_ANALYSIS section 1); a Flask service with `/add` and `/identify` (README).
- The course supplied `embedding.py` and `preprocessing.py` ("already implemented"), the tests, the autograder workflow and the classroom settings; the assignment's to-do list is the three FAISS indexes (`bruteforce`, `hnsw`, `lsh`), `search`, and the two endpoints (README, IMPLEMENTATION_ASSIGNMENT).
- Embeddings: InceptionResnetV1 from facenet-pytorch with CASIA-WebFace or VGGFace2 weights, 160 by 160 input, 512 numbers; MTCNN face cropping (CASE_ANALYSIS header and tasks).
- Docker packaging is part of the assignment (IMPLEMENTATION_ASSIGNMENT objectives). Autograder: "scores it 120/120" (README status).
- Data: 2,265 gallery images of 1,000 people and 999 probe images; 629 of 1,000 people have a single enrolment photo (CASE_ANALYSIS, operations section).
- The five analysis tasks: model selection (VGGFace2), preprocessing design (MTCNN cropping plus the normalisation the weights expect), index selection (HNSW, efSearch 64, 54 times faster than exact search at 100,000 vectors), number of candidates returned (N = 3), images per identity (at least 2). Selected settings are in `ironclad/design_config.json`.
- Grey-Resolve side (its own files): README acknowledgement ("Course code informed the design; this repository is an independent, re-scoped implementation"); `docs/PORT_INVENTORY.md` (what ports from IronClad, written against the course folder); ARCHITECTURE ("Ported from IronClad's FaissHNSW (M=16, efConstruction=40 baseline)"; "replaces IronClad's raw MTCNN-confidence gate"); evidence-run commands read the course gallery (`multi_image_gallery`) from that folder.
- Authorship, per the author's statement in AUTHOR_CONTEXT.md (2026-10-25): he wrote IronClad mostly with Claude Code as an assistant; the course supplied the skeleton and the two extraction modules (embedding and preprocessing); the search side, tests, packaging and report were built by him with Claude Code. Open items are in `Grey-Resolve.questions.md`.

## E20. Rewrite log (2026-10-25, third pass)

Guide rewritten in the author's first-person voice under questionnaire v3.2; previous versions kept as `Grey-Resolve.v2draft.md`, `Grey-Resolve.v3draft1.md`, `Grey-Resolve.v3draft2.md`. The dedicated FaceNet subsection was removed (FaceNet now appears in the origin paragraph and the alternatives paragraph only). Acronym pairings were rewritten to "short form, full name in parentheses, then what it is and why it is used": MIT, AI, JHU, MTCNN, HNSW, LSH, FAISS, SCRFD, ONNX, ArcFace, FMR, FNMR, EER, LFW, plus name-style pairings for SQLite, Qdrant, Docker, AdaFace, MagFace, CR-FIQA, FIQA. Statements about checking are impersonal ("the 241 tests pass"). Unconfirmed items were moved out of the guide into `Grey-Resolve.questions.md` (not published with the guide).

## E21. Clean re-run log (2026-10-25, questionnaire v3.2, written from the questionnaire alone)

Purpose: test whether the questionnaire alone produces the form the author wants. Old drafts and reader checks were not opened. Inputs: questionnaire v3.2, AUTHOR_CONTEXT, the earlier map and evidence notes (as research notes), the fresh clone (read-only, `git status` clean, 41 commits), the IronClad origin files (README, IMPLEMENTATION_ASSIGNMENT, first 120 lines of CASE_ANALYSIS; nothing copied).

**Re-verified this run (labels: Confirmed unless noted).**
- Quality formula `detection/quality.py:37-61`: 0.5 sharpness (Laplacian variance / 500, capped), 0.25 exposure (1 - 2|mean - 0.5|), 0.25 size (min side / 80). Toy example 0.5, 0.8, 0.75 gives 0.6375 (Ran it, arithmetic).
- Gate `detection/base.py:46-61`: full at or above quality_min, low at or above half of it, else reject. Strict profile 0.60 and so 0.30 (`configs/threshold_profiles.yaml`).
- Context score `fusion/linear.py:87-107` and weights lines 27-30; toy example 0.4 + 0.3 x 0.5^0.5 + 0.1 = 0.7121 (Ran it). Blend toy: 0.7 x 0.62 + 0.3 x 0.2 = 0.494; 0.7 x 0.58 + 0.3 x 0.9 = 0.676 (Ran it).
- Tiebreak `evaluation/fusion_experiment.py:427-459` (top two only, gap gate, query context non-empty).
- Ingest gates, search does not: `pipeline.py:75-94` versus `103-149` (Confirmed). The journey image was redrawn so the gate sits on the adding path only (an earlier version of the picture would have shown the gate on a search).
- Sweep numbers read from `docs/results/20261006T191330Z.json`: EER and FMR at 1% misses for brightness 0.75 (0.0217, 0.1495), low-res 0.1 (0.0457, 0.8727), blur 8 (0.0781, 0.5472), tilt 45 (0.0141, 0.2564); probes dropped 42, 19, 12, 1 of 138. Genuine pairs 213; gate on clean 6 of 213 versus 4 of 199 (Confirmed).
- Fusion table from `20261007T125746Z.json`: 106 queries, face-only 0.9057, blended 0.7736; gap at most 0.05 n=18: face-only 9/18, blend 3/18, tiebreak 8/18 (0.444), selective overall 0.849 (Confirmed, from the earlier read of the same file and RESULTS.md).
- Wilson ranges recomputed with Python in this run: 9/18 0.29-0.71, 3/18 0.06-0.39 (Ran it).
- Speed claim (HNSW p95 up to 1.6 ms at 50,000 against about 29 ms exact) is Per the docs; the random-vector recall caveat (0.125 at 50,000 in a probe) is carried over from E3 CL-21 and E4 (Ran it, earlier pass), not re-run now.
- Coverage (95 percent), the 12 failing tests without the undeclared package, rule attacks and the 91 percent clue-versus-rival figure are carried over from E3, E4, E7 and E18 (Ran it, earlier pass; not re-run in this clean pass).

**Corrections made against the earlier notes or the first instinct.**
- IronClad already tested photo noise (11 conditions: blur, resizing, brightness) in its Task 1 (CASE_ANALYSIS summary), so the Explainer does not say Grey-Resolve added damage testing; it says Grey-Resolve keeps the idea, adds a tilt test and adds clues and the blend.
- `docs/PORT_INVENTORY.md` says "No code is copied into Grey-Resolve" (it is an inventory), while ARCHITECTURE and PLAN say "Ported from IronClad's FaissHNSW"; the Explainer reports both without deciding how much was copied (see questions 6).
- Tool attribution: the Explainer says only "AI coding agents" for Grey-Resolve; the two commit credits stay here in E8. In the origin chapter, "mostly with Claude Code" applies to IronClad only, per the author.

**Image provenance (images/).** New: `grey-resolve-journey.png`, an HTML card strip rendered with the system Chrome (headless screenshot, scale 2); source kept in the run scratchpad; every number is the toy example above. Reused unchanged and re-checked by eye against the code: `grey-resolve-parts.png`, `grey-resolve-learned-vs-rules.png` (diagram sources in E18), `grey-resolve-damage-types.png`, `grey-resolve-damage-results.png` (bars match the JSON values above), `grey-resolve-blend-results.png` (0.91/0.77/0.91 and 0.50/0.17/0.44 match). Not used this time: `grey-resolve-face-finder.png`, `grey-resolve-study.png`.

**Length.** Plain `wc -w` of the Explainer including table pipes, glossary and code list: about 8,400 words (target 5,000-8,000; slightly over).

**Shortcomings register, claims, rule attacks, tests and CI, secret scan:** unchanged from E3, E4, E7, E11, E12 above.

## E22. Revision log (2026-10-25, v4 draft 2 from reader check)

Applied all ten reader-check fixes to `Grey-Resolve.md`; prior text archived as `Grey-Resolve.v4draft1.md`. Changes: paragraph one now has the plain "fixed rules wrapped around two learned models" sentence, and the deterministic/probabilistic paragraph was split into three short ones with the learned-vs-rules picture moved to the opening; origin chapter halved (tool list cut; IronClad facts kept: JHU 705.603, supplied FaceNet-style network and MTCNN, FAISS indexes, service, tests, Docker, report, 120/120, mostly Claude Code); LSH, ONNX, Flask, Qdrant, buffalo_l, RetinaFace, CR-FIQA, file names removed; SciPy, README, AdaFace, MagFace, InsightFace, WebFace600K, index, landmarks, weights, cut-off, pair, confidence range defined at first use; parts chapter rewritten as prose; shortcomings stated once (testing chapter holds the four holes, shortcomings chapter points back); interview section cut to five questions and three tough ones; "careful no" and "well tested" softened to single run, 18 close calls, stand-in models; code map labelled for engineers; table column renamed "Impostor pairs let in at a strict cut-off". Word count by `wc -w`: 7,929.

Images viewed: `grey-resolve-damage-types.png` (accurate: blur 8, low-res 0.1, brightness +0.75, tilt shown as a squeeze/trapezoid; caption says drawn cartoon face), `grey-resolve-face-finder.png` (illustration with hand-placed points; caption says so; now used in the face models chapter, with that caveat in the text), `grey-resolve-study.png` (boxes match the study side: damager, made-up people generator, same face models, measurer, saved results, charts, demo page; now used in the what-it-does chapter). Numbers carried over from E21 unchanged; no new runs.
