# Grey-Resolve Port Inventory

Source repo: `C:\Users\pchri\JHUAIEngineering\705.603-Creating_AI-Enabled_Systems\Modules\Mod8-VectorSearchSystems-3\705-603-fall-2026-ironclad-pcschmidt` (JHU 705.603 "IronClad" assignment).
Target repo: Grey-Resolve Phase 1 — ArcFace/AdaFace embedding backbone (InsightFace or AdaFace repo), real FIQA quality gate, FAISS HNSW + SQLite metadata sidecar, degraded-input ROC/FMR/FNMR benchmark harness.

This is an inventory only. No code is copied into Grey-Resolve.

Grey-Resolve Phase 1 changes that drive every port decision below:

1. **Embedding swap**: `facenet_pytorch.InceptionResnetV1` (512-d, 160x160 input) -> ArcFace/AdaFace (512-d, typically 112x112 aligned input). The FAISS `dim=512` contract survives; the preprocessing and model loading do not.
2. **Quality-gate swap**: MTCNN detection confidence (`return_prob=True` / `crop_images` diagnostics) -> a real FIQA score (e.g., SER-FIQ, FaceQnet, MagFace-embed quality, or CR-FIQA). The "gate vs. baseline vs. always-crop" experimental design in `task2` survives; the score source does not.
3. **Retrieval storage**: FAISS index with in-memory Python `metadata` list -> FAISS (HNSW) plus a SQLite metadata sidecar keyed by FAISS row id.
4. **Metrics**: identification metrics (Top-1, mAP@50, MRR, Hit@N) -> add verification metrics (genuine/impostor score distributions, ROC, FMR(t), FNMR(t), EER) for degraded inputs.

---

## 1. File-by-file inventory

### `ironclad/app.py` (Flask service, 5.3 KB)

Module-level Flask app exposing `/add` (multipart `image` + `name`) and `/identify` (multipart `image` + `k`). It wires `Embedding` -> `Preprocessing` -> `FaissSearch` over a configured FAISS index (`FaissBruteForce` / `FaissHNSW` / `FaissLSH` chosen from `design_config.json`), guards the catalog with a `RLock` (`catalog_lock`), and over-fetches then dedupes candidates with `utils.metrics.unique_identities`. `prepare_image()` is the MTCNN quality gate: if `design["preprocessing"] == "gated"` an `MTCNN(image_size=160, ...)` detector crops the face and only the crop is used when its confidence >= `confidence_threshold`, otherwise the original image is passed through.

- **facenet-pytorch/MTCNN coupling: high.** Direct `from facenet_pytorch import MTCNN`, hard-coded `dim=512`, `Preprocessing(image_size=160)`, MTCNN `(crop, confidence)` contract in `prepare_image`.
- **Grey-Resolve port: small refactor + local rewrite.** Keep the route contract, validation/400 semantics, catalog lock, and the over-fetch/dedup loop in `/identify`. Rewrite `prepare_image` around an aligner + FIQA gate (score + accept/reject policy instead of MTCNN confidence), swap `Embedding` construction, and persist index metadata to SQLite instead of `index.metadata` list. The module-level `index` / `model` / `preprocessor` / `search` names are an autograder contract and can be renamed.

### `ironclad/modules/extraction/preprocessing.py`

Class `Preprocessing(image_size=160)` with `process(probe: PIL.Image) -> torch.Tensor`: `torchvision.transforms` pipeline (Resize to 160x160, ToTensor, ImageNet mean/std normalize), batch dim added, CPU only. Instructor-provided and untouched in the assignment.

- **Coupling: medium.** No facenet import, but the pipeline is sized/normalized for the FaceNet input contract (160x160; ImageNet normalization).
- **Grey-Resolve port: rewrite.** ArcFace/AdaFace expect a 112x112 aligned crop with their own normalization (typically `(x/255 - 0.5) / 0.5`). Keep the "one class, one `process(image) -> tensor`" interface shape.

### `ironclad/modules/extraction/embedding.py`

Class `Embedding(pretrained='casia-webface'|'vggface2', device='cpu')` wrapping `facenet_pytorch.InceptionResnetV1(pretrained=...).eval()`; `encode(image_tensor) -> np.ndarray` runs `torch.no_grad()` inference and returns a squeezed 512-d NumPy vector.

- **Coupling: total.** The whole class is facenet-pytorch.
- **Grey-Resolve port: rewrite.** New embedder class behind the same `encode(tensor) -> np.ndarray` interface, loading ArcFace/AdaFace weights via InsightFace `FaceAnalysis`/onnxruntime or the AdaFace repo. Keep "returns float32 numpy 512-d" so the FAISS layer and caches stay valid.

### `ironclad/modules/retrieval/search.py`

Class `FaissSearch(faiss_index, metric, p=3)`: single-probe `search(query_vector, k)` over any index wrapper. Validates the query with `index.validation.vectors`, enforces metric/index agreement, caps `k` to `ntotal`, resolves metadata via `faiss_index.get_metadata`, and implements an approximate `minkowski` rerank (`_compute_minkowski` over reconstructed candidate vectors). Scores: squared L2, cosine/dot similarity, Hamming for LSH.

- **Coupling: none.** Pure numpy/FAISS; embedding-agnostic apart from `dim`.
- **Grey-Resolve port: small refactor.** Keep as-is except replace `self.faiss_index.get_metadata(i)` with a SQLite sidecar lookup (`index -> identity, source path, quality score`), and optionally extend `search` to batch probes. The Minkowski rerank can be dropped (not needed for cosine verification benchmarks).

### `ironclad/modules/retrieval/index/bruteforce.py`, `hnsw.py`, `lsh.py`

Three FAISS index wrappers with a shared shape: `__init__(dim, metric/params)`, `add_embeddings(embeddings, metadata)`, `get_metadata(idx)`, `save/load` (pickle of the whole wrapper). `FaissBruteForce` uses `faiss.IndexFlatL2/IndexFlatIP`; `FaissHNSW` uses `faiss.IndexHNSWFlat(dim, M, metric_type)` with `efConstruction`/`efSearch`; `FaissLSH` uses `faiss.IndexLSH(dim, nbits)`. All delegate validation to `index/validation.py` and keep a Python `self.metadata` list.

- **Coupling: none.** Fully model-agnostic (512 comes from the caller).
- **Grey-Resolve port: small refactor (HNSW) / port as-is for reference (Flat, LSH).** Grey-Resolve keeps FAISS HNSW as the production index: keep `FaissHNSW` logic but replace `self.metadata` with a SQLite sidecar (row id <-> identity/image/quality columns), and replace pickle `save/load` with `faiss.write_index` + DB migration. `FaissBruteForce` is still needed as the exact-search ground-truth oracle for ANN-recall and ROC computations. `FaissLSH` is optional; drop it unless you want a second approximate-index comparison. `validation.py` (`positive_integer`, `metric_name`, `vectors`, `add`, `metadata_at`) ports as-is.

### `ironclad/utils/metrics.py`

Small pure-numpy metric library: `unique_identities(ranking, n)` (first-occurrence dedupe), `average_precision(relevance, total_relevant, k)` (AP@K with `min(total_relevant, k)` denominator), `reciprocal_rank(relevance)`, `hit_at_n(ranking, identity, n)`, `ann_recall(approximate, exact)`.

- **Coupling: none.**
- **Grey-Resolve port: port as-is + extend.** These remain the retrieval/ranking metrics. Add new verification functions: genuine/impostor pair builders, `roc_curve(scores, labels)`, `fmr_fnmr(scores, labels, threshold)`, `eer(...)`, and threshold-at-FMR helpers. That extension is the core new metric work for Phase 1.

### `ironclad/design_config.json`

Runtime configuration consumed at app startup: `model` ("vggface2"), `index` ("hnsw"), `metric` ("cosine"), `candidate_count` (50), `preprocessing` ("gated"), `confidence_threshold` (0.99), `index_parameters` (M/efConstruction/efSearch), `scope` disclaimer. `analysis/report.py` writes the final measured recommendation back into this file. `app.py` validates it strictly at startup.

- **Grey-Resolve port: small refactor.** Same shape; change `model` values to ArcFace/AdaFace checkpoints, `confidence_threshold` becomes a FIQA threshold (different scale), add FIQA backend and SQLite path keys. The "config is validated at startup and the report writes the measured config back" pattern is worth keeping.

### `analysis/common.py` (15 KB — the harness core)

Shared deterministic analysis library. Notable pieces:
- Constants: `SEED=603`, `MODELS=("casia-webface","vggface2")`, `CONDITIONS` (clean + blur 1-3 + resize 120/80/40 + brightness 0.5/0.75/1.25/1.5 — 11 conditions), `NS=(1,3,5,10,20,50)`, `CACHE/RESULTS/FIGURES` dirs.
- `audit_data()` / `load_manifest()`: walks `ironclad/storage/multi_image_gallery` and `probe`, skips AppleDouble `._*` sidecars, records per-image `sha256`, `pixel_sha256`, size, brightness, contrast, dHash to `results/manifest.csv`, then writes `dataset_audit.json` (identity counts, gallery/probe-only identities, exact pixel duplicate groups, dHash near-duplicate candidates, quality summary) and `environment.json` (package versions, CPU, threads). Hard-asserts expected counts: gallery 2265 images / probe 999.
- `perturb(image, condition)`: applies blur/brightness/resize degradation — the "degraded input" generator.
- `crop_images(rows, condition)`: **MTCNN-coupled** — runs `facenet_pytorch.MTCNN` per image (with `select_boxes`/`extract`), saves 160px crops to `cache/crops_<key>/<n>.png`, records `diagnostics.csv` (`path, confidence, detected, detection_ms, x1..y2`). This is the quality-gate measurement surface.
- `encode(rows, model_name, condition, cropped)`: **facenet-coupled** — hashes the checkpoint file (`20180402-114759-vggface2.pt` / `20180408-102900-casia-webface.pt` under the Torch hub cache), batches through `Embedding`/`Preprocessing`, caches to `cache/embeddings_<sha256>.npz` (`vectors` + `metadata` JSON with data signature, weight hash, condition, environment versions, preprocessing file hash, pipeline version).
- `build_index(kind, values, labels, **params)`, `neighbors(wrapper, queries, unique_count)`: FAISS index construction + adaptive over-fetch until enough unique identities.
- `per_probe(indices, gallery_labels, probe_rows)`: per-probe ranking metrics (`top1, map50, mrr, hit1..hit50`).
- `summarize`, `bootstrap` (identity-cluster bootstrap CI), `paired_delta` (paired comparison with CI), `tuning_mask` (identity-hash 20% tuning split), `draw_curves`, `source_signature`/`cache_key` (cache invalidation).

- **Coupling: medium-high.** `crop_images` (MTCNN), `encode` (facenet checkpoint names, 160px preprocessing), `MODELS`, and `environment()`'s `facenet-pytorch` version pin are the coupled surface; everything else (audit, perturb, caching, bootstrap, splits) is generic.
- **Grey-Resolve port: small refactor + targeted rewrite.** Keep audit/manifest, `perturb`, cache-key/signature machinery, `bootstrap`/`paired_delta`/`tuning_mask`, `per_probe`/`summarize`, `draw_curves`. Rewrite `crop_images` as `align_and_score` (insightface alignment + FIQA score in `diagnostics.csv`), rewrite `encode` for the new backbone (new checkpoint hashing, 112px preprocessing, keep the `.npz` cache format — extend `metadata` with backend + weight hash). Add genuine/impostor pair generation and score extraction for the verification metrics.

### `analysis/experiments.py` (24 KB — five preregistered experiments)

Functions `task1()`..`task5()` plus helpers `splits()`, `baseline()`, `measurement()`, `prepare_crops()`, `crop_confidence()`, `combine(original, cropped, confidence, threshold)`, `configured_vectors()`, `tune_indices()`, `benchmark(wrapper, queries, exact_indices)`, and `TASKS = {1: task1, ...}`.

- `task1()`: for both weight sets, encodes gallery + probes under all 11 conditions with exact cosine search; reports Top-1/mAP/MRR/Hit@N with bootstrap CIs, paired casia-vs-vggface2 deltas, PCA embedding scatter, severity curves, and a model-selection rule (`task1_selection.json`). This is the **degraded-input robustness** experiment.
- `task2()`: the **quality-gate** experiment. Compares three policies — `baseline` (original image), `crop` (always MTCNN crop), `gated` (crop only when MTCNN confidence >= threshold) — over thresholds (0.9..0.999) tuned on a 20% identity split (`tuning_mask`), evaluated on heldout probes with paired deltas, detection-rate diagnostics, and recovered/regressed error examples.
- `task3()`: index benchmark — `tune_indices` grid over HNSW (M/efConstruction/efSearch) and LSH (nbits), then `benchmark()` measures p50/p95 latency, QPS, ANN recall@10 vs exact, serialized index bytes, build time, RSS delta, single-add latency, at real + synthetic scale (10k/30k/100k distractors), with log-log extrapolation to 1e6/1e9 (`task3_extrapolation.csv`) and an index-selection rule.
- `task4()`: **CMC / candidate-count** experiment — Hit@N for N in (1,3,5,10,20,50) across models x indexes on heldout probes, plateau-deficit CI vs Hit@50, adaptive-batch latency, and an N-selection rule per plateau fraction (`task4_selection.json`).
- `task5()`: **enrollment-saturation** experiment — for cohorts of identities with >=10 (and >=5 sensitivity) gallery images, subsamples m=1..10 gallery images per identity over 30 seeds, measures Top-1/mAP/Hit@5 and search latency, and selects the smallest m within 1pp of peak.

- **Coupling: high in `task2`/`crop_confidence`/`combine`** (MTCNN confidence semantics), medium in `task1`/`configured_vectors` (two facenet weight sets, crop flag), low in `task3`/`task4`/`task5` (index/ranking logic is generic).
- **Grey-Resolve port:**
  - `task1` -> **small refactor** (new backbone pair, e.g., ArcFace-r100 vs AdaFace-r100; same condition grid; add per-condition ROC/FMR/FNMR — see Section 2).
  - `task2` -> **rewrite** around FIQA: `combine`/`crop_confidence` become `gate(original_aligned, quality_score, threshold)`; the baseline/crop/gated policy comparison, tuning-split discipline, paired heldout deltas, and error-example extraction all carry over.
  - `task3` -> **port as-is** with the SQLite sidecar in `build_index`/`neighbors`; keep `benchmark()` (it is a good, honest latency/recall harness).
  - `task4` -> **extend** CMC into ROC/FMR/FNMR (see Section 2).
  - `task5` -> **port as-is** (enrollment saturation is still a good portfolio analysis).
  - `benchmark`, `tune_indices`, `per_probe`, `summarize`, `paired_delta` are reusable nearly unchanged.

### `analysis/common.py` sibling: `analysis/report.py` (36 KB)

Single `generate_report()` (plus `table()` markdown formatter and `load_json()`): refuses to run until all required result artifacts exist (`task*_summary.csv`, `task*_selection.json`, `dataset_audit.json`, `duplicate_review.json`, ...), then renders the self-contained `CASE_ANALYSIS.md` (tables, figures, per-task interpretation, limitations) and writes the final measured configuration into `ironclad/design_config.json`. Content is facenet/MTCNN-specific prose; structure is generic.

- **Coupling: high in prose, none in mechanics.**
- **Grey-Resolve port: rewrite content, keep mechanics.** Keep `table()`, the "no report without evidence" gate, and the write-config-back step. Rewrite all narrative around ArcFace/AdaFace + FIQA + ROC/FMR/FNMR.

### `analysis/evidence.py` (2 KB)

`validate_notebooks(root)`: every `task*.ipynb` must be fully executed with no error outputs; `validate_report(root)`: additionally checks all local markdown links resolve, Table/Figure captions are sequential, no TODO/TBD/FIXME, and `design_config.json["candidate_count"]` is one of the tested N values. Raises `ValueError`/`FileNotFoundError` otherwise.

- **Coupling: none.** Pure JSON/markdown validation.
- **Grey-Resolve port: port as-is** (adjust paths/report name; update the tested-N list if NS changes; extend to check that ROC/FMR/FNMR artifacts exist).

### `analysis/run.py` (7 KB)

CLI: `audit | crops | notebooks | execute | all | 1..5`. `make_notebooks()` generates the five evidence notebooks (markdown design question + method + limitations, code cells calling `TASKS[n]()`), `execute_notebooks()` runs them via `nbclient` with a 2 h timeout and writes executed notebooks back. `DESCRIPTIONS` dict holds each task's research question and method summary.

- **Coupling: low** (prose in `DESCRIPTIONS` and notebook templates is facenet/MTCNN-flavored).
- **Grey-Resolve port: port as-is + edit text.** Keep the notebook-generation/execution/evidence loop; update `DESCRIPTIONS`, notebook limitation text (add: verification metrics, FIQA gate, no unknown-person rejection claims), and add a `verify` action for the ROC/FMR/FNMR task.

### `analysis/tests/` (3 test modules)

- `test_metrics.py`: pure-metric toy cases (`average_precision`, `reciprocal_rank`, `unique_identities`, `ann_recall`), FAISS wrapper atomic-add validation across `FaissBruteForce/FaissHNSW/FaissLSH`, `FaissSearch` k-capping/non-mutation/metric-mismatch, Minkowski rerank. **Model-agnostic. Port as-is.**
- `test_analysis.py`: condition-contract tests (11 conditions, `perturb` semantics), `combine` gate-vector logic, `tuning_mask`/`paired_delta` identity partitioning, `per_probe` metric separation, `cache_key` determinism, `table()` rendering, `validate_notebooks` failure modes, `benchmark()` recall/CI shape. **Mostly model-agnostic**; tests of `combine`/`crop_images` specifics change with the FIQA gate. **Port with small refactor.**
- `test_app.py`: Flask test-client contract tests for `/add` and `/identify` (unique-identity over-fetch, invalid k, empty gallery, grayscale upload, inference-failure 500, gated-preprocessing threshold behavior via mocked `detector`). **Coupled to `dim=512` and MTCNN `prepare_image`.** Port the contract tests; rewrite the gate test for FIQA.

### Root / infrastructure (notable)

- `test/unit_test.py`, `test/unit_test_2.py`, `test/grade_runner.py`: instructor autograder tests for `FaissHNSW`/`FaissLSH`/`FaissSearch`/`/add`/`/identify`. Drop (course-specific).
- `setup.sh`: downloads the dataset tarball (JHU SharePoint link) into `ironclad/storage`. **Do not port the link**; see Section 3.
- `Dockerfile`, `local_test.sh`, `.classroom50.yaml`, `.github/workflows/autograde.yaml`: course CI. Drop or replace with a plain Dockerfile + your own CI.
- `ironclad/storage/probe/sample_gallery.py`: the instructor's gallery/probe splitter (shuffles `instructor_utils/original/*`, copies `*_001.jpg`-style files into `probe/` and `multi_image_gallery/`). Evidence for dataset layout provenance (see Section 3).

---

## 2. Analysis task -> Grey-Resolve degraded-input ROC/FMR/FNMR benchmark mapping

| IronClad task | What it measures | Grey-Resolve mapping | What must change |
|---|---|---|---|
| `task1.ipynb` / `task1()` | Identification robustness of 2 weight sets under 11 conditions (clean, blur 1-3, resize 120/80/40, brightness 0.5/0.75/1.25/1.5); Top-1/mAP50/MRR/Hit@N + paired CIs + PCA scatter | **Degraded-input robustness benchmark (core)** | Swap weight sets to ArcFace vs AdaFace; keep the `CONDITIONS` grid and `perturb()`; add genuine/impostor score extraction per condition so each condition yields a score distribution, not just rankings |
| `task2.ipynb` / `task2()` | Quality-gate policy: baseline vs always-crop vs confidence-gated MTCNN crop, threshold tuned on identity split, heldout paired deltas | **FIQA gate benchmark (core)** | Replace MTCNN confidence with FIQA score in `crop_images`/`crop_confidence`/`combine`; re-derive thresholds on the score's own scale; keep baseline/crop-or-align/gated policy matrix, `tuning_mask` split, paired delta + bootstrap CI, recovered/regressed examples; add gate-operating-point analysis (FMR/FNMR vs FIQA threshold) |
| `task3.ipynb` / `task3()` | Index choice: HNSW/LSH/Flat tuning, p50/p95 latency, QPS, ANN recall@10, serialized size, build/RSS, scale extrapolation | **Index + SQLite sidecar benchmark** | Keep `benchmark()`/`tune_indices()`; add SQLite sidecar write/read latency and row-count integrity checks; note HNSW search now feeds both identification and verification (pairwise scoring needs exact scores — keep `FaissBruteForce` as oracle) |
| `task4.ipynb` / `task4()` | **CMC curve**: heldout Hit@N for N in (1,3,5,10,20,50), plateau deficit vs Hit@50, N-selection rule | **Verification extension -> ROC / FMR / FNMR (core new work)** | Task 4 is an identification "how many candidates" question. Grey-Resolve's Phase 1 deliverable is a *verification* benchmark: build genuine pairs (same identity, probe vs gallery) and impostor pairs (cross identity), score with cosine similarity per condition, then compute ROC (FMR vs 1-FNMR), FMR(t)/FNMR(t) curves, EER, and FMR@FNMR operating points. Keep the CMC/Hit@N analysis alongside (it still answers the candidate-count API question), and keep the bootstrap-CI + heldout discipline. New code needed: pair construction, score matrix, `roc`/`eer`/`fmr_fnmr` metrics (extend `utils/metrics.py`), DET-style plots in `draw_curves` |
| `task5.ipynb` / `task5()` | Enrollment saturation: Top-1 vs gallery images per identity (m=1..10, 30 seeds, fixed cohorts) | **Enrollment-quality analysis** | Port unchanged except backbone; optionally add "EER vs m" as the verification-flavored companion to Top-1 vs m |

Shared harness changes across all tasks: `MODELS` becomes the ArcFace/AdaFace pair; `encode()` and `crop_images()` are rewritten (alignment + FIQA); `per_probe` gains verification outputs or a sibling `per_pair()`; `bootstrap`/`paired_delta`/`tuning_mask`/`source_signature`/cache keys are reused as-is; `run.py` gains a 6th task or folds ROC/FMR/FNMR into task 4's notebook; `evidence.py` and `report.py` gate on the new artifacts.

Suggested Grey-Resolve benchmark artifact set (mirroring `analysis/results/`): `task4_scores_<model>_<condition>.npz` (genuine/impostor scores), `task4_roc.csv`, `task4_eer.json`, `task4_fmr_fnmr.csv`, `figures/task4_roc_<condition>.png`, `figures/task4_det.png`.

---

## 3. Dataset / loader assumptions and shipping constraints

**Layout and loaders observed:**

- `ironclad/storage/multi_image_gallery/<First_Last>/<First_Last>_NNNN.jpg` (2,265 real images, 1,000 identity dirs) and `ironclad/storage/probe/<First_Last>/<First_Last>_NNNN.jpg` (999 real images, 999 identity dirs). The split was made by `storage/probe/sample_gallery.py` from an `instructor_utils/original/<Identity>/*.jpg` tree: `*_002.jpg` and later went to gallery (renumbered from 001), `*_001.jpg` became the probe (renumbered 002). `audit_data()` asserts exactly these counts (gallery 2265 / probe 999) and fails on drift.
- The archive contains macOS AppleDouble sidecars (`._*.jpg`, 176 bytes); every loader (`audit_data`) filters `._*` explicitly. Any Grey-Resolve loader must do the same or ignore non-image suffixes.
- Naming and identities match the **LFW ("Labeled Faces in the Wild")** convention (`First_Last_0001.jpg` per identity folder). Note: "casia-webface" in this repo refers to one of the two `facenet_pytorch.InceptionResnetV1` pretrained weight sets (`MODELS = ("casia-webface", "vggface2")`), not to the image archive itself. The audit itself only claims "Provided course archive; collection/pretraining overlap not independently certified" — provenance is not certified anywhere in the repo. Treat the images as LFW-derived, research-only data. Both LFW and CASIA-WebFace (and VGGFace2) carry non-commercial, research-only terms; none of them can ship in a public portfolio repo.
- Dataset acquisition: `setup.sh` downloads `multi_image_identities.tar` from a JHU SharePoint link and extracts it. `analysis/common.py` raises `FileNotFoundError("Extract the supplied dataset into ...")` if the folder is missing.
- Cache formats in `analysis/cache/`:
  - `embeddings_<sha256>.npz`: `vectors` (float32 `[N, 512]`) + `metadata` (JSON: data signature, model, checkpoint file hash, condition, `cropped` flag, batch size, environment package versions, `preprocessing.py` file hash, `pipeline` version, timing, image count).
  - `crops_<sha256>/<row>.png` + `diagnostics.csv` (columns: `path, confidence, detected, detection_ms, x1, y1, x2, y2`).
  - Keys are SHA-256 of sorted JSON details; `source_signature()` re-hashes every source image and raises if the dataset changed, so caches can never silently mix datasets.
- Audit outputs in `analysis/results/`: `manifest.csv` (split, identity, path, sha256, pixel_sha256, width, height, brightness, contrast, dhash) and `dataset_audit.json` (identity counts, gallery/probe-only identities, exact pixel duplicate groups, dHash near-duplicate candidates with `cross_split` flags, quality summary, `provenance_limit`).

**What Grey-Resolve must do to avoid shipping the non-commercial dataset:**

1. **Do not copy `ironclad/storage/`, `analysis/cache/`, or any image into the new repo.** Also exclude `manifest.csv`/`dataset_audit.json` if you keep the same dataset (they are derived metadata of non-commercial data and contain identity names).
2. **Ship a fetch script, not data.** Replace `setup.sh` with a `scripts/fetch_data.py` (or documented manual step) that downloads from a *redistributable* source or from a user-supplied path/URL, verifies a checksum, and extracts to a git-ignored `data/` directory. Keep the dataset out of git (`.gitignore`) exactly as the assignment does.
3. **Prefer a license-clean dataset for the shipped benchmark.** Options: (a) keep LFW-style workflow but document "user must obtain LFW themselves under its research-only license"; (b) build the gallery/probe benchmark from a permissively licensed face set (e.g., a CC0/CC-BY set or synthetic/generated faces) so the repo can include a tiny committed smoke-test subset; (c) commit only synthetic placeholder images for CI and run the real benchmark locally. Whichever you choose, keep the `audit_data()` contract: expected counts, sidecar filtering, duplicate screening, and an explicit `provenance_limit` statement in every report.
4. **Keep the cache-signature discipline.** The `source_signature` / `cache_key` pattern is what makes results reproducible without shipping data or caches; port it unchanged and record dataset name + license + checksum in `environment.json`/`dataset_audit.json`.
5. **Document the license in README** (dataset terms, model weights terms — ArcFace/AdaFace weights also have non-commercial clauses in some releases; check the exact checkpoint you choose).

---

## 4. Port effort table

| File | Effort | Notes |
|---|---|---|
| `ironclad/app.py` | Medium | Route contract + over-fetch loop reusable; `prepare_image` MTCNN gate -> align + FIQA gate; SQLite sidecar replaces `index.metadata` |
| `ironclad/modules/extraction/embedding.py` | High (rewrite) | `InceptionResnetV1` -> ArcFace/AdaFace loader (InsightFace/AdaFace); keep `encode()` interface |
| `ironclad/modules/extraction/preprocessing.py` | High (rewrite) | 160px ImageNet-normalize -> 112px aligned ArcFace normalization; interface shape kept |
| `ironclad/modules/retrieval/search.py` | Low | Model-agnostic; swap `get_metadata` for SQLite lookup; drop or keep Minkowski rerank |
| `ironclad/modules/retrieval/index/hnsw.py` | Medium | Keep FAISS HNSW logic; SQLite metadata sidecar + `faiss.write_index` persistence instead of pickle/list |
| `ironclad/modules/retrieval/index/bruteforce.py` | Low | Needed as exact oracle; metadata -> SQLite |
| `ironclad/modules/retrieval/index/lsh.py` | Low / drop | Optional second approximate index; drop unless comparison is wanted |
| `ironclad/modules/retrieval/index/validation.py` | Low (as-is) | Pure validation helpers |
| `ironclad/utils/metrics.py` | Medium | Port ranking metrics as-is; add genuine/impostor pairing, ROC, FMR/FNMR, EER |
| `ironclad/design_config.json` | Low | Same schema; new model names, FIQA threshold scale, SQLite path, backend key |
| `analysis/common.py` | Medium | Rewrite `crop_images` (MTCNN->align+FIQA) and `encode` (checkpoints, 112px); everything else ports; add pair/score generation |
| `analysis/experiments.py` | High | `task2` rewritten for FIQA; `task4` extended CMC -> ROC/FMR/FNMR; `task1/3/5` small refactor; `benchmark`/`tune_indices` reusable |
| `analysis/report.py` | Medium | Mechanics (`table`, evidence gate, config write-back) kept; all narrative rewritten |
| `analysis/evidence.py` | Low (as-is) | Adjust artifact list/paths; add ROC artifacts to the gate |
| `analysis/run.py` | Low | Keep notebook generate/execute loop; edit `DESCRIPTIONS` and template prose; add verify action |
| `analysis/tests/test_metrics.py` | Low (as-is) | Model-agnostic; extend with ROC/FMR/FNMR tests |
| `analysis/tests/test_analysis.py` | Low-Medium | Port most; `combine`/gate tests change with FIQA policy |
| `analysis/tests/test_app.py` | Medium | Contract tests portable; gate test rewritten for FIQA; `dim=512` assumptions kept or parameterized |
| `setup.sh` / `local_test.sh` / `.classroom50.yaml` / `autograde.yaml` / `Dockerfile` | Drop / rewrite | Course infra; replace with own fetch script + CI; never port the dataset URL |
| `test/unit_test*.py`, `test/grade_runner.py` | Drop | Autograder-only |
| `ironclad/storage/**`, `analysis/cache/**`, `analysis/results/**`, `analysis/figures/**`, `task*.ipynb` outputs | Drop | Data/caches/results: non-commercial images and derived artifacts stay out of the repo |
| `CASE_ANALYSIS.md`, `README.md`, `IMPLEMENTATION_ASSIGNMENT.md` | Rewrite (reference only) | Use as structural reference for the Grey-Resolve write-up; content is course/facenet-specific |

**Summary:** the retrieval layer (search, index wrappers, validation), metric utilities, analysis harness skeleton (audit, perturbation grid, cache signatures, bootstrap, tuning split, notebook evidence loop, report gate) ports with little or low-effort change. The extraction layer (embedding + preprocessing) is a full rewrite for ArcFace/AdaFace. The quality-gate experiment (`task2`) and the MTCNN crop pipeline are rewritten around FIQA. The main new engineering is the verification metric layer: genuine/impostor pair construction, score distributions per degradation condition, and ROC/FMR/FNMR/EER computation + plotting extending `task4`.
