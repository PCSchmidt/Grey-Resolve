# Degraded-input ROC/FMR/FNMR benchmark

Phase 1 "operational evaluation" deliverable: verification metrics (FMR, FNMR,
ROC, EER) for face embeddings under controlled image degradations. Extends the
IronClad task4 CMC analysis to verification (docs/PORT_INVENTORY.md section 2).

Two layers:

- `src/grey_resolve/evaluation/experiment.py` -- pure, deterministic sweep
  functions (data in, data out; no file formats, no plotting).
- `benchmarks/evaluate_roc.py` -- thin argparse CLI over those functions.

## How to run

1. Fetch the backbone weights once (non-commercial; never committed):

   ```
   python scripts/fetch_backbone.py
   ```

2. Lay out the dataset as `<root>/<identity>/*.png|jpg` (one directory per
   identity, like the IronClad gallery). Every image must contain one
   detectable face.

3. Run the sweep:

   ```
   python benchmarks/evaluate_roc.py --data-root path/to/gallery [--limit N]
   ```

   `--limit N` caps images per identity for quick runs. `--severities 0.25 0.5
   0.75` is the default grid; `--quality-threshold` / `--operating-threshold`
   control the gating ablation. `--det-model` / `--rec-model` override the
   weight paths (default `~/.insightface/models`, or `$GREY_RESOLVE_MODEL_DIR`).

## Outputs

Each run writes `benchmarks/out/<run-stamp>/`:

- `results.json` -- full record: run config (dataset, counts, models,
  severities), one block per condition with `eer`, `fmr_at_fnmr_0.01`, the complete
  `roc_points` (`threshold`/`fmr`/`fnmr`), and pair counts; plus the
  `gated_vs_ungated` ablation (FMR/FNMR/EER for the full set vs the
  quality-passing subset at a shared operating threshold).
- `summary.csv` -- one row per condition: `condition,severity,eer,fmr_at_fnmr_0.01`.

Metric definitions live in `src/grey_resolve/evaluation/metrics.py`: FMR =
impostor pairs accepted at the threshold; FNMR = genuine pairs rejected; EER is
the interpolated FMR == FNMR point; `fmr_at_fnmr_0.01` is FMR at the lowest-FMR
operating point with FNMR <= 0.01. A pair is accepted when
`cosine_score >= threshold`. Scores are cosine similarities in [-1, 1].

Protocol per condition: the probe set is every clean image degraded with the
condition's operator at the condition's severity; the gallery is the same
images, clean. Genuine pairs are same-identity probe-vs-gallery combinations,
impostor pairs are cross-identity. The clean baseline (condition `clean`,
severity 0) is always included.

## Honest limitations

- **Synthetic/derived data only.** The degradations are deterministic synthetic
  operators (blur, brightness, resolution loss, projective off-angle). They
  approximate capture artifacts; they are not real sensor/lens/compression
  pipelines. The gating ablation uses the heuristic FIQA-lite assessor, which
  is a stand-in for measured face quality, not a perceptual-quality claim.
- **Severity is the operator's own parameter**, forwarded verbatim
  (`sigma` / `delta` / `scale` / `angle_deg`), so units differ per operator. For
  `downsample`, severity is the resolution *retention* factor: 0.25 is a
  stronger degradation than 0.75. The default grid values are therefore not
  comparable across operators.
- **Same-item genuine pairs are included.** Probe row i vs gallery row i (the
  same source file, degraded vs clean) counts as a genuine pair, which is
  optimistic when an identity has few images. Use `--limit`-free datasets with
  several images per identity to dilute this effect.
- **Detection misses abort the run.** The embed adapter raises on an image with
  no detected face instead of silently dropping it.
- **No plotting and no confidence intervals** in this artifact set (CSV/JSON
  only); bootstrap CIs and DET plots belong to the analysis-notebook layer
  (PORT_INVENTORY task4 mapping).
- Metrics are computed on all pairs; no held-out threshold tuning is performed
  here (the `fmr_at_fnmr_0.01` operating point is derived from the same scores).
