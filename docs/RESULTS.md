# Grey-Resolve — Phase 1 Preliminary Results

Research prototype. Aggregate metrics from synthetic/local research data only; no real
operational data. Candidate ranking evaluation — not identity-assertion accuracy.

**Run:** `benchmarks/out/20261006T175659Z` (2026-10-06). Reproduce with:
`python benchmarks/evaluate_roc.py --data-root <gallery> --max-identities N --limit K`.
Raw outputs live in `benchmarks/out/` (gitignored); this file records aggregate metrics only.

**Setup.** Backbone: InsightFace SCRFD detection + w600k_r50 ArcFace (512-d). Quality
gate: heuristic FIQA-lite (sharpness/exposure/size), gate threshold 0.5, operating
threshold 0.5. Data: 50 images / 10 identities from the LFW-derived local course gallery
(research-only; never redistributed). Probe-vs-gallery pairs per
`docs/PORT_INVENTORY.md` §2. This subset is small and easy — see limitations.

## Degraded-input sweep (EER per condition)

| Condition | Severity | EER | FMR@FNMR=1% | Scored probes |
|---|---|---|---|---|
| clean | — | 0.000 | 0.000 | 50/50 |
| brightness | Δ=0.25 | 0.000 | 0.000 | 50/50 |
| brightness | Δ=0.50 | 0.000 | 0.000 | 50/50 |
| brightness | Δ=0.75 | 0.028 | 0.039 | 33/50 (17 detection failures) |
| downsample | ×0.5 | 0.000 | 0.000 | 50/50 |
| downsample | ×0.25 | 0.000 | 0.000 | 50/50 |
| downsample | ×0.1 | **0.100** | **0.628** | 45/50 |
| gaussian_blur | σ=1 | 0.000 | 0.000 | 50/50 |
| gaussian_blur | σ=3 | 0.000 | 0.000 | 50/50 |
| gaussian_blur | σ=8 | **0.069** | **0.267** | 42/50 |
| off_angle | 15° | 0.000 | 0.000 | 50/50 |
| off_angle | 30° | 0.000 | 0.000 | 50/50 |
| off_angle | 45° | 0.000 | 0.000 | 50/50 |

Severity is each operator's own strength parameter; the per-operator grid exists
because a shared scale is meaningless (an earlier run with severities 0.25/0.5/0.75
for every operator showed zero effect for blur σ=0.75px and rotation 0.75°).

## Quality-gate ablation (FMR/FNMR at fixed threshold 0.5)

| Condition | Full FNMR | Gated FNMR | Gate pass rate | Dropped |
|---|---|---|---|---|
| clean | 0.056 | **0.036** | 0.94 | 0 |
| brightness Δ=0.25 | 0.080 | 0.060 | 0.54 | 0 |
| brightness Δ=0.50 | 0.208 | (too few pass) | 0.02 | 0 |
| downsample ×0.25 | 0.228 | **0.104** | 0.46 | 0 |
| gaussian_blur σ=1 | 0.064 | **0.000** | 0.54 | 0 |
| gaussian_blur σ=3 | 0.376 | **0.000** | 0.04 | 0 |
| off_angle 15° | 0.068 | 0.045 | 0.94 | 0 |
| off_angle 45° | 0.096 | 0.100 | 0.98 | 0 |

**Finding 1 — the gate works as designed at the operating point.** The quality-passing
subset's FNMR is lower than the full set in almost every condition and never materially
worse; where quality is genuinely bad (heavy blur/brightness), the gate rejects most
probes instead of letting bad matches through.

**Finding 2 — detection failure is a first-class outcome.** At harsh degradation SCRFD
detects no face at all (17/50 at brightness Δ=0.75, 8/50 at blur σ=8). The harness
excludes and counts these probes per condition (`n_query_dropped`) rather than aborting.

**Finding 3 — 2D rotation is not a stressor after alignment.** `off_angle` (homography
rotation up to 45°) has near-zero effect because SCRFD re-detects and `norm_crop`
re-aligns the face. A meaningful pose stressor needs yaw/pitch (3D) or genuinely
off-angle imagery — recorded as future work.

**Finding 4 — ranking quality ≠ operating-point calibration.** EER can be 0.0 while
FNMR at threshold 0.5 is high (downsample ×0.25: EER 0.0, FNMR 0.228): degradation
compresses score magnitude without breaking ranking. Operational thresholds must be
calibrated per condition — this is exactly why the ROC/FMR/FNMR framing replaces the
course project's CMC Hit@N.

**Finding 5 — this dataset cannot show FMR effects (honest limitation).** FMR = 0 at
threshold 0.5 in every condition; 10 well-lit frontal identities are too easy for
impostor comparisons against ArcFace. The "gate improves FMR at fixed FNMR" claim needs
a harder, near-tie set — that is the Phase 2 synthetic scenario dataset's job
(`docs/SYNTHETIC_SCENARIO_SPEC.md`), not something to fake on easy data.

## Limitations (read before quoting any number)

- 50 images / 10 identities: coarse EER estimates; treat as pipeline validation, not
  benchmark-grade numbers. A larger run is in progress.
- Single gallery draw, one seed; no confidence intervals yet.
- Heuristic FIQA-lite gate, not measured FIQA (MagFace assessor is planned).
- All data is LFW-derived research data or synthetic; nothing here transfers to
  real-world operational accuracy claims.
