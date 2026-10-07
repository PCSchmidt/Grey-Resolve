# Grey-Resolve — Phase 1 Results

Research prototype. Aggregate metrics from local research data only; no real operational
data. Candidate-ranking evaluation — not identity-assertion accuracy.

**Primary run:** `benchmarks/out/20261006T191330Z` (2026-10-06): 138 images / 100 identities
from the LFW-derived local course gallery (research-only; never redistributed).
Reproduce: `python benchmarks/evaluate_roc.py --data-root <gallery> --max-identities 100 --limit 2`.
Raw outputs live in `benchmarks/out/` (gitignored); this file records aggregate metrics only.
An earlier 50-image pilot run (`20261006T175659Z`) is superseded except where noted.

**Setup.** Backbone: InsightFace SCRFD detection + w600k_r50 ArcFace (512-d). Quality
gate: heuristic FIQA-lite (sharpness/exposure/size), gate threshold 0.5, operating
threshold 0.5. Probe-vs-gallery pairs per `docs/PORT_INVENTORY.md` §2.

## Degraded-input sweep (EER per condition)

| Condition | Severity | EER | FMR@FNMR=1% | Scored probes |
|---|---|---|---|---|
| clean | — | 0.000 | 0.000 | 137/138 |
| brightness | Δ=0.25 | 0.000 | 0.000 | 137/138 |
| brightness | Δ=0.50 | 0.005 | 0.000 | 137/138 |
| brightness | Δ=0.75 | 0.022 | 0.150 | 96/138 (42 detection failures) |
| downsample | ×0.5 | 0.000 | 0.000 | 137/138 |
| downsample | ×0.25 | 0.005 | 0.000 | 137/138 |
| downsample | ×0.1 | 0.046 | **0.873** | 119/138 |
| gaussian_blur | σ=1 | 0.005 | 0.000 | 137/138 |
| gaussian_blur | σ=3 | 0.005 | 0.000 | 137/138 |
| gaussian_blur | σ=8 | **0.078** | **0.547** | 126/138 |
| off_angle | 15° | 0.000 | 0.000 | 137/138 |
| off_angle | 30° | 0.000 | 0.000 | 137/138 |
| off_angle | 45° | 0.014 | 0.256 | 137/138 |

Severity is each operator's own strength parameter; the per-operator grid exists because
a shared scale is meaningless (a pilot run at shared severities 0.25/0.5/0.75 showed zero
effect for blur σ=0.75px and rotation 0.75°).

## Quality-gate ablation (at fixed operating threshold 0.5)

| Condition | Full FNMR | Gated FNMR | Gate pass rate | Dropped |
|---|---|---|---|---|
| clean | 0.028 | **0.020** | 0.94 | 1 |
| brightness Δ=0.25 | 0.028 | **0.018** | 0.56 | 1 |
| brightness Δ=0.50 | 0.108 | **0.000** | 0.04 | 1 |
| downsample ×0.5 | 0.028 | **0.012** | 0.83 | 1 |
| downsample ×0.25 | 0.108 | **0.047** | 0.47 | 1 |
| gaussian_blur σ=1 | 0.033 | **0.011** | 0.50 | 1 |
| gaussian_blur σ=3 | 0.127 | **0.000** | 0.09 | 1 |
| off_angle 15° | 0.028 | **0.020** | 0.93 | 1 |
| off_angle 30° | 0.033 | **0.025** | 0.93 | 1 |
| off_angle 45° | 0.047 | 0.049 | 0.96 | 1 |

Extreme conditions (brightness Δ=0.75, downsample ×0.1, blur σ=8) drive the gate's pass
rate to ~0; with too few passing probes the gated metrics are reported as `None` rather
than estimated from noise. FMR = 0 in every cell (see Finding 5).

## Findings

**1 — The quality gate works as designed at the operating point.** The quality-passing
subset's FNMR is lower than the full set in nearly every condition and never materially
worse (the sole tie-level exception, off_angle 45° at 0.047→0.049, is noise-level). Where
quality is genuinely bad the gate rejects almost everything instead of letting bad
matches through.

**2 — Detection failure is a first-class outcome.** SCRFD finds no face at all under harsh
degradation (42/138 probes at brightness Δ=0.75, 12/138 at blur σ=8). The harness excludes
and counts these per condition (`n_query_dropped`) instead of aborting.

**3 — Rotation hurts mildly on a varied set (revised).** The 50-image pilot showed zero
effect up to 45°; at 100 identities, 45° gives EER 0.014 / FMR@FNMR=1% 0.256. Landmark
re-alignment (SCRFD + `norm_crop`) neutralizes much of 2D rotation, but not all of it on
more varied imagery. A hard pose stressor still needs yaw/pitch or genuinely off-angle
imagery — future work.

**4 — Ranking quality ≠ operating-point calibration.** EER stays near zero while
FMR@FNMR=1% explodes under severe degradation (downsample ×0.1: EER 0.046 but
FMR@FNMR=1% = 0.873). Degradation compresses score magnitude without fully breaking
ranking; operational thresholds must be calibrated per condition. This is why the
ROC/FMR/FNMR framing replaces the course project's CMC Hit@N.

**5 — This dataset cannot show FMR effects (honest limitation).** FMR = 0 at threshold
0.5 in every condition, even at 100 identities — well-lit frontal faces are too easy for
impostor comparisons against ArcFace. The "gate improves FMR at fixed FNMR" claim needs
hard, near-tie comparisons; that is the Phase 2 synthetic-scenario near-tie evaluation
(`docs/SYNTHETIC_SCENARIO_SPEC.md`), not something to fake on easy data.

## Limitations (read before quoting any number)

- Single gallery draw, one seed; no confidence intervals yet.
- 138 images / 100 identities (many identities have a single image): still coarse for
  per-condition EER; treat as strong pipeline evidence, not a benchmark-suite number.
- Heuristic FIQA-lite gate, not measured FIQA (MagFace assessor is planned).
- All data is LFW-derived research data; nothing here transfers to real-world operational
  accuracy claims.

---

# Phase 2 — Fusion Ablation (first evidence run, 2026-10-06)

Run `benchmarks/out/20261006T225702Z`. 100 real identities (LFW-derived local gallery,
2 images each), 30 synthetic personas with disjoint source images, 60 fabricated media
items, adaptive near-tie band [0.129, 1.0] at the 98th cross-persona-cosine quantile
(5 near-tie sets). Profile `strict_surveillance` (alpha=0.7, beta=0.3). Persona-level
ranking (candidates = personas; face score = max over the persona's non-query items).

| Scope | Face-only hit@1 | Fused hit@1 | Δ | MRR (face → fused) |
|---|---|---|---|---|
| overall (60 queries) | 1.000 | 1.000 | +0.000 | 1.000 → 1.000 |
| near-tie (20 queries) | 1.000 | 1.000 | +0.000 | 1.000 → 1.000 |

Context-noise sweep (rates 0.0–0.5): fused hit@1 stays 1.000, delta +0.000 at every rate.

**Finding 6 — Negative result: context fusion adds nothing on clean face evidence.**
Face-only resolves every query, so there is nothing for context to break. The diagnostic
explains why: "near-tie" sets were selected by persona-mean cosine, but at query time the
true persona's face-score gap is large (mean 0.69, median 0.81 over the near-tie queries).
These are not actual ties when the query image is a clean photo. Per the scenario spec,
this negative region is reported plainly rather than suppressed.

**Interpretation — the fusion hypothesis is untested, not falsified.** The ambiguity
regime never materialized on clean queries. Phase 1 shows where it lives: under
degradation, face evidence genuinely degrades (downsample ×0.1 → FMR@FNMR=1% = 0.873;
blur σ=8 → 0.547). The ablation must run with (a) degraded query images — the scenario
records already carry degradation descriptors — and (b) query-time tie selection
(queries whose top-2 face-score gap is small), not just persona-mean near-ties.

**Tooling findings from this run.** (i) The planning-doc near-tie band (0.65–0.75 cosine)
is empty for ArcFace — cross-persona cosines on this data top out near 0.16; the band is
now resolved adaptively at the 98th quantile of measured cross-persona cosines.
(ii) If synthetic personas share source images, "ties" are duplicate-photo artifacts
(cosine exactly 1.0) and the face-only baseline always loses them; the CLI now warns and
the evidence run uses disjoint source images per persona.

## Degraded-query ablation (2026-10-07, runs 012020Z / 012245Z / 012503Z)

Same seeded scenario (30 personas, disjoint source images) as the clean run above; query
photos degraded before embedding while gallery evidence stays clean. The face-gap subsets
(`gap_le`) restrict to queries whose top-1-vs-top-2 face-score gap is <= the cutoff --
the actual-tie regime.

| Query condition | Scope | Face-only hit@1 | Fused hit@1 | Δ |
|---|---|---|---|---|
| downsample ×0.25 | overall | 1.000 | 1.000 | +0.000 |
| blur σ=3 | overall | 1.000 | 1.000 | +0.000 |
| downsample ×0.1 | overall | 0.904 | 0.865 | **−0.038** |
| downsample ×0.1 | gap ≤ 0.2 (n=26) | 0.808 | 0.808 | +0.000 |
| downsample ×0.1 | gap ≤ 0.1 (n=9) | 0.667 | 0.778 | **+0.111** |
| downsample ×0.1 | gap ≤ 0.05 (n=4) | 0.500 | 0.750 | **+0.250** |

MRR on the tightest slice: 0.559 → 0.760. Detection failures under ×0.1: 18/60 (excluded
and counted). Context-noise sweep (×0.1): fused delta vs face-only is −0.038 at noise 0.0
and degrades to −0.115 at noise 0.4.

**Finding 7 — Small-run positive, later overturned (kept for honesty).** On the small
run, fusion beat face-only on tight-gap queries (gap <= 0.05: +0.25 hit@1 at n=4;
gap <= 0.1: +0.11 at n=9) while slightly hurting overall (−0.038). That positive was a
small-n fluke -- see Finding 8. The negative parts of Finding 7 did hold up: fusion is
inert where face evidence is clear, and mild degradation (downsample ×0.25, blur σ=3)
never creates ambiguity at all (face gaps ~0.55-0.59), consistent with Finding 4.

**Design consequence — fusion must be selective.** Rank by face outside the ambiguous
band, fuse inside it. Selective gap-gated fusion was implemented and tested
(`--selective-gap`); its effectiveness depends entirely on fusion being net-positive in
the tied regime, which Finding 8 calls into question.

Caveats from the small run: the tight-gap slices were tiny (n=4 and n=9). The persona-mean
near-tie sets did NOT correspond to query-time ties (near-tie Δ = 0.000), confirming that
gap-based subsetting, not mean-embedding clustering, defines the relevant regime.

## Scale run: the small-run positive reverses (2026-10-07, runs 111729Z / 113329Z)

2x scale (200 real identities / 60 personas / 120 items, same seed and protocol,
`--selective-gap 0.1`). Downsample ×0.1 queries: 230 embedded, 32 detection failures.

| Method | overall hit@1 | gap <= 0.1 (n=27) | gap <= 0.05 (n=18) |
|---|---|---|---|
| face-only | **0.906** | **0.630** | **0.500** |
| fused | 0.774 (−0.132) | 0.407 (−0.222) | **0.167 (−0.333)** |
| selective (gate 0.1) | 0.849 (−0.057) | — | — |

Control (downsample ×0.25): all methods ~0.983, fusion inert, gaps ~0.59. Context-noise
sweep at ×0.1: fused delta degrades from −0.132 to −0.208 (noise 0.5); selective −0.057 to
−0.085. "Fusion stops helping" is 0.0 for every scope.

**Finding 8 — Honest negative at scale: this context model does not resolve face ties.**
At n=18 genuine tight-gap queries, fused ranking is *below* face-only by −0.333 hit@1
(0.500 -> 0.167) -- worse than the small run suggested, in the opposite direction. Fusion
fails below the face-only baseline precisely where context was supposed to decide, and
selective gating (face path outside gap <= 0.1) reduces but does not eliminate the damage
(overall −0.057). Two candidate explanations to diagnose: (a) `context_score` handling of
absent/drifted/noisy query context is misleading (only ~55% of queries carry clean
truthful context at the configured rates), (b) score-scale mismatch -- under
alpha=0.7/beta=0.3 a full context swing (1.0) overrides a face gap up to ~0.43, far wider
than the tied regime the gate selects. Until the mechanism is diagnosed and fixed, the
claim "context breaks face near-ties" is NOT supported on this data; what remains
supported: the degradation/quality-gate findings (1-4), the tie-regime *definition*
(gap-based subsetting), and the value of honest negative reporting (Findings 6-8).
