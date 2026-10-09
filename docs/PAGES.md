# GitHub Pages hosting

The demo UI is a pure static page under `docs/` — no build step, no CI.

**URL:** `https://pcschmidt.github.io/Grey-Resolve/`

## One-time setup (repository settings)

GitHub → Settings → Pages → Build and deployment:
- Source: **Deploy from a branch**
- Branch: **main** / folder **/docs** → Save

The page is served from `docs/index.html`; `docs/figures/`, `docs/results/`, and
`docs/demo/` are served alongside it. After the first deploy the URL above resolves.

## What the demo shows

The **Resolution Console**: a synthetic, clearly-labeled visualization of Grey-Resolve's
mechanics — streaming media contacts with quality-gate decisions, a 3D embedding
manifold with near-tie resolution, live degradation modes, and telemetry sourced from
the sanitized real run artifacts in `docs/results/`. Everything is fabricated demo data
except the metric values, which trace to committed run outputs (see `docs/RESULTS.md`).

## Reading the Resolver (near-ties, gap <= 0.1)

The feed replays 13 scripted queries; 8 are near-ties. Each tie plays face bars, then
GEO/TIME/ENTITY context rows, then **FUSED** bars computed truthfully
(0.7 x face + 0.3 x mean context). The verdict is the real top-2 tiebreak rule applied
to those numbers, so the bars and the verdict never disagree:

| Verdict | Meaning |
|---|---|
| `CONTEXT CONFIRMED FACE #1` | clean context agrees with the face order; nothing changes |
| `CONTEXT FLIPPED FACE #1 -- CORRECT` | clean context overturns a wrong face #1 (Q-13) |
| `CONTEXT FLIPPED FACE #1 -- WRONG` | misleading context pulls a correct face #1 away (Q-09, Q-05) |
| `CONTEXT ABSENT -- FACE ORDER KEPT` | no context is no evidence; face order is untouched |

Correct and wrong flips are both shown on purpose: at scale they roughly cancel
(tiebreak hit@1 0.444 vs face-only 0.500 at gap <= 0.05, n=18; `docs/RESULTS.md`).
Timing: the feed holds a tie contact back until the resolver has finished the previous
tie (`VERDICT_AT_MS` + `VERDICT_HOLD_MS` in `panels.js`), so no verdict is cut off.
A headless run over two feed cycles reached all 8 verdicts with no page errors.

## Content policy (do not weaken)

- No face imagery anywhere in the UI (procedural abstract tiles only) — model-weight and
  image licensing stay clean.
- The "SYNTHETIC DEMO -- RESEARCH PROTOTYPE" banner is persistent and not dismissible.
- Candidate-ranking semantics only; no identification or watchlist language.

## Rebuilding

- Figures / sanitized results: `python benchmarks/make_figures.py`
- Demo assets are hand-written JS in `docs/demo/`; no build required.
