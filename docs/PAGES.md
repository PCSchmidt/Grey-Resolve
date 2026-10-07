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

## Content policy (do not weaken)

- No face imagery anywhere in the UI (procedural abstract tiles only) — model-weight and
  image licensing stay clean.
- The "SYNTHETIC DEMO -- RESEARCH PROTOTYPE" banner is persistent and not dismissible.
- Candidate-ranking semantics only; no identification or watchlist language.

## Rebuilding

- Figures / sanitized results: `python benchmarks/make_figures.py`
- Demo assets are hand-written JS in `docs/demo/`; no build required.
