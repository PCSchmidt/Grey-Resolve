# Synthetic Scenario Dataset — Spec (Phase 2)

Status: **implemented** (`src/grey_resolve/scenario/generator.py`, config in
`configs/scenario_v0.yaml`; the evaluated outcome is in `docs/RESULTS.md` findings 6-9).
This document specifies the fabricated scenario dataset that Phase 2's fusion scorer was
developed and evaluated on. **Everything in this dataset is synthetic.**
No real persons, real identities, real incidents, or real intelligence data appear in it.

## Purpose

Phase 2 evaluates `P(identity) = alpha*S_face + beta*S_context`. This requires media items
with (a) face imagery degraded in controlled ways and (b) contextual metadata with
realistic structure — timestamps, coarse geo, source platforms, text entities — where
context is *informative but imperfect*. The dataset exists only so the fusion scorer can be
ablated honestly (face-only vs fused) on a fixed, reproducible scenario set.

## Design requirements

1. **Fully generated, deterministic.** One command produces the dataset from a fixed seed.
   Output is versioned and content-hashed.
2. **Fabricated identities.** "Identities" are generated personas: synthetic face imagery
   (procedural or from permissively-licensed synthetic-face generators such as
   StyleGAN-generated faces / generated.photos-style sources — license documented per
   source) plus a fabricated biography. Names are clearly fictional.
3. **Informative-but-imperfect context.** Context signals must correlate with identity at a
   tunable strength; hard-coding `S_context` to always agree with the face label would make
   the ablation meaningless.
4. **Provenance labeling.** Every item carries `synthetic: true` and a generator version
   stamp. Downstream code must never mistake this data for real data.

## Schema

### Identity record
```json
{
  "identity_id": "syn-id-0001",
  "display_name": "Fabricated Persona 1",
  "synthetic": true,
  "archetype": "maritime-logistics-figure",
  "home_geo_cluster": "geo-alpha",
  "active_period": ["2024-01-01", "2024-06-30"],
  "known_associates": ["syn-id-0007", "syn-id-0012"],
  "text_aliases": ["alias-one", "alias-two"]
}
```

### Media item
```json
{
  "media_id": "syn-media-000001",
  "identity_id": "syn-id-0001",
  "synthetic": true,
  "image_ref": "images/000001.png",
  "degradation": {"type": "blur", "severity": 0.4},
  "timestamp": "2024-03-14T09:20:00Z",
  "geo_cluster": "geo-alpha",
  "source_platform": "platform-a",
  "text_entities": ["alias-one", "geo-alpha"],
  "generator_version": "0.1.0",
  "seed": 42
}
```

## Scenario structure

- **N identities** (default 30) across **G geo clusters** (default 5) and
  **T active-period windows** (default 4 quarters).
- **M media items per identity** (default 20), split:
  - clean reference shots (10-20%),
  - degraded shots using `data/synthetic_noise/` operators (blur, brightness,
    downsampling, off-angle) at graded severities,
  - ambiguous-context shots (see below).
- **Ambiguity injection** (the core experimental control):
  - *Near-tie face sets*: clusters of 2-3 visually similar synthetic identities whose
    clean-embedding cosine lands in the 0.65-0.75 band — the regime where context should
    decide.
  - *Context noise*: a tunable fraction of items get wrong/stale geo, timestamp jitter,
    or misleading text entities (default 15%; swept in experiments).
  - *Context absence*: a fraction of items carry no context at all (default 20%) so the
    scorer cannot always lean on it.
- **Query set**: held-out media items with the same identity distribution and degradation
  profile; queries never share exact media files with the index set.

## Generation pipeline

```text
generate_scenarios.py --seed 42 --config configs/scenario_v0.yaml --out data/scenarios/v0/
  1. sample personas (identities, geo, periods, associates)
  2. generate/assign synthetic face imagery per identity
  3. apply degradation operators with sampled severities
  4. sample context metadata with tunable signal strength + noise
  5. inject near-tie ambiguity sets
  6. write manifest.json (schema version, seed, config hash, file hashes)
```

Outputs: `identities.json`, `media.jsonl`, `images/`, `manifest.json`.
The manifest config hash is embedded in every benchmark result for traceability.

## Evaluation use

- **Primary ablation**: face-only ranking vs fused ranking on the query set
  (metrics: top-1 accuracy on near-tie subsets, MRR, FMR@fixed FNMR).
- **Signal-strength sweep**: context signal strength and noise rate are swept to show
  *when* fusion helps and when it hurts. Reporting the negative region is required.
- **What this cannot show**: any real-world accuracy. Results are properties of a
  generator we control. The report and README must state this wherever numbers appear.

## Open questions (resolved)

All three open questions are resolved by design decision (Phase 2 implementation):

1. **The repo ships NO face imagery -- not even synthetic/GAN faces.** Synthetic
   face images (StyleGAN / generated.photos-style) carry murky licenses. Instead the
   generator fabricates ALL personas and context metadata on top of **user-supplied
   local face images**: paths referenced in place, or copied by the user into
   `data/scenarios/` (gitignored). The generator never writes, copies, or
   redistributes pixels; caller-held numpy arrays are referenced as `inline/NNNNNN`.
2. **Near-tie ambiguity sets are SELECTED, not faked.** `select_near_ties` measures
   embedding cosine between the user-supplied images and keeps cross-persona pairs
   whose cosine lands in the measured 0.65-0.75 band from
   `configs/threshold_profiles.yaml` (grouped into 2-3-persona ambiguity sets). This
   is more realistic than generated look-alike faces and stays license-clean.
3. **Text entities are template-generated strings** from the fictional template
   vocabulary in `configs/scenario_v0.yaml`; the committed scenario content is
   **parameters + seeds only** (the config file). Generated outputs
   (`identities.json`, `media.jsonl`, `manifest.json`) live under `data/scenarios/`
   (gitignored) and are reproducible from the config seed.

Everything generated is marked `synthetic: true` and stamped with the generator
version, seed, and config hash (see `src/grey_resolve/scenario/generator.py`).

## Provenance

Written as part of Grey-Resolve Phase 1 planning. Adapted in spirit from the Phase 2
design in `PLAN.md`. Update this document before implementing `src/grey_resolve/fusion/`.
