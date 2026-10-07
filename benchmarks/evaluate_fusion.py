"""Fusion ablation benchmark CLI (face-only vs fused ranking on near-tie sets).

Loads a labeled image dataset laid out as ``<root>/<identity>/*.png|jpg`` -- the
same loader as ``benchmarks/evaluate_roc.py`` (imported, not duplicated) --
embeds it with the Grey-Resolve adapters (SCRFD detection + InsightFace ArcFace),
builds the deterministic synthetic scenario on top of those local images
(``grey_resolve.scenario.generator``: fabricated personas/context; the repo
ships no face imagery), selects near-tie ambiguity sets by MEASURED embedding
cosine, and runs the Phase 2 ablation:

- ``run_ambiguity_ablation``: face-only vs fused persona ranking (hit@1, hit@k,
  MRR) overall, on near-tie queries, and per near-tie set, plus the face-score
  gap summary.
- ``run_context_noise_sweep``: query-context corruption rates 0.0..0.5 with the
  delta vs face-only -- the negative region is reported, never suppressed.
- optional ``--weight-ratios``: small beta/alpha sweep (``run_context_weight_sweep``).

Outputs under ``benchmarks/out/<run-stamp>/``:
- ``results.json``  -- config (incl. seed + scenario config hash), all metrics
- ``summary.csv``   -- ablation rows + one row per noise rate
- ``scenario/``     -- the generated synthetic scenario + manifest (traceability)

No plotting: CSV/JSON only. Backbone weights must be fetched first
(``python scripts/fetch_backbone.py``). Metrics are properties of the generator
we control -- they do NOT transfer to real-world accuracy claims
(docs/SYNTHETIC_SCENARIO_SPEC.md "Evaluation use").

Usage:
    python benchmarks/evaluate_fusion.py --data-root path/to/gallery [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from evaluate_roc import (  # same-dir module, script-style CLI (sys.path[0] = benchmarks/)
    MODEL_FILES,
    _build_adapters,
    _FaceEmbedder,
    _model_dir,
    load_dataset,
)

from grey_resolve.config import ThresholdProfile, load_threshold_profiles
from grey_resolve.evaluation.fusion_experiment import (
    RankingItem,
    run_ambiguity_ablation,
    run_context_noise_sweep,
    run_context_weight_sweep,
)
from grey_resolve.fusion.linear import LinearFusionScorer
from grey_resolve.scenario.generator import (
    config_hash,
    fabricate_media,
    generate_identities,
    load_scenario,
    load_scenario_config,
    select_near_ties,
    write_scenario,
)

DEFAULT_NOISE_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
DEFAULT_K = 5
CSV_HEADER = [
    "row",
    "method",
    "scope",
    "noise_rate",
    "hit_at_1",
    "hit_at_k",
    "mrr",
    "n_queries",
    "delta_hit_at_1",
    "delta_mrr",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True, help="dataset dir: <root>/<identity>/*.png|jpg")
    parser.add_argument("--limit", type=int, default=None, help="cap images per identity (quick runs)")
    parser.add_argument(
        "--max-identities",
        type=int,
        default=None,
        help="cap the number of identity dirs, in sorted order (quick runs)",
    )
    parser.add_argument("--out-dir", default="benchmarks/out", help="output root for run directories")
    parser.add_argument("--seed", type=int, default=42, help="scenario + noise-sweep seed")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="hit@k cutoff for the ranking metrics")
    parser.add_argument("--profile", default="strict_surveillance", help="threshold profile name")
    parser.add_argument(
        "--near-tie-quantile",
        type=float,
        default=0.98,
        help="adaptive near-tie band: lower edge at this quantile of measured "
        "cross-persona cosines (0 disables; the config band is then used)",
    )
    parser.add_argument("--n-identities", type=int, default=None, help="override scenario persona count")
    parser.add_argument(
        "--query-degradation",
        nargs=2,
        default=None,
        metavar=("NAME", "SEVERITY"),
        help="degrade QUERY pixels with this operator/severity before embedding "
        "(gallery evidence stays clean); the face-ambiguity regime for the fusion ablation",
    )
    parser.add_argument(
        "--gap-thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2],
        help="face-gap cutoffs for the tight-gap ('gap_le') ablation subsets",
    )
    parser.add_argument(
        "--media-per-identity",
        type=int,
        default=None,
        help="override scenario media count per persona (keep <= images per persona source pool)",
    )
    parser.add_argument(
        "--near-tie-band",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="explicit near-tie band (overrides --near-tie-quantile and the config band)",
    )
    parser.add_argument(
        "--scenario-config",
        default="configs/scenario_v0.yaml",
        help="synthetic scenario config YAML (seed is overridden by --seed)",
    )
    parser.add_argument(
        "--noise-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_NOISE_RATES),
        help="context noise rates for the sweep (default: 0.0 0.1 ... 0.5)",
    )
    parser.add_argument(
        "--weight-ratios",
        type=float,
        nargs="+",
        default=None,
        help="optional small sweep of beta/alpha ratios (alpha=1)",
    )
    parser.add_argument("--det-model", default=None, help=f"SCRFD weights (default: {_model_dir() / MODEL_FILES[0]})")
    parser.add_argument("--rec-model", default=None, help=f"ArcFace weights (default: {_model_dir() / MODEL_FILES[1]})")
    return parser.parse_args(argv)


def _embed_images(images: Sequence, embed_fn: _FaceEmbedder) -> tuple[dict[int, object], int]:
    """Embed every image once; return (index -> vector for kept images, n_dropped).

    Images where face detection fails have no embedding and are excluded (and
    counted) instead of failing the run.
    """
    vectors: dict[int, object] = {}
    for idx, image in enumerate(images):
        vec = embed_fn(image)
        if vec is not None:
            vectors[idx] = vec
    return vectors, len(images) - len(vectors)


def _persona_embeddings(media, emb_by_index: dict[int, object]) -> tuple[list, list[str]]:
    """Mean embedding per persona over its media items' embedded source images."""
    import numpy as np

    per_persona: dict[str, dict[int, object]] = {}
    for m in media:
        ref = m.image_ref
        if not ref.startswith("inline/"):
            continue
        idx = int(ref.split("/", 1)[1])
        vec = emb_by_index.get(idx)
        if vec is None:
            continue
        per_persona.setdefault(m.identity_id, {})[idx] = vec
    persona_ids: list[str] = []
    means: list = []
    for pid in sorted(per_persona):
        rows = [np.asarray(v, dtype=np.float64) for _, v in sorted(per_persona[pid].items())]
        mean = np.mean(np.stack(rows), axis=0)
        if float(np.linalg.norm(mean)) == 0.0:
            continue
        persona_ids.append(pid)
        means.append(mean)
    return means, persona_ids


def _fmt(value: object) -> str:
    return "" if value is None else str(value)


def _warn_on_shared_assets(media, n_labels: "int | None") -> None:
    """Warn when distinct personas reference the same source images.

    Shared assets create duplicate persona evidence (cosine exactly 1.0), which
    makes near-tie results degenerate: "ties" between personas are then
    identical-photo artifacts, not look-alike ambiguity. Avoid by keeping
    n_identities <= number of real label groups and media_per_identity <= the
    smallest group size.
    """
    refs_by_persona: dict[str, set] = {}
    for m in media:
        refs_by_persona.setdefault(m.identity_id, set()).add(m.image_ref)
    owners: dict = {}
    for pid, refs in refs_by_persona.items():
        for ref in refs:
            owners.setdefault(ref, set()).add(pid)
    shared = {ref: pids for ref, pids in owners.items() if len(pids) > 1}
    if shared:
        n_personas = len(refs_by_persona)
        print(
            f"warning: {len(shared)} source images are shared across personas "
            f"({n_personas} personas from"
            + (f" {n_labels} label groups" if n_labels is not None else " an unlabeled pool")
            + ") -- near-tie results may be degenerate duplicate-photo artifacts; "
            "reduce --n-identities or --media-per-identity"
        )


def _resolve_near_tie_band(vectors, persona_ids, args, fallback_band) -> tuple[float, float]:
    """Pick the near-tie band: explicit flag > adaptive quantile > config band.

    The adaptive default exists because ArcFace cross-persona cosines on
    LFW-derived data are far lower than legacy facenet scales (observed 99th
    percentile ~0.16): a fixed 0.65-0.75 band selects nothing. "Near-tie" is
    therefore defined relative to the measured distribution -- the hardest
    cross-persona comparisons in the set.
    """
    if args.near_tie_band is not None:
        low, high = float(args.near_tie_band[0]), float(args.near_tie_band[1])
        if not (low < high):
            raise SystemExit("error: --near-tie-band must satisfy LOW < HIGH")
        return (low, high)
    q = float(args.near_tie_quantile)
    if q > 0:
        if q >= 1.0:
            raise SystemExit("error: --near-tie-quantile must be in [0, 1)")
        import numpy as np

        from grey_resolve.evaluation.metrics import build_pairs

        try:
            scores, is_genuine = build_pairs(vectors, persona_ids)
        except ValueError:
            return tuple(float(v) for v in fallback_band)
        cross = scores[~is_genuine]
        if cross.size:
            # clamp below 1.0 so the band is always valid even when the
            # hardest cross-persona pairs are near-duplicates (cosine = 1.0)
            low = min(float(np.quantile(cross, q)), 1.0 - 1e-9)
            return (low, 1.0)
    return tuple(float(v) for v in fallback_band)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = _model_dir()
    det_model = Path(args.det_model) if args.det_model else model_dir / MODEL_FILES[0]
    rec_model = Path(args.rec_model) if args.rec_model else model_dir / MODEL_FILES[1]

    try:
        images, labels = load_dataset(
            args.data_root, args.limit, max_identities=args.max_identities
        )
        detector, extractor, _assessor = _build_adapters(det_model, rec_model)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc  # includes the fetch-backbone hint

    embed_fn = _FaceEmbedder(detector, extractor)
    print(f"loaded {len(images)} images / {len(set(labels))} identities from {args.data_root}")
    print("embedding images ...")
    emb_by_index, n_embed_dropped = _embed_images(images, embed_fn)
    print(f"  embeddings: {len(emb_by_index)}/{len(images)} images ({n_embed_dropped} detection failures)")

    try:
        scenario_config = load_scenario_config(args.scenario_config)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    overrides = {"seed": int(args.seed)}
    if args.n_identities is not None:
        overrides["n_identities"] = int(args.n_identities)
    if args.media_per_identity is not None:
        overrides["media_per_identity"] = int(args.media_per_identity)
    scenario_config = replace(scenario_config, **overrides)

    identities = generate_identities(scenario_config)
    media = fabricate_media(images, identities, scenario_config, labels=labels)
    _warn_on_shared_assets(media, len(labels) if labels is not None else None)
    means, persona_ids = _persona_embeddings(media, emb_by_index)
    resolved_band = _resolve_near_tie_band(
        means, persona_ids, args, scenario_config.near_tie_band
    )
    try:
        near_tie_sets = select_near_ties(
            means,
            persona_ids,
            band=resolved_band,
            max_sets=scenario_config.max_near_tie_sets,
        )
    except ValueError as exc:
        print(f"note: near-tie selection skipped ({exc})")
        near_tie_sets = []
    print(
        f"scenario: {len(identities)} personas / {len(media)} media items / "
        f"{len(near_tie_sets)} near-tie sets (resolved band "
        f"{[round(v, 4) for v in resolved_band]}, quantile {args.near_tie_quantile})"
    )
    if not near_tie_sets:
        print(
            "note: no near-tie sets in band -- near-tie metrics will be empty; "
            "raise --near-tie-quantile (e.g. 0.95) or widen --near-tie-band"
        )

    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    write_scenario(out_dir / "scenario", identities, media, scenario_config, near_tie_sets=near_tie_sets)
    scenario = load_scenario(out_dir / "scenario")

    query_by_index: dict[int, object] = {}
    n_query_view_dropped = 0
    query_degradation: tuple[str, float] | None = None
    if args.query_degradation:
        from grey_resolve.degradation.operators import DEGRADATIONS

        op_name, sev_str = args.query_degradation
        if op_name not in DEGRADATIONS:
            raise SystemExit(
                f"error: unknown degradation {op_name!r}; known: {sorted(DEGRADATIONS)}"
            )
        try:
            severity = float(sev_str)
        except ValueError as exc:
            raise SystemExit(f"error: bad --query-degradation severity {sev_str!r}") from exc
        op = DEGRADATIONS[op_name]
        for idx, img in enumerate(images):
            if idx not in emb_by_index:
                continue
            try:
                degraded = op(img, severity)
            except ValueError as exc:
                raise SystemExit(f"error: {exc}") from exc
            qv = embed_fn(degraded)
            if qv is None:
                n_query_view_dropped += 1
            else:
                query_by_index[idx] = qv
        query_degradation = (op_name, severity)
        print(
            f"query view: {op_name} sev={severity:g} -- {len(query_by_index)} embedded, "
            f"{n_query_view_dropped} detection failures (excluded from queries)"
        )

    items: list[RankingItem] = []
    query_ids: list[str] = []
    n_items_dropped = 0
    for m in scenario.media:
        if not m.image_ref.startswith("inline/"):
            n_items_dropped += 1
            continue
        idx = int(m.image_ref.split("/", 1)[1])
        vec = emb_by_index.get(idx)
        if vec is None:
            n_items_dropped += 1
            continue
        if query_degradation is None:
            items.append(RankingItem(m.media_id, m.identity_id, vec, m.to_context()))
            query_ids.append(m.media_id)
        else:
            qvec = query_by_index.get(idx)
            items.append(
                RankingItem(m.media_id, m.identity_id, vec, m.to_context(), query_embedding=qvec)
            )
            if qvec is not None:
                query_ids.append(m.media_id)
    print(f"ranking pool: {len(items)} items ({n_items_dropped} without embeddings dropped)")

    try:
        profiles = load_threshold_profiles()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.profile not in profiles:
        raise SystemExit(f"error: unknown profile {args.profile!r}; known: {sorted(profiles)}")
    profile = profiles[args.profile]
    scorer = LinearFusionScorer(profile)
    print(f"fusion profile {profile.name}: alpha={profile.alpha} beta={profile.beta}")

    try:
        ablation = run_ambiguity_ablation(
            items, scorer, near_tie_sets, k=args.k,
            query_ids=query_ids, gap_thresholds=list(args.gap_thresholds),
        )
        noise = run_context_noise_sweep(
            items, scorer, near_tie_sets, list(args.noise_rates), seed=int(args.seed),
            k=args.k, query_ids=query_ids,
        )
        weight_sweep = None
        if args.weight_ratios:
            def _factory(alpha: float, beta: float) -> LinearFusionScorer:
                return LinearFusionScorer(
                    ThresholdProfile(
                        f"weight-{alpha:g}-{beta:g}",
                        profile.quality_min,
                        profile.ambiguous_low,
                        profile.ambiguous_high,
                        alpha,
                        beta,
                    )
                )

            weight_sweep = run_context_weight_sweep(
                items,
                _factory,
                [(1.0, float(r)) for r in args.weight_ratios],
                near_tie_sets,
                k=args.k,
            )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    config = {
        "data_root": str(Path(args.data_root).resolve()),
        "limit": args.limit,
        "max_identities": args.max_identities,
        "seed": int(args.seed),
        "k": int(args.k),
        "profile": {
            "name": profile.name,
            "alpha": profile.alpha,
            "beta": profile.beta,
            "ambiguous_low": profile.ambiguous_low,
            "ambiguous_high": profile.ambiguous_high,
        },
        "scenario_config_path": str(args.scenario_config),
        "scenario_config_hash": config_hash(scenario_config),
        "scenario_config": scenario_config.to_dict(),
        "query_degradation": (
            [query_degradation[0], float(query_degradation[1])] if query_degradation else None
        ),
        "gap_thresholds": [float(t) for t in args.gap_thresholds],
        "n_query_view_dropped": int(n_query_view_dropped),
        "noise_rates": [float(p) for p in args.noise_rates],
        "weight_ratios": [float(r) for r in args.weight_ratios] if args.weight_ratios else None,
        "det_model": str(det_model),
        "rec_model": str(rec_model),
        "n_images": len(images),
        "n_identities": len(set(labels)),
        "n_items": len(items),
        "n_items_dropped": n_items_dropped,
        "n_embed_dropped": n_embed_dropped,
        "near_tie_band": [float(v) for v in resolved_band],
        "near_tie_quantile": float(args.near_tie_quantile),
        "near_tie_sets": [s.to_dict() for s in near_tie_sets],
        "run_stamp": run_stamp,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    results = {
        "config": config,
        "ablation": ablation,
        "context_noise_sweep": noise,
        "context_weight_sweep": weight_sweep,
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for method in ("face_only", "fused"):
            for scope in ("overall", "near_tie"):
                m = ablation[method][scope]
                writer.writerow(
                    ["ablation", method, scope, "", _fmt(m["hit_at_1"]), _fmt(m["hit_at_k"]),
                     _fmt(m["mrr"]), _fmt(m["n_queries"]), "", ""]
                )
        for t_key, block in ablation.get("gap_le", {}).items():
            for method in ("face_only", "fused"):
                m = block[method]
                d = block["delta"] if method == "fused" else {"hit_at_1": "", "mrr": ""}
                writer.writerow(
                    ["gap_le", method, f"gap<={t_key}", "", _fmt(m["hit_at_1"]), _fmt(m["hit_at_k"]),
                     _fmt(m["mrr"]), _fmt(m["n_queries"]),
                     _fmt(d["hit_at_1"]) if method == "fused" else "",
                     _fmt(d["mrr"]) if method == "fused" else ""]
                )
        for row in noise["rows"]:
            for scope in ("overall", "near_tie"):
                m = row["fused"][scope]
                d = row["delta"][scope]
                writer.writerow(
                    ["noise_sweep", "fused", scope, row["noise_rate"], _fmt(m["hit_at_1"]),
                     _fmt(m["hit_at_k"]), _fmt(m["mrr"]), _fmt(m["n_queries"]),
                     _fmt(d["hit_at_1"]), _fmt(d["mrr"])]
                )

    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {out_dir / 'scenario'}")
    for method in ("face_only", "fused"):
        for scope in ("overall", "near_tie"):
            m = ablation[method][scope]
            if m["n_queries"]:
                print(
                    f"  ablation {method:<9} {scope:<8} hit@1={m['hit_at_1']:.3f} "
                    f"hit@{ablation['k']}={m['hit_at_k']:.3f} mrr={m['mrr']:.3f} n={m['n_queries']}"
                )
    for t_key, block in ablation.get("gap_le", {}).items():
        fo, fu, d = block["face_only"], block["fused"], block["delta"]
        if block["n_queries"]:
            print(
                f"  gap<={t_key:<5} n={block['n_queries']:<4} face hit@1={fo['hit_at_1']:.3f} "
                f"fused hit@1={fu['hit_at_1']:.3f} delta={d['hit_at_1']:+.3f} "
                f"(mrr {fo['mrr']:.3f}->{fu['mrr']:.3f})"
            )
        else:
            print(f"  gap<={t_key:<5} n=0")
    gap = ablation["face_score_gap"]["near_tie"]
    if gap["n"]:
        print(f"  near-tie face gap: mean={gap['mean']:.4f} median={gap['median']:.4f} (n={gap['n']})")
    print("  noise sweep (fused hit@1 / delta vs face-only):")
    for row in noise["rows"]:
        for scope in ("overall",):
            m, d = row["fused"][scope], row["delta"][scope]
            if m["n_queries"]:
                print(
                    f"    rate={row['noise_rate']:<4} hit@1={m['hit_at_1']:.3f} "
                    f"mrr={m['mrr']:.3f} delta_hit@1={d['hit_at_1']:+.3f} n_corrupted={row['n_corrupted']}"
                )
    first_neg = noise["first_negative_delta_rate"]
    print(f"  fusion stops helping at rate: overall={first_neg['overall']} near_tie={first_neg['near_tie']}")
    if weight_sweep is not None and weight_sweep["peak"] is not None:
        peak = weight_sweep["peak"]
        print(
            f"  weight peak ({weight_sweep['peak_scope']}): alpha={peak['alpha']} beta={peak['beta']} "
            f"hit@1={peak[weight_sweep['peak_scope']]['hit_at_1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
