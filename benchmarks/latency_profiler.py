"""Latency profiler CLI: p50/p95 ingest/search latency + index-size sweep (Phase 3).

Thin wrapper over ``grey_resolve.evaluation.latency`` following the
``benchmarks/evaluate_roc.py`` conventions (ARCHITECTURE.md "Evaluation
pipeline").

Two modes:

1. Synthetic index-size sweep (default): random unit-norm vectors indexed into
   FAISS HNSW and the exact BruteForceIndex for comparison; per-search latency
   is measured across index sizes, dims, and k values.
2. ``--pipeline`` (optional): end-to-end ``Pipeline.ingest`` / ``Pipeline.search``
   latency with the real adapters (SCRFD + ArcFace + heuristic FIQA) on
   synthetic numpy-noise images (the no-face fast path), or on real images via
   ``--data-root <root>/<identity>/*.png|jpg``. Backbone weights must be fetched
   first (``python scripts/fetch_backbone.py``).

Outputs under ``benchmarks/out/latency-<run-stamp>/``:
- ``results.json``  -- config + config hash, sweep rows, pipeline latency report
- ``summary.csv``   -- one row per measured stage/cell (percentiles in ms)

No plotting: CSV/JSON only.

Usage:
    python benchmarks/latency_profiler.py [--sizes 1000 10000 50000] [--repeats 20]
    python benchmarks/latency_profiler.py --pipeline [--data-root path/to/gallery]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from grey_resolve.evaluation.latency import index_size_sweep, percentiles, time_calls

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_SIZES = (1000, 10000, 50000)
DEFAULT_DIMS = (512,)
DEFAULT_K = (1, 10, 100)
MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
FETCH_HINT = "fetch the backbone first: python scripts/fetch_backbone.py"
INDEX_KINDS = ("faiss_hnsw", "bruteforce")
CSV_COLUMNS = (
    "kind",
    "stage",
    "index_size",
    "dim",
    "k",
    "build_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "n_samples",
)
_ADD_CHUNK = 10_000  # bound peak memory while filling large indexes


def _model_dir() -> Path:
    """InsightFace weights dir (honours GREY_RESOLVE_MODEL_DIR like the tests)."""
    return Path(os.environ.get("GREY_RESOLVE_MODEL_DIR", Path.home() / ".insightface" / "models"))


def _unit_rows(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Return [n, dim] float32 random unit-norm rows (zero rows are redrawn)."""
    rows = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    while np.any(norms == 0.0):  # vanishingly rare, but zero rows are invalid
        bad = norms[:, 0] == 0.0
        rows[bad] = rng.standard_normal((int(bad.sum()), dim)).astype(np.float32)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / norms


def _synthetic_builder(kind: str, seed: int) -> Callable[[int, int], object]:
    """Return ``(dim, size) ->`` an index filled with deterministic random unit vectors.

    Both index kinds receive identical data for a given ``(seed, dim, size)``
    so the comparison is fair; ids are ``0..size-1``.
    """

    def build(dim: int, size: int):
        from grey_resolve.index.bruteforce import BruteForceIndex  # heavy import, deferred
        from grey_resolve.index.faiss_index import FaissHNSWIndex

        if kind == "faiss_hnsw":
            index = FaissHNSWIndex(dim=dim)
        elif kind == "bruteforce":
            index = BruteForceIndex(dim=dim)
        else:
            raise ValueError(f"unknown index kind: {kind!r}")
        rng = np.random.default_rng([seed, dim, size])
        for start in range(0, size, _ADD_CHUNK):
            n = min(_ADD_CHUNK, size - start)
            index.add(
                _unit_rows(rng, n, dim),
                np.arange(start, start + n, dtype=np.int64),
            )
        return index

    return build


def _run_sweep(args: argparse.Namespace) -> list[dict]:
    """Run the synthetic index-size sweep for every configured index kind and dim."""
    rows: list[dict] = []
    for kind in INDEX_KINDS:
        print(f"sweeping {kind}: sizes={list(args.sizes)} dims={list(args.dims)} k={list(args.k)}")
        build = _synthetic_builder(kind, args.seed)
        for dim in args.dims:
            queries = _unit_rows(np.random.default_rng([args.seed, dim]), args.queries, dim)
            for row in index_size_sweep(
                build,
                dims=[dim],
                sizes=list(args.sizes),
                k=list(args.k),
                queries=queries,
                warmup=args.warmup,
                repeats=args.repeats,
            ):
                rows.append({"kind": kind, **row})
    return rows


# --------------------------------------------------------------------- pipeline mode


def _load_images(root: str | Path, limit: int | None) -> list[np.ndarray]:
    """Load ``<root>/<identity>/*.{png,jpg,jpeg}`` as BGR uint8 images (evaluate_roc layout)."""
    import cv2  # heavy import, deferred

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root} (expected <root>/<identity>/*.png|jpg)")
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    files = sorted(
        f
        for ident_dir in root.iterdir()
        if ident_dir.is_dir() and not ident_dir.name.startswith("._")
        for f in ident_dir.iterdir()
        if f.suffix.lower() in IMAGE_SUFFIXES and not f.name.startswith("._")
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no images found under {root} (expected <root>/<identity>/*.png|jpg)")
    images: list[np.ndarray] = []
    for file in files:
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode image: {file}")
        images.append(image)
    return images


def _build_pipeline(det_model: Path, rec_model: Path):
    """Wire SCRFD + ArcFace + heuristic FIQA + HNSW + SQLite into a Pipeline."""
    from grey_resolve.config import ThresholdProfile
    from grey_resolve.detection.quality import HeuristicQualityAssessor
    from grey_resolve.detection.scrfd import ScrfdDetector
    from grey_resolve.embeddings.insightface_arcface import InsightFaceArcFaceExtractor
    from grey_resolve.index.faiss_index import FaissHNSWIndex
    from grey_resolve.index.sqlite_store import SqliteMetadataStore
    from grey_resolve.pipeline import Pipeline

    missing = [str(p) for p in (det_model, rec_model) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"backbone weights not found: {', '.join(missing)}; {FETCH_HINT}")
    profile = ThresholdProfile(
        name="latency_profile",
        quality_min=0.5,
        ambiguous_low=0.4,
        ambiguous_high=0.7,
        alpha=0.7,
        beta=0.3,
    )
    return Pipeline(
        detector=ScrfdDetector(model_file=det_model),
        assessor=HeuristicQualityAssessor(),
        extractor=InsightFaceArcFaceExtractor(model_file=rec_model),
        index=FaissHNSWIndex(dim=512),
        store=SqliteMetadataStore(":memory:"),
        profile=profile,
    )


def _run_pipeline(args: argparse.Namespace, det_model: Path, rec_model: Path) -> dict:
    """Time Pipeline.ingest / Pipeline.search end-to-end on synthetic or real images."""
    from grey_resolve.types import MediaItem

    if args.data_root:
        images = _load_images(args.data_root, args.images)
        mode = "data_root"
    else:
        rng = np.random.default_rng([args.seed, 2])
        images = [rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for _ in range(args.images)]
        mode = "synthetic_noise"
    print(f"pipeline mode: {mode}, {len(images)} images, search_k={args.search_k}")

    pipeline = _build_pipeline(det_model, rec_model)
    media = [
        MediaItem(media_id=f"latency-{i}", image_ref=f"{mode}://{i}") for i in range(len(images))
    ]
    calls = list(zip(media, images))

    def _ingest(state: dict) -> None:
        item, image = calls[state["i"] % len(calls)]
        state["i"] += 1
        pipeline.ingest(item, image)

    ingest_state = {"i": 0}
    ingest_samples = time_calls(lambda: _ingest(ingest_state), args.repeats, args.warmup)

    def _search(state: dict) -> None:
        _, image = calls[state["i"] % len(calls)]
        state["i"] += 1
        pipeline.search(image, k=args.search_k)

    search_state = {"i": 0}
    search_samples = time_calls(lambda: _search(search_state), args.repeats, args.warmup)

    return {
        "mode": mode,
        "n_images": len(images),
        "search_k": int(args.search_k),
        "final_index_size": int(pipeline.index_size),
        "ingest_ms": _ms_report(ingest_samples),
        "search_ms": _ms_report(search_samples),
    }


def _ms_report(samples: Sequence[float]) -> dict:
    """Percentile report with every timing converted from seconds to milliseconds."""
    stats = percentiles(samples)
    return {
        "p50": float(stats["p50"]) * 1000.0,
        "p90": float(stats["p90"]) * 1000.0,
        "p95": float(stats["p95"]) * 1000.0,
        "p99": float(stats["p99"]) * 1000.0,
        "mean": float(stats["mean"]) * 1000.0,
        "n": int(stats["n"]),
    }


# --------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES), help="index sizes (vector counts) to sweep")
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS), help="embedding dimensionalities to sweep")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K), help="top-k values to sweep")
    parser.add_argument("--queries", type=int, default=8, help="query vectors per sweep cell")
    parser.add_argument("--repeats", type=int, default=20, help="timed calls per measured cell")
    parser.add_argument("--warmup", type=int, default=3, help="untimed warmup calls per measured cell")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic vectors/images")
    parser.add_argument("--pipeline", action="store_true", help="also time Pipeline.ingest/search end-to-end")
    parser.add_argument("--data-root", default=None, help="real images: <root>/<identity>/*.png|jpg (with --pipeline)")
    parser.add_argument("--images", type=int, default=8, help="images for --pipeline (synthetic count or real-image cap)")
    parser.add_argument("--search-k", type=int, default=10, help="k for Pipeline.search in --pipeline mode")
    parser.add_argument("--out-dir", default="benchmarks/out", help="output root for run directories")
    parser.add_argument("--det-model", default=None, help=f"SCRFD weights (default: {_model_dir() / MODEL_FILES[0]})")
    parser.add_argument("--rec-model", default=None, help=f"ArcFace weights (default: {_model_dir() / MODEL_FILES[1]})")
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> None:
    """Reject non-positive sweep/call parameters before any measurement starts."""
    for name in ("sizes", "dims", "k"):
        values = getattr(args, name)
        if not values or any(v <= 0 for v in values):
            raise ValueError(f"--{name} must contain positive integers, got {values}")
    if args.queries < 1:
        raise ValueError(f"--queries must be >= 1, got {args.queries}")
    if args.repeats < 1:
        raise ValueError(f"--repeats must be >= 1, got {args.repeats}")
    if args.warmup < 0:
        raise ValueError(f"--warmup must be >= 0, got {args.warmup}")
    if args.images < 1:
        raise ValueError(f"--images must be >= 1, got {args.images}")
    if args.search_k < 1:
        raise ValueError(f"--search-k must be >= 1, got {args.search_k}")


def _csv_row(kind: str, stage: str, stats: dict, index_size: object, dim: object, k: object, build_ms: object) -> list:
    """Flatten one measurement into the uniform summary.csv layout."""
    return [
        kind,
        stage,
        index_size,
        dim,
        k,
        build_ms,
        round(float(stats["p50"]), 6),
        round(float(stats["p90"]), 6),
        round(float(stats["p95"]), 6),
        round(float(stats["p99"]), 6),
        round(float(stats["mean"]), 6),
        int(stats["n"]),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    model_dir = _model_dir()
    det_model = Path(args.det_model) if args.det_model else model_dir / MODEL_FILES[0]
    rec_model = Path(args.rec_model) if args.rec_model else model_dir / MODEL_FILES[1]
    if args.pipeline:
        # fail fast on missing weights, before the expensive sweep
        missing = [str(p) for p in (det_model, rec_model) if not p.exists()]
        if missing:
            raise SystemExit(f"error: backbone weights not found: {', '.join(missing)}; {FETCH_HINT}")

    sweep_rows = _run_sweep(args)
    pipeline_report = None
    if args.pipeline:
        try:
            pipeline_report = _run_pipeline(args, det_model, rec_model)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"error: {exc}") from exc

    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / f"latency-{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "sizes": [int(s) for s in args.sizes],
        "dims": [int(d) for d in args.dims],
        "k": [int(kk) for kk in args.k],
        "queries": int(args.queries),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "seed": int(args.seed),
        "pipeline": bool(args.pipeline),
        "data_root": str(Path(args.data_root).resolve()) if args.data_root else None,
        "images": int(args.images),
        "search_k": int(args.search_k),
        "index_kinds": list(INDEX_KINDS),
        "det_model": str(det_model),
        "rec_model": str(rec_model),
        "run_stamp": run_stamp,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    config["config_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps({"config": config, "sweep": sweep_rows, "pipeline": pipeline_report}, indent=2),
        encoding="utf-8",
    )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for row in sweep_rows:
            stats = {
                "p50": row["search_p50_ms"],
                "p90": row["search_p90_ms"],
                "p95": row["search_p95_ms"],
                "p99": row["search_p99_ms"],
                "mean": row["search_mean_ms"],
                "n": row["n_samples"],
            }
            writer.writerow(
                _csv_row(row["kind"], "search", stats, row["index_size"], row["dim"], row["k"], round(float(row["build_ms"]), 6))
            )
        if pipeline_report is not None:
            for stage in ("ingest", "search"):
                writer.writerow(
                    _csv_row(
                        "pipeline",
                        stage,
                        pipeline_report[f"{stage}_ms"],
                        pipeline_report["final_index_size"],
                        "",
                        pipeline_report["search_k"] if stage == "search" else "",
                        "",
                    )
                )

    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"config hash: {config['config_hash']}")
    for row in sweep_rows:
        print(
            f"  {row['kind']:<11} size={row['index_size']:<6} dim={row['dim']:<4} k={row['k']:<4} "
            f"p50={row['search_p50_ms']:.3f}ms p95={row['search_p95_ms']:.3f}ms "
            f"build={row['build_ms']:.1f}ms n={row['n_samples']}"
        )
    if pipeline_report is not None:
        for stage in ("ingest", "search"):
            stats = pipeline_report[f"{stage}_ms"]
            print(
                f"  pipeline/{stage:<7} p50={stats['p50']:.3f}ms p95={stats['p95']:.3f}ms "
                f"mean={stats['mean']:.3f}ms n={stats['n']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
