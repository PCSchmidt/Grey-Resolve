"""Degraded-input ROC/FMR/FNMR benchmark CLI (thin wrapper over the experiment core).

Loads a labeled image dataset laid out as ``<root>/<identity>/*.png|jpg``
(IronClad gallery style), embeds it with the Grey-Resolve adapters (SCRFD
detection + InsightFace ArcFace + heuristic FIQA), and runs the degraded-input
verification sweep for every operator in
``grey_resolve.degradation.operators.DEGRADATIONS`` at the given severities plus
the clean baseline.

Outputs under ``benchmarks/out/<run-stamp>/``:
- ``results.json``  -- config, per-condition metrics, full ROC points, gating ablation
- ``summary.csv``   -- one row per condition: condition, severity, eer, fmr_at_fnmr_0.01

No plotting: CSV/JSON only. Backbone weights must be fetched first
(``python scripts/fetch_backbone.py``).

Usage:
    python benchmarks/evaluate_roc.py --data-root path/to/gallery [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from grey_resolve.degradation.operators import DEGRADATIONS
from grey_resolve.evaluation.experiment import (
    FMR_AT_FNM_KEY,
    DegradationCondition,
    gated_vs_ungated,
    run_condition_sweep,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
# Severity is each operator's own strength parameter (sigma / delta / scale /
# angle_deg) -- there is no shared normalized scale, so the default grid is
# per-operator. Values chosen to span "mild" to "severe" for each.
DEFAULT_SEVERITY_GRID = {
    "gaussian_blur": (1.0, 3.0, 8.0),    # sigma in px
    "brightness": (0.25, 0.5, 0.75),     # delta as fraction of range
    "downsample": (0.5, 0.25, 0.1),      # retained scale (smaller = harsher)
    "off_angle": (15.0, 30.0, 45.0),     # rotation in degrees
}
DEFAULT_SEVERITIES = (0.25, 0.5, 0.75)   # fallback when --severities is given for all ops
MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
FETCH_HINT = "fetch the backbone first: python scripts/fetch_backbone.py"


def _model_dir() -> Path:
    """InsightFace weights dir (honours GREY_RESOLVE_MODEL_DIR like the tests)."""
    return Path(os.environ.get("GREY_RESOLVE_MODEL_DIR", Path.home() / ".insightface" / "models"))


def load_dataset(
    root: str | Path, limit: int | None = None
) -> tuple[list[np.ndarray], list[str]]:
    """Load ``<root>/<identity>/*.{png,jpg,jpeg}`` as BGR uint8 images + identity labels.

    Args:
        root: dataset directory containing one sub-directory per identity.
        limit: optional cap on images per identity (files in sorted order).

    Returns:
        ``(images, labels)`` with one uint8 (H, W, 3) BGR image and one identity
        label per loaded file.

    Raises:
        FileNotFoundError: root missing or no images found.
        ValueError: root is not a directory, an image fails to decode, fewer than
            two identities are present, or ``limit`` is not positive.
    """
    import cv2  # heavy import, deferred

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"dataset root not found: {root} (expected <root>/<identity>/*.png|jpg)"
        )
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")

    images: list[np.ndarray] = []
    labels: list[str] = []
    for ident_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in ident_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if limit is not None:
            files = files[:limit]
        for file in files:
            image = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to decode image: {file}")
            images.append(image)
            labels.append(ident_dir.name)

    if not images:
        raise FileNotFoundError(f"no images found under {root} (expected <root>/<identity>/*.png|jpg)")
    if len(set(labels)) < 2:
        raise ValueError(f"need at least 2 identities under {root} (found {sorted(set(labels))})")
    return images, labels


class _DetectionCache:
    """Memoized face detection shared by the embed and quality adapters.

    Degradation produces fresh arrays per condition, so the key is a content
    hash; detection runs once per unique image instead of twice.
    """

    def __init__(self, detector, max_entries: int = 512):
        import collections

        self._detector = detector
        self._cache: "collections.OrderedDict" = collections.OrderedDict()
        self._max = max_entries

    def detect(self, image: np.ndarray):
        import hashlib

        key = hashlib.sha1(image.tobytes()).digest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        observations = self._detector.detect(image)
        self._cache[key] = observations
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return observations


class _FaceEmbedder:
    """embed_fn adapter: detect the most confident face, then embed it.

    Returns ``None`` when no face is detected -- heavy degradation causes real
    detection failures, and the experiment core excludes and counts those
    probes per condition instead of aborting the run.
    """

    def __init__(self, detector, extractor):
        self._detector = detector
        self._extractor = extractor

    def __call__(self, image: np.ndarray) -> "np.ndarray | None":
        observations = self._detector.detect(image)
        if not observations:
            return None
        return self._extractor.extract(image, observations[0])


class _FaceQuality:
    """quality_fn adapter: heuristic FIQA score of the most confident face (0.0 if none)."""

    def __init__(self, detector, assessor):
        self._detector = detector
        self._assessor = assessor

    def __call__(self, image: np.ndarray) -> float:
        observations = self._detector.detect(image)
        if not observations:
            return 0.0
        return self._assessor.assess(image, observations[0]).score


def _build_adapters(det_model: Path, rec_model: Path):
    """Construct SCRFD + ArcFace + heuristic FIQA; fail clearly when weights are missing."""
    from grey_resolve.detection.quality import HeuristicQualityAssessor
    from grey_resolve.detection.scrfd import ScrfdDetector
    from grey_resolve.embeddings.insightface_arcface import InsightFaceArcFaceExtractor

    missing = [str(p) for p in (det_model, rec_model) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"backbone weights not found: {', '.join(missing)}; {FETCH_HINT}")
    detector = _DetectionCache(ScrfdDetector(model_file=det_model))
    extractor = InsightFaceArcFaceExtractor(model_file=rec_model)
    return detector, extractor, HeuristicQualityAssessor()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True, help="dataset dir: <root>/<identity>/*.png|jpg")
    parser.add_argument("--limit", type=int, default=None, help="cap images per identity (quick runs)")
    parser.add_argument(
        "--severities",
        type=float,
        nargs="+",
        default=None,
        help="override: apply these severities to every operator "
        "(default: per-operator grid, see DEFAULT_SEVERITY_GRID)",
    )
    parser.add_argument("--out-dir", default="benchmarks/out", help="output root for run directories")
    parser.add_argument("--det-model", default=None, help=f"SCRFD weights (default: {_model_dir() / MODEL_FILES[0]})")
    parser.add_argument("--rec-model", default=None, help=f"ArcFace weights (default: {_model_dir() / MODEL_FILES[1]})")
    parser.add_argument("--quality-threshold", type=float, default=0.5, help="FIQA gate for the gating ablation")
    parser.add_argument("--operating-threshold", type=float, default=0.5, help="cosine threshold for gated FMR/FNMR")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = _model_dir()
    det_model = Path(args.det_model) if args.det_model else model_dir / MODEL_FILES[0]
    rec_model = Path(args.rec_model) if args.rec_model else model_dir / MODEL_FILES[1]

    try:
        images, labels = load_dataset(args.data_root, args.limit)
        detector, extractor, assessor = _build_adapters(det_model, rec_model)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    embed_fn = _FaceEmbedder(detector, extractor)
    quality_fn = _FaceQuality(detector, assessor)
    if args.severities:
        grid = {name: tuple(args.severities) for name in sorted(DEGRADATIONS)}
    else:
        grid = {
            name: DEFAULT_SEVERITY_GRID.get(name, DEFAULT_SEVERITIES)
            for name in sorted(DEGRADATIONS)
        }
    conditions = [
        DegradationCondition(name, severity)
        for name in sorted(DEGRADATIONS)
        for severity in grid[name]
    ]

    print(f"loaded {len(images)} images / {len(set(labels))} identities from {args.data_root}")
    for name, sevs in grid.items():
        print(f"  grid {name}: {list(sevs)}")
    print(f"running {len(conditions)} conditions + clean baseline ...")
    sweep = run_condition_sweep(images, labels, conditions, embed_fn)
    gated = gated_vs_ungated(
        images,
        labels,
        conditions,
        embed_fn,
        quality_fn,
        args.quality_threshold,
        threshold=args.operating_threshold,
    )

    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "data_root": str(Path(args.data_root).resolve()),
        "limit": args.limit,
        "severity_grid": {name: [float(s) for s in sevs] for name, sevs in grid.items()},
        "degradations": sorted(DEGRADATIONS),
        "quality_threshold": float(args.quality_threshold),
        "operating_threshold": float(args.operating_threshold),
        "det_model": str(det_model),
        "rec_model": str(rec_model),
        "n_images": len(images),
        "n_identities": len(set(labels)),
        "run_stamp": run_stamp,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps({"config": config, "sweep": sweep, "gated_vs_ungated": gated}, indent=2),
        encoding="utf-8",
    )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "severity", "eer", FMR_AT_FNM_KEY])
        for row in sweep:
            writer.writerow([row["condition"], row["severity"], row["eer"], row[FMR_AT_FNM_KEY]])

    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    for row in sweep:
        print(f"  {row['condition']:<14} sev={row['severity']:<5} eer={row['eer']:.4f} {FMR_AT_FNM_KEY}={row[FMR_AT_FNM_KEY]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
