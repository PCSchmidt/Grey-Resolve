"""Render portfolio figures + sanitized aggregate results from benchmark runs.

Discovers run directories under ``benchmarks/out/`` (or takes explicit ``--runs``),
renders the figure(s) matching each run's ``results.json`` schema into
``docs/figures/<run-stamp>/<name>.png`` (dpi 150), writes ``docs/figures/index.md``
linking every PNG to its run + a one-line description, and copies a sanitized
aggregate ``results.json`` into ``docs/results/<run-stamp>.json``.

Sanitization (see :func:`sanitize_results`) -- aggregate metrics only, no local
filesystem layout:

- ``config.data_root`` / ``config.det_model`` / ``config.rec_model`` local paths
  are replaced with their basenames; any other absolute filesystem path string
  anywhere in the document is reduced to its basename as a safety net.
- Per-threshold ``roc_points`` curve rows are dropped (the rendered figures are
  the curve view; the copy keeps aggregate metrics only).

Rendering skips gracefully when ``matplotlib`` is not installed: a clear message
is printed and sanitized results + the index are still written.

Usage:
    python benchmarks/make_figures.py [--runs RUN_OR_STAMP ...] [--dpi 150]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from grey_resolve.plotting import (
    METHOD_LABELS,
    eer_bar_figure,
    fusion_methods_figure,
    gate_ablation_figure,
    roc_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "benchmarks" / "out"
DEFAULT_FIGURES_ROOT = REPO_ROOT / "docs" / "figures"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "docs" / "results"
#: Config keys holding local filesystem paths that must not leak into docs/.
PATH_CONFIG_KEYS = ("data_root", "det_model", "rec_model")
#: Per-point rows dropped from the sanitized copy (aggregate metrics only).
NON_AGGREGATE_KEYS = ("roc_points",)
#: One-line description per figure name (used by the index).
FIGURE_DESCRIPTIONS = {
    "roc_curve": "ROC / error trade-off (FMR vs FNMR, log axes) per degradation condition.",
    "eer_fmr_bars": "EER and FMR@FNMR=1% per degradation condition (grouped bars).",
    "gate_ablation": "Full vs quality-gated FNMR per condition (quality-gate ablation).",
    "fusion_methods": (
        "Persona-ranking hit@1 per method (face-only / fused / selective / tiebreak) "
        "by scope and face-gap slice."
    ),
}
_ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[/\\])")


def path_basename(value: str) -> str:
    """Basename of a Windows- or POSIX-style path (or the value itself).

    Args:
        value: a path string with ``\\`` or ``/`` separators, or a bare name.

    Returns:
        The final path component; *value* unchanged when it has no separators.
    """
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _scrub_absolute_paths(obj: Any) -> Any:
    """Recursively replace absolute filesystem path strings with their basenames."""
    if isinstance(obj, dict):
        return {key: _scrub_absolute_paths(item) for key, item in obj.items()}
    if isinstance(obj, list):
        return [_scrub_absolute_paths(item) for item in obj]
    if isinstance(obj, str) and _ABSOLUTE_PATH_RE.match(obj):
        return path_basename(obj)
    return obj


def sanitize_results(results: Mapping[str, Any]) -> dict[str, Any]:
    """Return an aggregate-metrics-only copy of *results* with local paths removed.

    Args:
        results: a full ``results.json`` document (``config``, ``sweep`` /
            ``gated_vs_ungated`` for ROC runs, ``ablation`` for fusion runs).

    Returns:
        A deep copy where ``config.data_root`` / ``config.det_model`` /
        ``config.rec_model`` are reduced to basenames, any other absolute path
        string is reduced to its basename, and per-threshold ``roc_points`` rows
        are dropped. The input is never mutated.
    """
    clean = _scrub_absolute_paths(copy.deepcopy(dict(results)))
    config = clean.get("config")
    if isinstance(config, dict):
        for key in PATH_CONFIG_KEYS:
            value = config.get(key)
            if isinstance(value, str):
                config[key] = path_basename(value)
    sweep = clean.get("sweep")
    if isinstance(sweep, list):
        for entry in sweep:
            if isinstance(entry, dict):
                for key in NON_AGGREGATE_KEYS:
                    entry.pop(key, None)
    return clean


def discover_runs(runs_root: Path) -> list[Path]:
    """Run directories under *runs_root* that contain a ``results.json``, sorted."""
    if not runs_root.is_dir():
        return []
    return sorted(
        (p for p in runs_root.iterdir() if p.is_dir() and (p / "results.json").is_file()),
        key=lambda p: p.name,
    )


def resolve_run_dirs(runs_root: Path, runs: Sequence[str] | None) -> list[Path]:
    """Resolve ``--runs`` entries (run dirs or run stamps) to run directories.

    Args:
        runs_root: directory that holds run-stamp sub-directories.
        runs: explicit run dir paths / run stamps, or ``None`` to discover all.

    Returns:
        The resolved run directories, sorted by name. Unknown entries are
        reported on stderr and skipped.

    Raises:
        ValueError: *runs* is given but no entry resolves to a run directory.
    """
    if not runs:
        return discover_runs(runs_root)
    resolved: list[Path] = []
    for entry in runs:
        candidate = Path(entry)
        if (candidate / "results.json").is_file():
            resolved.append(candidate)
        elif (runs_root / entry / "results.json").is_file():
            resolved.append(runs_root / entry)
        else:
            print(f"make_figures: skipping unknown run: {entry}", file=sys.stderr)
    if not resolved:
        raise ValueError(f"no run directories found among --runs entries: {list(runs)}")
    return sorted(set(resolved), key=lambda p: p.name)


def figures_for_results(results: Mapping[str, Any], run_id: str) -> list[tuple[str, Any]]:
    """Build the figures a run's ``results.json`` supports, in canonical order.

    Args:
        results: one run's ``results.json`` document.
        run_id: run stamp cited in every figure title.

    Returns:
        ``(name, figure)`` pairs for ``roc_curve``/``eer_fmr_bars`` (runs with a
        ``sweep`` block), ``gate_ablation`` (runs with ``gated_vs_ungated``) and
        ``fusion_methods`` (runs with an ``ablation`` block).
    """
    out: list[tuple[str, Any]] = []
    sweep = results.get("sweep")
    # ROC sweeps carry per-condition blocks; other tools reuse the "sweep" key
    # (e.g. latency sweeps) and have no matching figure here.
    roc_entries = [
        e for e in sweep if isinstance(e, Mapping) and "condition" in e
    ] if isinstance(sweep, list) else []
    if roc_entries:
        out.append(("roc_curve", roc_figure(roc_entries, run_id=run_id)))
        out.append(("eer_fmr_bars", eer_bar_figure(roc_entries, run_id=run_id)))
    gated = results.get("gated_vs_ungated")
    gated_entries = [
        e for e in gated if isinstance(e, Mapping) and "full" in e
    ] if isinstance(gated, list) else []
    if gated_entries:
        out.append(("gate_ablation", gate_ablation_figure(gated_entries, run_id=run_id)))
    ablation = results.get("ablation")
    if isinstance(ablation, Mapping) and any(
        isinstance(ablation.get(m), Mapping) for m in METHOD_LABELS
    ):
        out.append(("fusion_methods", fusion_methods_figure(ablation, run_id=run_id)))
    return out


def build_index(records: Sequence[tuple[str, str]], *, figures_skipped: bool) -> str:
    """Render ``docs/figures/index.md`` content.

    Args:
        records: ``(run_id, figure_name)`` pairs, one per written PNG (or per
            figure that would have been written).
        figures_skipped: True when matplotlib was unavailable.

    Returns:
        The markdown document text (deterministic given the records).
    """
    lines = [
        "# Grey-Resolve benchmark figures",
        "",
        "research prototype -- local research data -- aggregate metrics from local",
        "research data only; no real operational data (see `docs/RESULTS.md`).",
        "Candidate-ranking evaluation -- not identity-assertion accuracy.",
        "",
        "Generated by `python benchmarks/make_figures.py` (dpi 150). Raw run artifacts",
        "under `benchmarks/out/` are gitignored; sanitized aggregate copies live in",
        "`docs/results/`.",
        "",
    ]
    if figures_skipped:
        lines += [
            "> Figure rendering was skipped: `matplotlib` is not installed",
            "> (`uv pip install -e \".[figures]\"`). Sanitized results were still written.",
            "",
        ]
    if not records:
        lines += ["No figures recorded.", ""]
        return "\n".join(lines)
    lines += ["| Figure | Run | Description | Sanitized results |", "|---|---|---|---|"]
    for run_id, name in sorted(records):
        filename = f"{name}.png"
        description = FIGURE_DESCRIPTIONS.get(name, "")
        lines.append(
            f"| [{filename}]({run_id}/{filename}) | `{run_id}` | {description} "
            f"| [results](../results/{run_id}.json) |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI argument parser for ``make_figures.py``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        metavar="RUN",
        help="run directory paths or run stamps (default: discover under --runs-root)",
    )
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT), help="run-dir root")
    parser.add_argument("--figures-root", default=str(DEFAULT_FIGURES_ROOT), help="PNG output root")
    parser.add_argument(
        "--results-root", default=str(DEFAULT_RESULTS_ROOT), help="sanitized JSON output root"
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution (default 150)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render figures + sanitized results for every discovered/selected run.

    Returns:
        0 on success (including the graceful matplotlib-missing skip), 2 when no
        run directories resolve.
    """
    args = parse_args(argv)
    try:
        run_dirs = resolve_run_dirs(Path(args.runs_root), args.runs)
    except ValueError as exc:
        print(f"make_figures: {exc}", file=sys.stderr)
        return 2
    if not run_dirs:
        print(
            f"make_figures: no run dirs with results.json under {args.runs_root}", file=sys.stderr
        )
        return 2

    figures_root = Path(args.figures_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
    except ImportError:
        matplotlib = None
    if matplotlib is not None:
        matplotlib.use("Agg")

    records: list[tuple[str, str]] = []
    for run_dir in run_dirs:
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        run_id = run_dir.name
        (results_root / f"{run_id}.json").write_text(
            json.dumps(sanitize_results(results), indent=1) + "\n", encoding="utf-8"
        )
        if matplotlib is None:
            continue
        import matplotlib.pyplot as plt

        for name, figure in figures_for_results(results, run_id):
            png_dir = figures_root / run_id
            png_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(png_dir / f"{name}.png", dpi=args.dpi)
            plt.close(figure)
            records.append((run_id, name))
            print(f"wrote {png_dir / (name + '.png')}")

    if matplotlib is None:
        print(
            "make_figures: matplotlib is not installed -- figure rendering skipped. "
            "Install it with: uv pip install -e \".[figures]\" (or: uv pip install matplotlib). "
            "Sanitized results were still written to "
            f"{results_root}.",
            file=sys.stderr,
        )

    figures_root.mkdir(parents=True, exist_ok=True)
    index_text = build_index(records, figures_skipped=matplotlib is None)
    (figures_root / "index.md").write_text(index_text, encoding="utf-8")
    print(f"wrote {figures_root / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
