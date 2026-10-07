"""Figure builders for the Grey-Resolve portfolio writeup.

Turn in-memory ``results.json`` blocks (schemas from ``benchmarks/evaluate_roc.py``
and ``benchmarks/evaluate_fusion.py``) into matplotlib ``Figure`` objects.

Design rules:

- ``matplotlib`` is imported lazily inside each function, so importing this
  module never requires the optional ``figures`` extra.
- Every figure states "research prototype -- local research data" and the
  benchmark run id in its title.
- Functions only build figures; they never call ``show()`` and never save.
  Callers decide whether to save, display, or close the figure.

Example::

    from grey_resolve.plotting import roc_figure
    fig = roc_figure(results["sweep"], run_id=results["config"]["run_stamp"])
    fig.savefig("roc.png", dpi=150)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

PROTOTYPE_NOTE: Final[str] = "research prototype -- local research data"
EER_LABEL: Final[str] = "EER"
FMR_AT_FNM_LABEL: Final[str] = "FMR@FNMR=1%"
FULL_LABEL: Final[str] = "Full set"
GATED_LABEL: Final[str] = "Quality-gated"
#: Method block key -> legend label (canonical draw order).
METHOD_LABELS: Final[Mapping[str, str]] = {
    "face_only": "face-only",
    "fused": "fused",
    "selective": "selective (gap-gated)",
    "tiebreak": "tiebreak (top-2)",
}
METHOD_COLORS: Final[Mapping[str, str]] = {
    "face_only": "#4c72b0",
    "fused": "#dd8452",
    "selective": "#55a868",
    "tiebreak": "#c44e52",
}
#: Schema key for FMR at FNMR = 1% (see ``benchmarks/evaluate_roc.py``).
FMR_AT_FNM_KEY: Final[str] = "fmr_at_fnmr_0.01"
#: Error rates of exactly 0 are drawn at this floor so log axes stay readable.
ZERO_ERROR_FLOOR: Final[float] = 5e-5


def condition_label(condition: str, severity: float | None) -> str:
    """Human-readable degradation label, e.g. ``"downsample 0.1"`` or ``"clean"``.

    Args:
        condition: degradation operator name (``"clean"`` for the baseline).
        severity: operator strength parameter, or ``None`` when not recorded.

    Returns:
        The condition name for the clean baseline, otherwise
        ``"<condition> <severity>"`` with severity formatted with ``g``.
    """
    if condition == "clean" or severity is None:
        return condition
    return f"{condition} {float(severity):g}"


def _set_title(ax: Any, headline: str, run_id: str) -> None:
    """Two-line axes title: figure headline plus the prototype note and run id."""
    ax.set_title(f"{headline}\n{PROTOTYPE_NOTE} (run {run_id})", fontsize=11)


def _finite(value: float) -> bool:
    """True when *value* is a real plottable number (not NaN/inf)."""
    return math.isfinite(value)


def _metric(metrics: Mapping[str, Any] | None, key: str) -> float:
    """Read ``metrics[key]`` as float; NaN when absent or ``None`` (unreported)."""
    if not isinstance(metrics, Mapping):
        return float("nan")
    value = metrics.get(key)
    return float("nan") if value is None else float(value)


def _annotate_bars(ax: Any, rects: Sequence[Any], values: Sequence[float]) -> None:
    """Label each finite bar with its value (``%.3f``), skip unreported ones."""
    for rect, value in zip(rects, values, strict=True):
        if not _finite(value):
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def roc_figure(
    sweep_results: Sequence[Mapping[str, Any]], *, run_id: str = "unspecified run"
) -> Figure:
    """ROC / error-trade-off curves: one FMR-vs-FNMR line per degradation condition.

    Args:
        sweep_results: the ``sweep`` list from an ``evaluate_roc.py``
            ``results.json`` -- each entry has ``condition``, ``severity`` and
            ``roc_points`` (``threshold``/``fmr``/``fnmr`` rows).
        run_id: benchmark run id cited in the figure title.

    Returns:
        A matplotlib figure with log-log error-trade-off axes. Zero error rates
        are drawn at :data:`ZERO_ERROR_FLOOR` so the log axes stay readable.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    cmap = plt.get_cmap("tab20")
    for index, entry in enumerate(sweep_results):
        points = sorted(
            (entry.get("roc_points") or []),
            key=lambda p: (float(p["fmr"]), float(p["fnmr"])),
        )
        if not points:
            continue
        fmr = [max(float(p["fmr"]), ZERO_ERROR_FLOOR) for p in points]
        fnmr = [max(float(p["fnmr"]), ZERO_ERROR_FLOOR) for p in points]
        ax.plot(
            fmr,
            fnmr,
            linewidth=1.6,
            color=cmap(index % cmap.N),
            label=condition_label(str(entry.get("condition", "?")), entry.get("severity")),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ZERO_ERROR_FLOOR, 1.0)
    ax.set_ylim(ZERO_ERROR_FLOOR, 1.0)
    ax.set_xlabel("FMR (false match rate) -- log scale")
    ax.set_ylabel("FNMR (false non-match rate) -- log scale")
    _set_title(ax, "Verification error trade-off (FMR vs FNMR)", run_id)
    ax.grid(True, which="both", alpha=0.3)
    if ax.get_lines():
        ax.legend(fontsize=8, loc="lower left", ncol=2 if len(sweep_results) > 8 else 1)
    fig.tight_layout()
    return fig


def eer_bar_figure(
    sweep_results: Sequence[Mapping[str, Any]], *, run_id: str = "unspecified run"
) -> Figure:
    """Grouped bars per degradation condition: EER and FMR at FNMR = 1%.

    Args:
        sweep_results: the ``sweep`` list from an ``evaluate_roc.py``
            ``results.json`` (``condition``, ``severity``, ``eer``,
            ``fmr_at_fnmr_0.01`` per entry).
        run_id: benchmark run id cited in the figure title.

    Returns:
        A matplotlib figure with two bars (EER, FMR@FNMR=1%) per condition.
    """
    import matplotlib.pyplot as plt

    labels = [
        condition_label(str(e.get("condition", "?")), e.get("severity")) for e in sweep_results
    ]
    eer_values = [_metric(e, "eer") for e in sweep_results]
    fmr_values = [_metric(e, FMR_AT_FNM_KEY) for e in sweep_results]
    x = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(7.5, 0.85 * len(labels) + 3.0), 5.5))
    eer_rects = ax.bar(
        [xi - width / 2.0 for xi in x], eer_values, width, label=EER_LABEL, color="#4c72b0"
    )
    fmr_rects = ax.bar(
        [xi + width / 2.0 for xi in x],
        fmr_values,
        width,
        label=FMR_AT_FNM_LABEL,
        color="#dd8452",
    )
    _annotate_bars(ax, eer_rects, eer_values)
    _annotate_bars(ax, fmr_rects, fmr_values)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("error rate")
    ax.set_ylim(0.0, 1.08)
    _set_title(ax, "EER and FMR@FNMR=1% per degradation condition", run_id)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def gate_ablation_figure(
    gated_results: Sequence[Mapping[str, Any]], *, run_id: str = "unspecified run"
) -> Figure:
    """Grouped bars per condition: full-set FNMR vs quality-gated FNMR.

    Args:
        gated_results: the ``gated_vs_ungated`` list from an ``evaluate_roc.py``
            ``results.json`` -- ``full``/``gated`` metric blocks per condition.
            ``gated`` FNMR of ``None`` (too few quality-passing probes) is drawn
            as an empty bar with an ``n/a`` marker.
        run_id: benchmark run id cited in the figure title.

    Returns:
        A matplotlib figure comparing full vs gated FNMR per condition.
    """
    import matplotlib.pyplot as plt

    labels = [
        condition_label(str(e.get("condition", "?")), e.get("severity")) for e in gated_results
    ]
    full_values = [_metric(e.get("full"), "fnmr") for e in gated_results]
    gated_values = [_metric(e.get("gated"), "fnmr") for e in gated_results]
    operating = _metric(gated_results[0] if gated_results else None, "operating_threshold")
    operating_note = f"{operating:g}" if _finite(operating) else "?"
    x = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(7.5, 0.85 * len(labels) + 3.0), 5.5))
    full_rects = ax.bar(
        [xi - width / 2.0 for xi in x], full_values, width, label=FULL_LABEL, color="#4c72b0"
    )
    gated_rects = ax.bar(
        [xi + width / 2.0 for xi in x], gated_values, width, label=GATED_LABEL, color="#55a868"
    )
    _annotate_bars(ax, full_rects, full_values)
    _annotate_bars(ax, gated_rects, gated_values)
    for xi, value in zip(x, gated_values, strict=True):
        if not _finite(value):
            ax.text(xi + width / 2.0, 0.012, "n/a", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel(f"FNMR at operating threshold {operating_note}")
    finite_max = max((v for v in full_values + gated_values if _finite(v)), default=0.12)
    ax.set_ylim(0.0, max(0.12, 1.08 * finite_max))
    _set_title(ax, "Quality-gate ablation: full vs gated FNMR", run_id)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def _fusion_scope_rows(
    ablation: Mapping[str, Any],
) -> tuple[list[str], list[tuple[str, dict[str, float]]]]:
    """Build (method keys, scope rows) for the fusion bar chart.

    Rows are ``overall``, ``near-tie`` and then the ``gap_le`` slices ordered by
    gap cutoff (tightest first); each row maps method key to hit@1 (NaN when the
    run recorded no metrics for that method on that scope).
    """
    methods = [m for m in METHOD_LABELS if isinstance(ablation.get(m), Mapping)]
    if not methods:
        raise ValueError("ablation_results has no ranking-method blocks (face_only/fused/...)")
    rows: list[tuple[str, dict[str, float]]] = []
    for scope_key, scope_name in (("overall", "overall"), ("near_tie", "near-tie")):
        anchor = next(
            (
                ablation[m][scope_key]
                for m in methods
                if isinstance(ablation[m].get(scope_key), Mapping)
            ),
            None,
        )
        if not isinstance(anchor, Mapping):
            continue
        n = anchor.get("n_queries")
        label = f"{scope_name} (n={n})" if n is not None else scope_name
        values = {m: _metric(ablation[m].get(scope_key), "hit_at_1") for m in methods}
        rows.append((label, values))
    gap_le = ablation.get("gap_le")
    if isinstance(gap_le, Mapping):
        for key in sorted(gap_le, key=float):
            slice_block = gap_le[key]
            if not isinstance(slice_block, Mapping):
                continue
            n = slice_block.get("n_queries")
            label = f"gap <= {float(key):g} (n={n})" if n is not None else f"gap <= {float(key):g}"
            values = {m: _metric(slice_block.get(m), "hit_at_1") for m in methods}
            rows.append((label, values))
    return methods, rows


def fusion_methods_figure(
    ablation_results: Mapping[str, Any], *, run_id: str = "unspecified run"
) -> Figure:
    """Grouped hit@1 bars per ranking method, per scope and face-gap slice.

    Args:
        ablation_results: the ``ablation`` block of an ``evaluate_fusion.py``
            ``results.json`` -- method blocks (``face_only``, ``fused``,
            ``selective``, ``tiebreak``; whichever the run recorded) with
            ``overall``/``near_tie`` metrics and optional ``gap_le`` slices.
        run_id: benchmark run id cited in the figure title.

    Returns:
        A matplotlib figure with one bar per method per scope/slice; methods or
        scopes a run did not record simply do not appear.

    Raises:
        ValueError: no ranking-method block is present.
    """
    import matplotlib.pyplot as plt

    methods, rows = _fusion_scope_rows(ablation_results)
    labels = [label for label, _ in rows]
    x = list(range(len(labels)))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(labels) + 3.5), 5.5))
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2.0) * width
        values = [row_values[method] for _, row_values in rows]
        rects = ax.bar(
            [xi + offset for xi in x],
            values,
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
        _annotate_bars(ax, rects, values)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("hit@1 (persona ranking)")
    ax.set_ylim(0.0, 1.15)
    _set_title(ax, "Fusion ablation: hit@1 by scope and face-gap slice", run_id)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
