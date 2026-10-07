"""Portfolio figure rendering for Grey-Resolve (research prototype figures).

Pure-ish functions that turn benchmark ``results.json`` blocks into matplotlib
figures; ``matplotlib`` is imported lazily so the core package stays importable
without the optional ``figures`` extra (``pip install -e ".[figures]"``).
"""

from __future__ import annotations

from grey_resolve.plotting.curves import (
    FMR_AT_FNM_LABEL,
    FULL_LABEL,
    GATED_LABEL,
    METHOD_LABELS,
    PROTOTYPE_NOTE,
    condition_label,
    eer_bar_figure,
    fusion_methods_figure,
    gate_ablation_figure,
    roc_figure,
)

__all__ = [
    "FMR_AT_FNM_LABEL",
    "FULL_LABEL",
    "GATED_LABEL",
    "METHOD_LABELS",
    "PROTOTYPE_NOTE",
    "condition_label",
    "eer_bar_figure",
    "fusion_methods_figure",
    "gate_ablation_figure",
    "roc_figure",
]
