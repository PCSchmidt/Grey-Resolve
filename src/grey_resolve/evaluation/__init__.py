"""Verification evaluation core: FMR/FNMR, ROC, EER, pair construction.

Extends the IronClad task4 CMC analysis to verification metrics
(docs/PORT_INVENTORY.md section 2). Pure numpy over score arrays.
"""

from grey_resolve.evaluation.metrics import (
    build_pairs,
    eer,
    fmr_at_fnmr,
    fmr_fnmr_at_threshold,
    roc_curve,
)

__all__ = [
    "build_pairs",
    "eer",
    "fmr_at_fnmr",
    "fmr_fnmr_at_threshold",
    "roc_curve",
]
