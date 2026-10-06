"""Verification metrics: genuine/impostor pairs, FMR/FNMR, ROC, EER.

Pure numpy score-array core for the degraded-input benchmark (the task4 CMC ->
ROC/FMR/FNMR extension, docs/PORT_INVENTORY.md section 2). Score conventions per
ARCHITECTURE.md: face scores are cosine similarities in [-1, 1]; higher scores
mean "more similar", and a pair is accepted when ``score >= threshold``.

Rate definitions used throughout:
- FMR  (false match rate)    = impostor pairs with ``score >= threshold`` / all impostor pairs.
- FNMR (false non-match rate)= genuine pairs with ``score <  threshold`` / all genuine pairs.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "build_pairs",
    "fmr_fnmr_at_threshold",
    "roc_curve",
    "eer",
    "fmr_at_fnmr",
]


def _as_2d_float(embeddings: object, name: str = "embeddings") -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D of shape (N, dim)")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} must contain at least 2 rows to form pairs")
    if arr.shape[1] < 1:
        raise ValueError(f"{name} must have dim >= 1")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _validate_scores(
    scores: object, is_genuine: object
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a score/mask pair; return (float64 scores, bool mask)."""
    s = np.asarray(scores, dtype=np.float64)
    g = np.asarray(is_genuine)
    if s.ndim != 1:
        raise ValueError("scores must be 1-D")
    if g.ndim != 1:
        raise ValueError("is_genuine must be 1-D")
    if s.shape[0] != g.shape[0]:
        raise ValueError("scores and is_genuine must have the same length")
    if s.shape[0] == 0:
        raise ValueError("scores must be non-empty (no pairs)")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    g = g.astype(bool)
    n_gen = int(g.sum())
    n_imp = int(g.size - n_gen)
    if n_gen == 0 or n_imp == 0:
        raise ValueError(
            "need at least one genuine and one impostor pair "
            f"(got {n_gen} genuine, {n_imp} impostor)"
        )
    return s, g


def build_pairs(
    embeddings: object, labels: object
) -> tuple[np.ndarray, np.ndarray]:
    """Build all unordered pairs from embeddings and score them with cosine similarity.

    Pairs are ``(i, j)`` with ``i < j`` in lexicographic order (row-major upper
    triangle) -- self-pairs excluded, deterministic ordering. Pairs with equal
    labels are genuine; all others are impostor.

    Args:
        embeddings: array-like of shape (N, dim), N >= 2. Rows must be non-zero.
        labels: array-like of shape (N,); any dtype supporting ``==``.

    Returns:
        ``(scores, is_genuine)``: float64 scores in [-1, 1] of length
        ``N*(N-1)//2``, and a boolean mask of the same length (True = genuine).
    """
    emb = _as_2d_float(embeddings)
    lab = np.asarray(labels)
    if lab.ndim != 1:
        raise ValueError("labels must be 1-D")
    if lab.shape[0] != emb.shape[0]:
        raise ValueError("labels must have length N matching embeddings")
    norms = np.linalg.norm(emb, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("embeddings must not contain zero vectors")
    unit = emb / norms[:, None]
    i, j = np.triu_indices(emb.shape[0], k=1)
    scores = np.einsum("ij,ij->i", unit[i], unit[j])
    is_genuine = lab[i] == lab[j]
    return scores.astype(np.float64), is_genuine.astype(bool)


def fmr_fnmr_at_threshold(
    scores: object, is_genuine: object, threshold: float
) -> tuple[float, float]:
    """Compute (FMR, FNMR) at one decision threshold.

    A pair is accepted as a match when ``score >= threshold``:
    - ``fmr``   = impostor pairs with score >= threshold, over all impostor pairs.
    - ``fnmr``  = genuine pairs with score < threshold, over all genuine pairs.

    Raises:
        ValueError: empty/mismatched scores, missing genuine or impostor pairs,
            or a non-finite threshold.
    """
    s, g = _validate_scores(scores, is_genuine)
    t = float(threshold)
    if not np.isfinite(t):
        raise ValueError("threshold must be finite")
    imp = s[~g]
    gen = s[g]
    fmr = float(np.mean(imp >= t))
    fnmr = float(np.mean(gen < t))
    return fmr, fnmr


def roc_curve(
    scores: object,
    is_genuine: object,
    thresholds: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep decision thresholds and return ``(fmr, fnmr, thresholds)``.

    If ``thresholds`` is None, every unique score is used as a threshold plus a
    sentinel just above the maximum score, so the curve spans the full range:
    the first point is (FMR=1, FNMR=0) and the last is (FMR=0, FNMR=1).
    Returned arrays are sorted by ascending threshold, so FMR is non-increasing
    and FNMR non-decreasing along the arrays.

    Raises:
        ValueError: empty/mismatched scores, missing genuine or impostor pairs,
            or an empty/non-finite ``thresholds`` array.
    """
    s, g = _validate_scores(scores, is_genuine)
    if thresholds is None:
        uniq = np.unique(s)
        sentinel = np.nextafter(uniq[-1], np.inf)
        thr = np.concatenate([uniq, [sentinel]])
    else:
        thr = np.asarray(thresholds, dtype=np.float64)
        if thr.ndim != 1 or thr.shape[0] == 0:
            raise ValueError("thresholds must be a non-empty 1-D array")
        if not np.all(np.isfinite(thr)):
            raise ValueError("thresholds must be finite")
    order = np.argsort(thr, kind="stable")
    thr = thr[order]

    imp = np.sort(s[~g])
    gen = np.sort(s[g])
    # fmr(t) = fraction of impostor scores >= t; fnmr(t) = fraction of genuine scores < t.
    fmr = (imp.size - np.searchsorted(imp, thr, side="left")) / imp.size
    fnmr = np.searchsorted(gen, thr, side="left") / gen.size
    return fmr.astype(np.float64), fnmr.astype(np.float64), thr.astype(np.float64)


def eer(scores: object, is_genuine: object) -> float:
    """Equal error rate: the FMR == FNMR operating point, by linear interpolation.

    Interpolation runs between the two adjacent ROC points where
    ``fmr - fnmr`` changes sign. For a constant score set the EER is 0.5;
    for perfect separation it is 0.0.

    Raises:
        ValueError: empty/mismatched scores, or missing genuine/impostor pairs.
    """
    fmr, fnmr, _ = roc_curve(scores, is_genuine)
    diff = fmr - fnmr  # non-increasing along ascending thresholds
    if diff[0] <= 0.0:
        return float((fmr[0] + fnmr[0]) / 2.0)
    if diff[-1] >= 0.0:
        return float((fmr[-1] + fnmr[-1]) / 2.0)
    i = int(np.argmax(diff <= 0.0))  # first point at or below zero
    j = i - 1
    w = diff[j] / (diff[j] - diff[i])  # in (0, 1]
    return float(fmr[j] + w * (fmr[i] - fmr[j]))


def fmr_at_fnmr(
    scores: object, is_genuine: object, target_fnmr: float
) -> float:
    """FMR at the best operating point whose FNMR is at most ``target_fnmr``.

    Uses linear interpolation between adjacent ROC points where FNMR crosses
    the target. ``target_fnmr`` must be in [0, 1]; 0.0 requests "FMR at
    FNMR=0" (the strictest threshold that never rejects genuine pairs).

    Raises:
        ValueError: empty/mismatched scores, missing genuine/impostor pairs,
            or a target outside [0, 1].
    """
    s, g = _validate_scores(scores, is_genuine)
    target = float(target_fnmr)
    if not np.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target_fnmr must be a finite float in [0, 1]")
    fmr, fnmr, _ = roc_curve(s, g)
    # fnmr is non-decreasing along ascending thresholds; find its crossing.
    if target >= fnmr[-1]:
        return float(fmr[-1])
    i = int(np.argmax(fnmr >= target))  # first point at or above target
    if fnmr[i] == target or i == 0:
        return float(fmr[i])
    j = i - 1
    w = (target - fnmr[j]) / (fnmr[i] - fnmr[j])
    return float(fmr[j] + w * (fmr[i] - fmr[j]))
