"""Degraded-input verification experiment core: condition sweeps and gating ablations.

Pure, deterministic, side-effect-free functions (data in, data out) for the
Phase 1 operational evaluation (PLAN.md Phase 1; ARCHITECTURE.md "Evaluation
pipeline") -- the task4 CMC -> ROC/FMR/FNMR extension described in
docs/PORT_INVENTORY.md section 2.

Protocol per condition: the *probe* set is every clean labeled image degraded
with the condition's operator at the condition's severity; the *gallery* is the
same images, clean. Genuine pairs are same-identity probe-vs-gallery
combinations (including the same-item pair ``i == j``); impostor pairs are
cross-identity. Pair scores are cosine similarities produced by
``metrics.build_pairs`` -- all verification math lives in
``grey_resolve.evaluation.metrics``.

Severity semantics: ``severity`` is forwarded verbatim to the degradation
operator as its strength parameter (``sigma`` / ``delta`` / ``scale`` /
``angle_deg``), so units differ per operator (see
``grey_resolve.degradation.operators.DEGRADATIONS``). Severity 0 is the
identity for every operator; the reserved ``"clean"`` condition skips
degradation entirely and requires severity 0.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np

from grey_resolve.degradation.operators import DEGRADATIONS
from grey_resolve.evaluation.metrics import (
    build_pairs,
    eer,
    fmr_at_fnmr,
    fmr_fnmr_at_threshold,
    roc_curve,
)

__all__ = [
    "CLEAN_CONDITION",
    "CLEAN_NAME",
    "FMR_AT_FNM_KEY",
    "DegradationCondition",
    "DegradedEmbeddings",
    "EmbedFn",
    "QualityFn",
    "degrade_embeddings",
    "gated_vs_ungated",
    "probe_gallery_scores",
    "run_condition_sweep",
    "score_condition",
]

CLEAN_NAME = "clean"
CLEAN_SEVERITY = 0.0
FNM_TARGET = 0.01
FMR_AT_FNM_KEY = f"fmr_at_fnmr_{FNM_TARGET:g}"

EmbedFn = Callable[[np.ndarray], np.ndarray]
"""Maps one uint8 (H, W, 3) image to a 1-D embedding vector (any finite dtype)."""

QualityFn = Callable[[np.ndarray], float]
"""Maps one uint8 (H, W, 3) image to a finite quality score (higher = better)."""

ConditionLike = Union["DegradationCondition", tuple]


@dataclass(frozen=True)
class DegradationCondition:
    """One benchmark condition: degradation operator name + severity.

    Args:
        name: key of ``grey_resolve.degradation.operators.DEGRADATIONS``, or the
            reserved ``"clean"`` (no degradation).
        severity: the operator's own strength parameter (``sigma`` / ``delta`` /
            ``scale`` / ``angle_deg``); must be finite and >= 0. The ``"clean"``
            condition requires severity 0.0.

    Raises:
        ValueError: unknown name, non-finite/negative severity, or a non-zero
            severity on the ``"clean"`` condition.
    """

    name: str
    severity: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("condition name must be a non-empty string")
        if self.name != CLEAN_NAME and self.name not in DEGRADATIONS:
            known = ", ".join(sorted((CLEAN_NAME, *DEGRADATIONS)))
            raise ValueError(f"unknown degradation {self.name!r}; expected one of: {known}")
        try:
            sev = float(self.severity)
        except (TypeError, ValueError):
            raise ValueError("severity must be a finite float >= 0") from None
        if not np.isfinite(sev) or sev < 0.0:
            raise ValueError("severity must be a finite float >= 0")
        if self.name == CLEAN_NAME and sev != 0.0:
            raise ValueError("the 'clean' condition requires severity 0.0")
        object.__setattr__(self, "severity", sev)


CLEAN_CONDITION = DegradationCondition(CLEAN_NAME, CLEAN_SEVERITY)
"""The no-degradation baseline condition (name 'clean', severity 0.0)."""


@dataclass(frozen=True)
class DegradedEmbeddings:
    """Probe (degraded) and gallery (clean) embedding matrices for one condition.

    ``query`` and ``gallery`` both have shape (N, dim); ``labels`` has length N
    and is shared, so probe row i and gallery row i describe the same source
    image (degraded vs clean).
    """

    query: np.ndarray
    gallery: np.ndarray
    labels: np.ndarray

    def pair_scores(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, is_genuine) for all probe-vs-gallery pairs."""
        return probe_gallery_scores(self.query, self.gallery, self.labels)


# --------------------------------------------------------------------- validation


def _validated_inputs(
    images: Sequence[np.ndarray], labels: object
) -> tuple[list[np.ndarray], np.ndarray]:
    """Validate the (images, labels) contract; return (image list, 1-D labels)."""
    imgs = list(images)
    lab = np.asarray(labels)
    if lab.ndim != 1:
        raise ValueError("labels must be 1-D")
    if lab.size == 0:
        raise ValueError("labels must be non-empty")
    if len(imgs) == 0:
        raise ValueError("images must be non-empty")
    if lab.shape[0] != len(imgs):
        raise ValueError(
            f"images and labels must have the same length (got {len(imgs)} images, "
            f"{lab.shape[0]} labels)"
        )
    for idx, img in enumerate(imgs):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"images[{idx}] must be a numpy.ndarray")
        if img.dtype != np.uint8:
            raise ValueError(f"images[{idx}] dtype must be uint8")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"images[{idx}] must have shape (H, W, 3)")
        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError(f"images[{idx}] must be non-empty")
    if len(imgs) < 2:
        raise ValueError("need at least 2 images to form probe/gallery pairs")
    if len(set(lab.tolist())) < 2:
        raise ValueError("labels must contain at least 2 distinct identities (need impostor pairs)")
    return imgs, lab


def _coerce_condition(condition: ConditionLike) -> DegradationCondition:
    """Accept a DegradationCondition or a (name, severity) tuple."""
    if isinstance(condition, DegradationCondition):
        return condition
    if isinstance(condition, tuple) and len(condition) == 2:
        return DegradationCondition(condition[0], condition[1])  # type: ignore[arg-type]
    raise TypeError("condition must be a DegradationCondition or a (name, severity) tuple")


def _embed_all(embed_fn: EmbedFn, images: Sequence[np.ndarray], role: str) -> np.ndarray:
    """Embed every image; return a float64 (N, dim) matrix."""
    vectors: list[np.ndarray] = []
    dim: int | None = None
    for idx, img in enumerate(images):
        vec = np.asarray(embed_fn(img), dtype=np.float64)
        if vec.ndim != 1 or vec.size == 0:
            raise ValueError(f"embed_fn must return a non-empty 1-D vector ({role} image {idx})")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"embed_fn returned non-finite values ({role} image {idx})")
        if not np.any(vec):
            raise ValueError(f"embed_fn returned a zero vector ({role} image {idx})")
        if dim is None:
            dim = int(vec.size)
        elif vec.size != dim:
            raise ValueError(
                f"embed_fn returned inconsistent dimensions ({role} image {idx}: "
                f"{vec.size} != {dim})"
            )
        vectors.append(vec)
    return np.stack(vectors)


def _degraded_images(images: Sequence[np.ndarray], condition: DegradationCondition) -> list[np.ndarray]:
    """Apply the condition's operator at its severity (pure; inputs untouched)."""
    if condition.name == CLEAN_NAME:
        return list(images)
    op = DEGRADATIONS[condition.name]
    operator = op if callable(op) else None
    if operator is None:  # pragma: no cover - registry is validated at condition construction
        raise ValueError(f"degradation {condition.name!r} is not callable")
    return [operator(img, condition.severity) for img in images]


def _plan_conditions(
    conditions: Sequence[ConditionLike], include_clean: bool
) -> list[DegradationCondition]:
    """Coerce conditions and prepend the clean baseline once, in stable order."""
    conds = [_coerce_condition(c) for c in conditions]
    planned: list[DegradationCondition] = []
    if include_clean and not any(c.name == CLEAN_NAME for c in conds):
        planned.append(CLEAN_CONDITION)
    planned.extend(conds)
    if not planned:
        raise ValueError("no conditions to evaluate (pass conditions or include_clean=True)")
    return planned


# ---------------------------------------------------------------- core: embeddings


def degrade_embeddings(
    images: Sequence[np.ndarray],
    labels: object,
    embed_fn: EmbedFn,
    condition: ConditionLike,
) -> DegradedEmbeddings:
    """Compute degraded probe embeddings and clean gallery embeddings.

    Every clean labeled image is degraded with ``condition``'s operator at its
    severity to form the probe set; the unmodified images form the gallery. The
    ``"clean"`` condition embeds the images once and reuses the result for both
    sides. Inputs are never mutated and ``embed_fn`` is only read.

    Args:
        images: N >= 2 uint8 images of shape (H, W, 3).
        labels: N labels (any dtype supporting ``==``); at least 2 distinct.
        embed_fn: callable mapping one image to a non-zero finite 1-D vector of
            consistent length across calls.
        condition: a DegradationCondition or (name, severity) tuple.

    Returns:
        DegradedEmbeddings with query (degraded probe), gallery (clean), labels.

    Raises:
        ValueError / TypeError: bad images, labels, condition, or embed_fn output.
    """
    cond = _coerce_condition(condition)
    imgs, lab = _validated_inputs(images, labels)
    gallery = _embed_all(embed_fn, imgs, "gallery")
    if cond.name == CLEAN_NAME:
        query = gallery.copy()
    else:
        query = _embed_all(embed_fn, _degraded_images(imgs, cond), "query")
    return DegradedEmbeddings(query=query, gallery=gallery, labels=lab.copy())


def probe_gallery_scores(
    query: object, gallery: object, labels: object
) -> tuple[np.ndarray, np.ndarray]:
    """Genuine/impostor cosine scores for every probe-vs-gallery pair.

    Scores come from ``metrics.build_pairs`` on the stacked ``[probe; gallery]``
    matrix; only the cross-block pairs (probe row i vs gallery row j) are kept,
    matching the PORT_INVENTORY protocol "genuine pairs (same identity, probe vs
    gallery) and impostor pairs (cross identity)". Ordering is deterministic:
    lexicographic (i, j).

    Args:
        query: probe embeddings, shape (N, dim), N >= 2.
        gallery: clean gallery embeddings, same shape as ``query``.
        labels: N labels; equal labels form genuine pairs.

    Returns:
        ``(scores, is_genuine)`` of length N*N.
    """
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    lab = np.asarray(labels)
    if q.ndim != 2 or g.ndim != 2:
        raise ValueError("query and gallery must be 2-D of shape (N, dim)")
    if q.shape != g.shape:
        raise ValueError("query and gallery must have the same shape")
    if lab.ndim != 1 or lab.shape[0] != q.shape[0]:
        raise ValueError("labels must be 1-D of length N matching query")
    if q.shape[0] < 2:
        raise ValueError("need at least 2 probe/gallery items")
    stacked = np.vstack([q, g])
    stacked_labels = np.concatenate([lab, lab])
    scores, is_genuine = build_pairs(stacked, stacked_labels)
    n = q.shape[0]
    i, j = np.triu_indices(2 * n, k=1)
    cross = (i < n) & (j >= n)
    return scores[cross], is_genuine[cross]


def score_condition(
    images: Sequence[np.ndarray],
    labels: object,
    embed_fn: EmbedFn,
    condition: ConditionLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: ``degrade_embeddings`` + ``probe_gallery_scores``."""
    return degrade_embeddings(images, labels, embed_fn, condition).pair_scores()


# ------------------------------------------------------------------- result blocks


def _condition_result(
    condition: DegradationCondition, scores: np.ndarray, is_genuine: np.ndarray
) -> dict:
    """Per-condition metrics dict: EER, FMR@FNMR target, and full ROC points."""
    fmr, fnmr, thresholds = roc_curve(scores, is_genuine)
    return {
        "condition": condition.name,
        "severity": condition.severity,
        "eer": float(eer(scores, is_genuine)),
        FMR_AT_FNM_KEY: float(fmr_at_fnmr(scores, is_genuine, FNM_TARGET)),
        "roc_points": [
            {"threshold": float(t), "fmr": float(f), "fnmr": float(n)}
            for t, f, n in zip(thresholds, fmr, fnmr)
        ],
        "n_pairs": int(scores.size),
        "n_genuine": int(is_genuine.sum()),
        "n_impostor": int((~is_genuine).sum()),
    }


def _rate_metrics(scores: np.ndarray, is_genuine: np.ndarray, threshold: float) -> dict:
    """FMR/FNMR at a fixed threshold plus EER and FMR@FNMR target."""
    fmr, fnmr = fmr_fnmr_at_threshold(scores, is_genuine, threshold)
    return {
        "n_pairs": int(scores.size),
        "n_genuine": int(is_genuine.sum()),
        "n_impostor": int((~is_genuine).sum()),
        "fmr": float(fmr),
        "fnmr": float(fnmr),
        "eer": float(eer(scores, is_genuine)),
        FMR_AT_FNM_KEY: float(fmr_at_fnmr(scores, is_genuine, FNM_TARGET)),
    }


# ------------------------------------------------------------------- sweep drivers


def run_condition_sweep(
    images: Sequence[np.ndarray],
    labels: object,
    conditions: Sequence[ConditionLike],
    embed_fn: EmbedFn,
    *,
    include_clean: bool = True,
) -> list[dict]:
    """Run the degraded-input verification sweep; one structured result per condition.

    The clean gallery is embedded once and reused across conditions. Unless a
    ``"clean"`` condition is passed explicitly, the clean baseline is prepended
    (``include_clean=True``). Conditions are evaluated in the given order after
    the baseline; the function is deterministic and side-effect free.

    Args:
        images: N >= 2 uint8 images of shape (H, W, 3).
        labels: N labels; at least 2 distinct identities.
        conditions: DegradationCondition values or (name, severity) tuples.
        embed_fn: image -> 1-D embedding vector callable.
        include_clean: prepend the clean baseline when absent.

    Returns:
        List of dicts, one per evaluated condition, each with keys
        ``"condition"``, ``"severity"``, ``"eer"``, ``"fmr_at_fnmr_0.01"``,
        ``"roc_points"`` (list of ``{"threshold", "fmr", "fnmr"}`` dicts), plus
        pair counts (``"n_pairs"``, ``"n_genuine"``, ``"n_impostor"``).
    """
    imgs, lab = _validated_inputs(images, labels)
    planned = _plan_conditions(conditions, include_clean)
    gallery = _embed_all(embed_fn, imgs, "gallery")
    results: list[dict] = []
    for cond in planned:
        if cond.name == CLEAN_NAME:
            query = gallery.copy()
        else:
            query = _embed_all(embed_fn, _degraded_images(imgs, cond), "query")
        scores, is_genuine = probe_gallery_scores(query, gallery, lab)
        results.append(_condition_result(cond, scores, is_genuine))
    return results


def gated_vs_ungated(
    images: Sequence[np.ndarray],
    labels: object,
    conditions: Sequence[ConditionLike],
    embed_fn: EmbedFn,
    quality_fn: QualityFn,
    quality_threshold: float,
    *,
    threshold: float = 0.5,
    include_clean: bool = True,
) -> list[dict]:
    """Quality-gate ablation: FMR/FNMR for the full set vs the quality-passing subset.

    The quality predicate is applied to each *degraded probe* image (the gate
    acts on what is being verified). Probes with
    ``quality_fn(image) >= quality_threshold`` form the gated subset; both sets
    are scored at the same operating ``threshold`` so FMR/FNMR are directly
    comparable. Embeddings of the gated subset are rows of the full matrices
    (``embed_fn`` is assumed deterministic). Degenerate subsets (fewer than 2
    probes or fewer than 2 identities) get ``None`` metrics instead of raising.

    Args:
        images: N >= 2 uint8 images of shape (H, W, 3).
        labels: N labels; at least 2 distinct identities.
        conditions: DegradationCondition values or (name, severity) tuples.
        embed_fn: image -> 1-D embedding vector callable.
        quality_fn: image -> finite quality score (higher = better).
        quality_threshold: probes with quality >= this value pass the gate.
        threshold: fixed cosine decision threshold for reported FMR/FNMR.
        include_clean: prepend the clean baseline when absent.

    Returns:
        List of dicts, one per evaluated condition, each with keys
        ``"condition"``, ``"severity"``, ``"quality_threshold"``,
        ``"operating_threshold"``, ``"full"`` and ``"gated"``. Both subsets carry
        ``"n"``, ``"pass_rate"``, ``"fmr"``, ``"fnmr"``, ``"eer"`` and
        ``"fmr_at_fnmr_0.01"``; the full set always has ``"pass_rate"`` 1.0.
    """
    imgs, lab = _validated_inputs(images, labels)
    if not callable(quality_fn):
        raise TypeError("quality_fn must be callable")
    qt = float(quality_threshold)
    if not np.isfinite(qt):
        raise ValueError("quality_threshold must be finite")
    op_thr = float(threshold)
    if not np.isfinite(op_thr):
        raise ValueError("threshold must be finite")
    planned = _plan_conditions(conditions, include_clean)
    gallery = _embed_all(embed_fn, imgs, "gallery")

    results: list[dict] = []
    for cond in planned:
        deg = _degraded_images(imgs, cond)
        if cond.name == CLEAN_NAME:
            query = gallery.copy()
        else:
            query = _embed_all(embed_fn, deg, "query")
        quality = np.array([_quality_value(quality_fn, im, i) for i, im in enumerate(deg)])
        mask = quality >= qt

        scores, is_genuine = probe_gallery_scores(query, gallery, lab)
        full = {"n": int(mask.size), "pass_rate": 1.0, **_rate_metrics(scores, is_genuine, op_thr)}

        n_pass = int(mask.sum())
        if n_pass >= 2 and len(set(lab[mask].tolist())) >= 2:
            sub_scores, sub_genuine = probe_gallery_scores(query[mask], gallery[mask], lab[mask])
            gated = {
                "n": n_pass,
                "pass_rate": float(n_pass / mask.size),
                **_rate_metrics(sub_scores, sub_genuine, op_thr),
            }
        else:
            gated = {
                "n": n_pass,
                "pass_rate": float(n_pass / mask.size),
                "n_pairs": None,
                "n_genuine": None,
                "n_impostor": None,
                "fmr": None,
                "fnmr": None,
                "eer": None,
                FMR_AT_FNM_KEY: None,
            }
        results.append(
            {
                "condition": cond.name,
                "severity": cond.severity,
                "quality_threshold": qt,
                "operating_threshold": op_thr,
                "full": full,
                "gated": gated,
            }
        )
    return results


def _quality_value(quality_fn: QualityFn, image: np.ndarray, idx: int) -> float:
    """Evaluate quality_fn with validation; raise on non-finite output."""
    score = float(quality_fn(image))
    if not np.isfinite(score):
        raise ValueError(f"quality_fn returned a non-finite score (probe {idx})")
    return score
