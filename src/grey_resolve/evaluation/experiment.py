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

EmbedFn = Callable[[np.ndarray], "np.ndarray | None"]
"""Maps one uint8 (H, W, 3) image to a 1-D embedding vector (any finite dtype).

Returning ``None`` means "no embedding for this image" (e.g. face detection
failed on a heavily degraded probe). Such images are excluded from scoring and
counted per condition (``n_query_dropped`` / ``n_gallery_dropped``) instead of
failing the run. Detection failure under degradation is a real outcome and must
be reported, not hidden.
"""

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
    n_dropped_query: int = 0
    n_dropped_gallery: int = 0

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


def _embed_all(
    embed_fn: EmbedFn, images: Sequence[np.ndarray], role: str, *, allow_empty: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Embed every image; return (float64 (K, dim) matrix, kept indices [K]).

    Images for which ``embed_fn`` returns ``None`` are skipped and excluded from
    the returned indices so callers can align rows across sides. Raises
    ValueError when no embeddings at all are produced or any returned vector is
    malformed.
    """
    vectors: list[np.ndarray] = []
    kept: list[int] = []
    dim: int | None = None
    for idx, img in enumerate(images):
        out = embed_fn(img)
        if out is None:
            continue
        vec = np.asarray(out, dtype=np.float64)
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
        kept.append(idx)
    if not vectors:
        if allow_empty:
            return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.int64)
        raise ValueError(f"no embeddings produced ({role}: embed_fn returned None for every image)")
    return np.stack(vectors), np.asarray(kept, dtype=np.int64)


def _aligned_rows(
    gidx: np.ndarray, qidx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align gallery/query rows on shared original-image indices.

    Returns ``(common_idx, gallery_rows, query_rows)`` such that
    ``gallery[gallery_rows]`` and ``query[query_rows]`` describe the same source
    images in ascending index order.
    """
    common = np.intersect1d(gidx, qidx, assume_unique=True)
    grow = np.searchsorted(gidx, common)
    qrow = np.searchsorted(qidx, common)
    return common, grow, qrow


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
        Images without an embedding on either side (``embed_fn`` returned
        ``None``) are excluded from all three arrays so rows stay aligned;
        ``n_dropped_query`` / ``n_dropped_gallery`` report the exclusions.

    Raises:
        ValueError / TypeError: bad images, labels, condition, or embed_fn output;
            fewer than 2 scoreable images or fewer than 2 distinct identities.
    """
    cond = _coerce_condition(condition)
    imgs, lab = _validated_inputs(images, labels)
    gallery, gidx = _embed_all(embed_fn, imgs, "gallery")
    if cond.name == CLEAN_NAME:
        query, qidx = gallery.copy(), gidx
    else:
        query, qidx = _embed_all(embed_fn, _degraded_images(imgs, cond), "query")
    common, grow, qrow = _aligned_rows(gidx, qidx)
    if common.size < 2 or len(set(lab[common].tolist())) < 2:
        raise ValueError(
            "fewer than 2 scoreable images or 2 distinct identities after embedding drops"
        )
    return DegradedEmbeddings(
        query=query[qrow],
        gallery=gallery[grow],
        labels=lab[common].copy(),
        n_dropped_query=int(len(imgs) - common.size),
        n_dropped_gallery=int(len(imgs) - gidx.size),
    )


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


def _safe_condition_result(
    condition: DegradationCondition,
    query: np.ndarray,
    gallery: np.ndarray,
    labels: np.ndarray,
    *,
    n_images: int,
    n_gallery: int,
    n_scored: int,
) -> dict:
    """Per-condition result with detection-failure counts and graceful degenerates.

    Conditions where too few images produced embeddings (or only one identity
    remains) report ``None`` metrics plus honest counts instead of raising:
    detection failure under heavy degradation is a real, reportable outcome.
    """
    counts = {
        "n_images": int(n_images),
        "n_gallery": int(n_gallery),
        "n_scored": int(n_scored),
        "n_query_dropped": int(n_images - n_scored),
        "n_gallery_dropped": int(n_images - n_gallery),
    }
    degenerate = n_scored < 2 or len(set(labels.tolist())) < 2
    if degenerate:
        return {
            "condition": condition.name,
            "severity": condition.severity,
            "eer": None,
            FMR_AT_FNM_KEY: None,
            "roc_points": [],
            "n_pairs": None,
            "n_genuine": None,
            "n_impostor": None,
            **counts,
        }
    scores, is_genuine = probe_gallery_scores(query, gallery, labels)
    return {**_condition_result(condition, scores, is_genuine), **counts}


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
    gallery, gidx = _embed_all(embed_fn, imgs, "gallery")
    if gidx.size < 2 or len(set(lab[gidx].tolist())) < 2:
        raise ValueError(
            "gallery produced fewer than 2 scoreable images or 2 distinct identities"
        )
    results: list[dict] = []
    for cond in planned:
        if cond.name == CLEAN_NAME:
            query, qidx = gallery.copy(), gidx
        else:
            query, qidx = _embed_all(
                embed_fn, _degraded_images(imgs, cond), "query", allow_empty=True
            )
        common, grow, qrow = _aligned_rows(gidx, qidx)
        results.append(
            _safe_condition_result(
                cond,
                query[qrow],
                gallery[grow],
                lab[common],
                n_images=len(imgs),
                n_gallery=int(gidx.size),
                n_scored=int(common.size),
            )
        )
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
        ``"operating_threshold"``, ``"n_dropped"``, ``"full"`` and ``"gated"``.
        Both subsets carry ``"n"``, ``"pass_rate"``, ``"fmr"``, ``"fnmr"``,
        ``"eer"`` and ``"fmr_at_fnmr_0.01"``; the full set always has
        ``"pass_rate"`` 1.0. ``pass_rate`` is over *scoreable* probes (images
        without embeddings are excluded and counted in ``n_dropped``).
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
    gallery, gidx = _embed_all(embed_fn, imgs, "gallery")
    if gidx.size < 2 or len(set(lab[gidx].tolist())) < 2:
        raise ValueError(
            "gallery produced fewer than 2 scoreable images or 2 distinct identities"
        )

    null_metrics = {
        "n_pairs": None,
        "n_genuine": None,
        "n_impostor": None,
        "fmr": None,
        "fnmr": None,
        "eer": None,
        FMR_AT_FNM_KEY: None,
    }
    results: list[dict] = []
    for cond in planned:
        deg = _degraded_images(imgs, cond)
        if cond.name == CLEAN_NAME:
            query, qidx = gallery.copy(), gidx
        else:
            query, qidx = _embed_all(embed_fn, deg, "query", allow_empty=True)
        common, grow, qrow = _aligned_rows(gidx, qidx)
        quality = np.array([_quality_value(quality_fn, im, i) for i, im in enumerate(deg)])

        n_scored = int(common.size)
        degenerate = n_scored < 2 or len(set(lab[common].tolist())) < 2
        if degenerate:
            full = {"n": n_scored, "pass_rate": 1.0, **null_metrics}
            gated = {"n": 0, "pass_rate": 0.0, **null_metrics}
        else:
            scores, is_genuine = probe_gallery_scores(query[qrow], gallery[grow], lab[common])
            full = {"n": n_scored, "pass_rate": 1.0, **_rate_metrics(scores, is_genuine, op_thr)}
            gate_mask = quality[common] >= qt
            n_pass = int(gate_mask.sum())
            if n_pass >= 2 and len(set(lab[common][gate_mask].tolist())) >= 2:
                sub_scores, sub_genuine = probe_gallery_scores(
                    query[qrow][gate_mask], gallery[grow][gate_mask], lab[common][gate_mask]
                )
                gated = {
                    "n": n_pass,
                    "pass_rate": float(n_pass / n_scored),
                    **_rate_metrics(sub_scores, sub_genuine, op_thr),
                }
            else:
                gated = {
                    "n": n_pass,
                    "pass_rate": float(n_pass / n_scored),
                    **null_metrics,
                }
        results.append(
            {
                "condition": cond.name,
                "severity": cond.severity,
                "quality_threshold": qt,
                "operating_threshold": op_thr,
                "n_dropped": int(len(imgs) - n_scored),
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
