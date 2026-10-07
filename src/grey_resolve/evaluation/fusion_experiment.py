"""Phase 2 fusion ablation core: face-only vs fused persona ranking (pure, deterministic).

Implements the key Phase 2 evidence for the "context breaks near-ties" claim
(docs/SYNTHETIC_SCENARIO_SPEC.md "Evaluation use"): a face-only vs fused ranking
ablation on measured near-tie ambiguity sets, a context-noise sweep that reports
the negative region honestly (fusion must be allowed to stop helping), and an
optional fusion-weight sweep.

Ranking semantics (persona-level identity resolution): for each query item (a
media item of persona P) the candidate set is the personas of its near-tie set
(including P); when the query has no near-tie set the candidate set falls back
to ALL personas. Each candidate persona's face score is the MAX face cosine over
that persona's gallery items (the query item itself is excluded from the
galleries), and the true label is P. Fused ranking scores the same candidates
with ``scorer.fuse(face_score, query_ctx, persona_ctx)`` where ``persona_ctx``
is the context of that persona's best face-scoring gallery item. Ties are
broken by ascending persona id (face) / ascending persona id (fused), so
rankings are deterministic.

Reported metrics per method are hit@1, hit@k, MRR, and n_queries -- overall and
restricted to queries whose candidate set is a near-tie set -- plus per-set
detail and the face-score gap distribution between the top-1 and top-2 candidate
personas. All metrics are per-query 1.0/0.0 values averaged over queries.

Everything here is pure and deterministic: no file IO, no model imports;
randomness flows only through seeded ``numpy.random.Generator`` streams
(``[seed, stream]``), so identical inputs always produce identical outputs.
Negative results are reported, never suppressed.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from grey_resolve.config import ThresholdProfile
from grey_resolve.fusion.base import FusionScorer
from grey_resolve.scenario.generator import NearTieSet
from grey_resolve.types import ContextMetadata

__all__ = [
    "NOISE_KINDS",
    "RankingItem",
    "hit_at_1",
    "hit_at_k",
    "mrr",
    "run_ambiguity_ablation",
    "run_context_noise_sweep",
    "run_context_weight_sweep",
]

NOISE_KINDS = ("replace", "drop")
"""Context corruption kinds for the noise sweep (see run_context_noise_sweep)."""


# --------------------------------------------------------------------- ranking metrics


def _first_rank(ranked_ids: Sequence[Hashable], true_id: Hashable) -> int | None:
    """Return the 1-based rank of ``true_id`` in ``ranked_ids``, or None if absent.

    When the id occurs more than once the first occurrence defines the rank.
    """
    for rank, cand in enumerate(ranked_ids, start=1):
        if cand == true_id:
            return rank
    return None


def hit_at_1(ranked_ids: Sequence[Hashable], true_id: Hashable) -> float:
    """1.0 when ``true_id`` ranks first, else 0.0 (empty list -> 0.0).

    Args:
        ranked_ids: candidate ids, best first (plain list; duplicates allowed).
        true_id: the correct candidate id.

    Returns:
        Per-query hit indicator as a float (1.0 / 0.0); callers average it.
    """
    return 1.0 if _first_rank(list(ranked_ids), true_id) == 1 else 0.0


def hit_at_k(ranked_ids: Sequence[Hashable], true_id: Hashable, k: int) -> float:
    """1.0 when ``true_id`` ranks within the top ``k``, else 0.0.

    Args:
        ranked_ids: candidate ids, best first.
        true_id: the correct candidate id.
        k: cutoff, an integer >= 1.

    Returns:
        Per-query hit indicator as a float (1.0 / 0.0).

    Raises:
        ValueError: ``k`` is not an integer >= 1.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be an integer >= 1")
    rank = _first_rank(list(ranked_ids), true_id)
    return 1.0 if rank is not None and rank <= int(k) else 0.0


def mrr(ranked_ids: Sequence[Hashable], true_id: Hashable) -> float:
    """Reciprocal rank of ``true_id`` in ``ranked_ids`` (0.0 when absent).

    Args:
        ranked_ids: candidate ids, best first.
        true_id: the correct candidate id.

    Returns:
        ``1.0 / rank`` as a float, or 0.0 when ``true_id`` does not appear.
    """
    rank = _first_rank(list(ranked_ids), true_id)
    return 0.0 if rank is None else 1.0 / rank


# --------------------------------------------------------------------------- data model


@dataclass(frozen=True)
class RankingItem:
    """One item in the ranking pool: face embedding plus optional context.

    Args:
        item_id: unique id (e.g. the scenario ``media_id``).
        persona_id: the item's persona label (the resolution target).
        embedding: 1-D face embedding (any finite non-zero vector; cosine is
            computed internally after normalization). This is the item's
            *gallery* view (clean evidence).
        context: the item's ContextMetadata (may be empty).
        query_embedding: optional *query* view of the same face as observed
            (e.g. embedded from a degraded photo). When set, queries score
            against candidate galleries with this vector while the item's own
            gallery evidence stays clean -- this is how the degraded-query
            ablation compresses face scores into the ambiguous regime.
    """

    item_id: str
    persona_id: str
    embedding: np.ndarray
    context: ContextMetadata = field(default_factory=ContextMetadata)
    query_embedding: np.ndarray | None = None


@dataclass(frozen=True)
class _Item:
    """Validated item: unit-norm embedding plus its metadata."""

    item_id: str
    persona_id: str
    unit: np.ndarray
    context: ContextMetadata
    query_unit: np.ndarray | None = None


@dataclass(frozen=True)
class _QueryEvidence:
    """Precomputed per-query candidate evidence (context-free of the scorer)."""

    query: _Item
    candidate_ids: tuple[str, ...]
    face_scores: tuple[float, ...]
    persona_contexts: tuple[ContextMetadata, ...]
    scope: str  # "near_tie" (candidate set is a near-tie set) or "overall" (fallback)
    set_key: str | None


# -------------------------------------------------------------------------- validation


def _validate_items(items: Sequence[RankingItem]) -> tuple[_Item, ...]:
    """Validate the item pool; return validated items sorted by ascending item_id.

    Raises:
        ValueError: empty pool, bad ids, duplicate item ids, malformed/non-finite/
            zero/inconsistent embeddings, or fewer than 2 distinct personas.
        TypeError: an element is not a RankingItem.
    """
    pool = list(items)
    if not pool:
        raise ValueError("items must be non-empty")
    out: list[_Item] = []
    seen: set[str] = set()
    dim: int | None = None
    for idx, it in enumerate(pool):
        if not isinstance(it, RankingItem):
            raise TypeError(f"items[{idx}] must be a RankingItem")
        if not isinstance(it.item_id, str) or not it.item_id:
            raise ValueError(f"items[{idx}].item_id must be a non-empty string")
        if not isinstance(it.persona_id, str) or not it.persona_id:
            raise ValueError(f"items[{idx}].persona_id must be a non-empty string")
        if it.item_id in seen:
            raise ValueError(f"item_id {it.item_id!r} is not unique")
        seen.add(it.item_id)
        vec = np.asarray(it.embedding, dtype=np.float64)
        if vec.ndim != 1 or vec.size == 0:
            raise ValueError(f"items[{idx}].embedding must be a non-empty 1-D vector")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"items[{idx}].embedding must be finite")
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise ValueError(f"items[{idx}].embedding must not be a zero vector")
        if dim is None:
            dim = int(vec.size)
        elif vec.size != dim:
            raise ValueError(
                f"items[{idx}].embedding has dimension {vec.size}, expected {dim}"
            )
        query_unit: np.ndarray | None = None
        if it.query_embedding is not None:
            qvec = np.asarray(it.query_embedding, dtype=np.float64)
            if qvec.ndim != 1 or qvec.size != vec.size:
                raise ValueError(
                    f"items[{idx}].query_embedding must be a 1-D vector of dim {vec.size}"
                )
            if not np.all(np.isfinite(qvec)):
                raise ValueError(f"items[{idx}].query_embedding must be finite")
            qnorm = float(np.linalg.norm(qvec))
            if qnorm == 0.0:
                raise ValueError(f"items[{idx}].query_embedding must not be a zero vector")
            query_unit = qvec / qnorm
        out.append(_Item(it.item_id, it.persona_id, vec / norm, it.context, query_unit))
    if len({it.persona_id for it in out}) < 2:
        raise ValueError("items must span at least 2 personas")
    out.sort(key=lambda it: it.item_id)
    return tuple(out)


def _validate_sets(
    near_tie_sets: Sequence[NearTieSet], items: Sequence[_Item]
) -> tuple[NearTieSet, ...]:
    """Validate near-tie sets against the item pool; return them in given order.

    Raises:
        ValueError: a malformed set (fewer than 2 distinct personas) or a set
            referencing a persona that has no items ("persona not in set").
        TypeError: an element is not a NearTieSet.
    """
    personas = {it.persona_id for it in items}
    out: list[NearTieSet] = []
    for idx, s in enumerate(near_tie_sets):
        if not isinstance(s, NearTieSet):
            raise TypeError(f"near_tie_sets[{idx}] must be a NearTieSet")
        ids = tuple(s.identity_ids)
        if len(set(ids)) < 2 or any(not isinstance(p, str) or not p for p in ids):
            raise ValueError(f"near_tie_sets[{idx}] must hold >= 2 distinct persona ids")
        missing = sorted(set(ids) - personas)
        if missing:
            raise ValueError(
                f"near_tie_sets[{idx}] references persona(s) with no items: {missing}"
            )
        out.append(s)
    return tuple(out)


def _resolve_query_ids(items: Sequence[_Item], query_ids: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve the query subset (default: every item) to sorted unique item ids.

    Raises:
        ValueError: an empty/unknown/duplicated query id ("mismatched ids").
    """
    if query_ids is None:
        return tuple(it.item_id for it in items)
    qids = [str(q) for q in query_ids]
    if not qids:
        raise ValueError("query_ids must be non-empty when provided")
    if len(set(qids)) != len(qids):
        raise ValueError("query_ids must not contain duplicates")
    known = {it.item_id for it in items}
    unknown = sorted(set(qids) - known)
    if unknown:
        raise ValueError(f"query_ids contain unknown item ids: {unknown}")
    return tuple(sorted(qids))


def _checked_k(k: int) -> int:
    """Validate the hit@k cutoff; return it as an int."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be an integer >= 1")
    return int(k)


# -------------------------------------------------------------------------- evidence


def _checked_gap_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    """Validate gap cutoffs: finite, >= 0, ascending-unique order preserved."""
    out: list[float] = []
    for t in thresholds:
        ft = float(t)
        if not np.isfinite(ft) or ft < 0:
            raise ValueError("gap_thresholds must be finite and >= 0")
        out.append(ft)
    return tuple(out)


def _prepare(
    items: Sequence[RankingItem],
    near_tie_sets: Sequence[NearTieSet],
    query_ids: Sequence[str] | None,
) -> tuple[list[_QueryEvidence], int]:
    """Build per-query candidate evidence; return (evidence, n_skipped queries).

    A query is skipped (and counted) when its own persona has no gallery item
    other than the query itself. Evidence order is ascending query item_id.
    """
    pool = _validate_items(items)
    sets = _validate_sets(near_tie_sets, pool)
    qids = _resolve_query_ids(pool, query_ids)

    by_persona: dict[str, list[_Item]] = {}
    for it in pool:
        by_persona.setdefault(it.persona_id, []).append(it)
    all_personas = tuple(sorted(by_persona))
    set_of_persona: dict[str, NearTieSet] = {}
    for s in sets:  # first set wins if personas overlap across sets
        for p in sorted(set(s.identity_ids)):
            set_of_persona.setdefault(p, s)

    want = set(qids)
    evidence: list[_QueryEvidence] = []
    n_skipped = 0
    for q in pool:
        if q.item_id not in want:
            continue
        own_gallery = [g for g in by_persona[q.persona_id] if g.item_id != q.item_id]
        if not own_gallery:
            n_skipped += 1
            continue
        s = set_of_persona.get(q.persona_id)
        if s is not None:
            candidates = tuple(sorted(set(s.identity_ids)))
            scope, set_key = "near_tie", "|".join(sorted(set(s.identity_ids)))
        else:
            candidates, scope, set_key = all_personas, "overall", None
        qvec = q.query_unit if q.query_unit is not None else q.unit
        face_scores: list[float] = []
        persona_contexts: list[ContextMetadata] = []
        for c in candidates:
            best_score = -np.inf
            best_ctx = ContextMetadata()
            first = True
            for g in by_persona[c]:
                if g.item_id == q.item_id:
                    continue
                score = float(np.dot(qvec, g.unit))
                if first or score > best_score:  # ties keep the smallest item_id
                    best_score, best_ctx, first = score, g.context, False
            face_scores.append(float(best_score))
            persona_contexts.append(best_ctx)
        evidence.append(
            _QueryEvidence(
                query=q,
                candidate_ids=candidates,
                face_scores=tuple(face_scores),
                persona_contexts=tuple(persona_contexts),
                scope=scope,
                set_key=set_key,
            )
        )
    return evidence, n_skipped


def _rank_personas(
    ev: _QueryEvidence,
    scorer: FusionScorer | None,
    method: str,
    query_context: ContextMetadata | None = None,
) -> list[str]:
    """Rank the candidate personas for one query; return persona ids best-first.

    ``method`` is ``"face"`` (face cosine only) or ``"fused"``
    (``scorer.fuse(face, query_ctx, persona_ctx)``). ``query_context`` overrides
    the query item's own context (used by the noise sweep). Ties break by
    ascending persona id.
    """
    if method == "face":
        scores = list(ev.face_scores)
    else:
        assert scorer is not None  # callers pass a scorer for "fused"
        q_ctx = ev.query.context if query_context is None else query_context
        scores = [
            float(scorer.fuse(face, q_ctx, pctx))
            for face, pctx in zip(ev.face_scores, ev.persona_contexts)
        ]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], ev.candidate_ids[i]))
    return [ev.candidate_ids[i] for i in order]


# --------------------------------------------------------------------------- aggregates


def _aggregate(ranked_lists: Sequence[Sequence[Hashable]], truths: Sequence[Hashable], k: int) -> dict[str, Any]:
    """Average per-query metrics over (ranked_list, true_id) pairs."""
    n = len(ranked_lists)
    if n == 0:
        return {"n_queries": 0, "hit_at_1": None, "hit_at_k": None, "mrr": None}
    h1 = sum(hit_at_1(r, t) for r, t in zip(ranked_lists, truths)) / n
    hk = sum(hit_at_k(r, t, k) for r, t in zip(ranked_lists, truths)) / n
    mr = sum(mrr(r, t) for r, t in zip(ranked_lists, truths)) / n
    return {"n_queries": int(n), "hit_at_1": float(h1), "hit_at_k": float(hk), "mrr": float(mr)}


def _method_block(
    evidence: Sequence[_QueryEvidence],
    ranked_lists: Sequence[list[str]],
    k: int,
    include_per_set: bool,
) -> dict[str, Any]:
    """Per-method metrics: overall, near-tie-only, and optionally per-set."""
    truths = [ev.query.persona_id for ev in evidence]
    overall = _aggregate(ranked_lists, truths, k)
    near_idx = [i for i, ev in enumerate(evidence) if ev.scope == "near_tie"]
    near = _aggregate([ranked_lists[i] for i in near_idx], [truths[i] for i in near_idx], k)
    block: dict[str, Any] = {"overall": overall, "near_tie": near}
    if include_per_set:
        per_set: dict[str, Any] = {}
        for key in sorted({ev.set_key for ev in evidence if ev.set_key is not None}):
            idx = [i for i, ev in enumerate(evidence) if ev.set_key == key]
            per_set[key] = _aggregate(
                [ranked_lists[i] for i in idx], [truths[i] for i in idx], k
            )
        block["per_set"] = per_set
    return block


def _face_gap(ev: _QueryEvidence) -> float | None:
    """Top-1 vs top-2 face-score gap for one query (None with <2 candidates)."""
    ranked = sorted(ev.face_scores, reverse=True)
    return float(ranked[0] - ranked[1]) if len(ranked) >= 2 else None


def _gap_summary(evidence: Sequence[_QueryEvidence]) -> dict[str, Any]:
    """Mean/median face-score gap between the top-1 and top-2 candidate personas."""
    gaps = []
    for ev in evidence:
        ranked = sorted(ev.face_scores, reverse=True)
        if len(ranked) >= 2:
            gaps.append(float(ranked[0] - ranked[1]))
    if not gaps:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": len(gaps),
        "mean": float(np.mean(gaps)),
        "median": float(np.median(gaps)),
    }


def _delta_block(fused: Mapping[str, Any], face: Mapping[str, Any]) -> dict[str, Any]:
    """Per-scope metric deltas (fused minus face_only); None when undefined."""
    out: dict[str, Any] = {}
    for scope in ("overall", "near_tie"):
        f, b = fused.get(scope, {}), face.get(scope, {})
        out[scope] = {
            key: (
                None
                if f.get(key) is None or b.get(key) is None
                else float(f[key] - b[key])
            )
            for key in ("hit_at_1", "hit_at_k", "mrr")
        }
    return out


def _score_fused(
    evidence: Sequence[_QueryEvidence],
    scorer: FusionScorer,
    overrides: Mapping[str, ContextMetadata] | None,
) -> list[list[str]]:
    """Fused rankings per query, with optional per-query context overrides."""
    return [
        _rank_personas(
            ev,
            scorer,
            "fused",
            query_context=None if overrides is None else overrides.get(ev.query.item_id),
        )
        for ev in evidence
    ]


# ------------------------------------------------------------------------ ablation


def run_ambiguity_ablation(
    items: Sequence[RankingItem],
    scorer: FusionScorer,
    near_tie_sets: Sequence[NearTieSet],
    *,
    k: int = 5,
    query_ids: Sequence[str] | None = None,
    gap_thresholds: Sequence[float] = (),
) -> dict[str, Any]:
    """Face-only vs fused persona ranking on near-tie ambiguity sets.

    For each query item (persona P) the candidate personas are its near-tie set
    (including P), or all personas when P has no near-tie set. Candidate face
    scores are the MAX face cosine over each persona's gallery items (the query
    item excluded); the true label is P. Method (a) ranks by face cosine alone;
    method (b) ranks by ``scorer.fuse(face, query_ctx, persona_ctx)``.

    Args:
        items: RankingItem pool (item_id, persona_id, embedding, context).
        scorer: fusion scorer with ``fuse`` (e.g. LinearFusionScorer).
        near_tie_sets: measured NearTieSet values (may be empty: every query
            then falls back to the all-personas candidate set).
        k: hit@k cutoff (integer >= 1).
        query_ids: optional query subset (default: every item).
        gap_thresholds: optional face-gap cutoffs; for each threshold the
            output includes metrics restricted to queries whose top-1 vs
            top-2 face-score gap is <= the cutoff ("gap_le" block) -- the
            actual-tie regime where fusion should matter.

    Returns:
        JSON-friendly dict with ``"k"``, ``"n_query_items"``, ``"n_skipped"``,
        ``"face_only"`` and ``"fused"`` blocks (``"overall"``, ``"near_tie"``,
        ``"per_set"`` each holding ``hit_at_1`` / ``hit_at_k`` / ``mrr`` /
        ``n_queries``), ``"gap_le"`` (one block per gap cutoff with
        ``face_only`` / ``fused`` / ``delta`` metrics on the tight-gap
        queries), and ``"face_score_gap"`` (mean/median top-1-vs-top-2
        face-score gap per scope). Empty scopes report ``None`` metrics with
        ``n_queries`` 0.

    Raises:
        ValueError / TypeError: invalid items, near-tie sets, query ids, or k
            (see the module validators); e.g. empty items, a near-tie persona
            with no items ("persona not in set"), or unknown query ids
            ("mismatched ids").
    """
    kk = _checked_k(k)
    if not callable(getattr(scorer, "fuse", None)) or not callable(
        getattr(scorer, "context_score", None)
    ):
        raise TypeError("scorer must provide fuse() and context_score()")
    thresholds = _checked_gap_thresholds(gap_thresholds)
    evidence, n_skipped = _prepare(items, near_tie_sets, query_ids)
    truths = [ev.query.persona_id for ev in evidence]
    face_ranked = [_rank_personas(ev, None, "face") for ev in evidence]
    fused_ranked = _score_fused(evidence, scorer, None)
    gap_le: dict[str, Any] = {}
    for t in thresholds:
        idx = [
            i
            for i, ev in enumerate(evidence)
            if (_gap := _face_gap(ev)) is not None and _gap <= t
        ]
        face_sub = _aggregate([face_ranked[i] for i in idx], [truths[i] for i in idx], kk)
        fused_sub = _aggregate([fused_ranked[i] for i in idx], [truths[i] for i in idx], kk)
        gap_le[f"{t:g}"] = {
            "gap_max": float(t),
            "n_queries": face_sub["n_queries"],
            "face_only": face_sub,
            "fused": fused_sub,
            "delta": {
                key: (
                    None
                    if fused_sub[key] is None or face_sub[key] is None
                    else float(fused_sub[key] - face_sub[key])
                )
                for key in ("hit_at_1", "hit_at_k", "mrr")
            },
        }
    return {
        "k": kk,
        "n_query_items": len(evidence),
        "n_skipped": int(n_skipped),
        "face_only": _method_block(evidence, face_ranked, kk, include_per_set=True),
        "fused": _method_block(evidence, fused_ranked, kk, include_per_set=True),
        "gap_le": gap_le,
        "face_score_gap": {
            "overall": _gap_summary(evidence),
            "near_tie": _gap_summary([ev for ev in evidence if ev.scope == "near_tie"]),
        },
    }


# ---------------------------------------------------------------------- noise sweep


def _corrupt_context(
    ctx: ContextMetadata,
    kind: str,
    donor: _Item,
) -> ContextMetadata:
    """Corrupt one query context with the given kind (deterministic given the donor).

    Kinds (mirroring the scenario generator's noise semantics):
    - ``"replace"``: replace the context fields with plausible-wrong values
      taken from another persona's item (the donor's full context).
    - ``"drop"``: drop the text entities (other fields unchanged).
    """
    if kind == "replace":
        return donor.context
    return ContextMetadata(
        timestamp=ctx.timestamp,
        geo_cluster=ctx.geo_cluster,
        source_platform=ctx.source_platform,
        text_entities=(),
    )


def run_context_noise_sweep(
    items: Sequence[RankingItem],
    scorer: FusionScorer,
    near_tie_sets: Sequence[NearTieSet],
    noise_rates: Sequence[float],
    seed: int,
    *,
    k: int = 5,
    query_ids: Sequence[str] | None = None,
    noise_kinds: Sequence[str] = NOISE_KINDS,
) -> dict[str, Any]:
    """Sweep query-context corruption rates and report fused metrics vs face-only.

    For each noise rate p, each query item's context is corrupted with
    probability p (deterministic ``numpy.random.Generator([seed, stream])``
    stream per rate, queries rolled in ascending item_id order). The fused
    ranking is rerun on the corrupted query contexts; candidate personas and
    their contexts are unchanged. Face-only ranking is context-free, so its
    baseline is computed once.

    The per-rate rows report fused hit@1 / hit@k / MRR plus the delta versus
    face-only; negative deltas are reported verbatim. ``"first_negative_delta_rate"``
    names the first rate whose hit@1 delta is negative per scope (None when the
    swept region stays non-negative) so readers see where fusion stops helping.

    Args:
        items: RankingItem pool.
        scorer: fusion scorer with ``fuse``.
        near_tie_sets: measured NearTieSet values (may be empty).
        noise_rates: rates in [0, 1], evaluated in the given order.
        seed: non-negative integer seed for the corruption streams.
        k: hit@k cutoff.
        query_ids: optional query subset.
        noise_kinds: corruption kinds drawn uniformly per corrupted query
            (subset of ``NOISE_KINDS``; default both).

    Returns:
        JSON-friendly dict with ``"seed"``, ``"noise_kinds"``, ``"face_only"``
        baseline, ``"rows"`` (one per rate: ``"noise_rate"``, ``"n_corrupted"``,
        ``"fused"``, ``"delta"`` per scope), and ``"first_negative_delta_rate"``.

    Raises:
        ValueError / TypeError: invalid items, sets, query ids, k, seed,
            noise_rates, or noise_kinds.
    """
    kk = _checked_k(k)
    if not callable(getattr(scorer, "fuse", None)) or not callable(
        getattr(scorer, "context_score", None)
    ):
        raise TypeError("scorer must provide fuse() and context_score()")
    rates = [float(p) for p in noise_rates]
    if not rates:
        raise ValueError("noise_rates must be non-empty")
    for p in rates:
        if not np.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("noise_rates must be finite and in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    kinds = tuple(str(kd) for kd in noise_kinds)
    if not kinds or any(kd not in NOISE_KINDS for kd in kinds):
        raise ValueError(f"noise_kinds must be a non-empty subset of {NOISE_KINDS}")

    evidence, _ = _prepare(items, near_tie_sets, query_ids)
    face_ranked = [_rank_personas(ev, None, "face") for ev in evidence]
    face_block = _method_block(evidence, face_ranked, kk, include_per_set=False)

    pool = _validate_items(items)
    donors_by_other: dict[str, tuple[_Item, ...]] = {}
    for ev in evidence:
        if ev.query.persona_id not in donors_by_other:
            donors_by_other[ev.query.persona_id] = tuple(
                it for it in pool if it.persona_id != ev.query.persona_id
            )

    rows: list[dict[str, Any]] = []
    first_negative: dict[str, Any] = {"overall": None, "near_tie": None}
    for p in rates:
        rng = np.random.default_rng([int(seed), round(p * 1_000_000)])
        overrides: dict[str, ContextMetadata] = {}
        for ev in evidence:  # ascending item_id order
            if rng.random() < p:
                kind = kinds[int(rng.integers(0, len(kinds)))]
                donors = donors_by_other[ev.query.persona_id]
                donor = donors[int(rng.integers(0, len(donors)))]
                overrides[ev.query.item_id] = _corrupt_context(ev.query.context, kind, donor)
        fused_ranked = _score_fused(evidence, scorer, overrides)
        fused_block = _method_block(evidence, fused_ranked, kk, include_per_set=False)
        delta = _delta_block(fused_block, face_block)
        for scope in ("overall", "near_tie"):
            d = delta[scope]["hit_at_1"]
            if first_negative[scope] is None and d is not None and d < 0.0:
                first_negative[scope] = float(p)
        rows.append(
            {
                "noise_rate": float(p),
                "n_corrupted": len(overrides),
                "fused": fused_block,
                "delta": delta,
            }
        )
    return {
        "seed": int(seed),
        "noise_kinds": list(kinds),
        "face_only": face_block,
        "rows": rows,
        "first_negative_delta_rate": first_negative,
    }


# --------------------------------------------------------------------- weight sweep


def run_context_weight_sweep(
    items: Sequence[RankingItem],
    scorer_cls_or_factory: Callable[..., FusionScorer] | type,
    weights: Sequence[tuple[float, float]],
    near_tie_sets: Sequence[NearTieSet] = (),
    *,
    k: int = 5,
    query_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Sweep fusion (alpha, beta) weights; report fused ranking where it peaks.

    For each ``(alpha, beta)`` pair a scorer is built and the fused ranking is
    evaluated with the same protocol as :func:`run_ambiguity_ablation`. The
    peak is the first row maximizing hit@1 (ties broken by MRR) on the
    near-tie scope, falling back to the overall scope when no near-tie query
    is scored. Small and intentionally optional.

    Args:
        items: RankingItem pool.
        scorer_cls_or_factory: a scorer class (instantiated with a
            ThresholdProfile carrying (alpha, beta)) or a factory
            ``factory(alpha, beta) -> scorer``.
        weights: non-empty sequence of (alpha, beta) pairs (finite, >= 0, not
            both zero).
        near_tie_sets: measured NearTieSet values (may be empty).
        k: hit@k cutoff.
        query_ids: optional query subset.

    Returns:
        JSON-friendly dict with ``"rows"`` (per weight pair: ``"alpha"``,
        ``"beta"``, ``"beta_alpha_ratio"``, ``"overall"``, ``"near_tie"``),
        ``"peak"`` (the winning row or None), and ``"peak_scope"``.

    Raises:
        ValueError / TypeError: invalid items, sets, query ids, k, or weights.
    """
    kk = _checked_k(k)
    pairs = [(float(a), float(b)) for a, b in weights]
    if not pairs:
        raise ValueError("weights must be non-empty")
    for a, b in pairs:
        if not (np.isfinite(a) and np.isfinite(b)) or a < 0.0 or b < 0.0 or (a == 0.0 and b == 0.0):
            raise ValueError("each (alpha, beta) must be finite, >= 0, not both zero")

    def _build(alpha: float, beta: float) -> FusionScorer:
        if isinstance(scorer_cls_or_factory, type):
            scorer = scorer_cls_or_factory(
                ThresholdProfile("weight_sweep", 0.5, 0.65, 0.75, alpha, beta)
            )
        elif callable(scorer_cls_or_factory):
            scorer = scorer_cls_or_factory(alpha, beta)
        else:
            raise TypeError("scorer_cls_or_factory must be a scorer class or factory")
        if not callable(getattr(scorer, "fuse", None)) or not callable(
            getattr(scorer, "context_score", None)
        ):
            raise TypeError("scorer must provide fuse() and context_score()")
        return scorer

    evidence, _ = _prepare(items, near_tie_sets, query_ids)
    truths = [ev.query.persona_id for ev in evidence]
    rows: list[dict[str, Any]] = []
    for alpha, beta in pairs:
        scorer = _build(alpha, beta)
        ranked = _score_fused(evidence, scorer, None)
        near_idx = [i for i, ev in enumerate(evidence) if ev.scope == "near_tie"]
        rows.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "beta_alpha_ratio": (float(beta / alpha) if alpha > 0.0 else None),
                "overall": _aggregate(ranked, truths, kk),
                "near_tie": _aggregate(
                    [ranked[i] for i in near_idx], [truths[i] for i in near_idx], kk
                ),
            }
        )

    scope = "near_tie" if any(r["near_tie"]["n_queries"] > 0 for r in rows) else "overall"
    scored = [r for r in rows if r[scope]["hit_at_1"] is not None]
    peak = max(scored, key=lambda r: (r[scope]["hit_at_1"], r[scope]["mrr"])) if scored else None
    return {"rows": rows, "peak": peak, "peak_scope": scope}
