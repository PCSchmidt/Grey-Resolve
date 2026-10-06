"""Tests for the Phase 2 fusion ablation core (pure math, fixed embeddings).

Hand-computed small cases with fake numpy embeddings and LinearFusionScorer --
no weights, no network. Covers ranking metrics, the face-only vs fused ablation
(including the honest negative case), the context-noise sweep (negative region
reported), the optional weight sweep, and the ValueError edges.
"""

from __future__ import annotations

import numpy as np
import pytest

from grey_resolve.config import ThresholdProfile
from grey_resolve.evaluation.fusion_experiment import (
    NOISE_KINDS,
    RankingItem,
    hit_at_1,
    hit_at_k,
    mrr,
    run_ambiguity_ablation,
    run_context_noise_sweep,
    run_context_weight_sweep,
)
from grey_resolve.fusion.linear import LinearFusionScorer
from grey_resolve.scenario.generator import NearTieSet
from grey_resolve.types import ContextMetadata

# --------------------------------------------------------------------------- helpers


def _scorer(alpha: float = 0.5, beta: float = 0.5) -> LinearFusionScorer:
    return LinearFusionScorer(ThresholdProfile("test", 0.5, 0.65, 0.75, alpha, beta))


def _emb(deg: float) -> np.ndarray:
    """Unit 2-vector at ``deg`` degrees (fixed, deterministic)."""
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)], dtype=np.float64)


def _ctx(geo: str, entity: str) -> ContextMetadata:
    return ContextMetadata(geo_cluster=geo, text_entities=(entity,))


ALPHA_CTX = _ctx("geo-alpha", "alias-one")
BETA_CTX = _ctx("geo-beta", "alias-two")


def _tie_set() -> NearTieSet:
    return NearTieSet(("p1", "p2"), (), ())


# --------------------------------------------------------------- ranking metrics


def test_hit_at_1_tiny_examples():
    assert hit_at_1(["a", "b"], "a") == 1.0
    assert hit_at_1(["a", "b"], "b") == 0.0
    assert hit_at_1(["b", "a"], "b") == 1.0
    assert hit_at_1([], "a") == 0.0
    assert hit_at_1(["b", "c"], "a") == 0.0


def test_mrr_tiny_examples():
    assert mrr(["a", "b", "c"], "a") == 1.0
    assert mrr(["a", "b", "c"], "b") == pytest.approx(0.5)
    assert mrr(["a", "b", "c"], "c") == pytest.approx(1.0 / 3.0)
    assert mrr([], "a") == 0.0
    assert mrr(["b"], "a") == 0.0


def test_hit_at_k_tiny_examples():
    assert hit_at_k(["a", "b", "c"], "c", 3) == 1.0
    assert hit_at_k(["a", "b", "c"], "c", 2) == 0.0
    assert hit_at_k(["a", "b", "c"], "b", 2) == 1.0
    assert hit_at_k(["a", "b", "c"], "a", 1) == hit_at_1(["a", "b", "c"], "a")
    assert hit_at_k([], "a", 5) == 0.0


def test_hit_at_k_invalid_cutoff():
    with pytest.raises(ValueError):
        hit_at_k(["a"], "a", 0)
    with pytest.raises(ValueError):
        hit_at_k(["a"], "a", -1)
    with pytest.raises(ValueError):
        hit_at_k(["a"], "a", 1.5)
    with pytest.raises(ValueError):
        hit_at_k(["a"], "a", True)


def test_ranking_metrics_duplicate_ids_use_first_rank():
    assert hit_at_1(["b", "a", "a"], "a") == 0.0
    assert mrr(["b", "a", "a"], "a") == pytest.approx(0.5)


# ------------------------------------------------------- ablation: informative case


def _tie_items() -> list[RankingItem]:
    # Face scores tie at 1.0 between the true persona p2 (via g@0 deg) and the
    # distractor p1 (d1@0 deg); the tie-break ranks p1 first, so face-only
    # fails on qA. Context is perfectly informative (p2 items carry ALPHA_CTX).
    return [
        RankingItem("qA", "p2", _emb(0), ALPHA_CTX),
        RankingItem("qB", "p2", _emb(10), ALPHA_CTX),
        RankingItem("g", "p2", _emb(0), ALPHA_CTX),
        RankingItem("g2", "p2", _emb(12), ALPHA_CTX),
        RankingItem("d1", "p1", _emb(0), BETA_CTX),
    ]


def test_ablation_informative_context_beats_face_only():
    res = run_ambiguity_ablation(
        _tie_items(), _scorer(), [_tie_set()], k=2, query_ids=["qA", "qB"]
    )
    face = res["face_only"]["near_tie"]
    fused = res["fused"]["near_tie"]
    # qA: face tie lost on the persona-id tie-break -> hit@1 0; qB: face edge -> 1.
    assert face["hit_at_1"] == pytest.approx(0.5)
    assert face["mrr"] == pytest.approx(0.75)
    assert face["hit_at_k"] == pytest.approx(1.0)
    assert face["n_queries"] == 2
    # Fused: context (geo + entities) breaks both ties towards p2.
    assert fused["hit_at_1"] == pytest.approx(1.0)
    assert fused["mrr"] == pytest.approx(1.0)
    assert fused["hit_at_1"] > face["hit_at_1"]
    assert res["k"] == 2
    assert res["n_query_items"] == 2
    assert res["n_skipped"] == 0
    # Per-set detail agrees with the near-tie scope.
    per_set = res["fused"]["per_set"]["p1|p2"]
    assert per_set["hit_at_1"] == pytest.approx(1.0)
    assert per_set["n_queries"] == 2
    # Overall scope holds the same two queries here.
    assert res["fused"]["overall"]["n_queries"] == 2


def test_ablation_face_score_gap_summary():
    res = run_ambiguity_ablation(
        _tie_items(), _scorer(), [_tie_set()], k=2, query_ids=["qA", "qB"]
    )
    gap = res["face_score_gap"]["near_tie"]
    # qA gap: 1.0 - 1.0 = 0.0; qB gap: cos(2 deg) - cos(10 deg) = 0.014583...
    assert gap["n"] == 2
    assert gap["mean"] == pytest.approx((0.0 + (np.cos(np.deg2rad(2)) - np.cos(np.deg2rad(10)))) / 2)
    assert gap["median"] == pytest.approx(gap["mean"])


# ---------------------------------------------------------- ablation: negative case


def test_ablation_anti_informative_context_is_worse():
    # Face slightly favors the true persona (cos 10 deg vs 15 deg), but the
    # query context matches the distractor exactly: fused ranking flips.
    items = [
        RankingItem("q", "p2", _emb(0), BETA_CTX),
        RankingItem("a2", "p2", _emb(10), ALPHA_CTX),
        RankingItem("d1", "p1", _emb(15), BETA_CTX),
    ]
    res = run_ambiguity_ablation(items, _scorer(), [_tie_set()], k=2, query_ids=["q"])
    face = res["face_only"]["near_tie"]
    fused = res["fused"]["near_tie"]
    assert face["hit_at_1"] == pytest.approx(1.0)
    assert face["mrr"] == pytest.approx(1.0)
    assert fused["hit_at_1"] == pytest.approx(0.0)
    assert fused["mrr"] == pytest.approx(0.5)
    assert fused["hit_at_1"] < face["hit_at_1"]  # the honest negative region
    gap = res["face_score_gap"]["near_tie"]
    assert gap["mean"] == pytest.approx(np.cos(np.deg2rad(10)) - np.cos(np.deg2rad(15)))


# ---------------------------------------------------------------- scopes and skips


def test_ablation_fallback_scope_without_near_tie_set():
    items = [
        RankingItem("q3", "p3", _emb(0), ALPHA_CTX),
        RankingItem("g3", "p3", _emb(1), ALPHA_CTX),
        RankingItem("qA", "p2", _emb(0), ALPHA_CTX),
        RankingItem("g", "p2", _emb(0), ALPHA_CTX),
        RankingItem("d1", "p1", _emb(0), BETA_CTX),
    ]
    res = run_ambiguity_ablation(
        items, _scorer(), [_tie_set()], k=2, query_ids=["q3", "qA"]
    )
    face = res["face_only"]
    assert face["overall"]["n_queries"] == 2  # both queries, each with its own candidate set
    assert face["near_tie"]["n_queries"] == 1  # only qA's candidate set is a near-tie set
    assert "p1|p2" in face["per_set"]
    assert face["per_set"]["p1|p2"]["n_queries"] == 1
    # q3 falls back to all personas (p1, p2, p3) -> 3-candidate ranking.
    assert res["fused"]["overall"]["n_queries"] == 2


def test_ablation_empty_near_tie_sets_fall_back_to_all_personas():
    items = [
        RankingItem("q", "p2", _emb(0), ALPHA_CTX),
        RankingItem("g", "p2", _emb(0), ALPHA_CTX),
        RankingItem("d1", "p1", _emb(0), BETA_CTX),
    ]
    res = run_ambiguity_ablation(items, _scorer(), [], k=2, query_ids=["q"])
    assert res["face_only"]["near_tie"]["n_queries"] == 0
    assert res["face_only"]["near_tie"]["hit_at_1"] is None
    assert res["fused"]["overall"]["n_queries"] == 1


def test_ablation_skips_query_without_own_gallery_evidence():
    items = [
        RankingItem("q", "p1", _emb(0), ALPHA_CTX),  # only p1 item -> no counterpart
        RankingItem("g", "p2", _emb(0), BETA_CTX),
        RankingItem("g2", "p2", _emb(1), BETA_CTX),
    ]
    res = run_ambiguity_ablation(items, _scorer(), [_tie_set()], k=2, query_ids=["q"])
    assert res["n_skipped"] == 1
    assert res["n_query_items"] == 0
    assert res["face_only"]["overall"]["hit_at_1"] is None


def test_ablation_is_deterministic_regardless_of_input_order():
    items = _tie_items()
    res1 = run_ambiguity_ablation(items, _scorer(), [_tie_set()], k=2, query_ids=["qA", "qB"])
    res2 = run_ambiguity_ablation(list(reversed(items)), _scorer(), [_tie_set()], k=2, query_ids=["qB", "qA"])
    assert res1 == res2


# -------------------------------------------------------------------- noise sweep


def test_noise_sweep_reports_negative_region():
    # kinds=("replace",) always misleads: corrupted queries flip to the
    # distractor, so deltas degrade from +0.5 (rate 0) to -0.5 (rate 1).
    res = run_context_noise_sweep(
        _tie_items(),
        _scorer(),
        [_tie_set()],
        [0.0, 0.5, 1.0],
        seed=7,
        k=2,
        query_ids=["qA", "qB"],
        noise_kinds=("replace",),
    )
    rows = res["rows"]
    assert [r["noise_rate"] for r in rows] == [0.0, 0.5, 1.0]
    face = res["face_only"]["near_tie"]
    assert face["hit_at_1"] == pytest.approx(0.5)
    assert face["mrr"] == pytest.approx(0.75)
    d0, d1, d2 = (r["delta"]["near_tie"] for r in rows)
    # Hand-computed: no corruption at rate 0 -> fused fixes both queries.
    assert d0["hit_at_1"] == pytest.approx(0.5)
    assert d0["mrr"] == pytest.approx(0.25)
    # All corrupted at rate 1 -> fused flips both queries.
    assert d2["hit_at_1"] == pytest.approx(-0.5)
    assert d2["mrr"] == pytest.approx(-0.25)
    # Monotone-ish degradation; the negative region is reported, not suppressed.
    assert d0["hit_at_1"] >= d1["hit_at_1"] >= d2["hit_at_1"]
    assert d1["hit_at_1"] in (-0.5, 0.0, 0.5)
    assert rows[-1]["n_corrupted"] == 2
    assert rows[0]["n_corrupted"] == 0
    # first_negative_delta_rate is the first row with a negative hit@1 delta.
    first = res["first_negative_delta_rate"]["near_tie"]
    assert first is not None
    neg = [r["noise_rate"] for r in rows if r["delta"]["near_tie"]["hit_at_1"] < 0.0]
    assert first == neg[0]
    assert all(r["delta"]["near_tie"]["hit_at_1"] >= 0.0 for r in rows if r["noise_rate"] < first)


def test_noise_sweep_default_kinds_and_determinism():
    kwargs = {"k": 2, "query_ids": ["qA", "qB"]}
    a = run_context_noise_sweep(_tie_items(), _scorer(), [_tie_set()], [0.3, 0.8], seed=11, **kwargs)
    b = run_context_noise_sweep(_tie_items(), _scorer(), [_tie_set()], [0.3, 0.8], seed=11, **kwargs)
    assert a == b  # identical inputs -> identical outputs
    assert a["noise_kinds"] == list(NOISE_KINDS)
    # Deltas stay within [-1, 1] and are reported for every rate.
    for row in a["rows"]:
        d = row["delta"]["near_tie"]["hit_at_1"]
        assert -1.0 <= d <= 1.0
        assert row["fused"]["near_tie"]["n_queries"] == 2


def test_noise_sweep_validation():
    items = _tie_items()
    with pytest.raises(ValueError):
        run_context_noise_sweep(items, _scorer(), [_tie_set()], [], seed=1, query_ids=["qA"])
    with pytest.raises(ValueError):
        run_context_noise_sweep(items, _scorer(), [_tie_set()], [1.5], seed=1, query_ids=["qA"])
    with pytest.raises(ValueError):
        run_context_noise_sweep(items, _scorer(), [_tie_set()], [-0.1], seed=1, query_ids=["qA"])
    with pytest.raises(ValueError):
        run_context_noise_sweep(items, _scorer(), [_tie_set()], [0.5], seed=-1, query_ids=["qA"])
    with pytest.raises(ValueError):
        run_context_noise_sweep(
            items, _scorer(), [_tie_set()], [0.5], seed=1, query_ids=["qA"], noise_kinds=()
        )
    with pytest.raises(ValueError):
        run_context_noise_sweep(
            items, _scorer(), [_tie_set()], [0.5], seed=1, query_ids=["qA"], noise_kinds=("bogus",)
        )


# ------------------------------------------------------------------- weight sweep


def test_weight_sweep_peaks_where_context_helps():
    weights = [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5)]
    res = run_context_weight_sweep(
        _tie_items(), LinearFusionScorer, weights, [_tie_set()], k=2, query_ids=["qA", "qB"]
    )
    rows = res["rows"]
    assert [(r["alpha"], r["beta"]) for r in rows] == weights
    # beta = 0 is face-only ranking: qA stays broken; context weights fix it.
    assert rows[0]["near_tie"]["hit_at_1"] == pytest.approx(0.5)
    assert rows[2]["near_tie"]["hit_at_1"] == pytest.approx(1.0)
    assert rows[1]["beta_alpha_ratio"] == pytest.approx(0.3 / 0.7)
    assert res["peak_scope"] == "near_tie"
    # Both context-weighted rows fix qA; the first maximum in row order wins.
    assert (res["peak"]["alpha"], res["peak"]["beta"]) == (0.7, 0.3)


def test_weight_sweep_accepts_factory():
    def factory(alpha: float, beta: float) -> LinearFusionScorer:
        return LinearFusionScorer(ThresholdProfile("w", 0.5, 0.65, 0.75, alpha, beta))

    res = run_context_weight_sweep(
        _tie_items(), factory, [(1.0, 0.0), (0.5, 0.5)], [_tie_set()], k=2, query_ids=["qA", "qB"]
    )
    assert res["peak"]["near_tie"]["hit_at_1"] == pytest.approx(1.0)


def test_weight_sweep_validation():
    items = _tie_items()
    with pytest.raises(ValueError):
        run_context_weight_sweep(items, LinearFusionScorer, [], [_tie_set()], query_ids=["qA"])
    with pytest.raises(ValueError):
        run_context_weight_sweep(items, LinearFusionScorer, [(0.0, 0.0)], [_tie_set()], query_ids=["qA"])
    with pytest.raises(ValueError):
        run_context_weight_sweep(items, LinearFusionScorer, [(1.0, -0.1)], [_tie_set()], query_ids=["qA"])
    with pytest.raises(TypeError):
        run_context_weight_sweep(items, object(), [(1.0, 0.5)], [_tie_set()], query_ids=["qA"])


# ------------------------------------------------------------------ error edges


def test_error_empty_items():
    with pytest.raises(ValueError):
        run_ambiguity_ablation([], _scorer(), [_tie_set()])


def test_error_persona_not_in_set():
    items = _tie_items()
    bad = NearTieSet(("p1", "ghost"), (), ())
    with pytest.raises(ValueError, match="ghost"):
        run_ambiguity_ablation(items, _scorer(), [bad])
    with pytest.raises(ValueError, match="ghost"):
        run_context_noise_sweep(items, _scorer(), [bad], [0.5], seed=1)
    single = NearTieSet(("p1",), (), ())
    with pytest.raises(ValueError):
        run_ambiguity_ablation(items, _scorer(), [single])


def test_error_mismatched_ids():
    items = _tie_items()
    with pytest.raises(ValueError, match="unknown item ids"):
        run_ambiguity_ablation(items, _scorer(), [_tie_set()], query_ids=["qA", "nope"])
    with pytest.raises(ValueError, match="duplicates"):
        run_ambiguity_ablation(items, _scorer(), [_tie_set()], query_ids=["qA", "qA"])
    with pytest.raises(ValueError):
        run_ambiguity_ablation(items, _scorer(), [_tie_set()], query_ids=[])
    dup = [
        RankingItem("x", "p2", _emb(0), ALPHA_CTX),
        RankingItem("x", "p1", _emb(1), BETA_CTX),
    ]
    with pytest.raises(ValueError, match="not unique"):
        run_ambiguity_ablation(dup, _scorer(), [])


def test_error_bad_embeddings():
    ok = _tie_items()
    zero = [RankingItem("z", "p3", np.zeros(2), ALPHA_CTX)] + ok
    with pytest.raises(ValueError, match="zero"):
        run_ambiguity_ablation(zero, _scorer(), [_tie_set()])
    mixed = [RankingItem("m", "p3", np.ones(3), ALPHA_CTX)] + ok
    with pytest.raises(ValueError, match="dimension"):
        run_ambiguity_ablation(mixed, _scorer(), [_tie_set()])
    nan = [RankingItem("n", "p3", np.array([np.nan, 1.0]), ALPHA_CTX)] + ok
    with pytest.raises(ValueError, match="finite"):
        run_ambiguity_ablation(nan, _scorer(), [_tie_set()])
    one_persona = [
        RankingItem("a", "p", _emb(0), ALPHA_CTX),
        RankingItem("b", "p", _emb(1), ALPHA_CTX),
    ]
    with pytest.raises(ValueError, match="2 personas"):
        run_ambiguity_ablation(one_persona, _scorer(), [])


def test_error_bad_k_and_scorer():
    items = _tie_items()
    with pytest.raises(ValueError, match="k must be"):
        run_ambiguity_ablation(items, _scorer(), [_tie_set()], k=0)
    with pytest.raises(TypeError):
        run_ambiguity_ablation(items, object(), [_tie_set()])
    not_an_item = [object()]
    with pytest.raises(TypeError):
        run_ambiguity_ablation(not_an_item, _scorer(), [])
