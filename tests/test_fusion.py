"""Tests for the linear fusion scorer (no model weights, pure math)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import numpy as np
import pytest

from grey_resolve.config import ThresholdProfile, load_threshold_profiles
from grey_resolve.fusion.base import FusionScorer
from grey_resolve.fusion.linear import (
    ENTITY_WEIGHT,
    GEO_WEIGHT,
    TIME_HALF_LIFE_DAYS,
    TIME_WEIGHT,
    LinearFusionScorer,
)
from grey_resolve.types import ContextMetadata

T0 = datetime(2024, 3, 14, 9, 20, 0, tzinfo=UTC)


def _strict() -> LinearFusionScorer:
    return LinearFusionScorer(ThresholdProfile("strict_surveillance", 0.60, 0.65, 0.75, 0.7, 0.3))


def _profile(alpha: float = 0.7, beta: float = 0.3, low: float = 0.65, high: float = 0.75) -> ThresholdProfile:
    return ThresholdProfile("custom", 0.5, low, high, alpha, beta)


# --- weights and context score ---------------------------------------------


def test_weights_sum_to_one_and_match_profile():
    assert GEO_WEIGHT + TIME_WEIGHT + ENTITY_WEIGHT == pytest.approx(1.0)
    profiles = load_threshold_profiles("configs/threshold_profiles.yaml")
    for name, profile in profiles.items():
        scorer = LinearFusionScorer(profile)
        assert scorer.profile.name == name
        assert scorer.alpha == profile.alpha
        assert scorer.beta == profile.beta


def test_protocol_conformance():
    assert isinstance(_strict(), FusionScorer)


def test_context_score_identical_context_is_one():
    ctx = ContextMetadata(
        timestamp=T0,
        geo_cluster="geo-alpha",
        source_platform="platform-a",
        text_entities=("alias-one", "geo-alpha"),
    )
    assert _strict().context_score(ctx, ctx) == pytest.approx(1.0)


def test_context_score_geo_only_semantics():
    scorer = _strict()
    g = ContextMetadata(geo_cluster="geo-alpha")
    other = ContextMetadata(geo_cluster="geo-beta")
    assert scorer.context_score(g, g) == pytest.approx(1.0)
    assert scorer.context_score(g, other) == pytest.approx(0.0)
    assert scorer.context_score(g, ContextMetadata()) == pytest.approx(0.0)


def test_context_score_temporal_decay():
    scorer = _strict()
    q = ContextMetadata(timestamp=T0)
    values = []
    for days in (0.0, 15.0, 30.0, 60.0, 120.0):
        c = ContextMetadata(timestamp=T0 + timedelta(days=days))
        values.append(scorer.context_score(q, c))
    assert values[0] == pytest.approx(1.0)
    assert values[2] == pytest.approx(0.5 ** (30.0 / TIME_HALF_LIFE_DAYS))
    assert values[3] == pytest.approx(0.25)
    assert all(a > b for a, b in pairwise(values))


def test_context_score_entity_jaccard():
    scorer = _strict()
    a = ContextMetadata(text_entities=("a", "b"))
    b = ContextMetadata(text_entities=("b", "c"))
    assert scorer.context_score(a, b) == pytest.approx(1.0 / 3.0)
    assert scorer.context_score(a, a) == pytest.approx(1.0)
    assert scorer.context_score(a, ContextMetadata(text_entities=("c", "d"))) == pytest.approx(0.0)
    assert scorer.context_score(a, ContextMetadata()) == pytest.approx(0.0)


def test_context_score_weighted_combination():
    scorer = _strict()
    q = ContextMetadata(timestamp=T0, geo_cluster="geo-alpha", text_entities=("a", "b"))
    c = ContextMetadata(
        timestamp=T0 + timedelta(days=30),
        geo_cluster="geo-alpha",
        text_entities=("a", "b"),
    )
    # geo 1.0 * 0.4 + temporal 0.5 * 0.3 + entities 1.0 * 0.3 = 0.85
    assert scorer.context_score(q, c) == pytest.approx(0.85)


def test_context_score_renormalizes_over_available_signals():
    scorer = _strict()
    q = ContextMetadata(geo_cluster="g", text_entities=("a",))
    c = ContextMetadata(geo_cluster="g", text_entities=("b",))
    # temporal excluded (missing): (0.4 * 1 + 0.3 * 0) / 0.7, not 0.4
    assert scorer.context_score(q, c) == pytest.approx(GEO_WEIGHT / (GEO_WEIGHT + ENTITY_WEIGHT))


def test_context_score_no_shared_signal_is_zero():
    scorer = _strict()
    empty = ContextMetadata()
    assert scorer.context_score(empty, empty) == pytest.approx(0.0)
    q = ContextMetadata(timestamp=T0)
    assert scorer.context_score(q, empty) == pytest.approx(0.0)


def test_context_score_symmetry_and_bounds():
    scorer = _strict()
    cases = [
        (ContextMetadata(), ContextMetadata(timestamp=T0, geo_cluster="g", text_entities=("a",))),
        (ContextMetadata(geo_cluster="g", timestamp=T0), ContextMetadata(geo_cluster="h", timestamp=T0 + timedelta(days=3))),
        (ContextMetadata(text_entities=("a", "b", "c")), ContextMetadata(text_entities=("b",))),
        (
            ContextMetadata(timestamp=T0, geo_cluster="g", text_entities=("a",)),
            ContextMetadata(timestamp=T0 + timedelta(days=45), geo_cluster="g", text_entities=("a", "z")),
        ),
    ]
    for q, c in cases:
        s1 = scorer.context_score(q, c)
        s2 = scorer.context_score(c, q)
        assert 0.0 <= s1 <= 1.0
        assert s1 == pytest.approx(s2)
    rng = np.random.default_rng(0)
    for _ in range(200):
        def rnd_ctx() -> ContextMetadata:
            ts = T0 + timedelta(days=float(rng.normal(0, 100))) if rng.random() < 0.7 else None
            geo = f"geo-{int(rng.integers(0, 4))}" if rng.random() < 0.7 else None
            ents = tuple(f"e{k}" for k in rng.integers(0, 5, size=int(rng.integers(0, 3))))
            return ContextMetadata(timestamp=ts, geo_cluster=geo, text_entities=ents)

        s = scorer.context_score(rnd_ctx(), rnd_ctx())
        assert 0.0 <= s <= 1.0


# --- fuse -------------------------------------------------------------------


def test_fuse_alpha_beta_honored():
    scorer = _strict()
    same = ContextMetadata(geo_cluster="g")
    empty = ContextMetadata()
    assert scorer.fuse(1.0, same, same) == pytest.approx(1.0)  # 0.7 * 1 + 0.3 * 1
    assert scorer.fuse(0.5, same, same) == pytest.approx(0.65)  # 0.7 * 0.5 + 0.3 * 1
    assert scorer.fuse(0.0, same, same) == pytest.approx(0.3)  # beta * 1
    assert scorer.fuse(1.0, empty, empty) == pytest.approx(0.7)  # alpha * 1
    assert scorer.fuse(0.0, empty, empty) == pytest.approx(0.0)


def test_fuse_from_yaml_profiles_matches_formula():
    for profile in load_threshold_profiles("configs/threshold_profiles.yaml").values():
        scorer = LinearFusionScorer(profile)
        ctx = ContextMetadata(geo_cluster="g")
        for face in (-1.0, -0.2, 0.0, 0.4, 0.75, 1.0):
            expected = min(1.0, max(0.0, profile.alpha * face + profile.beta * 1.0))
            assert scorer.fuse(face, ctx, ctx) == pytest.approx(expected)


def test_fuse_monotonic_in_face_score_and_clipped():
    scorer = _strict()
    q = ContextMetadata(timestamp=T0, geo_cluster="g", text_entities=("a",))
    c = ContextMetadata(timestamp=T0 + timedelta(days=10), geo_cluster="h", text_entities=("a", "b"))
    faces = np.linspace(-1.0, 1.0, 21)
    fused = [scorer.fuse(float(f), q, c) for f in faces]
    assert all(0.0 <= v <= 1.0 for v in fused)
    assert all(a <= b for a, b in pairwise(fused))


def test_fuse_clips_to_unit_interval():
    greedy = LinearFusionScorer(_profile(alpha=1.0, beta=1.0))
    ctx = ContextMetadata(geo_cluster="g")
    assert greedy.fuse(1.0, ctx, ctx) == pytest.approx(1.0)  # 2.0 clipped up
    assert greedy.fuse(-1.0, ctx, ctx) == pytest.approx(0.0)  # clipped down
    assert _strict().fuse(-1.0, ContextMetadata(), ContextMetadata()) == pytest.approx(0.0)


def test_fuse_rejects_non_finite_face_score():
    scorer = _strict()
    ctx = ContextMetadata()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            scorer.fuse(bad, ctx, ctx)


# --- ambiguous band ---------------------------------------------------------


def test_in_ambiguous_band_boundaries():
    scorer = _strict()
    assert scorer.in_ambiguous_band(0.65) is True
    assert scorer.in_ambiguous_band(0.75) is True
    assert scorer.in_ambiguous_band(0.70) is True
    assert scorer.in_ambiguous_band(0.649999) is False
    assert scorer.in_ambiguous_band(0.750001) is False
    assert scorer.in_ambiguous_band(0.2) is False
    assert scorer.in_ambiguous_band(0.95) is False
    broad = LinearFusionScorer(load_threshold_profiles("configs/threshold_profiles.yaml")["broad_lead"])
    assert broad.in_ambiguous_band(0.65) is True
    assert broad.in_ambiguous_band(0.55) is False


def test_in_ambiguous_band_rejects_non_finite():
    with pytest.raises(ValueError):
        _strict().in_ambiguous_band(float("nan"))


# --- construction validation ------------------------------------------------


def test_invalid_profile_raises():
    with pytest.raises(ValueError):
        LinearFusionScorer(_profile(alpha=-0.1))
    with pytest.raises(ValueError):
        LinearFusionScorer(_profile(beta=-0.1))
    with pytest.raises(ValueError):
        LinearFusionScorer(_profile(alpha=0.0, beta=0.0))
    with pytest.raises(ValueError):
        LinearFusionScorer(_profile(low=0.8, high=0.6))
    with pytest.raises(ValueError):
        LinearFusionScorer(_profile(low=-0.1, high=0.6))
    with pytest.raises(ValueError):
        LinearFusionScorer(_profile(low=0.1, high=1.2))
    with pytest.raises(ValueError):
        LinearFusionScorer(_profile(alpha=float("nan")))
