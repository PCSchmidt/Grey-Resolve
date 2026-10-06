"""Tests for the FMR/FNMR/ROC verification metrics core."""

from __future__ import annotations

import numpy as np
import pytest

from grey_resolve.evaluation import (
    build_pairs,
    eer,
    fmr_at_fnmr,
    fmr_fnmr_at_threshold,
    roc_curve,
)


def _separated(n_gen: int = 8, n_imp: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Hand-made score set with perfect genuine/impostor separation."""
    scores = np.concatenate([np.full(n_gen, 0.9), np.full(n_imp, 0.1)])
    is_genuine = np.concatenate([np.ones(n_gen, dtype=bool), np.zeros(n_imp, dtype=bool)])
    return scores, is_genuine


# ---------------------------------------------------------------- pair construction


def test_build_pairs_counts_for_tiny_labeled_set():
    # 5 samples with group sizes 2, 2, 1:
    # genuine = C(2,2) + C(2,2) + C(1,2) = 2; total C(5,2) = 10; impostor = 8.
    emb = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    labels = np.array([0, 0, 1, 1, 2])
    scores, is_genuine = build_pairs(emb, labels)
    assert scores.shape == (10,)
    assert is_genuine.shape == (10,)
    assert int(is_genuine.sum()) == 2
    assert int((~is_genuine).sum()) == 8


def test_build_pairs_deterministic_ordering_and_cosine():
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1])
    scores, is_genuine = build_pairs(emb, labels)
    # pairs in lexicographic (i<j) order: (0,1), (0,2), (1,2)
    assert np.all(is_genuine == np.array([True, False, False]))
    assert scores[0] == pytest.approx(1.0)  # identical vectors -> cosine 1
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(0.0)


def test_build_pairs_ignores_embedding_scale():
    emb = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    labels = np.array(["a", "a", "b"])
    scores, is_genuine = build_pairs(emb, labels)
    assert scores[0] == pytest.approx(1.0)  # cosine is scale-invariant
    assert bool(is_genuine[0]) is True


def test_build_pairs_rejects_bad_input():
    with pytest.raises(ValueError):
        build_pairs(np.zeros((1, 3)), np.array([0]))  # fewer than 2 samples
    with pytest.raises(ValueError):
        build_pairs(np.zeros((3, 3)), np.array([0, 1]))  # length mismatch
    with pytest.raises(ValueError):
        build_pairs(np.zeros(3), np.array([0, 1, 2]))  # not 2-D
    with pytest.raises(ValueError):
        build_pairs(np.zeros((2, 3)), np.array([0, 1]))  # zero vectors


# ------------------------------------------------- FMR/FNMR at a fixed threshold


def test_fmr_fnmr_hand_computed_at_threshold():
    scores = np.array([0.9, 0.8, 0.4, 0.3])
    is_genuine = np.array([True, False, True, False])
    fmr, fnmr = fmr_fnmr_at_threshold(scores, is_genuine, 0.5)
    # impostors {0.8, 0.3}: 1 of 2 >= 0.5 -> fmr = 0.5
    # genuine   {0.9, 0.4}: 1 of 2 <  0.5 -> fnmr = 0.5
    assert fmr == pytest.approx(0.5)
    assert fnmr == pytest.approx(0.5)


def test_fmr_fnmr_boundary_semantics():
    scores = np.array([0.9, 0.8, 0.4, 0.3])
    is_genuine = np.array([True, False, True, False])
    # score == threshold counts as a match (accepted)
    fmr, fnmr = fmr_fnmr_at_threshold(scores, is_genuine, 0.8)
    assert fmr == pytest.approx(0.5)  # impostor 0.8 is accepted at t=0.8
    assert fnmr == pytest.approx(0.5)  # genuine 0.4 is rejected
    fmr, fnmr = fmr_fnmr_at_threshold(scores, is_genuine, 0.9)
    assert fmr == pytest.approx(0.0)
    assert fnmr == pytest.approx(0.5)  # genuine 0.9 is not < 0.9


# ------------------------------------------------------------------------- ROC


def test_roc_perfect_separation_corners():
    scores, is_genuine = _separated()
    fmr, fnmr, thr = roc_curve(scores, is_genuine)
    assert fmr.shape == fnmr.shape == thr.shape
    assert fmr[0] == pytest.approx(1.0) and fnmr[0] == pytest.approx(0.0)
    assert fmr[-1] == pytest.approx(0.0) and fnmr[-1] == pytest.approx(1.0)
    # sorted thresholds, monotone rates
    assert np.all(np.diff(thr) >= 0)
    assert np.all(np.diff(fmr) <= 0)
    assert np.all(np.diff(fnmr) >= 0)
    # perfect separation: point with both rates zero exists
    assert np.any((fmr == 0.0) & (fnmr == 0.0))


def test_roc_hand_computed_on_small_set():
    scores = np.array([0.9, 0.2])
    is_genuine = np.array([True, False])
    fmr, fnmr, thr = roc_curve(scores, is_genuine)
    assert list(thr) == pytest.approx([0.2, 0.9, float(np.nextafter(0.9, np.inf))])
    assert fmr == pytest.approx([1.0, 0.0, 0.0])
    assert fnmr == pytest.approx([0.0, 0.0, 1.0])


def test_roc_custom_thresholds_sorted_and_validated():
    scores, is_genuine = _separated()
    fmr, fnmr, thr = roc_curve(scores, is_genuine, thresholds=[0.5, 0.05])
    assert list(thr) == pytest.approx([0.05, 0.5])
    assert fmr == pytest.approx([1.0, 0.0])
    assert fnmr == pytest.approx([0.0, 0.0])
    with pytest.raises(ValueError):
        roc_curve(scores, is_genuine, thresholds=[])
    with pytest.raises(ValueError):
        roc_curve(scores, is_genuine, thresholds=[0.5, np.nan])


def test_roc_all_same_scores_behaves_sane():
    scores = np.full(6, 0.42)
    is_genuine = np.array([True, True, True, False, False, False])
    fmr, fnmr, thr = roc_curve(scores, is_genuine)
    assert fmr[0] == pytest.approx(1.0) and fnmr[0] == pytest.approx(0.0)
    assert fmr[-1] == pytest.approx(0.0) and fnmr[-1] == pytest.approx(1.0)
    assert eer(scores, is_genuine) == pytest.approx(0.5)


# -------------------------------------------------------------------------- EER


def test_eer_perfect_separation_is_zero():
    scores, is_genuine = _separated()
    assert eer(scores, is_genuine) == pytest.approx(0.0)


def test_eer_fully_overlapping_is_half():
    scores = np.array([0.7, 0.7, 0.2, 0.2])
    is_genuine = np.array([True, False, True, False])
    assert eer(scores, is_genuine) == pytest.approx(0.5)


def test_eer_hand_computed_interpolation():
    # genuine {0.6, 0.4}, impostor {0.5, 0.3}: ROC diff hits 0 at t=0.5 -> EER 0.5
    scores = np.array([0.6, 0.4, 0.5, 0.3])
    is_genuine = np.array([True, True, False, False])
    assert eer(scores, is_genuine) == pytest.approx(0.5)


def test_eer_is_bounded_and_ordered_by_separation():
    scores, is_genuine = _separated()
    good = eer(scores, is_genuine)
    assert good == pytest.approx(0.0)
    noisy_scores = scores.copy()
    noisy_scores[0] = 0.05  # one genuine now scores below every impostor
    bad = eer(noisy_scores, is_genuine)
    assert bad == pytest.approx(0.125)  # hand-computed interpolation
    assert 0.0 < bad <= 0.5


# ------------------------------------------------------------- operating points


def test_fmr_at_fnmr_perfect_separation():
    scores, is_genuine = _separated()
    assert fmr_at_fnmr(scores, is_genuine, 0.0) == pytest.approx(0.0)


def test_fmr_at_fnmr_hand_computed():
    # genuine {0.6, 0.4}, impostor {0.5, 0.3} (same set as EER test)
    scores = np.array([0.6, 0.4, 0.5, 0.3])
    is_genuine = np.array([True, True, False, False])
    # strictest threshold with FNMR=0 is t=0.4 -> impostors >= 0.4: {0.5} -> FMR 0.5
    assert fmr_at_fnmr(scores, is_genuine, 0.0) == pytest.approx(0.5)
    # FNMR target 1.0 allows the loosest threshold -> FMR 0
    assert fmr_at_fnmr(scores, is_genuine, 1.0) == pytest.approx(0.0)


def test_fmr_at_fnmr_interpolates_between_points():
    # genuine {0.5, 0.3}, impostor {0.5, 0.4}: FNMR crosses 0.75 between the
    # (FMR=0.5, FNMR=0.5) and (FMR=0.0, FNMR=1.0) points -> interpolated FMR 0.25.
    scores = np.array([0.5, 0.3, 0.5, 0.4])
    is_genuine = np.array([True, True, False, False])
    assert fmr_at_fnmr(scores, is_genuine, 0.75) == pytest.approx(0.25)
    # exact plateau value -> best (lowest-FMR) point on the plateau
    assert fmr_at_fnmr(scores, is_genuine, 0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fmr_at_fnmr(scores, is_genuine, 1.5)
    with pytest.raises(ValueError):
        fmr_at_fnmr(scores, is_genuine, -0.1)


# ------------------------------------------------------------------ edge cases


def test_empty_and_degenerate_inputs_raise():
    with pytest.raises(ValueError):
        fmr_fnmr_at_threshold(np.array([]), np.array([], dtype=bool), 0.5)
    with pytest.raises(ValueError):
        roc_curve(np.array([]), np.array([], dtype=bool))
    with pytest.raises(ValueError):
        eer(np.array([]), np.array([], dtype=bool))
    with pytest.raises(ValueError):
        fmr_at_fnmr(np.array([]), np.array([], dtype=bool), 0.1)
    with pytest.raises(ValueError):
        eer(np.array([0.1]), np.array([True]))  # no impostor pairs
    with pytest.raises(ValueError):
        eer(np.array([0.1, 0.2]), np.array([False, False]))  # no genuine pairs
    with pytest.raises(ValueError):
        fmr_fnmr_at_threshold([0.1, 0.2], [True], 0.5)  # length mismatch
    with pytest.raises(ValueError):
        fmr_fnmr_at_threshold([0.1, np.nan], [True, False], 0.5)  # non-finite
    with pytest.raises(ValueError):
        fmr_fnmr_at_threshold([0.1, 0.2], [True, False], float("inf"))


def test_end_to_end_build_pairs_to_roc():
    # two identities with duplicated embeddings -> deterministic scores
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    scores, is_genuine = build_pairs(emb, labels)
    assert int(is_genuine.sum()) == 2  # (0,1) and (2,3)
    value = eer(scores, is_genuine)
    assert 0.0 <= value <= 0.5  # same-identity pairs score higher -> below chance
    fmr, fnmr = fmr_fnmr_at_threshold(scores, is_genuine, 0.5)
    assert fnmr == pytest.approx(0.0)  # both genuine pairs score > 0.5
