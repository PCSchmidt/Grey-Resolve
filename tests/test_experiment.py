"""Tests for the degraded-input experiment core (sweeps, gating ablation).

All tests use deterministic fake embedders over tiny synthetic numpy images --
no backbone weights, no real-person imagery, no file I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from grey_resolve.degradation.operators import DEGRADATIONS, gaussian_blur
from grey_resolve.evaluation.experiment import (
    CLEAN_CONDITION,
    CLEAN_NAME,
    FMR_AT_FNM_KEY,
    DegradationCondition,
    degrade_embeddings,
    gated_vs_ungated,
    probe_gallery_scores,
    run_condition_sweep,
    score_condition,
)

# --------------------------------------------------------------- synthetic fixtures


def _identity_image(ident: int, variant: int = 0, h: int = 16, w: int = 16) -> np.ndarray:
    """Flat image whose color one-hot encodes the identity (blur-invariant)."""
    img = np.full((h, w, 3), 10, dtype=np.uint8)
    img[..., ident] = min(255, 150 + 30 * variant)
    return img


def _color_embed(image: np.ndarray) -> np.ndarray:
    """Per-channel mean: same identity -> collinear vectors, distinct -> distinct."""
    return image.astype(np.float64).mean(axis=(0, 1))


def _identity_dataset(n_idents: int = 3, n_variants: int = 2):
    images, labels = [], []
    for ident in range(n_idents):
        for variant in range(n_variants):
            images.append(_identity_image(ident, variant))
            labels.append(f"id{ident}")
    return images, labels


def _stripe_image(freq: int, h: int = 32, w: int = 32) -> np.ndarray:
    """Vertical sine stripes at ``freq`` cycles across the width (sharp texture)."""
    x = np.arange(w)
    pattern = 127.0 + 70.0 * np.sin(2 * np.pi * freq * x / w)
    img = np.dstack([np.tile(pattern, (h, 1))] * 3)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _stripe_embed(image: np.ndarray) -> np.ndarray:
    """DC-removed per-column means: sensitive to blur, separates stripe frequencies."""
    gray = image.astype(np.float64).mean(axis=2)
    cols = gray.mean(axis=0)
    return cols - cols.mean()


def _stripe_dataset(freqs=(4, 6, 8)):
    images, labels = [], []
    for k, freq in enumerate(freqs):
        for _ in range(2):
            images.append(_stripe_image(freq))
            labels.append(f"stripe{freq}")
    return images, labels


# ----------------------------------------------------------------- sweep structure


def test_sweep_result_keys_for_each_condition():
    images, labels = _identity_dataset()
    conditions = [DegradationCondition(name, 0.5) for name in sorted(DEGRADATIONS)]
    results = run_condition_sweep(images, labels, conditions, _color_embed)
    assert [r["condition"] for r in results] == [CLEAN_NAME, *sorted(DEGRADATIONS)]
    for r in results:
        assert set(r) >= {"condition", "severity", "eer", FMR_AT_FNM_KEY, "roc_points"}
        assert isinstance(r["roc_points"], list) and len(r["roc_points"]) >= 2
        for pt in r["roc_points"]:
            assert set(pt) == {"threshold", "fmr", "fnmr"}
        assert 0.0 <= r["eer"] <= 1.0
        assert 0.0 <= r[FMR_AT_FNM_KEY] <= 1.0


def test_sweep_includes_clean_baseline_first():
    images, labels = _identity_dataset()
    results = run_condition_sweep(images, labels, [("gaussian_blur", 0.5)], _color_embed)
    assert results[0]["condition"] == CLEAN_NAME
    assert results[0]["severity"] == 0.0
    assert len(results) == 2


def test_sweep_clean_not_duplicated_when_passed_explicitly():
    images, labels = _identity_dataset()
    results = run_condition_sweep(
        images, labels, [CLEAN_CONDITION, ("brightness", 0.25)], _color_embed
    )
    assert [r["condition"] for r in results] == [CLEAN_NAME, "brightness"]


def test_sweep_can_skip_clean_baseline():
    images, labels = _identity_dataset()
    results = run_condition_sweep(
        images, labels, [("downsample", 0.5)], _color_embed, include_clean=False
    )
    assert [r["condition"] for r in results] == ["downsample"]


def test_sweep_no_conditions_raises():
    images, labels = _identity_dataset()
    with pytest.raises(ValueError):
        run_condition_sweep(images, labels, [], _color_embed, include_clean=False)


# --------------------------------------------------------------- metric correctness


def test_perfect_separation_eer_is_zero():
    images, labels = _identity_dataset()
    results = run_condition_sweep(images, labels, [], _color_embed)
    clean = results[0]
    assert clean["eer"] == pytest.approx(0.0, abs=1e-9)
    assert clean[FMR_AT_FNM_KEY] == pytest.approx(0.0, abs=1e-9)
    # ROC spans the full range
    assert clean["roc_points"][0]["fmr"] == pytest.approx(1.0)
    assert clean["roc_points"][-1]["fnmr"] == pytest.approx(1.0)


def test_pair_counts_are_probe_cross_gallery():
    images, labels = _identity_dataset(n_idents=3, n_variants=2)  # 3x2 -> 6 items
    scores, is_genuine = score_condition(images, labels, _color_embed, CLEAN_CONDITION)
    n = len(images)
    assert scores.shape == (n * n,)  # all probe-vs-gallery combinations
    # genuine = sum over identities of m_i^2 (incl. same-item pairs) = 3 * 4
    assert int(is_genuine.sum()) == 12
    assert int((~is_genuine).sum()) == n * n - 12


def test_probe_gallery_scores_order_and_genuineness():
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array(["a", "b"])
    scores, is_genuine = probe_gallery_scores(query, gallery, labels)
    # lexicographic (i, j): (0,0), (0,1), (1,0), (1,1)
    assert np.all(is_genuine == np.array([True, False, False, True]))
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


# --------------------------------------------------------- severity + degradation


def test_severity_ordering_affects_scores_blur_sensitive_embed():
    images, labels = _stripe_dataset()
    severities = [0.25, 0.5, 0.75]
    means = []
    for sev in severities:
        scores, is_genuine = score_condition(images, labels, _stripe_embed, ("gaussian_blur", sev))
        means.append(float(scores[is_genuine].mean()))
    assert means[0] > means[1] > means[2], f"blur must shrink genuine scores: {means}"


def test_severity_is_the_operator_parameter():
    images, labels = _stripe_dataset()
    emb = degrade_embeddings(images, labels, _stripe_embed, ("gaussian_blur", 1.5))
    expected = np.stack([_stripe_embed(gaussian_blur(im, 1.5)) for im in images])
    assert np.allclose(emb.query, expected)
    clean = np.stack([_stripe_embed(im) for im in images])
    assert np.allclose(emb.gallery, clean)


def test_degrade_embeddings_clean_condition_matches_gallery():
    images, labels = _identity_dataset()
    emb = degrade_embeddings(images, labels, _color_embed, CLEAN_CONDITION)
    assert np.allclose(emb.query, emb.gallery)
    assert np.array_equal(emb.labels, np.asarray(labels))
    # equal but not aliased
    emb.query[0, 0] = -1.0
    assert emb.gallery[0, 0] != -1.0


def test_degrade_embeddings_does_not_mutate_images():
    images, labels = _identity_dataset()
    originals = [im.copy() for im in images]
    degrade_embeddings(images, labels, _color_embed, ("off_angle", 30.0))
    for im, orig in zip(images, originals):
        assert np.array_equal(im, orig)


def test_run_condition_sweep_is_deterministic():
    images, labels = _stripe_dataset()
    conds = [("gaussian_blur", 0.5), ("brightness", 0.25)]
    r1 = run_condition_sweep(images, labels, conds, _stripe_embed)
    r2 = run_condition_sweep(images, labels, conds, _stripe_embed)
    assert r1 == r2


# ----------------------------------------------------------------- gating ablation


def test_gated_subset_is_smaller_or_equal():
    images, labels = _identity_dataset(n_idents=3, n_variants=2)  # 6 images
    # quality = channel-0 mean: id0 images (val 150/180) pass 100.0; others (val 10) fail
    quality_fn = lambda im: float(im[..., 0].mean())
    results = gated_vs_ungated(
        images, labels, [("gaussian_blur", 0.5)], _color_embed, quality_fn, 100.0
    )
    gated = results[0]["gated"]
    full = results[0]["full"]
    assert full["n"] == 6
    assert 0 < gated["n"] < full["n"]
    assert gated["pass_rate"] == pytest.approx(gated["n"] / full["n"])


def test_gated_open_gate_keeps_full_set():
    images, labels = _identity_dataset()
    results = gated_vs_ungated(
        images, labels, [], _color_embed, lambda im: 1.0, 0.5
    )
    gated, full = results[0]["gated"], results[0]["full"]
    assert gated["n"] == full["n"]
    assert gated["pass_rate"] == pytest.approx(1.0)
    assert gated[FMR_AT_FNM_KEY] == full[FMR_AT_FNM_KEY]


def test_gated_reports_fmr_fnmr_for_both_sets():
    images, labels = _identity_dataset()
    results = gated_vs_ungated(
        images, labels, [("gaussian_blur", 0.5)], _color_embed,
        lambda im: float(im.mean()), 0.0, threshold=0.4,
    )
    r = results[0]
    assert r["quality_threshold"] == 0.0
    assert r["operating_threshold"] == 0.4
    for subset in (r["full"], r["gated"]):
        for key in ("n", "pass_rate", "fmr", "fnmr", "eer", FMR_AT_FNM_KEY):
            assert key in subset
        assert 0.0 <= subset["fmr"] <= 1.0
        assert 0.0 <= subset["fnmr"] <= 1.0


def test_gated_degenerate_subset_reports_none_metrics():
    images, labels = _identity_dataset()
    results = gated_vs_ungated(
        images, labels, [], _color_embed, lambda im: 0.0, 1.0  # nothing passes
    )
    gated = results[0]["gated"]
    assert gated["n"] == 0
    assert gated["pass_rate"] == 0.0
    assert gated["fmr"] is None and gated["fnmr"] is None and gated["eer"] is None
    assert results[0]["full"]["eer"] == pytest.approx(0.0, abs=1e-9)


def test_gated_severity_changes_pass_rate_with_quality_sensitive_to_blur():
    images, labels = _stripe_dataset()
    quality_fn = lambda im: float(im.astype(np.float64).var())  # blur kills variance
    results = gated_vs_ungated(
        images, labels,
        [("gaussian_blur", sev) for sev in (0.25, 1.5, 3.0)],
        _stripe_embed, quality_fn, 500.0, include_clean=False,
    )
    rates = [r["gated"]["pass_rate"] for r in results]
    assert rates == pytest.approx([1.0, 1.0 / 3.0, 0.0])


# ------------------------------------------------------------------------ purity


def test_pure_functions_write_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    images, labels = _stripe_dataset()
    run_condition_sweep(images, labels, [("gaussian_blur", 0.5)], _stripe_embed)
    gated_vs_ungated(images, labels, [], _stripe_embed, lambda im: 1.0, 0.5)
    assert list(tmp_path.iterdir()) == []


# -------------------------------------------------------------------- error edges


def test_empty_labels_raises():
    images, _ = _identity_dataset()
    with pytest.raises(ValueError):
        degrade_embeddings(images, [], _color_embed, CLEAN_CONDITION)
    with pytest.raises(ValueError):
        run_condition_sweep(images, [], [], _color_embed)


def test_mismatched_lengths_raises():
    images, labels = _identity_dataset()
    with pytest.raises(ValueError):
        degrade_embeddings(images, labels[:-1], _color_embed, CLEAN_CONDITION)
    with pytest.raises(ValueError):
        run_condition_sweep(images, labels[:-1], [], _color_embed)


def test_empty_images_raises():
    with pytest.raises(ValueError):
        degrade_embeddings([], [], _color_embed, CLEAN_CONDITION)


def test_single_identity_labels_raise():
    images = [_identity_image(0), _identity_image(0, 1)]
    with pytest.raises(ValueError):
        degrade_embeddings(images, ["a", "a"], _color_embed, CLEAN_CONDITION)


def test_bad_image_contract_raises():
    images, labels = _identity_dataset(n_idents=2, n_variants=1)
    bad = [images[0].astype(np.float32), images[1]]
    with pytest.raises(ValueError):
        degrade_embeddings(bad, labels, _color_embed, CLEAN_CONDITION)
    flat = [np.zeros((8, 8), dtype=np.uint8), images[1]]
    with pytest.raises(ValueError):
        degrade_embeddings(flat, labels, _color_embed, CLEAN_CONDITION)
    with pytest.raises(TypeError):
        degrade_embeddings(["not an image", images[1]], labels, _color_embed, CLEAN_CONDITION)


def test_unknown_condition_name_raises():
    images, labels = _identity_dataset()
    with pytest.raises(ValueError):
        DegradationCondition("motion_blur", 0.5)
    with pytest.raises(ValueError):
        degrade_embeddings(images, labels, _color_embed, ("motion_blur", 0.5))


def test_invalid_severity_raises():
    with pytest.raises(ValueError):
        DegradationCondition("gaussian_blur", float("nan"))
    with pytest.raises(ValueError):
        DegradationCondition("gaussian_blur", -0.5)
    with pytest.raises(ValueError):
        DegradationCondition("clean", 0.5)  # clean requires severity 0


def test_bad_embed_fn_output_raises():
    images, labels = _identity_dataset(n_idents=2, n_variants=1)
    with pytest.raises(ValueError):
        degrade_embeddings(images, labels, lambda im: np.zeros(4), CLEAN_CONDITION)
    with pytest.raises(ValueError):
        degrade_embeddings(images, labels, lambda im: np.array([1.0, float("nan")]), CLEAN_CONDITION)

    def ragged(im):
        return np.array([1.0, 2.0]) if im[0, 0, 0] == 10 else np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        degrade_embeddings(images, labels, ragged, CLEAN_CONDITION)


def test_gated_bad_quality_values_raise():
    images, labels = _identity_dataset()
    with pytest.raises(ValueError):
        gated_vs_ungated(images, labels, [], _color_embed, lambda im: float("nan"), 0.5)
    with pytest.raises(ValueError):
        gated_vs_ungated(images, labels, [], _color_embed, lambda im: 1.0, float("inf"))
