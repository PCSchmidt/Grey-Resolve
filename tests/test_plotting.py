"""Tests for the portfolio figure pipeline (plotting + make_figures).

Deterministic hand-made results dicts mirroring the real ``results.json``
schemas (``benchmarks/evaluate_roc.py`` / ``benchmarks/evaluate_fusion.py``);
matplotlib runs on the Agg backend. No network, no real run dependency.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grey_resolve.plotting import (
    FMR_AT_FNM_LABEL,
    FULL_LABEL,
    GATED_LABEL,
    PROTOTYPE_NOTE,
    eer_bar_figure,
    fusion_methods_figure,
    gate_ablation_figure,
    roc_figure,
)

_MAKE_FIGURES_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "make_figures.py"
_spec = importlib.util.spec_from_file_location("make_figures", _MAKE_FIGURES_PATH)
make_figures = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("make_figures", make_figures)
_spec.loader.exec_module(make_figures)

RUN_ID = "20261006T191330Z"


# --------------------------------------------------------------- tiny real-schema fixtures


def _roc_points() -> list[dict[str, float]]:
    """Three-point error trade-off curve (real schema: threshold/fmr/fnmr)."""
    return [
        {"threshold": 0.3, "fmr": 0.10, "fnmr": 0.02},
        {"threshold": 0.5, "fmr": 0.01, "fnmr": 0.05},
        {"threshold": 0.7, "fmr": 0.0, "fnmr": 0.20},
    ]


def _tiny_sweep() -> list[dict]:
    """Two-condition sweep block mirroring evaluate_roc.py's results.json."""
    return [
        {
            "condition": "clean",
            "severity": 0.0,
            "eer": 0.0,
            "fmr_at_fnmr_0.01": 0.0,
            "roc_points": _roc_points(),
            "n_pairs": 10,
            "n_genuine": 3,
            "n_impostor": 7,
            "n_images": 4,
            "n_gallery": 3,
            "n_scored": 3,
            "n_query_dropped": 0,
            "n_gallery_dropped": 0,
        },
        {
            "condition": "downsample",
            "severity": 0.1,
            "eer": 0.046,
            "fmr_at_fnmr_0.01": 0.873,
            "roc_points": _roc_points(),
            "n_pairs": 10,
            "n_genuine": 3,
            "n_impostor": 7,
            "n_images": 4,
            "n_gallery": 3,
            "n_scored": 3,
            "n_query_dropped": 1,
            "n_gallery_dropped": 0,
        },
    ]


def _tiny_gated() -> list[dict]:
    """Gate-ablation block mirroring evaluate_roc.py's gated_vs_ungated."""
    return [
        {
            "condition": "clean",
            "severity": 0.0,
            "quality_threshold": 0.5,
            "operating_threshold": 0.5,
            "n_dropped": 1,
            "full": {"n": 3, "pass_rate": 1.0, "fmr": 0.0, "fnmr": 0.028},
            "gated": {"n": 2, "pass_rate": 0.94, "fmr": 0.0, "fnmr": 0.020},
        },
        {
            "condition": "brightness",
            "severity": 0.75,
            "quality_threshold": 0.5,
            "operating_threshold": 0.5,
            "n_dropped": 1,
            "full": {"n": 3, "pass_rate": 1.0, "fmr": 0.0, "fnmr": 0.30},
            # extreme condition: too few quality-passing probes -> unreported
            "gated": {"n": 0, "pass_rate": 0.0, "fmr": None, "fnmr": None},
        },
    ]


def _tiny_ablation() -> dict:
    """Ablation block mirroring evaluate_fusion.py's results.json (all methods)."""
    def _metrics(hit: float, n: int) -> dict:
        return {"n_queries": n, "hit_at_1": hit, "hit_at_k": hit, "mrr": hit}

    return {
        "k": 5,
        "n_query_items": 30,
        "n_skipped": 0,
        "selective_gap": 0.1,
        "face_only": {
            "overall": _metrics(0.906, 30),
            "near_tie": _metrics(0.63, 11),
            "per_set": {},
        },
        "fused": {"overall": _metrics(0.774, 30), "near_tie": _metrics(0.41, 11), "per_set": {}},
        "selective": {"overall": _metrics(0.849, 30), "near_tie": _metrics(0.55, 11), "per_set": {}},
        "tiebreak": {"overall": _metrics(0.906, 30), "near_tie": _metrics(0.57, 11), "per_set": {}},
        "gap_le": {
            "0.05": {
                "gap_max": 0.05,
                "n_queries": 18,
                "face_only": _metrics(0.5, 18),
                "fused": _metrics(0.167, 18),
                "tiebreak": _metrics(0.444, 18),
                "delta": {"hit_at_1": -0.333, "hit_at_k": -0.166, "mrr": -0.26},
            },
            "0.1": {
                "gap_max": 0.1,
                "n_queries": 27,
                "face_only": _metrics(0.63, 27),
                "fused": _metrics(0.407, 27),
                "tiebreak": _metrics(0.63, 27),
                "delta": {"hit_at_1": -0.222, "hit_at_k": -0.1, "mrr": -0.15},
            },
        },
        "face_score_gap": {"overall": {"n": 30, "mean": 0.17, "median": 0.16}},
    }


def _tiny_results() -> dict:
    """Full ROC-run results document including the path-holding config."""
    return {
        "config": {
            "data_root": "C:\\Users\\me\\gallery",
            "limit": 2,
            "max_identities": 100,
            "det_model": "C:\\Users\\me\\.insightface\\models\\det_10g.onnx",
            "rec_model": "C:\\Users\\me\\.insightface\\models\\w600k_r50.onnx",
            "run_stamp": RUN_ID,
            "created_utc": "2026-10-06T19:13:30+00:00",
        },
        "sweep": _tiny_sweep(),
        "gated_vs_ungated": _tiny_gated(),
    }


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------- figure content


def test_roc_figure_axes_legend_and_title() -> None:
    fig = roc_figure(_tiny_sweep(), run_id=RUN_ID)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert "FMR" in ax.get_xlabel()
    assert "FNMR" in ax.get_ylabel()
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == ["clean", "downsample 0.1"]
    assert [line.get_label() for line in ax.get_lines()] == ["clean", "downsample 0.1"]
    title = ax.get_title()
    assert PROTOTYPE_NOTE in title
    assert RUN_ID in title


def test_eer_bar_figure_groups_eer_and_fmr_bars() -> None:
    fig = eer_bar_figure(_tiny_sweep(), run_id=RUN_ID)
    ax = fig.axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == ["EER", FMR_AT_FNM_LABEL]
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == ["clean", "downsample 0.1"]
    assert PROTOTYPE_NOTE in ax.get_title()
    assert RUN_ID in ax.get_title()


def test_gate_ablation_figure_full_vs_gated_with_na() -> None:
    fig = gate_ablation_figure(_tiny_gated(), run_id=RUN_ID)
    ax = fig.axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == [FULL_LABEL, GATED_LABEL]
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == ["clean", "brightness 0.75"]
    # the unreported gated FNMR (None) is drawn as an "n/a" marker, not a value
    text_contents = [t.get_text() for t in ax.texts]
    assert "n/a" in text_contents
    assert "operating threshold 0.5" in ax.get_ylabel()
    assert PROTOTYPE_NOTE in ax.get_title()
    assert RUN_ID in ax.get_title()


def test_fusion_methods_figure_scopes_and_methods() -> None:
    fig = fusion_methods_figure(_tiny_ablation(), run_id=RUN_ID)
    ax = fig.axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == [
        "face-only",
        "fused",
        "selective (gap-gated)",
        "tiebreak (top-2)",
    ]
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels[0] == "overall (n=30)"
    assert tick_labels[1] == "near-tie (n=11)"
    # gap slices tightest first
    assert tick_labels[2:] == ["gap <= 0.05 (n=18)", "gap <= 0.1 (n=27)"]
    assert "hit@1" in ax.get_ylabel()
    assert PROTOTYPE_NOTE in ax.get_title()
    assert RUN_ID in ax.get_title()


def test_fusion_methods_figure_partial_methods_no_gap_slices() -> None:
    """Early fusion runs recorded only face_only/fused and no gap_le slices."""
    ablation = {
        "k": 5,
        "face_only": {"overall": {"n_queries": 8, "hit_at_1": 1.0}},
        "fused": {"overall": {"n_queries": 8, "hit_at_1": 1.0}},
    }
    fig = fusion_methods_figure(ablation, run_id="20261006T225702Z")
    ax = fig.axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == ["face-only", "fused"]
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == ["overall (n=8)"]


def test_fusion_methods_figure_rejects_empty_ablation() -> None:
    with pytest.raises(ValueError):
        fusion_methods_figure({"k": 5}, run_id=RUN_ID)


# --------------------------------------------------------------- make_figures schema detection


def test_figures_for_results_ignores_non_roc_sweeps() -> None:
    """Latency-style runs reuse the "sweep" key but have no condition blocks."""
    latency_results = {
        "config": {"sizes": [100], "run_stamp": "20261007T142923Z"},
        "sweep": [{"kind": "faiss_hnsw", "index_size": 100, "build_ms": 248.5, "search_p50_ms": 0.04}],
        "pipeline": {"rows": []},
    }
    assert make_figures.figures_for_results(latency_results, "latency-20261007T142923Z") == []


# --------------------------------------------------------------- make_figures sanitizer


def test_sanitize_results_replaces_local_paths() -> None:
    results = _tiny_results()
    clean = make_figures.sanitize_results(results)
    assert clean["config"]["data_root"] == "gallery"
    assert clean["config"]["det_model"] == "det_10g.onnx"
    assert clean["config"]["rec_model"] == "w600k_r50.onnx"
    # input is never mutated
    assert results["config"]["data_root"] == "C:\\Users\\me\\gallery"
    # non-path config values survive unchanged
    assert clean["config"]["run_stamp"] == RUN_ID
    assert clean["config"]["max_identities"] == 100


def test_sanitize_results_keeps_aggregate_metrics_only() -> None:
    clean = make_figures.sanitize_results(_tiny_results())
    for entry in clean["sweep"]:
        assert "roc_points" not in entry
        assert entry["eer"] is not None
        assert entry["fmr_at_fnmr_0.01"] is not None
    assert len(clean["sweep"]) == 2
    assert clean["gated_vs_ungated"][0]["gated"]["fnmr"] == 0.020
    # raw sweep data in the input keeps its curve rows
    assert "roc_points" in _tiny_results()["sweep"][0]


def test_sanitize_results_scrubs_other_absolute_paths() -> None:
    results = {"config": {"scenario_config_path": "configs/scenario_v0.yaml",
                          "extra_root": "/tmp/scratch/cache"},
               "sweep": []}
    clean = make_figures.sanitize_results(results)
    assert clean["config"]["extra_root"] == "cache"
    assert clean["config"]["scenario_config_path"] == "configs/scenario_v0.yaml"


# --------------------------------------------------------------- end-to-end CLI


def test_make_figures_end_to_end(tmp_path: Path) -> None:
    runs_root = tmp_path / "out"
    fusion_run = runs_root / "20261007T125746Z"
    fusion_run.mkdir(parents=True)
    fusion_results = {
        "config": {
            "data_root": "/home/me/gallery",
            "det_model": "/home/me/models/det_10g.onnx",
            "rec_model": "/home/me/models/w600k_r50.onnx",
            "run_stamp": "20261007T125746Z",
        },
        "ablation": _tiny_ablation(),
        "context_noise_sweep": {"seed": 42, "rows": []},
    }
    (fusion_run / "results.json").write_text(json.dumps(fusion_results), encoding="utf-8")
    roc_run = runs_root / RUN_ID
    roc_run.mkdir()
    (roc_run / "results.json").write_text(json.dumps(_tiny_results()), encoding="utf-8")

    figures_root = tmp_path / "figures"
    results_root = tmp_path / "results"
    rc = make_figures.main(
        [
            "--runs-root",
            str(runs_root),
            "--figures-root",
            str(figures_root),
            "--results-root",
            str(results_root),
            "--dpi",
            "150",
        ]
    )
    assert rc == 0

    for name in ("roc_curve", "eer_fmr_bars", "gate_ablation"):
        png = figures_root / RUN_ID / f"{name}.png"
        assert png.is_file() and png.stat().st_size > 0
    assert (figures_root / "20261007T125746Z" / "fusion_methods.png").is_file()

    index_text = (figures_root / "index.md").read_text(encoding="utf-8")
    assert PROTOTYPE_NOTE in index_text
    assert f"{RUN_ID}/roc_curve.png" in index_text
    assert "20261007T125746Z/fusion_methods.png" in index_text
    assert f"../results/{RUN_ID}.json" in index_text

    sanitized = json.loads((results_root / f"{RUN_ID}.json").read_text(encoding="utf-8"))
    assert sanitized["config"]["data_root"] == "gallery"
    assert "roc_points" not in sanitized["sweep"][0]
    sanitized_fusion = json.loads(
        (results_root / "20261007T125746Z.json").read_text(encoding="utf-8")
    )
    assert sanitized_fusion["config"]["det_model"] == "det_10g.onnx"
    assert sanitized_fusion["ablation"]["face_only"]["overall"]["hit_at_1"] == 0.906

    # a latency-style run gets a sanitized copy but no figures
    latency_run = runs_root / "latency-20260101T000000Z"
    latency_run.mkdir()
    latency_results = {
        "config": {
            "data_root": None,
            "det_model": "/home/me/models/det_10g.onnx",
            "rec_model": "/home/me/models/w600k_r50.onnx",
        },
        "sweep": [{"kind": "faiss_hnsw", "index_size": 100, "search_p50_ms": 0.04}],
    }
    (latency_run / "results.json").write_text(json.dumps(latency_results), encoding="utf-8")
    rc = make_figures.main(
        [
            "--runs-root",
            str(runs_root),
            "--figures-root",
            str(figures_root),
            "--results-root",
            str(results_root),
        ]
    )
    assert rc == 0
    assert not (figures_root / "latency-20260101T000000Z").exists()
    sanitized_latency = json.loads(
        (results_root / "latency-20260101T000000Z.json").read_text(encoding="utf-8")
    )
    assert sanitized_latency["config"]["det_model"] == "det_10g.onnx"


def test_make_figures_selects_runs_by_stamp(tmp_path: Path) -> None:
    runs_root = tmp_path / "out"
    for stamp in ("20260101T000000Z", "20260102T000000Z"):
        run_dir = runs_root / stamp
        run_dir.mkdir(parents=True)
        results = _tiny_results()
        results["config"]["run_stamp"] = stamp
        (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")

    figures_root = tmp_path / "figures"
    results_root = tmp_path / "results"
    rc = make_figures.main(
        [
            "--runs",
            "20260101T000000Z",
            "--runs-root",
            str(runs_root),
            "--figures-root",
            str(figures_root),
            "--results-root",
            str(results_root),
        ]
    )
    assert rc == 0
    assert (figures_root / "20260101T000000Z" / "roc_curve.png").is_file()
    assert not (figures_root / "20260102T000000Z").exists()
    assert (results_root / "20260101T000000Z.json").is_file()
    assert not (results_root / "20260102T000000Z.json").is_file()
