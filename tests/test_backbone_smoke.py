"""Smoke tests for the backbone adapters. Skipped unless weights are fetched.

Weights are downloaded at runtime (scripts/fetch_backbone.py) and are
non-commercial; they are never committed. These tests embed synthetic crops
only -- no real-person imagery is used or required.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from grey_resolve.types import normalize_embedding

_MODEL_DIR = Path(os.environ.get("GREY_RESOLVE_MODEL_DIR", Path.home() / ".insightface" / "models"))
_DET = _MODEL_DIR / "det_10g.onnx"
_REC = _MODEL_DIR / "w600k_r50.onnx"

pytestmark = pytest.mark.skipif(
    not (_DET.exists() and _REC.exists()),
    reason="backbone weights not fetched (run scripts/fetch_backbone.py)",
)


def test_extractor_returns_unit_norm_512():
    from grey_resolve.embeddings.insightface_arcface import InsightFaceArcFaceExtractor

    extractor = InsightFaceArcFaceExtractor(model_file=_REC)
    rng = np.random.default_rng(11)
    crop = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    emb = extractor.extract_aligned(crop)
    assert emb.shape == (512,)
    assert float(np.linalg.norm(emb)) == pytest.approx(1.0, abs=1e-4)
    # deterministic for the same input
    emb2 = extractor.extract_aligned(crop)
    assert np.allclose(emb, emb2)


def test_extractor_id_and_dimension():
    from grey_resolve.embeddings.insightface_arcface import InsightFaceArcFaceExtractor

    extractor = InsightFaceArcFaceExtractor(model_file=_REC)
    assert extractor.dimension == 512
    assert extractor.model_id == "insightface/w600k_r50"


def test_detector_runs_on_noise_and_finds_nothing():
    from grey_resolve.detection.scrfd import ScrfdDetector

    detector = ScrfdDetector(model_file=_DET)
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 256, size=(320, 320, 3), dtype=np.uint8)
    assert detector.detect(noise) == []
