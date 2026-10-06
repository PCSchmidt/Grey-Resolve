"""Smoke tests for configuration loading and core contracts (no heavy deps)."""

from __future__ import annotations

import numpy as np
import pytest

from grey_resolve.config import load_hnsw_config, load_threshold_profiles
from grey_resolve.types import ContextMetadata, MediaItem, normalize_embedding


def test_hnsw_config_matches_baseline():
    cfg = load_hnsw_config("configs/index_hnsw.yaml")
    assert cfg.m == 16
    assert cfg.ef_construction == 40
    assert cfg.ef_search == 64


def test_threshold_profiles_load():
    profiles = load_threshold_profiles("configs/threshold_profiles.yaml")
    assert set(profiles) == {"strict_surveillance", "broad_lead"}
    strict = profiles["strict_surveillance"]
    assert strict.quality_min > profiles["broad_lead"].quality_min
    assert strict.alpha + strict.beta == pytest.approx(1.0)


def test_normalize_embedding_unit_norm():
    vec = normalize_embedding(np.array([3.0, 4.0], dtype=np.float32))
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_embedding(np.zeros(4, dtype=np.float32))


def test_media_item_defaults():
    item = MediaItem(media_id="m1", image_ref="images/1.png")
    assert item.context == ContextMetadata()
    assert item.synthetic is False
