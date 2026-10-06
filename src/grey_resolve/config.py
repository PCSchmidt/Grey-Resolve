"""Configuration loading for index hyperparameters and operating profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_INDEX = Path("configs/index_hnsw.yaml")
_DEFAULT_PROFILES = Path("configs/threshold_profiles.yaml")


@dataclass(frozen=True)
class HNSWConfig:
    """FAISS HNSW hyperparameters (cosine metric on normalized vectors)."""

    m: int = 16
    ef_construction: int = 40
    ef_search: int = 64


@dataclass(frozen=True)
class ThresholdProfile:
    """One operating profile: quality gate + fusion weights."""

    name: str
    quality_min: float
    ambiguous_low: float
    ambiguous_high: float
    alpha: float
    beta: float


def load_hnsw_config(path: str | Path = _DEFAULT_INDEX) -> HNSWConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    hnsw = raw.get("hnsw", raw)
    return HNSWConfig(
        m=int(hnsw.get("m", 16)),
        ef_construction=int(hnsw.get("ef_construction", 40)),
        ef_search=int(hnsw.get("ef_search", 64)),
    )


def load_threshold_profiles(path: str | Path = _DEFAULT_PROFILES) -> dict[str, ThresholdProfile]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    profiles: dict[str, ThresholdProfile] = {}
    for name, spec in (raw.get("profiles") or {}).items():
        profiles[name] = ThresholdProfile(
            name=name,
            quality_min=float(spec["quality_min"]),
            ambiguous_low=float(spec["ambiguous_low"]),
            ambiguous_high=float(spec["ambiguous_high"]),
            alpha=float(spec["alpha"]),
            beta=float(spec["beta"]),
        )
    if not profiles:
        raise ValueError(f"no threshold profiles found in {path}")
    return profiles
