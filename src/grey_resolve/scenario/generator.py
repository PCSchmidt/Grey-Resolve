"""Deterministic synthetic scenario generation for Phase 2 fusion experiments.

This module fabricates personas and context metadata on top of USER-SUPPLIED
local face images. The repository ships NO face imagery -- not even synthetic or
GAN faces -- because such imagery carries murky licenses (see the resolved
design decisions in docs/SYNTHETIC_SCENARIO_SPEC.md). Images are referenced by
path or held by the caller as numpy arrays; nothing here writes, copies, or
redistributes pixels. Near-tie ambiguity sets are SELECTED by measured
embedding cosine in the configured band, never faked.

Everything generated is marked ``synthetic: true`` and stamped with the
generator version, the seed, and the config hash.

The module is pure and deterministic: identical ``(config, images, labels)``
inputs produce identical records. Two RNG streams are derived from
``config.seed``: stream 0 fabricates personas, stream 1 fabricates media.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from grey_resolve.evaluation.metrics import build_pairs
from grey_resolve.types import ContextMetadata, MediaItem

__all__ = [
    "GENERATOR_VERSION",
    "SCHEMA_VERSION",
    "DegradationSpec",
    "NearTieSet",
    "Scenario",
    "ScenarioConfig",
    "SyntheticIdentity",
    "SyntheticMedia",
    "TemplateVocab",
    "config_hash",
    "fabricate_media",
    "generate_identities",
    "generate_scenario",
    "load_scenario",
    "load_scenario_config",
    "select_near_ties",
    "write_scenario",
]

SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "0.1.0"

_DEFAULT_CONFIG_PATH = Path("configs/scenario_v0.yaml")

_SEVERITY_LOW = 0.1
_SEVERITY_HIGH = 0.8
_STALE_SHIFT_DAYS = (180.0, 365.0)

# Corruption kinds applied when context_noise_rate fires (one per item):
# - "entities": misleading text entities drawn strictly OUTSIDE the persona's
#   truth vocabulary (its text_aliases + home geo cluster name).
# - "timestamp": a stale timestamp pushed outside the persona's active period.
# - "geo": a wrong geo cluster plus a text entity naming that foreign cluster.
# Every corruption is detectable from the written records; benign geo drift
# (geo_drift_rate) keeps text entities inside the persona's truth vocabulary.
_NOISE_KINDS = ("entities", "timestamp", "geo")


@dataclass(frozen=True)
class TemplateVocab:
    """Fictional template vocabulary used to fabricate scenario text."""

    geo_clusters: tuple[str, ...] = (
        "geo-alpha",
        "geo-beta",
        "geo-gamma",
        "geo-delta",
        "geo-epsilon",
    )
    platforms: tuple[str, ...] = (
        "platform-a",
        "platform-b",
        "platform-c",
        "platform-d",
        "platform-e",
    )
    archetypes: tuple[str, ...] = (
        "maritime-logistics-figure",
        "orbital-freight-broker",
        "glacier-research-liaison",
        "desert-solar-auditor",
        "island-telecom-archivist",
        "river-barge-coordinator",
        "tundra-mapping-enthusiast",
        "fictional-conference-circuit-speaker",
    )
    alias_templates: tuple[str, ...] = (
        "alias-one",
        "alias-two",
        "alias-three",
        "alias-four",
        "alias-five",
    )
    entity_templates: tuple[str, ...] = (
        "entity-ember",
        "entity-harbor",
        "entity-quartz",
        "entity-lantern",
        "entity-cascade",
    )
    degradation_types: tuple[str, ...] = (
        "gaussian_blur",
        "brightness",
        "downsample",
        "off_angle",
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict of the vocabulary."""
        return {
            "geo_clusters": list(self.geo_clusters),
            "platforms": list(self.platforms),
            "archetypes": list(self.archetypes),
            "alias_templates": list(self.alias_templates),
            "entity_templates": list(self.entity_templates),
            "degradation_types": list(self.degradation_types),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TemplateVocab:
        """Build a vocabulary from a mapping; unknown keys are ignored."""
        return cls(
            geo_clusters=tuple(raw.get("geo_clusters", cls.geo_clusters)),
            platforms=tuple(raw.get("platforms", cls.platforms)),
            archetypes=tuple(raw.get("archetypes", cls.archetypes)),
            alias_templates=tuple(raw.get("alias_templates", cls.alias_templates)),
            entity_templates=tuple(raw.get("entity_templates", cls.entity_templates)),
            degradation_types=tuple(raw.get("degradation_types", cls.degradation_types)),
        )


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario parameters (configs/scenario_v0.yaml). See the spec for meaning.

    Raises ValueError for out-of-range parameters.
    """

    n_identities: int = 30
    n_geo_clusters: int = 5
    n_periods: int = 4
    media_per_identity: int = 20
    context_noise_rate: float = 0.15
    context_absent_rate: float = 0.20
    geo_drift_rate: float = 0.20
    clean_media_rate: float = 0.15
    timestamp_jitter_days: float = 3.0
    near_tie_band: tuple[float, float] = (0.65, 0.75)
    max_near_tie_sets: int = 10
    seed: int = 42
    templates: TemplateVocab = TemplateVocab()

    def __post_init__(self) -> None:
        for name in ("n_identities", "n_geo_clusters", "n_periods", "media_per_identity", "max_near_tie_sets"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in (
            "context_noise_rate",
            "context_absent_rate",
            "geo_drift_rate",
            "clean_media_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if float(self.timestamp_jitter_days) < 0.0:
            raise ValueError("timestamp_jitter_days must be >= 0")
        _validate_band(self.near_tie_band)
        for field_name, values in self.templates.to_dict().items():
            if not values:
                raise ValueError(f"templates.{field_name} must be non-empty")
            if any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f"templates.{field_name} must contain non-empty strings")
        if len(set(self.templates.alias_templates)) < 1:
            raise ValueError("templates.alias_templates must contain at least one distinct alias")

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-friendly dict (used for the config hash)."""
        return {
            "n_identities": self.n_identities,
            "n_geo_clusters": self.n_geo_clusters,
            "n_periods": self.n_periods,
            "media_per_identity": self.media_per_identity,
            "context_noise_rate": self.context_noise_rate,
            "context_absent_rate": self.context_absent_rate,
            "geo_drift_rate": self.geo_drift_rate,
            "clean_media_rate": self.clean_media_rate,
            "timestamp_jitter_days": self.timestamp_jitter_days,
            "near_tie_band": list(self.near_tie_band),
            "max_near_tie_sets": self.max_near_tie_sets,
            "seed": self.seed,
            "templates": self.templates.to_dict(),
        }


def _validate_band(band: Sequence[float]) -> tuple[float, float]:
    """Validate a (low, high) cosine band; return it as a float tuple."""
    try:
        low, high = (float(band[0]), float(band[1]))
        extra = len(band)  # type: ignore[arg-type]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("band must be a 2-sequence of finite floats") from exc
    if extra != 2:
        raise ValueError("band must have exactly 2 values")
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError("band values must be finite")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("band must satisfy 0 <= low < high <= 1")
    return (low, high)


def config_hash(config: ScenarioConfig) -> str:
    """Return the sha256 hex digest of the canonical config serialization.

    Any parameter change (including the seed or template vocab) changes the
    hash; identical configs always produce the same hash.
    """
    blob = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_scenario_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> ScenarioConfig:
    """Load a ScenarioConfig from a YAML file (``scenario`` + ``templates`` keys)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    scen = raw.get("scenario", raw)
    templates = TemplateVocab.from_dict(raw.get("templates") or {})
    band = scen.get("near_tie_band", (0.65, 0.75))
    return ScenarioConfig(
        n_identities=int(scen.get("n_identities", 30)),
        n_geo_clusters=int(scen.get("n_geo_clusters", 5)),
        n_periods=int(scen.get("n_periods", 4)),
        media_per_identity=int(scen.get("media_per_identity", 20)),
        context_noise_rate=float(scen.get("context_noise_rate", 0.15)),
        context_absent_rate=float(scen.get("context_absent_rate", 0.20)),
        geo_drift_rate=float(scen.get("geo_drift_rate", 0.20)),
        clean_media_rate=float(scen.get("clean_media_rate", 0.15)),
        timestamp_jitter_days=float(scen.get("timestamp_jitter_days", 3.0)),
        near_tie_band=(float(band[0]), float(band[1])),
        max_near_tie_sets=int(scen.get("max_near_tie_sets", 10)),
        seed=int(scen.get("seed", 42)),
        templates=templates,
    )


@dataclass(frozen=True)
class SyntheticIdentity:
    """One fabricated persona (docs/SYNTHETIC_SCENARIO_SPEC.md identity schema)."""

    identity_id: str
    display_name: str
    synthetic: bool
    archetype: str
    home_geo_cluster: str
    active_period: tuple[str, str]
    known_associates: tuple[str, ...]
    text_aliases: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON schema representation."""
        return {
            "identity_id": self.identity_id,
            "display_name": self.display_name,
            "synthetic": self.synthetic,
            "archetype": self.archetype,
            "home_geo_cluster": self.home_geo_cluster,
            "active_period": list(self.active_period),
            "known_associates": list(self.known_associates),
            "text_aliases": list(self.text_aliases),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SyntheticIdentity:
        """Parse one identity record."""
        period = raw["active_period"]
        return cls(
            identity_id=str(raw["identity_id"]),
            display_name=str(raw["display_name"]),
            synthetic=bool(raw["synthetic"]),
            archetype=str(raw["archetype"]),
            home_geo_cluster=str(raw["home_geo_cluster"]),
            active_period=(str(period[0]), str(period[1])),
            known_associates=tuple(str(a) for a in raw["known_associates"]),
            text_aliases=tuple(str(a) for a in raw["text_aliases"]),
        )


@dataclass(frozen=True)
class DegradationSpec:
    """Degradation descriptor recorded for a media item.

    The generator never touches pixels; this is the degradation *label* a
    downstream pipeline applies to the referenced user-supplied image.
    ``type == "none"`` marks clean reference shots (severity 0.0).
    """

    type: str
    severity: float

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON schema representation."""
        return {"type": self.type, "severity": self.severity}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DegradationSpec:
        """Parse one degradation descriptor."""
        return cls(type=str(raw["type"]), severity=float(raw["severity"]))


@dataclass(frozen=True)
class SyntheticMedia:
    """One fabricated media record (spec MediaItem schema + provenance stamps).

    ``image_ref`` is a posix path string for path assets, or ``inline/NNNNNN``
    for caller-held numpy arrays (NNNNNN is the asset index in the supplied
    image list). ``timestamp``/``geo_cluster``/``source_platform``/``text_entities``
    are all empty when the item carries no context (context_absent_rate).
    """

    media_id: str
    identity_id: str
    synthetic: bool
    image_ref: str
    degradation: DegradationSpec
    timestamp: datetime | None
    geo_cluster: str | None
    source_platform: str | None
    text_entities: tuple[str, ...]
    generator_version: str
    seed: int
    config_hash: str

    def to_context(self) -> ContextMetadata:
        """Return the grey_resolve.types.ContextMetadata view of the context."""
        return ContextMetadata(
            timestamp=self.timestamp,
            geo_cluster=self.geo_cluster,
            source_platform=self.source_platform,
            text_entities=self.text_entities,
        )

    def to_media_item(self) -> MediaItem:
        """Return the grey_resolve.types.MediaItem view of this record."""
        return MediaItem(
            media_id=self.media_id,
            image_ref=self.image_ref,
            context=self.to_context(),
            synthetic=self.synthetic,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON schema representation (spec key order)."""
        return {
            "media_id": self.media_id,
            "identity_id": self.identity_id,
            "synthetic": self.synthetic,
            "image_ref": self.image_ref,
            "degradation": self.degradation.to_dict(),
            "timestamp": self.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ") if self.timestamp else None,
            "geo_cluster": self.geo_cluster,
            "source_platform": self.source_platform,
            "text_entities": list(self.text_entities),
            "generator_version": self.generator_version,
            "seed": self.seed,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SyntheticMedia:
        """Parse one media record."""
        ts = raw.get("timestamp")
        timestamp = (
            datetime.strptime(str(ts), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            if ts
            else None
        )
        entities = raw.get("text_entities") or ()
        return cls(
            media_id=str(raw["media_id"]),
            identity_id=str(raw["identity_id"]),
            synthetic=bool(raw["synthetic"]),
            image_ref=str(raw["image_ref"]),
            degradation=DegradationSpec.from_dict(raw["degradation"]),
            timestamp=timestamp,
            geo_cluster=str(raw["geo_cluster"]) if raw.get("geo_cluster") is not None else None,
            source_platform=str(raw["source_platform"]) if raw.get("source_platform") is not None else None,
            text_entities=tuple(str(e) for e in entities),
            generator_version=str(raw["generator_version"]),
            seed=int(raw["seed"]),
            config_hash=str(raw["config_hash"]),
        )


@dataclass(frozen=True)
class NearTieSet:
    """One near-tie ambiguity set: 2-3 personas with in-band pairwise cosines.

    ``identity_ids`` are the distinct persona labels (sorted); ``pairs`` are
    ``(i, j)`` row indices into the embedding array with ``i < j`` and
    ``cosines[k]`` the measured cosine of ``pairs[k]``.
    """

    identity_ids: tuple[str, ...]
    pairs: tuple[tuple[int, int], ...]
    cosines: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (for manifest recording)."""
        return {
            "identity_ids": list(self.identity_ids),
            "pairs": [list(p) for p in self.pairs],
            "cosines": list(self.cosines),
        }


@dataclass(frozen=True)
class Scenario:
    """A generated scenario: identities, media records, and the manifest."""

    identities: tuple[SyntheticIdentity, ...]
    media: tuple[SyntheticMedia, ...]
    manifest: dict[str, Any]


def _rng_stream(config: ScenarioConfig, stream: int) -> np.random.Generator:
    """Return the deterministic RNG for one generation stream."""
    return np.random.default_rng([int(config.seed), int(stream)])


def _expand_names(vocab: Sequence[str], n: int, prefix: str) -> tuple[str, ...]:
    """Return ``n`` distinct names: the vocab prefix plus generated extras."""
    names = list(dict.fromkeys(vocab))
    if n <= len(names):
        return tuple(names[:n])
    extras = [f"{prefix}-extra-{k}" for k in range(1, n - len(names) + 1)]
    return tuple(names + extras)


def _period_windows(n_periods: int) -> tuple[tuple[datetime, datetime], ...]:
    """Return ``n_periods`` consecutive 3-month windows starting 2024-01-01.

    Each window is ``(start, end)`` with ``start`` inclusive and ``end``
    exclusive, UTC.
    """
    windows: list[tuple[datetime, datetime]] = []
    for p in range(n_periods):
        year = 2024 + p // 4
        month = 1 + 3 * (p % 4)
        start = datetime(year, month, 1, tzinfo=UTC)
        end = datetime(year + 1, 1, 1, tzinfo=UTC) if month == 10 else datetime(year, month + 3, 1, tzinfo=UTC)
        windows.append((start, end))
    return tuple(windows)


def generate_identities(config: ScenarioConfig) -> list[SyntheticIdentity]:
    """Fabricate ``config.n_identities`` synthetic personas (deterministic).

    Identities use RNG stream 0. Fields follow the spec identity schema:
    ``identity_id`` "syn-id-0001"-style, ``display_name`` "Fabricated Persona N"
    (obviously fictional), geo clusters and active periods spread round-robin,
    associates and text aliases sampled from the template vocab.

    Raises ValueError if the config is invalid (see ScenarioConfig).
    """
    rng = _rng_stream(config, 0)
    geo_names = _expand_names(config.templates.geo_clusters, config.n_geo_clusters, "geo")
    windows = _period_windows(config.n_periods)
    aliases = tuple(config.templates.alias_templates)
    identities: list[SyntheticIdentity] = []
    all_ids = [f"syn-id-{i:04d}" for i in range(1, config.n_identities + 1)]
    for i in range(1, config.n_identities + 1):
        identity_id = f"syn-id-{i:04d}"
        start, end = windows[(i - 1) % len(windows)]
        n_assoc = min(int(rng.integers(0, 3)), config.n_identities - 1)
        others = [oid for oid in all_ids if oid != identity_id]
        associates = (
            tuple(sorted(others[idx] for idx in rng.choice(len(others), size=n_assoc, replace=False)))
            if n_assoc
            else ()
        )
        n_alias = min(2, len(aliases))
        picked = rng.choice(len(aliases), size=n_alias, replace=False)
        identities.append(
            SyntheticIdentity(
                identity_id=identity_id,
                display_name=f"Fabricated Persona {i}",
                synthetic=True,
                archetype=config.templates.archetypes[(i - 1) % len(config.templates.archetypes)],
                home_geo_cluster=geo_names[(i - 1) % len(geo_names)],
                active_period=(
                    start.strftime("%Y-%m-%d"),
                    (end - timedelta(days=1)).strftime("%Y-%m-%d"),
                ),
                known_associates=associates,
                text_aliases=tuple(sorted(aliases[idx] for idx in picked)),
            )
        )
    return identities


def _asset_refs(images: Sequence[Any]) -> list[str]:
    """Map supplied images to stable image_ref strings."""
    refs: list[str] = []
    for k, img in enumerate(images):
        if isinstance(img, np.ndarray):
            refs.append(f"inline/{k:06d}")
        elif isinstance(img, (str, Path)):
            refs.append(Path(img).as_posix())
        else:
            raise TypeError(f"images[{k}] must be a numpy.ndarray, str, or Path")
    return refs


def _sample_subset(rng: np.random.Generator, values: Sequence[str], k: int) -> tuple[str, ...]:
    """Sample ``k`` distinct values (sorted output) without replacement."""
    k = min(k, len(values))
    idx = rng.choice(len(values), size=k, replace=False)
    return tuple(sorted(values[int(j)] for j in idx))


def fabricate_media(
    images: Sequence[Any],
    identities: Sequence[SyntheticIdentity],
    config: ScenarioConfig,
    labels: Sequence[str] | None = None,
) -> list[SyntheticMedia]:
    """Fabricate ``config.media_per_identity`` media records per persona.

    User-supplied images (paths or numpy arrays) are assigned to personas and
    cycled deterministically; no pixels are read or written. If ``labels`` is
    given (one per image, e.g. real anchor labels), images are grouped by
    label and each persona is bound to one label group, spreading the supplied
    labels across personas.

    Context fabrication per item (RNG stream 1), in order of precedence:

    - With probability ``config.context_absent_rate`` the item carries NO
      context (timestamp/geo/platform/entities all empty).
    - Otherwise, with probability ``config.context_noise_rate`` exactly one
      corruption fires: misleading text entities (drawn strictly outside the
      persona's truth vocabulary of its text_aliases + home geo), a stale
      timestamp outside the persona's active period, or a wrong geo cluster
      plus a text entity naming it. Each corruption is detectable from the
      written records.
    - Otherwise the context is truthful: timestamp inside the active period
      (uniform plus gaussian jitter of ``timestamp_jitter_days``, clipped),
      home geo cluster -- drifted to another cluster with probability
      ``config.geo_drift_rate`` (entities stay truthful) -- a random source
      platform, and text entities sampled from the persona's truth vocabulary.

    Each record is stamped ``synthetic=True`` with ``generator_version``,
    ``config.seed``, and ``config_hash(config)``. Degradation descriptors are
    sampled labels (``clean_media_rate`` clean); applying them to pixels is a
    downstream concern.

    Raises ValueError for empty images/identities, label length mismatches, or
    a template vocab too small to fabricate misleading entities.
    """
    if not identities:
        raise ValueError("identities must be non-empty")
    if not images:
        raise ValueError("images must be non-empty")
    if labels is not None and len(labels) != len(images):
        raise ValueError("labels must have one entry per image")
    refs = _asset_refs(images)

    if labels is None:
        label_of_asset: list[str | None] = [None] * len(refs)
    else:
        label_of_asset = [str(lab) for lab in labels]
    group_order = list(dict.fromkeys(label_of_asset))
    groups: dict[str | None, list[int]] = {lab: [] for lab in group_order}
    for k, lab in enumerate(label_of_asset):
        groups[lab].append(k)

    rng = _rng_stream(config, 1)
    windows = _period_windows(config.n_periods)
    geo_names = list(_expand_names(config.templates.geo_clusters, config.n_geo_clusters, "geo"))
    platforms = tuple(config.templates.platforms)
    deg_types = tuple(config.templates.degradation_types)
    noise_pool = sorted(
        set(config.templates.alias_templates) | set(config.templates.entity_templates) | set(geo_names)
    )
    chash = config_hash(config)

    media: list[SyntheticMedia] = []
    counter = 0
    for i, ident in enumerate(identities):
        truth_vocab = sorted(set(ident.text_aliases) | {ident.home_geo_cluster})
        pool = [e for e in noise_pool if e not in set(truth_vocab)]
        if not pool:
            raise ValueError("template vocab too small to fabricate misleading entities")
        window = windows[i % len(windows)]
        start, end = window
        end_last = end - timedelta(seconds=1)
        total_secs = int((end - start).total_seconds())
        group = groups[group_order[i % len(group_order)]]
        for j in range(config.media_per_identity):
            counter += 1
            absent_roll, noise_roll, drift_roll = rng.random(), rng.random(), rng.random()
            off = int(rng.integers(0, total_secs))
            jitter = float(rng.normal(0.0, config.timestamp_jitter_days))
            platform = platforms[int(rng.integers(0, len(platforms)))]
            clean_roll = float(rng.random())
            if clean_roll < config.clean_media_rate:
                degradation = DegradationSpec(type="none", severity=0.0)
            else:
                degradation = DegradationSpec(
                    type=deg_types[int(rng.integers(0, len(deg_types)))],
                    severity=round(float(rng.uniform(_SEVERITY_LOW, _SEVERITY_HIGH)), 4),
                )

            timestamp: datetime | None
            geo_cluster: str | None
            source_platform: str | None
            entities: tuple[str, ...]
            if absent_roll < config.context_absent_rate:
                timestamp, geo_cluster, source_platform, entities = None, None, None, ()
            else:
                base_ts = start + timedelta(seconds=off)
                timestamp = min(max(base_ts + timedelta(days=jitter), start), end_last)
                timestamp = timestamp.replace(microsecond=0)
                home_idx = geo_names.index(ident.home_geo_cluster)
                if drift_roll < config.geo_drift_rate and len(geo_names) > 1:
                    geo_cluster = geo_names[(home_idx + 1 + int(rng.integers(0, len(geo_names) - 1))) % len(geo_names)]
                else:
                    geo_cluster = ident.home_geo_cluster
                source_platform = platform
                entities = _sample_subset(rng, truth_vocab, 2)
                if noise_roll < config.context_noise_rate:
                    kind = _NOISE_KINDS[int(rng.integers(0, len(_NOISE_KINDS)))]
                    if kind == "entities":
                        entities = _sample_subset(rng, pool, 2)
                    elif kind == "timestamp":
                        shift = float(rng.uniform(*_STALE_SHIFT_DAYS))
                        sign = 1 if int(rng.integers(0, 2)) else -1
                        timestamp = (base_ts + timedelta(days=sign * shift)).replace(microsecond=0)
                    else:  # "geo": wrong cluster + entity naming it (detectable)
                        if len(geo_names) > 1:
                            geo_cluster = geo_names[(home_idx + 1 + int(rng.integers(0, len(geo_names) - 1))) % len(geo_names)]
                            entities = tuple(sorted({geo_cluster, *_sample_subset(rng, truth_vocab, 1)}))
                        else:
                            entities = _sample_subset(rng, pool, 2)

            if labels is None:
                asset_idx = (i + j) % len(refs)
            else:
                asset_idx = group[(i + j) % len(group)]
            media.append(
                SyntheticMedia(
                    media_id=f"syn-media-{counter:06d}",
                    identity_id=ident.identity_id,
                    synthetic=True,
                    image_ref=refs[asset_idx],
                    degradation=degradation,
                    timestamp=timestamp,
                    geo_cluster=geo_cluster,
                    source_platform=source_platform,
                    text_entities=entities,
                    generator_version=GENERATOR_VERSION,
                    seed=int(config.seed),
                    config_hash=chash,
                )
            )
    return media


# --- Near-tie selection ------------------------------------------------------

EmbedFn = Callable[[np.ndarray], "np.ndarray | None"]
EmbedInput = Any  # np.ndarray of precomputed vectors, or EmbedFn + images


def _vectors_from_input(
    embeddings: EmbedInput,
    labels: Sequence[str],
    images: Sequence[np.ndarray] | None,
) -> np.ndarray:
    """Resolve precomputed vectors or an embed_fn over images to a 2-D array."""
    if callable(embeddings):
        if images is None:
            raise ValueError("images must be provided when embeddings is an embed_fn")
        rows: list[np.ndarray] = []
        for k, img in enumerate(images):
            vec = embeddings(img)
            if vec is None:
                raise ValueError(f"embed_fn returned None for images[{k}]")
            rows.append(np.asarray(vec, dtype=np.float64).ravel())
        arr = np.stack(rows) if rows else np.empty((0, 0))
    else:
        arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("embeddings must be 2-D of shape (N, dim)")
    if arr.shape[0] != len(labels):
        raise ValueError("labels must have length N matching embeddings")
    return arr


def select_near_ties(
    embeddings: EmbedInput,
    labels: Sequence[str],
    band: tuple[float, float] = (0.65, 0.75),
    max_sets: int | None = None,
    *,
    images: Sequence[np.ndarray] | None = None,
) -> list[NearTieSet]:
    """Select measured near-tie ambiguity sets (2-3 personas each).

    ``embeddings`` is either precomputed vectors of shape (N, dim) or an
    ``embed_fn`` mapping one image to a vector (``images`` required then).
    Cross-persona pairs whose cosine lands in ``band`` (inclusive) are the
    near-tie candidates. Candidates are scanned in deterministic order
    (cosine descending, then row index); each candidate seeds a set greedily,
    extended to a third persona when one has in-band pairs to both members.
    Every persona appears in at most one set. Returns at most ``max_sets``
    (all of them when None), ordered deterministically.

    Raises ValueError for invalid bands, mismatched/empty labels, bad
    ``max_sets``, malformed embeddings, zero vectors, or an embed_fn that
    returns None.
    """
    low, high = _validate_band(band)
    lab = list(labels)
    if not lab or any(not isinstance(l, str) or not l for l in lab):
        raise ValueError("labels must be non-empty strings")
    if max_sets is not None and int(max_sets) < 1:
        raise ValueError("max_sets must be >= 1 or None")
    arr = _vectors_from_input(embeddings, lab, images)
    if arr.shape[0] < 2:
        raise ValueError("need at least 2 embeddings to form near-tie pairs")
    if not np.all(np.isfinite(arr)):
        raise ValueError("embeddings must be finite")
    if np.any(np.linalg.norm(arr, axis=1) == 0.0):
        raise ValueError("embeddings must not contain zero vectors")

    scores, is_genuine = build_pairs(arr, lab)
    ii, jj = np.triu_indices(arr.shape[0], k=1)
    candidates = [
        (int(i), int(j), float(s))
        for i, j, s, gen in zip(ii, jj, scores, is_genuine)
        if not gen and low <= float(s) <= high
    ]
    candidates.sort(key=lambda t: (-t[2], t[0], t[1]))

    by_pair: dict[tuple[str, str], tuple[tuple[int, int, float], ...]] = {}
    for i, j, s in candidates:
        key = tuple(sorted((lab[i], lab[j])))  # type: ignore[assignment]
        by_pair.setdefault(key, ())  # type: ignore[arg-type]
        by_pair[key] = by_pair[key] + ((i, j, s),)  # type: ignore[index]

    def _first_entry(a: str, b: str) -> tuple[int, int, float] | None:
        entries = by_pair.get(tuple(sorted((a, b))))  # type: ignore[arg-type]
        return entries[0] if entries else None

    used: set[str] = set()
    all_labels = sorted(set(lab))
    sets: list[NearTieSet] = []
    for i, j, s in candidates:
        if max_sets is not None and len(sets) >= max_sets:
            break
        la, lb = lab[i], lab[j]
        if la in used or lb in used:
            continue
        members = [la, lb]
        pairs: list[tuple[int, int, float]] = [(i, j, s)]
        thirds: list[tuple[float, str, tuple[int, int, float], tuple[int, int, float]]] = []
        for other in all_labels:
            if other in used or other in members:
                continue
            e1 = _first_entry(la, other)
            e2 = _first_entry(lb, other)
            if e1 is not None and e2 is not None:
                thirds.append((-(e1[2] + e2[2]), other, e1, e2))
        if thirds:
            thirds.sort(key=lambda t: (t[0], t[1]))
            _, other, e1, e2 = thirds[0]
            pairs.extend([e1, e2])
            members.append(other)
        used.update(members)
        pairs.sort(key=lambda t: (t[0], t[1]))
        sets.append(
            NearTieSet(
                identity_ids=tuple(sorted(members)),
                pairs=tuple((p[0], p[1]) for p in pairs),
                cosines=tuple(p[2] for p in pairs),
            )
        )
    return sets


# --- Output writing / reading -----------------------------------------------


def write_scenario(
    out_dir: str | Path,
    identities: Sequence[SyntheticIdentity],
    media: Sequence[SyntheticMedia],
    config: ScenarioConfig,
    near_tie_sets: Sequence[NearTieSet] | None = None,
) -> dict[str, Any]:
    """Write ``identities.json``, ``media.jsonl``, and ``manifest.json``.

    Face images are NEVER written: ``image_ref`` values point at user-supplied
    paths or caller-held inline arrays. The manifest records the schema
    version, generator version, seed, config hash + config snapshot, per-file
    sha256 hashes, counts, and optionally the near-tie sets. Returns the
    manifest dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids_blob = (
        json.dumps([ident.to_dict() for ident in identities], indent=2) + "\n"
    ).encode("utf-8")
    media_blob = (
        "".join(json.dumps(item.to_dict()) + "\n" for item in media)
    ).encode("utf-8")
    (out / "identities.json").write_bytes(ids_blob)
    (out / "media.jsonl").write_bytes(media_blob)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": int(config.seed),
        "config_hash": config_hash(config),
        "config": config.to_dict(),
        "n_identities": len(identities),
        "n_media": len(media),
        "file_hashes": {
            "identities.json": hashlib.sha256(ids_blob).hexdigest(),
            "media.jsonl": hashlib.sha256(media_blob).hexdigest(),
        },
        "images_written": False,
        "image_refs_external": True,
    }
    if near_tie_sets is not None:
        manifest["near_tie_sets"] = [s.to_dict() for s in near_tie_sets]
    (out / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    )
    return manifest


def load_scenario(directory: str | Path) -> Scenario:
    """Read a scenario written by :func:`write_scenario`.

    Verifies per-file sha256 hashes against the manifest. Returns the parsed
    records. Raises ValueError when a file hash does not match the manifest.
    """
    d = Path(directory)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("file_hashes", {})
    for name in ("identities.json", "media.jsonl"):
        blob = (d / name).read_bytes()
        expected = hashes.get(name)
        actual = hashlib.sha256(blob).hexdigest()
        if expected is not None and expected != actual:
            raise ValueError(f"file hash mismatch for {name}: manifest={expected} actual={actual}")
    raw_ids = json.loads((d / "identities.json").read_text(encoding="utf-8"))
    identities = tuple(SyntheticIdentity.from_dict(r) for r in raw_ids)
    lines = (d / "media.jsonl").read_text(encoding="utf-8").splitlines()
    media = tuple(SyntheticMedia.from_dict(json.loads(line)) for line in lines if line.strip())
    return Scenario(identities=identities, media=media, manifest=manifest)


def generate_scenario(
    images: Sequence[Any],
    out_dir: str | Path,
    config: ScenarioConfig,
    labels: Sequence[str] | None = None,
    near_tie_sets: Sequence[NearTieSet] | None = None,
) -> Scenario:
    """Generate personas + media from user-supplied images and write + reload.

    Composes :func:`generate_identities`, :func:`fabricate_media`, and
    :func:`write_scenario`, then returns :func:`load_scenario` of the output
    directory so the caller sees exactly what was written.
    """
    identities = generate_identities(config)
    media = fabricate_media(images, identities, config, labels=labels)
    write_scenario(out_dir, identities, media, config, near_tie_sets=near_tie_sets)
    return load_scenario(out_dir)
