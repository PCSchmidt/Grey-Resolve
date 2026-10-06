"""Tests for the deterministic synthetic scenario generator (no model weights).

User-supplied images are tiny numpy arrays; nothing here loads or writes face
imagery. All randomness is seeded through ScenarioConfig.seed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from grey_resolve.scenario import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    DegradationSpec,
    ScenarioConfig,
    SyntheticIdentity,
    SyntheticMedia,
    TemplateVocab,
    config_hash,
    fabricate_media,
    generate_identities,
    generate_scenario,
    load_scenario,
    load_scenario_config,
    select_near_ties,
    write_scenario,
)


def _tiny_images(n: int = 3) -> list[np.ndarray]:
    """Return n distinct tiny uint8 "user-supplied" arrays."""
    return [np.zeros((4, 4, 3), dtype=np.uint8) + k for k in range(n)]


def _small_config(**overrides: object) -> ScenarioConfig:
    base = {"n_identities": 6, "n_geo_clusters": 3, "n_periods": 2, "media_per_identity": 4}
    base.update(overrides)
    return ScenarioConfig(**base)  # type: ignore[arg-type]


# --- config -----------------------------------------------------------------


def test_config_file_loads_with_expected_defaults():
    cfg = load_scenario_config("configs/scenario_v0.yaml")
    assert cfg.n_identities == 30
    assert cfg.n_geo_clusters == 5
    assert cfg.n_periods == 4
    assert cfg.media_per_identity == 20
    assert cfg.context_noise_rate == pytest.approx(0.15)
    assert cfg.context_absent_rate == pytest.approx(0.20)
    assert cfg.near_tie_band == (0.65, 0.75)
    assert cfg.seed == 42
    for name in ("geo_clusters", "platforms", "alias_templates", "entity_templates"):
        assert getattr(cfg.templates, name), f"missing template vocab: {name}"
    assert all(g.startswith("geo-") for g in cfg.templates.geo_clusters)


def test_config_hash_stable_and_sensitive_to_changes():
    cfg = _small_config()
    assert config_hash(cfg) == config_hash(_small_config())
    assert config_hash(cfg) != config_hash(_small_config(context_noise_rate=0.16))
    assert config_hash(cfg) != config_hash(_small_config(seed=43))
    assert config_hash(cfg) != config_hash(replace(cfg, templates=TemplateVocab(alias_templates=("alias-x", "alias-y"))))


def test_config_validation_errors():
    with pytest.raises(ValueError):
        _small_config(n_identities=0)
    with pytest.raises(ValueError):
        _small_config(media_per_identity=0)
    with pytest.raises(ValueError):
        _small_config(context_noise_rate=1.5)
    with pytest.raises(ValueError):
        _small_config(context_absent_rate=-0.1)
    with pytest.raises(ValueError):
        _small_config(timestamp_jitter_days=-1.0)
    with pytest.raises(ValueError):
        _small_config(near_tie_band=(0.8, 0.6))
    with pytest.raises(ValueError):
        _small_config(near_tie_band=(0.5,))
    with pytest.raises(ValueError):
        _small_config(near_tie_band=(-0.2, 0.5))
    with pytest.raises(ValueError):
        _small_config(max_near_tie_sets=0)
    with pytest.raises(ValueError):
        _small_config(templates=TemplateVocab(platforms=()))
    with pytest.raises(ValueError):
        _small_config(near_tie_band=(0.5,))  # type: ignore[arg-type]


# --- personas ---------------------------------------------------------------


def test_identity_schema_and_synthetic_flags():
    cfg = _small_config()
    identities = generate_identities(cfg)
    assert len(identities) == cfg.n_identities
    ids = [i.identity_id for i in identities]
    assert ids == [f"syn-id-{k:04d}" for k in range(1, cfg.n_identities + 1)]
    for k, ident in enumerate(identities, start=1):
        assert ident.synthetic is True
        assert ident.display_name == f"Fabricated Persona {k}"
        assert ident.archetype in cfg.templates.archetypes
        assert ident.home_geo_cluster in cfg.templates.geo_clusters
        start, end = ident.active_period
        assert start < end
        assert start.startswith("2024") and end.startswith("2024")
        assert set(ident.text_aliases) <= set(cfg.templates.alias_templates)
        assert ident.text_aliases
        assert ident.identity_id not in ident.known_associates
        assert set(ident.known_associates) <= set(ids)
        assert len(set(ident.known_associates)) == len(ident.known_associates)


def test_seeded_reproducibility_and_seed_sensitivity():
    cfg = _small_config()
    imgs = _tiny_images()
    run_a = (generate_identities(cfg), fabricate_media(imgs, generate_identities(cfg), cfg))
    run_b = (generate_identities(cfg), fabricate_media(imgs, generate_identities(cfg), cfg))
    assert [i.to_dict() for i in run_a[0]] == [i.to_dict() for i in run_b[0]]
    assert [m.to_dict() for m in run_a[1]] == [m.to_dict() for m in run_b[1]]

    other = _small_config(seed=43)
    ids_other = generate_identities(other)
    med_other = fabricate_media(imgs, ids_other, other)
    assert [i.to_dict() for i in ids_other] != [i.to_dict() for i in run_a[0]]
    assert [m.to_dict() for m in med_other] != [m.to_dict() for m in run_a[1]]


# --- media ------------------------------------------------------------------


def test_media_schema_stamps_and_counts():
    cfg = _small_config()
    identities = generate_identities(cfg)
    imgs = _tiny_images(2)
    media = fabricate_media(imgs, identities, cfg)
    assert len(media) == cfg.n_identities * cfg.media_per_identity
    assert [m.media_id for m in media] == [f"syn-media-{k:06d}" for k in range(1, len(media) + 1)]
    valid_ids = {i.identity_id for i in identities}
    for m in media:
        assert m.synthetic is True
        assert m.generator_version == GENERATOR_VERSION
        assert m.seed == cfg.seed
        assert m.config_hash == config_hash(cfg)
        assert m.identity_id in valid_ids
        assert m.image_ref in {"inline/000000", "inline/000001"}
        assert m.degradation.type in set(cfg.templates.degradation_types) | {"none"}
        assert 0.0 <= m.degradation.severity <= 1.0
        if m.degradation.type == "none":
            assert m.degradation.severity == 0.0
        if m.source_platform is not None:
            assert m.source_platform in cfg.templates.platforms
    per_identity = {i.identity_id: 0 for i in identities}
    for m in media:
        per_identity[m.identity_id] += 1
    assert set(per_identity.values()) == {cfg.media_per_identity}


def test_path_image_refs_are_referenced_not_copied(tmp_path: Path):
    cfg = _small_config(n_identities=2, media_per_identity=3)
    fake = tmp_path / "faces" / "shot.png"
    fake.parent.mkdir()
    fake.write_bytes(b"not-an-image")
    identities = generate_identities(cfg)
    media = fabricate_media([str(fake)], identities, cfg)
    assert all(m.image_ref == fake.as_posix() for m in media)
    assert fake.read_bytes() == b"not-an-image"


def test_context_noise_and_absent_rates_roughly_honored():
    cfg = replace(ScenarioConfig(), n_identities=50, media_per_identity=20)
    identities = generate_identities(cfg)
    by_id = {i.identity_id: i for i in identities}
    media = fabricate_media(_tiny_images(5), identities, cfg)
    assert len(media) == 1000

    def is_absent(m: SyntheticMedia) -> bool:
        return (
            m.timestamp is None
            and m.geo_cluster is None
            and m.source_platform is None
            and not m.text_entities
        )

    def is_noisy(m: SyntheticMedia, ident: SyntheticIdentity) -> bool:
        vocab = set(ident.text_aliases) | {ident.home_geo_cluster}
        if any(e not in vocab for e in m.text_entities):
            return True
        if m.timestamp is not None:
            start = date.fromisoformat(ident.active_period[0])
            end = date.fromisoformat(ident.active_period[1])
            if not (start <= m.timestamp.date() <= end):
                return True
        return False

    n_absent = sum(is_absent(m) for m in media)
    non_absent = [m for m in media if not is_absent(m)]
    assert n_absent / len(media) == pytest.approx(cfg.context_absent_rate, abs=0.05)
    n_noisy = sum(is_noisy(m, by_id[m.identity_id]) for m in non_absent)
    assert n_noisy / len(non_absent) == pytest.approx(cfg.context_noise_rate, abs=0.05)
    truthful = [m for m in non_absent if not is_noisy(m, by_id[m.identity_id])]
    n_drift = sum(m.geo_cluster != by_id[m.identity_id].home_geo_cluster for m in truthful)
    assert n_drift / len(truthful) == pytest.approx(cfg.geo_drift_rate, abs=0.06)


def test_zero_rates_give_all_truthful_context():
    cfg = _small_config(context_noise_rate=0.0, context_absent_rate=0.0, geo_drift_rate=0.0)
    identities = generate_identities(cfg)
    media = fabricate_media(_tiny_images(), identities, cfg)
    for m in media:
        ident = next(i for i in identities if i.identity_id == m.identity_id)
        assert m.timestamp is not None
        assert m.geo_cluster == ident.home_geo_cluster
        assert set(m.text_entities) <= set(ident.text_aliases) | {ident.home_geo_cluster}
        start = datetime.strptime(ident.active_period[0], "%Y-%m-%d").replace(tzinfo=UTC)
        end = datetime.strptime(ident.active_period[1], "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
        assert start <= m.timestamp < end


def test_labels_are_spread_across_personas():
    cfg = _small_config(n_identities=4, media_per_identity=5)
    identities = generate_identities(cfg)
    labels = ["real-a", "real-a", "real-b"]
    media = fabricate_media(_tiny_images(3), identities, cfg, labels=labels)
    ref_to_label = {f"inline/{k:06d}": lab for k, lab in enumerate(labels)}
    by_identity: dict[str, set[str]] = {}
    for m in media:
        by_identity.setdefault(m.identity_id, set()).add(ref_to_label[m.image_ref])
    assert all(len(labs) == 1 for labs in by_identity.values())
    used = {next(iter(labs)) for labs in by_identity.values()}
    assert used == {"real-a", "real-b"}


def test_media_helpers_map_to_core_types():
    cfg = _small_config()
    identities = generate_identities(cfg)
    m = fabricate_media(_tiny_images(), identities, cfg)[0]
    ctx = m.to_context()
    assert ctx.timestamp == m.timestamp
    assert ctx.geo_cluster == m.geo_cluster
    assert ctx.source_platform == m.source_platform
    assert ctx.text_entities == m.text_entities
    item = m.to_media_item()
    assert item.media_id == m.media_id
    assert item.image_ref == m.image_ref
    assert item.synthetic is True


def test_fabricate_media_input_errors():
    cfg = _small_config()
    identities = generate_identities(cfg)
    with pytest.raises(ValueError):
        fabricate_media([], identities, cfg)
    with pytest.raises(ValueError):
        fabricate_media(_tiny_images(), [], cfg)
    with pytest.raises(ValueError):
        fabricate_media(_tiny_images(2), identities, cfg, labels=["only-one"])
    with pytest.raises(TypeError):
        fabricate_media([5], identities, cfg)  # type: ignore[list-item]


# --- near-tie selection -----------------------------------------------------


def _angle_vectors(angles: list[float]) -> np.ndarray:
    return np.array([[math.cos(a), math.sin(a)] for a in angles], dtype=np.float64)


def test_near_tie_sets_are_in_band_cross_persona_and_disjoint():
    delta = math.acos(0.70)
    labels = [f"p{k}" for k in range(6)]
    vecs = _angle_vectors([k * delta for k in range(6)])
    sets = select_near_ties(vecs, labels, (0.65, 0.75), None)
    assert len(sets) == 3
    seen: set[str] = set()
    for s in sets:
        assert 2 <= len(s.identity_ids) <= 3
        assert not (set(s.identity_ids) & seen)
        seen |= set(s.identity_ids)
        assert len(s.pairs) == len(s.cosines)
        for (i, j), cos in zip(s.pairs, s.cosines):
            assert labels[i] != labels[j]
            assert 0.65 <= cos <= 0.75
    assert seen == set(labels)


def test_near_tie_excludes_same_persona_pairs():
    # Only the same-persona pair (rows 0, 1) lands in the band -> no cross-persona set.
    labels = ["p1", "p1", "p2"]
    vecs = _angle_vectors([0.0, math.acos(0.70), 2.0])
    assert select_near_ties(vecs, labels, (0.65, 0.75), None) == []


def test_near_tie_extends_to_three_personas():
    gram = np.full((3, 3), 0.70)
    np.fill_diagonal(gram, 1.0)
    trio = np.linalg.cholesky(gram)  # rows are 3 vectors with pairwise cosine 0.70
    vecs = np.vstack([trio, np.array([[-1.0, 0.0, 0.0]])])
    labels = ["a", "b", "c", "far"]
    sets = select_near_ties(vecs, labels, (0.65, 0.75), None)
    assert len(sets) == 1
    assert sets[0].identity_ids == ("a", "b", "c")
    assert len(sets[0].pairs) == 3
    assert all(0.65 <= c <= 0.75 for c in sets[0].cosines)


def test_near_tie_max_sets_and_determinism():
    delta = math.acos(0.70)
    labels = [f"p{k}" for k in range(6)]
    vecs = _angle_vectors([k * delta for k in range(6)])
    a = select_near_ties(vecs, labels, (0.65, 0.75), 2)
    b = select_near_ties(vecs, labels, (0.65, 0.75), 2)
    assert len(a) == 2
    assert a == b
    assert select_near_ties(vecs, labels, (0.65, 0.75), None) == select_near_ties(vecs, labels, (0.65, 0.75), None)


def test_near_tie_accepts_embed_fn_over_images():
    delta = math.acos(0.70)
    labels = [f"p{k}" for k in range(6)]
    vecs = _angle_vectors([k * delta for k in range(6)])
    images = [v.reshape(1, 1, 2) for v in vecs]

    def embed_fn(img: np.ndarray) -> np.ndarray:
        return np.asarray(img, dtype=np.float64).ravel()

    assert select_near_ties(embed_fn, labels, (0.65, 0.75), None, images=images) == select_near_ties(
        vecs, labels, (0.65, 0.75), None
    )


def test_near_tie_validation_errors():
    vecs = _angle_vectors([0.0, math.acos(0.70)])
    labels = ["p1", "p2"]
    with pytest.raises(ValueError):
        select_near_ties(vecs, labels, (0.8, 0.6))
    with pytest.raises(ValueError):
        select_near_ties(vecs, labels, (-0.1, 0.5))
    with pytest.raises(ValueError):
        select_near_ties(vecs, labels, (0.65,))
    with pytest.raises(ValueError):
        select_near_ties(vecs, labels, (0.1, 0.2, 0.3))
    with pytest.raises(ValueError):
        select_near_ties(vecs, ["only-one"])
    with pytest.raises(ValueError):
        select_near_ties(vecs, ["", "p2"])
    with pytest.raises(ValueError):
        select_near_ties(vecs, labels, (0.65, 0.75), 0)
    with pytest.raises(ValueError):
        select_near_ties(vecs.ravel(), labels)
    with pytest.raises(ValueError):
        select_near_ties(np.zeros((2, 3)), labels)
    with pytest.raises(ValueError):
        select_near_ties(np.array([[1.0, 0.0]]), ["solo"])
    with pytest.raises(ValueError):
        select_near_ties(lambda img: img, labels)  # embed_fn without images
    with pytest.raises(ValueError):
        select_near_ties(lambda img: None, labels, images=[np.zeros(2), np.zeros(2)])


# --- output writing / reading ----------------------------------------------


def test_write_and_load_scenario_round_trip(tmp_path: Path):
    cfg = _small_config()
    identities = generate_identities(cfg)
    media = fabricate_media(_tiny_images(2), identities, cfg)
    out = tmp_path / "v0"
    manifest = write_scenario(out, identities, media, cfg)
    scenario = load_scenario(out)
    assert scenario.identities == tuple(identities)
    assert scenario.media == tuple(media)
    assert scenario.manifest == manifest
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["seed"] == cfg.seed
    assert manifest["config_hash"] == config_hash(cfg)
    assert manifest["n_identities"] == len(identities)
    assert manifest["n_media"] == len(media)
    assert manifest["images_written"] is False
    for name, expected in manifest["file_hashes"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected


def test_manifest_config_hash_changes_when_config_changes(tmp_path: Path):
    cfg = _small_config()
    identities = generate_identities(cfg)
    media = fabricate_media(_tiny_images(), identities, cfg)
    m1 = write_scenario(tmp_path / "a", identities, media, cfg)
    m2 = write_scenario(tmp_path / "b", identities, media, cfg)
    assert m1["config_hash"] == m2["config_hash"]
    changed = _small_config(context_absent_rate=0.30)
    m3 = write_scenario(tmp_path / "c", identities, media, changed)
    assert m3["config_hash"] != m1["config_hash"]


def test_write_scenario_records_near_tie_sets(tmp_path: Path):
    cfg = _small_config()
    identities = generate_identities(cfg)
    media = fabricate_media(_tiny_images(), identities, cfg)
    delta = math.acos(0.70)
    sets = select_near_ties(_angle_vectors([0.0, delta]), ["a", "b"], (0.65, 0.75), None)
    out = tmp_path / "ties"
    manifest = write_scenario(out, identities, media, cfg, near_tie_sets=sets)
    assert manifest["near_tie_sets"] == [s.to_dict() for s in sets]
    assert load_scenario(out).manifest["near_tie_sets"] == manifest["near_tie_sets"]


def test_generate_scenario_writes_only_metadata_files(tmp_path: Path):
    cfg = _small_config()
    out = tmp_path / "v0"
    scenario = generate_scenario(_tiny_images(2), out, cfg)
    assert sorted(p.name for p in out.iterdir()) == ["identities.json", "manifest.json", "media.jsonl"]
    assert len(scenario.identities) == cfg.n_identities
    assert len(scenario.media) == cfg.n_identities * cfg.media_per_identity


def test_generate_scenario_is_reproducible(tmp_path: Path):
    cfg = _small_config()
    a = generate_scenario(_tiny_images(2), tmp_path / "a", cfg)
    b = generate_scenario(_tiny_images(2), tmp_path / "b", cfg)
    assert a.manifest == b.manifest
    assert (tmp_path / "a" / "media.jsonl").read_bytes() == (tmp_path / "b" / "media.jsonl").read_bytes()
    different = generate_scenario(_tiny_images(2), tmp_path / "c", _small_config(seed=7))
    assert different.manifest["config_hash"] != a.manifest["config_hash"]


def test_load_scenario_detects_tampered_files(tmp_path: Path):
    cfg = _small_config()
    out = tmp_path / "v0"
    generate_scenario(_tiny_images(2), out, cfg)
    with (out / "media.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"media_id": "tampered"}) + "\n")
    with pytest.raises(ValueError):
        load_scenario(out)


def test_records_round_trip_through_json():
    ident = SyntheticIdentity(
        identity_id="syn-id-0001",
        display_name="Fabricated Persona 1",
        synthetic=True,
        archetype="maritime-logistics-figure",
        home_geo_cluster="geo-alpha",
        active_period=("2024-01-01", "2024-03-31"),
        known_associates=("syn-id-0002",),
        text_aliases=("alias-one",),
    )
    item = SyntheticMedia(
        media_id="syn-media-000001",
        identity_id="syn-id-0001",
        synthetic=True,
        image_ref="inline/000000",
        degradation=DegradationSpec(type="blur", severity=0.4),
        timestamp=datetime(2024, 3, 14, 9, 20, 0, tzinfo=UTC),
        geo_cluster="geo-alpha",
        source_platform="platform-a",
        text_entities=("alias-one", "geo-alpha"),
        generator_version=GENERATOR_VERSION,
        seed=42,
        config_hash="deadbeef",
    )
    assert SyntheticIdentity.from_dict(json.loads(json.dumps(ident.to_dict()))) == ident
    assert SyntheticMedia.from_dict(json.loads(json.dumps(item.to_dict()))) == item
    no_ctx = replace(item, timestamp=None, geo_cluster=None, source_platform=None, text_entities=())
    assert SyntheticMedia.from_dict(json.loads(json.dumps(no_ctx.to_dict()))) == no_ctx
