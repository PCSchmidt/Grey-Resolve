"""Synthetic scenario generation for Phase 2 fusion experiments (deterministic).

The repo ships NO face imagery: personas + context metadata are fabricated on
top of user-supplied local face images, and near-tie ambiguity sets are
selected by measured embedding cosine. See docs/SYNTHETIC_SCENARIO_SPEC.md.
"""

from grey_resolve.scenario.generator import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    DegradationSpec,
    NearTieSet,
    Scenario,
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
