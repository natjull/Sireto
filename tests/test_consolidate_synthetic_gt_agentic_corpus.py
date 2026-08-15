from __future__ import annotations

import hashlib
import json

import pytest

from scripts import consolidate_synthetic_gt_agentic_corpus as consolidate
from scripts import run_synthetic_gt_agentic_loop as loop


def row(seed_id: str, variant_id: str, family: str) -> dict[str, str]:
    return {
        "seed_id": seed_id,
        "variant_id": variant_id,
        "families_json": loop.canonical_json([family]),
    }


def test_family_quotas_preserve_rare_families_and_fill_exact_limit() -> None:
    quotas = consolidate.family_quotas(
        {"COMMON_A": 100, "COMMON_B": 50, "RARE": 3},
        limit=100,
        preserve_below=3,
    )
    assert quotas["RARE"] == 3
    assert quotas["COMMON_A"] == 65
    assert quotas["COMMON_B"] == 32
    assert sum(quotas.values()) == 100


def test_selection_is_deterministic_and_keeps_every_rare_row() -> None:
    rows = [
        *(row(f"a-{index}", "v1", "COMMON") for index in range(10)),
        *(row(f"b-{index}", "v2", "RARE") for index in range(2)),
    ]
    first, quotas = consolidate.select_rows(
        list(rows), limit=7, selection_seed=42, preserve_below=2
    )
    second, _ = consolidate.select_rows(
        list(reversed(rows)), limit=7, selection_seed=42, preserve_below=2
    )
    assert quotas == {"COMMON": 5, "RARE": 2}
    assert [(value["seed_id"], value["variant_id"]) for value in first] == [
        (value["seed_id"], value["variant_id"]) for value in second
    ]
    assert sum(json.loads(value["families_json"])[0] == "RARE" for value in first) == 2


def test_generator_fidelity_requires_exact_raw_luna_fields() -> None:
    crm = {
        "name": "SOCIETE EXEMPLE",
        "address": "1 R EXEMPLE",
        "postcode": "75001",
        "city": "PARIS",
        "insee": "75056",
    }
    response = {
        "seed": {"siret": "12345678900012", "siren": "123456789"},
        "variants": [{
            "variant_id": "v1",
            "crm": crm,
            "corruption_families_observed": ["ADDRESS_ABBREVIATION"],
            "transformation_summary": "Abréviation directe.",
        }],
    }
    raw = loop.canonical_json(response)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    task = {
        "role": "GENERATOR",
        "status": "COMPLETED",
        "raw_response": raw,
        "task_id": "task-1",
    }
    variant = {
        "seed_id": "seed-1",
        "target_siret": "12345678900012",
        "target_siren": "123456789",
        "variant_id": "v1",
        "crm_json": loop.canonical_json(crm),
        "families_json": loop.canonical_json(["ADDRESS_ABBREVIATION"]),
        "transformation_summary": "Abréviation directe.",
        "generator_response_sha256": digest,
    }
    consolidate.check_generator_fidelity(variant, {digest: task})
    variant["crm_json"] = loop.canonical_json({**crm, "name": "TEXTE MODIFIE"})
    with pytest.raises(ValueError, match="differs from raw Luna"):
        consolidate.check_generator_fidelity(variant, {digest: task})


def test_family_quotas_refuse_insufficient_accepted_rows() -> None:
    with pytest.raises(ValueError, match="only 2 accepted variants"):
        consolidate.family_quotas({"A": 2}, limit=3, preserve_below=0)
