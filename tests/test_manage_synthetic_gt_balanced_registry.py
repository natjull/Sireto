import hashlib
import json
from pathlib import Path

import pytest

from scripts.manage_synthetic_gt_balanced_registry import (
    empty_registry,
    quarantine,
    register,
    sha256,
    snapshot,
)


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def fixture(tmp_path: Path):
    seed = tmp_path / "seed.jsonl"
    promoted = tmp_path / "promoted.jsonl"
    manifest = tmp_path / "manifest.json"
    registry = tmp_path / "registry.json"
    fragment = {
        "field": "name", "relation": "LEGAL_FORM_REMOVE",
        "inspiration_ref": "a" * 64,
        "operation_parameters": {"removed_legal_forms": ["sarl"]},
    }
    address_fragment = {
        "field": "address", "relation": "ADDRESS_ABBREVIATE",
        "inspiration_ref": "b" * 64,
        "operation_parameters": {"pairs": [{"source": "rue", "target": "r"}]},
    }
    write_jsonl(seed, [{
        "seed_id": "P000:12345678900012",
        "target_siret": "12345678900012", "target_siren": "123456789",
        "seed_card": {"composite_contracts": [{
            "variant_id": "v1", "difficulty": "EASY",
            "augmentation_stratum": "NEAR_CLEAN_CONTROL",
            "field_relations": {
                "name": "LEGAL_FORM_REMOVE", "address": "ADDRESS_ABBREVIATE",
            },
            "field_inspirations": {"name": fragment, "address": address_fragment},
        }, {
            "variant_id": "v2", "difficulty": "HARD",
            "augmentation_stratum": "FAIL_BOTH_MODELS",
            "field_relations": {
                "name": "TOKEN_ORDER", "address": "ADDRESS_TOKEN_SUBSET",
            },
            "field_inspirations": {},
        }]},
    }])
    write_jsonl(promoted, [{
        "seed_id": "P000:12345678900012", "variant_id": "v1",
        "target_siret": "12345678900012", "target_siren": "123456789",
        "final_decision": "ACCEPT",
        "crm": {"name": "ALPHA", "address": "1 R TEST", "postcode": "75001",
                "city": "PARIS", "insee": "75056"},
        "full_sirene_qualification": {
            "decision": "EXACT_IDENTIFIABLE", "exact_witness": "G_N_A",
            "target_naturally_returned": True,
            "candidate_sirets": {"G_N_A": ["12345678900012"]},
        },
    }])
    promotion_manifest = {
        "schema_version": "sireto-synthetic-gt-full-exact-promotion-2",
        "run_id": "run-P000", "promotion_mode": "per-variant",
        "exact_witness": "G_N_A", "positive_injection": False,
        "qualification_uses_retrieval_or_model_scores": False,
        "promoted_variants": 1, "promoted_sha256": sha256(promoted),
        "source_hashes": {"seed_input": sha256(seed)},
    }
    manifest.write_text(json.dumps(promotion_manifest), encoding="utf-8")
    return seed, promoted, manifest, registry


def test_registry_counts_only_promoted_exact_contracts(tmp_path) -> None:
    seed, promoted, manifest, registry_path = fixture(tmp_path)
    args = type("Args", (), {
        "registry": registry_path, "target": 20_000, "batch_id": "P000",
        "seed_input": seed, "promoted": promoted, "promotion_manifest": manifest,
    })()
    registry = register(args)
    current = snapshot(registry)
    assert current["promoted_variants"] == 1
    assert current["difficulty_counts"] == {"EASY": 1}
    assert current["augmentation_stratum_counts"] == {"NEAR_CLEAN_CONTROL": 1}
    assert current["distinct_target_sirets"] == 1
    assert len(current["inspiration_ref_counts"]) == 2

    # Registration is idempotent after a crash/retry.
    assert register(args)["summary"] == current


def test_registry_rejects_mutated_promoted_artifact(tmp_path) -> None:
    seed, promoted, manifest, registry_path = fixture(tmp_path)
    args = type("Args", (), {
        "registry": registry_path, "target": 20_000, "batch_id": "P000",
        "seed_input": seed, "promoted": promoted, "promotion_manifest": manifest,
    })()
    register(args)
    promoted.write_text(promoted.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registered artifact changed"):
        snapshot(json.loads(registry_path.read_text()))


def test_quarantine_overlay_removes_only_bound_exact_keys(tmp_path) -> None:
    seed, promoted, manifest, registry_path = fixture(tmp_path)
    register(type("Args", (), {
        "registry": registry_path, "target": 20_000, "batch_id": "P000",
        "seed_input": seed, "promoted": promoted, "promotion_manifest": manifest,
    })())
    report_path = tmp_path / "quarantine.json"
    report = {
        "schema_version": "sireto-synthetic-gt-balanced-realism-quarantine-1",
        "source_registry": {"sha256": sha256(registry_path)},
        "quarantined_rows": 1,
        "records": [{
            "seed_id": "P000:12345678900012", "variant_id": "v1",
            "quarantined": True, "reason_codes": ["STREET_TYPE_COMPONENT_LOCK"],
        }],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    derived_path = tmp_path / "registry-v2.json"
    derived = quarantine(type("Args", (), {
        "source_registry": registry_path, "report": report_path,
        "output_registry": derived_path,
    })())
    assert derived["summary"]["promoted_variants"] == 0
    assert derived["summary"]["quarantined_variants"] == 1
    assert derived["summary"]["excluded_target_sirets"] == ["12345678900012"]
    assert derived["summary"]["batch_counts"] == {"P000": 0}

    report_path.write_text(json.dumps({**report, "quarantined_rows": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        snapshot(derived)
