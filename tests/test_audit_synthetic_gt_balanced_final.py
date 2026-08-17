from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_synthetic_gt_balanced_final as audit
from scripts import manage_synthetic_gt_balanced_registry as registry_lib
from scripts import run_synthetic_gt_agentic_loop as loop


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _fragment(field: str, relation: str, ref: str) -> dict:
    return {
        "schema_version": "sireto-synthetic-field-inspiration-1",
        "field": field,
        "relation": relation,
        "inspiration_ref": ref * 64,
        "provenance_digest": ref * 64,
        "source_fold": 2,
        "source_legacy_split": "train",
        "source_state": "A",
        "official_value": "SARL ALPHA" if field == "name" else "1 RUE TEST",
        "observed_crm_value": "ALPHA" if field == "name" else "1 R TEST",
        "operation_parameters": {},
    }


def _fixture(tmp_path: Path) -> argparse.Namespace:
    artifacts = tmp_path / "balanced"
    artifacts.mkdir()
    seed_path = artifacts / "P000_seed_input.jsonl"
    promoted_dir = artifacts / "P000_promoted"
    promoted_dir.mkdir()
    promoted_path = promoted_dir / "promoted.jsonl"
    promotion_manifest_path = promoted_dir / "manifest.json"
    ledger_path = artifacts / "P000.sqlite"
    full_audit_path = artifacts / "P000_full_sirene_audit.json"
    ledger_path.write_bytes(b"sealed-ledger")

    contract = {
        "variant_id": "v1",
        "difficulty": "HARD",
        "augmentation_stratum": "FAIL_BOTH_MODELS",
        "field_relations": {
            "name": "LEGAL_FORM_REMOVE",
            "address": "ADDRESS_ABBREVIATE",
        },
        "field_inspirations": {
            "name": _fragment("name", "LEGAL_FORM_REMOVE", "a"),
            "address": _fragment("address", "ADDRESS_ABBREVIATE", "b"),
        },
        "targeting_evidence": {
            "identity_free_aggregate": True,
            "catalog_sha256": "c" * 64,
        },
    }
    seed = {
        "seed_id": "P000:12345678900012",
        "target_siret": "12345678900012",
        "target_siren": "123456789",
        "seed_card": {
            "composite_contracts": [contract],
            "official_context": {
                "target": {
                    "names": [{"kind": "OFFICIAL_NAME", "value": "SARL ALPHA"}],
                    "address": {
                        "number": "1", "repetition_index": "", "street_type": "RUE",
                        "street": "TEST", "postcode": "75001", "city": "PARIS",
                        "insee": "75056",
                    },
                    "state": "A",
                }
            },
        },
    }
    _write_jsonl(seed_path, [seed])
    qualification = {
        "decision": "EXACT_IDENTIFIABLE",
        "exact_witness": "G_N_A",
        "target_naturally_returned": True,
        "candidate_sirets": {"G_N_A": ["12345678900012"]},
        "operational_equivalence": False,
        "operational_equivalent_sirets": [],
    }
    promoted = {
        "seed_id": seed["seed_id"],
        "variant_id": "v1",
        "target_siret": seed["target_siret"],
        "target_siren": seed["target_siren"],
        "source_kind": "SIRENE_ONLY_TRAIN",
        "oof_fold": -1,
        "difficulty": "HARD",
        "augmentation_stratum": "FAIL_BOTH_MODELS",
        "crm": {
            "name": "ALPHA", "address": "1 R TEST", "postcode": "75001",
            "city": "PARIS", "insee": "75056",
        },
        "corruption_families_observed": ["OBSERVED_COMPOSITE_ANALOGY"],
        "transformation_summary": "Forme juridique retirée et rue abrégée.",
        "variant_contract_sha256": loop.digest_json(contract),
        "generator_response_sha256": "d" * 64,
        "critic_decision": "ACCEPT",
        "adjudicator_decision": None,
        "final_decision": "ACCEPT",
        "final_reason": "AGENTIC_REVIEW_PASS",
        "promotion_provenance": {
            "accepted_task_id": "task-1",
            "generator_response_sha256": "d" * 64,
            "critic_decision": "ACCEPT",
        },
        "full_sirene_qualification": qualification,
    }
    _write_jsonl(promoted_path, [promoted])

    sirene_establishments = tmp_path / "establishments.parquet"
    sirene_legal_units = tmp_path / "units.parquet"
    sirene_establishments.write_bytes(b"establishments-snapshot")
    sirene_legal_units.write_bytes(b"legal-units-snapshot")
    sirene_hashes = {
        "sirene_establishments": registry_lib.sha256(sirene_establishments),
        "sirene_legal_units": registry_lib.sha256(sirene_legal_units),
    }
    full_audit = {
        "schema_version": "sireto-synthetic-gt-full-sirene-audit-1",
        "run_id": "run-P000",
        "ledger_sha256": registry_lib.sha256(ledger_path),
        "source_hashes": sirene_hashes,
        "positive_injection": False,
        "qualification_uses_retrieval_or_model_scores": False,
        "rows": [{
            "seed_id": seed["seed_id"], "variant_id": "v1",
            "variant_promotable_exact": True,
            "full_sirene_qualification": qualification,
        }],
    }
    _write_json(full_audit_path, full_audit)
    promotion_manifest = {
        "schema_version": "sireto-synthetic-gt-full-exact-promotion-2",
        "run_id": "run-P000",
        "promotion_mode": "per-variant",
        "exact_witness": "G_N_A",
        "positive_injection": False,
        "qualification_uses_retrieval_or_model_scores": False,
        "promoted_variants": 1,
        "promoted_sha256": registry_lib.sha256(promoted_path),
        "source_hashes": {
            "seed_input": registry_lib.sha256(seed_path),
            "db": registry_lib.sha256(ledger_path),
            "full_audit": registry_lib.sha256(full_audit_path),
        },
    }
    _write_json(promotion_manifest_path, promotion_manifest)
    registry_path = artifacts / "production_registry.json"
    registry_lib.register(argparse.Namespace(
        registry=registry_path,
        target=1,
        batch_id="P000",
        seed_input=seed_path,
        promoted=promoted_path,
        promotion_manifest=promotion_manifest_path,
    ))

    crm_path = tmp_path / "crm.csv"
    pd.DataFrame([
        {
            "crm_name": "REAL TRAIN", "crm_adresse": "2 RUE REELLE", "crm_cp": "69001",
            "crm_commune": "LYON", "crm_insee": "69123", "gt_siret": "98765432100018",
            "sirene_etat": "A",
        },
        {
            "crm_name": "REAL DEV", "crm_adresse": "3 RUE REELLE", "crm_cp": "69002",
            "crm_commune": "LYON", "crm_insee": "69123", "gt_siret": "98765432200015",
            "sirene_etat": "F",
        },
    ]).to_csv(crm_path, sep=";", index=False)
    folds_path = tmp_path / "folds.parquet"
    pd.DataFrame([
        {
            "query_id": "0", "siren_component_id": "987654321",
            "oof_fold": 2, "legacy_split": "train",
        },
        {
            "query_id": "1", "siren_component_id": "987654322",
            "oof_fold": 0, "legacy_split": "train",
        },
    ]).to_parquet(folds_path, index=False)
    corpus_plan_path = tmp_path / "corpus_plan.json"
    _write_json(corpus_plan_path, {
        "sources": {
            "crm_ok_gt": {"path": str(crm_path), "sha256": registry_lib.sha256(crm_path), "row_count": 2},
            "fold_assignments": {"path": str(folds_path), "sha256": registry_lib.sha256(folds_path)},
            "sirene_establishments": {"path": str(sirene_establishments), "sha256": sirene_hashes["sirene_establishments"]},
            "sirene_legal_units": {"path": str(sirene_legal_units), "sha256": sirene_hashes["sirene_legal_units"]},
        },
        "generator": {"allowed_oof_folds": [2, 3, 4], "allowed_legacy_split": "train"},
        "population": {
            "expected_joined_rows": 2,
            "allowed_rows": 1,
            "allowed_by_fold": {"2": 1},
            "allowed_components": 1,
            "allowed_sirens": 1,
            "forbidden_oof_folds": [0, 1],
            "forbidden_legacy_splits": ["dev", "test"],
            "expected_target_state_counts": {"A": 1},
        },
    })
    balanced_plan_path = tmp_path / "balanced_plan.json"
    _write_json(balanced_plan_path, {"objective": {"promoted_variant_target": 1}})
    return argparse.Namespace(
        registry=registry_path,
        corpus_plan=corpus_plan_path,
        balanced_plan=balanced_plan_path,
        realism_review=None,
        realism_sample_size=1,
        realism_salt="TEST-SALT",
        output=tmp_path / "audit-pending",
        full_audit=full_audit_path,
    )


def _sample_row(index: int, difficulty: str, stratum: str, name_relation: str, location_relation: str) -> dict:
    return {
        "batch_id": "P000", "seed_id": f"seed-{index}", "variant_id": "v1",
        "target_siret": f"{index:014d}", "difficulty": difficulty,
        "augmentation_stratum": stratum, "name_relation": name_relation,
        "location_field": "address", "location_relation": location_relation,
        "official_baseline": {"names": [f"NAME {index}"], "address": f"{index} RUE TEST"},
        "crm": {"name": f"NAME {index}", "address": f"{index} R TEST"},
        "contract": {"field_inspirations": {}}, "transformation_summary": "test",
    }


def test_stratified_sample_is_bounded_deterministic_and_covers_fine_cells() -> None:
    rows = [
        _sample_row(1, "EASY", "TRAIN", "ALIAS", "ABBREVIATE"),
        _sample_row(2, "EASY", "TRAIN", "ALIAS", "SUBSET"),
        _sample_row(3, "HARD", "FAIL", "ORDER", "ABBREVIATE"),
        _sample_row(4, "HARD", "FAIL", "ORDER", "SUBSET"),
        _sample_row(5, "HARD", "FAIL", "ORDER", "SUBSET"),
    ]
    first, dimensions = audit.stratified_realism_sample(rows, 4, "salt")
    second, _ = audit.stratified_realism_sample(list(reversed(rows)), 4, "salt")
    assert dimensions == [
        "difficulty", "augmentation_stratum", "name_relation", "location_relation"
    ]
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert len(first) == 4
    assert len({
        (row["difficulty"], row["augmentation_stratum"], row["name_relation"], row["location_relation"])
        for row in first
    }) == 4
    assert len({row["seed_id"] for row in first}) == len(first)


def test_stratified_sample_keeps_only_one_surface_per_seed() -> None:
    rows = [
        _sample_row(1, "EASY", "TRAIN", "ALIAS", "ABBREVIATE"),
        {
            **_sample_row(2, "EASY", "TRAIN", "ALIAS", "SUBSET"),
            "seed_id": "seed-1", "variant_id": "v2",
        },
        _sample_row(3, "HARD", "FAIL", "ORDER", "SUBSET"),
    ]
    sample, _ = audit.stratified_realism_sample(rows, 3, "fresh-v2")
    assert len(sample) == 2
    assert len({row["seed_id"] for row in sample}) == 2


def test_stratified_sample_excludes_every_seed_from_prior_sample(tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.jsonl"
    _write_jsonl(prior_path, [{
        "schema_version": audit.SAMPLE_SCHEMA_VERSION,
        "sample_id": "a" * 64,
        "seed_id": "seed-1",
    }])
    excluded, provenance = audit.excluded_realism_seeds(prior_path)
    rows = [
        _sample_row(1, "EASY", "TRAIN", "ALIAS", "ABBREVIATE"),
        _sample_row(2, "EASY", "TRAIN", "ALIAS", "SUBSET"),
        _sample_row(3, "HARD", "FAIL", "ORDER", "SUBSET"),
    ]
    sample, _ = audit.stratified_realism_sample(rows, 2, "fresh-v2", excluded)
    assert {row["seed_id"] for row in sample} == {"seed-2", "seed-3"}
    assert provenance["excluded_prior_sample_rows"] == 1
    assert provenance["excluded_prior_sample_seed_ids"] == 1
    assert provenance["excluded_prior_sample_sha256"] == registry_lib.sha256(prior_path)


def test_realism_review_is_exact_and_pauses_only_at_two_certain() -> None:
    sample = [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}]
    reviews = [
        {"schema_version": audit.REVIEW_SCHEMA_VERSION, "sample_id": "a", "decision": "CERTAIN_FALSE_REALISM", "reason": "malformed"},
        {"schema_version": audit.REVIEW_SCHEMA_VERSION, "sample_id": "b", "decision": "BORDERLINE", "reason": "telegraphic"},
        {"schema_version": audit.REVIEW_SCHEMA_VERSION, "sample_id": "c", "decision": "PASS", "reason": ""},
    ]
    assert audit.realism_review_summary(sample, reviews)["verdict"] == "PASS"
    reviews[1] = {
        "schema_version": audit.REVIEW_SCHEMA_VERSION, "sample_id": "b",
        "decision": "CERTAIN_FALSE_REALISM", "reason": "empty identity",
    }
    assert audit.realism_review_summary(sample, reviews)["verdict"] == "PAUSE_DOWNSTREAM"
    with pytest.raises(ValueError, match="cover"):
        audit.realism_review_summary(sample, reviews[:-1])


def test_exact_and_operational_views_are_never_merged() -> None:
    rows = [
        {"full_sirene_qualification": {"decision": "EXACT_IDENTIFIABLE", "operational_equivalence": True}},
        {"full_sirene_qualification": {"decision": "AMBIGUOUS_OFFICIAL", "operational_equivalence": True}},
    ]
    result = audit.qualification_summary(rows)
    assert result["exact_identifiable_rows"] == 1
    assert result["rows_with_operational_equivalent_alternative"] == 2
    assert result["operational_only_rows"] == 1
    assert result["views_are_separate"] is True


def test_real_rows_exclude_train_identity_connected_to_forbidden_rows(tmp_path: Path) -> None:
    crm_path = tmp_path / "crm.csv"
    pd.DataFrame([
        {
            "crm_name": "SAFE", "crm_adresse": "1 RUE A", "crm_cp": "75001",
            "crm_commune": "PARIS", "crm_insee": "75056",
            "gt_siret": "11111111100011", "sirene_etat": "A",
        },
        {
            "crm_name": "SIREN LEAK", "crm_adresse": "2 RUE A", "crm_cp": "75001",
            "crm_commune": "PARIS", "crm_insee": "75056",
            "gt_siret": "22222222200012", "sirene_etat": "A",
        },
        {
            "crm_name": "COMPONENT LEAK", "crm_adresse": "3 RUE A", "crm_cp": "75001",
            "crm_commune": "PARIS", "crm_insee": "75056",
            "gt_siret": "33333333300013", "sirene_etat": "F",
        },
        {
            "crm_name": "DEV SAME SIREN", "crm_adresse": "4 RUE A", "crm_cp": "75001",
            "crm_commune": "PARIS", "crm_insee": "75056",
            "gt_siret": "22222222200020", "sirene_etat": "A",
        },
        {
            "crm_name": "TEST SAME COMPONENT", "crm_adresse": "5 RUE A", "crm_cp": "75001",
            "crm_commune": "PARIS", "crm_insee": "75056",
            "gt_siret": "44444444400014", "sirene_etat": "F",
        },
    ]).to_csv(crm_path, sep=";", index=False)
    folds_path = tmp_path / "folds.parquet"
    pd.DataFrame([
        {"query_id": "0", "siren_component_id": "safe", "oof_fold": 2, "legacy_split": "train"},
        {"query_id": "1", "siren_component_id": "train-siren", "oof_fold": 2, "legacy_split": "train"},
        {"query_id": "2", "siren_component_id": "shared", "oof_fold": 3, "legacy_split": "train"},
        {"query_id": "3", "siren_component_id": "dev", "oof_fold": 0, "legacy_split": "dev"},
        {"query_id": "4", "siren_component_id": "shared", "oof_fold": 1, "legacy_split": "test"},
    ]).to_parquet(folds_path, index=False)
    plan = {
        "generator": {"allowed_oof_folds": [2, 3, 4], "allowed_legacy_split": "train"},
        "population": {
            "expected_joined_rows": 5,
            "allowed_rows": 1,
            "allowed_by_fold": {"2": 1},
            "allowed_components": 1,
            "allowed_sirens": 1,
            "forbidden_oof_folds": [0, 1],
            "forbidden_legacy_splits": ["dev", "test"],
            "expected_target_state_counts": {"A": 1},
        },
    }

    real_all, real_train = audit._real_rows(crm_path, folds_path, plan)

    assert len(real_all) == 5
    assert [row["query_id"] for row in real_train] == ["0"]


def test_final_audit_reuses_sealed_full_sirene_and_includes_real_distribution(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    output = audit.run(args)
    report = json.loads((output / "report.json").read_text())
    assert report["promoted_variants"] == 1
    assert report["qualification"]["exact_identifiable_rows"] == 1
    assert report["distribution"]["real_all"]["rows"] == 2
    assert report["distribution"]["real_train_folds_2_3_4"]["rows"] == 1
    assert report["distribution"]["raw_available_train_union"]["rows"] == 2
    assert report["realism_audit"]["status"] == "PENDING_BOUNDED_REVIEW"
    assert report["final_status"] == "PENDING_BOUNDED_REALISM_REVIEW"
    assert report["deterministic_invariants"]["all_promoted_rows_full_sirene_exact"] is True


def test_completed_review_is_bound_to_the_same_sample(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    pending = audit.run(args)
    sample = [json.loads(line) for line in (pending / "realism_sample.jsonl").read_text().splitlines()]
    review_path = tmp_path / "review.jsonl"
    _write_jsonl(review_path, [{
        "schema_version": audit.REVIEW_SCHEMA_VERSION,
        "sample_id": sample[0]["sample_id"],
        "decision": "PASS",
        "reason": "Surface CRM réaliste.",
    }])
    args.realism_review = review_path
    args.output = tmp_path / "audit-final"
    final = audit.run(args)
    report = json.loads((final / "report.json").read_text())
    assert report["final_status"] == "PASS"
    assert report["realism_audit"]["decision_counts"]["PASS"] == 1


def test_incomplete_registry_fails_before_hashing_large_sources(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    registry = json.loads(args.registry.read_text())
    registry["promoted_variant_target"] = 2
    registry["summary"]["promoted_variants"] = 1
    _write_json(args.registry, registry)
    _write_json(args.balanced_plan, {"objective": {"promoted_variant_target": 2}})
    with pytest.raises(ValueError, match="post-production only"):
        audit.run(args)


def test_final_audit_rejects_mutated_full_sirene_sidecar(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.full_audit.write_text(args.full_audit.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="full-SIRENE audit hash mismatch"):
        audit.run(args)
