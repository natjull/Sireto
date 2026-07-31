from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = REPOSITORY / "config/v4_13_fresh_labels_minimal_plan.json"
SCHEMA_PATH = (
    REPOSITORY / "config/v4_13_fresh_labels_minimal_plan.schema.json"
)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )


def compact_canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def load_plan() -> dict:
    raw = PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    assert raw == canonical_json(plan)
    return plan


def load_schema() -> dict:
    raw = SCHEMA_PATH.read_bytes()
    schema = json.loads(raw)
    assert raw == canonical_json(schema)
    return schema


def validate_exact_object_keys(value: object, schema: dict, path: str = "/") -> None:
    if isinstance(value, dict):
        assert path in schema["exact_object_keys"], f"unregistered object {path}"
        assert sorted(value) == schema["exact_object_keys"][path]
        for key, child in value.items():
            child_path = path.rstrip("/") + "/" + key
            validate_exact_object_keys(child, schema, child_path)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                validate_exact_object_keys(child, schema, path + "/*")


def test_plan_is_recursively_closed_and_canonical() -> None:
    plan = load_plan()
    schema = load_schema()
    assert schema["target_schema_version"] == plan["schema_version"]
    assert schema["additional_object_fields"] == "REJECT"
    validate_exact_object_keys(plan, schema)


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_closed_schema_rejects_nested_key_mutations(mutation: str) -> None:
    plan = copy.deepcopy(load_plan())
    if mutation == "extra":
        plan["retrieval"]["frozen_policy"]["unexpected"] = True
    else:
        del plan["source_protocol"]["manifest_only_0a"]["selection"]
    with pytest.raises(AssertionError):
        validate_exact_object_keys(plan, load_schema())


def test_contract_schema_and_frozen_historical_authorities_are_pinned() -> None:
    plan = load_plan()
    for pin in (plan["contract"], plan["schema"]):
        assert hashlib.sha256((REPOSITORY / pin["path"]).read_bytes()).hexdigest() == (
            pin["sha256"]
        )
    historical = plan["retrieval"]["historical_contract"]
    assert historical == {
        "git_commit": "eb0e6a3ca034a0b1e78ae77e5bde780608a836d7",
        "path": "docs/retrieval_selective_recall100_contract.md",
        "sha256": "51e078610441644d582b0d83c631e26119134f36ad8d4bf559e92df4a4aaecf1",
    }
    assert plan["retrieval"]["historical_evaluator"] == {
        "git_commit": "5a0e67f",
        "path": "scripts/evaluate_retrieval_admission.py",
        "sha256": "b24ee3f52ab5d713c92114ac13d3b1e99498bb40a3ca6cca015bb991dd237c45",
    }


def test_existing_registry_payloads_and_all_keysets_are_exact() -> None:
    plan = load_plan()
    for registry in plan["registries"].values():
        manifest = Path(registry["root"]) / "manifest.json"
        if not manifest.exists():
            manifest = Path(registry["root"]) / "payload_manifest.json"
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
            registry["manifest_sha256"]
        )
    compatibility_root = Path(plan["registries"]["consumed_compatibility"]["root"])
    assert set(plan["contamination"]["compatibility_keysets"]) == {
        "service_id",
        "input_siret_lineage",
        "siret_masked",
        "fuzzy_historical",
    }
    for pin in plan["contamination"]["compatibility_keysets"].values():
        assert hashlib.sha256(
            (compatibility_root / pin["file"]).read_bytes()
        ).hexdigest() == pin["sha256"]
    comparison = plan["contamination"]["comparison"]
    assert comparison["maximum_hit_count_each_keyset"] == 0
    assert comparison["maximum_hit_count_keyset_union"] == 0
    assert comparison["maximum_hit_count_siren_registry"] == 0
    assert comparison["all_authoritatively_known_sirens_included"] is True


def test_historical_hmac_is_read_only_and_not_a_new_pki() -> None:
    policy = load_plan()["contamination"]["keychain_read_only"]
    assert policy == {
        "account": "SIRETO",
        "allowed_api": "SecItemCopyMatching",
        "authentication_ui": "FAIL",
        "create_update_delete_forbidden": True,
        "key_id": "SIRETO_V412_COMPATIBILITY_LINEAGE_HMAC_V1",
        "key_sha256": "639ea96ba64008c7ec8c2cd69dce39f0c02c0d0d1e802b42d3d0bf02a707c4fe",
        "secret_output_forbidden": True,
        "service": "com.sireto.v412.compatibility-hmac",
    }


def test_availability_and_payload_open_are_distinct_one_shot_gates() -> None:
    plan = load_plan()
    source = plan["source_protocol"]
    assert source["manifest_only_0a"]["payload_open_forbidden"] is True
    assert source["manifest_only_0a"]["selection"] == (
        "MINIMUM_ARRIVAL_EPOCH_NS_THEN_MANIFEST_SHA256"
    )
    assert source["payload_open_0b"][
        "marker_created_o_excl_before_first_payload_fd"
    ] is True
    assert source["payload_open_0b"]["second_open_after_crash_forbidden"] is True
    assert plan["implementation"]["status"] == "UNIMPLEMENTED_NOT_AUTHORIZED"
    assert plan["gates"]["preregistration"]["independent_go_required"] == 2
    assert plan["gates"]["implementation"]["independent_go_required"] == 2


def test_authority_mapping_cannot_use_similarity_or_self_assert_truth() -> None:
    plan = load_plan()
    mapping = plan["artifact_schemas"]["authoritative_mapping"]
    assert mapping["join"] == "EXACT_SOURCE_RECORD_ID_ONLY"
    assert mapping["types"]["matching_pipeline_used"] == "boolean_false"
    assert not {
        "crm_name_raw",
        "crm_address_raw",
        "crm_postcode_raw",
        "crm_city_raw",
        "crm_insee_raw",
    } & set(mapping["columns_in_order"])
    forbidden = set(plan["authority_catalog"]["forbidden_truth_creators"])
    assert {
        "NAME_ADDRESS_SIMILARITY",
        "SIRENE_ALONE",
        "RETRIEVAL",
        "MODEL_OUTPUT",
        "LLM",
        "USER_VALIDATION",
    } <= forbidden
    assert set(plan["authority_catalog"]["allowed"]) == {
        "SOURCE_SYSTEM_OFFICIAL_SIRET",
        "CONTRACT_OR_BILLING_SIRET",
        "SEALED_ADMINISTRATIVE_DOCUMENT",
    }


def test_opaque_id_split_and_oof_have_frozen_vectors() -> None:
    projection = ["f" * 64, "e" * 64, 1, "CRM-42"]
    digest = hashlib.sha256(
        b"SIRETO-V413-OPAQUE-QUERY-ID\0" + compact_canonical_json(projection)
    ).hexdigest()
    opaque = "".join(chr(ord("a") + int(nibble, 16)) for nibble in digest)
    assert opaque == (
        "hchckchghiebinmcgfdfbaifncohacngafphapcneafambbgeheamcplanppofhj"
    )

    component = ["a" * 64, "b" * 64]
    component_key = compact_canonical_json(component)
    split_digest = hashlib.sha256(
        b"SIRETO-V413-FRESH-SPLIT\0" + component_key
    ).hexdigest()
    split_uint64 = int.from_bytes(bytes.fromhex(split_digest)[:8], "big")
    assert split_digest == (
        "29a3f72efed27efc4dc1bef9ba5b5d75a3baa43fc9f93490c861bf2134d5a958"
    )
    assert split_uint64 == 3000513557974646524
    split = load_plan()["split"]
    assert split_uint64 < split["fit_upper_exclusive_uint64"]
    assert split["fit_upper_exclusive_uint64"] == 12912720851596686131
    assert split["dev_upper_exclusive_uint64"] == 15679732462653118873

    fold_digest = hashlib.sha256(
        b"SIRETO-V413-RANKER-OOF-FOLD\0" + component_key
    ).digest()
    assert int.from_bytes(fold_digest[:8], "big") % 5 == 3


def test_retrieval_model_and_final_test_gates_are_closed() -> None:
    plan = load_plan()
    retrieval = plan["retrieval"]
    assert retrieval["candidate_maximum_absolute"] == 100
    assert retrieval["truth_absent_is_error"] is True
    assert retrieval["fit_execution_count_maximum"] == 1
    assert retrieval["dev_execution_count_maximum"] == 1
    assert retrieval["frozen_policy"] == {
        "overlay_quotas": {"name_char": 10, "name_word": 1},
        "rrf_constant": 60,
        "tie_break": "SIRET_ASCENDING",
        "weights": {
            "address_exact": 2.0,
            "address_word": 0.5,
            "current_sparse": 2.0,
            "name_char": 1.0,
            "name_exact": 2.0,
            "name_word": 1.0,
            "siren_head": 1.0,
        },
    }
    assert plan["model"]["oof"]["folds"] == 5
    assert plan["model"]["acceptor_training"].endswith("OOF_ONLY")
    assert plan["model"]["ambiguous_or_unresolved_auto_maximum"] == 0
    assert plan["model"]["threshold_selection"]["minimum_auto"] == (
        "MAX_50_OR_CEIL_25_PERCENT_DEV_ROWS"
    )
    final = plan["final_test"]
    assert final["events_in_order"] == [
        "OPENING_O_EXCL_BEFORE_FIRST_TEST_QUERY_FD",
        "CANDIDATES_AND_DECISIONS_SEALED",
        "ORACLE_OPEN",
        "METRICS_SEALED",
        "TERMINAL_RECEIPT",
    ]
    assert final["rescoring_allowed"] is False
    assert final["terminal_verdicts"] == ["GO", "PIVOT", "STOP"]
    assert final["gate"]["minimum_identifiable_coverage"] == 0.8
    assert final["gate"]["minimum_exact_siret_recall_at_100"] == 0.99
    assert final["gate"]["minimum_auto_exact_siret_precision_observed"] == 0.998


def test_no_source_observation_or_implementation_is_smuggled_into_plan() -> None:
    plan = load_plan()
    assert "current_evidence" not in plan
    assert plan["status"] == "PREREGISTRATION_AMENDMENT_AWAITING_TWO_GO"
    for path in plan["implementation"]["components"].values():
        assert path.startswith("scripts/")
    assert plan["implementation"]["tests"] == "tests/test_v413_fresh_intake.py"
    assert plan["preregistration_lock"]["status"] == "NOT_CREATED"
