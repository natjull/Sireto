from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    REPOSITORY
    / "config/v4_12_fresh_s1_local_producer_authority_plan.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
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


def _plan() -> dict:
    raw = PLAN_PATH.read_bytes()
    value = json.loads(raw)
    assert raw == _canonical(value)
    return value


def test_local_producer_plan_is_canonical_and_pins_authorities() -> None:
    plan = _plan()
    assert plan["status"] == (
        "PREREGISTERED_DO_NOT_IMPLEMENT_UNTIL_TWO_INDEPENDENT_AUDITS"
    )
    for role in ("contract", "s1_contract", "s1_plan"):
        authority = plan["authorities"][role]
        assert _sha(REPOSITORY / authority["path"]) == authority["sha256"]
    for role in ("s1_authoritative_commit", "s1_certification_commit"):
        oid = plan["authorities"][role]
        assert len(oid) == 40
        subprocess.run(
            ["git", "cat-file", "-e", f"{oid}^{{commit}}"],
            cwd=REPOSITORY,
            check=True,
        )


def test_local_producer_identity_is_fixed_and_pre_crm() -> None:
    plan = _plan()
    assert plan["producer"] == {
        "producer_id": "SIRETO_LOCAL_CRM_EXPORT_PRODUCER_V1",
        "source_system": "LOCAL_CRM_EXPORT_V1",
        "portfolio_id": "SIRETO_FRESH_HOLDOUT_V1",
        "source_record_id_semantics": (
            "V411_SERVICE_ID_NORM_EQUIVALENT_SOURCE_RECORD_ID_V1"
        ),
        "producer_key_id": (
            "SIRETO_V412_FRESH_S1_LOCAL_PRODUCER_ED25519_V1"
        ),
        "producer_export_ledger_id": (
            "SIRETO_V412_FRESH_S1_LOCAL_EXPORT_LEDGER_V1"
        ),
        "next_expected_export_sequence": 1,
    }
    assert set(plan["forbidden_inputs"]) >= {
        "CRM",
        "EVIDENCE",
        "LABEL",
        "ORACLE",
        "CANDIDATE",
        "RETRIEVAL",
        "MODEL",
    }


def test_private_key_can_only_live_in_keychain() -> None:
    keychain = _plan()["keychain"]
    assert keychain["secret_length_bytes"] == 32
    assert keychain["create_api"] == "SECITEMADD_IN_PROCESS"
    assert keychain["read_api"] == "SECITEMCOPYMATCHING_IN_PROCESS"
    assert keychain["authentication_ui"] == "FAIL"
    assert keychain["private_key_export_allowed"] is False
    assert keychain["binding_attribute"] == {
        "keychain_attribute": "kSecAttrGeneric",
        "value": "RAW_32_BYTE_SHA256_OF_EXACT_CLAIM_BYTES",
        "verified_on_every_read": True,
    }
    assert keychain["secitemadd_dictionary_exact"] == {
        "kSecClass": "kSecClassGenericPassword",
        "kSecAttrService": keychain["service"],
        "kSecAttrAccount": keychain["account"],
        "kSecAttrLabel": (
            "SIRETO_V412_FRESH_S1_LOCAL_PRODUCER_ED25519_V1"
        ),
        "kSecAttrGeneric": "RAW_32_BYTE_SHA256_OF_EXACT_CLAIM_BYTES",
        "kSecAttrSynchronizable": False,
        "kSecAttrAccessible": (
            "kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly"
        ),
        "kSecUseDataProtectionKeychain": True,
        "kSecValueData": "RAW_32_BYTE_ED25519_SEED",
    }
    assert keychain["secitemadd_dictionary_extra"] == "REJECT"
    assert keychain["secitemcopymatching_query_exact"] == {
        "kSecClass": "kSecClassGenericPassword",
        "kSecAttrService": keychain["service"],
        "kSecAttrAccount": keychain["account"],
        "kSecAttrSynchronizable": False,
        "kSecUseDataProtectionKeychain": True,
        "kSecUseAuthenticationUI": "kSecUseAuthenticationUIFail",
        "kSecReturnData": True,
        "kSecReturnAttributes": True,
        "kSecMatchLimit": "kSecMatchLimitOne",
    }
    assert keychain["secitemcopymatching_query_extra"] == "REJECT"
    result_policy = keychain["secitemcopymatching_result_policy"]
    assert result_policy["result_type"] == "CFDICTIONARY"
    assert result_policy["missing_required_key"] == "STOP"
    assert result_policy["required_projection_mismatch"] == "STOP"
    assert result_policy["all_nonprojected_returned_keys"] == (
        "IGNORE_WITHOUT_SERIALIZE_LOG_OR_DECISION"
    )
    expected_returned = dict(keychain["secitemadd_dictionary_exact"])
    expected_returned.pop("kSecUseDataProtectionKeychain")
    assert result_policy[
        "required_persisted_attributes_verified_exactly"
    ] == expected_returned
    assert set(keychain["forbidden_secret_channels"]) == {
        "ARGV",
        "ENVIRONMENT",
        "STDIN",
        "TEMP_FILE",
        "LOG",
        "RECEIPT",
        "MANIFEST",
        "SECURITY_CLI",
    }


def test_schemas_are_closed_and_genesis_signature_is_non_recursive() -> None:
    schemas = _plan()["schemas"]
    for schema in schemas.values():
        assert schema["exact_fields"]
        assert set(schema["nullable"]) <= set(schema["exact_fields"])
        if "types" in schema:
            assert set(schema["types"]) == set(schema["exact_fields"])
    genesis = schemas["ledger_genesis"]
    assert genesis["signed_projection_excludes"] == ["signature_base64"]
    assert genesis["types"]["producer_export_sequence"] == "integer_zero"
    assert genesis["types"]["producer_export_previous_entry_sha256"] == "null"
    assert "private" not in " ".join(
        schemas["payload"]["exact_fields"]
        + schemas["seal"]["exact_fields"]
        + schemas["receipt"]["exact_fields"]
    ).lower()
    expected_nested_types = {
        "producer": "producer_object",
        "keychain_locator": "keychain_locator_object",
        "s1_authorities": "s1_authorities_object",
        "implementation": "implementation_pin_object",
        "runtime": "runtime_pin_object",
    }
    for field, schema_name in expected_nested_types.items():
        assert schemas["payload"]["types"][field] == schema_name
    for schema_name in ("keychain_locator_object", "keychain_policy_object"):
        schema = schemas[schema_name]
        assert "data_protection_keychain" in schema["exact_fields"]
        assert schema["types"]["data_protection_keychain"] == "boolean_true"


def test_claim_binds_lock_authorization_nonce_and_keychain_item() -> None:
    plan = _plan()
    claim = plan["schemas"]["claim"]
    assert claim["exact_fields"] == [
        "schema_version",
        "plan_sha256",
        "execution_lock_sha256",
        "authorization_sha256",
        "attempt_binding_nonce_base64",
        "attempt_binding_nonce_sha256",
        "logical_time_utc",
        "claim_state",
    ]
    assert claim["types"]["attempt_binding_nonce_base64"] == (
        "base64_random_32_bytes"
    )
    assert claim["types"]["attempt_binding_nonce_sha256"] == "sha256"
    assert claim["types"]["execution_lock_sha256"] == "sha256"
    assert claim["types"]["authorization_sha256"] == "sha256"
    assert "key_intent" not in plan["paths"]
    assert "key_intent" not in plan["schemas"]


def test_one_shot_recovery_distinguishes_foreign_and_owned_items() -> None:
    one_shot = _plan()["one_shot"]
    assert one_shot["claim_create"] == (
        "O_EXCL_DURABLE_WITH_LOCK_AUTHORIZATION_AND_RANDOM_NONCE_"
        "BEFORE_ANY_KEYCHAIN_ACCESS"
    )
    assert one_shot["foreign_item_without_claim"] == (
        "STOP_FOREIGN_KEYCHAIN_ITEM"
    )
    assert one_shot["new_claim_item_present"] == (
        "STOP_FOREIGN_KEYCHAIN_ITEM"
    )
    assert one_shot["existing_claim_no_receipt_item_absent"] == (
        "CREATE_NEW_SEED_AND_SECITEMADD_BOUND_TO_CLAIM_SHA256"
    )
    assert one_shot["existing_claim_no_receipt_item_binding_matches"] == (
        "READ_KEY_AND_CONTINUE_SAME_ATTEMPT"
    )
    assert one_shot[
        "existing_claim_no_receipt_item_binding_absent_or_mismatch"
    ] == "STOP_FOREIGN_KEYCHAIN_ITEM"
    assert one_shot["secitemadd_duplicate"] == "STOP_FOREIGN_KEYCHAIN_ITEM"
    assert one_shot["valid_receipt_present"] == (
        "RETURN_IDEMPOTENT_WITHOUT_SECRET_ACCESS"
    )
    assert one_shot["rerun_new_attempt_allowed"] is False


def test_provisioning_remains_locked_and_root_absent() -> None:
    plan = _plan()
    assert plan["future_implementation"]["provisioner"]["sha256"] == (
        "UNIMPLEMENTED"
    )
    assert plan["future_implementation"]["tests"]["sha256"] == "UNIMPLEMENTED"
    assert plan["gates"]["implementation"]["verdict"] == (
        "GO_S1_LOCAL_PRODUCER_IMPLEMENTATION"
    )
    assert plan["gates"]["provision"]["verdict"] == (
        "GO_S1_LOCAL_PRODUCER_PROVISION"
    )
    sequence = plan["sequence"]
    assert sequence.index("COMMIT_ONE_SHOT_AUTHORIZATION") < sequence.index(
        "TWO_GO_S1_LOCAL_PRODUCER_PROVISION_AUDITS"
    )
    assert sequence.index(
        "TWO_GO_S1_LOCAL_PRODUCER_PROVISION_AUDITS"
    ) < sequence.index("RUN_ONCE")
    assert "AUTHORIZATION_COMMITTED" in plan["gates"]["provision"]["requires"]
    assert not Path(plan["paths"]["root"]).exists()
    assert not (REPOSITORY / plan["paths"]["authorization"]).exists()
    assert not (REPOSITORY / plan["paths"]["execution_lock"]).exists()
