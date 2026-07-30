from __future__ import annotations

import hashlib
import json
from pathlib import Path


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


def test_one_shot_recovery_distinguishes_foreign_and_owned_items() -> None:
    one_shot = _plan()["one_shot"]
    assert one_shot["claim_create"] == (
        "O_EXCL_DURABLE_BEFORE_ANY_KEYCHAIN_ACCESS"
    )
    assert one_shot["foreign_item_without_claim"] == (
        "STOP_FOREIGN_KEYCHAIN_ITEM"
    )
    assert one_shot["claim_no_receipt_item_absent"] == (
        "CREATE_KEY_AND_CONTINUE_SAME_ATTEMPT"
    )
    assert one_shot["claim_no_receipt_item_present"] == (
        "READ_KEY_AND_CONTINUE_SAME_ATTEMPT"
    )
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
    assert not Path(plan["paths"]["root"]).exists()
    assert not (REPOSITORY / plan["paths"]["authorization"]).exists()
    assert not (REPOSITORY / plan["paths"]["execution_lock"]).exists()
