from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = REPOSITORY / "config/v4_12_fresh_s1_plan.json"


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


def test_s1_preregistration_is_canonical_and_pins_authorities() -> None:
    plan = _plan()
    assert plan["status"] == (
        "PREREGISTERED_S1_DO_NOT_IMPLEMENT_UNTIL_TWO_INDEPENDENT_AUDITS"
    )
    for role in (
        "contract",
        "base_contract",
        "base_plan",
        "development_inputs",
        "forbidden_artifacts",
    ):
        authority = plan["authorities"][role]
        assert _sha(REPOSITORY / authority["path"]) == authority["sha256"]
    receipt = plan["authorities"]["r3_receipt"]
    assert _sha(Path(receipt["path"])) == receipt["sha256"]
    value = json.loads(Path(receipt["path"]).read_bytes())
    for field in ("schema_version", "verdict", "reason_code", "terminal_result"):
        assert value[field] == receipt[field]


def test_s1_consumed_sirens_registry_is_materially_pinned() -> None:
    registry = _plan()["registries"]["consumed_sirens"]
    manifest_path = Path(registry["manifest_path"])
    assert _sha(manifest_path) == registry["manifest_sha256"]
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["build_id"] == registry["build_id"]
    assert manifest["tree_payload_sha256"] == registry["logical_payload_sha256"]
    assert (
        manifest["files"]["consumed_sirens.parquet"]["sha256"]
        == registry["consumed_sirens_parquet_sha256"]
    )
    assert registry["available_to"] == ["WORKER_E"]
    assert registry["latest_resolution_allowed"] is False


def test_s1_compatibility_registry_and_keysets_are_materially_pinned() -> None:
    registry = _plan()["registries"]["compatibility"]
    root = Path(registry["root"])
    assert _sha(root / "payload_manifest.json") == (
        registry["payload_manifest_sha256"]
    )
    assert _sha(root / "seal.json") == registry["seal_sha256"]
    payload = json.loads((root / "payload_manifest.json").read_bytes())
    files = {record["relative_path"]: record["sha256"] for record in payload["payload_files"]}
    assert files["service_id_keyset.parquet"] == registry["keysets"]["service_id"]
    assert files["siret_masked_keyset.parquet"] == registry["keysets"]["siret_masked"]
    assert files["fuzzy_historical_keyset.parquet"] == (
        registry["keysets"]["fuzzy_historical"]
    )
    assert files["input_siret_lineage_keyset.parquet"] == (
        registry["keysets"]["input_siret_lineage"]
    )
    provenance = json.loads((root / "provenance.json").read_bytes())
    assert provenance["hmac_key_id"] == registry["hmac"]["key_id"]
    assert provenance["hmac_key_sha256"] == registry["hmac"]["key_sha256"]
    assert registry["available_to"] == ["WORKER_Q"]
    assert registry["latest_resolution_allowed"] is False


def test_s1_worker_boundaries_keep_query_and_truth_physically_separate() -> None:
    boundaries = _plan()["boundaries"]
    worker_q = boundaries["worker_q"]
    worker_e = boundaries["worker_e"]
    scorer = boundaries["scorer"]
    assert "CRM_PAYLOAD" in worker_q["inputs"]
    assert "EVIDENCE_PAYLOAD" in worker_q["forbidden"]
    assert "ORACLE" in worker_q["forbidden"]
    assert "EVIDENCE_PAYLOAD" in worker_e["inputs"]
    assert "PRIVATE_MINIMAL_BRIDGE" in worker_e["inputs"]
    assert {
        "CRM_NAME",
        "CRM_ADDRESS",
        "CRM_POSTCODE",
        "CRM_CITY",
        "CRM_INSEE",
    } <= set(worker_e["forbidden"])
    assert scorer["inputs"] == ["SEALED_SAFE_QUERIES"]
    assert "PRIVATE_ORACLE" in scorer["forbidden"]
    assert "PRIVATE_IDENTITY_BRIDGE" in scorer["forbidden"]


def test_s1_catalogs_are_required_before_any_real_crm_open() -> None:
    plan = _plan()
    required_catalog_fields = {
        "schema_version",
        "catalog_id",
        "builder_commit",
        "tests_sha256",
        "runtime",
        "config_sha256",
        "manifest_sha256",
    }
    for catalog in plan["catalogs"].values():
        assert catalog["status"] == (
            "MUST_BE_BUILT_SEALED_AND_PINNED_BEFORE_REAL_CRM_OPEN"
        )
        assert required_catalog_fields <= set(catalog["exact_fields"])
    evidence = plan["catalogs"]["evidence"]
    assert evidence["similarity_can_create_truth"] is False
    assert evidence["truth_creators"] == [
        "SEALED_SOURCE_RECORD_TO_SIRET_CONTRACT_MAPPING",
        "PREREGISTERED_OFFICIAL_IDENTIFIER",
        "SEALED_ADMINISTRATIVE_DOCUMENT",
    ]


def test_s1_manifests_have_closed_fields_types_and_nullability() -> None:
    schemas = _plan()["schema_contract"]
    assert schemas["extra_fields"] == "REJECT"
    for schema in schemas.values():
        if not isinstance(schema, dict):
            continue
        assert set(schema["nullable"]) <= set(schema["fields"])
        assert set(schema["types"]) <= set(schema["fields"])
    source = schemas["source_manifest"]
    assert source["types"]["v411_service_id_equivalence_attested"] == (
        "boolean_exact"
    )
    assert source["types"]["lineage_attestation_reference"] == "string_nonempty"


def test_s1_admission_claims_before_payload_and_is_one_shot() -> None:
    plan = _plan()
    admission = plan["boundaries"]["admission"]
    assert "CRM_PAYLOAD" in admission["forbidden"]
    assert admission["creates_before_payload_open"] == [
        "GLOBAL_SELECTION_CLAIM",
        "ARRIVAL_RECEIPT",
        "DYNAMIC_COLLECTION_LOCK",
    ]
    one_shot = plan["one_shot"]
    assert one_shot["global_claim_create"] == "O_EXCL_BEFORE_ANY_PAYLOAD_OPEN"
    assert one_shot["claim_without_receipt_after_possible_open"] == (
        "STOP_NO_RERUN"
    )
    assert one_shot["recovery_source"] == "SEALED_TREES_ONLY_NEVER_INBOX"
    assert one_shot["second_collection_allowed"] is False


def test_s1_gate_is_synthetic_exhaustive_and_blocks_real_crm() -> None:
    gate = _plan()["required_gate"]
    assert gate["fixture"]["batch_count_minimum"] >= 3
    assert gate["fixture"]["portfolio_count_minimum"] >= 2
    assert gate["fixture"]["rows_minimum"] > gate["fixture"]["late_error_after_row"]
    assert gate["test_families"] == [
        "AUTHORITIES_AND_PREOPEN",
        "SCHEMAS_AND_FDS",
        "CONTAMINATION_AND_QUALIFICATION",
        "EXHAUSTIVENESS",
        "PII_AND_DIAGNOSTICS",
        "CRASH_AND_RECOVERY",
        "READY_AND_ONE_SHOT",
    ]
    assert gate["independent_audits_required"] == 2
    assert gate["required_verdict"] == "GO_S1_IMPLEMENTATION"
    assert gate["real_crm_open_before_gate"] is False


def test_s1_north_star_and_model_freeze_are_preserved() -> None:
    plan = _plan()
    metrics = plan["metrics"]
    assert metrics["coverage"] == {
        "numerator": "MATCH_EXACT",
        "denominator": "ALL_SOURCE_ROWS",
        "minimum": 0.8,
        "match_exact_minimum": 657,
    }
    assert metrics["retrieval"]["candidate_maximum_absolute"] == 100
    assert metrics["retrieval"]["recall_siret_exact_at_100_minimum"] == 0.99
    assert metrics["retrieval"]["truth_absent_is_miss"] is True
    assert metrics["publish_together"] == ["historical", "V2", "V3", "fresh_holdout"]
    forbidden = (
        set(plan["boundaries"]["worker_q"]["forbidden"])
        | set(plan["boundaries"]["worker_e"]["forbidden"])
    )
    assert {"RETRIEVAL", "MODEL"} <= forbidden


def test_s1_real_roots_remain_unopened_at_preregistration() -> None:
    roots = _plan()["roots"]
    for role in ("inbox", "quarantine", "sealed", "audit", "temp", "ready"):
        assert not Path(roots[role]).exists()
