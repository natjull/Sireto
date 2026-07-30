from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from scripts import build_v412_fresh_intake_synthetic_fixture as core_builder


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = REPOSITORY / "config/v4_12_fresh_s0_r3_plan.json"
CONTRACT_PATH = REPOSITORY / "docs/v4_12_fresh_s0_r3_contract.md"


def _canonical(value: object, *, final_lf: bool = True) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _opaque(domain: str, values: dict) -> str:
    digest = hashlib.sha256(
        domain.encode("utf-8") + _canonical(values, final_lf=False)
    ).hexdigest()
    return "".join(chr(ord("a") + int(character, 16)) for character in digest)


def _plan() -> dict:
    raw = PLAN_PATH.read_bytes()
    value = json.loads(raw)
    assert raw == _canonical(value)
    return value


def _resolve(value: object, dotted_path: str) -> object:
    current = value
    for component in dotted_path.split("."):
        assert isinstance(current, dict), dotted_path
        assert component in current, dotted_path
        current = current[component]
    return current


def _parent(value: object, dotted_path: str) -> tuple[dict, str]:
    components = dotted_path.split(".")
    assert components and all(components), dotted_path
    current = value
    for component in components[:-1]:
        assert isinstance(current, dict), dotted_path
        assert component in current, dotted_path
        current = current[component]
    assert isinstance(current, dict), dotted_path
    return current, components[-1]


def _materialize_r3(plan: dict) -> dict:
    authority = plan["base_authorities"][plan["inheritance"]["base_plan_role"]]
    base_path = REPOSITORY / authority["path"]
    assert _sha(base_path) == authority["sha256"]
    effective = deepcopy(json.loads(base_path.read_bytes()))

    removals = plan["inheritance"]["removals"]
    assert len(removals) == len(set(removals))
    for target in removals:
        parent, leaf = _parent(effective, target)
        assert leaf in parent, target
        del parent[leaf]

    overrides = plan["inheritance"]["overrides"]
    targets = [record["target"] for record in overrides]
    assert len(targets) == len(set(targets))
    for record in overrides:
        source_value = deepcopy(_resolve(plan, record["source"]))
        components = record["target"].split(".")
        if len(components) == 1:
            parent, leaf = effective, components[0]
        else:
            parent, leaf = _parent(effective, record["target"])
        if leaf in parent:
            assert type(parent[leaf]) is type(source_value), record["target"]
        parent[leaf] = source_value
    return effective


def test_r3_plan_is_canonical_and_contract_is_pinned() -> None:
    plan = _plan()
    assert plan["status"] == (
        "PREREGISTERED_R3_DO_NOT_IMPLEMENT_UNTIL_TWO_INDEPENDENT_AUDITS"
    )
    assert plan["contract"] == {
        "path": "docs/v4_12_fresh_s0_r3_contract.md",
        "sha256": _sha(CONTRACT_PATH),
    }
    for record in plan["base_authorities"].values():
        assert _sha(REPOSITORY / record["path"]) == record["sha256"]


def test_r3_identity_is_independently_recomputed() -> None:
    plan = _plan()
    identity = plan["execution_identity"]
    assert _opaque(
        identity["run"]["domain"], identity["run"]["values"]
    ) == identity["run"]["result"]
    assert _opaque(
        identity["attempt"]["domain"], identity["attempt"]["values"]
    ) == identity["attempt"]["result"]
    assert identity["run"]["result"] == (
        "kbfkbicacgcgabcddiiacogfkndicooigeebcdaghpdgklgebocfhkinnniladkl"
    )
    assert identity["attempt"]["result"] == (
        "afjgbfncbfdbcakcjiclhmlnmgmemcjmllkhdfgogjjncompjojcnbkelopdklgp"
    )


def test_r3_control_hash_is_rebuilt_from_core_fixture() -> None:
    plan = _plan()
    core_path = REPOSITORY / plan["base_authorities"]["core_plan"]["path"]
    core_raw = core_path.read_bytes()
    core = json.loads(core_raw)
    run_id = plan["execution_identity"]["run"]["result"]
    csv_bytes = core["fixture"]["csv"]["exact_utf8_text"].encode("utf-8")
    evidence_bytes = core_builder._empty_evidence_parquet(core)
    payloads = core_builder._manifest_objects(
        core, run_id, csv_bytes, evidence_bytes
    )
    control = {
        "schema_version": core["control_manifest"]["schema"],
        "synthetic_fixture": True,
        "fixture_spec_sha256": core["control_manifest"][
            "fixture_spec_sha256"
        ],
        "synthetic_run_id": run_id,
        "logical_time_utc": core["fixture"]["logical_time_utc"],
        "batch_count": 1,
        "expected_source_row_count": 6,
        "producer_exclusions": [],
        "collection_source_manifest_sha256": hashlib.sha256(
            payloads["collection_source_manifest.json"]
        ).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            payloads["source_manifest.json"]
        ).hexdigest(),
        "crm_safe_csv_sha256": hashlib.sha256(
            payloads["crm_safe.csv"]
        ).hexdigest(),
        "evidence_source_manifest_sha256": hashlib.sha256(
            payloads["evidence_source_manifest.json"]
        ).hexdigest(),
        "evidence_source_parquet_sha256": hashlib.sha256(
            payloads["evidence_source.parquet"]
        ).hexdigest(),
    }
    assert hashlib.sha256(_canonical(control)).hexdigest() == (
        plan["execution_identity"]["attempt"]["values"][
            "fixture_control_manifest_sha256"
        ]
    )


def test_r3_predecessor_and_gate_are_immutable_authorities() -> None:
    plan = _plan()
    for record in (plan["predecessor"], plan["gate_authority"]["result"]):
        path = Path(
            record["receipt_path"]
            if "receipt_path" in record
            else record["path"]
        )
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == (
            record["receipt_sha256"]
            if "receipt_sha256" in record
            else record["sha256"]
        )
        assert raw == _canonical(json.loads(raw))
    gate = json.loads(
        Path(plan["gate_authority"]["result"]["path"]).read_bytes()
    )
    assert gate["status"] == "GO"
    assert gate["canary_denied_count"] == 11
    assert set(gate["canary_errnos"]) <= {1, 13}
    assert gate["r1_rejection_output_tree_unchanged"] is True


def test_r3_schema_chain_is_closed_and_root_is_absent() -> None:
    plan = _plan()
    schemas = plan["schemas"]
    assert "execution_identity" in schemas["execution_lock"]["exact_fields"]
    assert "execution_identity" in schemas["worker_spec"]["exact_fields"]
    assert schemas["worker_spec"]["schema_version"].endswith("worker-spec-2")
    assert schemas["control_result"]["schema_version"].endswith(
        "control-result-2"
    )
    assert "worker_failure" in schemas["control_result"]["exact_fields"]
    assert schemas["worker_failure"]["matrix"] == {
        "IDENTITY": [
            "EXECUTION_IDENTITY_SCHEMA_INVALID",
            "RUN_DERIVATION_MISMATCH",
            "ATTEMPT_DERIVATION_MISMATCH",
            "SPEC_CONTROL_IDENTITY_MISMATCH",
        ],
        "WORKER_RUNTIME": ["INTERNAL_ERROR"],
    }
    assert not Path(plan["paths"]["allowed_root"]).exists()


def test_r3_baseline_blobs_match_the_gate_commit_state() -> None:
    plan = _plan()
    for record in plan["implementation_baseline"]["artifacts"].values():
        assert _sha(REPOSITORY / record["path"]) == record["sha256"]
    assert plan["implementation_baseline"]["full_suite_passed"] == 1095
    audit = plan["required_implementation_gate"]["audit_binding"]
    assert audit["independent_audits_required"] == 2
    assert audit["required_verdict"] == "GO_R3_IMPLEMENTATION"
    assert audit["must_bind_preregistration_commit"] is True
    assert audit["must_bind_plan_sha256"] is True
    assert audit["must_bind_contract_sha256"] is True


def test_r3_closed_overlay_materializes_from_the_pinned_r2_plan() -> None:
    plan = _plan()
    effective = _materialize_r3(plan)
    assert effective["authorization"]["fixed_path"] == plan["paths"]["authorization"]
    assert effective["paths"]["allowed_root"] == plan["paths"]["allowed_root"]
    assert effective["execution_identity"] == plan["execution_identity"]
    assert effective["predecessor"] == plan["predecessor"]
    assert effective["gate_authority"] == plan["gate_authority"]
    assert effective["r3_successor"] == plan["r3_successor"]
    assert effective["execution_lock"]["exact_fields"] == (
        plan["schemas"]["execution_lock"]["exact_fields"]
    )
    assert "r2_smoke" not in effective["execution_lock"]["types"]
    assert effective["execution_lock"]["types"]["runtime_smoke"] == (
        "runtime_smoke_attestation"
    )
    assert effective["execution_lock"]["types"]["execution_identity"] == (
        "execution_identity"
    )
    assert effective["launch_receipt"]["schema_version"].endswith(
        "launch-receipt-3"
    )
    assert effective["schema_definitions"]["worker_spec"] == (
        plan["schemas"]["worker_spec"]
    )
    assert effective["schema_definitions"]["control_result"] == (
        plan["schemas"]["control_result"]
    )
    assert "r2_smoke_attestation" not in effective["schema_definitions"]
    assert effective["enum_definitions"]["source_blob_roles"] == (
        plan["source_blob_roles"]
    )
    assert effective["execution_lock"]["implementation_blob_roles"] == (
        plan["source_blob_roles"]
    )
    assert "r2_fixture_builder" not in effective["future_implementation"]["artifacts"]
    assert "r2_fixture_tests" not in effective["future_implementation"]["artifacts"]
    assert effective["future_implementation"]["artifacts"]["r3_fixture_builder"] == (
        plan["future_implementation"]["artifacts"]["r3_fixture_builder"]
    )
    assert effective["future_implementation"]["artifacts"]["r3_fixture_tests"] == (
        plan["future_implementation"]["artifacts"]["r3_fixture_tests"]
    )


def test_r3_inherited_authorities_resolve_in_the_pinned_r2_plan() -> None:
    plan = _plan()
    base_path = REPOSITORY / plan["base_authorities"]["r2_plan"]["path"]
    base = json.loads(base_path.read_bytes())
    for authority in plan["inherited_authorities"].values():
        if "base_path" in authority:
            _resolve(base, authority["base_path"])
        for dotted_path in authority.get("base_paths", []):
            _resolve(base, dotted_path)


def test_r3_overlay_rejects_hostile_missing_source_and_target_parent() -> None:
    plan = _plan()
    hostile_source = deepcopy(plan)
    hostile_source["inheritance"]["overrides"][0]["source"] = "missing.source"
    try:
        _materialize_r3(hostile_source)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing overlay source accepted")

    hostile_target = deepcopy(plan)
    hostile_target["inheritance"]["overrides"][0]["target"] = (
        "missing.parent.contract"
    )
    try:
        _materialize_r3(hostile_target)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing overlay target parent accepted")
