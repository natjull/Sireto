from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN = (
    REPOSITORY
    / "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _closed_objects(plan: dict, dotted: str) -> list[dict]:
    if dotted == "$":
        return [plan]
    values: list[object] = [plan]
    for component in dotted.split("."):
        next_values: list[object] = []
        for value in values:
            if component == "*":
                if isinstance(value, dict):
                    next_values.extend(value.values())
                elif isinstance(value, list):
                    next_values.extend(value)
                else:
                    raise AssertionError(dotted)
            else:
                assert isinstance(value, dict)
                next_values.append(value[component])
        values = next_values
    assert all(isinstance(value, dict) for value in values)
    return values  # type: ignore[return-value]


def _validate_closed_result(
    value: dict, schema: dict, equalities: dict[str, object]
) -> None:
    if list(value) != schema["exact_fields"]:
        raise ValueError("FIELDS")
    for field, rule in schema["fields"].items():
        candidate = value[field]
        expected_type = int if rule["type"] == "integer" else str
        if type(candidate) is not expected_type:
            raise ValueError("TYPE")
        expected = (
            rule["const"]
            if "const" in rule
            else equalities[rule["equals"]]
        )
        if candidate != expected:
            raise ValueError("VALUE")


def test_preflight_plan_is_canonical_and_cross_pinned() -> None:
    raw = PLAN.read_bytes()
    plan = json.loads(raw)
    assert raw == _canonical(plan)
    for role in ("contract", "execution_lock"):
        authority = plan[role]
        assert hashlib.sha256(
            (REPOSITORY / authority["path"]).read_bytes()
        ).hexdigest() == authority["sha256"]


def test_preflight_plan_objects_and_field_rules_are_closed() -> None:
    plan = json.loads(PLAN.read_bytes())
    schema = plan["plan_schema"]
    assert set(schema) == {
        "closed_object_fields",
        "field_rule_variants",
        "schema_version",
    }
    for dotted, exact_fields in schema["closed_object_fields"].items():
        for value in _closed_objects(plan, dotted):
            assert set(value) == set(exact_fields), dotted
    variants = {frozenset(fields) for fields in schema["field_rule_variants"]}
    assert variants == {
        frozenset(("const", "type")),
        frozenset(("equals", "type")),
    }
    for result_schema in (
        plan["lifecycle"]["claim"]["schema"],
        plan["output"]["schema"],
    ):
        assert list(result_schema["fields"]) == sorted(
            result_schema["exact_fields"]
        )
        assert all(
            frozenset(rule) in variants
            for rule in result_schema["fields"].values()
        )


def test_preflight_query_cannot_return_secret_or_attributes() -> None:
    plan = json.loads(PLAN.read_bytes())
    assert plan["query_exact"] == {
        "kSecClass": "kSecClassGenericPassword",
        "kSecAttrService": (
            "com.sireto.v412.fresh-s1-producer-ed25519"
        ),
        "kSecAttrAccount": "SIRETO",
        "kSecAttrSynchronizable": False,
        "kSecUseDataProtectionKeychain": True,
        "kSecUseAuthenticationUI": "kSecUseAuthenticationUIFail",
        "kSecMatchLimit": "kSecMatchLimitOne",
    }
    assert set(plan["query_forbidden_keys"]).isdisjoint(plan["query_exact"])
    assert plan["query_canonicalization"] == {
        "allow_nan": False,
        "encoding": "UTF-8",
        "ensure_ascii": False,
        "final_lf": True,
        "separators": [",", ":"],
        "sort_keys": True,
    }
    query_raw = _canonical(plan["query_exact"])
    assert hashlib.sha256(query_raw).hexdigest() == plan["query_sha256"]
    assert plan["query_sha256"] == (
        "0d5d2fe817391a4d91e51a57b3eaa447"
        "cad932c8c081b64a8b630bdc566fb96f"
    )
    assert plan["success"] == {
        "osstatus": -25300,
        "verdict": "KEYCHAIN_LOCATOR_ABSENT",
    }


def test_preflight_output_is_closed_and_real_output_absent() -> None:
    plan = json.loads(PLAN.read_bytes())
    schema = plan["output"]["schema"]
    assert schema["exact_fields"] == [
        "schema_version",
        "verdict",
        "authority_execution_lock_sha256",
        "query_sha256",
        "osstatus",
        "logical_time_utc",
        "preflight_plan_sha256",
        "preflight_execution_lock_sha256",
        "implementation_commit",
        "implementation_sha256",
        "implementation_tests_sha256",
    ]
    constants = {
        "schema_version": (
            "sireto-v4.12-fresh-s1-local-producer-"
            "keychain-preflight-result-2"
        ),
        "verdict": "KEYCHAIN_LOCATOR_ABSENT",
        "authority_execution_lock_sha256": plan["execution_lock"]["sha256"],
        "query_sha256": plan["query_sha256"],
        "osstatus": -25300,
        "logical_time_utc": plan["logical_time_utc"],
    }
    equalities = {
        "canonical_preflight_plan.sha256": "a" * 64,
        "preflight_execution_lock.file_sha256": "b" * 64,
        "preflight_execution_lock.implementation.git_commit": "c" * 40,
        "preflight_execution_lock.implementation.preflight_sha256": "d" * 64,
        "preflight_execution_lock.implementation.tests_sha256": "e" * 64,
    }
    assert set(schema) == {"exact_fields", "fields"}
    for field, value in constants.items():
        assert schema["fields"][field] == {
            "type": "integer" if type(value) is int else "string",
            "const": value,
        }
    valid = {
        field: (
            rule["const"]
            if "const" in rule
            else equalities[rule["equals"]]
        )
        for field, rule in schema["fields"].items()
    }
    valid = {field: valid[field] for field in schema["exact_fields"]}
    _validate_closed_result(valid, schema, equalities)
    for field in schema["exact_fields"]:
        mutated = dict(valid)
        mutated[field] = (
            valid[field] + "x"
            if type(valid[field]) is str
            else valid[field] + 1
        )
        with pytest.raises(ValueError):
            _validate_closed_result(mutated, schema, equalities)
    with pytest.raises(ValueError):
        _validate_closed_result({**valid, "extra": "x"}, schema, equalities)
    with pytest.raises(ValueError):
        _validate_closed_result(
            {key: value for key, value in valid.items() if key != "verdict"},
            schema,
            equalities,
        )
    assert schema["fields"]["verdict"]["const"] == plan["success"]["verdict"]
    assert schema["fields"]["osstatus"]["const"] == plan["success"]["osstatus"]
    assert not (REPOSITORY / plan["output"]["path"]).exists()


def test_preflight_is_bound_to_future_code_lock_and_closed_gates() -> None:
    plan = json.loads(PLAN.read_bytes())
    assert all(
        artifact["sha256"] == "UNIMPLEMENTED"
        for artifact in plan["future_implementation"].values()
    )
    assert not (
        REPOSITORY / plan["preflight_execution_lock"]["path"]
    ).exists()
    sequence = plan["sequence"]
    assert sequence.index("TWO_GO_PREFLIGHT_PREREG_AUDITS") < sequence.index(
        "IMPLEMENT_PREFLIGHT_AND_SEALER_WITH_FAKE_FRAMEWORKS_ONLY"
    )
    assert sequence.index(
        "TWO_GO_PREFLIGHT_IMPLEMENTATION_AUDITS"
    ) < sequence.index("SEAL_PREFLIGHT_EXECUTION_LOCK_ONCE")
    assert sequence.index(
        "TWO_GO_PREFLIGHT_LOCK_MATERIAL_AUDITS"
    ) < sequence.index("RUN_PREFLIGHT_ONCE")
    assert {
        gate["verdict"] for gate in plan["gates"].values()
    } == {
        "GO_PREFLIGHT_PREREG_NEXT_IMPLEMENTATION",
        "GO_PREFLIGHT_IMPLEMENTATION_NEXT_LOCK",
        "GO_PREFLIGHT_LOCK_MATERIAL_NEXT_RUN",
    }


def test_preflight_absence_guards_and_crash_policy_are_closed() -> None:
    plan = json.loads(PLAN.read_bytes())
    guards = plan["runtime_absence_guards"]
    assert [guard["path"] for guard in guards] == [
        "config/v4_12_fresh_s1_local_producer_authorization.json",
        (
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/"
            "fresh_holdout_intake_authorities/local_producer"
        ),
        (
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/"
            "fresh_holdout_intake_authorities/local_producer/"
            "claims/provision.claim.json"
        ),
    ]
    assert all(guard["required_state"] == "ABSENT" for guard in guards)
    assert all(
        not (
            (REPOSITORY / guard["path"])
            if guard["resolution"] == "REPOSITORY"
            else Path(guard["path"])
        ).exists()
        for guard in guards
    )
    lifecycle = plan["lifecycle"]
    assert lifecycle["entry_order"].index(
        "CREATE_DURABLE_CLAIM_O_EXCL"
    ) < lifecycle["entry_order"].index("CALL_SECITEMCOPYMATCHING_ONCE")
    assert all(
        "NO_REQUERY" in disposition
        or disposition == "RETURN_STORED_RESULT_WITHOUT_KEYCHAIN"
        or disposition == "STOP_OUTPUT_WITHOUT_CLAIM"
        for disposition in lifecycle["existing_state_policy"].values()
    )
    assert not (REPOSITORY / lifecycle["claim"]["path"]).exists()
