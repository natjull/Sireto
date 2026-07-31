from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import stat

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


def _validate_nested_lock(
    value: dict,
    *,
    exact_fields: list[str],
    fields: dict,
    equalities: dict[str, object],
    nested: dict[str, tuple[list[str], dict]],
) -> None:
    if list(value) != exact_fields:
        raise ValueError("FIELDS")
    for field in exact_fields:
        rule = fields[field]
        candidate = value[field]
        if rule["type"] == "object":
            if type(candidate) is not dict or "schema" not in rule:
                raise ValueError("TYPE")
            child_exact, child_fields = nested[rule["schema"]]
            _validate_nested_lock(
                candidate,
                exact_fields=child_exact,
                fields=child_fields,
                equalities=equalities,
                nested=nested,
            )
            continue
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


def _materialize_nested(
    *,
    exact_fields: list[str],
    fields: dict,
    equalities: dict[str, object],
    nested: dict[str, tuple[list[str], dict]],
) -> dict:
    result: dict[str, object] = {}
    for field in exact_fields:
        rule = fields[field]
        if "schema" in rule:
            child_exact, child_fields = nested[rule["schema"]]
            result[field] = _materialize_nested(
                exact_fields=child_exact,
                fields=child_fields,
                equalities=equalities,
                nested=nested,
            )
        elif "const" in rule:
            result[field] = rule["const"]
        else:
            result[field] = equalities[rule["equals"]]
    return result


def _reference_absent(
    path: Path, *, stat_at=os.stat
) -> bool:
    if not path.is_absolute():
        raise ValueError("ABSOLUTE_PATH_REQUIRED")
    parts = path.parts[1:]
    directory_fd = os.open(
        path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        for index, component in enumerate(parts):
            try:
                metadata = stat_at(
                    component,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    return True
                raise ValueError("FD_TRAVERSAL_ERROR") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("SYMLINK")
            if index == len(parts) - 1:
                raise ValueError("ENTRY_EXISTS")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("PARENT_NOT_DIRECTORY")
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise ValueError("FD_TRAVERSAL_ERROR") from exc
            opened = os.fstat(next_fd)
            if (opened.st_dev, opened.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                os.close(next_fd)
                raise ValueError("PARENT_REPLACED")
            os.close(directory_fd)
            directory_fd = next_fd
        raise ValueError("ROOT_EXISTS")
    finally:
        os.close(directory_fd)


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
        frozenset(("schema", "type")),
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


def test_preflight_claim_schema_rejects_every_mutation() -> None:
    plan = json.loads(PLAN.read_bytes())
    schema = plan["lifecycle"]["claim"]["schema"]
    equalities = {
        "canonical_preflight_plan.sha256": hashlib.sha256(
            PLAN.read_bytes()
        ).hexdigest(),
        "preflight_execution_lock.file_sha256": "b" * 64,
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
        wrong_value = dict(valid)
        wrong_value[field] = valid[field] + "x"
        with pytest.raises(ValueError):
            _validate_closed_result(wrong_value, schema, equalities)
        wrong_type = dict(valid)
        wrong_type[field] = 1
        with pytest.raises(ValueError):
            _validate_closed_result(wrong_type, schema, equalities)
    with pytest.raises(ValueError):
        _validate_closed_result({**valid, "extra": "x"}, schema, equalities)
    with pytest.raises(ValueError):
        _validate_closed_result(
            {key: value for key, value in valid.items() if key != "state"},
            schema,
            equalities,
        )


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


def test_preflight_execution_lock_schema_rejects_nested_mutations() -> None:
    plan = json.loads(PLAN.read_bytes())
    schema = plan["preflight_execution_lock"]["schema"]
    authority_lock = json.loads(
        (REPOSITORY / plan["execution_lock"]["path"]).read_bytes()
    )
    equalities: dict[str, object] = {
        "canonical_preflight_plan.contract.sha256": plan["contract"]["sha256"],
        "canonical_preflight_plan.sha256": hashlib.sha256(
            PLAN.read_bytes()
        ).hexdigest(),
        "runtime.os.getuid": os.getuid(),
        "git.commit_containing_all_implementation_blobs": "a" * 40,
    }
    for role in ("preflight", "tests", "sealer", "sealer_tests"):
        equalities[f"git_blob_sha256_at_commit.{role}_path"] = role[0] * 64
    for field, value in authority_lock["runtime"].items():
        equalities[
            f"authority_execution_lock.runtime.{field}"
        ] = value
    nested = {
        "preflight_execution_lock.implementation": (
            schema["implementation_exact_fields"],
            schema["implementation_fields"],
        ),
        "preflight_execution_lock.runtime": (
            schema["runtime_exact_fields"],
            schema["runtime_fields"],
        ),
    }
    valid = _materialize_nested(
        exact_fields=schema["exact_fields"],
        fields=schema["fields"],
        equalities=equalities,
        nested=nested,
    )
    _validate_nested_lock(
        valid,
        exact_fields=schema["exact_fields"],
        fields=schema["fields"],
        equalities=equalities,
        nested=nested,
    )
    for section_key, fields in (
        (None, schema["exact_fields"]),
        ("implementation", schema["implementation_exact_fields"]),
        ("runtime", schema["runtime_exact_fields"]),
    ):
        for field in fields:
            wrong_value = json.loads(json.dumps(valid))
            target = (
                wrong_value
                if section_key is None
                else wrong_value[section_key]
            )
            candidate = target[field]
            if type(candidate) is str:
                target[field] = candidate + "x"
            elif type(candidate) is int:
                target[field] = candidate + 1
            else:
                target[field] = {**candidate, "extra": "x"}
            with pytest.raises(ValueError):
                _validate_nested_lock(
                    wrong_value,
                    exact_fields=schema["exact_fields"],
                    fields=schema["fields"],
                    equalities=equalities,
                    nested=nested,
                )
            wrong_type = json.loads(json.dumps(valid))
            target = (
                wrong_type
                if section_key is None
                else wrong_type[section_key]
            )
            candidate = target[field]
            target[field] = (
                1
                if type(candidate) is str
                else "WRONG_TYPE"
            )
            with pytest.raises(ValueError):
                _validate_nested_lock(
                    wrong_type,
                    exact_fields=schema["exact_fields"],
                    fields=schema["fields"],
                    equalities=equalities,
                    nested=nested,
                )
            missing = json.loads(json.dumps(valid))
            target = (
                missing
                if section_key is None
                else missing[section_key]
            )
            target.pop(field)
            with pytest.raises(ValueError):
                _validate_nested_lock(
                    missing,
                    exact_fields=schema["exact_fields"],
                    fields=schema["fields"],
                    equalities=equalities,
                    nested=nested,
                )
        extra = json.loads(json.dumps(valid))
        target = (
            extra
            if section_key is None
            else extra[section_key]
        )
        target["extra"] = "x"
        with pytest.raises(ValueError):
            _validate_nested_lock(
                extra,
                exact_fields=schema["exact_fields"],
                fields=schema["fields"],
                equalities=equalities,
                nested=nested,
            )


def test_preflight_absence_guards_and_crash_policy_are_closed(
    tmp_path: Path,
) -> None:
    plan = json.loads(PLAN.read_bytes())
    assert plan["absence_semantics"] == {
        "absent_only_when": (
            "FIRST_MISSING_COMPONENT_OPENAT_OR_FSTATAT_RETURNS_ENOENT"
        ),
        "allowed_syscall": (
            "FD_ANCHORED_OPENAT_O_DIRECTORY_O_NOFOLLOW_"
            "AND_FSTATAT_NOFOLLOW"
        ),
        "existing_entry_policy": "STOP",
        "other_errno_policy": "STOP",
        "parent_traversal": (
            "EACH_EXISTING_COMPONENT_MUST_BE_REAL_DIRECTORY_NO_SYMLINK"
        ),
        "symlink_policy": "STOP_INCLUDING_DANGLING_SYMLINK",
    }
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
        _reference_absent(
            (REPOSITORY / guard["path"])
            if guard["resolution"] == "REPOSITORY"
            else Path(guard["path"])
        )
        for guard in guards
    )
    assert _reference_absent(tmp_path / "missing")
    existing_file = tmp_path / "file"
    existing_file.touch()
    existing_directory = tmp_path / "directory"
    existing_directory.mkdir()
    target = tmp_path / "target"
    target.touch()
    valid_symlink = tmp_path / "valid-symlink"
    valid_symlink.symlink_to(target)
    dangling_symlink = tmp_path / "dangling-symlink"
    dangling_symlink.symlink_to(tmp_path / "missing-target")
    parent_symlink = tmp_path / "parent-symlink"
    parent_symlink.symlink_to(existing_directory, target_is_directory=True)
    for path in (
        existing_file,
        existing_directory,
        valid_symlink,
        dangling_symlink,
        parent_symlink / "child",
    ):
        with pytest.raises(ValueError):
            _reference_absent(path)

    def denied(
        component: str, *, dir_fd: int, follow_symlinks: bool
    ):
        if component == "denied":
            raise PermissionError(errno.EACCES, "denied", component)
        return os.stat(
            component,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    with pytest.raises(ValueError, match="FD_TRAVERSAL_ERROR"):
        _reference_absent(tmp_path / "denied", stat_at=denied)
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
    requirements = plan["implementation_test_requirements"]
    assert {
        "AUTHORIZATION_DANGLING_SYMLINK",
        "ROOT_VALID_SYMLINK",
        "PRODUCER_CLAIM_DIRECTORY_PRESENT",
        "PARENT_SYMLINK",
        "FD_TRAVERSAL_PERMISSION_ERROR",
        "PARENT_REPLACEMENT_RACE",
    }.issubset(requirements["absence_guard_cases"])
    assert {
        "CLAIM_PRESENT_RESULT_MISSING",
        "CRASH_AFTER_NATIVE_CALL_BEFORE_RESULT",
        "RESULT_NONCANONICAL",
    }.issubset(requirements["lifecycle_cases"])
    expected_cases = set(requirements["absence_guard_cases"]) | set(
        requirements["lifecycle_cases"]
    )
    assert set(requirements["expected_native_call_count_by_case"]) == (
        expected_cases
    )
    assert all(
        count == 0
        for count in requirements[
            "expected_native_call_count_by_case"
        ].values()
    )
    assert requirements["expected_native_call_count_by_case"][
        "CLAIM_AND_RESULT_VALID"
    ] == 0
    assert requirements["mutation_targets"] == [
        "CLAIM_EACH_FIELD_EXTRA_MISSING_WRONG_TYPE_WRONG_VALUE",
        "LOCK_TOP_EACH_FIELD_EXTRA_MISSING_WRONG_TYPE_WRONG_VALUE",
        (
            "LOCK_IMPLEMENTATION_EACH_FIELD_EXTRA_MISSING_"
            "WRONG_TYPE_WRONG_VALUE"
        ),
        (
            "LOCK_RUNTIME_EACH_FIELD_EXTRA_MISSING_"
            "WRONG_TYPE_WRONG_VALUE"
        ),
        "GUARDS_EACH_FIELD_EXTRA_MISSING_WRONG_TYPE_WRONG_VALUE",
        "LIFECYCLE_EACH_FIELD_EXTRA_MISSING_WRONG_TYPE_WRONG_VALUE",
    ]
