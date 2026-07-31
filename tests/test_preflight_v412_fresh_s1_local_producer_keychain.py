from __future__ import annotations

import copy
import ctypes
import errno
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import threading
from types import SimpleNamespace
from typing import Any

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPOSITORY
    / "scripts/preflight_v412_fresh_s1_local_producer_keychain.py"
)
SPEC = importlib.util.spec_from_file_location("v412_preflight_subject", MODULE_PATH)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


class FakeNative:
    def __init__(self, status: int = -25300) -> None:
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def __call__(self, query: dict[str, Any]) -> int:
        self.calls.append(copy.deepcopy(query))
        return self.status


def _git_reader(repo: Path, commit: str, path: str) -> bytes:
    assert commit == "a" * 40
    return (repo / path).read_bytes()


def run_synthetic(
    root: Path,
    native: FakeNative,
    *,
    checkpoint=None,
):
    return subject.run_preflight(
        root, native, checkpoint=checkpoint, git_reader=_git_reader
    )


def _canonical(value: Any) -> bytes:
    return subject.canonical_json(value)


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def synthetic_repository(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "repository"
    plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
        ).read_text()
    )
    contract = b"synthetic contract\n"
    authority = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s1_local_producer_execution_lock.json"
        ).read_text()
    )
    authority_raw = _canonical(authority)
    plan["contract"]["sha256"] = subject.sha256_bytes(contract)
    plan["execution_lock"]["sha256"] = subject.sha256_bytes(authority_raw)
    for place in (
        plan["preflight_execution_lock"]["schema"]["fields"],
        plan["output"]["schema"]["fields"],
    ):
        place["authority_execution_lock_sha256"]["const"] = subject.sha256_bytes(
            authority_raw
        )
    absent_root = tmp_path / "external" / "producer"
    plan["runtime_absence_guards"] = [
        {
            "path": "config/synthetic_authorization.json",
            "required_state": "ABSENT",
            "resolution": "REPOSITORY",
        },
        {
            "path": str(absent_root),
            "required_state": "ABSENT",
            "resolution": "ABSOLUTE",
        },
        {
            "path": str(absent_root / "claims/provision.claim.json"),
            "required_state": "ABSENT",
            "resolution": "ABSOLUTE",
        },
    ]
    for key, item in plan["future_implementation"].items():
        source = REPOSITORY / item["path"]
        destination = root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copyfile(source, destination)
        else:
            destination.write_text(f"{key}\n")
    for authority_path in (
        authority["plan_path"],
        authority["contract_path"],
        authority["implementation"]["provisioner_path"],
        authority["implementation"]["tests_path"],
    ):
        destination = root / authority_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / authority_path, destination)
    _write(root / plan["contract"]["path"], contract)
    _write(root / plan["execution_lock"]["path"], authority_raw)
    plan_raw = _canonical(plan)
    plan_sha = subject.sha256_bytes(plan_raw)
    implementation = {
        "git_commit": "a" * 40,
        "preflight_path": plan["future_implementation"]["preflight"]["path"],
        "preflight_sha256": subject.sha256_bytes(
            (root / plan["future_implementation"]["preflight"]["path"]).read_bytes()
        ),
        "tests_path": plan["future_implementation"]["tests"]["path"],
        "tests_sha256": subject.sha256_bytes(
            (root / plan["future_implementation"]["tests"]["path"]).read_bytes()
        ),
        "sealer_path": plan["future_implementation"]["sealer"]["path"],
        "sealer_sha256": subject.sha256_bytes(
            (root / plan["future_implementation"]["sealer"]["path"]).read_bytes()
        ),
        "sealer_tests_path": plan["future_implementation"]["sealer_tests"]["path"],
        "sealer_tests_sha256": subject.sha256_bytes(
            (
                root
                / plan["future_implementation"]["sealer_tests"]["path"]
            ).read_bytes()
        ),
    }
    lock = {
        "schema_version": plan["preflight_execution_lock"]["schema"][
            "schema_version"
        ],
        "purpose": "READ_ONLY_KEYCHAIN_LOCATOR_ABSENCE_PREFLIGHT",
        "preflight_plan_path": str(subject.PLAN_RELATIVE),
        "preflight_plan_sha256": plan_sha,
        "contract_path": plan["contract"]["path"],
        "contract_sha256": plan["contract"]["sha256"],
        "authority_execution_lock_path": plan["execution_lock"]["path"],
        "authority_execution_lock_sha256": plan["execution_lock"]["sha256"],
        "implementation": implementation,
        "runtime": authority["runtime"],
        "expected_uid": os.getuid(),
        "logical_time_utc": plan["logical_time_utc"],
        "query_sha256": plan["query_sha256"],
        "claim_path": plan["lifecycle"]["claim"]["path"],
        "output_path": plan["output"]["path"],
    }
    _write(root / subject.PLAN_RELATIVE, plan_raw)
    _write(
        root / plan["preflight_execution_lock"]["path"],
        _canonical(lock),
    )
    (root / "reports/v9").mkdir(parents=True)
    return root, plan


def _rewrite_plan_and_rebind_lock(root: Path, plan: dict[str, Any]) -> None:
    plan_raw = _canonical(plan)
    (root / subject.PLAN_RELATIVE).write_bytes(plan_raw)
    lock_path = root / plan["preflight_execution_lock"]["path"]
    lock = json.loads(lock_path.read_bytes())
    lock["preflight_plan_sha256"] = subject.sha256_bytes(plan_raw)
    lock_path.write_bytes(_canonical(lock))


def test_query_is_exact_status_only_contract() -> None:
    plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
        ).read_text()
    )
    query = subject.build_query_contract(plan)
    assert len(query) == 7
    assert not set(query) & set(plan["query_forbidden_keys"])
    assert subject.sha256_bytes(_canonical(query)) == plan["query_sha256"]


def test_native_bridge_rejects_any_query_value_mutation_before_corefoundation() -> None:
    plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
        ).read_bytes()
    )
    query = subject.build_query_contract(plan)
    query["kSecAttrService"] += ".divergent"
    bridge = object.__new__(subject.MacStatusOnlyKeychain)
    with pytest.raises(subject.PreflightError, match="QUERY_VALUE_PIN"):
        bridge.query_status(query)


def test_happy_path_writes_claim_then_result_and_replay_is_zero_query(
    tmp_path: Path,
) -> None:
    root, plan = synthetic_repository(tmp_path)
    native = FakeNative()
    transitions: list[str] = []
    result = run_synthetic(root, native, checkpoint=transitions.append)
    assert result["verdict"] == "KEYCHAIN_LOCATOR_ABSENT"
    assert len(native.calls) == 1
    assert transitions == [
        "AFTER_CLAIM_BEFORE_NATIVE_CALL",
        "AFTER_NATIVE_CALL_BEFORE_RESULT",
    ]
    assert stat.S_IMODE(
        (root / plan["lifecycle"]["claim"]["path"]).stat().st_mode
    ) == 0o600
    replay = FakeNative()
    assert run_synthetic(root, replay) == result
    assert replay.calls == []


@pytest.mark.parametrize("status", [0, -25299, -50, 1])
def test_only_item_not_found_is_success(tmp_path: Path, status: int) -> None:
    root, plan = synthetic_repository(tmp_path)
    native = FakeNative(status)
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, native)
    assert len(native.calls) == 1
    assert not (root / plan["output"]["path"]).exists()


@pytest.mark.parametrize(
    "boundary",
    ["AFTER_CLAIM_BEFORE_NATIVE_CALL", "AFTER_NATIVE_CALL_BEFORE_RESULT"],
)
def test_crash_boundaries_are_never_requeried(
    tmp_path: Path, boundary: str
) -> None:
    root, _ = synthetic_repository(tmp_path)
    first = FakeNative()

    def crash(name: str) -> None:
        if name == boundary:
            raise subject.InjectedCrash(name)

    with pytest.raises(subject.InjectedCrash):
        run_synthetic(root, first, checkpoint=crash)
    second = FakeNative()
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, second)
    assert second.calls == []


@pytest.mark.parametrize(
    "kind",
    [
        "AUTHORIZATION_FILE_PRESENT",
        "AUTHORIZATION_DIRECTORY_PRESENT",
        "AUTHORIZATION_VALID_SYMLINK",
        "AUTHORIZATION_DANGLING_SYMLINK",
        "ROOT_FILE_PRESENT",
        "ROOT_DIRECTORY_PRESENT",
        "ROOT_VALID_SYMLINK",
        "ROOT_DANGLING_SYMLINK",
        "PRODUCER_CLAIM_FILE_PRESENT",
        "PRODUCER_CLAIM_DIRECTORY_PRESENT",
        "PRODUCER_CLAIM_VALID_SYMLINK",
        "PRODUCER_CLAIM_DANGLING_SYMLINK",
        "PARENT_SYMLINK",
    ],
)
def test_guard_matrix_stops_before_native_query(
    tmp_path: Path, kind: str
) -> None:
    root, plan = synthetic_repository(tmp_path)
    auth = root / plan["runtime_absence_guards"][0]["path"]
    producer = Path(plan["runtime_absence_guards"][1]["path"])
    producer_claim = Path(plan["runtime_absence_guards"][2]["path"])
    target = auth
    if kind.startswith("ROOT_"):
        target = producer
    elif kind.startswith("PRODUCER_CLAIM_"):
        target = producer_claim
    if kind == "PARENT_SYMLINK":
        real = tmp_path / "real"
        real.mkdir()
        parent = producer.parent
        parent.parent.mkdir(parents=True, exist_ok=True)
        parent.symlink_to(real, target_is_directory=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind.endswith("FILE_PRESENT"):
            target.write_text("present")
        elif kind.endswith("DIRECTORY_PRESENT"):
            target.mkdir()
        elif kind.endswith("VALID_SYMLINK"):
            backing = tmp_path / f"backing-{kind}"
            backing.write_text("present")
            target.symlink_to(backing)
        else:
            target.symlink_to(tmp_path / "missing")
    native = FakeNative()
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, native)
    assert native.calls == []


@pytest.mark.parametrize(
    "state",
    [
        "CLAIM_MISSING_RESULT_PRESENT",
        "CLAIM_PRESENT_RESULT_MISSING",
        "CLAIM_PRESENT_RESULT_INVALID",
        "CLAIM_PARTIAL",
        "CLAIM_NONCANONICAL",
        "RESULT_PARTIAL",
        "RESULT_NONCANONICAL",
    ],
)
def test_invalid_lifecycle_states_never_query(
    tmp_path: Path, state: str
) -> None:
    root, plan = synthetic_repository(tmp_path)
    claim = root / plan["lifecycle"]["claim"]["path"]
    result = root / plan["output"]["path"]
    if state != "CLAIM_MISSING_RESULT_PRESENT":
        claim.write_bytes(b'{"partial":true}\n')
    if state not in {"CLAIM_PRESENT_RESULT_MISSING", "CLAIM_PARTIAL", "CLAIM_NONCANONICAL"}:
        result.write_bytes(b'{"partial":true}\n')
    if state == "CLAIM_NONCANONICAL":
        claim.write_bytes(b'{ "partial": true }\n')
    if state == "RESULT_NONCANONICAL":
        result.write_bytes(b'{ "partial": true }\n')
    native = FakeNative()
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, native)
    assert native.calls == []


def _claim_mutation_cases() -> list[tuple[str, str]]:
    plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
        ).read_bytes()
    )
    fields = plan["lifecycle"]["claim"]["schema"]["exact_fields"]
    return [
        (field, mutation)
        for field in fields
        for mutation in ("wrong_value", "wrong_type", "missing")
    ] + [("extra", "extra")]


@pytest.mark.parametrize("field,mutation", _claim_mutation_cases())
def test_every_claim_mutation_replays_with_zero_query(
    tmp_path: Path, field: str, mutation: str
) -> None:
    root, plan = synthetic_repository(tmp_path)
    first = FakeNative()
    run_synthetic(root, first)
    assert len(first.calls) == 1
    claim_path = root / plan["lifecycle"]["claim"]["path"]
    claim = json.loads(claim_path.read_bytes())
    if mutation == "extra":
        claim["extra"] = "forbidden"
    elif mutation == "missing":
        claim.pop(field)
    else:
        value = claim[field]
        claim[field] = (
            1
            if mutation == "wrong_type"
            else value + "-MUTATED"
        )
    claim_path.write_bytes(_canonical(claim))
    replay = FakeNative()
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, replay)
    assert replay.calls == []


def _guard_mutation_cases() -> list[tuple[int, str, str]]:
    return [
        (index, field, mutation)
        for index in range(3)
        for field in ("path", "required_state", "resolution")
        for mutation in ("wrong_value", "wrong_type", "missing")
    ] + [(index, "extra", "extra") for index in range(3)]


@pytest.mark.parametrize("index,field,mutation", _guard_mutation_cases())
def test_every_guard_mutation_is_zero_query(
    tmp_path: Path, index: int, field: str, mutation: str
) -> None:
    root, plan = synthetic_repository(tmp_path)
    guard = plan["runtime_absence_guards"][index]
    if mutation == "extra":
        guard["extra"] = "forbidden"
    elif mutation == "missing":
        guard.pop(field)
    elif mutation == "wrong_type":
        guard[field] = 1
    else:
        guard[field] = {
            "path": "",
            "required_state": "PRESENT",
            "resolution": "UNKNOWN",
        }[field]
    _rewrite_plan_and_rebind_lock(root, plan)
    native = FakeNative()
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, native)
    assert native.calls == []


def _lifecycle_mutation_cases() -> list[tuple[str, str | None]]:
    plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
        ).read_bytes()
    )
    policy_fields = list(plan["lifecycle"]["existing_state_policy"])
    return [
        ("order_wrong_type", None),
        ("order_wrong_value", None),
        ("order_missing", None),
        ("order_extra", None),
        *[
            (mutation, field)
            for field in policy_fields
            for mutation in (
                "policy_wrong_type",
                "policy_wrong_value",
                "policy_missing",
            )
        ],
        ("policy_extra", None),
    ]


@pytest.mark.parametrize("mutation,field", _lifecycle_mutation_cases())
def test_every_lifecycle_plan_mutation_is_zero_query(
    tmp_path: Path, mutation: str, field: str | None
) -> None:
    root, plan = synthetic_repository(tmp_path)
    lifecycle = plan["lifecycle"]
    if mutation == "order_wrong_type":
        lifecycle["entry_order"] = "wrong"
    elif mutation == "order_wrong_value":
        lifecycle["entry_order"] = list(reversed(lifecycle["entry_order"]))
    elif mutation == "order_missing":
        lifecycle["entry_order"].pop()
    elif mutation == "order_extra":
        lifecycle["entry_order"].append("EXTRA")
    elif mutation == "policy_extra":
        lifecycle["existing_state_policy"]["extra"] = "forbidden"
    elif mutation == "policy_missing":
        lifecycle["existing_state_policy"].pop(field)
    elif mutation == "policy_wrong_type":
        lifecycle["existing_state_policy"][field] = 1
    else:
        lifecycle["existing_state_policy"][field] += "-MUTATED"
    _rewrite_plan_and_rebind_lock(root, plan)
    native = FakeNative()
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, native)
    assert native.calls == []


def _lock_mutation_cases() -> list[tuple[str | None, str, str]]:
    plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
        ).read_bytes()
    )
    schema = plan["preflight_execution_lock"]["schema"]
    cases: list[tuple[str | None, str, str]] = []
    for section, fields in (
        (None, schema["exact_fields"]),
        ("implementation", schema["implementation_exact_fields"]),
        ("runtime", schema["runtime_exact_fields"]),
    ):
        for field in fields:
            for mutation in ("wrong_value", "wrong_type", "missing"):
                cases.append((section, field, mutation))
        cases.append((section, "extra", "extra"))
    return cases


@pytest.mark.parametrize("section,field,mutation", _lock_mutation_cases())
def test_every_lock_mutation_fails_before_query(
    tmp_path: Path,
    section: str | None,
    field: str,
    mutation: str,
) -> None:
    root, plan = synthetic_repository(tmp_path)
    lock_path = root / plan["preflight_execution_lock"]["path"]
    lock = json.loads(lock_path.read_text())
    target = lock if section is None else lock[section]
    if mutation == "extra":
        target["extra"] = "forbidden"
    elif mutation == "missing":
        target.pop(field)
    else:
        value = target[field]
        if mutation == "wrong_type":
            target[field] = 1 if type(value) is str else "wrong-type"
        elif type(value) is str:
            target[field] = value + "x"
        elif type(value) is int:
            target[field] = value + 1
        else:
            target[field] = {**value, "extra": "forbidden"}
    lock_path.write_bytes(_canonical(lock))
    native = FakeNative()
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, native)
    assert native.calls == []


def test_direct_lock_validator_rejects_runtime_divergent_from_authority(
    tmp_path: Path,
) -> None:
    root, plan = synthetic_repository(tmp_path)
    plan_raw = (root / subject.PLAN_RELATIVE).read_bytes()
    lock_path = root / plan["preflight_execution_lock"]["path"]
    lock = json.loads(lock_path.read_bytes())
    authority = json.loads(
        (root / plan["execution_lock"]["path"]).read_bytes()
    )
    lock["runtime"]["platform"] += "-MUTATED"
    with pytest.raises(subject.PreflightError, match="platform_VALUE"):
        subject.validate_preflight_lock(
            lock,
            plan=plan,
            plan_sha256=subject.sha256_bytes(plan_raw),
            lock_sha256=subject.sha256_bytes(_canonical(lock)),
            authority_lock=authority,
            repo_root=root,
            git_reader=_git_reader,
        )


@pytest.mark.parametrize(
    "case",
    [
        "FD_TRAVERSAL_PERMISSION_ERROR",
        "FD_TRAVERSAL_OTHER_ERROR",
    ],
)
def test_real_fd_guard_errors_are_zero_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    root, _ = synthetic_repository(tmp_path)
    real_stat = os.stat

    def fail(component, *args, **kwargs):
        if component == "synthetic_authorization.json":
            error = (
                errno.EACCES
                if case == "FD_TRAVERSAL_PERMISSION_ERROR"
                else errno.EIO
            )
            raise OSError(error, case)
        return real_stat(component, *args, **kwargs)

    monkeypatch.setattr(subject.os, "stat", fail)
    native = FakeNative()
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, native)
    assert native.calls == []


@pytest.mark.parametrize("operation", ["open", "stat", "fstat"])
@pytest.mark.parametrize("error", [errno.EACCES, errno.EIO])
def test_low_level_guard_errors_are_normalized_and_zero_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    error: int,
) -> None:
    root, _ = synthetic_repository(tmp_path)
    original_loader = subject.load_and_validate_controls
    original_operation = getattr(subject.os, operation)

    def load_then_arm(*args, **kwargs):
        controls = original_loader(*args, **kwargs)

        def fail(*_args, **_kwargs):
            raise OSError(error, f"injected-{operation}")

        monkeypatch.setattr(subject.os, operation, fail)
        return controls

    monkeypatch.setattr(subject, "load_and_validate_controls", load_then_arm)
    native = FakeNative()
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, native)
    monkeypatch.setattr(subject.os, operation, original_operation)
    assert native.calls == []


@pytest.mark.parametrize(
    "stage,expected_calls",
    [
        ("AFTER_CLAIM_BEFORE_NATIVE_CALL", 0),
        ("AFTER_NATIVE_CALL_BEFORE_RESULT", 1),
    ],
)
@pytest.mark.parametrize("error", [errno.EACCES, errno.EIO])
def test_fstat_errors_at_crash_boundaries_are_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    expected_calls: int,
    error: int,
) -> None:
    root, plan = synthetic_repository(tmp_path)
    real_fstat = subject.os.fstat

    def checkpoint(current: str) -> None:
        if current == stage:
            def fail(_fd: int):
                raise OSError(error, "injected-fstat")

            monkeypatch.setattr(subject.os, "fstat", fail)

    native = FakeNative()
    with pytest.raises(subject.PreflightError):
        run_synthetic(root, native, checkpoint=checkpoint)
    monkeypatch.setattr(subject.os, "fstat", real_fstat)
    assert len(native.calls) == expected_calls
    claim = root / plan["lifecycle"]["claim"]["path"]
    result = root / plan["output"]["path"]
    assert claim.exists()
    assert not result.exists()


@pytest.mark.parametrize("trigger_at", [1, 3])
def test_real_parent_replacement_race_is_detected_before_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trigger_at: int
) -> None:
    root, _ = synthetic_repository(tmp_path)
    config = root / "config"
    displaced = root / "config-displaced"
    authorization = config / "synthetic_authorization.json"
    real_stat = os.stat
    triggered = False
    observations = 0

    def race(component, *args, **kwargs):
        nonlocal triggered, observations
        if component == authorization.name:
            observations += 1
        if (
            component == authorization.name
            and observations == trigger_at
            and not triggered
        ):
            triggered = True
            config.rename(displaced)
            config.mkdir()
            authorization.write_text("present\n")
        return real_stat(component, *args, **kwargs)

    monkeypatch.setattr(subject.os, "stat", race)
    native = FakeNative()
    with pytest.raises(
        subject.PreflightError, match="DIRECTORY_NAMESPACE_REPLACED"
    ):
        run_synthetic(root, native)
    assert triggered
    assert authorization.exists()
    assert native.calls == []


def test_state_parent_replacement_after_claim_is_zero_query(
    tmp_path: Path,
) -> None:
    root, _ = synthetic_repository(tmp_path)
    state_parent = root / "reports/v9"
    displaced = root / "reports/v9-displaced"

    def replace(stage: str) -> None:
        assert stage == "AFTER_CLAIM_BEFORE_NATIVE_CALL"
        state_parent.rename(displaced)
        state_parent.mkdir()

    native = FakeNative()
    with pytest.raises(
        subject.PreflightError, match="DIRECTORY_NAMESPACE_REPLACED"
    ):
        run_synthetic(root, native, checkpoint=replace)
    assert native.calls == []


def test_state_parent_replacement_after_native_never_writes_result(
    tmp_path: Path,
) -> None:
    root, plan = synthetic_repository(tmp_path)
    state_parent = root / "reports/v9"
    displaced = root / "reports/v9-displaced"

    def replace(stage: str) -> None:
        if stage == "AFTER_NATIVE_CALL_BEFORE_RESULT":
            state_parent.rename(displaced)
            state_parent.mkdir()

    native = FakeNative()
    with pytest.raises(
        subject.PreflightError, match="DIRECTORY_NAMESPACE_REPLACED"
    ):
        run_synthetic(root, native, checkpoint=replace)
    assert len(native.calls) == 1
    assert not (state_parent / Path(plan["output"]["path"]).name).exists()


def test_preregistered_zero_query_matrix_is_exactly_26_cases() -> None:
    plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
        ).read_text()
    )
    matrix = plan["implementation_test_requirements"][
        "expected_native_call_count_by_case"
    ]
    assert len(matrix) == 26
    assert set(matrix.values()) == {0}


def test_concurrent_attempts_allow_at_most_one_native_query(
    tmp_path: Path,
) -> None:
    root, _ = synthetic_repository(tmp_path)
    native = FakeNative()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            barrier.wait()
            run_synthetic(root, native)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(native.calls) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], subject.PreflightError)


def test_write_is_exclusive_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    subject.write_exclusive_durable(path, b"first\n")
    with pytest.raises(subject.PreflightError):
        subject.write_exclusive_durable(path, b"second\n")
    assert path.read_bytes() == b"first\n"


class FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


class FakeFramework:
    pass


def test_full_fake_corefoundation_abi_uses_null_result_and_seven_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = FakeFramework()
    security = FakeFramework()
    releases: list[int] = []
    sets: list[tuple[int, int, int]] = []
    seen_result: list[Any] = []
    core.CFDictionaryCreateMutable = FakeFunction(lambda *_: 100)
    core.CFDictionarySetValue = FakeFunction(
        lambda dictionary, key, value: sets.append((dictionary, key, value))
    )
    core.CFStringCreateWithCString = FakeFunction(lambda *_: 200 + len(sets))
    core.CFRelease = FakeFunction(lambda pointer: releases.append(pointer))
    security.SecItemCopyMatching = FakeFunction(
        lambda dictionary, result: seen_result.append(result) or -25300
    )
    constants: dict[tuple[int, str], ctypes.c_void_p] = {}

    def constant(library: Any, name: str) -> ctypes.c_void_p:
        key = (id(library), name)
        constants.setdefault(key, ctypes.c_void_p(1000 + len(constants)))
        return constants[key]

    monkeypatch.setattr(subject, "_framework_constant", constant)
    bridge = subject.MacStatusOnlyKeychain(security, core)
    plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
        ).read_bytes()
    )
    query = subject.build_query_contract(plan)
    assert bridge.query_status(query) == -25300
    assert len(sets) == 7
    assert seen_result == [None]
    assert 100 in releases


def test_main_rejects_arguments_before_framework_load() -> None:
    with pytest.raises(subject.PreflightError, match="ARGV_FORBIDDEN"):
        subject.main(["program", "--forbidden"])


def test_main_requires_material_lock_before_framework_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "MacStatusOnlyKeychain",
        lambda: pytest.fail("framework must remain unloaded"),
    )
    with pytest.raises(subject.PreflightError, match="PREFLIGHT_LOCK_OPEN"):
        subject.main(["program"])


def test_source_has_no_secret_return_or_mutating_keychain_api() -> None:
    source = MODULE_PATH.read_text()
    for forbidden in (
        "kSecReturnData",
        "kSecReturnAttributes",
        "kSecValueData",
        "SecItemAdd",
        "SecItemDelete",
        "SecItemUpdate",
    ):
        assert forbidden not in source
    assert "SecItemCopyMatching(dictionary.value, None)" in source


def test_runtime_git_provenance_disables_replace_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subject.subprocess, "run", run)
    subject._verify_git_commit(tmp_path, "a" * 40)
    assert [command[1] for command, _ in calls] == [
        "cat-file",
        "merge-base",
    ]
    assert all(command[0] == "/usr/bin/git" for command, _ in calls)
    assert all(
        environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        and environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
        and environment["GIT_CONFIG_NOSYSTEM"] == "1"
        for _, environment in calls
    )
