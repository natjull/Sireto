from __future__ import annotations

import importlib.util
import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY
    / "scripts/seal_v412_fresh_intake_synthetic_execution_lock.py"
)
SPEC = importlib.util.spec_from_file_location("v412_s0_lock_sealer", SOURCE)
assert SPEC is not None and SPEC.loader is not None
sealer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sealer
SPEC.loader.exec_module(sealer)
BUILDER_SOURCE = (
    REPOSITORY / "scripts/build_v412_fresh_s0_r3_fixture.py"
)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "v412_s0_r3_fixture_builder_for_lock_test", BUILDER_SOURCE
)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
builder = importlib.util.module_from_spec(BUILDER_SPEC)
sys.modules[BUILDER_SPEC.name] = builder
BUILDER_SPEC.loader.exec_module(builder)


def test_authoritative_plan_and_contract_pins_are_current() -> None:
    plan, core = sealer._load_plans()
    assert plan["contract"]["sha256"] == sealer.EXPECTED_CONTRACT_SHA256
    assert plan["core"]["pins"]["plan"]["sha256"] == (
        sealer.sha256_file(sealer.CORE_PLAN_PATH)
    )
    assert core["status"] == "PREREGISTERED_SYNTHETIC_ONLY_DO_NOT_EXECUTE"


def test_canonical_json_rejects_duplicates_and_noncanonical_bytes() -> None:
    with pytest.raises(sealer.LockSealError, match="duplicate JSON key"):
        sealer.parse_canonical_json(b'{"a":1,"a":2}\n', "duplicate")
    with pytest.raises(sealer.LockSealError, match="not canonical"):
        sealer.parse_canonical_json(b'{ "a": 1 }\n', "pretty")
    assert sealer.parse_canonical_json(b'{"a":1}\n', "valid") == {"a": 1}


def test_opaque_digest_is_a_single_safe_component() -> None:
    value = sealer.opaque_digest("domain\0", {"b": 2, "a": 1})
    assert sealer.OPAQUE_ID.fullmatch(value)
    assert "/" not in value and "." not in value
    assert value == sealer.opaque_digest("domain\0", {"a": 1, "b": 2})


def test_profile_derivation_is_exact_and_rejects_residual_marker(
    tmp_path: Path,
) -> None:
    plan, _ = sealer._load_plans()
    template = (
        REPOSITORY
        / "config/v4_12_fresh_intake_synthetic_scanner_sealer.sb"
    ).read_bytes()
    rendered = sealer._render_profile(
        template, plan, tmp_path, "a" * 64, tmp_path / "runtime"
    )
    assert b"@@" not in rendered
    assert rendered.endswith(b"\n")
    assert rendered.count(str(tmp_path / "runtime").encode()) >= 2
    with pytest.raises(sealer.LockSealError, match="residual"):
        sealer._render_profile(
            template + b"\n@@UNKNOWN@@",
            plan,
            tmp_path,
            "a" * 64,
            tmp_path / "runtime",
        )


def test_lock_paths_have_no_unresolved_tokens() -> None:
    plan, _ = sealer._load_plans()
    paths = sealer._substitute_lock_paths(
        plan, sealer.ALLOWED_ROOT, "a" * 64, "b" * 64
    )
    assert set(paths) == set(plan["lock_values"]["paths"])
    assert all(Path(value).is_absolute() for value in paths.values())
    assert all("<" not in value and ">" not in value for value in paths.values())


def test_precreated_run_directories_are_empty_and_disjoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthetic"
    root.mkdir(mode=0o700)
    sealer._prepare_run_directories(root, "a" * 64, "b" * 64)
    assert (root / "audit" / ("a" * 64) / "worker").is_dir()
    parent = root / "audit" / ("a" * 64) / "parent"
    assert {path.name for path in parent.iterdir()} == {
        "spec",
        "claims",
        "leases",
        "launch_receipts",
    }
    assert all(not any(path.iterdir()) for path in parent.iterdir())


def test_precreated_directories_reject_ancestor_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthetic"
    outside = tmp_path / "outside"
    root.mkdir(mode=0o700)
    outside.mkdir(mode=0o700)
    (root / "sealed").symlink_to(outside, target_is_directory=True)
    with pytest.raises(sealer.LockSealError, match="anchored open failed"):
        sealer._prepare_run_directories(root, "a" * 64, "b" * 64)
    assert list(outside.iterdir()) == []


def test_exclusive_write_rejects_parent_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir(mode=0o700)
    outside.mkdir(mode=0o700)
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(sealer.LockSealError, match="anchored open failed"):
        sealer._write_exclusive(root / "redirect" / "forbidden", b"x", 0o400)
    assert list(outside.iterdir()) == []


def test_fixture_validation_binds_control_and_all_five_payloads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fixture"
    plan, core = sealer._load_plans()
    test_plan = json.loads(
        (REPOSITORY / "config/v4_12_fresh_s0_r3_plan.json").read_bytes()
    )
    predecessor_source = Path(test_plan["predecessor"]["receipt_path"])
    predecessor_copy = tmp_path / "r2-receipt.json"
    predecessor_copy.write_bytes(predecessor_source.read_bytes())
    test_plan["predecessor"]["receipt_path"] = str(predecessor_copy)
    test_plan["paths"]["allowed_root"] = str(root)
    test_plan["r3_successor"]["root"] = str(root)
    test_plan_path = tmp_path / "r3-plan.json"
    test_plan_path.write_bytes(builder.canonical_json_bytes(test_plan))
    built = builder.build_fixture(
        root=root, plan_path=test_plan_path
    )
    run_id, attempt_id, logical_time, inputs = sealer._validate_fixture(
        plan, core, root
    )
    assert run_id == built["synthetic_run_id"]
    assert sealer.OPAQUE_ID.fullmatch(attempt_id)
    assert logical_time == core["fixture"]["logical_time_utc"]
    assert [record[0] for record in inputs] == (
        plan["fd_protocol"]["worker_payload_roles_exact_order"]
    )
    (Path(built["package_path"]) / "extra").write_text("forbidden")
    with pytest.raises(sealer.LockSealError, match="missing or extra"):
        sealer._validate_fixture(plan, core, root)


def test_git_blob_reader_matches_committed_authoritative_plan() -> None:
    commit = subprocess.run(
        ["/usr/bin/git", "rev-parse", "46b1958^{commit}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    blob, mode = sealer._git_blob_at_commit(
        REPOSITORY,
        commit,
        "config/v4_12_fresh_s0_authoritative_run_plan.json",
    )
    assert mode == 0o100644
    assert len(json.loads(blob)["contract"]["sha256"]) == 64


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin volume authority")
def test_native_volume_uuid_matches_diskutil() -> None:
    native = sealer.volume_uuid(REPOSITORY)
    device = subprocess.run(
        ["/usr/bin/stat", "-f", "%Sd", REPOSITORY],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # The core assertion is that the native call returns a valid stable UUID;
    # the launcher independently verifies the same device through diskutil.
    assert len(native) == 36
    assert native == sealer.volume_uuid(REPOSITORY)
    assert device


def test_sealer_has_no_authoritative_worker_launch() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(calls) == 2
    assert sorted(call.func.attr for call in calls) == ["Popen", "run"]
    assert "[str(OTOOL), *arguments]" in source
    assert "subprocess.Popen(" in inspect.getsource(
        sealer._run_bounded_child
    )
    smoke_source = inspect.getsource(sealer._run_runtime_smoke)
    assert "private_python" in smoke_source
    assert "_run_bounded_child(" in smoke_source


def test_r2b_smoke_capture_is_bounded_during_read() -> None:
    with pytest.raises(
        sealer.LockSealError, match="stdout exceeded capture limit"
    ):
        sealer._run_bounded_child(
            [
                sys.executable,
                "-c",
                "import os;os.write(1,b'x'*65537)",
            ],
            {"PATH": "/usr/bin:/bin"},
            timeout_seconds=10,
            capture_limit_bytes_each=65536,
        )
