from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import socket
import stat
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = REPOSITORY / "config/v4_12_fresh_s0_authoritative_run_plan.json"
CONTRACT_PATH = REPOSITORY / "docs/v4_12_fresh_s0_authoritative_run_contract.md"
SEALER_PATH = (
    REPOSITORY
    / "scripts/seal_v412_fresh_intake_synthetic_execution_lock.py"
)
LAUNCHER_PATH = (
    REPOSITORY
    / "scripts/launch_v412_fresh_intake_synthetic_scanner_sealer.py"
)
WORKER_PATH = REPOSITORY / "scripts/run_v412_fresh_s0_worker.py"
PROFILE_PATH = (
    REPOSITORY
    / "config/v4_12_fresh_intake_synthetic_scanner_sealer.sb"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sealer = _load("v412_s0_sealer_integration", SEALER_PATH)
launcher = _load("v412_s0_launcher_integration", LAUNCHER_PATH)
worker = _load("v412_s0_worker_integration", WORKER_PATH)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(fd: int, volume_uuid: str) -> dict[str, object]:
    info = os.fstat(fd)
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "uid": info.st_uid,
        "volume_uuid": volume_uuid,
        "link_count": info.st_nlink,
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
    }


class _FixedResolver:
    def __init__(self, value: str) -> None:
        self.value = value

    def for_fd(self, _fd: int) -> str:
        return self.value


def test_authoritative_plan_contract_and_source_roles_are_cross_pinned() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert _sha(PLAN_PATH) == sealer.EXPECTED_PLAN_SHA256
    assert _sha(CONTRACT_PATH) == sealer.EXPECTED_CONTRACT_SHA256
    assert plan["contract"]["sha256"] == _sha(CONTRACT_PATH)
    assert tuple(sealer.IMPLEMENTATION_ROLE_PATHS) == tuple(
        plan["execution_lock"]["implementation_blob_roles"]
    )
    for relative in sealer.IMPLEMENTATION_ROLE_PATHS.values():
        assert (REPOSITORY / relative).is_file()


def test_all_fd_role_orders_match_across_sealer_launcher_and_worker() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert tuple(plan["fd_protocol"]["lock_input_roles_exact_order"]) == (
        launcher.LOCK_INPUT_ROLES
    )
    assert tuple(
        plan["fd_protocol"]["parent_retained_roles_exact_order"]
    ) == launcher.PARENT_RETAINED_ROLES
    assert tuple(
        plan["fd_protocol"]["worker_payload_roles_exact_order"]
    ) == launcher.WORKER_PAYLOAD_ROLES == worker.PAYLOAD_ROLES
    assert launcher.WRITE_DIRECTORY_ROLES == worker.WRITE_ROLES


def test_profile_real_template_renders_with_no_residual_placeholder(
    tmp_path: Path,
) -> None:
    plan, _core = sealer._load_plans()
    rendered = sealer._render_profile(
        PROFILE_PATH.read_bytes(),
        plan,
        tmp_path,
        "a" * 64,
        tmp_path / "runtime" / ("a" * 64),
    )
    assert b"@@" not in rendered
    assert rendered.endswith(b"\n")
    assert b"(deny default)" in rendered
    text = rendered.decode("utf-8")
    assert '(literal "/")' in text
    assert '(subpath "/")' not in text
    assert "DYLD_" not in text
    assert '(subpath "/opt")' not in text
    assert '(subpath "/opt/homebrew")' not in text
    assert f'(literal "{sealer.HOST_PYTHON_FRAMEWORK}")' in text
    assert (
        "Resources/Python.app/Contents/MacOS/Python"
        in text
    )
    assert "Versions/3.14/bin/python3.14" not in text


def test_launcher_worker_spec_is_accepted_by_worker(tmp_path: Path) -> None:
    run_id = "a" * 64
    attempt_id = "b" * 64
    volume_uuid = "00000000-0000-0000-0000-000000000001"
    payloads = []
    opened: list[int] = []
    write_fds: dict[str, int] = {}
    parent, child = socket.socketpair()
    try:
        for index, role in enumerate(worker.PAYLOAD_ROLES):
            path = tmp_path / f"payload-{index}"
            path.write_bytes(f"{role}\n".encode())
            path.chmod(0o600)
            fd = os.open(path, os.O_RDONLY)
            opened.append(fd)
            payloads.append(
                launcher.OpenAuthority(
                    role=role,
                    path=path,
                    fd=fd,
                    expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected_size=path.stat().st_size,
                    expected_identity=_identity(fd, volume_uuid),
                )
            )
        for role in worker.WRITE_ROLES:
            path = tmp_path / role.lower()
            path.mkdir(mode=0o700)
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            opened.append(fd)
            write_fds[role] = fd
        lock_raw = b'{"synthetic":"lock"}\n'
        lock = {
            "implementation_commit": "1" * 40,
            "synthetic_run_id": run_id,
            "attempt_id": attempt_id,
            "logical_time_utc": "2026-07-30T00:00:00Z",
        }
        value = launcher._worker_spec(
            lock,
            lock_raw,
            payloads,
            write_fds,
            _FixedResolver(volume_uuid),
        )
        spec_path = tmp_path / "worker-spec.json"
        spec_path.write_bytes(launcher.canonical_json(value))
        spec_fd = os.open(spec_path, os.O_RDONLY)
        opened.append(spec_fd)
        worker._validate_spec(value, spec_fd, child.fileno())
    finally:
        parent.close()
        child.close()
        for fd in reversed(opened):
            try:
                os.close(fd)
            except OSError:
                pass


def test_worker_control_messages_are_accepted_by_launcher() -> None:
    run_id = "a" * 64
    attempt_id = "b" * 64
    spec = {"synthetic_run_id": run_id, "attempt_id": attempt_id}
    lock = {"synthetic_run_id": run_id, "attempt_id": attempt_id}
    ready = worker._ready_message(spec)
    launcher._validate_ready(ready, lock, os.getpid())
    terminal = worker._terminal_message(
        spec,
        message_type="RESULT",
        reason_code="OK",
        terminal_result="INGESTED_SYNTHETIC_SCANNER_SEALER_V412",
        stability={
            "same_worker_process": True,
            "same_five_payload_fds": True,
            "monotonic_elapsed_seconds": "60.000000000",
        },
        output_authority={
            "sealed_input_payload_manifest_sha256": "1" * 64,
            "sealed_input_seal_sha256": "2" * 64,
            "terminal_tree_kind": "SCAN_OUTPUT",
            "terminal_tree_payload_manifest_sha256": "3" * 64,
            "terminal_tree_seal_sha256": "4" * 64,
            "journal_generation": 3,
            "journal_generation_manifest_sha256": "5" * 64,
            "journal_head_event_sha256": "6" * 64,
        },
    )
    launcher._validate_terminal(terminal, lock)


def test_canary_proof_schema_and_empty_stop_policy_are_aligned() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert worker.CANARY_REPORT_SCHEMA == (
        "sireto-v4.12-fresh-s0-canary-proof-1"
    )
    declared = plan["launch_receipt"]["types"]["canaries"]
    assert "exact_order" in declared
    assert "empty_array_for_stop_before_complete_canary_proof" in declared
    with pytest.raises(launcher.LauncherStop):
        launcher._validate_canaries([], plan, run_id="a" * 64)


def _minimal_r2b_lock(plan: dict) -> dict:
    run_id = "a" * 64
    runtime_root = (
        Path(plan["paths"]["allowed_root"]) / "runtime" / run_id
    )
    boundary = plan["r2_successor"]["runtime_boundary_amendment"]
    helper = boundary["private_python_helper"]
    records = [
        {
            "role": "PYTHON_EXECUTABLE",
            "source_path": helper["source_path"],
            "private_relative_path": (
                "rootfs/" + helper["source_path"].removeprefix("/")
            ),
            "size_bytes": 1,
            "sha256": helper["sha256"],
            "mode": "0500",
        },
        {
            "role": "PYTHON_STDLIB",
            "source_path": "/pinned/encodings/__init__.py",
            "private_relative_path": (
                "rootfs/opt/homebrew/Cellar/python@3.14/3.14.3_1/"
                "Frameworks/Python.framework/Versions/3.14/lib/python3.14/"
                "encodings/__init__.py"
            ),
            "size_bytes": 1,
            "sha256": "1" * 64,
            "mode": "0400",
        },
        {
            "role": "PYARROW",
            "source_path": "/pinned/pyarrow/__init__.py",
            "private_relative_path": (
                "rootfs/opt/homebrew/lib/python3.14/site-packages/"
                "pyarrow/__init__.py"
            ),
            "size_bytes": 1,
            "sha256": "2" * 64,
            "mode": "0400",
        },
    ]
    host = boundary["host_python_framework"]
    return {
        "implementation_commit": "3" * 40,
        "synthetic_run_id": run_id,
        "attempt_id": "b" * 64,
        "runtime": {
            "private_runtime_manifest": {"records": records},
        },
        "read_fds": [
            {
                "role": "HOST_PYTHON_FRAMEWORK",
                "absolute_path": host["path"],
                "identity": {},
                "volume_uuid": "00000000-0000-0000-0000-000000000001",
                "size_bytes": 1,
                "sha256": host["sha256"],
            }
        ],
        "_runtime_root_for_test": str(runtime_root),
    }


def test_r2b_environment_is_exactly_private_and_has_no_dyld() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    lock = _minimal_r2b_lock(plan)
    lock.pop("_runtime_root_for_test")
    environment = launcher._worker_environment(lock, plan)
    assert list(environment) == plan["launcher"]["environment_exact_keys"]
    assert set(environment) == {
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHOME",
        "PYTHONPATH",
        "TMPDIR",
    }
    assert not any(key.startswith("DYLD_") for key in environment)
    assert environment["PYTHONHOME"].startswith(
        f"{plan['paths']['allowed_root']}/runtime/{lock['synthetic_run_id']}/"
    )
    assert environment["PYTHONPATH"].split(":")[0].startswith(
        f"{plan['paths']['allowed_root']}/runtime/{lock['synthetic_run_id']}/"
    )


def test_r2b_macho_parser_finds_only_the_pinned_non_system_framework() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    boundary = plan["r2_successor"]["runtime_boundary_amendment"]
    helper = Path(boundary["private_python_helper"]["source_path"])
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == (
        boundary["private_python_helper"]["sha256"]
    )
    names = launcher._macho_dylib_load_names(helper.read_bytes())
    non_system = tuple(
        name
        for name in names
        if not name.startswith("/System/") and not name.startswith("/usr/lib/")
    )
    assert non_system == (boundary["host_python_framework"]["path"],)
    assert "subprocess" not in inspect.getsource(
        launcher._validate_python_helper_install_name
    )
    assert "subprocess" not in inspect.getsource(
        launcher._macho_dylib_load_names
    )


def test_r2b_stdlib_excludes_homebrew_framework_symlink_alias() -> None:
    stdlib = Path(sealer.sysconfig.get_path("stdlib")).resolve()
    aliases = [
        source
        for source in sealer._iter_source_files(stdlib)
        if source.resolve() == sealer.HOST_PYTHON_FRAMEWORK
    ]
    assert aliases, "the pinned Homebrew stdlib alias must remain observable"
    assert all(
        source.resolve() != sealer.HOST_PYTHON_FRAMEWORK
        for source in sealer._iter_stdlib_runtime_files(stdlib)
    )


def test_r2b_profile_transport_is_p_text_and_never_passes_profile_fd() -> None:
    source = inspect.getsource(launcher.run_authoritative_launch)
    lock_source = inspect.getsource(launcher._load_lock)
    assert '"-p",' in source
    assert 'f"/dev/fd/{profile_authority.fd}"' not in source
    assert "pass_fds.append(profile_authority.fd)" not in source
    assert "SIRETO_V412_FRESH_SYNTHETIC_S0_R2_AUTHORITATIVE_RUN" in (
        lock_source
    )
    assert launcher.LOCK_FIELDS[7] == "r2_smoke"


def test_r2b_smoke_attestation_is_fully_reconstructed_and_fail_closed(
    tmp_path: Path,
) -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    lock = _minimal_r2b_lock(plan)
    lock.pop("_runtime_root_for_test")
    profile_raw = b"(version 1)\n(deny default)\n"
    profile_path = tmp_path / "effective.sb"
    profile_path.write_bytes(profile_raw)
    profile_path.chmod(0o400)
    profile_fd = os.open(profile_path, os.O_RDONLY)
    authority = launcher.OpenAuthority(
        role="SANDBOX_PROFILE",
        path=profile_path,
        fd=profile_fd,
        expected_sha256=hashlib.sha256(profile_raw).hexdigest(),
        expected_size=len(profile_raw),
        expected_identity=None,
    )
    try:
        environment = launcher._worker_environment(lock, plan)
        argv = [
            os.fspath(launcher.SANDBOX_EXEC_PATH),
            "-p",
            profile_raw.decode("utf-8"),
            os.fspath(launcher._private_python_path(lock, plan)),
            "-c",
            launcher._private_import_assertion(lock, plan),
        ]
        required = plan["r2_successor"]["smoke_attestation"][
            "required_result"
        ]
        smoke = {
            "schema_version": (
                "sireto-v4.12-fresh-s0-r2-smoke-attestation-2"
            ),
            "implementation_commit": lock["implementation_commit"],
            "synthetic_run_id": lock["synthetic_run_id"],
            "attempt_id": lock["attempt_id"],
            "python_sha256": plan["r2_successor"][
                "runtime_boundary_amendment"
            ]["private_python_helper"]["sha256"],
            "profile_sha256": authority.expected_sha256,
            "environment_sha256": hashlib.sha256(
                launcher.canonical_json(environment, final_lf=False)
            ).hexdigest(),
            "argv_sha256": hashlib.sha256(
                launcher.canonical_json(argv, final_lf=False)
            ).hexdigest(),
            "pass_fds": [],
            **required,
        }
        fields = plan["schema_definitions"]["r2_smoke_attestation"][
            "exact_fields"
        ]
        projection = {key: smoke[key] for key in fields if key != "smoke_sha256"}
        smoke["smoke_sha256"] = hashlib.sha256(
            launcher.canonical_json(projection, final_lf=False)
        ).hexdigest()
        lock["r2_smoke"] = smoke
        launcher._validate_r2_smoke(lock, plan, authority)

        lock["r2_smoke"]["pass_fds"] = [profile_fd]
        with pytest.raises(launcher.LauncherStop, match="smoke authority"):
            launcher._validate_r2_smoke(lock, plan, authority)
    finally:
        os.close(profile_fd)


def test_r2b_smoke_capture_limit_is_enforced_while_reading() -> None:
    bounded_source = inspect.getsource(sealer._run_bounded_child)
    assert "stdin=subprocess.DEVNULL" in bounded_source
    assert "close_fds=True" in bounded_source
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
