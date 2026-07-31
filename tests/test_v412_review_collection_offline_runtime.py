from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v412_review_collection_offline_runtime.py"
SPEC = importlib.util.spec_from_file_location("v412_review_offline_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


@pytest.fixture(scope="session", autouse=True)
def native_worker_build(tmp_path_factory: pytest.TempPathFactory):
    destination = tmp_path_factory.mktemp("native-worker") / runtime.NATIVE_WORKER_BASENAME
    subprocess.run(
        [sys.executable, str(runtime.NATIVE_WORKER_BUILDER), str(destination)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    previous = runtime.NATIVE_WORKER_PATH
    runtime.NATIVE_WORKER_PATH = destination
    try:
        yield destination
    finally:
        runtime.NATIVE_WORKER_PATH = previous


def test_canonical_json_and_domain_separated_ids_are_exact() -> None:
    assert runtime.canonical_json({"é": 2, "a": 1}) == b'{"a":1,"\xc3\xa9":2}'
    with pytest.raises(ValueError):
        runtime.canonical_json({"bad": float("nan")})
    projection = ["q1", 2, "société exemple"]
    expected = hashlib.sha256(
        b"SIRETO-V412-R30-SEARCH\0"
        + json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert runtime.search_attempt_id(*projection) == expected
    assert runtime.dns_attempt_id(expected, "example.fr") != expected


def test_primary_journal_is_exclusive_chained_and_projected(tmp_path: Path) -> None:
    journal = runtime.AccessJournal(tmp_path / "run")
    intent = journal.intent(
        phase="IDENTITY_DISCOVERY",
        operation="OPEN_LOCAL",
        target_kind="PATH",
        target_canonical="fixture://identity",
        query_id="q1",
        query_ordinal=0,
    )
    journal.result(
        intent,
        outcome="SUCCESS",
        byte_count=3,
        content_sha256=hashlib.sha256(b"abc").hexdigest(),
    )
    head = journal.verify_complete()
    assert head == journal.records[-1]["event_sha256"]
    assert [record["event_ordinal"] for record in journal.records] == [0, 1, 2]
    assert journal.records[1]["previous_event_sha256"] == journal.records[0]["event_sha256"]
    projection = tmp_path / "run" / "access_journal.jsonl"
    assert journal.project_jsonl(projection) == hashlib.sha256(projection.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        journal.project_jsonl(projection)


def test_journal_tamper_and_unclosed_intent_fail_closed(tmp_path: Path) -> None:
    first = runtime.AccessJournal(tmp_path / "tampered")
    genesis = first.events_dir / "00000000000000000000.json"
    genesis.write_bytes(genesis.read_bytes().replace(b'"outcome":"NONE"', b'"outcome":"SUCCESS"'))
    with pytest.raises(runtime.IntegrityStop):
        first.verify_complete()

    second = runtime.AccessJournal(tmp_path / "pending")
    second.intent(
        phase="IDENTITY_DISCOVERY",
        operation="OPEN_LOCAL",
        target_kind="PATH",
        target_canonical="fixture://identity",
    )
    with pytest.raises(runtime.IntegrityStop, match="without RESULT"):
        second.verify_complete()


def test_broker_has_no_network_implementation_and_revocation_is_terminal(tmp_path: Path) -> None:
    journal = runtime.AccessJournal(tmp_path / "broker")
    journal.state_transition(phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN")
    broker = runtime.OfflineBroker(journal)
    with pytest.raises(runtime.OfflineNetworkDenied):
        broker.search_request("https://example.fr/", query_id="q1", query_ordinal=0)
    assert journal.records[-1]["outcome"] == "DENIED"
    broker.revoke()
    with pytest.raises(runtime.OfflineNetworkDenied):
        broker.dns_resolution("example.fr", query_id="q1", query_ordinal=0)
    assert journal.records[-1]["outcome"] == "STOP_INTEGRITY"
    with pytest.raises(runtime.IntegrityStop, match="irreversible"):
        broker.revoke()
    journal.verify_complete()

def test_worker_boundaries_reject_candidate_leakage_and_early_comparison(tmp_path: Path) -> None:
    with pytest.raises(runtime.IntegrityStop):
        runtime.IdentityInput.from_mapping(
            {
                "query_id": "q1",
                "crm_name": "Exemple",
                "crm_address": "1 rue Exemple",
                "crm_postcode": "75001",
                "top1_siret": "55210055400013",
            }
        )
    journal = runtime.AccessJournal(tmp_path / "boundaries")
    journal.state_transition(phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN")
    broker = runtime.OfflineBroker(journal)
    identity = runtime.IdentityDiscoveryWorker().run(
        runtime.IdentityInput("q1", "Exemple", "1 rue Exemple", "75001")
    )
    comparison = runtime.ComparisonInput("q1", ("55210055400013",))
    with pytest.raises(runtime.IntegrityStop, match="prior network revocation"):
        runtime.FrozenCandidateComparisonWorker().run(identity, comparison, broker=broker)


def test_launcher_runs_two_phases_offline_in_order(tmp_path: Path) -> None:
    result = runtime.SyntheticOfflineLauncher().run(
        output_root=tmp_path / "run",
        identity_input=runtime.IdentityInput("q1", "École Exemple", "2 rue A", "69001"),
        comparison_input=runtime.ComparisonInput(
            "q1", ("55210055400013", "78983652500020")
        ),
    )
    assert result.comparison.candidate_count == 2
    assert result.identity_worker_pid != result.comparison_worker_pid
    assert result.identity_worker_pid != __import__("os").getpid()
    assert result.comparison_worker_pid != __import__("os").getpid()
    lines = [json.loads(line) for line in (tmp_path / "run" / "access_journal.jsonl").read_text().splitlines()]
    transitions = [line["target_canonical"] for line in lines if line["event_kind"] == "STATE_TRANSITION"]
    assert transitions == ["IDENTITY_NETWORK_OPEN", "IDENTITY_SEALED_NETWORK_REVOKED"]
    assert all(line["operation"] not in runtime.NETWORK_OPERATIONS for line in lines)


def test_candidate_budget_is_an_absolute_cap() -> None:
    with pytest.raises(runtime.IntegrityStop, match="budget"):
        runtime.ComparisonInput.from_mapping(
            {"query_id": "q1", "candidate_sirets": ["55210055400013"] * 101}
        )
    with pytest.raises(runtime.IntegrityStop, match="budget"):
        runtime.ComparisonInput("q1", tuple(["55210055400013"] * 101))
    with pytest.raises(runtime.IntegrityStop, match="ASCII"):
        runtime.ComparisonInput.from_mapping(
            {"query_id": "q1", "candidate_sirets": ["١٢٣٤٥٦٧٨٩٠١٢٣٤"]}
        )


def test_state_machine_is_exact_and_irreversible(tmp_path: Path) -> None:
    journal = runtime.AccessJournal(tmp_path / "state")
    with pytest.raises(runtime.IntegrityStop, match="non-monotone"):
        journal.state_transition(
            phase="COMPARISON", target_state="IDENTITY_SEALED_NETWORK_REVOKED"
        )
    journal.state_transition(phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN")
    with pytest.raises(runtime.IntegrityStop, match="non-monotone"):
        journal.state_transition(
            phase="IDENTITY_DISCOVERY", target_state="IDENTITY_NETWORK_OPEN"
        )
    journal.state_transition(
        phase="IDENTITY_SEAL", target_state="IDENTITY_SEALED_NETWORK_REVOKED"
    )
    with pytest.raises(runtime.IntegrityStop, match="non-monotone"):
        journal.state_transition(phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN")
    journal.verify_complete()


def test_journal_rejects_symlink_substitution_without_escape(tmp_path: Path) -> None:
    root = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    journal = runtime.AccessJournal(root)
    (root / "journal_events").rename(root / "original_events")
    (root / "journal_events").symlink_to(outside, target_is_directory=True)
    with pytest.raises(runtime.IntegrityStop, match="substituted"):
        journal.state_transition(phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN")
    assert not (outside / "00000000000000000001.json").exists()


def test_journal_rejects_ancestor_swap_while_using_retained_fds(tmp_path: Path) -> None:
    root = tmp_path / "run"
    journal = runtime.AccessJournal(root)
    root.rename(tmp_path / "displaced-run")
    root.mkdir()
    with pytest.raises(runtime.IntegrityStop, match="ancestor link was substituted"):
        journal.state_transition(phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN")
    assert list(root.iterdir()) == []


def test_journal_rejects_non_string_schema_and_invalid_result_matrix(tmp_path: Path) -> None:
    journal = runtime.AccessJournal(tmp_path / "types")
    with pytest.raises(runtime.IntegrityStop, match="target canonical"):
        journal.intent(
            phase="IDENTITY_DISCOVERY",
            operation="OPEN_LOCAL",
            target_kind="PATH",
            target_canonical=123,  # type: ignore[arg-type]
            query_id="q1",
        )

    second = runtime.AccessJournal(tmp_path / "matrix")
    intent = second.intent(
        phase="IDENTITY_DISCOVERY",
        operation="SEARCH_REQUEST",
        target_kind="URL",
        target_canonical="https://example.invalid/",
        query_id="q1",
        query_ordinal=0,
    )
    with pytest.raises(runtime.IntegrityStop, match="requires error_type"):
        second.result(intent, outcome="TIMEOUT", error_type="PARSE")


def test_sandbox_denies_network_fork_exec_and_free_local_open(tmp_path: Path) -> None:
    canary = tmp_path / "worker-must-not-open.txt"
    canary.write_bytes(b"unchanged")
    response = runtime._run_sandboxed_worker(
        "CAPABILITY_PROBE",
        {
            "forbidden_path": str(canary),
            "schema_version": runtime.WORKER_SCHEMA,
        },
    )
    assert response["artifact"] == {
        "exec": True,
        "fork": True,
        "local_read": True,
        "local_write": True,
        "network": True,
        "reexec": True,
    }
    assert canary.read_bytes() == b"unchanged"


def test_sandbox_pathless_dyld_read_does_not_expose_usr_bin() -> None:
    response = runtime._run_sandboxed_worker(
        "CAPABILITY_PROBE",
        {
            "forbidden_path": "/usr/bin/true",
            "schema_version": runtime.WORKER_SCHEMA,
        },
    )
    assert response["artifact"]["local_read"] is True
    assert response["artifact"]["exec"] is True
    assert response["artifact"]["reexec"] is True


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "/dev/null",
        "/cores",
        "/.vol",
        "/.file",
        "/usr/local",
        "/usr/libexec",
        "/usr/standalone",
        "/System/Volumes/Preboot",
        "/System/Volumes/Update",
        "/System/Volumes/Hardware",
        "/System/Volumes/VM",
        "/System/Volumes/Preboot/BAF255B6-34FD-4982-9984-775AD146AA2F/PreLoginData/diagnostics/shutdown.0.log",
        "/System/Volumes/Update/0x1307",
        "/System/Volumes/Hardware/ProductDocuments/RegulatoryCertifications/RegulatoryCertification-A3401-J614s-CHN.lpdf/Contents/Info.plist",
    ],
)
def test_sandbox_denies_every_existing_non_runtime_root(forbidden_path: str) -> None:
    assert Path(forbidden_path).exists()
    response = runtime._run_sandboxed_worker(
        "CAPABILITY_PROBE",
        {
            "forbidden_path": forbidden_path,
            "schema_version": runtime.WORKER_SCHEMA,
        },
    )
    assert response["artifact"]["local_read"] is True


def test_sandbox_profile_pins_absent_non_runtime_roots() -> None:
    profile = runtime._sandbox_profile(
        Path("/private/tmp/closed") / runtime.NATIVE_WORKER_BASENAME
    )
    for path in (
        "/workspace", "/usr/X11", "/usr/X11R6", "/System/Volumes",
        "/dev", "/cores", "/.vol"
    ):
        if path.startswith("/."):
            assert '^/\\.[^/]+' in profile
        else:
            assert f'(subpath "{path}")' in profile


def test_runtime_root_inventory_fails_closed_on_new_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_listdir = runtime.os.listdir

    def injected(path):
        entries = real_listdir(path)
        if path == "/":
            return [*entries, "unexpected-root"]
        return entries

    monkeypatch.setattr(runtime.os, "listdir", injected)
    with pytest.raises(runtime.IntegrityStop, match="closed inventory"):
        runtime._assert_closed_runtime_roots()


def test_revocation_rejects_pending_network_intent(tmp_path: Path) -> None:
    journal = runtime.AccessJournal(tmp_path / "pending-network")
    journal.state_transition(phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN")
    intent = journal.intent(
        phase="IDENTITY_DISCOVERY",
        operation="SEARCH_REQUEST",
        target_kind="URL",
        target_canonical="https://example.invalid/",
        query_id="q1",
        query_ordinal=0,
    )
    with pytest.raises(runtime.IntegrityStop, match="pending at revocation"):
        journal.state_transition(
            phase="IDENTITY_SEAL", target_state="IDENTITY_SEALED_NETWORK_REVOKED"
        )
    assert journal.state == "IDENTITY_NETWORK_OPEN"
    journal.result(intent, outcome="DENIED", error_type="IO_INTEGRITY")
    journal.state_transition(
        phase="IDENTITY_SEAL", target_state="IDENTITY_SEALED_NETWORK_REVOKED"
    )
    journal.verify_complete()


def test_comparison_has_no_revocation_bypass_and_spawn_requires_live_broker() -> None:
    assert not hasattr(runtime.FrozenCandidateComparisonWorker, "run_after_revocation")
    payload = {
        "comparison_input": {"candidate_sirets": [], "query_id": "q1"},
        "identity_artifact": {"payload_sha256": "0" * 64, "query_id": "q1"},
        "network_state": "IDENTITY_SEALED_NETWORK_REVOKED",
        "schema_version": runtime.WORKER_SCHEMA,
    }
    with pytest.raises(runtime.IntegrityStop, match="live broker"):
        runtime._run_sandboxed_worker("FROZEN_CANDIDATE_COMPARISON", payload)


def test_raw_parent_component_is_rejected_before_normalization(tmp_path: Path) -> None:
    unsafe = Path(f"{tmp_path}/parent/../escaped")
    with pytest.raises(runtime.IntegrityStop, match="canonical absolute syntax"):
        runtime.AccessJournal(unsafe)
    assert not (tmp_path / "escaped").exists()


def test_native_worker_receipt_tamper_is_rejected(
    tmp_path: Path, native_worker_build: Path
) -> None:
    worker = tmp_path / runtime.NATIVE_WORKER_BASENAME
    shutil.copyfile(native_worker_build, worker)
    worker.chmod(0o500)
    receipt = json.loads(
        native_worker_build.with_suffix(".json").read_text(encoding="utf-8")
    )
    receipt["artifact_sha256"] = "0" * 64
    worker.with_suffix(".json").write_bytes(runtime.canonical_json(receipt) + b"\n")
    with pytest.raises(runtime.IntegrityStop, match="receipt"):
        runtime._open_native_worker(worker)


def test_coordinated_binary_and_receipt_mutation_cannot_move_trust_anchor(
    tmp_path: Path, native_worker_build: Path
) -> None:
    worker = tmp_path / runtime.NATIVE_WORKER_BASENAME
    shutil.copyfile(native_worker_build, worker)
    mutated = bytearray(worker.read_bytes())
    mutated[-1] ^= 1
    worker.write_bytes(mutated)
    worker.chmod(0o500)
    receipt = json.loads(
        native_worker_build.with_suffix(".json").read_text(encoding="utf-8")
    )
    receipt["artifact_sha256"] = hashlib.sha256(mutated).hexdigest()
    worker.with_suffix(".json").write_bytes(runtime.canonical_json(receipt) + b"\n")
    with pytest.raises(runtime.IntegrityStop, match="receipt"):
        runtime._open_native_worker(worker)


def test_builder_requires_canonical_basename(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(runtime.NATIVE_WORKER_BUILDER), str(tmp_path / "worker")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0
    assert runtime.NATIVE_WORKER_BASENAME in result.stderr


def test_two_native_builds_are_byte_identical(tmp_path: Path) -> None:
    outputs = []
    for dirname in ("one", "two"):
        destination = tmp_path / dirname / runtime.NATIVE_WORKER_BASENAME
        subprocess.run(
            [sys.executable, str(runtime.NATIVE_WORKER_BUILDER), str(destination)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        outputs.append(destination)
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert outputs[0].with_suffix(".json").read_bytes() == outputs[1].with_suffix(".json").read_bytes()


def test_staged_worker_path_swap_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_select = runtime.select.select
    swapped = False

    def swap_after_ready(readers, writers, errors, timeout=None):
        nonlocal swapped
        result = real_select(readers, writers, errors, timeout)
        if not swapped and readers and result[0]:
            targets = list(Path("/private/tmp").glob(
                f"sireto-v412-native-*/{runtime.NATIVE_WORKER_BASENAME}"
            ))
            assert len(targets) == 1
            target = targets[0]
            replacement = target.with_name("replacement")
            shutil.copyfile("/usr/bin/true", replacement)
            replacement.chmod(0o500)
            replacement.replace(target)
            swapped = True
        return result

    monkeypatch.setattr(runtime.select, "select", swap_after_ready)
    with pytest.raises(runtime.IntegrityStop, match="changed across spawn"):
        runtime._run_sandboxed_worker(
            "IDENTITY_DISCOVERY",
            {
                "identity_input": {
                    "crm_address": "1 rue A",
                    "crm_name": "Exemple",
                    "crm_postcode": "75001",
                    "query_id": "q-swap",
                },
                "schema_version": runtime.WORKER_SCHEMA,
            },
        )
    assert swapped


def test_worker_failure_path_cleans_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = set(Path("/private/tmp").glob("sireto-v412-native-*"))

    def fail_read(*_args, **_kwargs):
        raise runtime.IntegrityStop("synthetic deadline")

    monkeypatch.setattr(runtime, "_read_fd_all_before", fail_read)
    with pytest.raises(runtime.IntegrityStop, match="synthetic deadline"):
        runtime._run_sandboxed_worker(
            "IDENTITY_DISCOVERY",
            {
                "identity_input": {
                    "crm_address": "1 rue A",
                    "crm_name": "Exemple",
                    "crm_postcode": "75001",
                    "query_id": "q-timeout",
                },
                "schema_version": runtime.WORKER_SCHEMA,
            },
        )
    assert set(Path("/private/tmp").glob("sireto-v412-native-*")) == before


def test_pipe_creation_failure_closes_fds_and_cleans_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_staging = set(Path("/private/tmp").glob("sireto-v412-native-*"))
    before_fds = set(runtime.os.listdir("/dev/fd"))
    real_pipe = runtime.os.pipe
    calls = 0

    def fail_second_pipe():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic pipe failure")
        return real_pipe()

    monkeypatch.setattr(runtime.os, "pipe", fail_second_pipe)
    with pytest.raises(OSError, match="synthetic pipe failure"):
        runtime._run_sandboxed_worker(
            "IDENTITY_DISCOVERY",
            {
                "identity_input": {
                    "crm_address": "1 rue A",
                    "crm_name": "Exemple",
                    "crm_postcode": "75001",
                    "query_id": "q-pipe-failure",
                },
                "schema_version": runtime.WORKER_SCHEMA,
            },
        )
    assert calls == 2
    assert set(Path("/private/tmp").glob("sireto-v412-native-*")) == before_staging
    assert set(runtime.os.listdir("/dev/fd")) == before_fds
