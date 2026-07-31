#!/usr/bin/env python3
"""Build a deterministic, network-free V4.12 collection fixture and run.

The fixture is deliberately label-free.  The identity worker receives only
CRM fields.  Scripted DNS/HTTP responses are consumed in memory, the network
capability is then revoked, and only then are synthetic SIRENE records and the
candidate pool exposed to the comparison worker.
"""

from __future__ import annotations

import argparse
import base64
import contextvars
from contextlib import contextmanager
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import select
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import pyarrow.parquet as pq

try:
    import v412_review_collection_broker as m2broker
    import replay_v412_review_collection_policy as replay
    import v412_review_collection_offline_runtime as runtime
except ImportError:  # pragma: no cover - package import used by pytest
    from scripts import v412_review_collection_broker as m2broker
    from scripts import replay_v412_review_collection_policy as replay
    from scripts import v412_review_collection_offline_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "config/v4_12_review_collection_synthetic_fixture.json"
FIXTURE_SHA256 = "44d8f192bd638371e5d6a175916fa30c77c107532fad56079d52c6893762d2b6"
PUBLIC_SUFFIX_PATH = ROOT / "config/v4_12_review_public_suffixes.txt"
POLICY_PATH = ROOT / "config/v4_12_review_collection_policy.json"
DNS_VECTORS_PATH = ROOT / "config/v4_12_review_dns_security_vectors.json"
SCHEMA = "sireto-v4.12-r30-review-collection-synthetic-run-1"
MANIFEST_SCHEMA = "sireto-v4.12-r30-review-collection-synthetic-manifest-1"
SEAL_SCHEMA = "sireto-v4.12-r30-review-collection-synthetic-seal-1"
TREE_DOMAIN = b"SIRETO-V412-R30-SYNTHETIC-TREE\0"
RUN_ID_DOMAIN = b"SIRETO-V412-R30-SYNTHETIC-RUN\0"
M1_TREE_DOMAIN = b"SIRETO-V412-R30-SYNTHETIC-M1-TREE\0"
DOCKET_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
    "v4_12_review_adjudication_pilot/c7a9feecaf2d3c2a"
)
DOCKET_HASHES = {
    "identity/identity_discovery.parquet": "e5b62b5a614420d3d8260c2c0daa744043f956cd3aedcb05ca16301ea816b872",
    "identity/collection_plan.parquet": "d52ed433ee00a70bd95b5b453558d4432e83be4099f551e7d68c84602b0bbfe0",
    "comparison/docket.parquet": "71db76376d2af26de4aba4143bf54032d3c58d64f884022da9c746417e2c1cf1",
    "comparison/candidate_context.parquet": "c79592d500f24ca0bbddfc5f1608b4294da87783bb84c44a6cd5cb7e2d57bb93",
    "manifest.json": "12c964159a028c9c25d940b6bc156af2af55202c892f531c0ae9c9b34eed97eb",
    "summary.json": "eb708e27a9d9314780984f7e7265a79f5203d9a3b1fd7a072408f8863b7455a7",
    "seal.json": "4806fe5ad59315c3d170788ecbd6da8f2e50373f53a342151501f4d6dcd9f248",
}
DOCKET_SELECTION_SHA256 = "ec481d8db07165185fecc61bf437d868bfcbe4db6f4938a62b6c344e7000c2ee"
M1_SCHEMA = "sireto-v4.12-r30-review-collection-synthetic-m1-1"
M1_MANIFEST_SCHEMA = "sireto-v4.12-r30-review-collection-synthetic-m1-manifest-1"
FORBIDDEN_IDENTITY_TOKENS = frozenset(
    {"candidate", "confidence", "correct", "ground_truth", "hit", "label", "match", "rank", "score", "siren", "siret", "target", "top1"}
)
_NETWORK_AUDIT_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "sireto_v412_synthetic_network_audit_active", default=False
)
_NETWORK_AUDIT_INSTALLED = False


class FixtureError(RuntimeError):
    """The synthetic fixture or one of its deterministic invariants failed."""


def _network_audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _NETWORK_AUDIT_ACTIVE.get() or event != "socket.__new__":
        return
    family = args[1] if len(args) > 1 else None
    if family in {socket.AF_INET, socket.AF_INET6}:
        raise FixtureError("AF_INET/AF_INET6 is forbidden in the synthetic M1 run")


def _install_network_audit_hook() -> None:
    global _NETWORK_AUDIT_INSTALLED
    if not _NETWORK_AUDIT_INSTALLED:
        sys.addaudithook(_network_audit_hook)
        _NETWORK_AUDIT_INSTALLED = True


@contextmanager
def network_audit_scope():
    """Reject Python IPv4/IPv6 socket creation during one local build."""

    _install_network_audit_hook()
    token = _NETWORK_AUDIT_ACTIVE.set(True)
    try:
        yield
    finally:
        _NETWORK_AUDIT_ACTIVE.reset(token)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = canonical_json(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise FixtureError("short synthetic JSON write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_fixture(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = Path(path).read_bytes()
    if Path(path) == DEFAULT_FIXTURE and sha256_bytes(raw) != FIXTURE_SHA256:
        raise FixtureError("canonical synthetic fixture hash mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FixtureError("synthetic fixture is not valid UTF-8 JSON") from exc
    if type(value) is not dict or value.get("schema_version") != (
        "sireto-v4.12-r30-m1-m2-integration-fixture-1"
    ):
        raise FixtureError("synthetic fixture schema mismatch")
    required = {
        "expected_materialized_transport_sha256", "expected_sirene_snapshot_sha256",
        "identity_label_blind", "role", "search_response", "sirene_records",
    }
    if set(value) != required | {"schema_version"}:
        raise FixtureError("integration fixture has an open or incomplete schema")
    if value["identity_label_blind"] is not True or value["sirene_records"] != []:
        raise FixtureError("M1 fixture must be identity-blind and contain no SIRENE truth")
    lowered = raw.decode("utf-8").casefold()
    if any(token in lowered for token in ("candidate_siret", "predicted_siret", "top1", "ground_truth")):
        raise FixtureError("candidate-bearing field leaked into the fixture capability")
    response = value["search_response"]
    if set(response) != {"addresses", "body_base64", "headers", "http_status", "terminal"}:
        raise FixtureError("search response fixture schema mismatch")
    if response["terminal"] != "RESPONSE" or response["http_status"] != 200:
        raise FixtureError("integration fixture must be the pinned empty success response")
    try:
        body = base64.b64decode(response["body_base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise FixtureError("fixture body is not strict base64") from exc
    if body != b"<html></html>":
        raise FixtureError("integration fixture body must contain no result or evidence")
    return value, raw


def _load_policy() -> tuple[dict[str, Any], list[ipaddress._BaseNetwork]]:
    policy_raw = POLICY_PATH.read_bytes()
    if sha256_bytes(policy_raw) != replay.POLICY_SHA256:
        raise FixtureError("collection policy hash mismatch")
    policy = json.loads(policy_raw)
    dns_vectors = json.loads(DNS_VECTORS_PATH.read_text(encoding="utf-8"))
    networks = [
        ipaddress.ip_network(value, strict=True)
        for value in dns_vectors["forbidden_cidrs"]
    ]
    return policy, networks


def _journal_result(
    journal: runtime.AccessJournal,
    intent: runtime.Intent,
    *,
    outcome: str,
    error_type: str | None = None,
    http_status: int | None = None,
    payload: bytes | None = None,
) -> None:
    fields: dict[str, Any] = {
        "outcome": outcome,
        "error_type": error_type,
        "http_status": http_status,
    }
    if payload is not None and outcome == "SUCCESS":
        fields["byte_count"] = len(payload)
        fields["content_sha256"] = sha256_bytes(payload)
    journal.result(intent, **fields)


def _quota_decisions(rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    per_query: dict[int, int] = {}
    dossier_count = 0
    seen_domains: dict[str, tuple[int, int]] = {}
    output: list[dict[str, Any]] = []
    query_limit = policy["page_fetch"]["maximum_open_attempts_per_query"]
    dossier_limit = policy["page_fetch"]["maximum_open_attempts_per_dossier"]
    for row in sorted(rows, key=lambda value: (value["query_ordinal"], value["result_rank"])):
        query = row["query_ordinal"]
        domain = row["registrable_domain"]
        first_seen = seen_domains.get(domain)
        if not row["admissible"]:
            decision = "SKIP_INADMISSIBLE"
        elif first_seen is not None:
            decision = "SKIP_DUPLICATE_DOMAIN"
        elif dossier_count >= dossier_limit:
            decision = "SKIP_DOSSIER_QUOTA"
        elif per_query.get(query, 0) >= query_limit:
            decision = "SKIP_QUERY_QUOTA"
        else:
            decision = "OPEN_ATTEMPT"
            per_query[query] = per_query.get(query, 0) + 1
            dossier_count += 1
            seen_domains[domain] = (query, row["result_rank"])
        output.append(
            {
                "decision": decision,
                "domain_first_seen": list(first_seen) if decision == "SKIP_DUPLICATE_DOMAIN" else None,
                "query_ordinal": query,
                "registrable_domain": domain,
                "result_rank": row["result_rank"],
                "url": row["url"],
            }
        )
    return output


def _sirene_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["record_count"] != 1:
            disposition = "NONUNIQUE"
        elif record["administrative_state"] != "A":
            disposition = "CLOSED"
        else:
            disposition = "ACTIVE_UNIQUE"
        rows.append(
            {
                "administrative_state": record["administrative_state"],
                "disposition": disposition,
                "record_count": record["record_count"],
                "siret": record["siret"],
            }
        )
    return rows


def _tree_records(root: Path, excluded: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        info = path.stat()
        rows.append(
            {
                "mode": stat.S_IMODE(info.st_mode),
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": info.st_size,
            }
        )
    return rows


def _copy_pinned(source: Path, destination: Path, expected_sha256: str) -> None:
    if sha256_file(source) != expected_sha256:
        raise FixtureError(f"canonical docket hash mismatch: {source}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                offset = 0
                while offset < len(chunk):
                    written = os.write(descriptor, chunk[offset:])
                    if written <= 0:
                        raise FixtureError("short canonical docket copy")
                    offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if sha256_file(destination) != expected_sha256:
        raise FixtureError("copied docket capability differs from source")


def _run_native_batch_worker(role: str, payload: bytes) -> bytes:
    """Run one authenticated native worker over one complete phase batch."""

    if role not in {"IDENTITY_DISCOVERY", "FROZEN_CANDIDATE_COMPARISON"}:
        raise FixtureError("unknown M1 batch worker role")
    if len(payload) > runtime.MAX_WORKER_MESSAGE_BYTES:
        raise FixtureError("M1 native batch exceeds the closed worker limit")
    runtime._assert_closed_runtime_roots()
    worker_fd = runtime._open_native_worker(runtime.NATIVE_WORKER_PATH)
    sandbox_fd = runtime._open_authenticated_regular(runtime.SANDBOX_EXEC_PATH)
    descriptors = [worker_fd, sandbox_fd]
    staging: Path | None = None
    worker_copy: Path | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        if runtime._sha256_fd(sandbox_fd) != runtime.PINNED_NATIVE_HASHES["sandbox_exec_sha256"]:
            raise FixtureError("sandbox-exec differs from the pinned trust anchor")
        staging = Path(tempfile.mkdtemp(prefix="sireto-v412-m1-batch-", dir="/private/tmp"))
        worker_copy = staging / runtime.NATIVE_WORKER_BASENAME
        copy_fd = os.open(
            worker_copy,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o500,
        )
        descriptors.append(copy_fd)
        size = os.fstat(worker_fd).st_size
        offset = 0
        while offset < size:
            chunk = os.pread(worker_fd, min(1024 * 1024, size - offset), offset)
            if not chunk:
                raise FixtureError("native worker changed while staging M1 batch")
            runtime._write_fd_all(copy_fd, chunk)
            offset += len(chunk)
        os.fsync(copy_fd)
        os.close(copy_fd)
        descriptors.remove(copy_fd)
        staged_fd = runtime._open_authenticated_regular(worker_copy)
        descriptors.append(staged_fd)
        if runtime._sha256_fd(staged_fd) != runtime.PINNED_NATIVE_HASHES["artifact_sha256"]:
            raise FixtureError("staged M1 native worker hash mismatch")
        staging_fd = os.open(
            staging,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptors.append(staging_fd)
        pipes: list[tuple[int, int]] = []
        for _ in range(4):
            pair = os.pipe()
            pipes.append(pair)
            descriptors.extend(pair)
        (input_read, input_write), (output_read, output_write), (
            ready_read, ready_write,
        ), (gate_read, gate_write) = pipes
        process = subprocess.Popen(
            [
                os.fspath(runtime.SANDBOX_EXEC_PATH), "-p",
                runtime._sandbox_profile(worker_copy), os.fspath(worker_copy), role,
                str(input_read), str(output_write), str(ready_write), str(gate_read),
            ],
            close_fds=True,
            pass_fds=(input_read, output_write, ready_write, gate_read),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for descriptor in (input_read, output_write, ready_write, gate_read):
            os.close(descriptor)
            descriptors.remove(descriptor)
        deadline = time.monotonic() + runtime.WORKER_TIMEOUT_SECONDS
        if not select.select([ready_read], [], [], runtime.WORKER_TIMEOUT_SECONDS)[0] or os.read(ready_read, 1) != b"R":
            raise FixtureError(f"M1 native batch worker {role} READY failure")
        linked = os.stat(worker_copy, follow_symlinks=False)
        retained = os.fstat(staged_fd)
        if (
            (linked.st_dev, linked.st_ino) != (retained.st_dev, retained.st_ino)
            or runtime._sha256_fd(staged_fd) != runtime.PINNED_NATIVE_HASHES["artifact_sha256"]
        ):
            raise FixtureError("M1 native worker changed across spawn")
        os.unlink(worker_copy)
        os.fsync(staging_fd)
        runtime._write_fd_all_before(gate_write, b"G", deadline)
        os.close(gate_write)
        descriptors.remove(gate_write)
        runtime._write_fd_all_before(input_write, payload, deadline)
        os.close(input_write)
        descriptors.remove(input_write)
        native_output = runtime._read_fd_all_before(output_read, deadline, 68)
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
        if process.returncode != 0:
            raise FixtureError(f"M1 native batch worker failed rc={process.returncode}")
        expected_size = 32 if role == "IDENTITY_DISCOVERY" else 68
        if len(native_output) != expected_size:
            raise FixtureError("M1 native batch worker response shape mismatch")
        return native_output
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if worker_copy is not None:
            try:
                os.unlink(worker_copy)
            except FileNotFoundError:
                pass
        if staging is not None:
            try:
                os.rmdir(staging)
            except FileNotFoundError:
                pass


def _build_m1(root: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise M1 boundaries and the M2 broker without claiming M3 collection."""

    root.mkdir(mode=0o700, exist_ok=True)
    control = root / "control"
    identity_root = root / "identity_capability"
    fixture_root = root / "fixture_capability"
    comparison_root = root / "comparison_capability"
    sirene_root = root / "sirene_capability"
    for directory in (control, identity_root, fixture_root, comparison_root, sirene_root):
        directory.mkdir(mode=0o700)

    for relative in ("manifest.json", "summary.json", "seal.json"):
        if sha256_file(DOCKET_ROOT / relative) != DOCKET_HASHES[relative]:
            raise FixtureError(f"canonical docket control hash mismatch: {relative}")
    source_summary = json.loads((DOCKET_ROOT / "summary.json").read_text(encoding="utf-8"))
    source_seal = json.loads((DOCKET_ROOT / "seal.json").read_text(encoding="utf-8"))
    if (
        source_summary.get("selection_sha256") != DOCKET_SELECTION_SHA256
        or source_summary.get("label_columns_deserialized") != []
        or source_summary.get("label_semantics_opened") is not False
        or source_seal.get("tree_sha256")
        != "8a9aade72e741e393bdd5647ae440f38793da879462b185640dbf8ac6cf02df0"
    ):
        raise FixtureError("canonical docket control claims differ from the frozen source")

    # Only identity capabilities exist before revocation.  In particular, neither
    # comparison Parquet is copied, opened, or used to materialize the fixture.
    for name, relative in {
        "identity_discovery.parquet": "identity/identity_discovery.parquet",
        "collection_plan.parquet": "identity/collection_plan.parquet",
    }.items():
        _copy_pinned(DOCKET_ROOT / relative, identity_root / name, DOCKET_HASHES[relative])
    plan = m2broker.load_canonical_plan_from_parquets(
        identity_root / "identity_discovery.parquet",
        identity_root / "collection_plan.parquet",
    )
    if len(plan.rows) != 90 or len({row.query_id for row in plan.rows}) != 30:
        raise FixtureError("M2 did not authenticate the exact 30/90 identity plan")

    fixture, fixture_raw = _load_fixture(DEFAULT_FIXTURE)
    _copy_pinned(DEFAULT_FIXTURE, fixture_root / "fixture.json", FIXTURE_SHA256)
    response = fixture["search_response"]
    body = base64.b64decode(response["body_base64"], validate=True)
    exchanges = tuple(
        m2broker.FixtureExchange(
            "SEARCH",
            runtime.search_attempt_id(row.query_id, row.query_ordinal, row.search_query),
            m2broker.SEARCH_HOST,
            tuple(response["addresses"]),
            response["terminal"],
            response["http_status"],
            tuple(tuple(pair) for pair in response["headers"]),
            (body,),
        )
        for row in plan.rows
    )
    transport = m2broker.SealedFixtureTransport(
        exchanges,
        (),
        fixture["expected_materialized_transport_sha256"],
        fixture["expected_sirene_snapshot_sha256"],
    )
    audit_events: list[dict[str, str]] = []

    def audit_hook(operation: str, target: str) -> None:
        audit_events.append({"operation": operation, "target": target})

    phases = ["IDENTITY_CAPABILITIES_OPENED", "EXTERNAL_FIXTURE_AUTHENTICATED"]
    journal = runtime.AccessJournal(control / "journal")
    try:
        journal.state_transition(phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN")
        m2 = m2broker.InjectedOfflineBroker(
            journal=journal,
            plan=plan.rows,
            transport=transport,
            expected_transport_fixture_sha256=fixture["expected_materialized_transport_sha256"],
            expected_sirene_snapshot_sha256=fixture["expected_sirene_snapshot_sha256"],
            policy_bytes=POLICY_PATH.read_bytes(),
            public_suffix_bytes=PUBLIC_SUFFIX_PATH.read_bytes(),
            dns_vectors_bytes=DNS_VECTORS_PATH.read_bytes(),
            archive_store=m2broker.ExclusiveArchiveStore(identity_root / "raw", audit_hook),
            audit_hook=audit_hook,
        )
        search_rows = []
        broker_fact_rows: list[dict[str, Any]] = []
        for row in plan.rows:
            result = m2.search(row.query_id, row.query_ordinal)
            if result.status != "SUCCESS" or result.results or result.archive is None:
                raise FixtureError("external empty fixture produced evidence or a failed search")
            search_rows.append(
                {
                    "archive_sha256": result.archive.content_sha256,
                    "query_id": result.query_id,
                    "query_ordinal": result.query_ordinal,
                    "result_count": 0,
                    "search_attempt_id": result.search_attempt_id,
                    "status": result.status,
                }
            )
        phases.append("M2_EMPTY_SEARCH_FIXTURE_CONSUMED")
        lookup_plan = [
            {"query_id": query_id, "siret": siret}
            for query_id, siret in m2.sirene_lookup_plan
        ]
        if broker_fact_rows or lookup_plan:
            raise FixtureError("empty external fixture unexpectedly produced proof or lookup")

        identity_payload = canonical_json(
            {
                "plan_sha256": plan.canonical_sha256,
                "queries": [row.projection() for row in plan.rows],
                "transport_fixture_sha256": transport.fixture_sha256,
            }
        )
        native_identity = _run_native_batch_worker("IDENTITY_DISCOVERY", identity_payload)
        _write_json(identity_root / "search_responses.json", search_rows)
        _write_json(identity_root / "facts.json", broker_fact_rows)
        _write_json(sirene_root / "lookup_plan.json", lookup_plan)
        _write_json(control / "broker_audit_events.json", audit_events)
        identity_artifact = {
            "broker_implementation": "InjectedOfflineBroker",
            "facts_count": len(broker_fact_rows),
            "native_business_logic_executed": False,
            "native_protocol_digest_sha256": native_identity.hex(),
            "plan_sha256": plan.canonical_sha256,
            "search_archive_count": len(search_rows),
            "search_result_count": 0,
            "sirene_lookup_count": len(lookup_plan),
            "transport_fixture_sha256": transport.fixture_sha256,
            "worker_invocation_count": 1,
        }
        _write_json(identity_root / "artifact.json", identity_artifact)
        _write_json(
            identity_root / "seal.json",
            {
                "artifact_sha256": sha256_file(identity_root / "artifact.json"),
                "facts_sha256": sha256_file(identity_root / "facts.json"),
                "lookup_plan_sha256": sha256_file(sirene_root / "lookup_plan.json"),
                "schema_version": "sireto-v4.12-r30-m1-m2-identity-seal-1",
            },
        )
        phases.append("IDENTITY_SEALED")
        m2.revoke()
        if m2.state != "IDENTITY_SEALED_NETWORK_REVOKED":
            raise FixtureError("M2 revocation did not become terminal")
        phases.append("IDENTITY_SEALED_NETWORK_REVOKED")

        # Comparison material appears only after the irreversible M2 transition.
        for name, relative in {
            "docket.parquet": "comparison/docket.parquet",
            "candidate_context.parquet": "comparison/candidate_context.parquet",
        }.items():
            _copy_pinned(DOCKET_ROOT / relative, comparison_root / name, DOCKET_HASHES[relative])
        candidate_table = pq.read_table(comparison_root / "candidate_context.parquet")
        docket_table = pq.read_table(comparison_root / "docket.parquet", columns=["query_id"])
        if candidate_table.num_rows != 3000 or docket_table.num_rows != 30:
            raise FixtureError("post-revoke comparison capability shape mismatch")
        candidate_rows = candidate_table.to_pylist()
        candidates_by_query: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidate_rows:
            candidates_by_query.setdefault(candidate["query_id"], []).append(candidate)
        comparison_inputs = []
        for query_id in sorted(candidates_by_query):
            ordered = sorted(
                candidates_by_query[query_id],
                key=lambda item: (item["ranker_rank"], item["retrieval_rank"], item["candidate_siret"]),
            )
            comparison_inputs.append(
                {"candidate_sirets": [item["candidate_siret"] for item in ordered], "query_id": query_id}
            )
        comparison_payload = canonical_json(comparison_inputs)
        native_comparison = _run_native_batch_worker(
            "FROZEN_CANDIDATE_COMPARISON",
            native_identity + len(comparison_inputs).to_bytes(4, "big") + comparison_payload,
        )
        _write_json(comparison_root / "inputs.json", comparison_inputs)
        _write_json(
            comparison_root / "artifact.json",
            {
                "candidate_count": 3000,
                "dossier_count": 30,
                "native_business_logic_executed": False,
                "native_protocol_digest_sha256": native_comparison[36:].hex(),
                "worker_invocation_count": 1,
            },
        )
        phases.append("POST_REVOKE_COMPARISON_CAPABILITY_OPENED")
        journal_head = journal.verify_complete()
        journal_jsonl = journal.project_jsonl(control / "journal" / "access_journal.jsonl")
    finally:
        journal.close()

    _write_json(control / "phase_sequence.json", phases)
    summary = {
        "candidate_count_post_revoke": 3000,
        "dossier_count": 30,
        "facts_count": 0,
        "identity_worker_count": 1,
        "integration_scope": "M1_BOUNDARY_PLUS_M2_INTEGRATION_HARNESS_NOT_M3",
        "journal_head_sha256": journal_head,
        "journal_jsonl_sha256": journal_jsonl,
        "query_count": 90,
        "search_archive_count": 90,
        "total_collection_worker_count": 2,
        "worker_comparison_count": 1,
    }
    _write_json(root / "summary.json", summary)
    payload_files = _tree_records(root)
    manifest = {
        "artifact_role": "M1_BOUNDARY_PLUS_M2_INTEGRATION_HARNESS",
        "claims": {
            "identity_label_blind": True,
            "no_circular_candidate_proof": True,
            "no_historical_labels": True,
            "native_workers_execute_protocol_digest_only": True,
        },
        "docket": {"input_hashes": DOCKET_HASHES, "selection_sha256": DOCKET_SELECTION_SHA256},
        "fixture": {
            "external_sha256": sha256_bytes(fixture_raw),
            "role": fixture["role"],
            "transport_sha256": transport.fixture_sha256,
        },
        "m2_broker_source_sha256": sha256_file(Path(m2broker.__file__)),
        "output_claim": "NO_M3_NO_SECTION_5_NO_COLLECTION_EVIDENCE_CLAIM",
        "payload_files": payload_files,
        "schema_version": M1_MANIFEST_SCHEMA,
    }
    _write_json(root / "manifest.json", manifest)
    seal = {
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "payload_tree_sha256": sha256_bytes(M1_TREE_DOMAIN + canonical_json(payload_files)),
        "schema_version": "sireto-v4.12-r30-m1-m2-integration-seal-1",
    }
    _write_json(root / "seal.json", seal)
    return {"manifest": manifest, "seal": seal, "summary": summary}


def _publish(destination: Path, producer) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"synthetic destination already exists: {destination}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    try:
        with network_audit_scope():
            result = producer(temporary)
        directory_fd = os.open(temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary.rename(destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return result
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build(destination: Path) -> dict[str, Any]:
    """Build the real M1 harness: exactly two sequential collection workers."""

    policy, _ = _load_policy()
    return _publish(destination, lambda temporary: _build_m1(temporary, policy))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    result = build(args.destination)
    print(canonical_json(result["seal"]).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
