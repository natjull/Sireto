from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket

import pyarrow.parquet as pq
import pytest

from scripts import build_v412_review_collection_synthetic_fixture as builder


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("m1-m2-integration") / "run"
    builder.build(destination)
    return destination


def test_external_fixture_is_identity_blind_empty_and_externally_pinned() -> None:
    raw = builder.DEFAULT_FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == builder.FIXTURE_SHA256
    fixture = json.loads(raw)
    assert fixture["identity_label_blind"] is True
    assert fixture["role"] == "EXTERNAL_EMPTY_WEB_CAPABILITY_NOT_EVIDENCE"
    assert fixture["sirene_records"] == []
    assert fixture["search_response"]["body_base64"] == "PGh0bWw+PC9odG1sPg=="
    lowered = raw.decode("utf-8").casefold()
    assert not any(token in lowered for token in (
        "candidate_siret", "predicted_siret", "top1", "ground_truth",
    ))


def test_audit_hook_rejects_ipv4_and_ipv6() -> None:
    with builder.network_audit_scope():
        with pytest.raises(builder.FixtureError, match="AF_INET/AF_INET6"):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(builder.FixtureError, match="AF_INET/AF_INET6"):
            socket.socket(socket.AF_INET6, socket.SOCK_STREAM)


def test_run_exercises_real_m2_broker_without_m3_claim(built: Path) -> None:
    summary = load(built / "summary.json")
    manifest = load(built / "manifest.json")
    artifact = load(built / "identity_capability/artifact.json")
    assert summary == {
        **summary,
        "candidate_count_post_revoke": 3000,
        "dossier_count": 30,
        "facts_count": 0,
        "identity_worker_count": 1,
        "integration_scope": "M1_BOUNDARY_PLUS_M2_INTEGRATION_HARNESS_NOT_M3",
        "query_count": 90,
        "search_archive_count": 90,
        "total_collection_worker_count": 2,
        "worker_comparison_count": 1,
    }
    assert artifact["broker_implementation"] == "InjectedOfflineBroker"
    assert artifact["native_business_logic_executed"] is False
    assert artifact["facts_count"] == artifact["sirene_lookup_count"] == 0
    assert manifest["output_claim"] == "NO_M3_NO_SECTION_5_NO_COLLECTION_EVIDENCE_CLAIM"
    assert manifest["artifact_role"] == "M1_BOUNDARY_PLUS_M2_INTEGRATION_HARNESS"
    assert manifest["m2_broker_source_sha256"] == builder.sha256_file(
        Path(builder.m2broker.__file__)
    )


def test_only_identity_and_external_fixture_capabilities_exist_before_revoke(built: Path) -> None:
    phases = load(built / "control/phase_sequence.json")
    assert phases == [
        "IDENTITY_CAPABILITIES_OPENED",
        "EXTERNAL_FIXTURE_AUTHENTICATED",
        "M2_EMPTY_SEARCH_FIXTURE_CONSUMED",
        "IDENTITY_SEALED",
        "IDENTITY_SEALED_NETWORK_REVOKED",
        "POST_REVOKE_COMPARISON_CAPABILITY_OPENED",
    ]
    identity_schema = pq.read_schema(
        built / "identity_capability/identity_discovery.parquet"
    ).names
    assert "predicted_siret" not in identity_schema
    assert "candidate_siret" not in identity_schema
    source = Path(builder.__file__).read_text(encoding="utf-8")
    assert "class LocalFixtureBroker" not in source
    assert "m2broker.InjectedOfflineBroker(" in source


def test_m2_archives_all_90_searches_and_produces_no_circular_proof(built: Path) -> None:
    rows = load(built / "identity_capability/search_responses.json")
    assert len(rows) == 90
    assert all(row["status"] == "SUCCESS" and row["result_count"] == 0 for row in rows)
    archives = sorted((built / "identity_capability/raw/search").glob("*.bin"))
    assert len(archives) == 90
    assert all(path.read_bytes() == b"<html></html>" for path in archives)
    assert load(built / "identity_capability/facts.json") == []
    assert load(built / "sirene_capability/lookup_plan.json") == []
    manifest = load(built / "manifest.json")
    assert manifest["claims"] == {
        "identity_label_blind": True,
        "native_workers_execute_protocol_digest_only": True,
        "no_circular_candidate_proof": True,
        "no_historical_labels": True,
    }


def test_candidate_context_is_used_only_for_post_revoke_comparison(built: Path) -> None:
    source_rows = pq.read_table(
        built / "comparison_capability/candidate_context.parquet"
    ).to_pylist()
    by_query: dict[str, list[dict]] = {}
    for row in source_rows:
        by_query.setdefault(row["query_id"], []).append(row)
    inputs = load(built / "comparison_capability/inputs.json")
    assert len(inputs) == 30
    for item in inputs:
        expected = [
            row["candidate_siret"] for row in sorted(
                by_query[item["query_id"]],
                key=lambda row: (row["ranker_rank"], row["retrieval_rank"], row["candidate_siret"]),
            )
        ]
        assert len(expected) == 100
        assert item["candidate_sirets"] == expected


def test_exactly_two_native_processes_are_launched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = builder.subprocess.Popen
    launches = []

    def counting(*args, **kwargs):
        launches.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(builder.subprocess, "Popen", counting)
    destination = tmp_path / "run"
    builder.build(destination)
    assert len(launches) == 2
    assert load(destination / "summary.json")["total_collection_worker_count"] == 2


def test_comparison_files_are_neither_copied_nor_opened_before_m2_revoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revoked = False
    original_revoke = builder.m2broker.InjectedOfflineBroker.revoke
    original_copy = builder._copy_pinned
    original_read = builder.pq.read_table

    def tracked_revoke(self):
        nonlocal revoked
        result = original_revoke(self)
        revoked = True
        return result

    def tracked_copy(source, destination, expected):
        if "comparison" in source.parts:
            assert revoked
        return original_copy(source, destination, expected)

    def tracked_read(where, *args, **kwargs):
        if isinstance(where, (str, Path)) and "comparison_capability" in Path(where).parts:
            assert revoked
        return original_read(where, *args, **kwargs)

    monkeypatch.setattr(builder.m2broker.InjectedOfflineBroker, "revoke", tracked_revoke)
    monkeypatch.setattr(builder, "_copy_pinned", tracked_copy)
    monkeypatch.setattr(builder.pq, "read_table", tracked_read)
    builder.build(tmp_path / "boundary")
    assert revoked


def test_manifest_and_seal_recompute_exactly(built: Path) -> None:
    manifest = load(built / "manifest.json")
    seal = load(built / "seal.json")
    observed = builder._tree_records(built, frozenset({"manifest.json", "seal.json"}))
    assert manifest["payload_files"] == observed
    assert seal["manifest_sha256"] == hashlib.sha256(
        (built / "manifest.json").read_bytes()
    ).hexdigest()
    assert seal["payload_tree_sha256"] == hashlib.sha256(
        builder.M1_TREE_DOMAIN + builder.canonical_json(observed)
    ).hexdigest()


def test_two_builds_are_byte_identical(tmp_path: Path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    builder.build(first)
    builder.build(second)
    assert file_bytes(first) == file_bytes(second)
    with pytest.raises(FileExistsError):
        builder.build(first)
