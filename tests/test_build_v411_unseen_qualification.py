from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from scripts import build_v411_unseen_qualification as subject


def _queries(count: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": f"q{index}",
                "crm_record_id": f"crm{index}",
                "crm_name": f"ENTREPRISE {index}",
                "crm_address": f"{index + 1} RUE ALPHA",
                "crm_postcode": "75001",
                "crm_city": "PARIS",
                "crm_insee": "75056",
            }
            for index in range(count)
        ],
        columns=subject.QUERY_COLUMNS,
    )


def _direct(query_id: str, siret: str) -> dict:
    return {
        "query_id": query_id,
        "split": "descriptive_unseen",
        "partition_key": "insee:75056",
        "candidate_siret": siret,
        "candidate_siren": siret[:9],
        "candidate_state": "A",
        "candidate_names_json": '["ENTREPRISE"]',
        "candidate_address_hash": "1|RUE ALPHA",
        "exact_name_anchor": True,
        "exact_address_anchor": True,
        "direct_evidence_class": "NAME_AND_ADDRESS",
    }


def test_frozen_schemas_and_policy_are_exact():
    assert subject.POLICY_VERSION == "active-direct-current-v4.0"
    assert subject.SANITIZED_SCHEMA_VERSION == (
        "sireto-v4.11-descriptive-unseen-sanitized-1"
    )
    assert subject.SANITIZED_QUERIES_FILENAME == "queries_sanitized.parquet"
    assert len(subject.QUERY_COLUMNS) == 7
    assert set(subject.LABEL_KINDS) == {
        "MATCH_EXACT",
        "AMBIGUOUS",
        "UNRESOLVED",
    }
    assert "NO_MATCH" not in subject.LABEL_KINDS


def test_sanitized_schema_rejects_any_extra_or_empty_field():
    validated = subject.validate_query_schema(_queries(), expected_count=3)
    assert list(validated.columns) == subject.QUERY_COLUMNS

    leaked = _queries()
    leaked["ranker_score"] = 0.0
    with pytest.raises(ValueError, match="schema changed"):
        subject.validate_query_schema(leaked, expected_count=3)

    empty = _queries()
    empty.loc[0, "crm_name"] = ""
    with pytest.raises(ValueError, match="empty sanitized field"):
        subject.validate_query_schema(empty, expected_count=3)


def test_load_sanitized_artifact_reads_only_seven_column_file(tmp_path):
    queries = _queries()
    identity = {"frozen": "sanitized"}
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    root = tmp_path / build_id
    root.mkdir()
    path = root / subject.SANITIZED_QUERIES_FILENAME
    queries.to_parquet(path, index=False)
    manifest = {
        "schema_version": subject.SANITIZED_SCHEMA_VERSION,
        "experiment_id": subject.SANITIZED_EXPERIMENT_ID,
        "build_id": build_id,
        "build_identity": identity,
        "invariants": {
            "forbidden_source_columns_loaded": False,
            "labels_loaded": False,
            "retrieval_or_model_loaded": False,
            "input_siret_or_siren_exposed": False,
        },
        "outputs": {
            subject.SANITIZED_QUERIES_FILENAME: {
                "sha256": subject.file_sha256(path),
                "row_count": 3,
                "columns": subject.QUERY_COLUMNS,
            },
            "sealed_mapping.parquet": {"sha256": "not-opened"},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    loaded_manifest, loaded, observed_hash = subject.load_sanitized_artifact(
        root, expected_count=3
    )

    assert loaded_manifest["build_id"] == build_id
    pd.testing.assert_frame_equal(loaded, queries)
    assert observed_hash == subject.file_sha256(path)


def test_load_sanitized_artifact_rejects_wrong_declared_contract(tmp_path):
    queries = _queries()
    path = tmp_path / subject.SANITIZED_QUERIES_FILENAME
    queries.to_parquet(path, index=False)
    manifest = {
        "schema_version": subject.SANITIZED_SCHEMA_VERSION,
        "experiment_id": "WRONG_EXPERIMENT",
        "outputs": {
            subject.SANITIZED_QUERIES_FILENAME: {
                "sha256": subject.file_sha256(path),
                "row_count": 3,
                "columns": subject.QUERY_COLUMNS,
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="wrong sanitized experiment"):
        subject.load_sanitized_artifact(tmp_path, expected_count=3)


def test_mechanical_cardinality_creates_three_labels(monkeypatch):
    queries = _queries()
    monkeypatch.setattr(subject, "PartitionedCandidateStore", lambda path: object())
    monkeypatch.setattr(
        subject, "_planned_partition_key", lambda row, store: "insee:75056"
    )
    monkeypatch.setattr(subject, "_load_partition", lambda key, store: [{}])
    monkeypatch.setattr(
        subject,
        "build_active_partition_index",
        lambda rows: subject.ActivePartitionIndex({}, {}, 123, 130),
    )

    def fake_find(row, index, *, partition_key):
        if row["query_id"] == "q0":
            return [_direct("q0", "11111111100011")]
        if row["query_id"] == "q1":
            return [
                _direct("q1", "22222222200022"),
                _direct("q1", "22222222200030"),
            ]
        return []

    monkeypatch.setattr(subject, "find_direct_active_candidates", fake_find)
    evidence, labels = subject.qualify_queries(
        queries,
        partitions_dir="unused",
        snapshot_sha256=subject.EXPECTED_SNAPSHOT_SHA256,
    )

    assert list(evidence.columns) == subject.EVIDENCE_COLUMNS
    assert list(labels.columns) == subject.LABEL_COLUMNS
    assert labels["label_kind"].tolist() == [
        "MATCH_EXACT",
        "AMBIGUOUS",
        "UNRESOLVED",
    ]
    assert labels["direct_active_candidate_count"].tolist() == [1, 2, 0]
    assert labels.loc[0, "ground_truth_siret"] == "11111111100011"
    assert labels.loc[1:, "ground_truth_siret"].isna().all()
    assert evidence["active_universe_count"].eq(123).all()


def test_direct_rule_requires_frozen_exact_anchor():
    assert (
        subject._direct_rule(
            {"exact_name_anchor": True, "exact_address_anchor": False}
        )
        == "EXACT_NAME_STRONG_ADDRESS"
    )
    assert (
        subject._direct_rule(
            {"exact_name_anchor": False, "exact_address_anchor": True}
        )
        == "EXACT_ADDRESS_STRONG_NAME"
    )
    with pytest.raises(ValueError, match="lacks an exact anchor"):
        subject._direct_rule(
            {"exact_name_anchor": False, "exact_address_anchor": False}
        )


def test_validation_rejects_nonexact_truth_and_no_match():
    queries = _queries(1)
    evidence = pd.DataFrame(columns=subject.EVIDENCE_COLUMNS)
    labels = pd.DataFrame(
        [
            {
                "query_id": "q0",
                "label_kind": "NO_MATCH",
                "ground_truth_siret": None,
                "ground_truth_siren": None,
                "direct_active_candidate_count": 0,
                "evidence_refs_json": "[]",
                "qualification_reason": "NO_ACTIVE_DIRECT_MATCH",
                "snapshot_sha256": subject.EXPECTED_SNAPSHOT_SHA256,
                "policy_version": subject.POLICY_VERSION,
                "validator": subject.VALIDATOR,
                "human_validated": False,
            }
        ],
        columns=subject.LABEL_COLUMNS,
    )
    with pytest.raises(ValueError, match="forbidden label"):
        subject.validate_qualification(queries, evidence, labels)


def test_validation_rejects_inconsistent_truth_and_evidence_refs(monkeypatch):
    queries = _queries(1)
    monkeypatch.setattr(subject, "PartitionedCandidateStore", lambda path: object())
    monkeypatch.setattr(
        subject, "_planned_partition_key", lambda row, store: "insee:75056"
    )
    monkeypatch.setattr(subject, "_load_partition", lambda key, store: [{}])
    monkeypatch.setattr(
        subject,
        "build_active_partition_index",
        lambda rows: subject.ActivePartitionIndex({}, {}, 1, 1),
    )
    monkeypatch.setattr(
        subject,
        "find_direct_active_candidates",
        lambda row, index, partition_key: [_direct("q0", "11111111100011")],
    )
    evidence, labels = subject.qualify_queries(
        queries,
        partitions_dir="unused",
        snapshot_sha256=subject.EXPECTED_SNAPSHOT_SHA256,
    )
    broken = labels.copy()
    broken.loc[0, "ground_truth_siren"] = "999999999"
    with pytest.raises(ValueError, match="identifiers inconsistent"):
        subject.validate_qualification(queries, evidence, broken)
    broken = labels.copy()
    broken.loc[0, "ground_truth_siret"] = "22222222200022"
    broken.loc[0, "ground_truth_siren"] = "222222222"
    with pytest.raises(ValueError, match="truth differs from evidence"):
        subject.validate_qualification(queries, evidence, broken)
    broken = labels.copy()
    broken.loc[0, "evidence_refs_json"] = "[]"
    with pytest.raises(ValueError, match="evidence refs mismatch"):
        subject.validate_qualification(queries, evidence, broken)


def test_sanitized_link_requires_identity_manifest_and_query_hashes(tmp_path):
    queries = _queries()
    queries_path = tmp_path / subject.SANITIZED_QUERIES_FILENAME
    queries.to_parquet(queries_path, index=False)
    queries_sha = subject.file_sha256(queries_path)
    sanitized = {
        "build_id": "sanitized-build",
        "outputs": {
            subject.SANITIZED_QUERIES_FILENAME: {"sha256": queries_sha}
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(sanitized), encoding="utf-8")
    manifest_sha = subject.file_sha256(manifest_path)
    identity = {
        "sanitized_manifest_sha256": manifest_sha,
        "sanitized_queries_sha256": queries_sha,
        "sanitized_build_id": "sanitized-build",
    }
    inputs = {
        "sanitized_artifact_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
        },
        "sanitized_queries": {
            "path": str(queries_path),
            "sha256": queries_sha,
        },
    }
    subject._validate_sanitized_link(identity, inputs)
    broken = dict(identity)
    broken["sanitized_queries_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="link drift"):
        subject._validate_sanitized_link(broken, inputs)
