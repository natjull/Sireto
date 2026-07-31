from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.xgb_matcher.v411_scene import V411_ACCEPTOR_FEATURE_NAMES
from src.xgb_matcher.v412_service_bundle import (
    _capture_bundle_files,
    _capture_exact,
    _json_object,
    _validate_model_controls,
    _validate_runtime,
    load_frozen_v412_service_bundle,
    validate_frozen_v412_service_bundle,
)


REFERENCE = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/references/"
    "v4_12_service_parity/b4b7fef24c5e7036"
)


def test_capture_exact_rejects_hash_and_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    digest = hashlib.sha256(b"{}").hexdigest()

    assert _capture_exact(target, digest) == b"{}"
    with pytest.raises(ValueError, match="hash changed"):
        _capture_exact(target, "0" * 64)

    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="contains a symlink"):
        _capture_exact(link, digest)

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    nested = real_parent / "nested.json"
    nested.write_bytes(b"{}")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="contains a symlink"):
        _capture_exact(linked_parent / "nested.json", digest)


def test_model_control_mutation_is_fail_closed() -> None:
    payloads = _capture_bundle_files()
    metadata = json.loads(payloads["acceptor_metadata"])
    metadata["threshold"] += 0.01
    payloads["acceptor_metadata"] = json.dumps(metadata).encode()
    stack = json.loads(payloads["stack_manifest"])
    stack["components"]["acceptor"]["metadata_sha256"] = hashlib.sha256(
        payloads["acceptor_metadata"]
    ).hexdigest()
    payloads["stack_manifest"] = json.dumps(stack).encode()

    with pytest.raises(ValueError, match="acceptor metadata changed"):
        _validate_model_controls(payloads)


def test_runtime_and_bundle_attestation_are_fail_closed() -> None:
    payloads = _capture_bundle_files()
    certification = _json_object(
        payloads["certification_manifest"],
        "certification",
    )
    ranker = _json_object(payloads["ranker_manifest"], "ranker")
    certification["runtime"]["numpy"] = "0.0.0"
    with pytest.raises(ValueError, match="runtime changed"):
        _validate_runtime(certification, ranker)
    with pytest.raises(ValueError, match="unattested"):
        validate_frozen_v412_service_bundle(object())


def test_real_frozen_bundle_reproduces_all_downstream_stages() -> None:
    query = (
        pd.read_parquet(REFERENCE / "queries.parquet")
        .query("query_id == '10014'")
        .iloc[0]
        .to_dict()
    )
    expected_ranker = (
        pd.read_parquet(REFERENCE / "ranker_reference.parquet")
        .query("query_id == '10014'")
        .sort_values("ranker_rank", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_scene = (
        pd.read_parquet(REFERENCE / "scenes_reference.parquet")
        .query("query_id == '10014'")
        .iloc[0]
    )
    expected_acceptor = (
        pd.read_parquet(REFERENCE / "acceptor_reference.parquet")
        .query("query_id == '10014'")
        .iloc[0]
    )
    expected_guard = (
        pd.read_parquet(REFERENCE / "guard_reference.parquet")
        .query("query_id == '10014'")
        .iloc[0]
    )

    with load_frozen_v412_service_bundle(include_evidence=True) as bundle:
        assert bundle.evidence is not None
        retrieval = bundle.retrieval.build(query)
        evidence = bundle.evidence.build(query)
        trace = bundle.downstream.infer_one(
            query=query,
            candidates=retrieval.candidates,
            direct_evidence=evidence.query,
        )

    assert trace.scored_candidates["candidate_siret"].tolist() == (
        expected_ranker["candidate_siret"].tolist()
    )
    assert np.array_equal(
        trace.scored_candidates["ranker_score"].to_numpy(dtype=np.float32),
        expected_ranker["ranker_score"].to_numpy(dtype=np.float32),
    )
    assert all(
        np.float64(trace.scene[name]).tobytes()
        == np.float64(expected_scene[name]).tobytes()
        for name in V411_ACCEPTOR_FEATURE_NAMES
    )
    assert trace.predicted_siret == expected_scene["predicted_siret"]
    assert trace.acceptor_score == expected_acceptor["score"]
    assert trace.decision_v411 == expected_acceptor["decision"]
    assert trace.decision_v412 == expected_guard["decision_v412"]
    assert trace.review_reason_v412 == expected_guard["review_reason_v412"]
