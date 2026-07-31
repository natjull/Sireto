from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.xgb_matcher.v411_scene import V411_ACCEPTOR_FEATURE_NAMES
from src.xgb_matcher import v412_service_bundle as bundle_module
from src.xgb_matcher import v412_service as service_module
from src.xgb_matcher.v412_service_bundle import (
    _capture_bundle_files,
    _capture_exact,
    _json_object,
    _validate_model_controls,
    _validate_runtime,
    load_frozen_v412_service_bundle,
    validate_frozen_v412_service_bundle,
)
from src.xgb_matcher.v412_service_worker import PersistentV412Worker


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


def test_capture_rejects_an_imported_source_from_another_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    replacement = tmp_path / "v411_acceptor.py"
    replacement.write_text("# not the executed frozen source\n")
    monkeypatch.setattr(
        bundle_module._v411_acceptor,
        "__file__",
        str(replacement),
    )
    with pytest.raises(ValueError, match="executed source path changed"):
        _capture_bundle_files()


def test_nested_bundle_state_and_acceptor_classes_are_fail_closed() -> None:
    with load_frozen_v412_service_bundle(include_evidence=False) as bundle:
        bundle.downstream.acceptor.steps[-1][1].coef_[0, 0] += 1.0
        with pytest.raises(ValueError, match="STOP_V412_SERVICE_INTEGRITY"):
            validate_frozen_v412_service_bundle(bundle)

    with load_frozen_v412_service_bundle(include_evidence=False) as bundle:
        bundle.downstream.acceptor.classes_[0] = 7
        with pytest.raises(ValueError, match="mutated service bundle"):
            validate_frozen_v412_service_bundle(bundle)


def test_callback_code_mutation_is_fail_closed() -> None:
    with load_frozen_v412_service_bundle(include_evidence=False) as bundle:
        callback = bundle.retrieval.retriever
        original_code = callback.__code__

        def replacement(*args, **kwargs):
            return None

        try:
            callback.__code__ = replacement.__code__
            with pytest.raises(ValueError, match="mutated service bundle"):
                validate_frozen_v412_service_bundle(bundle)
        finally:
            callback.__code__ = original_code


def test_acceptor_class_and_downstream_method_mutations_are_fail_closed() -> None:
    with load_frozen_v412_service_bundle(include_evidence=False) as bundle:
        acceptor_class = type(bundle.downstream.acceptor)
        original = acceptor_class.predict_proba
        try:
            acceptor_class.predict_proba = lambda self, matrix: np.asarray(
                [[1.0, 0.0]]
            )
            with pytest.raises(ValueError, match="mutated service bundle"):
                validate_frozen_v412_service_bundle(bundle)
        finally:
            acceptor_class.predict_proba = original

    with load_frozen_v412_service_bundle(include_evidence=False) as bundle:
        bundle.downstream.apply_guard_to_trace = lambda **kwargs: None
        with pytest.raises(ValueError, match="mutated service bundle"):
            validate_frozen_v412_service_bundle(bundle)


def test_ranker_class_method_closure_mutation_is_fail_closed() -> None:
    with load_frozen_v412_service_bundle(include_evidence=False) as bundle:
        method = type(bundle.downstream.ranker).predict
        cell = next(
            closure_cell
            for closure_cell in (method.__closure__ or ())
            if callable(closure_cell.cell_contents)
        )
        original = cell.cell_contents

        def replacement(*args, **kwargs):
            return np.zeros(len(args[1]), dtype=np.float32)

        try:
            cell.cell_contents = replacement
            with pytest.raises(
                ValueError,
                match="mutated service bundle",
            ):
                validate_frozen_v412_service_bundle(bundle)
        finally:
            cell.cell_contents = original


def test_guard_global_callable_mutation_is_fail_closed() -> None:
    with load_frozen_v412_service_bundle(include_evidence=True) as bundle:
        original = service_module.apply_guard
        try:
            service_module.apply_guard = lambda **kwargs: (
                kwargs["decision_v411"],
                kwargs["review_reason_v411"],
            )
            with pytest.raises(
                ValueError,
                match="mutated service bundle",
            ):
                validate_frozen_v412_service_bundle(bundle)
        finally:
            service_module.apply_guard = original


@pytest.mark.parametrize(
    ("name", "replacement"),
    (
        (
            "V411_ACCEPTOR_FEATURE_NAMES",
            tuple(reversed(service_module.V411_ACCEPTOR_FEATURE_NAMES)),
        ),
        (
            "FORBIDDEN_FIELDS",
            frozenset(),
        ),
    ),
)
def test_service_policy_global_mutation_is_fail_closed(
    name: str,
    replacement: object,
) -> None:
    with load_frozen_v412_service_bundle(include_evidence=True) as bundle:
        original = getattr(service_module, name)
        try:
            setattr(service_module, name, replacement)
            with pytest.raises(
                ValueError,
                match="mutated service bundle",
            ):
                validate_frozen_v412_service_bundle(bundle)
        finally:
            setattr(service_module, name, original)


@pytest.mark.parametrize("target", ("ranker", "retrieval", "evidence"))
def test_instance_method_override_is_fail_closed(target: str) -> None:
    with load_frozen_v412_service_bundle(include_evidence=True) as bundle:
        if target == "ranker":
            bundle.downstream.ranker.predict = lambda matrix: np.zeros(
                len(matrix)
            )
        elif target == "retrieval":
            bundle.retrieval.build = lambda query: None
        else:
            assert bundle.evidence is not None
            bundle.evidence.build = lambda query: None
        with pytest.raises(ValueError, match="STOP_V412_SERVICE_INTEGRITY"):
            validate_frozen_v412_service_bundle(bundle)


@pytest.mark.parametrize("target", ("ranker_config", "evidence_config"))
def test_existing_configuration_value_mutation_is_fail_closed(
    target: str,
) -> None:
    with load_frozen_v412_service_bundle(include_evidence=True) as bundle:
        if target == "ranker_config":
            bundle.downstream.ranker.missing = 0.0
        else:
            assert bundle.evidence is not None
            bundle.evidence.max_index_cache_entries = 1
        with pytest.raises(ValueError, match="mutated service bundle"):
            validate_frozen_v412_service_bundle(bundle)


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


def test_real_bundle_attestation_is_stable_across_ten_requests() -> None:
    query = (
        pd.read_parquet(REFERENCE / "queries.parquet")
        .query("query_id == '10014'")
        .iloc[0]
        .to_dict()
    )
    with load_frozen_v412_service_bundle(include_evidence=True) as bundle:
        worker = PersistentV412Worker(bundle=bundle, mode="v412g")
        decisions = []
        for _index in range(10):
            decisions.append(worker.process(query).v412.decision_v412)
            validate_frozen_v412_service_bundle(bundle)
    assert decisions == ["AUTO_MATCH"] * 10
