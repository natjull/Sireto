from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import build_v45_hard_scenes as subject
from src.xgb_matcher.v41_retrieval import InputSiretState
from src.xgb_matcher.v9_dataset import file_sha256
from src.xgb_matcher.v9_scene import V9_SCENE_FEATURE_NAMES


def _adjudication(**overrides):
    row = {
        "audit_case_id": "case-1",
        "query_id": "service-1",
        "service_id": "service-1",
        "frozen_top1_siret": "12345678900011",
        "adjudication_label": "TOP1_WRONG",
        "evidence_validated": True,
        "acceptor_eligible": True,
        "acceptor_target": 0,
        "sampling_stratum": "TARGETED",
    }
    row.update(overrides)
    return row


def test_bind_adjudication_never_copies_label_on_scene_drift():
    bound = subject.bind_adjudication_to_replay(
        _adjudication(), "99999999900011"
    )

    assert bound["scene_status"] == "SCENE_DRIFT"
    assert bound["label_bound_to_replayed_top1"] is False
    assert bound["scene_adjudication_label"] is None
    assert bound["scene_acceptor_target"] is None
    assert bound["scene_training_eligible"] is False
    assert bound["frozen_adjudication_label"] == "TOP1_WRONG"


@pytest.mark.parametrize(
    ("label", "target"),
    [("TOP1_CORRECT", 1), ("TOP1_WRONG", 0), ("AMBIGUOUS", 0)],
)
def test_bind_adjudication_keeps_valid_label_only_for_exact_same_top1(
    label, target
):
    bound = subject.bind_adjudication_to_replay(
        _adjudication(adjudication_label=label, acceptor_target=target),
        "12345678900011",
    )

    assert bound["scene_status"] == "SCENE_COMPATIBLE"
    assert bound["scene_adjudication_label"] == label
    assert bound["scene_acceptor_target"] == target
    assert bound["scene_training_eligible"] is True


def test_unresolved_is_never_bound_even_without_drift():
    bound = subject.bind_adjudication_to_replay(
        _adjudication(
            adjudication_label="UNRESOLVED",
            acceptor_target=None,
            acceptor_eligible=False,
            evidence_validated=False,
        ),
        "12345678900011",
    )

    assert bound["scene_status"] == "SCENE_COMPATIBLE"
    assert bound["scene_adjudication_label"] is None
    assert bound["scene_acceptor_target"] is None
    assert bound["scene_training_eligible"] is False


def test_join_crm_rows_rejects_conflicting_duplicate_service(tmp_path):
    crm_path = tmp_path / "crm.csv"
    crm_path.write_text(
        "SERVICE ID;SITE\nservice-1;Alpha\nservice-1;Beta\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Ambiguous CRM source rows"):
        subject.join_crm_rows(pd.DataFrame([_adjudication()]), crm_path)


def test_load_canonical_gate_checks_exact_five_artifact_chain(
    tmp_path, monkeypatch
):
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    inputs = []
    provenance = []
    for index in range(5):
        path = tmp_path / f"artifact-{index}"
        inputs.append(
            {
                "path": str(path),
                "manifest_sha256": f"manifest-{index}",
                "adjudications_sha256": f"table-{index}",
                "case_count": 1,
            }
        )
        provenance.append({**inputs[-1], "build_id": f"build-{index}"})
    (gate_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": subject.GATE_SCHEMA_VERSION,
                "build_id": "gate-build",
                "verdict": "STOP_AUTONOMOUS_LABELING",
                "stop_requested": True,
                "inputs": inputs,
                "adjudication_manifest_hashes": [
                    item["manifest_sha256"] for item in inputs
                ],
                "outputs": {"gate_report.json": "placeholder"},
            }
        )
    )
    gate_report = {
        "verdict": "STOP_AUTONOMOUS_LABELING",
        "stop_requested": True,
        "source_status": "EXPLICITLY_EXHAUSTED",
        "metrics": {
            "unique_case_count": 172,
            "evidence_validated_count": 162,
            "acceptor_eligible_count": 162,
            "top1_correct_evidence_validated_count": 114,
            "top1_wrong_evidence_validated_count": 42,
            "unresolved_count": 10,
            "random_case_count": 57,
            "random_evidence_validated_count": 53,
            "label_counts": {
                "AMBIGUOUS": 6,
                "TOP1_CORRECT": 114,
                "TOP1_WRONG": 42,
                "UNRESOLVED": 10,
            },
        },
    }
    (gate_dir / "gate_report.json").write_text(json.dumps(gate_report))
    gate_manifest = json.loads((gate_dir / "manifest.json").read_text())
    gate_manifest["outputs"]["gate_report.json"] = file_sha256(
        gate_dir / "gate_report.json"
    )
    (gate_dir / "manifest.json").write_text(json.dumps(gate_manifest))
    monkeypatch.setattr(
        subject,
        "load_adjudication_artifacts",
        lambda paths: (pd.DataFrame([_adjudication()]), provenance),
    )

    rows, gate, observed = subject.load_canonical_gate(
        gate_dir, enforce_contract_counts=False
    )

    assert len(rows) == 1
    assert gate["verdict"] == "STOP_AUTONOMOUS_LABELING"
    assert len(observed) == 5


def test_link_v43_queue_requires_same_service_top1_and_stratum(
    tmp_path, monkeypatch
):
    queue_path = tmp_path / "queue.parquet"
    pd.DataFrame(
        [
            {
                "audit_case_id": "case-1",
                "service_id": "service-1",
                "top1_siret": "12345678900011",
                "sampling_stratum": "TARGETED",
                "source_row_number": 1,
            }
        ]
    ).to_parquet(queue_path, index=False)
    monkeypatch.setattr(
        subject,
        "EXPECTED_V43_QUEUE_SHA256",
        file_sha256(queue_path),
    )

    linked = subject.link_v43_hard_label_queue(
        pd.DataFrame([_adjudication()]), queue_path
    )

    assert linked.loc[0, "v43_source_row_number"] == 1


def test_v42_input_validation_rejects_changed_partition_signature(
    tmp_path, monkeypatch
):
    partitions = tmp_path / "partitions"
    global_store = tmp_path / "global"
    partitions.mkdir()
    global_store.mkdir()
    state = tmp_path / "state.parquet"
    state.write_bytes(b"state")
    config = subject.V41RetrievalConfig(
        variant=subject.V41RetrievalVariant.B_INPUT_EVIDENCE,
        max_candidates=100,
    )
    manifest = tmp_path / "v42.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": (
                    "sireto-v4.2-representative-retrieval-evaluation-1"
                ),
                "positive_injection": False,
                "retrieval": {
                    "v41_signature": config.signature(),
                    "v41_config": config.to_dict(),
                },
                "inputs": {
                    "partitions": {
                        "path": str(partitions),
                        "runtime_signature": "frozen-partitions",
                    },
                    "global_store": {
                        "path": str(global_store),
                        "runtime_signature": "global-signature",
                    },
                    "current_state_snapshot": {
                        "path": str(state),
                        "sha256": file_sha256(state),
                    },
                },
            }
        )
    )
    monkeypatch.setattr(
        subject,
        "_path_signature",
        lambda path: (
            "changed-partitions"
            if Path(path) == partitions
            else "global-signature"
        ),
    )

    with pytest.raises(ValueError, match="partitions differs"):
        subject.validate_v42_retrieval_inputs(
            experiment_manifest_path=manifest,
            config=config,
            partitions_dir=partitions,
            global_store_path=global_store,
            state_snapshot_path=state,
        )


@dataclass
class _SparseResult:
    gt_was_injected: bool = False
    idf_map: dict[str, float] | None = None
    default_idf: float = 1.0

    def __post_init__(self):
        if self.idf_map is None:
            self.idf_map = {}


class _Retriever:
    def __init__(self):
        self.kwargs = None

    def build(self, **kwargs):
        self.kwargs = kwargs
        candidate = {
            "siret": "12345678900011",
            "siren": "123456789",
            "etat_admin": "A",
            "nom_etablissement": "ALPHA",
            "denomination_unite_legale": "ALPHA",
            "adresse": "1 RUE ALPHA",
            "postcode": "75001",
            "city": "PARIS",
            "insee": "75056",
            "rrf_score": 0.5,
            "retrieval_channel_count": 1,
            "v41_channel_ranks": {"sparse_active": 1},
        }
        return SimpleNamespace(
            sparse_result=_SparseResult(),
            candidates=[candidate],
            input_siret=SimpleNamespace(
                normalized_siret=None, state=InputSiretState.INVALID
            ),
            channels={"sparse_active": [candidate["siret"]]},
        )


class _Ranker:
    def predict(self, matrix):
        return np.ones(len(matrix))


def test_replay_uses_no_positive_and_common_80_feature_scene():
    retriever = _Retriever()
    row = {
        **_adjudication(),
        "SITE": "Alpha",
        "SITE_CLI_ADRESSE": "1 rue Alpha",
        "SITE_CLI_COMMUNE": "Paris",
        "COMMUNE": "Paris",
        "CODE_POSTAL": "75001",
        "CODE_INSEE": "75056",
        "SIRET": "",
    }
    # Obtain the real frozen candidate-feature order from one generated row.
    from scripts.run_v41_shadow import _annotate_candidate
    from scripts.build_v41_training_dataset import build_legacy_55_features
    from src.xgb_matcher.features import (
        make_feature_rows_from_preprocessed,
        preprocess_crm_row,
    )
    from src.xgb_matcher.v41_features import build_v41_candidate_features

    crm = subject.build_crm_row(row)
    candidate = _annotate_candidate(_Retriever().build().candidates[0])
    legacy = make_feature_rows_from_preprocessed(
        preprocess_crm_row(crm), [candidate], include_semantic=False
    )[0]
    feature_order = list(
        {
            **build_legacy_55_features(legacy, candidate),
            **build_v41_candidate_features(candidate, input_siret=None),
        }
    )

    replay, ranked, features = subject.replay_ranked_scene(
        row=row,
        retriever=retriever,
        ranker=_Ranker(),
        ranker_feature_order=feature_order,
        persistent_cache=object(),
    )

    assert retriever.kwargs["gt_siret"] is None
    assert replay["candidate_count"] == 1
    assert replay["replayed_top1_siret"] == "12345678900011"
    assert len(ranked) == 1
    assert list(features) == list(V9_SCENE_FEATURE_NAMES)
    assert len(features) == 80


def test_validate_artifact_rejects_label_on_drift(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    scenes = pd.DataFrame(
        [
            {
                "scene_status": "SCENE_DRIFT",
                "scene_acceptor_target": 0,
                "scene_adjudication_label": "TOP1_WRONG",
                "candidate_count": 1,
                **{name: 0.0 for name in V9_SCENE_FEATURE_NAMES},
            }
        ]
    )
    scenes.to_parquet(artifact / "scene_compatibility.parquet", index=False)
    (artifact / "candidates.parquet").write_bytes(b"candidate-placeholder")
    (artifact / "summary.json").write_text("{}")
    outputs = {
        filename: file_sha256(artifact / filename)
        for filename in (
            "scene_compatibility.parquet",
            "candidates.parquet",
            "summary.json",
        )
    }
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": subject.SCHEMA_VERSION,
                "scene_feature_order": list(V9_SCENE_FEATURE_NAMES),
                "outputs": outputs,
            }
        )
    )

    with pytest.raises(ValueError, match="Drift scene carries"):
        subject.validate_artifact(artifact)


def _compatibility_population():
    rows = []

    def add(stratum, label, count, compatible_count):
        offset = len(rows)
        for index in range(count):
            compatible = index < compatible_count
            rows.append(
                {
                    "audit_case_id": f"case-{offset + index:03d}",
                    "sampling_stratum": stratum,
                    "frozen_adjudication_label": label,
                    "scene_status": (
                        "SCENE_COMPATIBLE" if compatible else "SCENE_DRIFT"
                    ),
                    "label_bound_to_replayed_top1": (
                        compatible and label != "UNRESOLVED"
                    ),
                }
            )

    add("RANDOM_POPULATION", "TOP1_CORRECT", 47, 47)
    add("RANDOM_POPULATION", "TOP1_WRONG", 5, 5)
    add("RANDOM_POPULATION", "AMBIGUOUS", 1, 1)
    add("RANDOM_POPULATION", "UNRESOLVED", 4, 4)
    add("AUTO_NEAR_THRESHOLD", "TOP1_CORRECT", 67, 55)
    add("AUTO_NEAR_THRESHOLD", "TOP1_WRONG", 37, 30)
    add("AUTO_NEAR_THRESHOLD", "AMBIGUOUS", 5, 4)
    add("AUTO_NEAR_THRESHOLD", "UNRESOLVED", 6, 6)
    return pd.DataFrame(rows)


def test_scene_compatibility_gate_applies_all_preregistered_thresholds():
    report = subject.compute_scene_compatibility_gate(
        _compatibility_population()
    )

    assert report["verdict"] == "GO_SCENE_COMPATIBILITY"
    assert all(report["checks"].values())
    assert report["counts"]["random_reliable_compatible"] == 53
    assert report["counts"]["random_negative_compatible"] == 6
    assert report["counts"]["targeted_top1_wrong_compatible"] == 30


def test_scene_compatibility_gate_pivots_when_one_random_negative_drifts():
    scenes = _compatibility_population()
    row = scenes[
        scenes["sampling_stratum"].eq("RANDOM_POPULATION")
        & scenes["frozen_adjudication_label"].eq("TOP1_WRONG")
    ].index[0]
    scenes.loc[row, "scene_status"] = "SCENE_DRIFT"
    scenes.loc[row, "label_bound_to_replayed_top1"] = False

    report = subject.compute_scene_compatibility_gate(scenes)

    assert report["verdict"] == "PIVOT_SCENE_DRIFT"
    assert report["checks"]["all_random_negatives_compatible"] is False
    assert report["training_authorized"] is False
