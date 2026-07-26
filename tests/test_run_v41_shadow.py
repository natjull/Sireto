import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_v41_training_dataset import FEATURE_ORDER
from scripts.run_v41_shadow import (
    V41RuntimeBundle,
    run_shadow,
    score_one_query,
    validate_inventory_chain,
    validate_model_bundle_hashes,
    validate_runtime_contract,
)
from src.xgb_matcher.retrieval import CandidatePoolResult
from src.xgb_matcher.v41_acceptor import V41RawLogisticAcceptor
from src.xgb_matcher.v41_release import V41ReleaseManifest
from src.xgb_matcher.v41_retrieval import (
    InputSiretQualification,
    InputSiretState,
    V41CandidatePoolResult,
    V41RetrievalConfig,
)
from src.xgb_matcher.v41_shadow import build_shadow_inventory
from src.xgb_matcher.v9_dataset import file_sha256
from src.xgb_matcher.v9_scene import V9_SCENE_FEATURE_NAMES


class _Ranker:
    def predict(self, matrix):
        return np.asarray([row[0] for row in matrix], dtype=float)


class _AcceptorModel:
    def predict_proba(self, matrix):
        positive = np.full(len(matrix), 0.99)
        return np.column_stack((1.0 - positive, positive))


class _Retriever:
    def __init__(self, *, fail_on=None):
        self.fail_on = fail_on
        self.calls = []
        self.crm_rows = []

    def build(self, **kwargs):
        query_id = kwargs["crm_row"]["query_id"]
        self.calls.append(query_id)
        self.crm_rows.append(dict(kwargs["crm_row"]))
        if query_id == self.fail_on:
            raise RuntimeError("synthetic interruption")
        input_siret = kwargs["input_siret"]
        candidate = {
            "siret": input_siret,
            "siren": input_siret[:9],
            "denomination": "ALPHA",
            "enseigne1": None,
            "enseigne2": None,
            "enseigne3": None,
            "numeroVoie": "1",
            "typeVoie": "RUE",
            "libelleVoie": "ALPHA",
            "complementAdresse": None,
            "postcode": "01000",
            "city": "BOURG",
            "insee": "01001",
            "etat_admin": "A",
            "is_siege": True,
            "rrf_score": 0.1,
            "retrieval_rank": 1,
            "retrieval_source": "sparse_active+input_siret_active",
            "retrieval_channel_count": 2,
            "v41_channel_ranks": {
                "sparse_active": 1,
                "input_siret_active": 1,
            },
        }
        return V41CandidatePoolResult(
            candidates=[candidate],
            input_siret=InputSiretQualification(
                raw_value=input_siret,
                normalized_siret=input_siret,
                siren=input_siret[:9],
                state=InputSiretState.ACTIVE,
                candidate=candidate,
            ),
            channels={"sparse_active": [input_siret]},
            sparse_result=CandidatePoolResult(
                candidates=[candidate],
                gt_was_injected=False,
            ),
        )


def _runtime():
    config = V41RetrievalConfig()
    signature = "retrieval-fixture"
    release = V41ReleaseManifest.build(
        retrieval_signature=signature,
        ranker_bundle_id="ranker-1",
        acceptor_bundle_id="acceptor-1",
        ranker_dataset_manifest_id="dataset-1",
        acceptor_dataset_manifest_id="acceptor-data-1",
        ranker_feature_order=FEATURE_ORDER,
        acceptor_feature_order=V9_SCENE_FEATURE_NAMES,
        ranker_variant="R1",
    )
    acceptor = V41RawLogisticAcceptor(
        model=_AcceptorModel(),
        threshold=0.9,
        feature_order=V9_SCENE_FEATURE_NAMES,
        model_bundle_id="acceptor-1",
        dataset_manifest_id="acceptor-data-1",
        retrieval_signature=signature,
    )
    return V41RuntimeBundle(
        dataset_manifest={
            "build_id": "dataset-1",
            "retrieval_signature": signature,
            "feature_order": FEATURE_ORDER,
            "positive_injection": False,
        },
        release=release,
        ranker_metadata={},
        ranker=_Ranker(),
        acceptor=acceptor,
        retrieval_config=config,
        artifact_hashes={"ranker/ranker.json": "ranker-hash"},
    )


def _inputs(tmp_path: Path):
    source = pd.DataFrame(
        {
            "SERVICE ID": ["eligible-1", "excluded", "eligible-2"],
            "SITE": ["ALPHA", "EXCLUDED", "ALPHA"],
            "SIRET": [
                "11111111100011",
                "99999999900099",
                "22222222200022",
            ],
            "SITE_CLI_ADRESSE": ["1 RUE ALPHA"] * 3,
            "COMMUNE": ["BOURG"] * 3,
            "CODE_POSTAL": ["01000"] * 3,
            "CODE_INSEE": ["01001"] * 3,
        }
    )
    inventory = build_shadow_inventory(
        source,
        denylists={"OLD_TEST": {"excluded"}},
    )
    inventory.loc[
        inventory["eligible_for_shadow"], "input_siret_state"
    ] = "ACTIVE"
    panel = inventory.loc[
        inventory["service_id"].eq("eligible-1")
    ].copy()
    source_path = tmp_path / "source.csv"
    inventory_path = tmp_path / "inventory.parquet"
    panel_path = tmp_path / "panel.parquet"
    old_test_path = tmp_path / "old_test.parquet"
    fresh_path = tmp_path / "fresh.parquet"
    manifest_path = tmp_path / "inventory_manifest.json"
    source.to_csv(source_path, sep=";", index=False)
    inventory.to_parquet(inventory_path, index=False)
    panel.to_parquet(panel_path, index=False)
    pd.DataFrame(
        {"crm_record_id": ["excluded"], "split": ["test"]}
    ).to_parquet(old_test_path, index=False)
    pd.DataFrame({"crm_record_id": pd.Series(dtype=str)}).to_parquet(
        fresh_path, index=False
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "sireto-shadow-v4.1-inventory-1",
                "source": {
                    "path": str(source_path),
                    "sha256": file_sha256(source_path),
                },
                "denylists": [
                    {
                        "name": "OLD_TEST",
                        "path": str(old_test_path),
                        "sha256": file_sha256(old_test_path),
                        "id_column": "crm_record_id",
                        "split_column": "split",
                        "split_value": "test",
                    },
                    {
                        "name": "FRESH_HOLDOUT",
                        "path": str(fresh_path),
                        "sha256": file_sha256(fresh_path),
                        "id_column": "crm_record_id",
                        "split_column": None,
                        "split_value": None,
                    },
                ],
                "outputs": {
                    "inventory.parquet": file_sha256(inventory_path),
                    "panel500.parquet": file_sha256(panel_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return source_path, inventory_path, panel_path, manifest_path


def test_shadow_scores_only_eligible_and_exports_evidence(tmp_path):
    source, inventory, panel, inventory_manifest = _inputs(tmp_path)
    source_hash_before = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    retriever = _Retriever()
    result = run_shadow(
        source_path=source,
        inventory_path=inventory,
        inventory_manifest_path=inventory_manifest,
        panel_path=panel,
        runtime=_runtime(),
        retriever=retriever,
        persistent_cache=None,
        output_root=tmp_path / "out",
        checkpoint_path=tmp_path / "checkpoint" / "run.sqlite",
        run_id="run",
        input_artifacts={
            "source": source,
            "inventory": inventory,
            "panel": panel,
        },
        expected_decision_count=2,
        expected_panel_count=1,
    )
    assert retriever.calls == ["eligible-1", "eligible-2"]
    decisions = pd.read_parquet(result / "decisions.parquet")
    top10 = pd.read_parquet(result / "candidates_top10.parquet")
    assert set(decisions["service_id"]) == {"eligible-1", "eligible-2"}
    assert set(decisions["routing_status"]) == {"AUTO"}
    assert set(top10["etat_admin"]) == {"A"}
    assert top10["candidate_names_json"].str.contains("ALPHA").all()
    assert "candidate_address" in top10
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["invariants"]["excluded_rows_scored"] == 0
    assert manifest["run_metadata"]["crm_writes_performed"] is False
    assert (
        __import__("hashlib").sha256(source.read_bytes()).hexdigest()
        == source_hash_before
    )


def test_checkpoint_resume_does_not_rescore_completed_ids(tmp_path):
    source, inventory, panel, inventory_manifest = _inputs(tmp_path)
    first = _Retriever(fail_on="eligible-2")
    kwargs = dict(
        source_path=source,
        inventory_path=inventory,
        inventory_manifest_path=inventory_manifest,
        panel_path=panel,
        runtime=_runtime(),
        persistent_cache=None,
        output_root=tmp_path / "out",
        checkpoint_path=tmp_path / "checkpoint" / "resume.sqlite",
        run_id="resume",
        input_artifacts={
            "source": source,
            "inventory": inventory,
            "panel": panel,
        },
        expected_decision_count=2,
        expected_panel_count=1,
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_shadow(retriever=first, **kwargs)
    assert first.calls == ["eligible-1", "eligible-2"]
    second = _Retriever()
    result = run_shadow(retriever=second, **kwargs)
    assert result.exists()
    assert second.calls == ["eligible-2"]


def test_inventory_chain_rejects_denylist_and_id_tampering(tmp_path):
    source, inventory, panel, inventory_manifest = _inputs(tmp_path)
    manifest = json.loads(inventory_manifest.read_text())
    old_test = Path(manifest["denylists"][0]["path"])
    pd.DataFrame(
        {"crm_record_id": ["eligible-1"], "split": ["test"]}
    ).to_parquet(old_test, index=False)
    with pytest.raises(ValueError, match="denylist hash mismatch"):
        validate_inventory_chain(
            source_path=source,
            inventory_path=inventory,
            panel_path=panel,
            inventory_manifest_path=inventory_manifest,
        )


def test_nan_crm_cells_are_empty_strings_not_literal_nan():
    retriever = _Retriever()
    runtime = _runtime()
    _, _, evidence = score_one_query(
        row={
            "service_id": "nan-row",
            "SITE": np.nan,
            "crm_name": np.nan,
            "SITE_CLI_ADRESSE": np.nan,
            "COMMUNE": np.nan,
            "CODE_POSTAL": np.nan,
            "CODE_INSEE": np.nan,
            "SIRET": "11111111100011",
        },
        retriever=retriever,
        ranker=runtime.ranker,
        ranker_feature_order=runtime.release.ranker_feature_order,
        acceptor=runtime.acceptor,
        persistent_cache=None,
        run_id="nan",
    )
    assert retriever.crm_rows[0]["crm_name"] == ""
    assert retriever.crm_rows[0]["crm_address"] == ""
    assert evidence["crm"]["name"] == ""
    assert all(
        value != "nan"
        for value in evidence["crm"].values()
        if isinstance(value, str)
    )


def test_checkpoint_resume_rejects_model_identity_change(tmp_path):
    source, inventory, panel, inventory_manifest = _inputs(tmp_path)
    checkpoint = tmp_path / "checkpoint" / "identity.sqlite"
    common = dict(
        source_path=source,
        inventory_path=inventory,
        inventory_manifest_path=inventory_manifest,
        panel_path=panel,
        retriever=_Retriever(),
        persistent_cache=None,
        output_root=tmp_path / "out",
        checkpoint_path=checkpoint,
        run_id="identity",
        input_artifacts={"source": source},
        expected_decision_count=2,
        expected_panel_count=1,
    )
    first_runtime = _runtime()
    run_shadow(runtime=first_runtime, **common)
    changed_runtime = _runtime()
    changed_runtime.artifact_hashes = {"ranker/ranker.json": "tampered"}
    with pytest.raises(ValueError, match="Checkpoint identity differs"):
        run_shadow(runtime=changed_runtime, **common)


def test_model_manifest_rejects_tampered_ranker_before_load(tmp_path):
    release = _runtime().release
    relative_names = (
        "ranker/ranker.json",
        "ranker/metadata.json",
        "acceptor/acceptor_model.joblib",
        "acceptor/metadata.json",
        "ranker_predictions.parquet",
        "acceptor_scenes.parquet",
        "split_assignments.parquet",
        "training_report.json",
        "release_manifest.json",
    )
    for name in relative_names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{name}".encode())
    manifest = {
        "schema_version": "v4.1-model-bundle-manifest-1",
        "release_id": release.release_id,
        "outputs": {
            name: file_sha256(tmp_path / name) for name in relative_names
        },
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest))
    validate_model_bundle_hashes(tmp_path, release=release)
    (tmp_path / "ranker" / "ranker.json").write_text("tampered")
    with pytest.raises(ValueError, match="ranker/ranker.json"):
        validate_model_bundle_hashes(tmp_path, release=release)


def test_release_validation_rejects_signature_or_feature_drift():
    config = V41RetrievalConfig()
    signature = __import__(
        "scripts.build_v41_training_dataset",
        fromlist=["retrieval_signature"],
    ).retrieval_signature(
        config,
        partitions_signature="partitions",
        global_store_signature="global",
    )
    release = V41ReleaseManifest.build(
        retrieval_signature=signature,
        ranker_bundle_id="ranker",
        acceptor_bundle_id="acceptor",
        ranker_dataset_manifest_id="dataset",
        acceptor_dataset_manifest_id="acceptor-data",
        ranker_feature_order=FEATURE_ORDER,
        acceptor_feature_order=["scene"],
        ranker_variant="R1",
    )
    ranker_metadata = {
        "model_bundle_id": "ranker",
        "dataset_manifest_id": "dataset",
        "retrieval_signature": signature,
        "feature_order": FEATURE_ORDER,
        "ranker_variant": "R1",
        "positive_injection": False,
    }
    acceptor_metadata = {
        "model_bundle_id": "acceptor",
        "dataset_manifest_id": "acceptor-data",
        "retrieval_signature": signature,
        "feature_order": ["scene"],
        "calibration_method": "raw",
        "confidence_kind": "ROUTING_SCORE_UNCALIBRATED",
    }
    manifest = {
        "build_id": "dataset",
        "feature_order": FEATURE_ORDER,
        "retrieval_signature": signature,
        "retrieval_config": {
            **config.__dict__,
            "variant": config.variant.value,
        },
        "positive_injection": False,
        "retrieval_gate": {
            "manifest_sha256": "gate",
            "selected_variant": config.variant.value,
        },
        "input_hashes": {
            "partitions": "partitions",
            "global_store": "global",
        },
    }
    validate_runtime_contract(
        dataset_manifest=manifest,
        release=release,
        ranker_metadata=ranker_metadata,
        acceptor_metadata=acceptor_metadata,
        retrieval_config=config,
        partitions_signature="partitions",
        global_store_signature="global",
    )
    broken = dict(manifest)
    broken["feature_order"] = FEATURE_ORDER[:-1]
    with pytest.raises(ValueError, match="feature order"):
        validate_runtime_contract(
            dataset_manifest=broken,
            release=release,
            ranker_metadata=ranker_metadata,
            acceptor_metadata=acceptor_metadata,
            retrieval_config=config,
            partitions_signature="partitions",
            global_store_signature="global",
        )
