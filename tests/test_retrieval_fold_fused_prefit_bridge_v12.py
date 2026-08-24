from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.xgb_matcher.retrieval_fold_fused_prefit_bridge_v12 as bridge
from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (
    ExecutionCaps,
    FeatureBatch,
    FoldFusedInvariantError,
    SceneRecord,
    TRAIN_FOLDS,
    sha256_file,
)
from src.xgb_matcher.retrieval_fold_fused_runtime_v12 import (
    PARITY_ROW_SCHEMA_VERSION,
    PREFIT_PARITY_CHECKS,
    PREFIT_PARITY_SCHEMA_VERSION,
)
from src.xgb_matcher.retrieval_fold_fused_sharded_oof_v12 import (
    ShardedPhysicalInputsV12,
    plan_query_shards,
    run_sharded_physical_sample,
)
from tests.test_retrieval_fold_fused_sharded_oof_v12 import (
    _Backend,
    _Contexts,
    _LazyBackend,
    _Ledger,
    _Sampler,
    _Source,
    _build,
    _labels,
    _labels_for_sequence,
    _lineage,
    _scene_index,
)


def _parity_row(observed: dict[str, object]) -> dict[str, object]:
    row = {
        "schema_version": PARITY_ROW_SCHEMA_VERSION,
        "query_id": str(observed["query_id"]),
        "fold": int(observed["fold"]),
        "outer_fold": int(observed["outer_fold"]),
        "master_count": int(observed["master_count"]),
        "fit_sample_count": int(observed["fit_sample_count"]),
        "legacy_parent_sha256": str(observed["fused_parent_sha256"]),
        "fused_parent_sha256": str(observed["fused_parent_sha256"]),
        "legacy_master_sha256": str(observed["fused_master_sha256"]),
        "fused_master_sha256": str(observed["fused_master_sha256"]),
        "legacy_features_sha256": str(observed["fused_features_sha256"]),
        "fused_features_sha256": str(observed["fused_features_sha256"]),
        "legacy_fit_rows_sha256": str(observed["fused_fit_rows_sha256"]),
        "fused_fit_rows_sha256": str(observed["fused_fit_rows_sha256"]),
    }
    row.update({name: True for name in PREFIT_PARITY_CHECKS})
    return row


def _fixture_observed_rows(
    *,
    index: pa.Table | None = None,
    candidate_count: int = 6,
) -> list[dict[str, object]]:
    scene_index = index if index is not None else _scene_index()
    labels = _labels_for_sequence(
        scene_index["query_id"].to_pylist(),
        scene_index["fold"].to_pylist(),
        candidate_count,
    )
    truth = {query_id: label.human_siret for query_id, label in labels.items()}
    contexts = _Contexts(truth, [], candidate_count)
    sampler = _Sampler()
    rows: list[dict[str, object]] = []
    for raw in scene_index.to_pylist():
        scene = SceneRecord(query_id=str(raw["query_id"]), fold=int(raw["fold"]))
        prepared = {}
        for outer in TRAIN_FOLDS:
            context = contexts.prepare(scene, outer_fold=outer)
            statistics = contexts.compute_full_master_statistics(context)
            features = contexts.materialize(
                context, statistics, candidate_indices=None
            )
            observed = bridge._observed_label_blind_row_v12(
                context, statistics, features
            )
            prepared[outer] = (context, statistics, observed)
        label = labels[scene.query_id]
        for outer in TRAIN_FOLDS:
            context, statistics, observed = prepared[outer]
            plan = sampler.choose(
                context, statistics, label, maximum_rows=min(4, candidate_count)
            )
            digest, count = bridge._fit_projection_v12(context, plan)
            observed["fused_fit_rows_sha256"] = digest
            observed["fit_sample_count"] = count
            rows.append(observed)
    return rows


def _write_prefit_reference(
    root: Path,
    *,
    index: pa.Table | None = None,
    candidate_count: int = 6,
    lineage: dict[str, str] | None = None,
) -> Path:
    root.mkdir()
    scene_index = index if index is not None else _scene_index()
    lineage_value = dict(lineage or _lineage())
    rows = [
        _parity_row(row)
        for row in _fixture_observed_rows(
            index=scene_index, candidate_count=candidate_count
        )
    ]
    filler_digest = bridge._sha256_json("filler")
    # The strict PREFIT loader requires the real 512 x 3 physical evidence.
    # Only six identities are consumed by this bounded sharded fixture.
    for filler_index in range(512 - scene_index.num_rows):
        query_id = f"prefit-filler-{filler_index:04d}"
        fold = TRAIN_FOLDS[filler_index % len(TRAIN_FOLDS)]
        for outer in TRAIN_FOLDS:
            rows.append(
                _parity_row(
                    {
                        "query_id": query_id,
                        "fold": fold,
                        "outer_fold": outer,
                        "master_count": 2,
                        "fit_sample_count": 2,
                        "fused_parent_sha256": filler_digest,
                        "fused_master_sha256": filler_digest,
                        "fused_features_sha256": filler_digest,
                        "fused_fit_rows_sha256": filler_digest,
                    }
                )
            )
    rows_path = root / "parity_rows.parquet"
    pq.write_table(pa.Table.from_pylist(rows), rows_path, compression="zstd")
    reference = {
        "schema_version": "sireto-retrieval-v12-fold-fused-reference-1",
        "engine": "LEGACY_SINGLE_CONTEXT_V12",
        "query_count": 512,
        "context_count": 1536,
        "folds": list(TRAIN_FOLDS),
        "lineage": lineage_value,
        "site_model_manifest_sha256": None,
        "site_model_sha256": None,
        "site_model_recipe_sha256": None,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
    }
    reference_path = root / "reference_manifest.json"
    reference_path.write_text(
        json.dumps(reference, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": PREFIT_PARITY_SCHEMA_VERSION,
        "status": "PASS",
        "phase": "PREFIT_NON_CIRCULAR",
        "reference_manifest_sha256": sha256_file(reference_path),
        "query_count": 512,
        "context_count": 1536,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "checks": {name: True for name in PREFIT_PARITY_CHECKS},
        "stop_on_any_divergence": True,
        "postfit_checks_status": "PENDING_SITE_MODELS",
        "lineage": lineage_value,
        "artifacts": {
            "reference_manifest": {
                "path": reference_path.name,
                "sha256": sha256_file(reference_path),
            },
            "parity_rows": {
                "path": rows_path.name,
                "sha256": sha256_file(rows_path),
                "rows": len(rows),
            },
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


def _sink_for_spec(prefit: Path, spec) -> bridge.PrefitBridgeSinkV12:
    return bridge.PrefitBridgeSinkV12.from_prefit_manifest(
        prefit,
        expected_prefit_lineage=_lineage(),
        bridge_lineage=_lineage(),
        expected_query_ids=spec.query_ids,
        expected_query_folds=spec.query_folds,
        expected_prefit_manifest_sha256=sha256_file(prefit),
    )


class _TracingSink:
    def __init__(self, delegate, events: list[tuple]) -> None:
        self.delegate = delegate
        self.events = events

    def observe_label_blind(self, context, statistics, features) -> None:
        self.events.append(("bridge_blind", context.query_id, context.outer_fold))
        self.delegate.observe_label_blind(context, statistics, features)

    def observe_fit(self, context, plan) -> None:
        self.events.append(("bridge_fit", context.query_id, context.outer_fold))
        self.delegate.observe_fit(context, plan)

    def seal(self, output_dir):
        return self.delegate.seal(output_dir)


def test_bridge_observes_three_complete_contexts_before_label_and_seals_direct_prefit(
    tmp_path: Path,
) -> None:
    prefit = _write_prefit_reference(tmp_path / "prefit")
    index = _scene_index()
    spec = plan_query_shards(index, tuple(index["query_id"].to_pylist()), 1)[0]
    events: list[tuple] = []
    sink = _TracingSink(_sink_for_spec(prefit, spec), events)
    artifact = _build(
        spec,
        tmp_path / "build",
        _labels(),
        events,
        prefit_bridge_sink=sink,
    )
    assert artifact.prefit_bridge is not None
    assert artifact.prefit_bridge.row_count == 18
    bridge.validate_prefit_bridge_against_reference_v12(
        artifact.prefit_bridge,
        prefit_manifest_path=prefit,
        expected_lineage=_lineage(),
    )
    for query_id in spec.query_ids:
        label_position = events.index(("label", query_id))
        blind_positions = [
            index
            for index, event in enumerate(events)
            if event[:2] == ("bridge_blind", query_id)
        ]
        fit_positions = [
            index
            for index, event in enumerate(events)
            if event[:2] == ("bridge_fit", query_id)
        ]
        assert len(blind_positions) == len(TRAIN_FOLDS)
        assert len(fit_positions) == len(TRAIN_FOLDS)
        assert max(blind_positions) < label_position < min(fit_positions)


def test_direct_bridge_fails_on_current_feature_divergence(tmp_path: Path) -> None:
    prefit = _write_prefit_reference(tmp_path / "prefit")
    index = _scene_index()
    query_id = str(index["query_id"][0].as_py())
    fold = int(index["fold"][0].as_py())
    spec = plan_query_shards(index, (query_id,), 1)[0]
    sink = _sink_for_spec(prefit, spec)
    labels = _labels()
    contexts = _Contexts(
        {key: value.human_siret for key, value in labels.items()}, []
    )
    context = contexts.prepare(SceneRecord(query_id=query_id, fold=fold), outer_fold=2)
    statistics = contexts.compute_full_master_statistics(context)
    features = contexts.materialize(context, statistics, candidate_indices=None)
    forged = FeatureBatch(
        query_id=features.query_id,
        scene_fold=features.scene_fold,
        outer_fold=features.outer_fold,
        candidate_indices=features.candidate_indices,
        candidate_sirets=features.candidate_sirets,
        candidate_parents=features.candidate_parents,
        feature_names=features.feature_names,
        values=np.ascontiguousarray(features.values + np.float32(1.0)),
        statistics_sha256=features.statistics_sha256,
        row_payloads=features.row_payloads,
    )
    with pytest.raises(FoldFusedInvariantError, match="fused_features_sha256"):
        sink.observe_label_blind(context, statistics, forged)


def test_w1_w3_merge_same_prefit_bridge_bytes_and_record(tmp_path: Path) -> None:
    prefit = _write_prefit_reference(tmp_path / "prefit")
    index = _scene_index()
    query_ids = tuple(str(value) for value in index["query_id"].to_pylist())
    entrypoint_prefix = "tests.test_retrieval_fold_fused_sharded_oof_v12"
    common = {
        "prefit_bridge": {
            "manifest_path": str(prefit),
            "manifest_sha256": sha256_file(prefit),
            "prefit_lineage": _lineage(),
        }
    }

    def execute(worker_count: int, name: str, *, bridge_enabled: bool = True):
        shard_count = 1 if worker_count == 1 else 3
        inputs = ShardedPhysicalInputsV12(
            scene_index=index,
            query_ids=query_ids,
            lineage=_lineage(),
            build_entrypoint=f"{entrypoint_prefix}:public_fixture_build_worker",
            build_payload=common if bridge_enabled else {},
            replay_entrypoint=f"{entrypoint_prefix}:public_fixture_replay_worker",
            replay_payload={},
            model_backend=_LazyBackend(tmp_path / name / "run" / "site_models"),
            caps=ExecutionCaps(fit_sample=4),
            maximum_spool_bytes=32 * 1024**2,
            maximum_workers=worker_count,
        )
        return run_sharded_physical_sample(
            inputs,
            plan_query_shards(index, query_ids, shard_count),
            tmp_path / name / "run",
            miniature_fit=True,
        )

    baseline = execute(1, "baseline", bridge_enabled=False)
    w1 = execute(1, "w1")
    w3 = execute(3, "w3")
    assert baseline.prefit_bridge is None
    assert w1.prefit_bridge is not None
    assert w3.prefit_bridge is not None
    assert w1.prefit_bridge.row_count == w3.prefit_bridge.row_count == 18
    assert w1.prefit_bridge.sha256 == w3.prefit_bridge.sha256
    assert w1.prefit_bridge.query_sequence_sha256 == w3.prefit_bridge.query_sequence_sha256
    assert w1.digests["prefit_bridge"] == w3.digests["prefit_bridge"]
    for name in (
        "train_matrix",
        "group_sequence",
        "heldout_spool",
        "model_by_outer",
        "prerank500",
        "top100",
        "outcomes",
    ):
        assert baseline.digests[name] == w1.digests[name]
    assert pq.read_table(w1.prefit_bridge.path).equals(
        pq.read_table(w3.prefit_bridge.path)
    )
    for artifact in (w1, w3):
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        assert manifest["digests"]["prefit_bridge"] == artifact.prefit_bridge.sha256
        assert manifest["prefit_bridge"]["rows"] == 18
        bridge.validate_prefit_bridge_against_reference_v12(
            artifact.prefit_bridge,
            prefit_manifest_path=prefit,
            expected_lineage=_lineage(),
        )


def test_bridge_loader_and_direct_validator_reject_tampering(tmp_path: Path) -> None:
    prefit = _write_prefit_reference(tmp_path / "prefit")
    index = _scene_index()
    spec = plan_query_shards(
        index, tuple(index["query_id"].to_pylist()), 1
    )[0]
    artifact = _build(
        spec,
        tmp_path / "build",
        _labels(),
        [],
        prefit_bridge_sink=_sink_for_spec(prefit, spec),
    ).prefit_bridge
    assert artifact is not None
    rows = pq.read_table(artifact.path).to_pylist()
    rows[0]["statistics_sha256"] = bridge._sha256_json("tampered-statistics")
    forged = bridge._write_bridge_artifact_v12(
        rows,
        query_sequence=tuple(
            zip(spec.query_ids, spec.query_folds, strict=True)
        ),
        lineage=artifact.lineage,
        reference_hashes={
            "prefit_manifest_sha256": artifact.prefit_manifest_sha256,
            "reference_manifest_sha256": artifact.reference_manifest_sha256,
            "parity_rows_sha256": artifact.parity_rows_sha256,
        },
        output_dir=tmp_path / "self_consistent_forgery",
    )
    # A statistic-only change is intentionally not part of historical PREFIT,
    # but produces different W1/W3 bridge bytes and cannot hide in that gate.
    assert forged.sha256 != artifact.sha256

    rows = pq.read_table(artifact.path).to_pylist()
    rows[0]["fused_master_sha256"] = bridge._sha256_json("forged-master")
    forged_prefit = bridge._write_bridge_artifact_v12(
        rows,
        query_sequence=tuple(
            zip(spec.query_ids, spec.query_folds, strict=True)
        ),
        lineage=artifact.lineage,
        reference_hashes={
            "prefit_manifest_sha256": artifact.prefit_manifest_sha256,
            "reference_manifest_sha256": artifact.reference_manifest_sha256,
            "parity_rows_sha256": artifact.parity_rows_sha256,
        },
        output_dir=tmp_path / "forged_prefit_projection",
    )
    with pytest.raises(FoldFusedInvariantError, match="fused_master_sha256"):
        bridge.validate_prefit_bridge_against_reference_v12(
            forged_prefit,
            prefit_manifest_path=prefit,
            expected_lineage=_lineage(),
        )
