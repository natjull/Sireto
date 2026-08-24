from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.xgb_matcher.retrieval_fold_fused_physical_benchmark_v12 as physical_benchmark
from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (
    ExecutionCaps,
    FeatureBatch,
    FittedSiteModel,
    FoldFusedInvariantError,
    FullMasterStatistics,
    HumanLabel,
    MasterContext,
    RankingSelection,
    SamplingPlan,
    SceneRecord,
    TrainMatrixArtifact,
    sha256_file,
)
from src.xgb_matcher.retrieval_fold_fused_runtime_v12 import _MasterPayload
from src.xgb_matcher.retrieval_fold_fused_prefit_bridge_v12 import (
    PrefitBridgeSinkV12,
)
from src.xgb_matcher.retrieval_fold_fused_sharded_oof_v12 import (
    ShardedPhysicalInputsV12,
    ShardSpecV12,
    WorkerJobV12,
    aggregate_sharded_gates_v12,
    build_shard_v12,
    fit_site_models_sequential_v12,
    fitted_site_model_records_v12,
    heldout_master_record_v12,
    load_fitted_site_model_records_v12,
    load_heldout_master_record_v12,
    load_shard_build_manifest_v12,
    load_shard_replay_manifest_v12,
    merge_heldout_shards_v12,
    merge_replay_shards_v12,
    merge_training_shards_v12,
    plan_query_shards,
    replay_shard_v12,
    run_sharded_physical_sample,
    spawn_manifest_workers_v12,
)
from src.xgb_matcher.retrieval_top100_cascade_v12 import CASCADE_PARENT_FEATURES


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _siret(parent: int, site: int) -> str:
    return f"{parent:09d}{site:05d}"


def _scene_index() -> pa.Table:
    records = []
    row_start = 0
    for fold in (2, 3, 4):
        for index in range(2):
            records.append(
                {
                    "query_id": f"f{fold}q{index}",
                    "fold": fold,
                    "row_start": row_start,
                    "row_count": 6,
                }
            )
            row_start += 6
    return pa.Table.from_pylist(records)


def _scene_index_512(candidate_count: int = 2) -> pa.Table:
    records = []
    row_start = 0
    for fold, count in ((2, 170), (3, 171), (4, 171)):
        for index in range(count):
            records.append(
                {
                    "query_id": f"f{fold}q{index}",
                    "fold": fold,
                    "row_start": row_start,
                    "row_count": candidate_count,
                }
            )
            row_start += candidate_count
    return pa.Table.from_pylist(records)


def _lineage() -> dict[str, str]:
    return {
        "scene_ir_sha256": _sha("scene"),
        "label_ledger_sha256": _sha("labels"),
        "scientific_config_sha256": _sha("config"),
    }


class _Source:
    def __init__(self, spec: ShardSpecV12, events: list[tuple]) -> None:
        self.spec = spec
        self.events = events

    def iter_scenes(self, *, folds: tuple[int, ...]):
        assert folds == (2, 3, 4)
        for query_id, fold in zip(
            self.spec.query_ids, self.spec.query_folds, strict=True
        ):
            self.events.append(("scene", query_id, fold))
            yield SceneRecord(query_id=query_id, fold=fold)


class _Ledger:
    def __init__(self, labels: dict[str, HumanLabel], events: list[tuple]) -> None:
        self.labels = labels
        self.events = events

    def get(self, query_id: str) -> HumanLabel:
        self.events.append(("label", query_id))
        return self.labels[query_id]


class _Contexts:
    feature_names = ("signal", "joint_name_geo", "parent_score_probability")

    def __init__(
        self,
        truth: dict[str, str],
        events: list[tuple],
        candidate_count: int = 6,
    ) -> None:
        self.truth = truth
        self.events = events
        self.candidate_count = candidate_count

    def _raw_rows(
        self, scene: SceneRecord, outer_fold: int
    ) -> tuple[dict[str, object], ...]:
        index = int(scene.query_id.rsplit("q", 1)[-1])
        base = 200_000_000 + scene.fold * 10_000 + index * 100
        output = []
        for candidate_index in range(self.candidate_count):
            parent = base + candidate_index
            output.append(
                {
                    "query_id": scene.query_id,
                    "fold": scene.fold,
                    "siret": _siret(parent, 1),
                    "siren": f"{parent:09d}",
                    "union_rank": candidate_index + 1,
                    "master_rank": candidate_index + 1,
                    "hydration_rank": candidate_index + 1,
                    "parent_rank": candidate_index + 1,
                    "site_rank": 1,
                    "parent_score": float(outer_fold) + candidate_index / 100,
                    "parent_feature_names": list(CASCADE_PARENT_FEATURES),
                    "parent_feature_values": [0.0] * len(CASCADE_PARENT_FEATURES),
                    "site_score": 0.0,
                    "signals": [],
                    "channels": [],
                    "sources": ["SIRENE"],
                    "lanes": [],
                    "selection_reasons": [],
                    "official_evidence_ids": [],
                    "official_evidence_sources": ["SIRENE"],
                    "candidate_names": [f"NAME {candidate_index}"],
                    "candidate_addresses": [f"{candidate_index} RUE TEST"],
                    "candidate_insee": "75056",
                    "candidate_postcode": "75001",
                    "candidate_number": str(candidate_index),
                    "candidate_state": "A",
                    "candidate_is_headquarters": candidate_index == 0,
                    "temporal_valid_from": "2020-01-01",
                    "temporal_valid_to": "",
                    "crm_name": "NAME 5",
                    "crm_address": "5 RUE TEST",
                    "crm_number": "5",
                    "crm_postcode": "75001",
                    "crm_insee": "75056",
                }
            )
        return tuple(output)

    def prepare(self, scene: SceneRecord, *, outer_fold: int) -> MasterContext:
        self.events.append(("prepare", scene.query_id, outer_fold))
        raw = self._raw_rows(scene, outer_fold)
        return MasterContext(
            query_id=scene.query_id,
            scene_fold=scene.fold,
            outer_fold=outer_fold,
            candidate_sirets=tuple(str(row["siret"]) for row in raw),
            candidate_parents=tuple(str(row["siren"]) for row in raw),
            opaque=_MasterPayload(raw),
        )

    def compute_full_master_statistics(
        self, context: MasterContext
    ) -> FullMasterStatistics:
        digest = _sha(f"stats:{context.query_id}:{context.outer_fold}")
        self.events.append(("stats", context.query_id, context.outer_fold))
        return FullMasterStatistics(
            query_id=context.query_id,
            outer_fold=context.outer_fold,
            master_count=len(context.candidate_sirets),
            candidate_order_sha256=context.candidate_order_sha256,
            statistics_sha256=digest,
            complete=True,
        )

    def materialize(
        self,
        context: MasterContext,
        statistics: FullMasterStatistics,
        *,
        candidate_indices,
    ) -> FeatureBatch:
        indices = (
            tuple(range(len(context.candidate_sirets)))
            if candidate_indices is None
            else tuple(candidate_indices)
        )
        rows = []
        for index in indices:
            truth = context.candidate_sirets[index] == self.truth[context.query_id]
            rows.append([10.0 if truth else float(5 - index), float(truth), 0.2])
        raw = context.opaque.raw_rows
        return FeatureBatch(
            query_id=context.query_id,
            scene_fold=context.scene_fold,
            outer_fold=context.outer_fold,
            candidate_indices=indices,
            candidate_sirets=tuple(context.candidate_sirets[index] for index in indices),
            candidate_parents=tuple(context.candidate_parents[index] for index in indices),
            feature_names=self.feature_names,
            values=np.asarray(rows, dtype=np.float32),
            statistics_sha256=statistics.statistics_sha256,
            row_payloads=tuple(raw[index] for index in indices),
        )


class _Sampler:
    def choose(
        self,
        context: MasterContext,
        statistics: FullMasterStatistics,
        label: HumanLabel,
        *,
        maximum_rows: int,
    ) -> SamplingPlan:
        truth = context.candidate_sirets.index(label.human_siret)
        return SamplingPlan(
            candidate_indices=tuple(dict.fromkeys((0, 1, truth)))[:maximum_rows],
            statistics_sha256=statistics.statistics_sha256,
            rule_id="fixture",
        )


class _Backend:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, artifact: TrainMatrixArtifact) -> FittedSiteModel:
        path = self.output_dir / f"outer{artifact.outer_fold}.model"
        path.write_text(
            json.dumps(
                {
                    "outer": artifact.outer_fold,
                    "matrix": artifact.sha256,
                    "groups": artifact.query_count,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return FittedSiteModel(
            outer_fold=artifact.outer_fold,
            handle={"outer": artifact.outer_fold},
            path=path,
            sha256=_sha(path.read_text(encoding="utf-8")),
            recipe_sha256=_sha("fixture-frozen-recipe"),
            feature_names=artifact.feature_names,
        )

    def score(self, model: FittedSiteModel, features: FeatureBatch) -> np.ndarray:
        return features.values[:, 0].astype(np.float64)


class _LazyBackend:
    """Keep fixture models under the core root without precreating it."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.delegate: _Backend | None = None

    def _get(self) -> _Backend:
        if self.delegate is None:
            self.delegate = _Backend(self.output_dir)
        return self.delegate

    def fit(self, artifact: TrainMatrixArtifact) -> FittedSiteModel:
        return self._get().fit(artifact)

    def score(self, model: FittedSiteModel, features: FeatureBatch) -> np.ndarray:
        return self._get().score(model, features)


class _Selector:
    def select(
        self,
        context: MasterContext,
        features: FeatureBatch,
        scores: np.ndarray,
        *,
        prerank_cap: int,
        product_cap: int,
    ) -> RankingSelection:
        order = tuple(
            sorted(
                range(len(scores)),
                key=lambda index: (-float(scores[index]), features.candidate_sirets[index]),
            )
        )
        prerank = order[:prerank_cap]
        product = prerank[: min(3, product_cap)]
        return RankingSelection(
            prerank_indices=prerank,
            product_indices=product,
            product_reasons=("fixture",) * len(product),
        )


def _labels_for_sequence(
    query_ids,
    folds,
    candidate_count: int = 6,
) -> dict[str, HumanLabel]:
    output = {}
    for query_id, fold in zip(query_ids, folds, strict=True):
        index = int(str(query_id).rsplit("q", 1)[-1])
        base = 200_000_000 + int(fold) * 10_000 + index * 100
        truth = _siret(base + candidate_count - 1, 1)
        output[str(query_id)] = HumanLabel(
            query_id=str(query_id),
            fold=int(fold),
            human_siret=truth,
            operational_sirets=(truth,),
            audit_metadata={"official_catalog_present": True},
        )
    return output


def _labels(candidate_count: int = 6) -> dict[str, HumanLabel]:
    index = _scene_index()
    return _labels_for_sequence(
        index["query_id"].to_pylist(),
        index["fold"].to_pylist(),
        candidate_count,
    )


def _build(
    spec: ShardSpecV12,
    path: Path,
    labels: dict[str, HumanLabel],
    events: list[tuple],
    candidate_count: int = 6,
    prefit_bridge_sink=None,
):
    truth = {query_id: value.human_siret for query_id, value in labels.items()}
    return build_shard_v12(
        spec,
        source=_Source(spec, events),
        labels=_Ledger(labels, events),
        contexts=_Contexts(truth, events, candidate_count),
        sampler=_Sampler(),
        admission_config=SimpleNamespace(channels=()),
        output_dir=path,
        lineage=_lineage(),
        caps=ExecutionCaps(fit_sample=4),
        prefit_bridge_sink=prefit_bridge_sink,
    )


def _replay(spec, build, path, labels, models):
    truth = {query_id: value.human_siret for query_id, value in labels.items()}
    return replay_shard_v12(
        spec,
        heldout=build.heldout_master,
        labels=_Ledger(labels, []),
        contexts=_Contexts(truth, []),
        model_backend=_Backend(path.parent / "scoring_backend"),
        models=models,
        selector=_Selector(),
        output_dir=path,
        lineage=_lineage(),
        build_manifest_sha256=build.manifest_sha256,
        caps=ExecutionCaps(fit_sample=4),
    )


def test_planner_is_weighted_contiguous_and_refuses_forbidden_folds() -> None:
    index = _scene_index()
    requested = tuple(reversed(index["query_id"].to_pylist()))
    specs = plan_query_shards(index, requested, 3)

    assert [spec.query_count for spec in specs] == [2, 2, 2]
    assert tuple(value for spec in specs for value in spec.query_ids) == tuple(
        index["query_id"].to_pylist()
    )
    assert [spec.sequence_start for spec in specs] == [0, 2, 4]
    assert [spec.sequence_stop for spec in specs] == [2, 4, 6]
    assert all(len(spec.source_ranges) == 1 for spec in specs)

    forged = index.set_column(
        index.schema.get_field_index("fold"),
        "fold",
        pa.array([1, 2, 3, 3, 4, 4]),
    )
    with pytest.raises(FoldFusedInvariantError, match="fold 0/1"):
        plan_query_shards(forged, requested, 3)


def test_w3_planner_preserves_the_preregistered_exact_fold_ranges() -> None:
    rows = []
    start = 0
    for fold, counts in ((2, (1_000, 1_000)), (3, (1, 1)), (4, (1, 1))):
        for ordinal, count in enumerate(counts):
            rows.append(
                {
                    "query_id": f"fold{fold}-{ordinal}",
                    "fold": fold,
                    "row_start": start,
                    "row_count": count,
                }
            )
            start += count
    index = pa.Table.from_pylist(rows)
    query_ids = tuple(index["query_id"].to_pylist())
    specs = plan_query_shards(index, query_ids, 3)

    assert [spec.shard_id for spec in specs] == ["fold2", "fold3", "fold4"]
    assert [spec.folds for spec in specs] == [(2,), (3,), (4,)]
    assert [spec.candidate_row_count for spec in specs] == [2_000, 2, 2]


def test_heldout_merge_is_byte_exact_across_global_row_group_boundary(
    tmp_path: Path,
) -> None:
    candidate_count = 2_000
    rows = []
    labels: dict[str, HumanLabel] = {}
    row_start = 0
    for fold in (2, 3, 4):
        for ordinal in range(3):
            query_id = f"f{fold}q{ordinal}"
            rows.append(
                {
                    "query_id": query_id,
                    "fold": fold,
                    "row_start": row_start,
                    "row_count": candidate_count,
                }
            )
            base = 200_000_000 + fold * 10_000 + ordinal * 100
            truth = _siret(base + candidate_count - 1, 1)
            labels[query_id] = HumanLabel(
                query_id=query_id,
                fold=fold,
                human_siret=truth,
                operational_sirets=(truth,),
            )
            row_start += candidate_count
    index = pa.Table.from_pylist(rows)
    query_ids = tuple(index["query_id"].to_pylist())
    w1_spec = plan_query_shards(index, query_ids, 1)[0]
    w1 = _build(
        w1_spec,
        tmp_path / "rg_w1",
        labels,
        [],
        candidate_count,
    )
    w3_specs = plan_query_shards(index, query_ids, 3)
    w3 = tuple(
        _build(
            spec,
            tmp_path / f"rg_w3_{ordinal}",
            labels,
            [],
            candidate_count,
        )
        for ordinal, spec in enumerate(w3_specs)
    )
    merged = merge_heldout_shards_v12(w3, tmp_path / "rg_merged")

    assert merged.artifact.sha256 == w1.heldout_master.sha256
    assert [
        pq.ParquetFile(merged.artifact.path).metadata.row_group(index).num_rows
        for index in range(pq.ParquetFile(merged.artifact.path).num_row_groups)
    ] == [16_000, 2_000]


def test_w3_build_merge_fit_replay_is_exactly_equal_to_w1(tmp_path: Path) -> None:
    index = _scene_index()
    query_ids = tuple(index["query_id"].to_pylist())
    labels = _labels()
    events: list[tuple] = []
    w1_spec = plan_query_shards(index, query_ids, 1)[0]
    w1 = _build(w1_spec, tmp_path / "w1_build", labels, events)
    w3_specs = plan_query_shards(index, query_ids, 3)
    w3 = tuple(
        _build(spec, tmp_path / f"w3_build_{index}", labels, events)
        for index, spec in enumerate(w3_specs)
    )

    merged_train = merge_training_shards_v12(w3, tmp_path / "merged_train")
    merged_heldout = merge_heldout_shards_v12(w3, tmp_path / "merged_heldout")
    for outer in (2, 3, 4):
        assert merged_train.matrices[outer].sha256 == w1.train_matrices[outer].sha256
        assert (
            merged_train.group_sequence_sha256[outer]
            == w1.train_group_sequence_sha256[outer]
        )
    assert merged_heldout.artifact.sha256 == w1.heldout_master.sha256
    assert merged_heldout.artifact.query_sequence_sha256 == w1_spec.query_sequence_sha256

    w1_models = fit_site_models_sequential_v12(
        _Backend(tmp_path / "w1_models"), w1.train_matrices
    )
    w3_models = fit_site_models_sequential_v12(
        _Backend(tmp_path / "w3_models"), merged_train
    )
    assert {
        outer: model.sha256 for outer, model in w1_models.items()
    } == {outer: model.sha256 for outer, model in w3_models.items()}

    w1_replay = _replay(
        w1_spec, w1, tmp_path / "w1_replay", labels, w1_models
    )
    w3_replays = tuple(
        _replay(spec, build, tmp_path / f"w3_replay_{index}", labels, w3_models)
        for index, (spec, build) in enumerate(zip(w3_specs, w3, strict=True))
    )
    merged_replay = merge_replay_shards_v12(w3_replays, tmp_path / "merged_replay")
    assert merged_replay.digests == {
        "prerank500": w1_replay.outputs["top500"].sha256,
        "top100": w1_replay.outputs["top100"].sha256,
        "outcomes": w1_replay.outputs["outcomes"].sha256,
    }
    assert merged_replay.outcomes == w1_replay.outcomes
    gates = aggregate_sharded_gates_v12(merged_replay.outcomes)
    assert gates["exact_recall_at_100"]["rate"] == 1.0
    assert gates["gate_eligible"] is False
    assert gates["decision"] == "NO_READY_DECISION"

    # Every build opens its label only after all three contexts exist and the
    # heldout row has been spooled.  The fixture records prepare/label order.
    for query_id in query_ids:
        positions = [
            index for index, event in enumerate(events) if query_id in event
        ]
        label_positions = [
            index
            for index, event in enumerate(events)
            if event == ("label", query_id)
        ]
        assert label_positions
        for label_position in label_positions:
            recent = events[max(0, label_position - 5) : label_position]
            assert sum(event[:2] == ("prepare", query_id) for event in recent) == 3


def test_shard_manifests_round_trip_and_reject_forged_fold(tmp_path: Path) -> None:
    index = _scene_index()
    query_ids = tuple(index["query_id"].to_pylist())
    labels = _labels()
    spec = plan_query_shards(index, query_ids, 1)[0]
    build = _build(spec, tmp_path / "build", labels, [])
    loaded = load_shard_build_manifest_v12(build.manifest_path)
    assert loaded.build_id == build.build_id
    assert loaded.spec == spec

    backend = _Backend(tmp_path / "models")
    models = fit_site_models_sequential_v12(backend, build.train_matrices)
    replay = _replay(spec, build, tmp_path / "replay", labels, models)
    loaded_replay = load_shard_replay_manifest_v12(replay.manifest_path)
    assert loaded_replay.spec == spec
    assert loaded_replay.digests == replay.digests

    raw = json.loads(build.manifest_path.read_text(encoding="utf-8"))
    raw["fold0_opened"] = True
    build.manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(FoldFusedInvariantError, match="unsafe"):
        load_shard_build_manifest_v12(build.manifest_path, verify_hashes=False)


def public_fixture_manifest_worker(payload: dict[str, object]) -> str:
    output = Path(str(payload["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"job": payload["job"]}), encoding="utf-8")
    return str(manifest)


def public_fixture_build_worker(payload: dict[str, object]) -> str:
    spec = ShardSpecV12.from_manifest(dict(payload["spec"]))
    common = dict(payload.get("common") or {})
    candidate_count = int(common.get("candidate_count", 6))
    labels = _labels_for_sequence(
        spec.query_ids, spec.query_folds, candidate_count
    )
    truth = {query_id: value.human_siret for query_id, value in labels.items()}
    bridge_raw = dict(common.get("prefit_bridge") or {})
    bridge_sink = (
        PrefitBridgeSinkV12.from_prefit_manifest(
            Path(str(bridge_raw["manifest_path"])),
            expected_prefit_lineage=dict(bridge_raw["prefit_lineage"]),
            bridge_lineage=dict(payload["lineage"]),
            expected_query_ids=spec.query_ids,
            expected_query_folds=spec.query_folds,
            expected_prefit_manifest_sha256=str(bridge_raw["manifest_sha256"]),
        )
        if bridge_raw
        else None
    )
    artifact = build_shard_v12(
        spec,
        source=_Source(spec, []),
        labels=_Ledger(labels, []),
        contexts=_Contexts(truth, [], candidate_count),
        sampler=_Sampler(),
        admission_config=SimpleNamespace(channels=()),
        output_dir=Path(str(payload["output_dir"])),
        lineage=dict(payload["lineage"]),
        caps=ExecutionCaps(**dict(payload["caps"])),
        maximum_spool_bytes=int(payload["maximum_spool_bytes"]),
        prefit_bridge_sink=bridge_sink,
    )
    return str(artifact.manifest_path)


def public_fixture_replay_worker(payload: dict[str, object]) -> str:
    spec = ShardSpecV12.from_manifest(dict(payload["spec"]))
    heldout = load_heldout_master_record_v12(dict(payload["heldout"]))
    models = load_fitted_site_model_records_v12(
        dict(payload["models"]),
        handle_loader=lambda path, outer: json.loads(path.read_text(encoding="utf-8")),
    )
    common = dict(payload.get("common") or {})
    candidate_count = int(common.get("candidate_count", 6))
    labels = _labels_for_sequence(
        spec.query_ids, spec.query_folds, candidate_count
    )
    truth = {query_id: value.human_siret for query_id, value in labels.items()}
    output = Path(str(payload["output_dir"]))
    artifact = replay_shard_v12(
        spec,
        heldout=heldout,
        labels=_Ledger(labels, []),
        contexts=_Contexts(truth, [], candidate_count),
        model_backend=_Backend(output.parent / f"score-{spec.shard_id}"),
        models=models,
        selector=_Selector(),
        output_dir=output,
        lineage=dict(payload["lineage"]),
        build_manifest_sha256=str(payload["build_manifest_sha256"]),
        caps=ExecutionCaps(**dict(payload["caps"])),
    )
    return str(artifact.manifest_path)


def test_spawn_workers_exchange_only_manifest_paths(tmp_path: Path) -> None:
    jobs = tuple(
        WorkerJobV12(
            job_id=f"job-{index}",
            payload={
                "job": index,
                "output_dir": str(tmp_path / f"worker-{index}"),
            },
        )
        for index in range(2)
    )
    paths = spawn_manifest_workers_v12(
        "tests.test_retrieval_fold_fused_sharded_oof_v12:public_fixture_manifest_worker",
        jobs,
        max_workers=2,
    )
    assert [json.loads(path.read_text())["job"] for path in paths] == [0, 1]

    with pytest.raises((TypeError, FoldFusedInvariantError)):
        WorkerJobV12("bad", {"arrow": pa.table({"x": [1]})})


def test_strict_replay_records_and_full_subprocess_dag_are_w1_w3_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_retrieval_fold_fused_prefit_bridge_v12 import (
        _write_prefit_reference,
    )

    index = _scene_index()
    query_ids = tuple(index["query_id"].to_pylist())
    prefit_path = _write_prefit_reference(tmp_path / "strict_prefit")
    prefit_sha = sha256_file(prefit_path)
    plan_sha = _sha("fixture-sharded-plan")
    bridge_common = {
        "prefit_bridge": {
            "manifest_path": str(prefit_path),
            "manifest_sha256": prefit_sha,
            "prefit_lineage": _lineage(),
        }
    }
    one = _build(
        plan_query_shards(index, query_ids, 1)[0],
        tmp_path / "record_build",
        _labels(),
        [],
    )
    heldout_record = heldout_master_record_v12(one.heldout_master)
    assert load_heldout_master_record_v12(heldout_record) == one.heldout_master
    forged_heldout = dict(heldout_record)
    forged_heldout["row_count"] = int(forged_heldout["row_count"]) + 1
    with pytest.raises(FoldFusedInvariantError, match="manifest drift"):
        load_heldout_master_record_v12(forged_heldout)

    record_models = fit_site_models_sequential_v12(
        _Backend(tmp_path / "record_models"), one.train_matrices
    )
    model_records = fitted_site_model_records_v12(record_models)
    loaded_models = load_fitted_site_model_records_v12(
        model_records,
        handle_loader=lambda path, outer: json.loads(path.read_text(encoding="utf-8")),
    )
    assert {
        outer: model.sha256 for outer, model in loaded_models.items()
    } == {outer: model.sha256 for outer, model in record_models.items()}

    entrypoint_prefix = "tests.test_retrieval_fold_fused_sharded_oof_v12"

    def execute(worker_count: int, name: str):
        inputs = ShardedPhysicalInputsV12(
            scene_index=index,
            query_ids=query_ids,
            lineage=_lineage(),
            build_entrypoint=f"{entrypoint_prefix}:public_fixture_build_worker",
            build_payload=bridge_common,
            replay_entrypoint=f"{entrypoint_prefix}:public_fixture_replay_worker",
            replay_payload={},
            model_backend=_LazyBackend(tmp_path / name / "run" / "site_models"),
            caps=ExecutionCaps(fit_sample=4),
            maximum_spool_bytes=16 * 1024**2,
            maximum_workers=worker_count,
            execution_evidence={
                "phase": "PHYSICAL_PARITY_BENCHMARK_512",
                "sharded_implementation_sha256": None,
                "runtime_environment_sha256": None,
                "prefit_parity_manifest_sha256": prefit_sha,
                "sharded_execution_plan_sha256": plan_sha,
            },
        )
        shard_count = 1 if worker_count == 1 else 3
        return run_sharded_physical_sample(
            inputs,
            plan_query_shards(index, query_ids, shard_count),
            tmp_path / name / "run",
            miniature_fit=True,
        )

    w1 = execute(1, "full_w1")
    w2 = execute(2, "full_w2")
    w3 = execute(3, "full_w3")
    assert w1.digests == w2.digests
    assert w1.digests == w3.digests
    w2_manifest = json.loads(w2.manifest_path.read_text(encoding="utf-8"))
    assert w2_manifest["worker_count"] == 2
    assert w2_manifest["shard_count"] == 3
    assert len(w2.specs) == 3
    assert set(w1.digests) == {
        "train_matrix",
        "group_sequence",
        "heldout_spool",
        "model_by_outer",
        "prerank500",
        "top100",
        "outcomes",
        "prefit_bridge",
    }
    assert w1.prefit_bridge is not None
    assert json.loads(w1.manifest_path.read_text(encoding="utf-8"))[
        "prefit_bridge"
    ]["rows"] == 18
    assert w3.query_count == 6
    assert w3.folds == (2, 3, 4)
    assert w3.gates["exact_recall_at_100"]["rate"] == 1.0

    # The authorization-side loader must reconstruct its product from the
    # complete core tree, not from copied claims in the outer benchmark.
    monkeypatch.setattr(physical_benchmark, "SAMPLE_QUERY_COUNT", 6)
    reconstructed = (
        physical_benchmark.physical_trial_product_from_sharded_manifest_v12(
            w3.manifest_path,
            expected_lineage=_lineage(),
            expected_worker_count=3,
            expected_prefit_parity_manifest_path=prefit_path,
            expected_prefit_parity_manifest_sha256=prefit_sha,
            expected_sharded_execution_plan_sha256=plan_sha,
        )
    )
    assert reconstructed.digests == {
        "prefit_bridge": w3.prefit_bridge.sha256,
        "train_matrices": physical_benchmark._sha256_json(
            {str(fold): w3.train.matrices[fold].sha256 for fold in (2, 3, 4)}
        ),
        "query_groups": physical_benchmark._sha256_json(
            {
                str(fold): w3.train.group_sequence_sha256[fold]
                for fold in (2, 3, 4)
            }
        ),
        "heldout_spool": w3.heldout.artifact.sha256,
        "miniature_models": physical_benchmark._sha256_json(
            {str(fold): w3.models[fold].sha256 for fold in (2, 3, 4)}
        ),
        "prerank500": w3.replay.outputs["top500"].sha256,
        "top100": w3.replay.outputs["top100"].sha256,
        "outcomes": w3.replay.outputs["outcomes"].sha256,
    }


def test_complete_core_benchmark_issues_a_replayable_authorization(
    tmp_path: Path,
) -> None:
    from tests.test_retrieval_fold_fused_prefit_bridge_v12 import (
        _write_prefit_reference,
    )
    from tests.test_retrieval_fold_fused_physical_benchmark_v12 import (
        _lineage as _physical_lineage,
        _measurer,
        _resources,
        _sharded_plan,
    )

    candidate_count = 2
    index = _scene_index_512(candidate_count)
    query_ids = tuple(index["query_id"].to_pylist())
    lineage = _physical_lineage()
    execution_lineage = physical_benchmark.sharded_execution_lineage_v12(
        lineage, workspace_root=Path(__file__).resolve().parents[1]
    )
    prefit_path = _write_prefit_reference(
        tmp_path / "prefit",
        index=index,
        candidate_count=candidate_count,
        lineage=lineage,
    )
    prefit_sha = physical_benchmark.sha256_file(prefit_path)
    plan_path = tmp_path / "plan.json"
    _sharded_plan(plan_path, lineage, prefit_sha)
    plan_sha = physical_benchmark.sha256_file(plan_path)
    common = {
        "candidate_count": candidate_count,
        "prefit_bridge": {
            "manifest_path": str(prefit_path),
            "manifest_sha256": prefit_sha,
            "prefit_lineage": lineage,
        },
    }

    def runner(worker_count: int, output_dir: Path):
        shard_count = 1 if worker_count == 1 else 3
        artifact = run_sharded_physical_sample(
            ShardedPhysicalInputsV12(
                scene_index=index,
                query_ids=query_ids,
                lineage=execution_lineage,
                build_entrypoint=(
                    "tests.test_retrieval_fold_fused_sharded_oof_v12:"
                    "public_fixture_build_worker"
                ),
                build_payload=common,
                replay_entrypoint=(
                    "tests.test_retrieval_fold_fused_sharded_oof_v12:"
                    "public_fixture_replay_worker"
                ),
                replay_payload={"candidate_count": candidate_count},
                model_backend=_LazyBackend(output_dir / "site_models"),
                caps=ExecutionCaps(fit_sample=4),
                maximum_spool_bytes=512 * 1024**2,
                maximum_workers=worker_count,
                execution_evidence={
                    "phase": "PHYSICAL_PARITY_BENCHMARK_512",
                    "sharded_implementation_sha256": execution_lineage[
                        "sharded_implementation_sha256"
                    ],
                    "runtime_environment_sha256": execution_lineage[
                        "runtime_environment_sha256"
                    ],
                    "prefit_parity_manifest_sha256": prefit_sha,
                    "sharded_execution_plan_sha256": plan_sha,
                },
            ),
            plan_query_shards(index, query_ids, shard_count),
            output_dir,
            miniature_fit=True,
        )
        return physical_benchmark.physical_trial_product_from_sharded_v12(artifact)

    benchmark_path = physical_benchmark.run_physical_worker_benchmark_v12(
        runner=runner,
        output_dir=tmp_path / "benchmark",
        expected_lineage=execution_lineage,
        prefit_parity_manifest_path=prefit_path,
        prefit_parity_manifest_sha256=prefit_sha,
        sharded_execution_plan_path=plan_path,
        sharded_execution_plan_sha256=plan_sha,
        measurer=_measurer(
            {
                1: _resources(wall=30.0, io_bytes=1_000),
                3: _resources(wall=10.0, io_bytes=1_100),
            }
        ),
    )
    authorization = physical_benchmark.load_sharded_oof_authorization_v12(
        benchmark_path,
        benchmark_path.parent / "execution_authorization.json",
        expected_lineage=lineage,
        expected_prefit_parity_manifest_sha256=prefit_sha,
        expected_sharded_execution_plan_sha256=plan_sha,
    )
    assert authorization.worker_count == 3
