from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (
    CompactTrainBatch,
    ExecutionCaps,
    ExistingV12SelectionAdapter,
    FeatureBatch,
    FittedSiteModel,
    FoldFusedInvariantError,
    FoldFusedOOFExecutor,
    FullMasterStatistics,
    GateThresholds,
    HeldoutMasterArtifact,
    HumanLabel,
    MasterContext,
    ParquetCompactTrainingStore,
    ParquetOOFOutputStore,
    ParityEvidence,
    REQUIRED_PARITY_CHECKS,
    RankingSelection,
    RunLineage,
    SamplingPlan,
    SceneRecord,
    TrainMatrixArtifact,
    XGBoostSiteLambdaMARTBackend,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _siret(parent: int, site: int) -> str:
    return f"{parent:09d}{site:05d}"


class FixtureSource:
    def __init__(self, scenes: list[SceneRecord], events: list[tuple]) -> None:
        self.scenes = scenes
        self.events = events
        self.passes = 0

    def iter_scenes(self, *, folds: tuple[int, ...]):
        assert folds == (2, 3, 4)
        self.passes += 1
        self.events.append(("source_pass", self.passes))
        yield from self.scenes


class FixtureLedger:
    def __init__(self, labels: dict[str, HumanLabel], events: list[tuple]) -> None:
        self.labels = labels
        self.events = events

    def expected_query_ids(self, *, folds: tuple[int, ...]):
        assert folds == (2, 3, 4)
        return tuple(self.labels)

    def get(self, query_id: str) -> HumanLabel:
        self.events.append(("label", query_id))
        return self.labels[query_id]


class FixtureContexts:
    feature_names = ("signal", "joint_name_geo", "parent_score_probability")

    def __init__(self, truth: dict[str, str], events: list[tuple]) -> None:
        self.truth = truth
        self.events = events

    def prepare(self, scene: SceneRecord, *, outer_fold: int) -> MasterContext:
        self.events.append(("prepare", scene.query_id, outer_fold))
        base = 200_000_000 + int(scene.fold) * 10_000 + int(scene.query_id[-1]) * 100
        parents = tuple(f"{base + index:09d}" for index in range(6))
        sirets = tuple(_siret(base + index, 1) for index in range(6))
        assert sirets[-1] == self.truth[scene.query_id]
        return MasterContext(
            query_id=scene.query_id,
            scene_fold=scene.fold,
            outer_fold=outer_fold,
            candidate_sirets=sirets,
            candidate_parents=parents,
            opaque=(scene.query_id, outer_fold),
        )

    def compute_full_master_statistics(
        self, context: MasterContext
    ) -> FullMasterStatistics:
        digest = _sha(f"stats:{context.query_id}:{context.outer_fold}")
        self.events.append(("stats", context.query_id, context.outer_fold, digest))
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
        self.events.append(
            (
                "materialize_full" if candidate_indices is None else "materialize_sample",
                context.query_id,
                context.outer_fold,
                indices,
            )
        )
        rows = []
        payloads = []
        for index in indices:
            is_truth = context.candidate_sirets[index] == self.truth[context.query_id]
            rows.append([10.0 if is_truth else float(5 - index), float(is_truth), 0.2])
            payloads.append({"official_source": "SIRENE"})
        return FeatureBatch(
            query_id=context.query_id,
            scene_fold=context.scene_fold,
            outer_fold=context.outer_fold,
            candidate_indices=indices,
            candidate_sirets=tuple(context.candidate_sirets[index] for index in indices),
            candidate_parents=tuple(context.candidate_parents[index] for index in indices),
            feature_names=self.feature_names,
            values=np.asarray(rows, dtype=np.float32).reshape(len(indices), 3),
            statistics_sha256=statistics.statistics_sha256,
            row_payloads=tuple(payloads),
        )


class FixtureSampler:
    def __init__(self, events: list[tuple], *, corrupt_digest: bool = False) -> None:
        self.events = events
        self.corrupt_digest = corrupt_digest

    def choose(
        self,
        context: MasterContext,
        statistics: FullMasterStatistics,
        label: HumanLabel,
        *,
        maximum_rows: int,
    ) -> SamplingPlan:
        self.events.append(("sample", context.query_id, context.outer_fold))
        truth = context.candidate_sirets.index(label.human_siret)
        indices = tuple(dict.fromkeys((0, 1, truth)))[:maximum_rows]
        return SamplingPlan(
            candidate_indices=indices,
            statistics_sha256=("wrong" if self.corrupt_digest else statistics.statistics_sha256),
            rule_id="fixture-v12-sampler",
        )


class FixtureBackend:
    def __init__(self, output_dir: Path, events: list[tuple]) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events = events

    def fit(self, artifact: TrainMatrixArtifact) -> FittedSiteModel:
        self.events.append(("fit", artifact.outer_fold))
        path = self.output_dir / f"fold{artifact.outer_fold}_model.fixture"
        path.write_text(f"fixture model outer={artifact.outer_fold}\n")
        return FittedSiteModel(
            outer_fold=artifact.outer_fold,
            handle={"outer": artifact.outer_fold},
            path=path,
            sha256=_sha(path.read_text()),
            recipe_sha256=_sha("frozen-site-recipe"),
            feature_names=artifact.feature_names,
        )

    def score(self, model: FittedSiteModel, features: FeatureBatch) -> np.ndarray:
        self.events.append(("score", features.query_id, model.outer_fold))
        return features.values[:, 0].astype(np.float64)


class FixtureHeldoutStore:
    def __init__(self, output_dir: Path, events: list[tuple]) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events = events
        self.contexts: list[MasterContext] = []
        self.artifact: HeldoutMasterArtifact | None = None

    def append(self, context: MasterContext) -> None:
        self.events.append(("spool", context.query_id, context.outer_fold))
        assert context.outer_fold == context.scene_fold
        self.contexts.append(context)

    def seal(self) -> HeldoutMasterArtifact:
        path = self.output_dir / "heldout.fixture"
        manifest = self.output_dir / "manifest.fixture"
        path.write_text("\n".join(context.query_id for context in self.contexts))
        manifest.write_text("sealed\n")
        self.artifact = HeldoutMasterArtifact(
            path=path,
            manifest_path=manifest,
            sha256=_sha(path.read_text()),
            manifest_sha256=_sha(manifest.read_text()),
            row_count=sum(len(context.candidate_sirets) for context in self.contexts),
            query_count=len(self.contexts),
            size_bytes=path.stat().st_size,
            maximum_bytes=20 * 1024**3,
            schema_sha256=_sha("fixture-schema"),
            query_sequence_sha256=_sha("fixture-sequence"),
        )
        self.events.append(("spool_seal", len(self.contexts)))
        return self.artifact

    def iter_contexts(self, artifact: HeldoutMasterArtifact):
        assert artifact == self.artifact
        self.events.append(("spool_replay", len(self.contexts)))
        yield from self.contexts


class FixtureSelector:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def select(
        self,
        context: MasterContext,
        features: FeatureBatch,
        scores: np.ndarray,
        *,
        prerank_cap: int,
        product_cap: int,
    ) -> RankingSelection:
        self.events.append(("select", context.query_id, context.outer_fold))
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
            product_reasons=("fixture_adaptive",) * len(product),
        )


def _fixture(tmp_path: Path, *, corrupt_digest: bool = False):
    events: list[tuple] = []
    scenes: list[SceneRecord] = []
    labels: dict[str, HumanLabel] = {}
    for fold in (2, 3, 4):
        for index in range(2):
            query_id = f"f{fold}q{index}"
            base = 200_000_000 + fold * 10_000 + index * 100
            truth = _siret(base + 5, 1)
            scenes.append(SceneRecord(query_id=query_id, fold=fold))
            labels[query_id] = HumanLabel(
                query_id=query_id,
                fold=fold,
                human_siret=truth,
                operational_sirets=(truth,),
                audit_metadata={"official_catalog_present": True},
            )
    source = FixtureSource(scenes, events)
    ledger = FixtureLedger(labels, events)
    contexts = FixtureContexts(
        {query_id: label.human_siret for query_id, label in labels.items()}, events
    )
    work = tmp_path / "fused"
    work.mkdir()
    executor = FoldFusedOOFExecutor(
        source=source,
        labels=ledger,
        contexts=contexts,
        sampler=FixtureSampler(events, corrupt_digest=corrupt_digest),
        train_store=ParquetCompactTrainingStore(work / "train"),
        heldout_store=FixtureHeldoutStore(work / "heldout", events),
        model_backend=FixtureBackend(work / "models", events),
        selector=FixtureSelector(events),
        output_store=ParquetOOFOutputStore(work / "outputs"),
        lineage=RunLineage(
            scene_ir_sha256=_sha("scene"),
            scene_ir_manifest_sha256=_sha("scene-manifest"),
            label_ledger_sha256=_sha("labels"),
            label_ledger_manifest_sha256=_sha("labels-manifest"),
            label_sidecar_sha256=_sha("label-sidecar"),
            scientific_config_sha256=_sha("science-config"),
            scientific_contract_sha256=_sha("science-contract"),
            parent_model_bundle_sha256=_sha("parent-bundle"),
            executor_config_sha256=_sha("executor-config"),
        ),
        manifest_path=work / "manifest.json",
        caps=ExecutionCaps(fit_sample=4),
        parity=ParityEvidence(),
    )
    return executor, source, events


def test_fold_fused_executor_streams_one_source_pass_and_replays_heldout_spool(
    tmp_path: Path,
) -> None:
    executor, source, events = _fixture(tmp_path)
    result = executor.run()

    assert source.passes == 1
    assert set(result.train_matrices) == {2, 3, 4}
    assert set(result.models) == {2, 3, 4}
    assert [event for event in events if event[0] == "fit"] == [
        ("fit", 2),
        ("fit", 3),
        ("fit", 4),
    ]
    # Six queries produce two <=128 training contexts and one heldout full
    # Master context each.  No full training Master is written.
    assert len([event for event in events if event[0] == "materialize_sample"]) == 12
    assert len([event for event in events if event[0] == "materialize_full"]) == 6
    assert len([event for event in events if event[0] == "spool"]) == 6
    assert len([event for event in events if event[0] == "spool_replay"]) == 1
    for artifact in result.train_matrices.values():
        assert artifact.query_count == 4
        assert artifact.row_count == 12
        table = pq.read_table(artifact.path)
        assert table.num_rows == 12
        assert table.column_names == [
            "query_id",
            "scene_fold",
            "candidate_siret",
            "candidate_parent",
            "site_label",
            "statistics_sha256",
            "feature_values",
        ]

    assert len(result.outcomes) == 6
    assert all(outcome.master_hit_at_2000 for outcome in result.outcomes)
    assert all(outcome.prerank_hit_at_500 for outcome in result.outcomes)
    assert all(outcome.exact_hit_at_100 for outcome in result.outcomes)
    assert result.outputs["top500"].rows == 36
    assert result.outputs["top100"].rows == 18
    assert result.outputs["outcomes"].rows == 6
    assert result.manifest["physical_plan"] == {
        "scene_ir_passes_after_parent_fit": 1,
        "training_contexts_per_query": 2,
        "heldout_full_master_contexts_per_query": 1,
        "heldout_replay_source": "SEALED_EXTERNAL_PARQUET_SPOOL",
        "source_or_sidecar_reopened_after_fit": False,
        "full_master_statistics_before_fit_sampling": True,
        "fit_sampling_maximum": 4,
        "validation_candidate_sampling": False,
        "wide_master_materialized": False,
    }
    spool_manifest = result.manifest["heldout_master_spool"]
    assert spool_manifest["actual_size_bytes"] == result.heldout_master.size_bytes
    assert spool_manifest["projected_national_size_bytes"] == result.heldout_master.size_bytes
    assert spool_manifest["projection_basis_queries"] == 6
    assert spool_manifest["maximum_bytes"] == 20 * 1024**3
    assert spool_manifest["budget_pass"] is True
    assert result.manifest["fold0_opened"] is False
    assert result.manifest["fold1_opened"] is False
    assert result.manifest["parity"]["status"] == "NOT_RUN"
    assert result.manifest["gates"]["gate_eligible"] is False
    assert result.manifest["gates"]["decision"] == "NO_READY_DECISION"


def test_labels_are_read_only_after_complete_stats_and_after_selection(
    tmp_path: Path,
) -> None:
    executor, _source, events = _fixture(tmp_path)
    executor.run()

    first_pass_end = next(
        index for index, event in enumerate(events) if event[0] == "fit"
    )
    training_events = events[:first_pass_end]
    for query_id in (f"f{fold}q{index}" for fold in (2, 3, 4) for index in range(2)):
        label_index = next(
            index
            for index, event in enumerate(training_events)
            if event == ("label", query_id)
        )
        stats_before_label = [
            event
            for event in training_events[:label_index]
            if event[0] == "stats" and event[1] == query_id
        ]
        assert len(stats_before_label) == 2

    validation_events = events[first_pass_end:]
    for query_id in (f"f{fold}q{index}" for fold in (2, 3, 4) for index in range(2)):
        select_index = next(
            index
            for index, event in enumerate(validation_events)
            if event[0] == "select" and event[1] == query_id
        )
        label_index = next(
            index
            for index, event in enumerate(validation_events)
            if event == ("label", query_id)
        )
        assert select_index < label_index


def test_sampling_must_cite_the_complete_master_statistics(tmp_path: Path) -> None:
    executor, _source, _events = _fixture(tmp_path, corrupt_digest=True)
    with pytest.raises(FoldFusedInvariantError, match="sealed full-Master statistics"):
        executor.run()


def test_master_headroom_is_diagnostic_not_a_generation_gate(tmp_path: Path) -> None:
    executor, _source, _events = _fixture(tmp_path)
    executor.gates = GateThresholds(
        master_recall_at_2000=1.01,
        prerank_recall_at_500=0.0,
        top100_recall=0.0,
        top100_wilson95_lower=0.0,
        minimum_each_outer_fold_top100=0.0,
    )
    gates = executor.run().manifest["gates"]

    assert gates["master_headroom_target_met"] is False
    assert gates["master_decision_blocking"] is False
    assert gates["recall_gates_pass"] is True
    assert gates["decision"] == "NO_READY_DECISION"


def test_oof_executor_refuses_postfit_parity(tmp_path: Path) -> None:
    executor, _source, _events = _fixture(tmp_path)
    executor.parity = ParityEvidence(
        reference_manifest_sha256=_sha("reference"),
        query_count=512,
        checks={name: True for name in REQUIRED_PARITY_CHECKS},
    )
    with pytest.raises(FoldFusedInvariantError, match="refuses POSTFIT parity"):
        executor.run()


def test_oof_cli_has_no_postfit_parity_argument() -> None:
    from scripts.run_retrieval_fold_fused_oof_v12 import parser

    required = [
        "--scene-ir",
        "scene",
        "--labels",
        "labels",
        "--sidecars",
        "sidecars",
        "--mmap-sidecar",
        "mmap",
        "--reader-parity-manifest",
        "reader.json",
        "--parent-model-bundle",
        "parent.json",
        "--parity-manifest",
        "postfit.json",
    ]
    with pytest.raises(SystemExit):
        parser().parse_args(required)


def test_fold_and_feature_leakage_guards_fail_closed() -> None:
    with pytest.raises(FoldFusedInvariantError, match="folds 2/3/4"):
        SceneRecord(query_id="forbidden", fold=1)

    with pytest.raises(FoldFusedInvariantError, match="identifier/label features"):
        FeatureBatch(
            query_id="q",
            scene_fold=2,
            outer_fold=2,
            candidate_indices=(0,),
            candidate_sirets=("12345678900001",),
            candidate_parents=("123456789",),
            feature_names=("ground_truth_siret",),
            values=np.zeros((1, 1), dtype=np.float32),
            statistics_sha256=_sha("stats"),
        )


def test_existing_v12_selector_adapter_keeps_score_then_siret_tie_break() -> None:
    config = SimpleNamespace(
        prerank_cap=500,
        final_cap=100,
        selection={"parent_probability_weight": 0.2, "repeat_parent_penalty": 0.035},
    )
    adapter = ExistingV12SelectionAdapter(config)
    sirets = ("12345678900003", "12345678900001", "12345678900002")
    parents = ("123456789",) * 3
    context = MasterContext(
        query_id="q",
        scene_fold=2,
        outer_fold=2,
        candidate_sirets=sirets,
        candidate_parents=parents,
    )
    features = FeatureBatch(
        query_id="q",
        scene_fold=2,
        outer_fold=2,
        candidate_indices=(0, 1, 2),
        candidate_sirets=sirets,
        candidate_parents=parents,
        feature_names=("joint_name_geo", "parent_score_probability"),
        values=np.zeros((3, 2), dtype=np.float32),
        statistics_sha256=_sha("stats"),
    )
    selected = adapter.select(
        context,
        features,
        np.asarray([1.0, 1.0, 1.0]),
        prerank_cap=500,
        product_cap=100,
    )
    assert selected.prerank_indices == (1, 2, 0)
    assert selected.product_indices == (1, 2, 0)


def test_xgboost_adapter_fits_frozen_ranker_shape_from_compact_parquet(
    tmp_path: Path,
) -> None:
    store = ParquetCompactTrainingStore(tmp_path / "train")
    feature_names = ("name_signal", "geo_signal")
    for query_index, scene_fold in enumerate((3, 4)):
        parent = 300_000_000 + query_index
        sirets = (_siret(parent, 1), _siret(parent + 10, 1), _siret(parent + 20, 1))
        store.append(
            CompactTrainBatch(
                query_id=f"q{query_index}",
                scene_fold=scene_fold,
                outer_fold=2,
                candidate_sirets=sirets,
                candidate_parents=tuple(value[:9] for value in sirets),
                feature_names=feature_names,
                values=np.asarray([[1.0, 1.0], [0.2, 0.4], [0.1, 0.3]], dtype=np.float32),
                labels=np.asarray([1, 0, 0], dtype=np.int8),
                statistics_sha256=_sha(f"stats-{query_index}"),
            )
        )
    artifact = store.seal_outer(2)
    backend = XGBoostSiteLambdaMARTBackend(
        params={
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@2",
            "tree_method": "hist",
            "n_estimators": 2,
            "max_depth": 2,
            "learning_rate": 0.2,
            "n_jobs": 1,
            "random_state": 7,
            "lambdarank_pair_method": "topk",
            "lambdarank_num_pair_per_sample": 2,
        },
        output_dir=tmp_path / "models",
    )
    model = backend.fit(artifact)
    features = FeatureBatch(
        query_id="heldout",
        scene_fold=2,
        outer_fold=2,
        candidate_indices=(0, 1),
        candidate_sirets=("40000000000001", "40000001000001"),
        candidate_parents=("400000000", "400000010"),
        feature_names=feature_names,
        values=np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        statistics_sha256=_sha("heldout-stats"),
    )
    scores = backend.score(model, features)
    assert model.outer_fold == 2
    assert model.path.exists()
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
