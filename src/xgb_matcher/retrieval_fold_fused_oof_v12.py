"""Fold-fused physical executor for the frozen retrieval-v12 site recipe.

This module changes the execution plan, not the scientific recipe.  It is
deliberately expressed against small protocols so that the SceneIR compiler
and the vectorised context kernel can evolve independently.

The executor performs exactly one SceneIR/cascade pass after the nested parent
models exist:

1. for every query, build the two training contexts, seal statistics on each
   complete Master, materialise at most 128 sampled feature rows, and spool the
   single held-out label-blind Master to external Parquet;
2. after fitting one site LambdaMART per outer fold, replay only that compact
   spool, compute the unchanged full-Master features, then emit PreRank500,
   top100 and outcome.  SceneIR and official sidecars are never reopened.

Human labels are supplied by a separate sealed ledger.  They are never passed
to SceneIR or the context engine.  Fold 0 and fold 1 are rejected before any
context is prepared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable
import hashlib
import json
import math
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ENGINE_ID = "FOLD_FUSED_CROSSFIT_COMPILER_V12_1"
MANIFEST_SCHEMA_VERSION = "sireto-retrieval-v12-fold-fused-oof-manifest-1"
PARITY_SCHEMA_VERSION = "sireto-retrieval-v12-fold-fused-parity-1"
GATE_SCHEMA_VERSION = "sireto-retrieval-v12-fold-fused-gates-1"
TRAIN_FOLDS = (2, 3, 4)
FORBIDDEN_FOLDS = (0, 1)

REQUIRED_PARITY_CHECKS = (
    "parent_scores_bit_identical",
    "master_candidates_and_ranks_byte_identical",
    "site_features_float32_bit_identical",
    "fit_row_ids_and_order_byte_identical",
    "prerank500_byte_identical",
    "top100_byte_identical",
)

_LABEL_KEYS = frozenset(
    {
        "human_siret",
        "human_ground_truth_siret",
        "historical_ground_truth_siret",
        "ground_truth_siret",
        "gt_siret",
        "acceptable_sirets_operational",
        "acceptable_sirets_operational_json",
        "operational_sirets_json",
    }
)
_FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "siret",
        "siren",
        "candidate_siret",
        "candidate_siren",
        "candidate_parent",
        "human_siret",
        "ground_truth_siret",
        "announcement_id",
        "bodacc_announcement_id",
        "crm_alias",
    }
)


class FoldFusedInvariantError(ValueError):
    """A physical-execution invariant was violated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _schema_hash(feature_names: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_json(list(feature_names)).encode("utf-8"))


def _candidate_order_hash(sirets: Sequence[str], parents: Sequence[str]) -> str:
    return _sha256_bytes(
        _canonical_json(list(zip(sirets, parents, strict=True))).encode("utf-8")
    )


def _assert_blind_mapping(value: Mapping[str, Any], *, where: str) -> None:
    forbidden = sorted(_LABEL_KEYS.intersection(str(key) for key in value))
    if forbidden:
        raise FoldFusedInvariantError(
            f"{where} contains forbidden label-side keys: {forbidden}"
        )


def _assert_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in feature_names)
    if not names or len(names) != len(set(names)):
        raise FoldFusedInvariantError("feature names must be non-empty and unique")
    forbidden = sorted(set(names).intersection(_FORBIDDEN_FEATURE_NAMES))
    if forbidden:
        raise FoldFusedInvariantError(
            f"identifier/label features are forbidden: {forbidden}"
        )
    return names


@dataclass(frozen=True)
class ExecutionCaps:
    master: int = 2_000
    fit_sample: int = 128
    prerank: int = 500
    product: int = 100

    def __post_init__(self) -> None:
        if self.master != 2_000 or self.fit_sample > 128:
            raise FoldFusedInvariantError(
                "v12 requires Master<=2000 and fit sampling<=128"
            )
        if self.prerank != 500 or self.product != 100:
            raise FoldFusedInvariantError("v12 requires PreRank500 and top100")
        if min(self.master, self.fit_sample, self.prerank, self.product) <= 0:
            raise FoldFusedInvariantError("execution caps must be positive")


@dataclass(frozen=True)
class GateThresholds:
    master_recall_at_2000: float = 0.995
    prerank_recall_at_500: float = 0.994
    top100_recall: float = 0.992
    top100_wilson95_lower: float = 0.990
    minimum_each_outer_fold_top100: float = 0.988


@dataclass(frozen=True)
class SceneRecord:
    """Label-blind query reference produced by SceneIR."""

    query_id: str
    fold: int
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.query_id):
            raise FoldFusedInvariantError("SceneIR query_id must be non-empty")
        if int(self.fold) not in TRAIN_FOLDS:
            raise FoldFusedInvariantError("fold-fused execution may open only folds 2/3/4")
        _assert_blind_mapping(self.metadata, where="SceneIR metadata")


@dataclass(frozen=True)
class HumanLabel:
    query_id: str
    fold: int
    human_siret: str
    operational_sirets: tuple[str, ...] = ()
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.fold) not in TRAIN_FOLDS:
            raise FoldFusedInvariantError("label ledger may expose only folds 2/3/4")
        if not str(self.query_id) or len(str(self.human_siret)) != 14:
            raise FoldFusedInvariantError("usable human labels require query_id and 14-digit SIRET")
        if not str(self.human_siret).isdigit():
            raise FoldFusedInvariantError("human SIRET must be numeric")


@dataclass(frozen=True)
class MasterContext:
    """A label-blind, context-specific complete Master descriptor."""

    query_id: str
    scene_fold: int
    outer_fold: int
    candidate_sirets: tuple[str, ...]
    candidate_parents: tuple[str, ...]
    opaque: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.scene_fold) not in TRAIN_FOLDS or int(self.outer_fold) not in TRAIN_FOLDS:
            raise FoldFusedInvariantError("context folds must be 2/3/4")
        if len(self.candidate_sirets) != len(self.candidate_parents):
            raise FoldFusedInvariantError("candidate SIRET/parent arrays differ in length")
        if len(set(self.candidate_sirets)) != len(self.candidate_sirets):
            raise FoldFusedInvariantError("Master candidates must be unique by SIRET")
        if any(len(value) != 14 or not value.isdigit() for value in self.candidate_sirets):
            raise FoldFusedInvariantError("Master candidate SIRETs must be 14 digits")
        if any(len(value) != 9 or not value.isdigit() for value in self.candidate_parents):
            raise FoldFusedInvariantError("Master candidate parents must be 9 digits")
        _assert_blind_mapping(self.metadata, where="Master metadata")

    @property
    def candidate_order_sha256(self) -> str:
        return _candidate_order_hash(self.candidate_sirets, self.candidate_parents)


@dataclass(frozen=True)
class FullMasterStatistics:
    query_id: str
    outer_fold: int
    master_count: int
    candidate_order_sha256: str
    statistics_sha256: str
    complete: bool
    opaque: Any = None


@dataclass(frozen=True)
class SamplingPlan:
    candidate_indices: tuple[int, ...]
    statistics_sha256: str
    rule_id: str


@dataclass(frozen=True)
class FeatureBatch:
    query_id: str
    scene_fold: int
    outer_fold: int
    candidate_indices: tuple[int, ...]
    candidate_sirets: tuple[str, ...]
    candidate_parents: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray
    statistics_sha256: str
    row_payloads: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        names = _assert_feature_names(self.feature_names)
        values = np.asarray(self.values)
        if values.dtype != np.float32:
            raise FoldFusedInvariantError("site feature matrices must be float32")
        if values.ndim != 2 or values.shape != (len(self.candidate_indices), len(names)):
            raise FoldFusedInvariantError("feature matrix shape does not match rows/schema")
        lengths = {
            len(self.candidate_indices),
            len(self.candidate_sirets),
            len(self.candidate_parents),
        }
        if len(lengths) != 1:
            raise FoldFusedInvariantError("feature row identities differ in length")
        if self.row_payloads and len(self.row_payloads) != len(self.candidate_indices):
            raise FoldFusedInvariantError("row payload count differs from feature rows")
        for payload in self.row_payloads:
            _assert_blind_mapping(payload, where="feature row payload")
        if not np.isfinite(values).all():
            raise FoldFusedInvariantError("site feature matrix contains non-finite values")

    def take(self, positions: Sequence[int]) -> "FeatureBatch":
        selected = tuple(int(value) for value in positions)
        return FeatureBatch(
            query_id=self.query_id,
            scene_fold=self.scene_fold,
            outer_fold=self.outer_fold,
            candidate_indices=tuple(self.candidate_indices[index] for index in selected),
            candidate_sirets=tuple(self.candidate_sirets[index] for index in selected),
            candidate_parents=tuple(self.candidate_parents[index] for index in selected),
            feature_names=self.feature_names,
            values=np.ascontiguousarray(self.values[list(selected)], dtype=np.float32),
            statistics_sha256=self.statistics_sha256,
            row_payloads=(
                tuple(self.row_payloads[index] for index in selected)
                if self.row_payloads
                else ()
            ),
        )


@dataclass(frozen=True)
class CompactTrainBatch:
    query_id: str
    scene_fold: int
    outer_fold: int
    candidate_sirets: tuple[str, ...]
    candidate_parents: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray
    labels: np.ndarray
    statistics_sha256: str

    def __post_init__(self) -> None:
        if self.scene_fold == self.outer_fold:
            raise FoldFusedInvariantError("heldout rows cannot enter an outer training matrix")
        if len(self.candidate_sirets) > 128:
            raise FoldFusedInvariantError("site fit batch exceeds 128 rows")
        if np.asarray(self.values).dtype != np.float32:
            raise FoldFusedInvariantError("compact features must remain float32")
        labels = np.asarray(self.labels)
        if labels.dtype != np.int8 or labels.shape != (len(self.candidate_sirets),):
            raise FoldFusedInvariantError("compact labels must be one int8 value per row")
        if int(labels.sum()) != 1 or len(labels) < 2:
            raise FoldFusedInvariantError("every retained LambdaMART group needs one positive and a negative")


@dataclass(frozen=True)
class TrainMatrixArtifact:
    outer_fold: int
    path: Path
    sha256: str
    row_count: int
    query_count: int
    fit_folds: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_schema_sha256: str


@dataclass(frozen=True)
class HeldoutMasterArtifact:
    path: Path
    manifest_path: Path
    sha256: str
    manifest_sha256: str
    row_count: int
    query_count: int
    size_bytes: int
    maximum_bytes: int
    schema_sha256: str
    query_sequence_sha256: str


@dataclass(frozen=True)
class FittedSiteModel:
    outer_fold: int
    handle: Any
    path: Path
    sha256: str
    recipe_sha256: str
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class RankingSelection:
    prerank_indices: tuple[int, ...]
    product_indices: tuple[int, ...]
    product_reasons: tuple[str, ...]


@dataclass(frozen=True)
class QueryOutcome:
    query_id: str
    fold: int
    human_siret: str
    master_candidate_count: int
    prerank_candidate_count: int
    product_candidate_count: int
    truth_master_rank: int | None
    truth_prerank_rank: int | None
    truth_product_rank: int | None
    master_hit_at_2000: bool
    prerank_hit_at_500: bool
    exact_hit_at_100: bool
    operational_hit_at_100: bool
    loss_reason: str
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactRecord:
    path: Path
    sha256: str
    rows: int


@dataclass(frozen=True)
class RunLineage:
    scene_ir_sha256: str
    scene_ir_manifest_sha256: str
    label_ledger_sha256: str
    label_ledger_manifest_sha256: str
    label_sidecar_sha256: str
    scientific_config_sha256: str
    scientific_contract_sha256: str
    parent_model_bundle_sha256: str
    executor_config_sha256: str
    source_candidate_sha256: str | None = None
    source_manifest_sha256: str | None = None
    official_sidecar_manifest_sha256: str | None = None
    mmap_sidecar_manifest_sha256: str | None = None
    reader_parity_evidence_sha256: str | None = None
    query_sample_sha256: str | None = None
    prefit_parity_evidence_sha256: str | None = None
    parity_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value is None:
                continue
            if len(str(value)) != 64 or any(char not in "0123456789abcdef" for char in str(value)):
                raise FoldFusedInvariantError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class ParityEvidence:
    reference_manifest_sha256: str | None = None
    query_count: int = 0
    checks: Mapping[str, bool] = field(default_factory=dict)

    def as_manifest(self) -> Mapping[str, Any]:
        if self.reference_manifest_sha256 is None:
            return {
                "schema_version": PARITY_SCHEMA_VERSION,
                "status": "NOT_RUN",
                "query_count": 0,
                "required_checks": list(REQUIRED_PARITY_CHECKS),
                "checks": {},
                "stop_on_any_divergence": True,
            }
        if (
            len(self.reference_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.reference_manifest_sha256
            )
        ):
            raise FoldFusedInvariantError(
                "parity reference manifest must be a lowercase SHA-256"
            )
        if self.query_count != 512:
            raise FoldFusedInvariantError("the preregistered parity gate requires 512 queries")
        missing = sorted(set(REQUIRED_PARITY_CHECKS) - set(self.checks))
        if missing:
            raise FoldFusedInvariantError(f"parity evidence misses checks: {missing}")
        status = "PASS" if all(bool(self.checks[name]) for name in REQUIRED_PARITY_CHECKS) else "FAIL"
        return {
            "schema_version": PARITY_SCHEMA_VERSION,
            "status": status,
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "query_count": int(self.query_count),
            "required_checks": list(REQUIRED_PARITY_CHECKS),
            "checks": {name: bool(self.checks[name]) for name in REQUIRED_PARITY_CHECKS},
            "stop_on_any_divergence": True,
        }


@dataclass(frozen=True)
class FoldFusedRunResult:
    manifest_path: Path
    train_matrices: Mapping[int, TrainMatrixArtifact]
    heldout_master: HeldoutMasterArtifact
    models: Mapping[int, FittedSiteModel]
    outputs: Mapping[str, ArtifactRecord]
    outcomes: tuple[QueryOutcome, ...]
    manifest: Mapping[str, Any]


@runtime_checkable
class SceneSource(Protocol):
    def iter_scenes(self, *, folds: tuple[int, ...]) -> Iterable[SceneRecord]: ...


@runtime_checkable
class SealedLabelLedger(Protocol):
    def expected_query_ids(self, *, folds: tuple[int, ...]) -> Sequence[str]: ...

    def get(self, query_id: str) -> HumanLabel: ...


@runtime_checkable
class VectorizedContextAPI(Protocol):
    def prepare(self, scene: SceneRecord, *, outer_fold: int) -> MasterContext: ...

    def compute_full_master_statistics(
        self, context: MasterContext
    ) -> FullMasterStatistics: ...

    def materialize(
        self,
        context: MasterContext,
        statistics: FullMasterStatistics,
        *,
        candidate_indices: Sequence[int] | None,
    ) -> FeatureBatch: ...


@runtime_checkable
class FitSampler(Protocol):
    def choose(
        self,
        context: MasterContext,
        statistics: FullMasterStatistics,
        label: HumanLabel,
        *,
        maximum_rows: int,
    ) -> SamplingPlan: ...


@runtime_checkable
class CompactTrainingStore(Protocol):
    def append(self, batch: CompactTrainBatch) -> None: ...

    def seal_outer(self, outer_fold: int) -> TrainMatrixArtifact: ...


@runtime_checkable
class HeldoutMasterStore(Protocol):
    """External, label-blind replay boundary for the one held-out context."""

    def append(self, context: MasterContext) -> None: ...

    def seal(self) -> HeldoutMasterArtifact: ...

    def iter_contexts(
        self, artifact: HeldoutMasterArtifact
    ) -> Iterable[MasterContext]: ...


@runtime_checkable
class SiteLambdaMARTBackend(Protocol):
    def fit(self, artifact: TrainMatrixArtifact) -> FittedSiteModel: ...

    def score(self, model: FittedSiteModel, features: FeatureBatch) -> np.ndarray: ...


@runtime_checkable
class ProductSelector(Protocol):
    def select(
        self,
        context: MasterContext,
        features: FeatureBatch,
        scores: np.ndarray,
        *,
        prerank_cap: int,
        product_cap: int,
    ) -> RankingSelection: ...


@runtime_checkable
class OOFOutputSink(Protocol):
    def append_rankings(
        self,
        context: MasterContext,
        features: FeatureBatch,
        scores: np.ndarray,
        selection: RankingSelection,
    ) -> None: ...

    def append_outcome(self, outcome: QueryOutcome) -> None: ...

    def seal(self) -> Mapping[str, ArtifactRecord]: ...


class ParquetCompactTrainingStore:
    """Streaming, narrow float32 training matrices, one file per outer fold."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[int, pq.ParquetWriter] = {}
        self._paths = {
            fold: self.output_dir / f"outer{fold}_site_train_matrix.parquet"
            for fold in TRAIN_FOLDS
        }
        self._rows = {fold: 0 for fold in TRAIN_FOLDS}
        self._queries: dict[int, set[str]] = {fold: set() for fold in TRAIN_FOLDS}
        self._features: tuple[str, ...] | None = None

    def _table(self, batch: CompactTrainBatch) -> pa.Table:
        names = _assert_feature_names(batch.feature_names)
        if self._features is None:
            self._features = names
        elif names != self._features:
            raise FoldFusedInvariantError("feature schema drift across compact matrices")
        width = len(names)
        flat = pa.array(batch.values.reshape(-1), type=pa.float32())
        matrix = pa.FixedSizeListArray.from_arrays(flat, width)
        count = len(batch.candidate_sirets)
        table = pa.table(
            {
                "query_id": pa.array([batch.query_id] * count, type=pa.string()),
                "scene_fold": pa.array([batch.scene_fold] * count, type=pa.int8()),
                "candidate_siret": pa.array(batch.candidate_sirets, type=pa.string()),
                "candidate_parent": pa.array(batch.candidate_parents, type=pa.string()),
                "site_label": pa.array(batch.labels, type=pa.int8()),
                "statistics_sha256": pa.array(
                    [batch.statistics_sha256] * count, type=pa.string()
                ),
                "feature_values": matrix,
            }
        )
        metadata = dict(table.schema.metadata or {})
        metadata[b"feature_names_json"] = _canonical_json(list(names)).encode("utf-8")
        metadata[b"feature_schema_sha256"] = _schema_hash(names).encode("ascii")
        return table.replace_schema_metadata(metadata)

    def append(self, batch: CompactTrainBatch) -> None:
        outer = int(batch.outer_fold)
        if outer not in TRAIN_FOLDS:
            raise FoldFusedInvariantError("unknown outer fold")
        if batch.query_id in self._queries[outer]:
            raise FoldFusedInvariantError("query appended twice to one outer train matrix")
        table = self._table(batch)
        writer = self._writers.get(outer)
        if writer is None:
            writer = pq.ParquetWriter(self._paths[outer], table.schema, compression="zstd")
            self._writers[outer] = writer
        writer.write_table(table)
        self._queries[outer].add(batch.query_id)
        self._rows[outer] += len(batch.candidate_sirets)

    def seal_outer(self, outer_fold: int) -> TrainMatrixArtifact:
        outer = int(outer_fold)
        writer = self._writers.pop(outer, None)
        if writer is None or self._features is None:
            raise FoldFusedInvariantError(f"outer{outer} has no trainable rows")
        writer.close()
        path = self._paths[outer]
        return TrainMatrixArtifact(
            outer_fold=outer,
            path=path,
            sha256=sha256_file(path),
            row_count=self._rows[outer],
            query_count=len(self._queries[outer]),
            fit_folds=tuple(fold for fold in TRAIN_FOLDS if fold != outer),
            feature_names=self._features,
            feature_schema_sha256=_schema_hash(self._features),
        )


class XGBoostSiteLambdaMARTBackend:
    """Adapter that fits the frozen XGBRanker recipe from compact matrices."""

    def __init__(self, *, params: Mapping[str, Any], output_dir: Path) -> None:
        self.params = dict(params)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recipe_sha256 = _sha256_bytes(
            _canonical_json(self.params).encode("utf-8")
        )

    @staticmethod
    def _load(artifact: TrainMatrixArtifact) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = pq.read_table(
            artifact.path, columns=["query_id", "site_label", "feature_values"]
        ).combine_chunks()
        query_ids = table["query_id"].to_pylist()
        if not query_ids:
            raise FoldFusedInvariantError("empty LambdaMART matrix")
        seen: set[str] = set()
        last: str | None = None
        qid = np.empty(len(query_ids), dtype=np.int64)
        group = -1
        for index, value in enumerate(query_ids):
            current = str(value)
            if current != last:
                if current in seen:
                    raise FoldFusedInvariantError("query groups must be contiguous")
                seen.add(current)
                group += 1
                last = current
            qid[index] = group
        feature_column = table["feature_values"].chunk(0)
        width = len(artifact.feature_names)
        values = np.asarray(feature_column.values, dtype=np.float32).reshape(-1, width)
        labels = np.asarray(table["site_label"], dtype=np.int8)
        return values, labels, qid

    def fit(self, artifact: TrainMatrixArtifact) -> FittedSiteModel:
        import xgboost as xgb

        values, labels, qid = self._load(artifact)
        model = xgb.XGBRanker(**self.params)
        model.fit(values, labels, qid=qid, verbose=False)
        path = self.output_dir / f"fold{artifact.outer_fold}_site_ranker.json"
        model.save_model(path)
        return FittedSiteModel(
            outer_fold=artifact.outer_fold,
            handle=model,
            path=path,
            sha256=sha256_file(path),
            recipe_sha256=self.recipe_sha256,
            feature_names=artifact.feature_names,
        )

    def score(self, model: FittedSiteModel, features: FeatureBatch) -> np.ndarray:
        if features.feature_names != model.feature_names:
            raise FoldFusedInvariantError("validation feature schema differs from fit")
        if not len(features.candidate_sirets):
            return np.empty(0, dtype=np.float64)
        return np.asarray(model.handle.predict(features.values), dtype=np.float64)


class ExistingV12SelectionAdapter:
    """Exact NumPy execution of the frozen adaptive top100 semantics.

    The historical :func:`select_top100` remains the scientific oracle.  This
    adapter performs the same ordering and marginal allocation directly on the
    already contiguous feature matrix, avoiding a per-query pandas frame and
    roughly 50,000 ``DataFrame.loc`` calls at Master=2,000.
    """

    def __init__(self, admission_config: Any) -> None:
        self.config = admission_config

    def select(
        self,
        context: MasterContext,
        features: FeatureBatch,
        scores: np.ndarray,
        *,
        prerank_cap: int,
        product_cap: int,
    ) -> RankingSelection:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (len(features.candidate_sirets),):
            raise FoldFusedInvariantError("one site score is required per Master row")
        if not np.isfinite(scores).all():
            raise FoldFusedInvariantError("site scores must be finite")
        if not len(scores):
            return RankingSelection((), (), ())
        if prerank_cap != int(self.config.prerank_cap) or product_cap != int(
            self.config.final_cap
        ):
            raise FoldFusedInvariantError("selection caps differ from frozen v12 config")
        feature_index = {name: index for index, name in enumerate(features.feature_names)}
        required = {"joint_name_geo", "parent_score_probability"}
        if not required.issubset(feature_index):
            raise FoldFusedInvariantError("v12 selector lacks adaptive allocation features")

        # ``sorted(key=(-score, siret))`` from the historical adapter.  SIRETs
        # are 14 decimal digits and therefore fit safely in signed int64.
        siret_numeric = np.fromiter(
            (int(value) for value in features.candidate_sirets),
            dtype=np.int64,
            count=len(scores),
        )
        order = np.lexsort((siret_numeric, -scores))
        prerank_array = np.asarray(
            order[: min(prerank_cap, len(order))], dtype=np.int64
        )
        prerank = tuple(int(value) for value in prerank_array)
        if not len(prerank_array):
            return RankingSelection((), (), ())

        # Inside PreRank500, ranked_rank is exactly 1..K in ``order``.  Keep
        # float64 operations and Python-float configuration values, matching
        # the pandas oracle's arithmetic.
        size = len(prerank_array)
        site_percentile = 1.0 - (
            np.arange(size, dtype=np.float64) / float(max(1, size))
        )
        parent_probability = features.values[
            prerank_array, feature_index["parent_score_probability"]
        ]
        # pandas keeps this scalar multiplication in the Series' float32
        # dtype, then promotes when adding the float64 site percentile.
        # Preserving that order is necessary for exact marginal tie behavior.
        parent_contribution = (
            float(self.config.selection["parent_probability_weight"])
            * parent_probability
        )
        base_utility = (
            site_percentile
            + parent_contribution
        )
        prerank_scores = scores[prerank_array]
        prerank_sirets = siret_numeric[prerank_array]
        joint = np.asarray(
            features.values[
                prerank_array, feature_index["joint_name_geo"]
            ]
            == 1,
            dtype=bool,
        )

        parent_codes_by_name: dict[str, int] = {}
        parent_codes = np.empty(size, dtype=np.int32)
        for position, master_index in enumerate(prerank_array):
            parent = str(features.candidate_parents[int(master_index)])
            code = parent_codes_by_name.setdefault(parent, len(parent_codes_by_name))
            parent_codes[position] = code
        parent_counts = np.zeros(len(parent_codes_by_name), dtype=np.int16)

        chosen: list[int] = []
        reasons: list[str] = []
        protected_positions = np.flatnonzero(joint)
        if len(protected_positions):
            protected_order = np.lexsort(
                (
                    prerank_sirets[protected_positions],
                    -prerank_scores[protected_positions],
                    -base_utility[protected_positions],
                )
            )
            for position in protected_positions[protected_order][
                : int(product_cap)
            ]:
                local = int(position)
                chosen.append(local)
                reasons.append("protected_joint_name_geo")
                parent_counts[parent_codes[local]] += 1

        remaining = np.ones(size, dtype=bool)
        if chosen:
            remaining[np.asarray(chosen, dtype=np.int64)] = False
        repeat_penalty = float(self.config.selection["repeat_parent_penalty"])
        # The oracle calls ``math.log1p(count)`` for every candidate.  A tiny
        # precomputed table preserves those exact scalar values while keeping
        # the greedy loop vectorized.
        count_penalties = np.asarray(
            [repeat_penalty * math.log1p(value) for value in range(product_cap + 1)],
            dtype=np.float64,
        )
        while len(chosen) < product_cap and remaining.any():
            candidates = np.flatnonzero(remaining)
            marginal = base_utility[candidates] - count_penalties[
                parent_counts[parent_codes[candidates]]
            ]
            best_value = marginal.max()
            tied = candidates[marginal == best_value]
            if len(tied) > 1:
                tied_scores = prerank_scores[tied]
                tied = tied[tied_scores == tied_scores.max()]
            if len(tied) > 1:
                tied = tied[
                    prerank_sirets[tied] == prerank_sirets[tied].min()
                ]
            best = int(tied[0])
            chosen.append(best)
            reasons.append("adaptive_marginal")
            remaining[best] = False
            parent_counts[parent_codes[best]] += 1

        product = tuple(int(prerank_array[position]) for position in chosen)
        return RankingSelection(prerank, product, tuple(reasons))


class ParquetOOFOutputStore:
    """Stream the three bounded OOF products without persisting full Masters."""

    _RANKING_SCHEMA = pa.schema(
        [
            ("query_id", pa.string()),
            ("fold", pa.int8()),
            ("candidate_siret", pa.string()),
            ("candidate_parent", pa.string()),
            ("rank", pa.int16()),
            ("site_score", pa.float64()),
            ("selection_reason", pa.string()),
            ("payload_json", pa.string()),
        ]
    )
    _OUTCOME_SCHEMA = pa.schema(
        [
            ("query_id", pa.string()),
            ("fold", pa.int8()),
            ("human_siret", pa.string()),
            ("master_candidate_count", pa.int16()),
            ("prerank_candidate_count", pa.int16()),
            ("product_candidate_count", pa.int16()),
            ("truth_master_rank", pa.int16()),
            ("truth_prerank_rank", pa.int16()),
            ("truth_product_rank", pa.int16()),
            ("master_hit_at_2000", pa.bool_()),
            ("prerank_hit_at_500", pa.bool_()),
            ("exact_hit_at_100", pa.bool_()),
            ("operational_hit_at_100", pa.bool_()),
            ("loss_reason", pa.string()),
            ("audit_metadata_json", pa.string()),
        ]
    )

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "top500": self.output_dir / "oof_prerank500.parquet",
            "top100": self.output_dir / "oof_top100.parquet",
            "outcomes": self.output_dir / "oof_query_outcomes.parquet",
        }
        self._writers = {
            "top500": pq.ParquetWriter(
                self.paths["top500"], self._RANKING_SCHEMA, compression="zstd"
            ),
            "top100": pq.ParquetWriter(
                self.paths["top100"], self._RANKING_SCHEMA, compression="zstd"
            ),
            "outcomes": pq.ParquetWriter(
                self.paths["outcomes"], self._OUTCOME_SCHEMA, compression="zstd"
            ),
        }
        self._rows = {name: 0 for name in self.paths}

    def _ranking_table(
        self,
        context: MasterContext,
        features: FeatureBatch,
        scores: np.ndarray,
        indices: Sequence[int],
        reasons: Sequence[str],
    ) -> pa.Table:
        records: list[dict[str, Any]] = []
        for rank, (index, reason) in enumerate(zip(indices, reasons, strict=True), start=1):
            payload = features.row_payloads[index] if features.row_payloads else {}
            records.append(
                {
                    "query_id": context.query_id,
                    "fold": context.scene_fold,
                    "candidate_siret": features.candidate_sirets[index],
                    "candidate_parent": features.candidate_parents[index],
                    "rank": rank,
                    "site_score": float(scores[index]),
                    "selection_reason": reason,
                    "payload_json": _canonical_json(payload),
                }
            )
        return pa.Table.from_pylist(records, schema=self._RANKING_SCHEMA)

    def append_rankings(
        self,
        context: MasterContext,
        features: FeatureBatch,
        scores: np.ndarray,
        selection: RankingSelection,
    ) -> None:
        top500_reasons = ("site_lambdamart_prerank",) * len(selection.prerank_indices)
        for name, indices, reasons in (
            ("top500", selection.prerank_indices, top500_reasons),
            ("top100", selection.product_indices, selection.product_reasons),
        ):
            table = self._ranking_table(context, features, scores, indices, reasons)
            if table.num_rows:
                self._writers[name].write_table(table)
                self._rows[name] += table.num_rows

    def append_outcome(self, outcome: QueryOutcome) -> None:
        row = {
            **{name: getattr(outcome, name) for name in outcome.__dataclass_fields__ if name != "audit_metadata"},
            "audit_metadata_json": _canonical_json(outcome.audit_metadata),
        }
        table = pa.Table.from_pylist([row], schema=self._OUTCOME_SCHEMA)
        self._writers["outcomes"].write_table(table)
        self._rows["outcomes"] += 1

    def seal(self) -> Mapping[str, ArtifactRecord]:
        for writer in self._writers.values():
            writer.close()
        return {
            name: ArtifactRecord(path, sha256_file(path), self._rows[name])
            for name, path in self.paths.items()
        }


class FoldFusedOOFExecutor:
    """Single-source-pass executor for the three nested site OOF models."""

    def __init__(
        self,
        *,
        source: SceneSource,
        labels: SealedLabelLedger,
        contexts: VectorizedContextAPI,
        sampler: FitSampler,
        train_store: CompactTrainingStore,
        heldout_store: HeldoutMasterStore,
        model_backend: SiteLambdaMARTBackend,
        selector: ProductSelector,
        output_store: OOFOutputSink,
        lineage: RunLineage,
        manifest_path: Path,
        caps: ExecutionCaps = ExecutionCaps(),
        gates: GateThresholds = GateThresholds(),
        parity: ParityEvidence = ParityEvidence(),
    ) -> None:
        self.source = source
        self.labels = labels
        self.contexts = contexts
        self.sampler = sampler
        self.train_store = train_store
        self.heldout_store = heldout_store
        self.model_backend = model_backend
        self.selector = selector
        self.output_store = output_store
        self.lineage = lineage
        self.manifest_path = Path(manifest_path)
        self.caps = caps
        self.gates = gates
        self.parity = parity
        self._feature_names: tuple[str, ...] | None = None

    def _validate_context(
        self,
        scene: SceneRecord,
        outer: int,
        context: MasterContext,
        statistics: FullMasterStatistics,
    ) -> None:
        self._validate_master_context(scene, outer, context)
        if (
            not statistics.complete
            or statistics.query_id != scene.query_id
            or statistics.outer_fold != outer
            or statistics.master_count != len(context.candidate_sirets)
            or statistics.candidate_order_sha256 != context.candidate_order_sha256
            or not statistics.statistics_sha256
        ):
            raise FoldFusedInvariantError("statistics were not sealed on the complete Master")

    def _validate_master_context(
        self, scene: SceneRecord, outer: int, context: MasterContext
    ) -> None:
        if (
            context.query_id != scene.query_id
            or context.scene_fold != scene.fold
            or context.outer_fold != outer
        ):
            raise FoldFusedInvariantError("context identity differs from SceneIR request")
        if not context.candidate_sirets:
            raise FoldFusedInvariantError("Master is unexpectedly empty")
        if len(context.candidate_sirets) > self.caps.master:
            raise FoldFusedInvariantError("Master exceeds 2,000 candidates")

    def _validate_features(
        self,
        context: MasterContext,
        statistics: FullMasterStatistics,
        features: FeatureBatch,
        expected_indices: Sequence[int],
    ) -> None:
        indices = tuple(int(value) for value in expected_indices)
        if (
            features.query_id != context.query_id
            or features.scene_fold != context.scene_fold
            or features.outer_fold != context.outer_fold
            or features.statistics_sha256 != statistics.statistics_sha256
            or features.candidate_indices != indices
        ):
            raise FoldFusedInvariantError("materialised features do not match the sealed context")
        expected_sirets = tuple(context.candidate_sirets[index] for index in indices)
        expected_parents = tuple(context.candidate_parents[index] for index in indices)
        if features.candidate_sirets != expected_sirets or features.candidate_parents != expected_parents:
            raise FoldFusedInvariantError("feature row identities/order differ from Master indices")
        if self._feature_names is None:
            self._feature_names = features.feature_names
        elif features.feature_names != self._feature_names:
            raise FoldFusedInvariantError("feature schema drift between contexts")

    @staticmethod
    def _validate_scene_sequence(
        observed: Sequence[tuple[str, int]], expected_ids: set[str]
    ) -> None:
        query_ids = [query_id for query_id, _fold in observed]
        if len(query_ids) != len(set(query_ids)):
            raise FoldFusedInvariantError("SceneIR contains duplicate query IDs")
        if set(query_ids) != expected_ids:
            raise FoldFusedInvariantError("SceneIR population differs from sealed label ledger")

    def _build_compact_training_matrices(
        self, expected_ids: set[str]
    ) -> tuple[
        Mapping[int, TrainMatrixArtifact],
        HeldoutMasterArtifact,
        Mapping[str, Any],
        tuple[tuple[str, int], ...],
    ]:
        observed: list[tuple[str, int]] = []
        diagnostics: dict[int, dict[str, int]] = {
            fold: {
                "queries_seen": 0,
                "groups_written": 0,
                "rows_written": 0,
                "missing_natural_positive": 0,
                "without_negative": 0,
                "operational_neutrals_excluded": 0,
            }
            for fold in TRAIN_FOLDS
        }
        observed_fit_folds: dict[int, set[int]] = {
            fold: set() for fold in TRAIN_FOLDS
        }
        for scene in self.source.iter_scenes(folds=TRAIN_FOLDS):
            if scene.fold not in TRAIN_FOLDS:
                raise FoldFusedInvariantError("SceneIR attempted to open fold 0/1")
            observed.append((scene.query_id, scene.fold))
            # All label-blind contexts and complete-Master statistics exist
            # before the label-side is consulted for sampling.
            prepared: dict[int, tuple[MasterContext, FullMasterStatistics]] = {}
            for outer in TRAIN_FOLDS:
                context = self.contexts.prepare(scene, outer_fold=outer)
                self._validate_master_context(scene, outer, context)
                if outer == scene.fold:
                    # The held-out Master is label-blind and is persisted
                    # before the label ledger is consulted.  Its expensive
                    # statistics/features are deferred until the models exist.
                    self.heldout_store.append(context)
                    continue
                statistics = self.contexts.compute_full_master_statistics(context)
                self._validate_context(scene, outer, context, statistics)
                prepared[outer] = (context, statistics)
            label = self.labels.get(scene.query_id)
            if label.query_id != scene.query_id or label.fold != scene.fold:
                raise FoldFusedInvariantError("label ledger identity/fold mismatch")
            for outer, (context, statistics) in prepared.items():
                diag = diagnostics[outer]
                diag["queries_seen"] += 1
                plan = self.sampler.choose(
                    context,
                    statistics,
                    label,
                    maximum_rows=self.caps.fit_sample,
                )
                indices = tuple(int(value) for value in plan.candidate_indices)
                if plan.statistics_sha256 != statistics.statistics_sha256:
                    raise FoldFusedInvariantError("fit sampling did not use sealed full-Master statistics")
                if len(indices) > self.caps.fit_sample or len(indices) != len(set(indices)):
                    raise FoldFusedInvariantError("fit sampling is duplicated or exceeds 128 rows")
                if any(index < 0 or index >= len(context.candidate_sirets) for index in indices):
                    raise FoldFusedInvariantError("fit sampling attempted candidate injection")
                if not indices:
                    diag["missing_natural_positive"] += 1
                    continue
                features = self.contexts.materialize(
                    context, statistics, candidate_indices=indices
                )
                self._validate_features(context, statistics, features, indices)
                human = label.human_siret
                neutral = set(label.operational_sirets) - {human}
                keep = [
                    index
                    for index, siret in enumerate(features.candidate_sirets)
                    if siret not in neutral
                ]
                diag["operational_neutrals_excluded"] += len(features.candidate_sirets) - len(keep)
                retained = features.take(keep)
                target = np.asarray(
                    [int(siret == human) for siret in retained.candidate_sirets],
                    dtype=np.int8,
                )
                if int(target.sum()) != 1:
                    diag["missing_natural_positive"] += 1
                    continue
                if len(target) < 2:
                    diag["without_negative"] += 1
                    continue
                batch = CompactTrainBatch(
                    query_id=scene.query_id,
                    scene_fold=scene.fold,
                    outer_fold=outer,
                    candidate_sirets=retained.candidate_sirets,
                    candidate_parents=retained.candidate_parents,
                    feature_names=retained.feature_names,
                    values=retained.values,
                    labels=target,
                    statistics_sha256=statistics.statistics_sha256,
                )
                self.train_store.append(batch)
                observed_fit_folds[outer].add(scene.fold)
                diag["groups_written"] += 1
                diag["rows_written"] += len(target)
        self._validate_scene_sequence(observed, expected_ids)
        artifacts = {
            outer: self.train_store.seal_outer(outer) for outer in TRAIN_FOLDS
        }
        heldout_artifact = self.heldout_store.seal()
        if (
            heldout_artifact.query_count != len(observed)
            or heldout_artifact.row_count < heldout_artifact.query_count
        ):
            raise FoldFusedInvariantError("heldout Master spool cardinality drift")
        for outer, artifact in artifacts.items():
            expected_fit_folds = tuple(fold for fold in TRAIN_FOLDS if fold != outer)
            if (
                artifact.fit_folds != expected_fit_folds
                or observed_fit_folds[outer] != set(expected_fit_folds)
            ):
                raise FoldFusedInvariantError("compact matrix has the wrong fit folds")
        return (
            artifacts,
            heldout_artifact,
            {str(key): value for key, value in diagnostics.items()},
            tuple(observed),
        )

    def _fit_models(
        self, artifacts: Mapping[int, TrainMatrixArtifact]
    ) -> Mapping[int, FittedSiteModel]:
        models = {outer: self.model_backend.fit(artifacts[outer]) for outer in TRAIN_FOLDS}
        if set(models) != set(TRAIN_FOLDS):
            raise FoldFusedInvariantError("exactly three site models are required")
        for outer, model in models.items():
            if model.outer_fold != outer or model.feature_names != artifacts[outer].feature_names:
                raise FoldFusedInvariantError("fitted model identity/schema mismatch")
        if len({model.recipe_sha256 for model in models.values()}) != 1:
            raise FoldFusedInvariantError("all three site models must use one frozen recipe")
        return models

    @staticmethod
    def _rank_of(indices: Sequence[int], truth_index: int | None) -> int | None:
        if truth_index is None:
            return None
        try:
            return list(indices).index(truth_index) + 1
        except ValueError:
            return None

    def _outcome(
        self,
        context: MasterContext,
        selection: RankingSelection,
        label: HumanLabel,
    ) -> QueryOutcome:
        try:
            truth_index = context.candidate_sirets.index(label.human_siret)
        except ValueError:
            truth_index = None
        master_rank = truth_index + 1 if truth_index is not None else None
        prerank_rank = self._rank_of(selection.prerank_indices, truth_index)
        product_rank = self._rank_of(selection.product_indices, truth_index)
        selected_sirets = {
            context.candidate_sirets[index] for index in selection.product_indices
        }
        operational = set(label.operational_sirets) | {label.human_siret}
        truth_parent_present = label.human_siret[:9] in set(context.candidate_parents)
        catalog = label.audit_metadata.get("official_catalog_present")
        if truth_index is None and catalog is False:
            loss = "OFFICIAL_CATALOG_ABSENT"
        elif truth_index is None and catalog is None:
            loss = "OFFICIAL_CATALOG_STATUS_UNKNOWN"
        elif truth_index is None and not truth_parent_present:
            loss = "PARENT_NOT_RETRIEVED"
        elif truth_index is None:
            loss = "SITE_NOT_EXPANDED"
        elif prerank_rank is None:
            loss = "EVICTED_MASTER_TO_500"
        elif product_rank is not None:
            loss = "HIT"
        else:
            loss = "EVICTED_500_TO_100"
        return QueryOutcome(
            query_id=context.query_id,
            fold=context.scene_fold,
            human_siret=label.human_siret,
            master_candidate_count=len(context.candidate_sirets),
            prerank_candidate_count=len(selection.prerank_indices),
            product_candidate_count=len(selection.product_indices),
            truth_master_rank=master_rank,
            truth_prerank_rank=prerank_rank,
            truth_product_rank=product_rank,
            master_hit_at_2000=truth_index is not None,
            prerank_hit_at_500=prerank_rank is not None,
            exact_hit_at_100=product_rank is not None,
            operational_hit_at_100=bool(selected_sirets.intersection(operational)),
            loss_reason=loss,
            audit_metadata=dict(label.audit_metadata),
        )

    def _score_heldout(
        self,
        models: Mapping[int, FittedSiteModel],
        expected_sequence: tuple[tuple[str, int], ...],
        heldout_artifact: HeldoutMasterArtifact,
    ) -> tuple[tuple[QueryOutcome, ...], Mapping[str, ArtifactRecord]]:
        observed: list[tuple[str, int]] = []
        outcomes: list[QueryOutcome] = []
        for context in self.heldout_store.iter_contexts(heldout_artifact):
            if (
                context.scene_fold not in TRAIN_FOLDS
                or context.outer_fold != context.scene_fold
            ):
                raise FoldFusedInvariantError("heldout spool attempted to open fold 0/1")
            scene = SceneRecord(query_id=context.query_id, fold=context.scene_fold)
            observed.append((context.query_id, context.scene_fold))
            # Only the heldout context is materialised at full Master width;
            # its source is the sealed spool, never SceneIR/sidecars.
            statistics = self.contexts.compute_full_master_statistics(context)
            self._validate_context(
                scene, context.scene_fold, context, statistics
            )
            indices = tuple(range(len(context.candidate_sirets)))
            features = self.contexts.materialize(
                context, statistics, candidate_indices=None
            )
            self._validate_features(context, statistics, features, indices)
            model = models[context.scene_fold]
            scores = self.model_backend.score(model, features)
            scores = np.asarray(scores, dtype=np.float64)
            if scores.shape != (len(indices),) or not np.isfinite(scores).all():
                raise FoldFusedInvariantError("site model returned invalid heldout scores")
            selection = self.selector.select(
                context,
                features,
                scores,
                prerank_cap=self.caps.prerank,
                product_cap=self.caps.product,
            )
            self._validate_selection(context, selection)
            self.output_store.append_rankings(context, features, scores, selection)
            # Evaluation-only label access happens after generation, feature
            # computation, scoring and selection have all completed.
            label = self.labels.get(scene.query_id)
            if label.fold != context.scene_fold:
                raise FoldFusedInvariantError("heldout label fold mismatch")
            outcome = self._outcome(context, selection, label)
            outcomes.append(outcome)
            self.output_store.append_outcome(outcome)
        if tuple(observed) != expected_sequence:
            raise FoldFusedInvariantError("SceneIR changed between training and heldout passes")
        return tuple(outcomes), self.output_store.seal()

    def _validate_selection(
        self, context: MasterContext, selection: RankingSelection
    ) -> None:
        master_count = len(context.candidate_sirets)
        prerank = selection.prerank_indices
        product = selection.product_indices
        if len(prerank) > self.caps.prerank or len(product) > self.caps.product:
            raise FoldFusedInvariantError("selector exceeded PreRank500/top100")
        if len(prerank) != len(set(prerank)) or len(product) != len(set(product)):
            raise FoldFusedInvariantError("selector emitted duplicate candidates")
        if any(index < 0 or index >= master_count for index in (*prerank, *product)):
            raise FoldFusedInvariantError("selector emitted a candidate outside Master")
        if not set(product).issubset(prerank):
            raise FoldFusedInvariantError("top100 must be selected inside PreRank500")
        if len(selection.product_reasons) != len(product):
            raise FoldFusedInvariantError("top100 reasons are incomplete")

    @staticmethod
    def _metric(successes: int, total: int) -> Mapping[str, Any]:
        if total <= 0:
            return {"successes": 0, "total": 0, "rate": None, "wilson_95": None}
        p = successes / total
        z = NormalDist().inv_cdf(0.975)
        denominator = 1.0 + z * z / total
        center = (p + z * z / (2.0 * total)) / denominator
        spread = z * math.sqrt(
            p * (1.0 - p) / total + z * z / (4.0 * total * total)
        ) / denominator
        return {
            "successes": successes,
            "total": total,
            "rate": p,
            "wilson_95": [max(0.0, center - spread), min(1.0, center + spread)],
        }

    def _gate_manifest(self, outcomes: Sequence[QueryOutcome]) -> Mapping[str, Any]:
        total = len(outcomes)
        master = self._metric(sum(value.master_hit_at_2000 for value in outcomes), total)
        prerank = self._metric(sum(value.prerank_hit_at_500 for value in outcomes), total)
        top100 = self._metric(sum(value.exact_hit_at_100 for value in outcomes), total)
        per_fold = {
            str(fold): self._metric(
                sum(value.exact_hit_at_100 for value in outcomes if value.fold == fold),
                sum(value.fold == fold for value in outcomes),
            )
            for fold in TRAIN_FOLDS
        }
        # Master@2000 is a diagnostic ceiling/headroom target in the frozen
        # v12 contract.  PreRank@500 is the first decision-blocking generation
        # gate; a low Master value must never be relabelled PIVOT_GENERATION by
        # this physical executor.
        master_headroom_target_met = bool(
            master["rate"] is not None
            and master["rate"] >= self.gates.master_recall_at_2000
        )
        recall_pass = bool(
            prerank["rate"] is not None
            and prerank["rate"] >= self.gates.prerank_recall_at_500
            and top100["rate"] is not None
            and top100["rate"] >= self.gates.top100_recall
            and top100["wilson_95"][0] >= self.gates.top100_wilson95_lower
            and all(
                value["rate"] is not None
                and value["rate"] >= self.gates.minimum_each_outer_fold_top100
                for value in per_fold.values()
            )
        )
        parity = self.parity.as_manifest()
        recall_and_parity_pass = bool(recall_pass and parity["status"] == "PASS")
        if parity["status"] != "PASS":
            decision = "NO_READY_DECISION"
        elif prerank["rate"] < self.gates.prerank_recall_at_500:
            decision = "PIVOT_GENERATION"
        elif not recall_pass:
            decision = "PIVOT_ADMISSION"
        else:
            decision = "PENDING_RUNTIME_PROFILE"
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "denominator": "ALL_USABLE_HUMAN_SIRET_LABELS_FOLDS_2_3_4",
            "master_recall_at_2000": master,
            "master_headroom_target_met": master_headroom_target_met,
            "master_decision_blocking": False,
            "prerank_recall_at_500": prerank,
            "exact_recall_at_100": top100,
            "exact_recall_at_100_by_outer_fold": per_fold,
            "thresholds": self.gates.__dict__,
            "caps_observed": {
                "maximum_master": max((value.master_candidate_count for value in outcomes), default=0),
                "maximum_prerank": max((value.prerank_candidate_count for value in outcomes), default=0),
                "maximum_product": max((value.product_candidate_count for value in outcomes), default=0),
            },
            "recall_gates_pass": recall_pass,
            "parity_gate": parity["status"],
            "latency_gate": "NOT_EVALUATED",
            "recall_and_parity_pass": recall_and_parity_pass,
            # This executor does not own the warm runtime profiler.  It can
            # never issue a full v12 gate claim by itself.
            "gate_eligible": False,
            "decision": decision,
        }

    def run(self) -> FoldFusedRunResult:
        if self.parity.as_manifest()["status"] != "NOT_RUN":
            raise FoldFusedInvariantError(
                "OOF executor refuses POSTFIT parity; certify the immutable run separately"
            )
        expected = tuple(str(value) for value in self.labels.expected_query_ids(folds=TRAIN_FOLDS))
        if not expected or len(expected) != len(set(expected)):
            raise FoldFusedInvariantError("sealed ledger population must be non-empty and unique")
        (
            artifacts,
            heldout_artifact,
            sampling_diagnostics,
            sequence,
        ) = self._build_compact_training_matrices(set(expected))
        models = self._fit_models(artifacts)
        outcomes, outputs = self._score_heldout(
            models, sequence, heldout_artifact
        )
        if {value.query_id for value in outcomes} != set(expected):
            raise FoldFusedInvariantError("outcomes do not exactly match the all-human ledger")
        parity_manifest = self.parity.as_manifest()
        gate_manifest = self._gate_manifest(outcomes)
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "engine_id": ENGINE_ID,
            "run_kind": "NESTED_OUTER_OOF_FOLD_FUSED",
            "status": "EXECUTED_NO_READY_CLAIM",
            "opened_folds": list(TRAIN_FOLDS),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "logical_recipe_changes": False,
            "physical_plan": {
                "scene_ir_passes_after_parent_fit": 1,
                "training_contexts_per_query": 2,
                "heldout_full_master_contexts_per_query": 1,
                "heldout_replay_source": "SEALED_EXTERNAL_PARQUET_SPOOL",
                "source_or_sidecar_reopened_after_fit": False,
                "full_master_statistics_before_fit_sampling": True,
                "fit_sampling_maximum": self.caps.fit_sample,
                "validation_candidate_sampling": False,
                "wide_master_materialized": False,
            },
            "caps": self.caps.__dict__,
            "lineage": self.lineage.__dict__,
            "sampling_diagnostics_by_outer": sampling_diagnostics,
            "train_matrices": {
                str(outer): {
                    "path": str(artifact.path),
                    "sha256": artifact.sha256,
                    "rows": artifact.row_count,
                    "queries": artifact.query_count,
                    "fit_folds": list(artifact.fit_folds),
                    "feature_schema_sha256": artifact.feature_schema_sha256,
                }
                for outer, artifact in artifacts.items()
            },
            "heldout_master_spool": {
                "path": str(heldout_artifact.path),
                "manifest_path": str(heldout_artifact.manifest_path),
                "sha256": heldout_artifact.sha256,
                "manifest_sha256": heldout_artifact.manifest_sha256,
                "rows": heldout_artifact.row_count,
                "queries": heldout_artifact.query_count,
                "actual_size_bytes": heldout_artifact.size_bytes,
                # The full OOF run is the national population, so its exact
                # measured size is also the final (non-sampled) projection.
                "projected_national_size_bytes": heldout_artifact.size_bytes,
                "projection_basis_queries": heldout_artifact.query_count,
                "maximum_bytes": heldout_artifact.maximum_bytes,
                "budget_pass": (
                    heldout_artifact.size_bytes <= heldout_artifact.maximum_bytes
                ),
                "schema_sha256": heldout_artifact.schema_sha256,
                "query_sequence_sha256": heldout_artifact.query_sequence_sha256,
            },
            "site_models": {
                str(outer): {
                    "path": str(model.path),
                    "sha256": model.sha256,
                    "recipe_sha256": model.recipe_sha256,
                    "feature_schema_sha256": _schema_hash(model.feature_names),
                }
                for outer, model in models.items()
            },
            "outputs": {
                name: {"path": str(value.path), "sha256": value.sha256, "rows": value.rows}
                for name, value in outputs.items()
            },
            "parity": parity_manifest,
            "gates": gate_manifest,
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.manifest_path)
        return FoldFusedRunResult(
            manifest_path=self.manifest_path,
            train_matrices=artifacts,
            heldout_master=heldout_artifact,
            models=models,
            outputs=outputs,
            outcomes=outcomes,
            manifest=manifest,
        )


__all__ = [
    "ArtifactRecord",
    "CompactTrainBatch",
    "CompactTrainingStore",
    "ENGINE_ID",
    "ExecutionCaps",
    "ExistingV12SelectionAdapter",
    "FeatureBatch",
    "FitSampler",
    "FittedSiteModel",
    "FoldFusedInvariantError",
    "FoldFusedOOFExecutor",
    "FoldFusedRunResult",
    "FullMasterStatistics",
    "GateThresholds",
    "HeldoutMasterArtifact",
    "HeldoutMasterStore",
    "HumanLabel",
    "MANIFEST_SCHEMA_VERSION",
    "MasterContext",
    "OOFOutputSink",
    "PARITY_SCHEMA_VERSION",
    "ParquetCompactTrainingStore",
    "ParquetOOFOutputStore",
    "ParityEvidence",
    "ProductSelector",
    "QueryOutcome",
    "RankingSelection",
    "REQUIRED_PARITY_CHECKS",
    "RunLineage",
    "SamplingPlan",
    "SceneRecord",
    "SceneSource",
    "SealedLabelLedger",
    "SiteLambdaMARTBackend",
    "TRAIN_FOLDS",
    "TrainMatrixArtifact",
    "VectorizedContextAPI",
    "XGBoostSiteLambdaMARTBackend",
    "sha256_file",
]
