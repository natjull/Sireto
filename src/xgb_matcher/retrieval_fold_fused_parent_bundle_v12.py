"""Streaming, fold-fused trainer for the six retrieval-v12 parent models.

The historical v12 workflow materialises every bootstrap parent before
``run_fit_parent_oof`` keeps at most 128 rows per query.  On the all-human OOF
population that intermediate is needlessly large.  This module preserves the
scientific recipe while changing only that physical plan:

1. stream a label-blind :class:`~retrieval_v12_scene_ir.SceneIRCache`;
2. compile each complete natural scene once;
3. assign the frozen deterministic bootstrap score and the exact legacy
   parent ordering/cap without consulting a human label;
4. only then look up the strict label ledger and reproduce
   ``_load_parent_fit_sample``'s <=128 sampling rule;
5. fit and seal the same six LambdaMART models and nested OOF routes.

Only the compact sampled rows are spooled, partitioned by fold, while models
are fitted.  The spool is removed before publication; no full parent-candidate
Parquet is created.  Fold 0 and fold 1 are rejected at every boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_fold_fused_v12 import (
    ALLOWED_FOLDS_V12,
    FoldFusedSceneV12,
)
from .retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    fit_parent_ranker_from_cascade_rows,
    normalize_siret,
    parent_feature_names,
)
from .retrieval_top100_cascade_v12 import (
    CASCADE_PARENT_FEATURES,
    RetrievalTop100CascadeV12Config,
    SidecarReaderV12,
)
from .retrieval_v12_scene_ir import SceneIRCache
from .retrieval_v12_parent_arrow import (
    compile_parent_arrow_scene_v12,
    parent_signal_complete_projection_v12,
)
from .retrieval_v12_parent_signal_capability import (
    ParentSignalCapabilityV12,
    load_parent_signal_capability_v12,
)


PARENT_OOF_BUNDLE_SCHEMA_V12 = "sireto-parent-oof-model-bundle-v12-1"
PARENT_BUNDLE_RUN_KIND_V12 = "PARENT_NESTED_OOF_MODEL_BUNDLE"
PARENT_FEATURE_ORIGIN_V12 = "CASCADE_PRE_EXPANSION"
PARENT_PHYSICAL_PLAN_V12 = "SCENE_IR_FUSED_BOOTSTRAP_SAMPLE_128_NO_FULL_PARENT_PARQUET"
PARENT_MODEL_KIND_V12 = "XGBOOST_BOOSTER"
ALLOWED_FOLDS = frozenset({2, 3, 4})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Insertion order intentionally matches the historical runner.  Apart from
# making manifests familiar, it also keeps deterministic model-fit ordering.
PARENT_FIT_FOLDS_V12: Mapping[str, tuple[int, ...]] = {
    "fit_2": (2,),
    "fit_3": (3,),
    "fit_4": (4,),
    "fit_23": (2, 3),
    "fit_24": (2, 4),
    "fit_34": (3, 4),
}

OUTER_PARENT_MAPS_V12: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    "2": {
        "2": {"model": "fit_34", "fit_folds": [3, 4], "role": "OUTER_HELDOUT"},
        "3": {"model": "fit_4", "fit_folds": [4], "role": "INNER_TRAIN"},
        "4": {"model": "fit_3", "fit_folds": [3], "role": "INNER_TRAIN"},
    },
    "3": {
        "3": {"model": "fit_24", "fit_folds": [2, 4], "role": "OUTER_HELDOUT"},
        "2": {"model": "fit_4", "fit_folds": [4], "role": "INNER_TRAIN"},
        "4": {"model": "fit_2", "fit_folds": [2], "role": "INNER_TRAIN"},
    },
    "4": {
        "4": {"model": "fit_23", "fit_folds": [2, 3], "role": "OUTER_HELDOUT"},
        "2": {"model": "fit_3", "fit_folds": [3], "role": "INNER_TRAIN"},
        "3": {"model": "fit_2", "fit_folds": [2], "role": "INNER_TRAIN"},
    },
}

PARENT_SAMPLE_COLUMNS_V12 = (
    "query_id",
    "candidate_parent",
    "parent_feature_origin",
    "sealed_parent_rank",
    "sealed_parent_score",
    *CASCADE_PARENT_FEATURES,
    "fold",
    "human_siret",
)

_PARENT_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("candidate_parent", pa.string(), nullable=False),
        pa.field("parent_feature_origin", pa.string(), nullable=False),
        pa.field("sealed_parent_rank", pa.float64(), nullable=False),
        pa.field("sealed_parent_score", pa.float64(), nullable=False),
        *[
            pa.field(name, pa.float32(), nullable=False)
            for name in CASCADE_PARENT_FEATURES
        ],
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("human_siret", pa.string(), nullable=False),
    ]
)


class SceneIRCacheLikeV12(Protocol):
    """The narrow, production-compatible SceneIR surface used by the trainer."""

    manifest: Mapping[str, Any]

    def iter_scenes(
        self,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 262_144,
    ) -> Any: ...


class ParentModelV12(Protocol):
    def save_model(self, path: Path | str) -> None: ...


ParentModelFitterV12 = Callable[
    [pd.DataFrame, AdmissionConfigV12],
    tuple[ParentModelV12, Mapping[str, int]],
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"fold-fused parent lineage lacks valid {field}")
    return value


def _canonical_json_line(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class BootstrapParentCandidateV12:
    """One naturally generated parent after the label-blind bootstrap cap."""

    query_id: str
    fold: int
    siren: str
    rank: int
    score: float
    feature_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.query_id or self.fold not in ALLOWED_FOLDS:
            raise ValueError("bootstrap parent escaped the OOF folds 2/3/4")
        if len(self.siren) != 9 or not self.siren.isdigit():
            raise ValueError("bootstrap parent requires a normalized nine-digit SIREN")
        if self.rank < 1 or not math.isfinite(self.score):
            raise ValueError("bootstrap parent rank/score is invalid")
        if len(self.feature_values) != len(CASCADE_PARENT_FEATURES):
            raise ValueError("bootstrap parent feature schema/order drift")
        if not all(math.isfinite(float(value)) for value in self.feature_values):
            raise ValueError("bootstrap parent features must be finite")


@dataclass(frozen=True)
class ParentFitSampleDiagnosticsV12:
    rows_before_sampling: int
    rows_after_sampling: int
    maximum_candidates_per_query: int
    natural_parent_positive_queries: int
    queries: int
    queries_by_fold: Mapping[str, int]
    natural_parent_rows_before_master_cap: int
    natural_positive_queries_before_master_cap: int
    ledger_queries_without_scene: int
    maximum_natural_parents_before_master_cap: int
    maximum_bounded_parents: int
    fit_sample_semantic_sha256: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "rows_before_sampling": self.rows_before_sampling,
            "rows_after_sampling": self.rows_after_sampling,
            "maximum_candidates_per_query": self.maximum_candidates_per_query,
            "natural_parent_positive_queries": self.natural_parent_positive_queries,
            "queries": self.queries,
            "queries_by_fold": dict(self.queries_by_fold),
            "natural_parent_rows_before_master_cap": (
                self.natural_parent_rows_before_master_cap
            ),
            "natural_positive_queries_before_master_cap": (
                self.natural_positive_queries_before_master_cap
            ),
            "ledger_queries_without_scene": self.ledger_queries_without_scene,
            "maximum_natural_parents_before_master_cap": (
                self.maximum_natural_parents_before_master_cap
            ),
            "maximum_bounded_parents": self.maximum_bounded_parents,
            "fit_sample_semantic_sha256": self.fit_sample_semantic_sha256,
            "label_attached_after_complete_natural_parent_pool": True,
            "full_parent_candidate_parquet_materialized": False,
        }


@dataclass(frozen=True)
class ParentBundleLineageV12:
    source_candidate_sha256: str
    source_manifest_sha256: str
    sidecar_manifest_sha256: str
    source_labels_sha256: str
    source_labels_manifest_sha256: str
    config_sha256: str
    scene_ir_manifest_sha256: str
    scene_ir_build_id: str

    def __post_init__(self) -> None:
        for field in (
            "source_candidate_sha256",
            "source_manifest_sha256",
            "sidecar_manifest_sha256",
            "source_labels_sha256",
            "source_labels_manifest_sha256",
            "config_sha256",
            "scene_ir_manifest_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        _require_sha256(self.scene_ir_build_id, field="scene_ir_build_id")

    @property
    def source_retrieval_lineage(self) -> dict[str, str]:
        return {
            "source_candidate_sha256": self.source_candidate_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "sidecar_manifest_sha256": self.sidecar_manifest_sha256,
        }


@dataclass(frozen=True)
class ParentBundleBuildV12:
    output_dir: Path
    manifest_path: Path
    model_paths: Mapping[str, Path]
    diagnostics: ParentFitSampleDiagnosticsV12


def validate_strict_parent_labels_v12(labels: pd.DataFrame) -> pd.DataFrame:
    """Validate the already sealed ledger projection used by parent fitting.

    Production callers must first use the canonical strict ledger loader.  The
    reduced frame is validated again here so a programmatic caller cannot
    accidentally bypass the fold and SIREN-disjoint constraints.
    """

    required = {"query_id", "fold", "human_siret"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"parent label ledger lacks columns: {missing}")
    ledger = labels[["query_id", "fold", "human_siret"]].copy()
    ledger["query_id"] = ledger["query_id"].astype(str)
    ledger["fold"] = ledger["fold"].astype(int)
    if ledger["query_id"].eq("").any() or ledger["query_id"].duplicated().any():
        raise ValueError("parent label ledger must be non-empty and unique by query")
    if set(ledger["fold"].unique()) != ALLOWED_FOLDS:
        raise ValueError("parent label ledger must contain exactly folds 2/3/4")
    normalized: list[str] = []
    for value in ledger["human_siret"]:
        exact = normalize_siret(value)
        if exact != str(value):
            raise ValueError("parent label ledger SIRETs must already be normalized")
        normalized.append(exact)
    ledger["human_siret"] = normalized
    truth_parent = ledger["human_siret"].str[:9]
    if pd.DataFrame({"parent": truth_parent, "fold": ledger["fold"]}).groupby(
        "parent"
    )["fold"].nunique().gt(1).any():
        raise ValueError("parent label ledger SIRENs overlap across OOF folds")
    return ledger.set_index("query_id", drop=False)


def deterministic_bootstrap_parent_pool_v12(
    scene: FoldFusedSceneV12,
    *,
    config: RetrievalTop100CascadeV12Config,
) -> tuple[BootstrapParentCandidateV12, ...]:
    """Reproduce pass-A's score/order/master cap without any label access."""

    if scene.query_fold not in ALLOWED_FOLDS or not scene.parents:
        raise ValueError("parent bootstrap requires a non-empty fold 2/3/4 scene")
    scored: list[tuple[Any, float]] = []
    for parent in scene.parents:
        # Positional form of `_default_parent_score`, in the same operation
        # order and Python-float precision.  Avoiding ~2k temporary dicts per
        # query is a physical optimisation only; the differential suite seals
        # exact scores/ranks and final XGBoost bytes.
        values = parent.feature_values
        score = 0.0
        score += 8.0 * values[0]
        score += 6.0 * values[1]
        score += 4.0 * values[2]
        score += 2.5 * values[3]
        score += 2.0 * values[4]
        score += 1.25 * values[5]
        score += 2.0 * values[6]
        score += min(3.0, values[7])
        score += min(2.0, 0.35 * values[8])
        score += min(2.0, 0.20 * values[13])
        score += min(1.5, 0.25 * values[14])
        score += 100.0 * values[15]
        score += min(1.0, max(0.0, values[17]))
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("deterministic bootstrap produced a non-finite score")
        scored.append((parent, score))
    ordered = sorted(
        scored,
        key=lambda item: (
            -item[1],
            item[0].minimum_input_rank,
            item[0].siren,
        ),
    )[: config.master_reference_cap]
    return tuple(
        BootstrapParentCandidateV12(
            query_id=scene.query_id,
            fold=scene.query_fold,
            siren=parent.siren,
            rank=rank,
            score=float(score),
            feature_values=tuple(float(value) for value in parent.feature_values),
        )
        for rank, (parent, score) in enumerate(ordered, start=1)
    )


def sample_bootstrap_parent_pool_v12(
    pool: Sequence[BootstrapParentCandidateV12],
    *,
    human_siret: str,
    maximum_candidates_per_query: int,
) -> pd.DataFrame:
    """Reproduce ``_load_parent_fit_sample`` exactly for one complete pool.

    The naturally present positive may replace the final head row.  A missing
    positive is never inserted.  This function receives only the already
    scored/bounded pool, making label use structurally incapable of changing
    generation, scores, ranks or the 2,000 cap.
    """

    if not pool:
        raise ValueError("parent sampling requires a complete non-empty pool")
    cap = int(maximum_candidates_per_query)
    if cap < 1:
        raise ValueError("parent fit sample cap must be positive")
    query_ids = {value.query_id for value in pool}
    folds = {value.fold for value in pool}
    if len(query_ids) != 1 or len(folds) != 1:
        raise ValueError("parent sampling pool mixes scenes")
    exact = normalize_siret(human_siret)
    truth_parent = exact[:9]
    ordered = sorted(pool, key=lambda value: value.rank)
    selected = list(ordered[:cap])
    positive = next((value for value in ordered if value.siren == truth_parent), None)
    if positive is not None and all(value.siren != truth_parent for value in selected):
        selected = [*selected[: max(0, cap - 1)], positive]

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in selected:
        if value.siren in seen:
            continue
        seen.add(value.siren)
        record: dict[str, Any] = {
            "query_id": value.query_id,
            "candidate_parent": value.siren,
            "parent_feature_origin": PARENT_FEATURE_ORIGIN_V12,
            "sealed_parent_rank": float(value.rank),
            "sealed_parent_score": float(value.score),
        }
        record.update(
            {
                name: float(feature)
                for name, feature in zip(
                    CASCADE_PARENT_FEATURES,
                    value.feature_values,
                    strict=True,
                )
            }
        )
        record["fold"] = int(value.fold)
        record["human_siret"] = exact
        records.append(record)
    output = pd.DataFrame.from_records(records, columns=PARENT_SAMPLE_COLUMNS_V12)
    if output.empty or len(output) > cap:
        raise AssertionError("parent fit sampling violated its per-query cap")
    return output


class _CompactParentSampleSpoolV12:
    """Temporary <=128/query Parquet spool; never stores the full parent pool."""

    def __init__(self, root: Path, *, flush_rows: int = 65_536) -> None:
        if flush_rows < 1:
            raise ValueError("parent sample spool flush_rows must be positive")
        self.paths = {
            fold: root / f".parent-fit-fold{fold}.parquet"
            for fold in sorted(ALLOWED_FOLDS)
        }
        self._writers: dict[int, pq.ParquetWriter] = {}
        self._buffers: dict[int, list[pd.DataFrame]] = {
            fold: [] for fold in ALLOWED_FOLDS
        }
        self._buffered_rows = {fold: 0 for fold in ALLOWED_FOLDS}
        self._rows = {fold: 0 for fold in ALLOWED_FOLDS}
        self._flush_rows = int(flush_rows)

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError("parent sample spool refuses an empty query frame")
        folds = set(frame["fold"].astype(int).unique())
        if len(folds) != 1 or not folds.issubset(ALLOWED_FOLDS):
            raise ValueError("parent sample spool frame mixes or opens folds")
        fold = folds.pop()
        self._buffers[fold].append(frame)
        self._buffered_rows[fold] += len(frame)
        self._rows[fold] += len(frame)
        if self._buffered_rows[fold] >= self._flush_rows:
            self._flush(fold)

    def _flush(self, fold: int) -> None:
        pieces = self._buffers[fold]
        if not pieces:
            return
        frame = pd.concat(pieces, ignore_index=True)
        frame = frame.loc[:, PARENT_SAMPLE_COLUMNS_V12].copy()
        for name in CASCADE_PARENT_FEATURES:
            frame[name] = frame[name].astype(np.float32)
        frame["fold"] = frame["fold"].astype(np.int8)
        table = pa.Table.from_pandas(
            frame,
            schema=_PARENT_SAMPLE_SCHEMA,
            preserve_index=False,
            safe=True,
        )
        writer = self._writers.get(fold)
        if writer is None:
            writer = pq.ParquetWriter(
                self.paths[fold],
                _PARENT_SAMPLE_SCHEMA,
                compression="zstd",
                use_dictionary=(
                    "query_id",
                    "candidate_parent",
                    "parent_feature_origin",
                    "human_siret",
                ),
            )
            self._writers[fold] = writer
        writer.write_table(table, row_group_size=self._flush_rows)
        self._buffers[fold] = []
        self._buffered_rows[fold] = 0

    def close(self) -> None:
        for fold in sorted(ALLOWED_FOLDS):
            self._flush(fold)
        for writer in self._writers.values():
            writer.close()
        missing = [fold for fold, path in self.paths.items() if not path.is_file()]
        if missing:
            raise ValueError(f"parent sample spool lacks folds: {missing}")
        for fold, path in self.paths.items():
            if int(pq.ParquetFile(path).metadata.num_rows) != self._rows[fold]:
                raise ValueError("parent sample spool row-count drift")

    def read_folds(self, folds: Sequence[int]) -> pd.DataFrame:
        requested = tuple(int(value) for value in folds)
        if not requested or not set(requested).issubset(ALLOWED_FOLDS):
            raise ValueError("parent model requested a forbidden fit fold")
        tables = [pq.read_table(self.paths[fold]) for fold in requested]
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        frame = table.to_pandas()
        if set(frame["fold"].astype(int).unique()) != set(requested):
            raise ValueError("parent sample spool fit-fold projection drift")
        return frame.loc[:, PARENT_SAMPLE_COLUMNS_V12]

    def remove(self) -> None:
        for path in self.paths.values():
            if path.exists():
                path.unlink()


def stream_parent_fit_sample_v12(
    scene_cache: SceneIRCacheLikeV12,
    *,
    sidecars: SidecarReaderV12,
    cascade_config: RetrievalTop100CascadeV12Config,
    admission_config: AdmissionConfigV12,
    labels: pd.DataFrame,
    consume: Callable[[pd.DataFrame], None],
    batch_size: int = 262_144,
    signal_capability: ParentSignalCapabilityV12 | None = None,
) -> ParentFitSampleDiagnosticsV12:
    """Compile once and emit only the exact legacy <=128/query fit rows."""

    if batch_size < 1:
        raise ValueError("SceneIR training batch_size must be positive")
    if set(ALLOWED_FOLDS_V12) != ALLOWED_FOLDS:
        raise AssertionError("fold-fused parent trainer/cascade fold contract drift")
    if tuple(parent_feature_names(admission_config)) != tuple(
        CASCADE_PARENT_FEATURES
    ):
        raise ValueError("parent model feature schema/order drift")
    cap = int(
        admission_config.fit_candidate_sampling["maximum_candidates_per_query"]
    )
    ledger = validate_strict_parent_labels_v12(labels)
    seen_queries: set[str] = set()
    seen_folds: set[int] = set()
    queries_by_fold = {fold: 0 for fold in sorted(ALLOWED_FOLDS)}
    rows_before = rows_after = natural_positives = 0
    full_rows = full_positives = 0
    max_full = max_bounded = 0
    semantic_digest = hashlib.sha256()

    # There is deliberately no label lookup before ``scene`` and ``pool`` are
    # fully computed.  Keeping this order explicit is the leakage boundary.
    projection: tuple[str, ...] | None = None
    if signal_capability is not None:
        source_path = getattr(scene_cache, "source", None)
        if source_path is None:
            raise ValueError(
                "signal-complete parent projection requires a real SceneIR source"
            )
        validated_capability = load_parent_signal_capability_v12(
            signal_capability.manifest_path, scene_cache  # type: ignore[arg-type]
        )
        if validated_capability != signal_capability:
            raise ValueError("parent signal capability object/manifest mismatch")
        observed_projection = parent_signal_complete_projection_v12(
            pq.read_schema(Path(source_path)).names
        )
        manifest = getattr(scene_cache, "manifest", {})
        if (
            signal_capability.source_candidate_sha256
            != manifest.get("source_candidate_sha256")
            or signal_capability.source_manifest_sha256
            != manifest.get("source_manifest_sha256")
            or signal_capability.candidate_rows
            != int(manifest.get("candidate_rows", signal_capability.candidate_rows))
            or signal_capability.projection != observed_projection
        ):
            raise ValueError("parent signal capability/SceneIR lineage mismatch")
        projection = signal_capability.projection
    for raw_scene in scene_cache.iter_scenes(
        columns=projection, batch_size=batch_size
    ):
        query_id = str(raw_scene.query_id)
        if query_id in seen_queries:
            raise ValueError("SceneIR parent trainer observed a duplicate query")
        scene = compile_parent_arrow_scene_v12(
            raw_scene.rows,
            query_id=query_id,
            sidecars=sidecars,
            config=cascade_config,
            signals_complete=signal_capability is not None,
        )
        full_parent_sirens = frozenset(parent.siren for parent in scene.parents)
        pool = deterministic_bootstrap_parent_pool_v12(scene, config=cascade_config)

        # Strict label attachment starts here, after the complete natural pool
        # and its deterministic bootstrap cap/ranks are immutable.
        if query_id not in ledger.index:
            raise ValueError("SceneIR parent pool contains a query outside label ledger")
        label = ledger.loc[query_id]
        if isinstance(label, pd.DataFrame):
            raise ValueError("parent label ledger is not unique by query")
        fold = int(label["fold"])
        if fold != scene.query_fold or fold not in ALLOWED_FOLDS:
            raise ValueError("SceneIR parent pool fold/label mismatch")
        human_siret = str(label["human_siret"])
        truth_parent = human_siret[:9]
        selected = sample_bootstrap_parent_pool_v12(
            pool,
            human_siret=human_siret,
            maximum_candidates_per_query=cap,
        )
        consume(selected)

        seen_queries.add(query_id)
        seen_folds.add(fold)
        queries_by_fold[fold] += 1
        rows_before += len(pool)
        rows_after += len(selected)
        natural_positives += int(any(value.siren == truth_parent for value in pool))
        full_rows += len(scene.parents)
        full_positives += int(truth_parent in full_parent_sirens)
        max_full = max(max_full, len(scene.parents))
        max_bounded = max(max_bounded, len(pool))
        for row in selected.to_dict("records"):
            semantic_digest.update(
                _canonical_json_line(
                    [
                        row["query_id"],
                        int(row["fold"]),
                        row["human_siret"],
                        row["candidate_parent"],
                        float(row["sealed_parent_rank"]),
                        float(row["sealed_parent_score"]),
                        [float(row[name]) for name in CASCADE_PARENT_FEATURES],
                    ]
                )
            )

    if not seen_queries or seen_folds != ALLOWED_FOLDS:
        raise ValueError("SceneIR parent trainer must cover folds 2/3/4")
    manifest = getattr(scene_cache, "manifest", {})
    declared_queries = manifest.get("query_count")
    if declared_queries is not None and int(declared_queries) != len(seen_queries):
        raise ValueError("SceneIR manifest/query stream count mismatch")
    declared_fold_counts = manifest.get("fold_counts")
    observed_fold_counts = {
        str(fold): queries_by_fold[fold] for fold in sorted(ALLOWED_FOLDS)
    }
    if declared_fold_counts is not None and declared_fold_counts != observed_fold_counts:
        raise ValueError("SceneIR manifest/query fold counts mismatch")
    if rows_after > cap * len(seen_queries):
        raise AssertionError("parent fit sample global cap exceeded")
    return ParentFitSampleDiagnosticsV12(
        rows_before_sampling=rows_before,
        rows_after_sampling=rows_after,
        maximum_candidates_per_query=cap,
        natural_parent_positive_queries=natural_positives,
        queries=len(seen_queries),
        queries_by_fold=observed_fold_counts,
        natural_parent_rows_before_master_cap=full_rows,
        natural_positive_queries_before_master_cap=full_positives,
        ledger_queries_without_scene=len(ledger.index.difference(seen_queries)),
        maximum_natural_parents_before_master_cap=max_full,
        maximum_bounded_parents=max_bounded,
        fit_sample_semantic_sha256=semantic_digest.hexdigest(),
    )


def collect_parent_fit_sample_v12(
    scene_cache: SceneIRCacheLikeV12,
    *,
    sidecars: SidecarReaderV12,
    cascade_config: RetrievalTop100CascadeV12Config,
    admission_config: AdmissionConfigV12,
    labels: pd.DataFrame,
    batch_size: int = 262_144,
    signal_capability: ParentSignalCapabilityV12 | None = None,
) -> tuple[pd.DataFrame, ParentFitSampleDiagnosticsV12]:
    """In-memory fixture/parity adapter for the streaming training boundary."""

    pieces: list[pd.DataFrame] = []
    diagnostics = stream_parent_fit_sample_v12(
        scene_cache,
        sidecars=sidecars,
        cascade_config=cascade_config,
        admission_config=admission_config,
        labels=labels,
        consume=pieces.append,
        batch_size=batch_size,
        signal_capability=signal_capability,
    )
    return (
        pd.concat(pieces, ignore_index=True).loc[:, PARENT_SAMPLE_COLUMNS_V12],
        diagnostics,
    )


def parent_bundle_lineage_from_inputs_v12(
    scene_cache: SceneIRCache,
    *,
    sidecar_dir: Path,
    labels_path: Path,
    labels_manifest_path: Path,
    config_path: Path,
) -> ParentBundleLineageV12:
    """Seal and cross-check every lineage edge used by the fused trainer."""

    manifest = scene_cache.manifest
    if manifest.get("folds") != [2, 3, 4]:
        raise ValueError("SceneIR parent lineage opened a forbidden fold")
    source_candidate_sha = _require_sha256(
        manifest.get("source_candidate_sha256"),
        field="source_candidate_sha256",
    )
    source_manifest_sha = _require_sha256(
        manifest.get("source_manifest_sha256"),
        field="source_manifest_sha256",
    )
    build_id = _require_sha256(manifest.get("build_id"), field="build_id")
    scene_manifest_path = Path(scene_cache.cache_dir) / "manifest.json"
    sidecar_manifest_path = Path(sidecar_dir).resolve() / "manifest.json"
    for path in (
        scene_manifest_path,
        sidecar_manifest_path,
        Path(labels_path),
        Path(labels_manifest_path),
        Path(config_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    sidecar_sha = sha256_file(sidecar_manifest_path)
    strict_source = manifest.get("strict_source_manifest")
    if (
        not isinstance(strict_source, Mapping)
        or strict_source.get("sidecar_manifest_sha256") != sidecar_sha
    ):
        raise ValueError("SceneIR source and parent sidecar snapshot disagree")
    return ParentBundleLineageV12(
        source_candidate_sha256=source_candidate_sha,
        source_manifest_sha256=source_manifest_sha,
        sidecar_manifest_sha256=sidecar_sha,
        source_labels_sha256=sha256_file(Path(labels_path)),
        source_labels_manifest_sha256=sha256_file(Path(labels_manifest_path)),
        config_sha256=sha256_file(Path(config_path)),
        scene_ir_manifest_sha256=sha256_file(scene_manifest_path),
        scene_ir_build_id=build_id,
    )


def build_fold_fused_parent_bundle_v12(
    scene_cache: SceneIRCacheLikeV12,
    *,
    sidecars: SidecarReaderV12,
    cascade_config: RetrievalTop100CascadeV12Config,
    admission_config: AdmissionConfigV12,
    labels: pd.DataFrame,
    lineage: ParentBundleLineageV12,
    output_dir: Path,
    batch_size: int = 262_144,
    signal_capability: ParentSignalCapabilityV12 | None = None,
    model_fitter: ParentModelFitterV12 = fit_parent_ranker_from_cascade_rows,
) -> ParentBundleBuildV12:
    """Stream, fit and atomically publish the compatible six-model bundle."""

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    spool = _CompactParentSampleSpoolV12(stage)
    try:
        diagnostics = stream_parent_fit_sample_v12(
            scene_cache,
            sidecars=sidecars,
            cascade_config=cascade_config,
            admission_config=admission_config,
            labels=labels,
            consume=spool.append,
            batch_size=batch_size,
            signal_capability=signal_capability,
        )
        spool.close()

        models: dict[str, Mapping[str, Any]] = {}
        fit_diagnostics: dict[str, Mapping[str, int]] = {}
        model_paths: dict[str, Path] = {}
        for name, fit_folds in PARENT_FIT_FOLDS_V12.items():
            fit_rows = spool.read_folds(fit_folds)
            if set(fit_rows["fold"].astype(int).unique()) != set(fit_folds):
                raise ValueError(f"parent model {name} fit-fold projection drift")
            model, current_diagnostics = model_fitter(
                fit_rows, admission_config
            )
            model_path = stage / f"parent_{name}.json"
            model.save_model(model_path)
            if not model_path.is_file():
                raise ValueError(f"parent model fitter did not seal {model_path.name}")
            model_paths[name] = model_path
            models[name] = {
                "kind": PARENT_MODEL_KIND_V12,
                "path": model_path.name,
                "sha256": sha256_file(model_path),
                "fit_folds": list(fit_folds),
            }
            fit_diagnostics[name] = {
                str(key): int(value) for key, value in current_diagnostics.items()
            }

        # The sampled spool is a transient execution detail.  Removing it
        # guarantees that the publication contains models/lineage only, not a
        # replacement multi-million-row parent candidate artifact.
        spool.remove()
        outer_maps = {
            f"outer{outer}": {
                "outer_heldout_fold": int(outer),
                "scorer_by_query_fold": mapping,
            }
            for outer, mapping in OUTER_PARENT_MAPS_V12.items()
        }
        manifest: dict[str, Any] = {
            "schema_version": PARENT_OOF_BUNDLE_SCHEMA_V12,
            "run_kind": PARENT_BUNDLE_RUN_KIND_V12,
            "gate_eligible_for_nested_candidate_generation": True,
            "parent_feature_origin": PARENT_FEATURE_ORIGIN_V12,
            "parent_feature_names": list(parent_feature_names(admission_config)),
            "parent_model_matrix_dtype": "float32",
            "opened_folds": [2, 3, 4],
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "source_retrieval_lineage": lineage.source_retrieval_lineage,
            "source_labels_sha256": lineage.source_labels_sha256,
            "source_labels_manifest_sha256": (
                lineage.source_labels_manifest_sha256
            ),
            "config_sha256": lineage.config_sha256,
            "source_scene_ir": {
                "build_id": lineage.scene_ir_build_id,
                "manifest_sha256": lineage.scene_ir_manifest_sha256,
                "candidate_data_copied": False,
            },
            "parent_signal_capability": (
                {
                    "build_id": signal_capability.build_id,
                    "manifest_sha256": sha256_file(
                        signal_capability.manifest_path
                    ),
                    "source_candidate_sha256": (
                        signal_capability.source_candidate_sha256
                    ),
                    "candidate_rows": signal_capability.candidate_rows,
                    "strong_projection": list(signal_capability.projection),
                }
                if signal_capability is not None
                else None
            ),
            "physical_plan": {
                "id": PARENT_PHYSICAL_PLAN_V12,
                "scene_compilations_per_query": 1,
                "bootstrap_parent_score": "DETERMINISTIC_V12_DEFAULT",
                "label_attached_after_complete_natural_parent_pool": True,
                "fit_rows_maximum_per_query": diagnostics.maximum_candidates_per_query,
                "full_parent_candidate_parquet_materialized": False,
                "temporary_fit_sample_removed_before_publication": True,
                "post_expansion_parent_rescore": False,
            },
            "models": models,
            "outer_maps": outer_maps,
            "diagnostics": fit_diagnostics,
            "fit_sampling": diagnostics.as_manifest(),
        }
        manifest_path = stage / "manifest.json"
        _write_json(manifest_path, manifest)
        os.rename(stage, target)
        return ParentBundleBuildV12(
            output_dir=target,
            manifest_path=target / "manifest.json",
            model_paths={
                name: target / path.name for name, path in model_paths.items()
            },
            diagnostics=diagnostics,
        )
    except Exception:
        for writer in spool._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "BootstrapParentCandidateV12",
    "OUTER_PARENT_MAPS_V12",
    "PARENT_FIT_FOLDS_V12",
    "PARENT_OOF_BUNDLE_SCHEMA_V12",
    "PARENT_SAMPLE_COLUMNS_V12",
    "ParentBundleBuildV12",
    "ParentBundleLineageV12",
    "ParentFitSampleDiagnosticsV12",
    "build_fold_fused_parent_bundle_v12",
    "collect_parent_fit_sample_v12",
    "deterministic_bootstrap_parent_pool_v12",
    "parent_bundle_lineage_from_inputs_v12",
    "sample_bootstrap_parent_pool_v12",
    "stream_parent_fit_sample_v12",
    "validate_strict_parent_labels_v12",
]
