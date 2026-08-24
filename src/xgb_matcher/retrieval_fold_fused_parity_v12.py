"""Sealed differential parity harness for the fold-fused v12 executor.

The harness deliberately has two phases.  ``PREFIT`` proves every invariant
that can be checked before the three site LambdaMART models exist.  ``FULL``
replays the same 512 scenes with an already sealed site-model manifest and
adds PreRank500/top100 parity.  This split avoids requiring models produced by
an OOF run as a prerequisite of that same run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_fold_fused_oof_v12 import (
    ExistingV12SelectionAdapter,
    FeatureBatch,
    FittedSiteModel,
    FoldFusedInvariantError,
    MasterContext,
    PARITY_SCHEMA_VERSION,
    REQUIRED_PARITY_CHECKS,
    SceneRecord,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_heldout_spool_v12 import (
    CALIBRATION_VERSION,
    ParquetHeldoutMasterSpoolV12,
    projected_national_spool_bytes,
)
from .retrieval_fold_fused_runtime_v12 import (
    PREFIT_PARITY_CHECKS,
    PREFIT_PARITY_SCHEMA_VERSION,
    ExistingV12FitSamplerAdapter,
    FoldFusedContextAdapterV12,
    StrictV12LabelLedger,
)
from .retrieval_fold_fused_v12 import (
    PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12,
)
from .retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    canonicalize_union,
    sample_precomputed_candidates,
    site_feature_names,
)
from .retrieval_top100_cascade_v12 import (
    CASCADE_PARENT_FEATURES,
    RetrievalTop100CascadeV12,
    RetrievalTop100CascadeV12Config,
)
from .retrieval_v12_scene_ir import SceneIRCache


PARITY_ROWS = 512
PARITY_ROW_SCHEMA_VERSION = "sireto-retrieval-v12-fold-fused-parity-rows-1"


def _canonical_json(value: Any) -> bytes:
    def compatible(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): compatible(val) for key, val in item.items()}
        if isinstance(item, (list, tuple, set, frozenset)):
            values = [compatible(val) for val in item]
            return sorted(values, key=repr) if isinstance(item, (set, frozenset)) else values
        if isinstance(item, np.generic):
            return compatible(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return {"__nonfinite_float__": repr(item)}
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return str(item)

    return (
        json.dumps(
            compatible(value),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _float32_bits(values: np.ndarray) -> bytes:
    matrix = np.ascontiguousarray(values, dtype=np.float32)
    return matrix.view(np.uint32).tobytes(order="C")


def _parent_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Seal both parent identities/order and their exact float32 score bytes."""

    matrix = np.asarray(
        [
            [
                float(row.get("parent_rank") or 0.0),
                float(row.get("parent_score") or 0.0),
                *[float(value) for value in row["parent_feature_values"]],
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    digest = hashlib.sha256()
    digest.update(_canonical_json(_parent_projection(rows)))
    digest.update(_float32_bits(matrix))
    return digest.hexdigest()


def _feature_digest(batch: FeatureBatch) -> str:
    """Seal site identities/schema/order together with exact float32 bytes."""

    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "candidate_sirets": batch.candidate_sirets,
                "candidate_parents": batch.candidate_parents,
                "feature_names": batch.feature_names,
            }
        )
    )
    digest.update(_float32_bits(batch.values))
    return digest.hexdigest()


class _ScalarParentScorer:
    """Replay the legacy one-parent prediction through the same model."""

    def __init__(self, matrix_scorer: Any) -> None:
        self.matrix_scorer = matrix_scorer

    def score(self, features: Mapping[str, float]) -> float:
        if tuple(features) != tuple(CASCADE_PARENT_FEATURES):
            raise FoldFusedInvariantError("legacy parent feature schema/order drift")
        values = np.asarray(
            [[float(features[name]) for name in CASCADE_PARENT_FEATURES]],
            dtype=np.float32,
        )
        scores = np.asarray(self.matrix_scorer.score_matrix(values)).reshape(-1)
        if scores.shape != (1,) or not np.isfinite(scores).all():
            raise FoldFusedInvariantError("legacy parent scorer returned invalid output")
        return float(scores[0])


@dataclass(frozen=True)
class SiteModelCatalogV12:
    manifest_path: Path
    manifest_sha256: str
    models: Mapping[int, FittedSiteModel]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        feature_names: Sequence[str],
        expected_lineage: Mapping[str, str],
    ) -> "SiteModelCatalogV12":
        import xgboost as xgb

        manifest_path = Path(path).resolve()
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        lineage = raw.get("lineage")
        if not isinstance(lineage, Mapping):
            raise FoldFusedInvariantError("site model manifest lacks run lineage")
        for key, expected in expected_lineage.items():
            if lineage.get(key) != expected:
                raise FoldFusedInvariantError(f"site model lineage mismatch: {key}")
        records = raw.get("site_models")
        if not isinstance(records, Mapping) or set(records) != {"2", "3", "4"}:
            raise FoldFusedInvariantError("site model manifest must seal three outer models")
        models: dict[int, FittedSiteModel] = {}
        recipes: set[str] = set()
        for outer in TRAIN_FOLDS:
            record = records[str(outer)]
            if not isinstance(record, Mapping):
                raise FoldFusedInvariantError("invalid site model record")
            model_path = Path(str(record.get("path") or ""))
            if not model_path.is_absolute():
                model_path = manifest_path.parent / model_path
            model_path = model_path.resolve()
            if not model_path.is_file() or sha256_file(model_path) != record.get("sha256"):
                raise FoldFusedInvariantError("site model content hash mismatch")
            recipe = str(record.get("recipe_sha256") or "")
            recipes.add(recipe)
            model = xgb.XGBRanker()
            model.load_model(model_path)
            models[outer] = FittedSiteModel(
                outer_fold=outer,
                handle=model,
                path=model_path,
                sha256=str(record["sha256"]),
                recipe_sha256=recipe,
                feature_names=tuple(feature_names),
            )
        if len(recipes) != 1 or any(len(value) != 64 for value in recipes):
            raise FoldFusedInvariantError("site models do not share one sealed recipe")
        return cls(manifest_path, sha256_file(manifest_path), models)


@dataclass(frozen=True)
class FoldFusedParityInputsV12:
    scene_cache: SceneIRCache
    query_sample: pd.DataFrame
    query_sample_path: Path
    ledger: StrictV12LabelLedger
    sidecars: Any
    parent_scorers: Mapping[str, Any]
    admission_config: AdmissionConfigV12
    cascade_config: RetrievalTop100CascadeV12Config
    lineage: Mapping[str, str]
    site_models: SiteModelCatalogV12 | None = None
    national_query_count: int = PARITY_ROWS
    maximum_spool_bytes: int = 20 * 1024**3


def _validate_query_sample(
    sample: pd.DataFrame, *, ledger: StrictV12LabelLedger
) -> pd.DataFrame:
    expected_columns = ["query_id", "fold"]
    if list(sample.columns) != expected_columns or len(sample) != PARITY_ROWS:
        raise FoldFusedInvariantError(
            "parity sample must contain exactly query_id/fold and 512 rows"
        )
    output = sample.copy()
    output["query_id"] = output["query_id"].astype(str)
    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
    keys = output["query_id"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    ordered = (
        output.assign(__key=keys)
        .sort_values(["__key", "query_id"], kind="stable")
        .index.tolist()
    )
    if (
        output["query_id"].eq("").any()
        or output["query_id"].duplicated().any()
        or not set(output["fold"]).issubset(set(TRAIN_FOLDS))
        or ordered != list(output.index)
    ):
        raise FoldFusedInvariantError("parity sample identity/fold/order drift")
    ledger_folds = ledger.fold_by_query()
    if any(ledger_folds.get(row.query_id) != int(row.fold) for row in output.itertuples()):
        raise FoldFusedInvariantError("parity sample differs from the sealed label ledger")
    return output.reset_index(drop=True)


def _parent_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "siret": str(row["siret"]),
            "siren": str(row["siren"]),
            "master_rank": row.get("master_rank"),
            "parent_rank": row.get("parent_rank"),
            "parent_score": row.get("parent_score"),
            "parent_feature_names": row.get("parent_feature_names"),
            "parent_feature_values": row.get("parent_feature_values"),
        }
        for row in rows
    ]


def _master_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ignored = {"retrieval_latency_ms"}
    return [
        {key: value for key, value in row.items() if key not in ignored}
        for row in rows
    ]


def _feature_batch_from_canonical(
    canonical: pd.DataFrame,
    *,
    context: MasterContext,
    features: Sequence[str],
    statistics_sha256: str,
    row_payloads: Sequence[Mapping[str, Any]],
) -> FeatureBatch:
    return FeatureBatch(
        query_id=context.query_id,
        scene_fold=context.scene_fold,
        outer_fold=context.outer_fold,
        candidate_indices=tuple(range(len(canonical))),
        candidate_sirets=tuple(canonical["candidate_siret"].astype(str)),
        candidate_parents=tuple(canonical["candidate_parent"].astype(str)),
        feature_names=tuple(features),
        values=np.ascontiguousarray(
            canonical[list(features)].fillna(0.0).to_numpy(dtype=np.float32),
            dtype=np.float32,
        ),
        statistics_sha256=statistics_sha256,
        row_payloads=tuple(row_payloads),
    )


def _score_site(model: FittedSiteModel, batch: FeatureBatch) -> np.ndarray:
    if batch.feature_names != model.feature_names:
        raise FoldFusedInvariantError("parity site model feature schema drift")
    return np.asarray(model.handle.predict(batch.values), dtype=np.float64)


def run_fold_fused_parity_v12(
    inputs: FoldFusedParityInputsV12,
    *,
    output_dir: Path,
    batch_size: int = 262_144,
) -> Path:
    """Run legacy-vs-fused parity and publish a content-sealed manifest."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if batch_size < 1:
        raise FoldFusedInvariantError("parity SceneIR batch size must be positive")
    sample = _validate_query_sample(inputs.query_sample, ledger=inputs.ledger)
    if (
        inputs.national_query_count < PARITY_ROWS
        or inputs.maximum_spool_bytes <= 0
    ):
        raise FoldFusedInvariantError("parity spool calibration bounds are invalid")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    calibration_temp = tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.spool-calibration.", dir=output_dir.parent
    )
    calibration_spool = ParquetHeldoutMasterSpoolV12(
        Path(calibration_temp.name) / "heldout_spool_calibration",
        config=inputs.admission_config,
        maximum_bytes=inputs.maximum_spool_bytes,
    )
    wanted = set(sample["query_id"])
    context_adapter = FoldFusedContextAdapterV12(
        sidecars=inputs.sidecars,
        parent_scorers=inputs.parent_scorers,
        cascade_config=inputs.cascade_config,
        admission_config=inputs.admission_config,
    )
    sampler = ExistingV12FitSamplerAdapter(inputs.admission_config)
    selector = ExistingV12SelectionAdapter(inputs.admission_config)
    features = tuple(site_feature_names(inputs.admission_config))
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()

    selective = getattr(inputs.scene_cache, "iter_selected_scenes", None)
    if callable(selective):
        scene_iterator = selective(tuple(wanted))
    else:  # narrow fixture/compatibility surface only
        scene_iterator = (
            scene
            for scene in inputs.scene_cache.iter_scenes(batch_size=batch_size)
            if scene.query_id in wanted
        )
    for scene_ir in scene_iterator:
        if scene_ir.fold not in TRAIN_FOLDS or scene_ir.query_id in observed:
            raise FoldFusedInvariantError("parity SceneIR population/fold drift")
        observed.add(scene_ir.query_id)
        scene = SceneRecord(
            query_id=scene_ir.query_id,
            fold=scene_ir.fold,
            payload=scene_ir,
        )
        raw_rows = scene_ir.as_mappings()
        metadata = dict(raw_rows[0])

        # Build all three label-blind fused contexts before opening the label.
        prepared: dict[int, tuple[MasterContext, Any, FeatureBatch, Any]] = {}
        for outer in TRAIN_FOLDS:
            context = context_adapter.prepare(scene, outer_fold=outer)
            statistics = context_adapter.compute_full_master_statistics(context)
            fused_features = context_adapter.materialize(
                context, statistics, candidate_indices=None
            )
            if outer == scene.fold:
                # Exactly the context later replayed by OOF; still label-blind
                # and appended before the ledger is opened below.
                calibration_spool.append(context)
            model_key = PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12[outer][scene.fold]
            legacy = RetrievalTop100CascadeV12(
                inputs.sidecars,
                inputs.cascade_config,
                parent_scorer=_ScalarParentScorer(inputs.parent_scorers[model_key]),
            ).cascade(raw_rows, query_id=scene.query_id)
            legacy_rows = tuple(legacy.master_rows(scene.query_id, metadata))
            fused_rows = tuple(context.opaque.raw_rows)

            legacy_parent_sha = _parent_digest(legacy_rows)
            fused_parent_sha = _parent_digest(fused_rows)
            parent_equal = legacy_parent_sha == fused_parent_sha
            master_equal = _canonical_json(_master_projection(legacy_rows)) == _canonical_json(
                _master_projection(fused_rows)
            )

            legacy_canonical = canonicalize_union(
                pd.DataFrame(legacy_rows),
                inputs.admission_config,
                labels_allowed=False,
            )
            legacy_batch = _feature_batch_from_canonical(
                legacy_canonical,
                context=context,
                features=features,
                statistics_sha256=statistics.statistics_sha256,
                row_payloads=legacy_rows,
            )
            legacy_features_sha = _feature_digest(legacy_batch)
            fused_features_sha = _feature_digest(fused_features)
            feature_equal = legacy_features_sha == fused_features_sha
            prepared[outer] = (
                context,
                statistics,
                fused_features,
                (legacy_canonical, legacy_batch, legacy_rows),
            )
            rows.append(
                {
                    "schema_version": PARITY_ROW_SCHEMA_VERSION,
                    "query_id": scene.query_id,
                    "fold": scene.fold,
                    "outer_fold": outer,
                    "parent_scores_bit_identical": parent_equal,
                    "master_candidates_and_ranks_byte_identical": master_equal,
                    "site_features_float32_bit_identical": feature_equal,
                    "fit_row_ids_and_order_byte_identical": False,
                    "prerank500_byte_identical": None,
                    "top100_byte_identical": None,
                    "master_count": len(context.candidate_sirets),
                    "legacy_parent_sha256": legacy_parent_sha,
                    "fused_parent_sha256": fused_parent_sha,
                    "legacy_master_sha256": _digest(_master_projection(legacy_rows)),
                    "fused_master_sha256": _digest(_master_projection(fused_rows)),
                    "legacy_features_sha256": legacy_features_sha,
                    "fused_features_sha256": fused_features_sha,
                    "legacy_fit_rows_sha256": None,
                    "fused_fit_rows_sha256": None,
                    "legacy_prerank500_sha256": None,
                    "fused_prerank500_sha256": None,
                    "legacy_top100_sha256": None,
                    "fused_top100_sha256": None,
                }
            )
            if not parent_equal or not master_equal or not feature_equal:
                raise FoldFusedInvariantError(
                    f"prefit parity divergence query={scene.query_id} outer={outer}"
                )

        # The natural human label is opened only after all three complete
        # Master/statistics/feature views above have been sealed.
        label = inputs.ledger.get(scene.query_id)
        for outer in TRAIN_FOLDS:
            context, statistics, fused_features, legacy_payload = prepared[outer]
            legacy_canonical, legacy_batch, _legacy_rows = legacy_payload
            plan = sampler.choose(
                context,
                statistics,
                label,
                maximum_rows=sampler.maximum_rows,
            )
            legacy_labelled = legacy_canonical.copy()
            legacy_labelled["human_siret"] = label.human_siret
            legacy_labelled["operational_sirets_json"] = json.dumps(
                list(label.operational_sirets), separators=(",", ":")
            )
            legacy_sample = sample_precomputed_candidates(
                legacy_labelled, inputs.admission_config
            )
            fused_sample_ids = tuple(
                context.candidate_sirets[index] for index in plan.candidate_indices
            )
            sample_equal = fused_sample_ids == tuple(
                legacy_sample["candidate_siret"].astype(str)
            )
            record = rows[-len(TRAIN_FOLDS) + TRAIN_FOLDS.index(outer)]
            record["fit_row_ids_and_order_byte_identical"] = sample_equal
            record["fit_sample_count"] = len(fused_sample_ids)
            record["legacy_fit_rows_sha256"] = _digest(
                tuple(legacy_sample["candidate_siret"].astype(str))
            )
            record["fused_fit_rows_sha256"] = _digest(fused_sample_ids)
            if not sample_equal:
                raise FoldFusedInvariantError(
                    f"fit sample parity divergence query={scene.query_id} outer={outer}"
                )

            if inputs.site_models is not None:
                model = inputs.site_models.models[outer]
                legacy_scores = _score_site(model, legacy_batch)
                fused_scores = _score_site(model, fused_features)
                legacy_selection = selector.select(
                    context,
                    legacy_batch,
                    legacy_scores,
                    prerank_cap=500,
                    product_cap=100,
                )
                fused_selection = selector.select(
                    context,
                    fused_features,
                    fused_scores,
                    prerank_cap=500,
                    product_cap=100,
                )
                legacy_prerank = tuple(
                    legacy_batch.candidate_sirets[index]
                    for index in legacy_selection.prerank_indices
                )
                fused_prerank = tuple(
                    fused_features.candidate_sirets[index]
                    for index in fused_selection.prerank_indices
                )
                legacy_product = tuple(
                    (
                        legacy_batch.candidate_sirets[index],
                        legacy_selection.product_reasons[position],
                    )
                    for position, index in enumerate(legacy_selection.product_indices)
                )
                fused_product = tuple(
                    (
                        fused_features.candidate_sirets[index],
                        fused_selection.product_reasons[position],
                    )
                    for position, index in enumerate(fused_selection.product_indices)
                )
                record["prerank500_byte_identical"] = legacy_prerank == fused_prerank
                record["top100_byte_identical"] = legacy_product == fused_product
                record["legacy_prerank500_sha256"] = _digest(legacy_prerank)
                record["fused_prerank500_sha256"] = _digest(fused_prerank)
                record["legacy_top100_sha256"] = _digest(legacy_product)
                record["fused_top100_sha256"] = _digest(fused_product)
                if not record["prerank500_byte_identical"] or not record["top100_byte_identical"]:
                    raise FoldFusedInvariantError(
                        f"postfit parity divergence query={scene.query_id} outer={outer}"
                    )

    if observed != wanted or len(rows) != PARITY_ROWS * len(TRAIN_FOLDS):
        raise FoldFusedInvariantError("parity run did not cover sealed 512 x 3 contexts")

    calibration_artifact = calibration_spool.seal()
    if calibration_artifact.query_count != PARITY_ROWS:
        raise FoldFusedInvariantError("spool calibration did not cover sealed parity sample")
    projected_spool_bytes = projected_national_spool_bytes(
        sample_bytes=calibration_artifact.size_bytes,
        sample_queries=calibration_artifact.query_count,
        national_queries=inputs.national_query_count,
    )
    if projected_spool_bytes > inputs.maximum_spool_bytes:
        raise FoldFusedInvariantError(
            "heldout spool 512-query projection exceeds the frozen temporary budget"
        )

    row_frame = pd.DataFrame(rows).sort_values(
        ["query_id", "outer_fold"], kind="stable"
    ).reset_index(drop=True)
    complete = inputs.site_models is not None
    checks = {
        name: bool(row_frame[name].eq(True).all())
        for name in PREFIT_PARITY_CHECKS
    }
    if complete:
        checks.update(
            {
                "prerank500_byte_identical": bool(
                    row_frame["prerank500_byte_identical"].eq(True).all()
                ),
                "top100_byte_identical": bool(
                    row_frame["top100_byte_identical"].eq(True).all()
                ),
            }
        )

    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        calibration_destination = stage / "heldout_spool_calibration"
        shutil.copytree(calibration_artifact.path.parent, calibration_destination)
        calibration_record = {
            "schema_version": CALIBRATION_VERSION,
            "status": "PASS",
            "sample_query_count": calibration_artifact.query_count,
            "sample_row_count": calibration_artifact.row_count,
            "actual_sample_size_bytes": calibration_artifact.size_bytes,
            "national_query_count": inputs.national_query_count,
            "projected_national_size_bytes": projected_spool_bytes,
            "maximum_bytes": inputs.maximum_spool_bytes,
            "budget_pass": True,
            "schema_sha256": calibration_artifact.schema_sha256,
            "fold0_opened": False,
            "fold1_opened": False,
            "label_columns": [],
            "artifact": {
                "path": "heldout_spool_calibration/heldout_master.parquet",
                "sha256": calibration_artifact.sha256,
                "size_bytes": calibration_artifact.size_bytes,
            },
            "artifact_manifest": {
                "path": "heldout_spool_calibration/manifest.json",
                "sha256": calibration_artifact.manifest_sha256,
            },
        }
        reference = {
            "schema_version": "sireto-retrieval-v12-fold-fused-reference-1",
            "engine": "LEGACY_SINGLE_CONTEXT_V12",
            "query_count": PARITY_ROWS,
            "context_count": len(row_frame),
            "folds": list(TRAIN_FOLDS),
            "lineage": dict(inputs.lineage),
            "site_model_manifest_sha256": (
                inputs.site_models.manifest_sha256 if inputs.site_models else None
            ),
            "site_model_sha256": (
                {
                    str(outer): inputs.site_models.models[outer].sha256
                    for outer in TRAIN_FOLDS
                }
                if inputs.site_models
                else None
            ),
            "site_model_recipe_sha256": (
                {
                    str(outer): inputs.site_models.models[outer].recipe_sha256
                    for outer in TRAIN_FOLDS
                }
                if inputs.site_models
                else None
            ),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
        }
        reference_path = stage / "reference_manifest.json"
        reference_path.write_bytes(_canonical_json(reference))
        rows_path = stage / "parity_rows.parquet"
        pq.write_table(
            pa.Table.from_pandas(row_frame, preserve_index=False),
            rows_path,
            compression="zstd",
        )
        manifest: dict[str, Any] = {
            "schema_version": (
                PARITY_SCHEMA_VERSION if complete else PREFIT_PARITY_SCHEMA_VERSION
            ),
            "status": "PASS" if all(checks.values()) else "FAIL",
            "phase": "FULL_POSTFIT" if complete else "PREFIT_NON_CIRCULAR",
            "reference_manifest_sha256": sha256_file(reference_path),
            "query_count": PARITY_ROWS,
            "context_count": len(row_frame),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "checks": checks,
            "stop_on_any_divergence": True,
            "postfit_checks_status": (
                "PASS" if complete else "PENDING_SITE_MODELS"
            ),
            "heldout_spool_calibration": calibration_record,
            "lineage": dict(inputs.lineage),
            "artifacts": {
                "reference_manifest": {
                    "path": reference_path.name,
                    "sha256": sha256_file(reference_path),
                },
                "parity_rows": {
                    "path": rows_path.name,
                    "sha256": sha256_file(rows_path),
                    "rows": len(row_frame),
                },
            },
        }
        if complete:
            manifest["required_checks"] = list(REQUIRED_PARITY_CHECKS)
            manifest["site_model_manifest_sha256"] = inputs.site_models.manifest_sha256
            manifest["site_model_sha256"] = {
                str(outer): inputs.site_models.models[outer].sha256
                for outer in TRAIN_FOLDS
            }
            manifest["site_model_recipe_sha256"] = {
                str(outer): inputs.site_models.models[outer].recipe_sha256
                for outer in TRAIN_FOLDS
            }
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        os.rename(stage, output_dir)
        calibration_temp.cleanup()
        return output_dir / "manifest.json"
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        calibration_temp.cleanup()
        raise


__all__ = [
    "FoldFusedParityInputsV12",
    "PARITY_ROW_SCHEMA_VERSION",
    "PARITY_ROWS",
    "SiteModelCatalogV12",
    "run_fold_fused_parity_v12",
]
