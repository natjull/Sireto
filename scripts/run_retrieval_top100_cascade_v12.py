#!/usr/bin/env python3
"""Run the bounded, label-blind retrieval v12 top-100 cascade.

The input must be a natural v11 lightweight union.  The command writes an
atomic directory containing a Master view (<=2,000) consumed by the SIRET
LambdaMART, a legacy heuristic hydration diagnostic (<=500), and parent
(<=64) Parquet views.  Learned PreRank@500 is produced later by admission from
the complete Master view.  This command does not attach or evaluate human
labels.  Folds 0 and 1 are rejected before scanning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_sidecars_v11 import RetrievalSidecarReader  # noqa: E402
from src.xgb_matcher.retrieval_top100_cascade_v12 import (  # noqa: E402
    CASCADE_PARENT_FEATURES,
    RescueParentHitV12,
    RetrievalTop100CascadeV12,
    RetrievalTop100CascadeV12Config,
    V12_CASCADE_SCHEMA_VERSION,
    V12_OUTPUT_CHANNELS,
    V12_PARENT_SCHEMA_VERSION,
    XGBoostParentScorerV12,
    production_projection,
)


MANIFEST_SCHEMA = "sireto-retrieval-top100-cascade-v12-manifest-1"
PARENT_MODEL_BUNDLE_SCHEMA = "sireto-parent-oof-model-bundle-v12-1"
PARENT_MODEL_RECIPE_MODE = "NESTED_OUTER_OOF"
ALLOWED_FOLDS = frozenset({2, 3, 4})
V11_SOURCE_MANIFEST_SCHEMA = (
    "sireto-retrieval-all-human-v11-candidates-manifest-1"
)
V11_CANDIDATE_SCHEMA = "sireto-retrieval-all-human-v11-union-1"
V11_LABEL_BLIND_RUN_KINDS = frozenset(
    {"LABEL_BLIND_RETRIEVAL", "LABEL_BLIND_RETRIEVAL_COMBINED"}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_SIGNAL_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("source", pa.string()),
            pa.field("channel", pa.string()),
            pa.field("rank", pa.int64()),
            pa.field("score", pa.float64()),
            pa.field("parent_identifier", pa.string()),
            pa.field("parent_rank", pa.int64()),
            pa.field("parent_support_count", pa.int64()),
            pa.field("site_rank", pa.int64()),
            pa.field("relation_direction", pa.string()),
            pa.field("relation_date", pa.string()),
        ]
    )
)
_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string()),
        pa.field("query_id", pa.string()),
        pa.field("fold", pa.int64()),
        pa.field("crm_name", pa.string()),
        pa.field("crm_address", pa.string()),
        pa.field("crm_insee", pa.string()),
        pa.field("crm_postcode", pa.string()),
        pa.field("retrieval_latency_ms", pa.float64()),
        pa.field("siret", pa.string()),
        pa.field("siren", pa.string()),
        pa.field("union_rank", pa.int64()),
        pa.field("master_rank", pa.int64()),
        pa.field("hydration_rank", pa.int64()),
        pa.field("parent_rank", pa.int64()),
        pa.field("site_rank", pa.int64()),
        pa.field("parent_score", pa.float64()),
        pa.field("parent_feature_names", pa.list_(pa.string())),
        pa.field("parent_feature_values", pa.list_(pa.float64())),
        pa.field("site_score", pa.float64()),
        pa.field("selection_lane", pa.string()),
        pa.field("lanes", pa.list_(pa.string())),
        pa.field("selection_reasons", pa.list_(pa.string())),
        pa.field("channels", pa.list_(pa.string())),
        pa.field("sources", pa.list_(pa.string())),
        pa.field("signals", _SIGNAL_TYPE),
        pa.field("official_evidence_ids", pa.list_(pa.string())),
        pa.field("official_evidence_sources", pa.list_(pa.string())),
        pa.field("candidate_names", pa.list_(pa.string())),
        pa.field("candidate_addresses", pa.list_(pa.string())),
        pa.field("candidate_insee", pa.string()),
        pa.field("candidate_postcode", pa.string()),
        pa.field("candidate_number", pa.string()),
        pa.field("candidate_state", pa.string()),
        pa.field("candidate_is_headquarters", pa.bool_()),
        pa.field("temporal_valid_from", pa.string()),
        pa.field("temporal_valid_to", pa.string()),
        *[
            field
            for channel in V12_OUTPUT_CHANNELS
            for field in (
                pa.field(f"{channel}_rank", pa.int64()),
                pa.field(f"{channel}_score", pa.float64()),
            )
        ],
    ]
)
_PARENT_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string()),
        pa.field("query_id", pa.string()),
        pa.field("fold", pa.int64()),
        pa.field("siren", pa.string()),
        pa.field("parent_rank", pa.int64()),
        pa.field("parent_score", pa.float64()),
        pa.field("support_sites", pa.int64()),
        pa.field("channels", pa.list_(pa.string())),
        pa.field("sources", pa.list_(pa.string())),
        pa.field("selection_reasons", pa.list_(pa.string())),
        pa.field("name_evidence", pa.bool_()),
        pa.field("address_evidence", pa.bool_()),
        pa.field("exact_name_evidence", pa.bool_()),
        pa.field("exact_address_evidence", pa.bool_()),
        pa.field("geographic_support", pa.bool_()),
        pa.field("rescue", pa.bool_()),
        pa.field("feature_names", pa.list_(pa.string())),
        pa.field("feature_values", pa.list_(pa.float64())),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    """Small dependency-free linear percentiles for runtime manifests."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "p99": None}

    def one(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    return {
        "count": len(ordered),
        "p50": one(0.50),
        "p95": one(0.95),
        "p99": one(0.99),
    }


@dataclass(frozen=True)
class _ParentModelRecipe:
    """Validated fold-aware parent scorers for one nested outer-OOF run.

    The model bundle is deliberately separate from candidate artefacts so it
    can be sealed before a cascade run.  Its selected outer map contains no
    labels: it only routes a query fold to the already-fitted parent model
    whose training folds are recorded in ``model_provenance``.
    """

    bundle_sha256: str
    config_sha256: str
    recipe_name: str
    outer_fold: int
    scorers_by_fold: Mapping[int, Any]
    model_provenance: Mapping[int, Mapping[str, Any]]
    model_sha256_by_name: Mapping[str, str]
    source_retrieval_lineage: Mapping[str, str]
    source_labels_sha256: str
    source_labels_manifest_sha256: str

    def scorer_for_fold(self, fold: int) -> Any:
        parsed = int(fold)
        if parsed not in self.scorers_by_fold:
            raise ValueError(
                f"sealed parent OOF recipe {self.recipe_name!r} has no model "
                f"for score fold {parsed}"
            )
        return self.scorers_by_fold[parsed]

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "gate_eligible": True,
            "outer_heldout_fold": self.outer_fold,
            "parent_feature_origin": "CASCADE_PRE_EXPANSION",
            "parent_feature_names": list(CASCADE_PARENT_FEATURES),
            "model_sha256_by_name": dict(sorted(self.model_sha256_by_name.items())),
            "training_lineage": {
                "source_retrieval": dict(self.source_retrieval_lineage),
                "label_ledger_sha256": self.source_labels_sha256,
                "label_ledger_manifest_sha256": self.source_labels_manifest_sha256,
            },
            "scorer_by_query_fold": {
                str(fold): dict(self.model_provenance[fold])
                for fold in sorted(self.model_provenance)
            },
        }


def _expected_nested_fit_folds(*, outer_fold: int, score_fold: int) -> set[int]:
    """Return the only leakage-free fit folds for one nested OOF score."""

    if outer_fold not in ALLOWED_FOLDS or score_fold not in ALLOWED_FOLDS:
        raise ValueError("v12 parent OOF recipes are restricted to folds 2/3/4")
    if score_fold == outer_fold:
        return set(ALLOWED_FOLDS.difference({outer_fold}))
    # Rows used to fit the site ranker must themselves have inner-OOF parent
    # scores.  Consequently neither the outer fold nor the row's own fold may
    # train that parent model.
    return set(ALLOWED_FOLDS.difference({outer_fold, score_fold}))


def _load_parent_model_recipe(
    bundle_path: Path,
    *,
    recipe_name: str,
    required_folds: set[int],
    scorer_loader: Any = None,
) -> _ParentModelRecipe:
    """Load and verify a content-sealed nested parent-model mapping.

    Every model hash is checked before any scorer is constructed.  The fold
    recipe is fixed rather than inferred: for outer fold ``H``, ``H`` is
    scored by a model fit on the other two folds; each training fold is scored
    by a model fit only on the remaining training fold.  This is the minimum
    two-pass bridge required to feed parent scores to an outer-OOF site model
    without using the held-out fold during parent fitting.
    """

    if scorer_loader is None:
        scorer_loader = XGBoostParentScorerV12.from_model_path
    path = bundle_path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != PARENT_MODEL_BUNDLE_SCHEMA:
        raise ValueError("unsupported v12 parent OOF model bundle schema")
    if raw.get("run_kind") != "PARENT_NESTED_OOF_MODEL_BUNDLE":
        raise ValueError("v12 parent model bundle has the wrong run_kind")
    if raw.get("gate_eligible_for_nested_candidate_generation") is not True:
        raise ValueError("v12 parent model bundle is not sealed as gate eligible")
    if raw.get("parent_feature_origin") != "CASCADE_PRE_EXPANSION":
        raise ValueError("v12 parent model bundle must use pre-expansion features")
    if tuple(raw.get("parent_feature_names") or ()) != CASCADE_PARENT_FEATURES:
        raise ValueError("v12 parent model bundle feature schema/order drift")
    if {int(value) for value in raw.get("opened_folds") or ()} != set(ALLOWED_FOLDS):
        raise ValueError("v12 parent model bundle must be trained only on folds 2/3/4")
    if raw.get("fold0_opened") is not False or raw.get("fold1_opened") is not False:
        raise ValueError("v12 parent model bundle must prove folds 0/1 stayed closed")
    if raw.get("positive_injection") is not False:
        raise ValueError("v12 parent model bundle must prove zero positive injection")
    bundle_config_sha256 = str(raw.get("config_sha256") or "")
    if not bundle_config_sha256:
        raise ValueError("v12 parent model bundle does not seal its config hash")
    source_retrieval_lineage = raw.get("source_retrieval_lineage")
    lineage_keys = {
        "source_candidate_sha256",
        "source_manifest_sha256",
        "sidecar_manifest_sha256",
    }
    if (
        not isinstance(source_retrieval_lineage, Mapping)
        or set(source_retrieval_lineage) != lineage_keys
        or any(
            not isinstance(source_retrieval_lineage.get(key), str)
            or re.fullmatch(r"[0-9a-f]{64}", source_retrieval_lineage[key]) is None
            for key in lineage_keys
        )
    ):
        raise ValueError("v12 parent model bundle lacks sealed retrieval lineage")
    source_labels_sha256 = raw.get("source_labels_sha256")
    source_labels_manifest_sha256 = raw.get("source_labels_manifest_sha256")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in (source_labels_sha256, source_labels_manifest_sha256)
    ):
        raise ValueError("v12 parent model bundle lacks sealed label-ledger lineage")

    outer_maps = raw.get("outer_maps")
    if not isinstance(outer_maps, Mapping) or recipe_name not in outer_maps:
        raise ValueError(f"unknown sealed parent OOF recipe {recipe_name!r}")
    recipe = outer_maps[recipe_name]
    if not isinstance(recipe, Mapping):
        raise ValueError("v12 parent model recipe must be an object")
    outer_fold = int(recipe.get("outer_heldout_fold", -1))
    if outer_fold not in ALLOWED_FOLDS:
        raise ValueError("v12 parent model recipe outer_heldout_fold must be 2, 3 or 4")

    requested = {int(value) for value in required_folds}
    if not requested or not requested.issubset(ALLOWED_FOLDS):
        raise ValueError("v12 parent model recipe requested folds must be within 2/3/4")
    models = raw.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("v12 parent model bundle has no models")
    expected_model_keys = {"fit_2", "fit_3", "fit_4", "fit_23", "fit_24", "fit_34"}
    if set(models) != expected_model_keys:
        raise ValueError("v12 parent model bundle must seal the six nested fit models")
    model_catalog: dict[str, tuple[Path, str, set[int], str]] = {}
    for model_key in sorted(expected_model_keys):
        entry = models[model_key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"parent model {model_key!r} must be an object")
        if entry.get("kind", "XGBOOST_BOOSTER") != "XGBOOST_BOOSTER":
            raise ValueError("sealed parent OOF recipes require XGBOOST_BOOSTER models")
        fit_folds = {int(value) for value in entry.get("fit_folds") or ()}
        encoded_fit_folds = {int(character) for character in model_key.removeprefix("fit_")}
        if fit_folds != encoded_fit_folds or not fit_folds.issubset(ALLOWED_FOLDS):
            raise ValueError(f"parent model {model_key!r} fit_folds drift")
        declared_path = str(entry.get("path") or "")
        if not declared_path:
            raise ValueError("parent model entry has an empty path")
        model_path = Path(declared_path)
        if not model_path.is_absolute():
            model_path = path.parent / model_path
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        declared_sha = str(entry.get("sha256") or "").lower()
        actual_sha = sha256_file(model_path)
        if declared_sha != actual_sha:
            raise ValueError(f"parent model hash mismatch for {model_key}")
        model_catalog[model_key] = (
            model_path,
            actual_sha,
            fit_folds,
            declared_path,
        )
    scorer_map = recipe.get("scorer_by_query_fold")
    if not isinstance(scorer_map, Mapping):
        raise ValueError("v12 parent model recipe has no scorer_by_query_fold")
    declared_score_folds = {int(value) for value in scorer_map}
    if declared_score_folds != set(ALLOWED_FOLDS):
        raise ValueError("nested v12 parent recipe must seal models for folds 2/3/4")

    # Validate *all* entries and hashes, even when the source currently contains
    # a single fold.  The emitted mapping hash then really attests to the whole
    # outer recipe rather than to a partially checked subset.
    provenance: dict[int, dict[str, Any]] = {}
    model_paths: dict[int, Path] = {}
    for score_fold in sorted(ALLOWED_FOLDS):
        route = scorer_map.get(str(score_fold), scorer_map.get(score_fold))
        if not isinstance(route, Mapping):
            raise ValueError(f"missing parent scorer route for query fold {score_fold}")
        model_key = str(route.get("model") or "")
        entry = models.get(model_key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"unknown parent model key {model_key!r}")
        model_path, actual_sha, fit_folds, declared_path = model_catalog[model_key]
        expected_fit = _expected_nested_fit_folds(
            outer_fold=outer_fold, score_fold=score_fold
        )
        if fit_folds != expected_fit:
            raise ValueError(
                f"leaky parent model for score fold {score_fold}: "
                f"fit_folds={sorted(fit_folds)}, expected={sorted(expected_fit)}"
            )
        expected_key = "fit_" + "".join(str(value) for value in sorted(fit_folds))
        if model_key != expected_key:
            raise ValueError(
                f"parent route for score fold {score_fold} must use {expected_key}"
            )
        expected_role = "OUTER_HELDOUT" if score_fold == outer_fold else "INNER_TRAIN"
        if route.get("role") != expected_role:
            raise ValueError(
                f"parent route for score fold {score_fold} must have role {expected_role}"
            )
        model_paths[score_fold] = model_path
        provenance[score_fold] = {
            "kind": "XGBOOST_BOOSTER",
            "model": model_key,
            "role": expected_role,
            "score_fold": score_fold,
            "fit_folds": sorted(fit_folds),
            "declared_path": declared_path,
            "sha256": actual_sha,
        }

    scorers = {fold: scorer_loader(model_paths[fold]) for fold in sorted(requested)}
    return _ParentModelRecipe(
        bundle_sha256=sha256_file(path),
        config_sha256=bundle_config_sha256,
        recipe_name=recipe_name,
        outer_fold=outer_fold,
        scorers_by_fold=scorers,
        model_provenance=provenance,
        model_sha256_by_name={
            model_key: values[1] for model_key, values in model_catalog.items()
        },
        source_retrieval_lineage=dict(source_retrieval_lineage),
        source_labels_sha256=source_labels_sha256,
        source_labels_manifest_sha256=source_labels_manifest_sha256,
    )


def _source_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"v11 source manifest lacks valid {field}")
    return value


def _validate_source_manifest_contract(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact label-blind v11 producer contract.

    In particular, self-declared ``SEALED_*`` fixture kinds are never treated
    as production lineage.  The national folds-2/3/4 artifact is emitted by
    the real v11 combiner, so its three shard manifests are also checked and
    their counts/folds must reconcile with the combined manifest.
    """

    if manifest.get("schema_version") != V11_SOURCE_MANIFEST_SCHEMA:
        raise ValueError("v12 source has the wrong v11 manifest schema")
    if manifest.get("candidate_schema_version") != V11_CANDIDATE_SCHEMA:
        raise ValueError("v12 source has the wrong v11 candidate schema")
    run_kind = manifest.get("run_kind")
    if run_kind not in V11_LABEL_BLIND_RUN_KINDS:
        raise ValueError("v12 source is not an official label-blind v11 run")
    if manifest.get("positive_injection") is not False:
        raise ValueError("v12 source does not prove zero positive injection")
    if manifest.get("human_labels_opened_by_retrieval") is not False:
        raise ValueError("v12 source does not prove label-blind retrieval")

    folds_raw = manifest.get("folds")
    if not isinstance(folds_raw, list) or any(
        not isinstance(value, int) for value in folds_raw
    ):
        raise ValueError("v11 source folds must be an explicit integer list")
    folds = set(folds_raw)
    if len(folds) != len(folds_raw) or not folds or not folds.issubset(ALLOWED_FOLDS):
        raise ValueError("v12 refuses any source declaring fold 0 or fold 1")

    candidate_sha = _require_sha256(
        manifest.get("candidate_parquet_sha256"),
        field="candidate_parquet_sha256",
    )
    for field in (
        "source_input_sha256",
        "config_sha256",
        "official_config_sha256",
        "sidecar_manifest_sha256",
    ):
        _require_sha256(manifest.get(field), field=field)
    if candidate_sha != sha256_file(path):
        raise ValueError("v12 source candidate hash mismatch")

    parquet_rows = int(pq.ParquetFile(path).metadata.num_rows)
    queries = manifest.get("queries")
    candidate_rows = manifest.get("candidate_rows")
    empty_queries = manifest.get("empty_queries")
    maximum = manifest.get("maximum_union_candidates")
    if not isinstance(queries, int) or queries <= 0:
        raise ValueError("v11 source manifest has an invalid query count")
    if not isinstance(candidate_rows, int) or candidate_rows != parquet_rows:
        raise ValueError("v11 source manifest candidate row count mismatch")
    if (
        not isinstance(empty_queries, int)
        or empty_queries < 0
        or empty_queries > queries
    ):
        raise ValueError("v11 source manifest has an invalid empty-query count")
    if not isinstance(maximum, int) or not 1 <= maximum <= 2_000:
        raise ValueError("v11 source manifest exceeds the 2,000-candidate cap")

    source_shards = manifest.get("source_shards")
    if run_kind == "LABEL_BLIND_RETRIEVAL_COMBINED":
        if not isinstance(source_shards, list) or len(source_shards) < 2:
            raise ValueError("combined v11 source lacks sealed source shards")
        shard_folds: set[int] = set()
        shard_queries = shard_rows = shard_empty = 0
        shard_maximum = 0
        stable_fields = (
            "schema_version",
            "candidate_schema_version",
            "source_input_sha256",
            "config_sha256",
            "official_config_sha256",
            "sidecar_manifest_sha256",
            "positive_injection",
            "human_labels_opened_by_retrieval",
        )
        seen_paths: set[Path] = set()
        for shard in source_shards:
            if not isinstance(shard, Mapping):
                raise ValueError("combined v11 source has an invalid shard entry")
            raw_shard_path = shard.get("path")
            if not isinstance(raw_shard_path, str) or not raw_shard_path:
                raise ValueError("combined v11 source shard lacks a path")
            shard_path = Path(raw_shard_path)
            if not shard_path.is_absolute() or shard_path in seen_paths:
                raise ValueError("combined v11 source shard paths must be unique/absolute")
            seen_paths.add(shard_path)
            shard_manifest_path = _source_manifest_path(shard_path)
            if not shard_path.is_file() or not shard_manifest_path.is_file():
                raise FileNotFoundError(
                    f"combined v11 source shard lineage is missing: {shard_path}"
                )
            declared_shard_sha = _require_sha256(shard.get("sha256"), field="shard sha256")
            declared_manifest_sha = _require_sha256(
                shard.get("manifest_sha256"), field="shard manifest_sha256"
            )
            if sha256_file(shard_manifest_path) != declared_manifest_sha:
                raise ValueError("combined v11 source shard manifest hash mismatch")
            if sha256_file(shard_path) != declared_shard_sha:
                raise ValueError("combined v11 source shard candidate hash mismatch")
            shard_manifest = json.loads(
                shard_manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(shard_manifest, Mapping):
                raise ValueError("combined v11 source shard manifest is not an object")
            if (
                shard_manifest.get("run_kind") != "LABEL_BLIND_RETRIEVAL"
                or shard_manifest.get("candidate_parquet_sha256")
                != declared_shard_sha
                or any(
                    shard_manifest.get(field) != manifest.get(field)
                    for field in stable_fields
                )
            ):
                raise ValueError("combined v11 source shard lineage mismatch")
            _require_sha256(
                shard_manifest.get("source_manifest_sha256"),
                field="shard source_manifest_sha256",
            )
            current_folds = shard_manifest.get("folds")
            if (
                not isinstance(current_folds, list)
                or len(current_folds) != 1
                or not isinstance(current_folds[0], int)
                or current_folds[0] not in ALLOWED_FOLDS
                or current_folds[0] in shard_folds
            ):
                raise ValueError("combined v11 source shards have invalid/overlapping folds")
            shard_folds.add(current_folds[0])
            current_rows = shard_manifest.get("candidate_rows")
            current_queries = shard_manifest.get("queries")
            current_empty = shard_manifest.get("empty_queries")
            current_maximum = shard_manifest.get("maximum_union_candidates")
            if (
                not isinstance(current_rows, int)
                or current_rows != int(pq.ParquetFile(shard_path).metadata.num_rows)
                or not isinstance(current_queries, int)
                or current_queries <= 0
                or not isinstance(current_empty, int)
                or current_empty < 0
                or current_empty > current_queries
                or not isinstance(current_maximum, int)
                or not 1 <= current_maximum <= 2_000
            ):
                raise ValueError("combined v11 source shard counts/cap are invalid")
            shard_rows += current_rows
            shard_queries += current_queries
            shard_empty += current_empty
            shard_maximum = max(shard_maximum, current_maximum)
        if (
            shard_folds != folds
            or shard_rows != candidate_rows
            or shard_queries != queries
            or shard_empty != empty_queries
            or shard_maximum != maximum
        ):
            raise ValueError("combined v11 source shard counts/folds do not reconcile")
    elif source_shards is not None:
        raise ValueError("non-combined v11 source may not declare source shards")

    return dict(manifest)


def _load_source_manifest(path: Path, *, allow_unsealed_fixture: bool) -> dict[str, Any]:
    manifest_path = _source_manifest_path(path)
    if not manifest_path.is_file():
        if allow_unsealed_fixture:
            return {
                "run_kind": "UNSEALED_TEST_FIXTURE",
                "folds": [2, 3, 4],
                "positive_injection": False,
                "human_labels_opened_by_retrieval": False,
            }
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("v11 source manifest must be a JSON object")
    return _validate_source_manifest_contract(path, manifest)


def _iter_query_groups(
    path: Path, *, batch_size: int = 65_536
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    parquet = pq.ParquetFile(path)
    columns = production_projection(parquet.schema_arrow.names)
    required = {"query_id", "siret", "siren"}
    if not required.issubset(columns):
        raise ValueError(f"v12 input is missing {sorted(required.difference(columns))}")
    current_id: str | None = None
    current: list[dict[str, Any]] = []
    closed: set[str] = set()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        for row in batch.to_pylist():
            query_id = str(row.get("query_id") or "")
            if not query_id:
                raise ValueError("v12 input contains an empty query_id")
            fold = int(row.get("fold", row.get("oof_fold", -1)))
            if fold not in ALLOWED_FOLDS:
                raise ValueError("v12 refuses folds outside 2/3/4")
            if current_id is None:
                current_id = query_id
            if query_id != current_id:
                closed.add(current_id)
                yield current_id, current
                if query_id in closed:
                    raise ValueError("v12 input query groups are not contiguous")
                current_id = query_id
                current = []
            current.append(row)
    if current_id is not None:
        yield current_id, current


class _PrecomputedRescue:
    def __init__(self, path: Path) -> None:
        table = pq.read_table(
            path,
            columns=["query_id", "siren", "rank", "score", "source"],
        )
        self._hits: dict[str, list[RescueParentHitV12]] = {}
        for row in table.to_pylist():
            self._hits.setdefault(str(row["query_id"]), []).append(
                RescueParentHitV12(
                    siren=str(row["siren"]),
                    rank=int(row["rank"]),
                    score=float(row["score"]),
                    source=str(row.get("source") or "contrastive_siren_rescue"),
                )
            )
        for values in self._hits.values():
            values.sort(key=lambda hit: (hit.rank, -hit.score, hit.siren))

    def retrieve(self, query: object, *, limit: int, query_id: str = ""):
        return self._hits.get(str(query_id), ())[:limit]


class _Sink:
    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = path
        self.schema = schema
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def write(self, values: Sequence[Mapping[str, Any]]) -> None:
        if not values:
            return
        table = pa.Table.from_pylist(list(values), schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table, row_group_size=65_536)
        self.rows += len(values)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fold": int(row.get("fold", row.get("oof_fold", -1))),
        "crm_name": str(row.get("crm_name") or ""),
        "crm_address": str(row.get("crm_address") or ""),
        "crm_insee": str(row.get("crm_insee") or row.get("insee") or ""),
        "crm_postcode": str(
            row.get("crm_postcode") or row.get("postcode") or ""
        ),
    }


def _upstream_retrieval_latency_ms(row: Mapping[str, Any]) -> float | None:
    raw = row.get("retrieval_latency_ms")
    if raw in (None, ""):
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("v12 upstream retrieval_latency_ms must be finite and non-negative")
    return value


def _candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = set(_CANDIDATE_SCHEMA.names)
    return [{key: row.get(key) for key in names} for row in rows]


def run(args: argparse.Namespace) -> Path:
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_manifest = _load_source_manifest(
        source, allow_unsealed_fixture=bool(args.allow_unsealed_fixture)
    )
    source_candidate_sha = source_manifest.get("candidate_parquet_sha256")
    if not isinstance(source_candidate_sha, str):
        source_candidate_sha = sha256_file(source)
    source_manifest_path = _source_manifest_path(source)
    source_manifest_sha = (
        sha256_file(source_manifest_path) if source_manifest_path.is_file() else None
    )
    source_folds = {int(value) for value in source_manifest["folds"]}
    config = RetrievalTop100CascadeV12Config.load(args.config)
    parent_choices = sum(
        value is not None
        for value in (args.parent_model, args.parent_weights, args.parent_model_bundle)
    )
    if parent_choices > 1:
        raise ValueError(
            "choose one of --parent-model, --parent-weights or --parent-model-bundle"
        )
    if parent_choices and args.allow_bootstrap_parent_score:
        raise ValueError("bootstrap parent scoring cannot accompany a frozen parent scorer")
    nested_parent_recipe: _ParentModelRecipe | None = None
    if args.parent_model_bundle is not None:
        if not args.parent_outer_recipe:
            raise ValueError("--parent-model-bundle requires --parent-outer-recipe")
        nested_parent_recipe = _load_parent_model_recipe(
            args.parent_model_bundle,
            recipe_name=str(args.parent_outer_recipe),
            required_folds=source_folds,
        )
        if nested_parent_recipe.config_sha256 != sha256_file(args.config):
            raise ValueError("parent OOF bundle/cascade config hash mismatch")
        parent_scorer = None
        parent_scorer_kind = "NESTED_OOF_FOLD_MODEL_MAPPING"
    elif args.parent_outer_recipe:
        raise ValueError("--parent-outer-recipe requires --parent-model-bundle")
    elif args.parent_model is not None:
        parent_scorer: Any = XGBoostParentScorerV12.from_model_path(args.parent_model)
        parent_scorer_kind = "XGBOOST_SHARED_PARENT_RANKER"
    elif args.parent_weights is not None:
        parent_scorer = json.loads(args.parent_weights.read_text(encoding="utf-8"))
        parent_scorer_kind = "FROZEN_LINEAR_WEIGHT_MAPPING"
    elif args.allow_bootstrap_parent_score:
        parent_scorer = None
        parent_scorer_kind = "DETERMINISTIC_BOOTSTRAP_NOT_GATE_ELIGIBLE"
    else:
        raise ValueError(
            "v12 requires the shared frozen parent ranker; deterministic "
            "bootstrap is fixture-only and must be explicit"
        )
    rescue = _PrecomputedRescue(args.rescue_hits) if args.rescue_hits else None
    if nested_parent_recipe is not None:
        expected_training_source = {
            "source_candidate_sha256": source_candidate_sha,
            "source_manifest_sha256": source_manifest_sha,
            "sidecar_manifest_sha256": sha256_file(args.sidecars / "manifest.json"),
        }
        if dict(nested_parent_recipe.source_retrieval_lineage) != expected_training_source:
            raise ValueError(
                "parent OOF bundle was fitted on another retrieval/sidecar snapshot"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    master_sink = _Sink(stage / "master_references.parquet", _CANDIDATE_SCHEMA)
    shortlist_sink = _Sink(stage / "hydration_shortlist.parquet", _CANDIDATE_SCHEMA)
    parent_candidate_sink = _Sink(stage / "parent_candidates.parquet", _PARENT_SCHEMA)
    parent_sink = _Sink(stage / "parent_shortlist.parquet", _PARENT_SCHEMA)
    query_count = rescue_trigger_count = rescue_execution_count = 0
    upstream_rescue_reference_count = 0
    max_source = max_master = max_shortlist = max_parent_candidates = max_parents = 0
    observed_folds: set[int] = set()
    missing_upstream_latency_queries = 0
    upstream_latencies_ms: list[float] = []
    cascade_latencies_ms: list[float] = []
    total_latencies_ms: list[float] = []
    try:
        with RetrievalSidecarReader(args.sidecars) as sidecars:
            if nested_parent_recipe is None:
                shared_cascade = RetrievalTop100CascadeV12(
                    sidecars,
                    config,
                    rescue_provider=rescue,
                    parent_scorer=parent_scorer,
                )
                cascades_by_fold: dict[int, RetrievalTop100CascadeV12] = {}
            else:
                shared_cascade = None
                cascades_by_fold = {
                    fold: RetrievalTop100CascadeV12(
                        sidecars,
                        config,
                        rescue_provider=rescue,
                        parent_scorer=nested_parent_recipe.scorer_for_fold(fold),
                    )
                    for fold in sorted(source_folds)
                }
            for query_id, rows in _iter_query_groups(source, batch_size=args.batch_size):
                query_count += 1
                max_source = max(max_source, len(rows))
                if len(rows) > int(source_manifest.get("maximum_union_candidates", 2_000)):
                    raise ValueError("v11 source query exceeds its declared candidate cap")
                metadata = _metadata(rows[0])
                fold = int(metadata["fold"])
                if any(
                    int(row.get("fold", row.get("oof_fold", -1))) != fold
                    for row in rows
                ):
                    raise ValueError("v12 query group mixes folds")
                observed_folds.add(fold)
                upstream_latency_ms = _upstream_retrieval_latency_ms(rows[0])
                cascade = cascades_by_fold.get(fold) if nested_parent_recipe else shared_cascade
                if cascade is None:
                    raise ValueError(f"v12 has no parent scorer for query fold {fold}")
                started = perf_counter()
                result = cascade.cascade(rows, query_id=query_id)
                upstream_rescue_reference_count += int(
                    result.strategy_counts.get(
                        "upstream_rescue_reference_count", 0
                    )
                )
                cascade_latency_ms = (perf_counter() - started) * 1_000.0
                if upstream_latency_ms is None:
                    missing_upstream_latency_queries += 1
                    total_latency_ms = None
                else:
                    total_latency_ms = upstream_latency_ms + cascade_latency_ms
                    upstream_latencies_ms.append(upstream_latency_ms)
                    total_latencies_ms.append(total_latency_ms)
                cascade_latencies_ms.append(cascade_latency_ms)
                metadata["retrieval_latency_ms"] = total_latency_ms
                if not result.master_references or not result.hydration_shortlist:
                    raise ValueError("v12 cannot silently drop an input query")
                master_rows = _candidate_rows(result.master_rows(query_id, metadata))
                shortlist_rows = _candidate_rows(
                    result.admission_rows(query_id, metadata)
                )
                parent_rows = [
                    {**row, "fold": metadata["fold"]}
                    for row in result.parent_rows(query_id)
                ]
                parent_candidate_rows = [
                    {**row, "fold": metadata["fold"]}
                    for row in result.parent_candidate_rows(query_id)
                ]
                master_sink.write(master_rows)
                shortlist_sink.write(shortlist_rows)
                parent_candidate_sink.write(parent_candidate_rows)
                parent_sink.write(parent_rows)
                max_master = max(max_master, len(master_rows))
                max_shortlist = max(max_shortlist, len(shortlist_rows))
                max_parent_candidates = max(
                    max_parent_candidates, len(parent_candidate_rows)
                )
                max_parents = max(max_parents, len(parent_rows))
                rescue_trigger_count += int(result.rescue_triggered)
                rescue_execution_count += int(result.rescue_executed)
        for sink in (
            master_sink,
            shortlist_sink,
            parent_candidate_sink,
            parent_sink,
        ):
            sink.close()
            if not sink.path.is_file():
                raise ValueError(f"v12 produced no rows for {sink.path.name}")
        if source_manifest.get("run_kind") in V11_LABEL_BLIND_RUN_KINDS:
            expected_nonempty_queries = int(source_manifest["queries"]) - int(
                source_manifest["empty_queries"]
            )
            if query_count != expected_nonempty_queries or max_source != int(
                source_manifest["maximum_union_candidates"]
            ):
                raise ValueError("v11 source query counts/cap do not match its manifest")
        if max_master > config.master_reference_cap:
            raise AssertionError("v12 CLI master cap exceeded")
        if max_shortlist > config.hydration_shortlist_cap:
            raise AssertionError("v12 CLI hydration cap exceeded")
        if max_parents > config.parent_shortlist_cap:
            raise AssertionError("v12 CLI parent cap exceeded")
        if max_parent_candidates > config.master_reference_cap:
            raise AssertionError("v12 CLI parent candidate cap exceeded")
        nested_manifest = (
            nested_parent_recipe.manifest_payload()
            if nested_parent_recipe is not None
            else None
        )
        source_is_sealed = source_manifest.get("run_kind") in V11_LABEL_BLIND_RUN_KINDS
        rescue_is_gate_safe = args.rescue_hits is None
        gate_eligible = bool(
            nested_parent_recipe is not None
            and observed_folds == set(ALLOWED_FOLDS)
            and source_is_sealed
            and rescue_is_gate_safe
            and upstream_rescue_reference_count == 0
            and missing_upstream_latency_queries == 0
        )
        if nested_manifest is not None:
            nested_manifest["gate_eligible"] = gate_eligible
        output_files = {
            sink.path.name: {
                "sha256": sha256_file(sink.path),
                "rows": sink.rows,
            }
            for sink in (
                master_sink,
                shortlist_sink,
                parent_candidate_sink,
                parent_sink,
            )
        }
        parent_model_hashes = (
            {
                str(fold): nested_parent_recipe.model_provenance[fold]["sha256"]
                for fold in sorted(nested_parent_recipe.model_provenance)
            }
            if nested_parent_recipe is not None
            else {}
        )
        candidate_generation = (
            {
                "mode": PARENT_MODEL_RECIPE_MODE,
                "heldout_parent_fit_folds": list(
                    nested_parent_recipe.model_provenance[
                        nested_parent_recipe.outer_fold
                    ]["fit_folds"]
                ),
                "training_fold_parent_fit_folds": {
                    str(fold): list(nested_parent_recipe.model_provenance[fold]["fit_folds"])
                    for fold in sorted(
                        ALLOWED_FOLDS.difference({nested_parent_recipe.outer_fold})
                    )
                },
            }
            if nested_parent_recipe is not None
            else {
                "mode": "SHARED_OR_BOOTSTRAP_NOT_NESTED_OOF",
                "heldout_parent_fit_folds": None,
                "training_fold_parent_fit_folds": {},
            }
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "candidate_schema_version": V12_CASCADE_SCHEMA_VERSION,
            "parent_schema_version": V12_PARENT_SCHEMA_VERSION,
            "run_kind": "LABEL_BLIND_TOP100_CASCADE",
            "source_candidate_sha256": source_candidate_sha,
            "source_manifest_sha256": source_manifest_sha,
            "config_sha256": sha256_file(args.config),
            "sidecar_manifest_sha256": sha256_file(args.sidecars / "manifest.json"),
            "parent_scorer_kind": parent_scorer_kind,
            "gate_eligible": gate_eligible,
            "bootstrap": bool(args.allow_bootstrap_parent_score),
            "outer_heldout_fold": (
                nested_parent_recipe.outer_fold if nested_parent_recipe else None
            ),
            "candidate_generation": candidate_generation,
            "parent_model_hashes": parent_model_hashes,
            "parent_model_bundle_sha256": (
                nested_parent_recipe.bundle_sha256 if nested_parent_recipe else None
            ),
            "nested_parent_oof": nested_manifest,
            "candidate_parent_scores": {
                "fields": ["parent_score", "parent_rank"],
                "parent_feature_names": list(CASCADE_PARENT_FEATURES),
                "parent_feature_values_embedded": True,
                "scorer_kind": parent_scorer_kind,
                "computed_before_parent_expansion": True,
                "complete_for_all_master_and_hydration_candidates": True,
                "sealed_by_nested_model_bundle": nested_parent_recipe is not None,
            },
            "parent_model_sha256": (
                sha256_file(args.parent_model) if args.parent_model else None
            ),
            "parent_weights_sha256": (
                sha256_file(args.parent_weights) if args.parent_weights else None
            ),
            "rescue_hits_sha256": (
                sha256_file(args.rescue_hits) if args.rescue_hits else None
            ),
            "opened_folds": sorted(observed_folds),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "human_labels_opened": False,
            "crm_alias_lookup": False,
            "query_count": query_count,
            "maximum_master_references": max_master,
            "maximum_hydration_shortlist": max_shortlist,
            "maximum_parent_candidates": max_parent_candidates,
            "maximum_parent_shortlist": max_parents,
            "master_rows": master_sink.rows,
            "hydration_rows": shortlist_sink.rows,
            "parent_candidate_rows": parent_candidate_sink.rows,
            "parent_rows": parent_sink.rows,
            "rescue_trigger_queries": rescue_trigger_count,
            "rescue_execution_queries": rescue_execution_count,
            "upstream_rescue_reference_count": upstream_rescue_reference_count,
            "retrieval_latency_ms": {
                "complete_end_to_end": missing_upstream_latency_queries == 0,
                "missing_upstream_queries": missing_upstream_latency_queries,
                "upstream": _percentiles(upstream_latencies_ms),
                "cascade": _percentiles(cascade_latencies_ms),
                "total": _percentiles(total_latencies_ms),
            },
            "files": output_files,
            "outputs": output_files,
            "views": {
                "master_references.parquet": {
                    "union_rank_semantics": "MASTER_RANK_WITHIN_MAX_2000",
                    "hydration_rank_preserved": True,
                    "parent_score_rank_preserved": True,
                    "site_lambdamart_input": True,
                    "admission_input": True,
                },
                "hydration_shortlist.parquet": {
                    "union_rank_semantics": "HEURISTIC_HYDRATION_RANK_WITHIN_MAX_500",
                    "master_rank_preserved": True,
                    "parent_score_rank_preserved": True,
                    "heuristic_only": True,
                    "admission_input": False,
                },
                "parent_shortlist.parquet": {
                    "parent_rank_semantics": "FROZEN_PRE_EXPANSION_PARENT_RANK"
                },
                "parent_candidates.parquet": {
                    "parent_rank_semantics": "FROZEN_PRE_EXPANSION_SCORE_RANK",
                    "contains_all_bounded_parent_training_candidates": True,
                    "maximum_candidates": config.master_reference_cap,
                },
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
        return output
    except Exception:
        for sink in (
            master_sink,
            shortlist_sink,
            parent_candidate_sink,
            parent_sink,
        ):
            sink.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/retrieval_all_human_v12.json",
    )
    parser.add_argument("--parent-model", type=Path)
    parser.add_argument("--parent-weights", type=Path)
    parser.add_argument(
        "--parent-model-bundle",
        type=Path,
        help=(
            "sealed six-model nested OOF bundle produced before candidate generation"
        ),
    )
    parser.add_argument(
        "--parent-outer-recipe",
        choices=("outer2", "outer3", "outer4"),
        help="outer-specific fold routing to select from --parent-model-bundle",
    )
    parser.add_argument("--rescue-hits", type=Path)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--allow-bootstrap-parent-score", action="store_true")
    parser.add_argument("--allow-unsealed-fixture", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
