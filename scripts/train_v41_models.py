#!/usr/bin/env python3
"""Train the V4.1 ranker ablation and raw query-level acceptor on fit/dev only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_v9_ranker import eligible_ranker_rows  # noqa: E402
from src.xgb_matcher.v41_acceptor import (  # noqa: E402
    V41RawLogisticAcceptor,
    fit_v41_raw_logistic_acceptor,
)
from src.xgb_matcher.v41_features import (  # noqa: E402
    V41_CANDIDATE_FEATURE_NAMES,
    build_v41_candidate_features,
    normalize_siren,
    normalize_siret,
    validate_v41_model_feature_order,
)
from src.xgb_matcher.v41_decision import v41_precheck_reason  # noqa: E402
from src.xgb_matcher.v41_release import V41ReleaseManifest  # noqa: E402
from src.xgb_matcher.v41_split import (  # noqa: E402
    assign_connected_siren_splits,
    validate_connected_siren_split,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402
from src.xgb_matcher.v9_scene import (  # noqa: E402
    V9_ACCEPTOR_EVIDENCE_BASE_FEATURE_NAMES,
    V9_SCENE_FEATURE_NAMES,
    build_query_scenes,
)


FORBIDDEN_BOUNDARY_VALUES = {"test", "holdout", "final_holdout", "final-test"}
DEFAULT_RANKER_PARAMS: dict[str, Any] = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@1",
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.035,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 3,
    "reg_lambda": 5.0,
    "n_jobs": -1,
}


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_hash(manifest: Mapping[str, Any], filename: str) -> str | None:
    value = (manifest.get("outputs") or {}).get(filename)
    if isinstance(value, Mapping):
        return str(value.get("sha256") or value.get("hash") or "") or None
    return str(value) if value else None


def _forbidden_column(column: str) -> bool:
    lowered = column.lower().replace("-", "_")
    parts = set(lowered.split("_"))
    return "holdout" in parts or "test" in parts


def _assert_fit_dev_only(frame: pd.DataFrame, *, name: str) -> None:
    forbidden_columns = [column for column in frame if _forbidden_column(str(column))]
    if forbidden_columns:
        raise ValueError(
            f"{name} contains forbidden test/holdout columns: {forbidden_columns}"
        )
    for column in ("split", "partition", "subset"):
        if column not in frame:
            continue
        values = frame[column].dropna().astype(str).str.strip().str.lower()
        forbidden = sorted(set(values).intersection(FORBIDDEN_BOUNDARY_VALUES))
        if forbidden:
            raise ValueError(
                f"{name}.{column} contains forbidden final boundary values: {forbidden}"
            )


def load_v41_canonical_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate a hash-addressed V4.1 dataset without final data."""
    manifest_path = dataset_dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    required_manifest = {
        "build_id",
        "retrieval_signature",
        "feature_order",
        "positive_injection",
        "outputs",
    }
    missing_manifest = required_manifest - set(manifest)
    if missing_manifest:
        raise ValueError(f"Manifest fields missing: {sorted(missing_manifest)}")
    if manifest["positive_injection"] is not False:
        raise ValueError("V4.1 training requires positive_injection=false")
    forbidden_manifest_keys = [
        str(key) for key in manifest if _forbidden_column(str(key))
    ]
    forbidden_outputs = [
        str(key)
        for key in (manifest.get("outputs") or {})
        if _forbidden_column(str(key))
    ]
    if forbidden_manifest_keys or forbidden_outputs:
        raise ValueError(
            "Manifest exposes a forbidden test/holdout boundary: "
            f"{forbidden_manifest_keys + forbidden_outputs}"
        )

    frames: dict[str, pd.DataFrame] = {}
    for filename in ("queries.parquet", "labels.parquet", "candidates.parquet"):
        path = dataset_dir / filename
        expected_hash = _manifest_hash(manifest, filename)
        if not expected_hash or file_sha256(path) != expected_hash:
            raise ValueError(f"Manifest hash mismatch: {filename}")
        frame = pd.read_parquet(path)
        _assert_fit_dev_only(frame, name=filename)
        frames[filename] = frame

    queries = frames["queries.parquet"].copy()
    labels = frames["labels.parquet"].copy()
    candidates = frames["candidates.parquet"].copy()
    required_queries = {
        "query_id",
        "input_siret",
        "input_siren",
        "input_siret_state",
    }
    required_labels = {
        "query_id",
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
    }
    required_candidates = {
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "is_ground_truth",
    }
    for name, frame, required in (
        ("queries", queries, required_queries),
        ("labels", labels, required_labels),
        ("candidates", candidates, required_candidates),
    ):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} columns missing: {sorted(missing)}")
    if queries["query_id"].astype(str).duplicated().any():
        raise ValueError("queries.query_id must be unique")
    if labels["query_id"].astype(str).duplicated().any():
        raise ValueError("labels.query_id must be unique")
    if set(queries["query_id"].astype(str)) != set(labels["query_id"].astype(str)):
        raise ValueError("queries and labels must contain exactly the same query IDs")
    unknown_candidate_queries = set(candidates["query_id"].astype(str)) - set(
        queries["query_id"].astype(str)
    )
    if unknown_candidate_queries:
        raise ValueError("Candidates reference unknown query IDs")
    if candidates.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("Candidate query/SIRET pairs must be unique")
    counts = candidates.groupby(candidates["query_id"].astype(str)).size()
    if not counts.empty and int(counts.max()) > 100:
        raise ValueError("The V4.1 candidate cap is 100")

    source_columns = [
        column
        for column in ("retrieval_source", "prediction_origin", "provenance")
        if column in candidates
    ]
    for column in source_columns:
        injected = candidates[column].fillna("").astype(str).str.lower().str.contains(
            "inject"
        )
        if injected.any():
            raise ValueError(f"Positive injection marker found in candidates.{column}")
    for column in ("gt_was_injected", "positive_injected", "is_injected"):
        if column in candidates and candidates[column].fillna(False).astype(bool).any():
            raise ValueError(f"Positive injection flag found in candidates.{column}")

    queries["query_id"] = queries["query_id"].astype(str)
    labels["query_id"] = labels["query_id"].astype(str)
    candidates["query_id"] = candidates["query_id"].astype(str)
    queries["input_siret"] = queries["input_siret"].map(normalize_siret)
    queries["input_siren"] = queries["input_siren"].map(normalize_siren)
    derived_input_siren = queries["input_siret"].map(
        lambda value: value[:9] if value else None
    )
    inconsistent_input = (
        queries["input_siren"].notna()
        & derived_input_siren.notna()
        & queries["input_siren"].ne(derived_input_siren)
    )
    if inconsistent_input.any():
        raise ValueError("input_siren must equal the first 9 digits of input_siret")
    queries["input_siren"] = queries["input_siren"].fillna(derived_input_siren)
    labels["ground_truth_siret"] = labels["ground_truth_siret"].map(normalize_siret)
    labels["ground_truth_siren"] = [
        normalize_siren(siren) or (siret[:9] if siret else None)
        for siren, siret in zip(
            labels["ground_truth_siren"],
            labels["ground_truth_siret"],
            strict=True,
        )
    ]
    allowed_label_kinds = {"MATCH_EXACT", "NO_MATCH", "AMBIGUOUS", "UNRESOLVED"}
    invalid_label_kinds = set(labels["label_kind"].astype(str)) - allowed_label_kinds
    if invalid_label_kinds:
        raise ValueError(f"Unsupported label kinds: {sorted(invalid_label_kinds)}")
    exact = labels["label_kind"].astype(str).eq("MATCH_EXACT")
    if labels.loc[exact, "ground_truth_siret"].isna().any():
        raise ValueError("MATCH_EXACT labels require a valid ground_truth_siret")
    if labels.loc[~exact, "ground_truth_siret"].notna().any():
        raise ValueError("Only MATCH_EXACT labels may carry ground_truth_siret")
    candidates["candidate_siret"] = candidates["candidate_siret"].map(normalize_siret)
    if candidates["candidate_siret"].isna().any():
        raise ValueError("Every candidate requires a valid 14-digit SIRET")
    candidates["candidate_siren"] = candidates["candidate_siret"].str[:9]

    truth = labels.set_index("query_id")["ground_truth_siret"]
    expected_positive = (
        candidates["candidate_siret"]
        == candidates["query_id"].map(truth)
    ).astype(int)
    observed_positive = candidates["is_ground_truth"].fillna(0).astype(int)
    if not expected_positive.equals(observed_positive):
        raise ValueError(
            "is_ground_truth must describe retrieved rows exactly; no repair or "
            "positive injection is allowed"
        )

    legacy_features = [
        str(feature)
        for feature in manifest["feature_order"]
        if str(feature) not in V41_CANDIDATE_FEATURE_NAMES
    ]
    if not legacy_features:
        raise ValueError("Manifest feature_order contains no legacy R0 features")
    missing_features = set(legacy_features) - set(candidates)
    if missing_features:
        raise ValueError(f"Candidate features missing: {sorted(missing_features)}")
    validate_v41_model_feature_order(
        legacy_features,
        require_v41_features=False,
    )
    return manifest, queries, labels, candidates


def prepare_v41_training_frames(
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_source = queries[
        ["query_id", "input_siret", "input_siren"]
    ].merge(
        labels[
            ["query_id", "ground_truth_siret", "ground_truth_siren"]
        ],
        on="query_id",
        validate="one_to_one",
    ).rename(
        columns={
            "ground_truth_siret": "target_siret",
            "ground_truth_siren": "target_siren",
        }
    )
    assignments = assign_connected_siren_splits(
        split_source,
        dev_fraction=0.2,
        oof_folds=5,
        seed=seed,
    )[["query_id", "siren_component_id", "split", "oof_fold"]]
    validate_connected_siren_split(assignments)

    query_output = queries.drop(
        columns=[
            column
            for column in ("split", "oof_fold", "siren_component_id")
            if column in queries
        ]
    ).merge(assignments, on="query_id", validate="one_to_one")
    label_output = labels.drop(
        columns=[
            column
            for column in ("split", "oof_fold", "siren_component_id")
            if column in labels
        ]
    ).merge(assignments, on="query_id", validate="one_to_one")
    candidate_output = candidates.drop(
        columns=[
            column
            for column in ("split", "oof_fold", "siren_component_id")
            if column in candidates
        ]
    ).merge(assignments, on="query_id", validate="many_to_one")

    input_by_query = query_output.set_index("query_id")["input_siret"]
    additions = [
        build_v41_candidate_features(
            row,
            input_siret=input_by_query.get(str(row["query_id"])),
        )
        for row in candidate_output.to_dict("records")
    ]
    for feature in V41_CANDIDATE_FEATURE_NAMES:
        candidate_output[feature] = [row[feature] for row in additions]
    return query_output, label_output, candidate_output


def _ranker(
    *,
    seed: int,
    ranker_params: Mapping[str, Any] | None = None,
) -> xgb.XGBRanker:
    params = {**DEFAULT_RANKER_PARAMS, **dict(ranker_params or {})}
    return xgb.XGBRanker(random_state=seed, **params)


def _fit_ranker(
    rows: pd.DataFrame,
    feature_order: Sequence[str],
    *,
    seed: int,
    ranker_params: Mapping[str, Any] | None,
) -> xgb.XGBRanker:
    ordered = rows.sort_values(["query_id", "candidate_siret"]).copy()
    groups = ordered.groupby("query_id", sort=False).size().to_numpy()
    model = _ranker(seed=seed, ranker_params=ranker_params)
    model.fit(
        ordered[list(feature_order)].astype(float).to_numpy(),
        ordered["is_ground_truth"].astype(int).to_numpy(),
        group=groups,
        verbose=False,
    )
    return model


def _score_rows(
    model: xgb.XGBRanker,
    rows: pd.DataFrame,
    feature_order: Sequence[str],
    *,
    origin: str,
    fold: int | None,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    passthrough = [
        column
        for column in (
            "query_id",
            "candidate_siret",
            "candidate_siren",
            "candidate_state",
            "etat_admin",
            "retrieval_rank",
            "retrieval_source",
            "retrieval_channel_count",
            "retrieval_agreement",
            "sparse_rank",
            "dense_rank",
            "rrf_score",
            *V41_CANDIDATE_FEATURE_NAMES,
            *V9_ACCEPTOR_EVIDENCE_BASE_FEATURE_NAMES,
        )
        if column in rows
    ]
    output = rows[passthrough].copy()
    output["score"] = model.predict(rows[list(feature_order)].astype(float).to_numpy())
    output["prediction_origin"] = origin
    output["fold"] = fold
    output["rank"] = (
        output.groupby("query_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return output


def train_ranker_variant(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_order: Sequence[str],
    seed: int,
    ranker_params: Mapping[str, Any] | None = None,
) -> tuple[xgb.XGBRanker, pd.DataFrame]:
    """Produce component-safe OOF fit predictions and out-of-sample dev predictions."""
    fit_labels = labels[labels["split"].eq("fit")]
    fit_candidates = candidates[candidates["split"].eq("fit")]
    predictions: list[pd.DataFrame] = []
    for fold in range(5):
        valid_ids = set(
            fit_labels.loc[fit_labels["oof_fold"].eq(fold), "query_id"]
        )
        if not valid_ids:
            continue
        train_labels = fit_labels[fit_labels["oof_fold"].ne(fold)]
        train_ids = set(train_labels["query_id"])
        train_rows = eligible_ranker_rows(
            fit_candidates[fit_candidates["query_id"].isin(train_ids)],
            train_labels,
        )
        if train_rows.empty:
            raise ValueError(f"OOF fold {fold} has no eligible ranker fit rows")
        fold_model = _fit_ranker(
            train_rows,
            feature_order,
            seed=seed + fold,
            ranker_params=ranker_params,
        )
        predictions.append(
            _score_rows(
                fold_model,
                fit_candidates[fit_candidates["query_id"].isin(valid_ids)],
                feature_order,
                origin="oof",
                fold=fold,
            )
        )

    final_fit_rows = eligible_ranker_rows(fit_candidates, fit_labels)
    if final_fit_rows.empty:
        raise ValueError("No eligible ranker fit rows")
    final_model = _fit_ranker(
        final_fit_rows,
        feature_order,
        seed=seed,
        ranker_params=ranker_params,
    )
    predictions.append(
        _score_rows(
            final_model,
            candidates[candidates["split"].eq("dev")],
            feature_order,
            origin="out_of_sample_dev",
            fold=None,
        )
    )
    nonempty = [part for part in predictions if not part.empty]
    return final_model, (
        pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    )


def _query_hits(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    exact = labels[labels["label_kind"].eq("MATCH_EXACT")].copy()
    if predictions.empty:
        return pd.Series(False, index=exact["query_id"].astype(str))
    top1 = (
        predictions.sort_values(["query_id", "rank"])
        .drop_duplicates("query_id")
        .set_index("query_id")["candidate_siret"]
    )
    truth = exact.set_index("query_id")["ground_truth_siret"]
    return truth.index.to_series().map(top1).fillna("").eq(truth.fillna(""))


def compare_ranker_variants(
    r0_predictions: pd.DataFrame,
    r1_predictions: pd.DataFrame,
    labels: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    segment_columns: Sequence[str],
) -> dict[str, Any]:
    dev_labels = labels[labels["split"].eq("dev")]
    r0_hits = _query_hits(r0_predictions, dev_labels)
    r1_hits = _query_hits(r1_predictions, dev_labels).reindex(r0_hits.index, fill_value=False)
    segments: dict[str, Any] = {}
    query_index = queries.set_index("query_id")
    deltas: list[float] = []
    for column in segment_columns:
        if column not in query_index:
            continue
        values = query_index[column].reindex(r0_hits.index).fillna("UNKNOWN").astype(str)
        for value in sorted(values.unique()):
            mask = values.eq(value)
            r0_rate = float(r0_hits[mask].mean())
            r1_rate = float(r1_hits[mask].mean())
            delta = r1_rate - r0_rate
            segments[f"{column}={value}"] = {
                "count": int(mask.sum()),
                "r0_hit_at_1": r0_rate,
                "r1_hit_at_1": r1_rate,
                "delta": delta,
            }
            deltas.append(delta)
    r0_rate = float(r0_hits.mean()) if len(r0_hits) else 0.0
    r1_rate = float(r1_hits.mean()) if len(r1_hits) else 0.0
    max_regression = max([0.0, *[-delta for delta in deltas]])
    checks = {
        "r1_not_below_r0": r1_rate >= r0_rate,
        "r1_hit_at_1_at_least_96pct": r1_rate >= 0.96,
        "no_segment_regression_over_2pp": max_regression <= 0.02,
    }
    return {
        "promote_r1": all(checks.values()),
        "selected_variant": "R1" if all(checks.values()) else "R0",
        "checks": checks,
        "r0_hit_at_1": r0_rate,
        "r1_hit_at_1": r1_rate,
        "max_segment_regression": max_regression,
        "segments": segments,
        "exact_dev_count": int(len(r0_hits)),
    }


def _append_missing_prediction_sentinels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    predicted_ids = set(predictions["query_id"].astype(str)) if not predictions.empty else set()
    missing = labels[~labels["query_id"].astype(str).isin(predicted_ids)]
    if missing.empty:
        return predictions
    sentinels = pd.DataFrame(
        {
            "query_id": missing["query_id"].astype(str),
            "candidate_siret": None,
            "candidate_siren": None,
            "score": 0.0,
            "prediction_origin": missing["split"].map(
                {"fit": "oof", "dev": "out_of_sample_dev"}
            ),
            "fold": missing["oof_fold"],
            "rank": 1,
        }
    )
    return pd.concat([predictions, sentinels], ignore_index=True)


def annotate_v41_prechecks(
    scenes: pd.DataFrame,
    predictions: pd.DataFrame,
    queries: pd.DataFrame,
) -> pd.DataFrame:
    """Keep every scene, and mark those deterministically routed to REVIEW."""
    query_by_id = queries.set_index("query_id")
    prediction_groups = {
        str(query_id): group.to_dict("records")
        for query_id, group in predictions.groupby("query_id", sort=False)
    }
    reasons: list[str | None] = []
    for query_id in scenes["query_id"].astype(str):
        query = query_by_id.loc[query_id]
        reason = v41_precheck_reason(
            input_siret=query.get("input_siret"),
            input_siret_state=str(query.get("input_siret_state") or "UNKNOWN"),
            candidates=prediction_groups.get(query_id, []),
        )
        reasons.append(reason.value if reason is not None else None)
    output = scenes.copy()
    output["precheck_review_reason"] = reasons
    output["acceptor_eligible"] = output["precheck_review_reason"].isna()
    return output


def train_v41_models(
    *,
    dataset_dir: Path,
    output_dir: Path,
    ranker_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the frozen V4.1 fit/dev training workflow."""
    if output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {output_dir}")
    manifest, queries, labels, candidates = load_v41_canonical_dataset(dataset_dir)
    queries, labels, candidates = prepare_v41_training_frames(
        queries,
        labels,
        candidates,
        seed=42,
    )
    legacy_features = [
        str(feature)
        for feature in manifest["feature_order"]
        if str(feature) not in V41_CANDIDATE_FEATURE_NAMES
    ]
    r1_features = [*legacy_features, *V41_CANDIDATE_FEATURE_NAMES]
    validate_v41_model_feature_order(r1_features)

    r0_model, r0_predictions = train_ranker_variant(
        candidates,
        labels,
        feature_order=legacy_features,
        seed=42,
        ranker_params=ranker_params,
    )
    r1_model, r1_predictions = train_ranker_variant(
        candidates,
        labels,
        feature_order=r1_features,
        seed=42,
        ranker_params=ranker_params,
    )
    segment_columns = list(
        manifest.get("critical_segment_columns") or ["input_siret_state"]
    )
    comparison = compare_ranker_variants(
        r0_predictions,
        r1_predictions,
        labels,
        queries,
        segment_columns=segment_columns,
    )
    selected_variant = comparison["selected_variant"]
    selected_model = r1_model if selected_variant == "R1" else r0_model
    selected_features = r1_features if selected_variant == "R1" else legacy_features
    selected_predictions = (
        r1_predictions if selected_variant == "R1" else r0_predictions
    )
    selected_predictions = _append_missing_prediction_sentinels(
        selected_predictions,
        labels,
    )

    scene_labels = labels.rename(columns={"split": "_split"}).copy()
    scene_labels["split"] = scene_labels.pop("_split")
    scenes = build_query_scenes(selected_predictions, scene_labels)
    scenes["ranker_prediction_is_out_of_sample"] = True
    scenes = annotate_v41_prechecks(
        scenes,
        selected_predictions,
        queries,
    )
    acceptor_scenes = scenes[scenes["acceptor_eligible"]].copy()
    if acceptor_scenes.empty:
        raise ValueError("No scene remains eligible for the V4.1 acceptor")
    acceptor_dataset_identity = {
        "ranker_dataset_manifest_id": str(manifest["build_id"]),
        "retrieval_signature": str(manifest["retrieval_signature"]),
        "ranker_variant": selected_variant,
        "ranker_feature_order": selected_features,
        "scene_feature_order": V9_SCENE_FEATURE_NAMES,
        "query_ids": sorted(acceptor_scenes["query_id"].astype(str)),
    }
    acceptor_dataset_manifest_id = hashlib.sha256(
        json.dumps(acceptor_dataset_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    acceptor, acceptor_report = fit_v41_raw_logistic_acceptor(
        acceptor_scenes,
        feature_order=V9_SCENE_FEATURE_NAMES,
        dataset_manifest_id=acceptor_dataset_manifest_id,
        retrieval_signature=str(manifest["retrieval_signature"]),
        target_precision=0.998,
        min_auto_count=100,
        seed=42,
    )

    ranker_identity = {
        "dataset_manifest_id": str(manifest["build_id"]),
        "retrieval_signature": str(manifest["retrieval_signature"]),
        "feature_order": selected_features,
        "ranker_variant": selected_variant,
        "seed": 42,
    }
    ranker_bundle_id = hashlib.sha256(
        json.dumps(ranker_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    ranker_metadata = {
        "schema_version": "v4.1-ranker-1",
        "model_bundle_id": ranker_bundle_id,
        **ranker_identity,
        "folds": 5,
        "fold_group": "connected_component(input_siren,target_siren)",
        "positive_injection": False,
        "fit_scene_prediction_origin": "oof",
        "dev_scene_prediction_origin": "out_of_sample_dev",
        "comparison": comparison,
    }
    acceptor_metadata = {
        "model_bundle_id": acceptor.model_bundle_id,
        "dataset_manifest_id": acceptor.dataset_manifest_id,
        "retrieval_signature": acceptor.retrieval_signature,
        "feature_order": acceptor.feature_order,
        "calibration_method": acceptor.calibration_method,
        "confidence_kind": acceptor.confidence_kind,
    }
    release = V41ReleaseManifest.build(
        retrieval_signature=str(manifest["retrieval_signature"]),
        ranker_bundle_id=ranker_bundle_id,
        acceptor_bundle_id=acceptor.model_bundle_id,
        ranker_dataset_manifest_id=str(manifest["build_id"]),
        acceptor_dataset_manifest_id=acceptor_dataset_manifest_id,
        ranker_feature_order=selected_features,
        acceptor_feature_order=acceptor.feature_order,
        ranker_variant=selected_variant,
    )
    release.validate_components(
        ranker_metadata=ranker_metadata,
        acceptor_metadata=acceptor_metadata,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    ranker_dir = output_dir / "ranker"
    ranker_dir.mkdir()
    selected_model.save_model(ranker_dir / "ranker.json")
    (ranker_dir / "metadata.json").write_text(
        json.dumps(ranker_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    acceptor.save(output_dir / "acceptor")
    selected_predictions.to_parquet(
        output_dir / "ranker_predictions.parquet",
        index=False,
    )
    scenes.to_parquet(output_dir / "acceptor_scenes.parquet", index=False)
    assignments = queries[
        ["query_id", "siren_component_id", "split", "oof_fold"]
    ]
    assignments.to_parquet(output_dir / "split_assignments.parquet", index=False)
    release.save(output_dir / "release_manifest.json")
    report = {
        "schema_version": "v4.1-training-report-1",
        "source_dataset_manifest_id": str(manifest["build_id"]),
        "acceptor_dataset_manifest_id": acceptor_dataset_manifest_id,
        "retrieval_signature": str(manifest["retrieval_signature"]),
        "split": {
            "strategy": "connected_component(input_siren,target_siren)",
            "fit_count": int(queries["split"].eq("fit").sum()),
            "dev_count": int(queries["split"].eq("dev").sum()),
            "dev_fraction_target": 0.2,
            "folds": 5,
            "seed": 42,
        },
        "ranker_comparison": comparison,
        "acceptor": acceptor_report,
        "prechecks": {
            "scene_count": int(len(scenes)),
            "acceptor_eligible_count": int(len(acceptor_scenes)),
            "review_reason_counts": scenes[
                "precheck_review_reason"
            ].value_counts(dropna=True).to_dict(),
        },
        "positive_injection": False,
        "test_or_holdout_consumed": False,
        "release_id": release.release_id,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_v41_models(
        dataset_dir=args.dataset,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
