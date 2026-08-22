#!/usr/bin/env python3
"""Train, cross-fit or run the all-human hierarchical admission v11."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_hierarchical_admission_v11 import (  # noqa: E402
    HierarchicalAdmissionConfig,
    HierarchicalModels,
    PARENT_FEATURES,
    SCHEMA_VERSION,
    SIRET_FEATURES,
    canonicalize_union,
    evaluate_all_human,
    fit_outer_oof_models,
    fit_hierarchical_models,
    outer_oof_evaluate,
    predict_top100,
    score_canonical_candidates,
    select_top100,
    summarize_outcomes,
)
from scripts.evaluate_retrieval_ltr_streaming import (  # noqa: E402
    iter_complete_query_frames,
)


DEFAULT_CONFIG = ROOT / "config/retrieval_all_human_v11.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_candidate_manifest(
    path: Path, *, require_labels_attached: bool
) -> tuple[Path, dict[str, Any]]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"A zero-injection candidate manifest is required: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("candidate_parquet_sha256")
    observed = sha256_file(path)
    if expected != observed:
        raise ValueError("Candidate manifest hash mismatch")
    if manifest.get("positive_injection") is not False:
        raise ValueError("Candidate manifest does not prove zero positive injection")
    if int(manifest.get("maximum_union_candidates", 2000)) > 2000:
        raise ValueError("Candidate manifest exceeds the full-union cap")
    if require_labels_attached:
        if manifest.get("labels_attached_after_retrieval") is not True:
            raise ValueError("OOF/fit requires the sealed post-retrieval label join")
        if manifest.get("all_label_queries_have_natural_candidates") is not True:
            raise ValueError("OOF/fit cannot drop empty human-labelled queries")
    elif manifest.get("human_labels_opened_by_retrieval") is not False:
        raise ValueError("Prediction requires a label-blind retrieval manifest")
    return manifest_path, manifest


def _write_json(path: Path, value: MappingLike) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


MappingLike = dict[str, Any]


def _prepare_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)


def _natural_positive_mask(group: pd.DataFrame) -> pd.Series:
    truth = str(group.iloc[0].get("human_ground_truth_siret") or "")
    candidate_column = "siret" if "siret" in group else "candidate_siret"
    return group[candidate_column].astype(str).eq(truth)


def _fit_sample(group: pd.DataFrame, config: HierarchicalAdmissionConfig) -> pd.DataFrame:
    sampling = config.fit_candidate_sampling
    cap = int(sampling["maximum_candidates_per_query"])
    if len(group) <= cap:
        return group
    union_head = int(sampling["union_head"])
    channel_head = int(sampling["channel_head"])
    selected: set[int] = set(
        pd.to_numeric(group["union_rank"], errors="coerce")
        .nsmallest(union_head)
        .index.tolist()
    )
    channel_selected: set[int] = set()
    for column in group.columns:
        if not column.endswith("_rank") or column in {"union_rank", "retrieval_rank"}:
            continue
        ranks = pd.to_numeric(group[column], errors="coerce")
        channel_selected.update(ranks.dropna().nsmallest(channel_head).index.tolist())
    selected.update(channel_selected)
    positive = set(group.index[_natural_positive_mask(group)].tolist())
    selected.update(positive)
    if len(selected) < cap:
        fill = (
            pd.to_numeric(group["union_rank"], errors="coerce")
            .drop(index=list(selected), errors="ignore")
            .nsmallest(cap - len(selected))
        )
        selected.update(fill.index.tolist())
    candidates = group.loc[sorted(selected)].copy()
    candidates["__positive"] = candidates.index.isin(positive)
    candidates["__channel_head"] = candidates.index.isin(channel_selected)
    candidates["__rank"] = pd.to_numeric(
        candidates["union_rank"], errors="coerce"
    ).fillna(2_147_483_647)
    candidates = candidates.sort_values(
        ["__positive", "__channel_head", "__rank"],
        ascending=[False, False, True],
        kind="stable",
    ).head(cap)
    return candidates.drop(
        columns=["__positive", "__channel_head", "__rank"]
    ).sort_values("union_rank", kind="stable")


def _load_fit_sample(
    path: Path, config: HierarchicalAdmissionConfig, *, query_batch_size: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    chunks: list[pd.DataFrame] = []
    groups: list[pd.DataFrame] = []
    before = after = queries = positives = 0

    def flush() -> None:
        nonlocal positives
        if not groups:
            return
        canonical = canonicalize_union(
            pd.concat(groups, ignore_index=True),
            config,
            labels_allowed=True,
            require_labels=True,
        )
        positives += int(
            canonical["candidate_siret"].eq(canonical["human_siret"]).sum()
        )
        chunks.append(canonical)
        groups.clear()

    for group in iter_complete_query_frames(path, batch_size=131_072):
        fold = int(group.iloc[0]["fold"])
        if fold not in config.train_folds:
            raise ValueError("Fit sample opened a non-training fold")
        queries += 1
        before += len(group)
        sampled = _fit_sample(group, config)
        after += len(sampled)
        groups.append(sampled)
        if len(groups) >= query_batch_size:
            flush()
    flush()
    if not chunks:
        raise ValueError("No sampled training candidates")
    return pd.concat(chunks, ignore_index=True), {
        "queries": queries,
        "rows_before": before,
        "rows_after": after,
        "natural_positives_retained": positives,
    }


def _selected_projection(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "query_id",
        "fold",
        "candidate_siret",
        "candidate_parent",
        "union_rank",
        "selected_rank",
        "final_score",
        "parent_rank",
        "site_rank",
        "selection_reason",
        "channels_json",
        "official_evidence_ids_json",
        "official_evidence_sources_json",
    ]
    return frame[columns].copy()


def _stream_outer_validation(
    path: Path,
    models: MappingLike,
    config: HierarchicalAdmissionConfig,
    selected_path: Path,
    *,
    query_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    writer: pq.ParquetWriter | None = None
    groups: list[pd.DataFrame] = []
    outcome_chunks: list[pd.DataFrame] = []
    fold_queries = {fold: 0 for fold in config.train_folds}
    max_union = 0

    def flush() -> None:
        nonlocal writer, max_union
        if not groups:
            return
        canonical = canonicalize_union(
            pd.concat(groups, ignore_index=True),
            config,
            labels_allowed=True,
            require_labels=True,
        )
        selected_parts: list[pd.DataFrame] = []
        for fold in config.train_folds:
            validation = canonical[canonical["fold"].eq(fold)].copy()
            if validation.empty:
                continue
            fold_queries[fold] += int(validation["query_id"].nunique())
            max_union = max(max_union, int(validation.groupby("query_id").size().max()))
            blind = validation.copy()
            blind["human_siret"] = ""
            selected = select_top100(
                score_canonical_candidates(models[fold], blind), config
            )
            outcomes, _ = evaluate_all_human(validation, selected, config)
            outcome_chunks.append(outcomes)
            selected_parts.append(_selected_projection(selected))
        if selected_parts:
            table = pa.Table.from_pandas(
                pd.concat(selected_parts, ignore_index=True),
                schema=writer.schema if writer else None,
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(selected_path, table.schema, compression="zstd")
            writer.write_table(table)
        groups.clear()

    try:
        for group in iter_complete_query_frames(path, batch_size=131_072):
            groups.append(group)
            if len(groups) >= query_batch_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    if writer is None or not outcome_chunks:
        raise ValueError("Streaming OOF produced no validation output")
    return pd.concat(outcome_chunks, ignore_index=True), {
        "queries_by_fold": {str(key): value for key, value in fold_queries.items()},
        "maximum_validation_union": max_union,
        "validation_candidate_sampling": False,
    }


def run_oof(args: argparse.Namespace) -> Path:
    config = HierarchicalAdmissionConfig.load(args.config)
    source_manifest_path, _ = verify_candidate_manifest(
        args.candidates, require_labels_attached=True
    )
    _prepare_output(args.output_dir)
    selected_path = args.output_dir / "selected_candidates.parquet"
    outcomes_path = args.output_dir / "query_outcomes.parquet"
    summary_path = args.output_dir / "summary.json"
    sampled, sample_diagnostics = _load_fit_sample(
        args.candidates,
        config,
        query_batch_size=args.fit_query_batch_size,
    )
    models, model_diagnostics = fit_outer_oof_models(sampled, config)
    model_paths: list[Path] = []
    for fold, fold_models in models.items():
        parent_path = args.output_dir / f"fold{fold}_parent_ranker.json"
        siret_path = args.output_dir / f"fold{fold}_siret_ranker.json"
        fold_models.parent.save_model(parent_path)
        fold_models.siret.save_model(siret_path)
        model_paths.extend((parent_path, siret_path))
    del sampled
    outcomes, stream_diagnostics = _stream_outer_validation(
        args.candidates,
        models,
        config,
        selected_path,
        query_batch_size=args.validation_query_batch_size,
    )
    summary = summarize_outcomes(outcomes, config)
    summary["nested_oof"] = {
        "outer_folds": list(config.train_folds),
        "model_diagnostics": model_diagnostics,
        "fit_sample": sample_diagnostics,
        "stream_validation": stream_diagnostics,
        "fold0_opened": False,
        "fold1_opened": False,
        "validation_candidate_sampling": False,
    }
    outcomes.to_parquet(outcomes_path, index=False, compression="zstd")
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_kind": "NESTED_OUTER_OOF",
        "policy_id": config.policy_id,
        "source_candidates_sha256": sha256_file(args.candidates),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "config_sha256": sha256_file(args.config),
        "opened_folds": list(config.train_folds),
        "fold0_opened": False,
        "fold1_opened": False,
        "validation_candidate_sampling": False,
        "positive_injection": False,
        "outputs": {
            path.name: sha256_file(path)
            for path in (selected_path, outcomes_path, summary_path, *model_paths)
        },
        "verdict": summary["verdict"],
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return args.output_dir


def _save_models(
    models: HierarchicalModels,
    output_dir: Path,
    config: HierarchicalAdmissionConfig,
    config_path: Path,
    candidates_path: Path,
    candidate_manifest_path: Path,
) -> None:
    parent_path = output_dir / "parent_ranker.json"
    siret_path = output_dir / "siret_ranker.json"
    metadata_path = output_dir / "metadata.json"
    models.parent.save_model(parent_path)
    models.siret.save_model(siret_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": config.policy_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_folds": list(config.train_folds),
        "parent_features": list(PARENT_FEATURES),
        "siret_features": list(SIRET_FEATURES),
        "diagnostics": dict(models.diagnostics),
        "human_labels_used_by": "TRAINER_ONLY",
        "positive_injection": False,
        "fold0_opened": False,
        "fold1_opened": False,
    }
    _write_json(metadata_path, metadata)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_kind": "FINAL_TRAIN_234",
        "config_sha256": sha256_file(config_path),
        "source_candidates_sha256": sha256_file(candidates_path),
        "source_manifest_sha256": sha256_file(candidate_manifest_path),
        "opened_folds": list(config.train_folds),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "outputs": {
            path.name: sha256_file(path)
            for path in (parent_path, siret_path, metadata_path)
        },
    }
    _write_json(output_dir / "manifest.json", manifest)


def run_fit(args: argparse.Namespace) -> Path:
    config = HierarchicalAdmissionConfig.load(args.config)
    source_manifest_path, _ = verify_candidate_manifest(
        args.candidates, require_labels_attached=True
    )
    _prepare_output(args.output_dir)
    frame = pd.read_parquet(args.candidates)
    models = fit_hierarchical_models(frame, config)
    _save_models(
        models,
        args.output_dir,
        config,
        args.config,
        args.candidates,
        source_manifest_path,
    )
    return args.output_dir


def _load_models(
    model_dir: Path, config: HierarchicalAdmissionConfig, config_path: Path
) -> HierarchicalModels:
    manifest_path = model_dir / "manifest.json"
    metadata_path = model_dir / "metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Model and inference configurations differ")
    if metadata.get("parent_features") != list(PARENT_FEATURES):
        raise ValueError("Parent model feature contract differs")
    if metadata.get("siret_features") != list(SIRET_FEATURES):
        raise ValueError("SIRET model feature contract differs")
    for filename in ("parent_ranker.json", "siret_ranker.json", "metadata.json"):
        if manifest.get("outputs", {}).get(filename) != sha256_file(model_dir / filename):
            raise ValueError(f"Model artifact hash mismatch: {filename}")
    parent = xgb.XGBRanker(**config.xgb_params("parent"))
    parent.load_model(model_dir / "parent_ranker.json")
    siret = xgb.XGBRanker(**config.xgb_params("siret"))
    siret.load_model(model_dir / "siret_ranker.json")
    return HierarchicalModels(parent=parent, siret=siret, diagnostics=metadata["diagnostics"])


def run_predict(args: argparse.Namespace) -> Path:
    config = HierarchicalAdmissionConfig.load(args.config)
    source_manifest_path, _ = verify_candidate_manifest(
        args.candidates, require_labels_attached=False
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    models = _load_models(args.model_dir, config, args.config)
    # predict_top100 rejects every populated human-label column.
    frame = pd.read_parquet(args.candidates)
    selected = predict_top100(models, frame, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False, compression="zstd")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_kind": "LABEL_BLIND_PREDICT",
        "source_candidates_sha256": sha256_file(args.candidates),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "model_manifest_sha256": sha256_file(args.model_dir / "manifest.json"),
        "config_sha256": sha256_file(args.config),
        "human_labels_opened": False,
        "positive_injection": False,
        "maximum_candidates": int(selected.groupby("query_id").size().max()),
        "candidate_parquet_sha256": sha256_file(args.output),
    }
    _write_json(args.output.with_suffix(args.output.suffix + ".manifest.json"), manifest)
    return args.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("oof", "fit"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--candidates", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if command == "oof":
            subparser.add_argument("--fit-query-batch-size", type=int, default=64)
            subparser.add_argument(
                "--validation-query-batch-size", type=int, default=32
            )
    predict = subparsers.add_parser("predict")
    predict.add_argument("--candidates", type=Path, required=True)
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "oof":
        output = run_oof(args)
    elif args.command == "fit":
        output = run_fit(args)
    else:
        output = run_predict(args)
    print(output)


if __name__ == "__main__":
    main()
