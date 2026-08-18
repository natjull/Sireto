#!/usr/bin/env python3
"""Build, train and evaluate the frozen lexical LambdaMART admission block.

Development commands never read fold 1.  The final ``evaluate --split test``
path requires both ``--open-test`` and a hash-bound one-shot authorization.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.retrieval_ltr_admission import (  # noqa: E402
    AdmissionConfig,
    DEFAULT_CONFIG_PATH,
    SCHEMA_VERSION,
    build_internal_union,
    evaluate_outcomes,
    feature_order,
    score_and_select,
    train_ranker,
)


RUN_SCHEMA_VERSION = "sireto-retrieval-ltr-run-v1"
TEST_AUTH_SCHEMA_VERSION = "sireto-retrieval-ltr-test-authorization-v1"
MappingLike = dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: MappingLike) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_unique_config(path: Path) -> AdmissionConfig:
    selected = AdmissionConfig.load(path)
    frozen = AdmissionConfig.load(DEFAULT_CONFIG_PATH)
    if dict(selected.raw) != dict(frozen.raw):
        raise ValueError("Only the single checked-in retrieval LTR config is allowed")
    return selected


def _read_filtered_parquet(path: Path, folds: Iterable[int]) -> pd.DataFrame:
    schema = set(pq.ParquetFile(path).schema.names)
    if "fold" not in schema:
        raise ValueError("Input parquet has no fold column")
    allowed = sorted(set(int(value) for value in folds))
    return pd.read_parquet(path, filters=[("fold", "in", allowed)])


def _new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable output already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _manifest_base(
    *,
    command: str,
    config_path: Path,
    test_opened: bool,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "admission_schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "python": platform.python_version(),
        "xgboost": xgb.__version__,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "test_opened": test_opened,
        "positive_injection": False,
        "contains_synthetic": False,
        "contains_dense": False,
    }


def _verify_manifest_output(path: Path) -> dict[str, Any]:
    manifest_path = path.parent / "manifest.json"
    manifest = _read_json(manifest_path)
    expected = manifest.get("outputs", {}).get(path.name)
    if expected != file_sha256(path):
        raise ValueError(f"Manifest hash mismatch: {path}")
    return manifest


def build_union_command(args: argparse.Namespace) -> Path:
    config = _load_unique_config(args.config)
    if args.scope == "train":
        folds = config.train_folds
    elif args.scope == "development":
        folds = (*config.train_folds, config.dev_fold)
    else:  # pragma: no cover - argparse guards this
        raise ValueError("Unsupported build scope")
    # There is intentionally no test scope here.
    raw = _read_filtered_parquet(args.candidates, folds)
    union, diagnostics = build_internal_union(raw, config, allowed_folds=folds)
    if config.test_fold in set(union["fold"].astype(int)):
        raise AssertionError("Fold 1 reached a development union")

    _new_output(args.output_dir)
    union_path = args.output_dir / "union.parquet"
    diagnostics_path = args.output_dir / "union_diagnostics.json"
    union.to_parquet(union_path, index=False, compression="zstd")
    _write_json(diagnostics_path, diagnostics)
    manifest = {
        **_manifest_base(
            command=f"build-union:{args.scope}",
            config_path=args.config,
            test_opened=False,
        ),
        "opened_folds": sorted(set(union["fold"].astype(int))),
        "inputs": {str(args.candidates.resolve()): file_sha256(args.candidates)},
        "outputs": {
            union_path.name: file_sha256(union_path),
            diagnostics_path.name: file_sha256(diagnostics_path),
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return args.output_dir


def train_command(args: argparse.Namespace) -> Path:
    config = _load_unique_config(args.config)
    union_manifest = _verify_manifest_output(args.union)
    if union_manifest.get("test_opened") is not False:
        raise ValueError("Training union must prove test_opened=false")
    union = _read_filtered_parquet(args.union, config.train_folds)
    observed = set(union["fold"].astype(int))
    if observed != set(config.train_folds):
        raise ValueError(
            f"Training requires all folds {config.train_folds}; observed {sorted(observed)}"
        )
    model, diagnostics = train_ranker(union, config)

    _new_output(args.output_dir)
    model_path = args.output_dir / "ranker.json"
    metadata_path = args.output_dir / "metadata.json"
    model.save_model(model_path)
    metadata = {
        "schema_version": RUN_SCHEMA_VERSION,
        "policy_id": config.policy_id,
        "feature_order": feature_order(config),
        "train_folds": list(config.train_folds),
        "dev_fold_read": False,
        "test_opened": False,
        "positive_injection": False,
        "training": diagnostics,
        "source_union_sha256": file_sha256(args.union),
        "config_sha256": file_sha256(args.config),
    }
    _write_json(metadata_path, metadata)
    manifest = {
        **_manifest_base(command="train", config_path=args.config, test_opened=False),
        "opened_folds": list(config.train_folds),
        "inputs": {
            str(args.union.resolve()): file_sha256(args.union),
            str(args.union.parent.joinpath("manifest.json").resolve()): file_sha256(
                args.union.parent / "manifest.json"
            ),
        },
        "outputs": {
            model_path.name: file_sha256(model_path),
            metadata_path.name: file_sha256(metadata_path),
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return args.output_dir


def _load_model(model_dir: Path, config_path: Path, config: AdmissionConfig) -> xgb.XGBRanker:
    model_path = model_dir / "ranker.json"
    metadata_path = model_dir / "metadata.json"
    manifest = _verify_manifest_output(model_path)
    if manifest.get("test_opened") is not False:
        raise ValueError("Frozen model provenance opened test")
    metadata = _read_json(metadata_path)
    if metadata.get("feature_order") != feature_order(config):
        raise ValueError("Frozen model feature order differs")
    if metadata.get("config_sha256") != file_sha256(config_path):
        raise ValueError("Frozen model used another config")
    if metadata.get("test_opened") is not False:
        raise ValueError("Frozen model metadata opened test")
    model = xgb.XGBRanker()
    model.load_model(model_path)
    return model


def _validate_test_authorization(
    *,
    path: Path,
    candidates: Path,
    config_path: Path,
    model_dir: Path,
) -> dict[str, Any]:
    authorization = _read_json(path)
    required = {
        "schema_version": TEST_AUTH_SCHEMA_VERSION,
        "purpose": "retrieval_ltr_fold1_one_shot",
        "status": "FROZEN_AUTHORIZED",
    }
    for key, expected in required.items():
        if authorization.get(key) != expected:
            raise ValueError(f"Invalid final authorization {key}")
    if not str(authorization.get("authorization_id") or "").strip():
        raise ValueError("Final authorization has no authorization_id")
    expected_hashes = {
        "candidates_sha256": file_sha256(candidates),
        "config_sha256": file_sha256(config_path),
        "model_sha256": file_sha256(model_dir / "ranker.json"),
        "model_metadata_sha256": file_sha256(model_dir / "metadata.json"),
    }
    for key, observed in expected_hashes.items():
        if authorization.get(key) != observed:
            raise ValueError(f"Final authorization hash mismatch: {key}")
    return authorization


def _consume_test_authorization_once(
    path: Path, authorization: dict[str, Any]
) -> Path:
    """Atomically burn the authorization before the sole fold-1 read."""
    marker = path.with_name(f"{path.name}.consumed.json")
    payload = {
        "schema_version": TEST_AUTH_SCHEMA_VERSION,
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": file_sha256(path),
        "consumed_at": datetime.now(timezone.utc).isoformat(),
        "reason": "fold1_read_started",
    }
    try:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Final authorization was already consumed: {marker}"
        ) from exc
    return marker


def _candidate_lists(selected: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for query_id, group in selected.sort_values(
        ["query_id", "selected_rank"], kind="stable"
    ).groupby("query_id", sort=True):
        values = group["candidate_siret"].astype(str).tolist()
        records.append(
            {
                "query_id": str(query_id),
                "candidate_count": len(values),
                "candidate_sirets_json": json.dumps(values, separators=(",", ":")),
            }
        )
    return pd.DataFrame(records)


def _evaluate_and_write(
    *,
    union: pd.DataFrame,
    model: xgb.XGBRanker,
    config: AdmissionConfig,
    config_path: Path,
    output_dir: Path,
    split: str,
    inputs: dict[str, str],
    authorization: dict[str, Any] | None,
    authorization_path: Path | None,
    union_diagnostics: dict[str, Any] | None,
) -> Path:
    selected, outcomes, latency = score_and_select(model, union, config)
    integrity = {
        "positive_injection_absent": True,
        "synthetic_rows_absent": True,
        "dense_channels_absent": True,
        "duplicate_selected_pairs_absent": not selected.duplicated(
            ["query_id", "candidate_siret"]
        ).any(),
    }
    summary = evaluate_outcomes(
        outcomes, config, integrity=integrity, latency=latency
    )
    summary.update(
        {
            "split": split,
            "opened_fold": config.test_fold if split == "test" else config.dev_fold,
            "test_opened": split == "test",
            "union_diagnostics": union_diagnostics,
        }
    )
    _new_output(output_dir)
    selected_path = output_dir / "selected_candidates.parquet"
    outcomes_path = output_dir / "query_outcomes.parquet"
    lists_path = output_dir / "candidate_lists.parquet"
    summary_path = output_dir / "summary.json"
    selected.to_parquet(selected_path, index=False, compression="zstd")
    outcomes.to_parquet(outcomes_path, index=False, compression="zstd")
    _candidate_lists(selected).to_parquet(lists_path, index=False, compression="zstd")
    _write_json(summary_path, summary)
    outputs = {
        selected_path.name: file_sha256(selected_path),
        outcomes_path.name: file_sha256(outcomes_path),
        lists_path.name: file_sha256(lists_path),
        summary_path.name: file_sha256(summary_path),
    }
    if split == "test":
        marker_path = output_dir / "ONE_SHOT_CONSUMED.json"
        _write_json(
            marker_path,
            {
                "authorization_id": authorization["authorization_id"],
                "authorization_sha256": file_sha256(authorization_path),
                "consumed_at": datetime.now(timezone.utc).isoformat(),
                "fold": config.test_fold,
                "verdict": summary["verdict"],
            },
        )
        outputs[marker_path.name] = file_sha256(marker_path)
    manifest = {
        **_manifest_base(
            command=f"evaluate:{split}",
            config_path=config_path,
            test_opened=split == "test",
        ),
        "opened_folds": [config.test_fold if split == "test" else config.dev_fold],
        "authorization_id": authorization.get("authorization_id") if authorization else None,
        "authorization_sha256": (
            file_sha256(authorization_path) if authorization_path else None
        ),
        "verdict": summary["verdict"],
        "inputs": inputs,
        "outputs": outputs,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def evaluate_command(args: argparse.Namespace) -> Path:
    config = _load_unique_config(args.config)
    model = _load_model(args.model_dir, args.config, config)
    model_inputs = {
        str((args.model_dir / "ranker.json").resolve()): file_sha256(
            args.model_dir / "ranker.json"
        ),
        str((args.model_dir / "metadata.json").resolve()): file_sha256(
            args.model_dir / "metadata.json"
        ),
    }
    if args.split == "dev":
        if args.open_test or args.authorization or args.candidates:
            raise ValueError("Dev evaluation cannot receive final-test arguments")
        if args.union is None:
            raise ValueError("Dev evaluation requires --union")
        union_manifest = _verify_manifest_output(args.union)
        if union_manifest.get("test_opened") is not False:
            raise ValueError("Development union opened test")
        union = _read_filtered_parquet(args.union, [config.dev_fold])
        if set(union["fold"].astype(int)) != {config.dev_fold}:
            raise ValueError("Development union does not contain fold 0")
        inputs = {
            **model_inputs,
            str(args.union.resolve()): file_sha256(args.union),
            str(args.union.parent.joinpath("manifest.json").resolve()): file_sha256(
                args.union.parent / "manifest.json"
            ),
        }
        return _evaluate_and_write(
            union=union,
            model=model,
            config=config,
            config_path=args.config,
            output_dir=args.output_dir,
            split="dev",
            inputs=inputs,
            authorization=None,
            authorization_path=None,
            union_diagnostics=None,
        )

    if not args.open_test:
        raise ValueError("Final test requires explicit --open-test")
    if args.authorization is None or args.candidates is None:
        raise ValueError("Final test requires --authorization and --candidates")
    if args.union is not None:
        raise ValueError("Final test builds fold-1 union once from --candidates")
    authorization = _validate_test_authorization(
        path=args.authorization,
        candidates=args.candidates,
        config_path=args.config,
        model_dir=args.model_dir,
    )
    # Output existence is checked before the only authorized fold-1 read.
    if args.output_dir.exists():
        raise FileExistsError(f"One-shot output already exists: {args.output_dir}")
    consumption_marker = _consume_test_authorization_once(
        args.authorization, authorization
    )
    raw = _read_filtered_parquet(args.candidates, [config.test_fold])
    union, diagnostics = build_internal_union(
        raw, config, allowed_folds=[config.test_fold]
    )
    if set(union["fold"].astype(int)) != {config.test_fold}:
        raise ValueError("Authorized candidate source has no fold 1")
    inputs = {
        **model_inputs,
        str(args.candidates.resolve()): file_sha256(args.candidates),
        str(args.authorization.resolve()): file_sha256(args.authorization),
        str(consumption_marker.resolve()): file_sha256(consumption_marker),
    }
    return _evaluate_and_write(
        union=union,
        model=model,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        split="test",
        inputs=inputs,
        authorization=authorization,
        authorization_path=args.authorization,
        union_diagnostics=diagnostics,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-union", help="Build folds 2/3/4 and optionally dev 0")
    build.add_argument("--candidates", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--scope", choices=("train", "development"), default="development")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    build.set_defaults(handler=build_union_command)

    train = commands.add_parser("train", help="Fit the frozen rank:ndcg policy on 2/3/4")
    train.add_argument("--union", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    train.set_defaults(handler=train_command)

    evaluate = commands.add_parser("evaluate", help="Evaluate dev or authorized test once")
    evaluate.add_argument("--split", choices=("dev", "test"), required=True)
    evaluate.add_argument("--union", type=Path)
    evaluate.add_argument("--candidates", type=Path)
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    evaluate.add_argument("--open-test", action="store_true")
    evaluate.add_argument("--authorization", type=Path)
    evaluate.set_defaults(handler=evaluate_command)
    return root


def main() -> None:
    args = parser().parse_args()
    print(args.handler(args))


if __name__ == "__main__":
    main()
