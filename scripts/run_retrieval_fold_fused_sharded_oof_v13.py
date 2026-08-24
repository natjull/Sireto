#!/usr/bin/env python3
"""Run authorized national retrieval-v13 OOF with fragment-only artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_fold_fused_national_runtime_v13 import (  # noqa: E402
    execute_national_fragment_oof_v13,
)
from src.xgb_matcher.retrieval_fold_fused_sharded_preflight_v12 import (  # noqa: E402
    prepare_sharded_execution_v12,
    require_external_sharded_output_v12,
)


DEFAULT_PLAN = ROOT / "config/retrieval_all_human_v12_sharded_execution_plan.json"


def _plan_defaults(path: Path) -> dict[str, Path]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    lineage = plan.get("input_lineage") or {}
    authorities = plan.get("authorities") or {}
    scene = lineage.get("scene_ir") or {}
    labels = lineage.get("labels") or {}
    sidecars = lineage.get("official_sidecars") or {}
    parent = lineage.get("parent_model_bundle") or {}
    prefit = lineage.get("prefit_parity") or {}

    def authority(name: str) -> Path:
        value = Path(str((authorities.get(name) or {}).get("path") or ""))
        return value if value.is_absolute() else ROOT / value

    return {
        "scene_ir": Path(str(scene.get("directory") or "")),
        "query_sample": Path(str(labels.get("runtime_query_sample_path") or "")),
        "labels": Path(str(labels.get("ledger_path") or "")),
        "sidecars": Path(str(sidecars.get("sidecar_manifest_path") or "")).parent,
        "mmap_sidecar": Path(str(sidecars.get("mmap_manifest_path") or "")).parent,
        "reader_parity_manifest": Path(
            str(sidecars.get("reader_parity_manifest_path") or "")
        ),
        "parent_model_bundle": Path(str(parent.get("manifest_path") or "")),
        "scientific_config": authority("scientific_config"),
        "executor_config": authority("fold_fused_executor_config"),
        "prefit_parity_manifest": Path(str(prefit.get("manifest_path") or "")),
    }


def _value(args: argparse.Namespace, defaults: dict[str, Path], name: str) -> Path:
    value = getattr(args, name)
    return Path(value if value is not None else defaults[name]).resolve()


def run(args: argparse.Namespace) -> Path:
    plan_path = DEFAULT_PLAN.resolve()
    defaults = _plan_defaults(plan_path)
    # Cache-neutral by construction.  The strict authorization loader inside
    # the national runtime rehashes the complete source through the post-W3
    # logical-I/O proof before any national worker is spawned.
    prepared = prepare_sharded_execution_v12(
        workspace_root=ROOT,
        scene_ir=_value(args, defaults, "scene_ir"),
        query_sample=_value(args, defaults, "query_sample"),
        labels=_value(args, defaults, "labels"),
        sidecars=_value(args, defaults, "sidecars"),
        mmap_sidecar=_value(args, defaults, "mmap_sidecar"),
        reader_parity_manifest=_value(
            args, defaults, "reader_parity_manifest"
        ),
        parent_model_bundle=_value(args, defaults, "parent_model_bundle"),
        scientific_config=_value(args, defaults, "scientific_config"),
        executor_config=_value(args, defaults, "executor_config"),
        prefit_parity_manifest=_value(
            args, defaults, "prefit_parity_manifest"
        ),
        sharded_execution_plan=plan_path,
        verify_source_hash=False,
    )
    output = require_external_sharded_output_v12(
        args.output_dir, executor=prepared.executor
    )
    artifact = execute_national_fragment_oof_v13(
        prepared=prepared,
        workspace_root=ROOT,
        output_dir=output,
        physical_benchmark_manifest=Path(
            args.physical_benchmark_manifest
        ).resolve(),
        physical_benchmark_manifest_sha256=(
            args.physical_benchmark_manifest_sha256
        ),
        execution_authorization=Path(args.execution_authorization).resolve(),
        execution_authorization_sha256=args.execution_authorization_sha256,
    )
    print(str(artifact.manifest_path))
    return artifact.manifest_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the separately authorized 23,571-query retrieval-v13 OOF "
            "with three fold-owned fragment actors and no merged Parquet"
        )
    )
    value.add_argument("--physical-benchmark-manifest", type=Path, required=True)
    value.add_argument(
        "--physical-benchmark-manifest-sha256", required=True
    )
    value.add_argument("--execution-authorization", type=Path, required=True)
    value.add_argument("--execution-authorization-sha256", required=True)
    value.add_argument("--scene-ir", type=Path)
    value.add_argument("--query-sample", type=Path)
    value.add_argument("--labels", type=Path)
    value.add_argument("--sidecars", type=Path)
    value.add_argument("--mmap-sidecar", type=Path)
    value.add_argument("--reader-parity-manifest", type=Path)
    value.add_argument("--parent-model-bundle", type=Path)
    value.add_argument("--scientific-config", type=Path)
    value.add_argument("--executor-config", type=Path)
    value.add_argument("--prefit-parity-manifest", type=Path)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
