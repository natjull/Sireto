#!/usr/bin/env python3
"""Run the sole fresh retrieval-v14 W3 benchmark; never national OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_fold_fused_physical_benchmark_v14 import (  # noqa: E402
    run_physical_benchmark_v14,
)


DEFAULT_PLAN = ROOT / "config/retrieval_all_human_v12_sharded_execution_plan.json"
DEFAULT_CONFIG = ROOT / "config/retrieval_all_human_v14_physical_execution.json"
DEFAULT_OUTPUT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
    "retrieval_fold_fused_physical_benchmark_v14_w3"
)
DEFAULT_PROJECTIONS = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
    "retrieval_v14_immutable_projections"
)


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
    plan = Path(args.sharded_execution_plan).resolve()
    defaults = _plan_defaults(plan)
    manifest = run_physical_benchmark_v14(
        config_path=Path(args.physical_config).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        workspace_root=ROOT,
        scene_ir=_value(args, defaults, "scene_ir"),
        query_sample=_value(args, defaults, "query_sample"),
        labels=_value(args, defaults, "labels"),
        sidecars=_value(args, defaults, "sidecars"),
        mmap_sidecar=_value(args, defaults, "mmap_sidecar"),
        reader_parity_manifest=_value(args, defaults, "reader_parity_manifest"),
        parent_model_bundle=_value(args, defaults, "parent_model_bundle"),
        scientific_config=_value(args, defaults, "scientific_config"),
        executor_config=_value(args, defaults, "executor_config"),
        prefit_parity_manifest=_value(args, defaults, "prefit_parity_manifest"),
        sharded_execution_plan=plan,
        immutable_projection_root=Path(args.immutable_projection_root).resolve(),
    )
    print(str(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run exactly one fresh W3 retrieval-v14 physical trial against "
            "the immutable v12 W1 baseline; never launch national OOF"
        )
    )
    value.add_argument("--physical-config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--sharded-execution-plan", type=Path, default=DEFAULT_PLAN)
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
    value.add_argument(
        "--immutable-projection-root", type=Path, default=DEFAULT_PROJECTIONS
    )
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
