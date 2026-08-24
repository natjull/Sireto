#!/usr/bin/env python3
"""Run the sealed physical W3/W1/(W2) benchmark on exactly 512 queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_fold_fused_physical_benchmark_v12 import (  # noqa: E402
    physical_trial_product_from_sharded_v12,
    run_physical_worker_benchmark_v12,
    sharded_execution_lineage_v12,
)
from src.xgb_matcher.retrieval_fold_fused_sharded_oof_v12 import (  # noqa: E402
    ShardedPhysicalInputsV12,
    plan_query_shards,
    run_sharded_physical_sample,
)
from src.xgb_matcher.retrieval_fold_fused_sharded_preflight_v12 import (  # noqa: E402
    prepare_sharded_execution_v12,
    require_external_sharded_output_v12,
)
from src.xgb_matcher.retrieval_fold_fused_sharded_runtime_v12 import (  # noqa: E402
    BUILD_ENTRYPOINT_V12,
    REPLAY_ENTRYPOINT_V12,
    LazyXGBoostSiteLambdaMARTBackendV12,
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
    plan_path = Path(args.sharded_execution_plan).resolve()
    defaults = _plan_defaults(plan_path)
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
    )
    output = require_external_sharded_output_v12(
        args.output_dir, executor=prepared.executor
    )
    execution_lineage = sharded_execution_lineage_v12(
        prepared.lineage, workspace_root=ROOT
    )

    def trial(worker_count: int, trial_dir: Path):
        inputs = ShardedPhysicalInputsV12(
            scene_index=prepared.scene_cache.index,
            query_ids=prepared.sample_query_ids,
            lineage=execution_lineage,
            build_entrypoint=BUILD_ENTRYPOINT_V12,
            build_payload=prepared.runtime_payload,
            replay_entrypoint=REPLAY_ENTRYPOINT_V12,
            replay_payload=prepared.runtime_payload,
            model_backend=LazyXGBoostSiteLambdaMARTBackendV12(
                params=prepared.admission.xgb_params("site"),
                output_dir=trial_dir / "site_models",
            ),
            caps=prepared.caps,
            gates=prepared.gates,
            maximum_spool_bytes=prepared.executor.maximum_temporary_gib * 1024**3,
            maximum_workers=worker_count,
            execution_evidence={
                "phase": "PHYSICAL_PARITY_BENCHMARK_512",
                "sharded_implementation_sha256": execution_lineage[
                    "sharded_implementation_sha256"
                ],
                "runtime_environment_sha256": execution_lineage[
                    "runtime_environment_sha256"
                ],
                "prefit_parity_manifest_sha256": (
                    prepared.prefit_parity_manifest_sha256
                ),
                "sharded_execution_plan_sha256": (
                    prepared.sharded_execution_plan_sha256
                ),
            },
        )
        shard_count = 1 if worker_count == 1 else 3
        specs = plan_query_shards(
            prepared.scene_cache.index,
            prepared.sample_query_ids,
            shard_count,
        )
        artifact = run_sharded_physical_sample(
            inputs, specs, trial_dir, miniature_fit=True
        )
        return physical_trial_product_from_sharded_v12(artifact)

    manifest = run_physical_worker_benchmark_v12(
        runner=trial,
        output_dir=output,
        expected_lineage=execution_lineage,
        prefit_parity_manifest_path=prepared.prefit_parity_manifest_path,
        prefit_parity_manifest_sha256=prepared.prefit_parity_manifest_sha256,
        sharded_execution_plan_path=prepared.sharded_execution_plan_path,
        sharded_execution_plan_sha256=prepared.sharded_execution_plan_sha256,
    )
    print(str(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Benchmark exact retrieval-v12 sharding on sealed 512 queries; "
            "authorize W3 or W2, otherwise stop, without opening folds 0/1"
        )
    )
    value.add_argument(
        "--sharded-execution-plan", type=Path, default=DEFAULT_PLAN
    )
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
