#!/usr/bin/env python3
"""Run national sharded v12 OOF only after sealed physical authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_fold_fused_physical_benchmark_v12 import (  # noqa: E402
    current_national_worker_guard_v12,
    load_sharded_oof_authorization_v12,
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
    authorization = load_sharded_oof_authorization_v12(
        args.physical_benchmark_manifest,
        args.execution_authorization,
        expected_lineage=prepared.lineage,
        expected_prefit_parity_manifest_sha256=(
            prepared.prefit_parity_manifest_sha256
        ),
        expected_sharded_execution_plan_sha256=(
            prepared.sharded_execution_plan_sha256
        ),
    )
    output = require_external_sharded_output_v12(
        args.output_dir, executor=prepared.executor
    )
    runtime_guard = current_national_worker_guard_v12(
        authorization.worker_count,
        prepared.sharded_execution_plan_path,
    )
    selected_workers = int(
        runtime_guard["selected_maximum_concurrent_workers"]
    )
    national_lineage = {
        **dict(authorization.lineage),
        "physical_benchmark_manifest_sha256": authorization.manifest_sha256,
        "execution_authorization_sha256": authorization.authorization_sha256,
        "runtime_resource_guard_sha256": str(runtime_guard["record_sha256"]),
    }
    inputs = ShardedPhysicalInputsV12(
        scene_index=prepared.scene_cache.index,
        query_ids=prepared.national_query_ids,
        lineage=national_lineage,
        build_entrypoint=BUILD_ENTRYPOINT_V12,
        build_payload=prepared.runtime_payload,
        replay_entrypoint=REPLAY_ENTRYPOINT_V12,
        replay_payload=prepared.runtime_payload,
        model_backend=LazyXGBoostSiteLambdaMARTBackendV12(
            params=prepared.admission.xgb_params("site"),
            output_dir=output / "site_models",
        ),
        caps=prepared.caps,
        gates=prepared.gates,
        maximum_spool_bytes=prepared.executor.maximum_temporary_gib * 1024**3,
        maximum_workers=selected_workers,
        execution_evidence={
            "phase": "NATIONAL_OOF_FOLDS_2_3_4",
            "physical_benchmark_manifest_sha256": authorization.manifest_sha256,
            "execution_authorization_sha256": authorization.authorization_sha256,
            "runtime_resource_guard": dict(runtime_guard),
            "fixed_shard_count": 3,
            "maximum_concurrent_workers": selected_workers,
        },
    )
    specs = plan_query_shards(
        prepared.scene_cache.index,
        prepared.national_query_ids,
        3,
    )
    artifact = run_sharded_physical_sample(
        inputs, specs, output, miniature_fit=False
    )
    print(str(artifact.manifest_path))
    return artifact.manifest_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run exact national fold-fused OOF with the worker count sealed "
            "by the 512-query physical parity benchmark"
        )
    )
    value.add_argument(
        "--sharded-execution-plan", type=Path, default=DEFAULT_PLAN
    )
    value.add_argument("--physical-benchmark-manifest", type=Path, required=True)
    value.add_argument("--execution-authorization", type=Path, required=True)
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
