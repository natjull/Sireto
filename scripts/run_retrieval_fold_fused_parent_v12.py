#!/usr/bin/env python3
"""Fit the six nested retrieval-v12 parent models directly from SceneIR.

This command is the fold-fused replacement for pass-A parent materialisation
followed by ``fit-parent-oof``.  It opens only folds 2/3/4, uses the canonical
strict label-ledger validator, and publishes a bundle accepted by the existing
nested cascade loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_fold_fused_parent_bundle_v12 import (  # noqa: E402
    build_fold_fused_parent_bundle_v12,
    parent_bundle_lineage_from_inputs_v12,
)
from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (  # noqa: E402
    FoldFusedInvariantError,
)
from src.xgb_matcher.retrieval_fold_fused_runtime_v12 import (  # noqa: E402
    StrictV12LabelLedger,
    load_executor_config_v12,
    reconcile_scene_ir_and_labels_v12,
)
from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (  # noqa: E402
    AdmissionConfigV12,
)
from src.xgb_matcher.retrieval_sidecars_v11 import (  # noqa: E402
    RetrievalSidecarReader,
)
from src.xgb_matcher.retrieval_top100_cascade_v12 import (  # noqa: E402
    RetrievalTop100CascadeV12Config,
)
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache  # noqa: E402
from src.xgb_matcher.retrieval_v12_parent_signal_capability import (  # noqa: E402
    load_parent_signal_capability_v12,
)


DEFAULT_CONFIG = ROOT / "config/retrieval_all_human_v12.json"
DEFAULT_EXECUTOR_CONFIG = (
    ROOT / "config/retrieval_all_human_v12_fold_fused_executor.json"
)


def _require_external_parent_output(
    output_dir: Path,
    *,
    scratch_root: Path,
    maximum_temporary_gib: int,
) -> tuple[Path, int]:
    """Fail closed before the compact spool can touch the system disk."""

    output = Path(output_dir).resolve()
    scratch = Path(scratch_root).resolve()
    if not scratch.is_dir():
        raise FileNotFoundError(scratch)
    if output == scratch or scratch not in output.parents:
        raise FoldFusedInvariantError(
            "fold-fused parent output must be a child of the configured "
            "external scratch root"
        )
    if output.exists():
        raise FileExistsError(output)
    free_bytes = shutil.disk_usage(scratch).free
    required_bytes = int(maximum_temporary_gib) * 1024**3
    if free_bytes < required_bytes:
        raise FoldFusedInvariantError(
            f"external scratch has {free_bytes / 1024**3:.1f} GiB free; "
            f"the preregistered temporary ceiling requires "
            f"{maximum_temporary_gib} GiB"
        )
    return output, int(free_bytes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream SceneIR once, retain <=128 natural bootstrap parents/query, "
            "and seal the six nested v12 parent LambdaMART models"
        )
    )
    parser.add_argument("--scene-ir", type=Path, required=True)
    parser.add_argument(
        "--source-override",
        type=Path,
        help="relocated immutable candidate parquet sealed by SceneIR",
    )
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument(
        "--parent-signal-capability",
        type=Path,
        required=True,
        help=(
            "exhaustive content-addressed proof allowing the strongly "
            "projected parent-only Arrow scanner"
        ),
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--executor-config", type=Path, default=DEFAULT_EXECUTOR_CONFIG
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path | None:
    config_path = Path(args.config).resolve()
    labels_path = Path(args.labels).resolve()
    sidecar_dir = Path(args.sidecars).resolve()
    executor = load_executor_config_v12(
        Path(args.executor_config),
        scientific_config_path=config_path,
        workspace_root=ROOT,
    )
    fit_plan = executor.raw.get("fit_materialization") or {}
    if fit_plan.get("maximum_parent_rows_per_query") != 128:
        raise FoldFusedInvariantError(
            "executor config must pin <=128 parent fit rows per query"
        )
    output_dir, free_bytes = _require_external_parent_output(
        Path(args.output_dir),
        scratch_root=executor.scratch_root,
        maximum_temporary_gib=executor.maximum_temporary_gib,
    )
    scene_cache = SceneIRCache(
        Path(args.scene_ir),
        source_override=(
            Path(args.source_override) if args.source_override is not None else None
        ),
    )
    admission_config = AdmissionConfigV12.load(config_path)
    cascade_config = RetrievalTop100CascadeV12Config.load(config_path)
    signal_capability = load_parent_signal_capability_v12(
        Path(args.parent_signal_capability), scene_cache
    )
    if (
        int(
            admission_config.fit_candidate_sampling[
                "maximum_candidates_per_query"
            ]
        )
        != 128
        or cascade_config.master_reference_cap != 2_000
    ):
        raise FoldFusedInvariantError(
            "scientific config must pin parent fit<=128 and Master<=2000"
        )

    # This loader validates the all-human population, exclusions, exact-only
    # operational sidecar, SIREN-disjoint folds and folds 0/1 closure.  The
    # resulting ledger is not supplied to the scene compiler; the core trainer
    # looks up one row only after that scene's natural parent pool is complete.
    ledger = StrictV12LabelLedger.load(
        labels_path, config_path=config_path
    )
    reconcile_scene_ir_and_labels_v12(scene_cache, ledger)
    labels = pd.DataFrame(
        [
            {
                "query_id": query_id,
                "fold": ledger.get(query_id).fold,
                "human_siret": ledger.get(query_id).human_siret,
            }
            for query_id in ledger.expected_query_ids(folds=(2, 3, 4))
        ]
    )
    labels_manifest_path = ledger.manifest_path
    lineage = parent_bundle_lineage_from_inputs_v12(
        scene_cache,
        sidecar_dir=sidecar_dir,
        labels_path=labels_path,
        labels_manifest_path=labels_manifest_path,
        config_path=config_path,
    )
    with RetrievalSidecarReader(sidecar_dir) as sidecars:
        report = {
            "status": "PREFLIGHT_PASS",
            "engine_id": executor.raw["engine_id"],
            "output_dir": str(output_dir),
            "scratch_root": str(executor.scratch_root),
            "scratch_free_gib": free_bytes / 1024**3,
            "maximum_temporary_gib": executor.maximum_temporary_gib,
            "scene_ir_build_id": lineage.scene_ir_build_id,
            "scene_ir_manifest_sha256": lineage.scene_ir_manifest_sha256,
            "source_candidate_sha256": lineage.source_candidate_sha256,
            "source_manifest_sha256": lineage.source_manifest_sha256,
            "sidecar_manifest_sha256": lineage.sidecar_manifest_sha256,
            "label_ledger_sha256": lineage.source_labels_sha256,
            "label_ledger_manifest_sha256": (
                lineage.source_labels_manifest_sha256
            ),
            "config_sha256": lineage.config_sha256,
            "folds": [2, 3, 4],
            "fold0_opened": False,
            "fold1_opened": False,
            "parent_fit_rows_per_query": 128,
            "parent_signal_capability_build_id": signal_capability.build_id,
            "parent_signal_capability_manifest": str(
                signal_capability.manifest_path
            ),
            "parent_arrow_projection_columns": len(
                signal_capability.projection
            ),
            "full_parent_candidate_parquet_materialized": False,
        }
        if args.preflight_only:
            print(json.dumps(report, indent=2, sort_keys=True))
            return None
        build = build_fold_fused_parent_bundle_v12(
            scene_cache,
            sidecars=sidecars,
            cascade_config=cascade_config,
            admission_config=admission_config,
            labels=labels,
            lineage=lineage,
            output_dir=output_dir,
            batch_size=int(args.batch_size),
            signal_capability=signal_capability,
        )
    return build.manifest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = run(args)
    if manifest_path is None:
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "models": sorted((manifest.get("models") or {}).keys()),
                "queries": (manifest.get("fit_sampling") or {}).get("queries"),
                "rows_after_sampling": (
                    manifest.get("fit_sampling") or {}
                ).get("rows_after_sampling"),
                "full_parent_candidate_parquet_materialized": False,
                "fold0_opened": False,
                "fold1_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
