#!/usr/bin/env python3
"""Run the preregistered fold-fused v12 OOF physical plan.

The command has an explicit ``--preflight-only`` mode.  A full run requires
the non-circular sealed 512-query PREFIT parity artifact and an output path on
the configured external scratch volume.  POSTFIT parity cannot be supplied to
this command: it must be produced from the three models emitted by this run,
then joined by the separate certifier without refitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (  # noqa: E402
    ExecutionCaps,
    ExistingV12SelectionAdapter,
    FoldFusedInvariantError,
    FoldFusedOOFExecutor,
    GateThresholds,
    ParquetCompactTrainingStore,
    ParquetOOFOutputStore,
    ParityEvidence,
    RunLineage,
    XGBoostSiteLambdaMARTBackend,
    sha256_file,
)
from src.xgb_matcher.retrieval_fold_fused_runtime_v12 import (  # noqa: E402
    ExistingV12FitSamplerAdapter,
    FoldFusedContextAdapterV12,
    FoldFusedOfficialSidecarsV12,
    SceneIRSourceAdapterV12,
    StrictV12LabelLedger,
    load_executor_config_v12,
    load_parent_scorer_catalog_v12,
    load_prefit_parity_evidence_v12,
    reconcile_scene_ir_and_labels_v12,
)
from src.xgb_matcher.retrieval_fold_fused_heldout_spool_v12 import (  # noqa: E402
    ParquetHeldoutMasterSpoolV12,
    validate_spool_calibration_v12,
)
from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (  # noqa: E402
    AdmissionConfigV12,
    site_feature_names,
)
from src.xgb_matcher.retrieval_top100_cascade_v12 import (  # noqa: E402
    RetrievalTop100CascadeV12Config,
)
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache  # noqa: E402
from src.xgb_matcher.retrieval_v12_reader_parity import (  # noqa: E402
    load_mmap_reader_parity_evidence_v12,
)


DEFAULT_CONFIG = ROOT / "config/retrieval_all_human_v12.json"
DEFAULT_EXECUTOR_CONFIG = (
    ROOT / "config/retrieval_all_human_v12_fold_fused_executor.json"
)


def _bundle_manifest(path: Path) -> Path:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _require_external_output(
    output_dir: Path,
    *,
    scratch_root: Path,
    maximum_temporary_gib: int,
) -> MappingResult:
    output = Path(output_dir).resolve()
    scratch = Path(scratch_root).resolve()
    if not scratch.is_dir():
        raise FileNotFoundError(scratch)
    if output == scratch or scratch not in output.parents:
        raise FoldFusedInvariantError(
            "fold-fused OOF output must be a child of the configured external scratch root"
        )
    if output.exists():
        raise FileExistsError(output)
    free_bytes = shutil.disk_usage(scratch).free
    required_bytes = int(maximum_temporary_gib) * 1024**3
    if free_bytes < required_bytes:
        raise FoldFusedInvariantError(
            f"external scratch has {free_bytes / 1024**3:.1f} GiB free; "
            f"the preregistered temporary ceiling requires {maximum_temporary_gib} GiB"
        )
    return MappingResult(output, free_bytes)


class MappingResult:
    def __init__(self, output_dir: Path, free_bytes: int) -> None:
        self.output_dir = output_dir
        self.free_bytes = int(free_bytes)


def _preflight(args: argparse.Namespace) -> tuple[dict, dict]:
    validated_executor = load_executor_config_v12(
        args.executor_config,
        scientific_config_path=args.config,
        workspace_root=ROOT,
    )
    admission = AdmissionConfigV12.load(args.config)
    cascade = RetrievalTop100CascadeV12Config.load(args.config)
    if admission.train_folds != (2, 3, 4):
        raise FoldFusedInvariantError("scientific config outer folds drifted")
    if (
        admission.internal_cap != 2_000
        or admission.prerank_cap != 500
        or admission.final_cap != 100
        or cascade.master_reference_cap != 2_000
        or cascade.hydration_shortlist_cap != 500
        or len(site_feature_names(admission)) != 319
    ):
        raise FoldFusedInvariantError("scientific v12 caps/319-feature schema drifted")

    scene_cache = SceneIRCache(
        args.scene_ir,
        verify_source_hash=True,
    )
    ledger = StrictV12LabelLedger.load(args.labels, config_path=args.config)
    reconcile_scene_ir_and_labels_v12(scene_cache, ledger)
    scene_manifest_path = Path(args.scene_ir).resolve() / "manifest.json"
    sidecar_manifest_path = Path(args.sidecars).resolve() / "manifest.json"
    mmap_manifest_path = Path(args.mmap_sidecar).resolve() / "manifest.json"

    parent_bundle_path = _bundle_manifest(args.parent_model_bundle)
    context = {
        "validated_executor": validated_executor,
        "admission": admission,
        "cascade": cascade,
        "scene_cache": scene_cache,
        "ledger": ledger,
        "scene_manifest_path": scene_manifest_path,
        "sidecar_manifest_path": sidecar_manifest_path,
        "mmap_manifest_path": mmap_manifest_path,
        "parent_bundle_path": parent_bundle_path,
    }
    report = {
        "status": "PREFLIGHT_HASHES_PENDING_SIDECAR_OPEN",
        "engine_id": validated_executor.raw["engine_id"],
        "query_count": len(ledger.expected_query_ids(folds=(2, 3, 4))),
        "scene_ir_build_id": scene_cache.manifest["build_id"],
        "scene_ir_manifest_sha256": sha256_file(scene_manifest_path),
        "scientific_config_sha256": validated_executor.scientific_config_sha256,
        "scientific_contract_sha256": validated_executor.scientific_contract_sha256,
        "executor_config_sha256": validated_executor.sha256,
        "label_ledger_sha256": ledger.ledger_sha256,
        "label_ledger_manifest_sha256": ledger.manifest_sha256,
        "folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "site_feature_count": len(site_feature_names(admission)),
        "caps": {"master": 2_000, "fit": 128, "prerank": 500, "product": 100},
    }
    return context, report


def run(args: argparse.Namespace) -> Path | None:
    context, report = _preflight(args)
    validated_executor = context["validated_executor"]
    scene_cache = context["scene_cache"]
    ledger = context["ledger"]
    output_check = None
    if not args.preflight_only:
        if args.output_dir is None:
            raise FoldFusedInvariantError("a full OOF run requires --output-dir")
        output_check = _require_external_output(
            args.output_dir,
            scratch_root=validated_executor.scratch_root,
            maximum_temporary_gib=validated_executor.maximum_temporary_gib,
        )
    with FoldFusedOfficialSidecarsV12.open(
        args.sidecars,
        args.mmap_sidecar,
        verify_hashes=True,
    ) as sidecars:
        if sidecars.sidecar_manifest_sha256 != sha256_file(
            context["sidecar_manifest_path"]
        ) or sidecars.mmap_manifest_sha256 != sha256_file(
            context["mmap_manifest_path"]
        ):
            raise FoldFusedInvariantError("sidecar manifest changed during preflight")
        local_record = (sidecars.official.manifest.get("files") or {}).get(
            "siren_local_sites.parquet"
        ) or {}
        source_rows = int(
            (sidecars.official.manifest.get("counts") or {}).get(
                "siren_local_sites", -1
            )
        )
        _reader_parity, reader_parity_sha = load_mmap_reader_parity_evidence_v12(
            args.reader_parity_manifest,
            expected_official_manifest_sha256=sidecars.sidecar_manifest_sha256,
            expected_mmap_manifest_sha256=sidecars.mmap_manifest_sha256,
            expected_source_local_sites_sha256=str(local_record.get("sha256") or ""),
            expected_source_rows=source_rows,
            expected_mmap_build_id=str(
                sidecars.mmap_sites.manifest.get("build_id") or ""
            ),
        )
        parent_bundle = load_parent_scorer_catalog_v12(
            context["parent_bundle_path"],
            config_sha256=validated_executor.scientific_config_sha256,
            source_candidate_sha256=str(
                scene_cache.manifest["source_candidate_sha256"]
            ),
            source_manifest_sha256=str(
                scene_cache.manifest["source_manifest_sha256"]
            ),
            sidecar_manifest_sha256=sidecars.sidecar_manifest_sha256,
            label_ledger_sha256=ledger.ledger_sha256,
            label_ledger_manifest_sha256=ledger.manifest_sha256,
        )
        report.update(
            {
                "status": "PREFLIGHT_PASS",
                "source_candidate_sha256": scene_cache.manifest[
                    "source_candidate_sha256"
                ],
                "source_manifest_sha256": scene_cache.manifest[
                    "source_manifest_sha256"
                ],
                "official_sidecar_manifest_sha256": sidecars.sidecar_manifest_sha256,
                "mmap_sidecar_manifest_sha256": sidecars.mmap_manifest_sha256,
                "reader_parity_evidence_sha256": reader_parity_sha,
                "parent_model_bundle_sha256": parent_bundle.sha256,
                "parent_model_sha256": dict(parent_bundle.model_sha256),
                "parity_gate": "NOT_REQUESTED_IN_PREFLIGHT",
            }
        )
        if args.preflight_only:
            print(json.dumps(report, indent=2, sort_keys=True))
            return None

        if args.prefit_parity_manifest is None:
            raise FoldFusedInvariantError(
                "national OOF requires sealed passing non-circular PREFIT parity"
            )
        expected_parity_lineage = {
            "scene_ir_sha256": str(scene_cache.manifest["build_id"]),
            "scene_ir_manifest_sha256": sha256_file(
                context["scene_manifest_path"]
            ),
            "source_candidate_sha256": str(
                scene_cache.manifest["source_candidate_sha256"]
            ),
            "source_manifest_sha256": str(
                scene_cache.manifest["source_manifest_sha256"]
            ),
            "label_ledger_sha256": ledger.ledger_sha256,
            "label_ledger_manifest_sha256": ledger.manifest_sha256,
            "scientific_config_sha256": validated_executor.scientific_config_sha256,
            "executor_config_sha256": validated_executor.sha256,
            "official_sidecar_manifest_sha256": sidecars.sidecar_manifest_sha256,
            "mmap_sidecar_manifest_sha256": sidecars.mmap_manifest_sha256,
            "reader_parity_evidence_sha256": reader_parity_sha,
            "parent_model_bundle_sha256": parent_bundle.sha256,
            "query_sample_sha256": str(
                ledger.manifest.get("runtime_query_sample_sha256") or ""
            ),
        }
        _prefit, prefit_parity_sha = load_prefit_parity_evidence_v12(
            args.prefit_parity_manifest,
            expected_lineage=expected_parity_lineage,
        )
        validate_spool_calibration_v12(
            args.prefit_parity_manifest,
            _prefit.get("heldout_spool_calibration") or {},
            expected_sample_queries=512,
            expected_national_queries=len(
                ledger.expected_query_ids(folds=(2, 3, 4))
            ),
            expected_maximum_bytes=(
                validated_executor.maximum_temporary_gib * 1024**3
            ),
        )
        parity = ParityEvidence()
        parity_sha = None
        report["parity_gate"] = "PREFIT_PASS_POSTFIT_PENDING"
        report["prefit_parity_evidence_sha256"] = prefit_parity_sha
        report["parity_evidence_sha256"] = parity_sha
        assert output_check is not None
        output = output_check.output_dir
        source = SceneIRSourceAdapterV12(
            scene_cache, batch_size=int(args.batch_size)
        )
        contexts = FoldFusedContextAdapterV12(
            sidecars=sidecars,
            parent_scorers=parent_bundle.scorers,
            cascade_config=context["cascade"],
            admission_config=context["admission"],
        )
        sampling_cap = int(
            context["admission"].fit_candidate_sampling[
                "maximum_candidates_per_query"
            ]
        )
        executor = FoldFusedOOFExecutor(
            source=source,
            labels=ledger,
            contexts=contexts,
            sampler=ExistingV12FitSamplerAdapter(context["admission"]),
            train_store=ParquetCompactTrainingStore(output / "train_matrices"),
            heldout_store=ParquetHeldoutMasterSpoolV12(
                output / "heldout_master_spool",
                config=context["admission"],
                maximum_bytes=validated_executor.maximum_temporary_gib
                * 1024**3,
            ),
            model_backend=XGBoostSiteLambdaMARTBackend(
                params=context["admission"].xgb_params("site"),
                output_dir=output / "models",
            ),
            selector=ExistingV12SelectionAdapter(context["admission"]),
            output_store=ParquetOOFOutputStore(output / "outputs"),
            lineage=RunLineage(
                scene_ir_sha256=str(scene_cache.manifest["build_id"]),
                scene_ir_manifest_sha256=sha256_file(
                    context["scene_manifest_path"]
                ),
                label_ledger_sha256=ledger.ledger_sha256,
                label_ledger_manifest_sha256=ledger.manifest_sha256,
                label_sidecar_sha256=ledger.operational_sidecar_sha256,
                scientific_config_sha256=validated_executor.scientific_config_sha256,
                scientific_contract_sha256=validated_executor.scientific_contract_sha256,
                parent_model_bundle_sha256=parent_bundle.sha256,
                executor_config_sha256=validated_executor.sha256,
                source_candidate_sha256=str(
                    scene_cache.manifest["source_candidate_sha256"]
                ),
                source_manifest_sha256=str(
                    scene_cache.manifest["source_manifest_sha256"]
                ),
                official_sidecar_manifest_sha256=sidecars.sidecar_manifest_sha256,
                mmap_sidecar_manifest_sha256=sidecars.mmap_manifest_sha256,
                reader_parity_evidence_sha256=reader_parity_sha,
                query_sample_sha256=str(
                    ledger.manifest.get("runtime_query_sample_sha256") or ""
                ),
                prefit_parity_evidence_sha256=prefit_parity_sha,
                parity_evidence_sha256=parity_sha,
            ),
            manifest_path=output / "manifest.json",
            caps=ExecutionCaps(fit_sample=sampling_cap),
            gates=GateThresholds(
                master_recall_at_2000=context["admission"].gates["oracle_2000"],
                prerank_recall_at_500=context["admission"].gates["prerank_500"],
                top100_recall=context["admission"].gates["recall_100"],
                top100_wilson95_lower=context["admission"].gates["wilson_100"],
                minimum_each_outer_fold_top100=context["admission"].gates[
                    "worst_fold_100"
                ],
            ),
            parity=parity,
        )
        result = executor.run()
        print(str(result.manifest_path))
        return result.manifest_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the sealed fold-fused retrieval-v12 OOF executor"
    )
    value.add_argument("--scene-ir", type=Path, required=True)
    value.add_argument("--labels", type=Path, required=True)
    value.add_argument("--sidecars", type=Path, required=True)
    value.add_argument("--mmap-sidecar", type=Path, required=True)
    value.add_argument("--reader-parity-manifest", type=Path, required=True)
    value.add_argument("--parent-model-bundle", type=Path, required=True)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--executor-config", type=Path, default=DEFAULT_EXECUTOR_CONFIG)
    value.add_argument("--prefit-parity-manifest", type=Path)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--batch-size", type=int, default=262_144)
    value.add_argument("--preflight-only", action="store_true")
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
