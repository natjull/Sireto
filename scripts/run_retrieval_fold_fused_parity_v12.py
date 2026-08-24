#!/usr/bin/env python3
"""Produce the sealed 512-query legacy-vs-fold-fused v12 parity evidence.

Without ``--site-model-manifest`` the command writes the non-circular PREFIT
gate required before OOF model fitting.  With a sealed three-model OOF
manifest it additionally compares PreRank500/top100 and writes the complete
schema accepted by the final v12 parity gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (  # noqa: E402
    FoldFusedInvariantError,
    sha256_file,
)
from src.xgb_matcher.retrieval_fold_fused_parity_v12 import (  # noqa: E402
    FoldFusedParityInputsV12,
    SiteModelCatalogV12,
    run_fold_fused_parity_v12,
)
from src.xgb_matcher.retrieval_fold_fused_runtime_v12 import (  # noqa: E402
    FoldFusedOfficialSidecarsV12,
    StrictV12LabelLedger,
    load_executor_config_v12,
    load_parent_scorer_catalog_v12,
    reconcile_scene_ir_and_labels_v12,
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
    value = Path(path).resolve()
    if value.is_dir():
        value = value / "manifest.json"
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def _sealed_query_sample(
    path: Path, *, ledger: StrictV12LabelLedger
) -> tuple[pd.DataFrame, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    declared_name = ledger.manifest.get("runtime_query_sample_path")
    declared_sha = ledger.manifest.get("runtime_query_sample_sha256")
    artifact = (ledger.manifest.get("artifacts") or {}).get(
        "runtime_query_sample"
    ) or {}
    expected_path = ledger.manifest_path.parent / str(declared_name or "")
    observed_sha = sha256_file(resolved)
    if (
        resolved != expected_path.resolve()
        or declared_sha != observed_sha
        or artifact.get("path") != declared_name
        or artifact.get("sha256") != observed_sha
        or artifact.get("rows") != 512
        or ledger.manifest.get("runtime_query_sample_rows") != 512
        or ledger.manifest.get("runtime_query_sample_rule")
        != "SHA256_QUERY_ID_ASC_FIRST_512"
    ):
        raise FoldFusedInvariantError(
            "parity query sample is not the exact ledger-sealed 512-row artifact"
        )
    return pd.read_parquet(resolved), observed_sha


def run(args: argparse.Namespace) -> Path:
    executor = load_executor_config_v12(
        args.executor_config,
        scientific_config_path=args.config,
        workspace_root=ROOT,
    )
    output_dir = Path(args.output_dir).resolve()
    if (
        not executor.scratch_root.is_dir()
        or output_dir == executor.scratch_root
        or executor.scratch_root not in output_dir.parents
    ):
        raise FoldFusedInvariantError(
            "parity output must be a child of the configured external scratch root"
        )
    admission = AdmissionConfigV12.load(args.config)
    cascade = RetrievalTop100CascadeV12Config.load(args.config)
    if len(site_feature_names(admission)) != 319:
        raise FoldFusedInvariantError("parity requires the exact 319-feature schema")
    cache = SceneIRCache(args.scene_ir, verify_source_hash=True)
    ledger = StrictV12LabelLedger.load(args.labels, config_path=args.config)
    reconcile_scene_ir_and_labels_v12(cache, ledger)
    sample, sample_sha = _sealed_query_sample(args.query_sample, ledger=ledger)
    scene_manifest = Path(args.scene_ir).resolve() / "manifest.json"

    with FoldFusedOfficialSidecarsV12.open(
        args.sidecars, args.mmap_sidecar, verify_hashes=True
    ) as sidecars:
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
            _bundle_manifest(args.parent_model_bundle),
            config_sha256=executor.scientific_config_sha256,
            source_candidate_sha256=str(cache.manifest["source_candidate_sha256"]),
            source_manifest_sha256=str(cache.manifest["source_manifest_sha256"]),
            sidecar_manifest_sha256=sidecars.sidecar_manifest_sha256,
            label_ledger_sha256=ledger.ledger_sha256,
            label_ledger_manifest_sha256=ledger.manifest_sha256,
        )
        lineage = {
            "scene_ir_sha256": str(cache.manifest["build_id"]),
            "scene_ir_manifest_sha256": sha256_file(scene_manifest),
            "source_candidate_sha256": str(
                cache.manifest["source_candidate_sha256"]
            ),
            "source_manifest_sha256": str(
                cache.manifest["source_manifest_sha256"]
            ),
            "label_ledger_sha256": ledger.ledger_sha256,
            "label_ledger_manifest_sha256": ledger.manifest_sha256,
            "scientific_config_sha256": executor.scientific_config_sha256,
            "executor_config_sha256": executor.sha256,
            "official_sidecar_manifest_sha256": sidecars.sidecar_manifest_sha256,
            "mmap_sidecar_manifest_sha256": sidecars.mmap_manifest_sha256,
            "reader_parity_evidence_sha256": reader_parity_sha,
            "parent_model_bundle_sha256": parent_bundle.sha256,
            "query_sample_sha256": sample_sha,
        }
        model_lineage = {
            key: value for key, value in lineage.items() if key != "query_sample_sha256"
        }
        site_models = (
            SiteModelCatalogV12.load(
                args.site_model_manifest,
                feature_names=site_feature_names(admission),
                expected_lineage=model_lineage,
            )
            if args.site_model_manifest is not None
            else None
        )
        manifest = run_fold_fused_parity_v12(
            FoldFusedParityInputsV12(
                scene_cache=cache,
                query_sample=sample,
                query_sample_path=Path(args.query_sample).resolve(),
                ledger=ledger,
                sidecars=sidecars,
                parent_scorers=parent_bundle.scorers,
                admission_config=admission,
                cascade_config=cascade,
                lineage=lineage,
                site_models=site_models,
                national_query_count=len(
                    ledger.expected_query_ids(folds=(2, 3, 4))
                ),
                maximum_spool_bytes=executor.maximum_temporary_gib * 1024**3,
            ),
            output_dir=output_dir,
            batch_size=args.batch_size,
        )
    print(str(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run sealed 512-query fold-fused v12 differential parity"
    )
    value.add_argument("--scene-ir", type=Path, required=True)
    value.add_argument("--query-sample", type=Path, required=True)
    value.add_argument("--labels", type=Path, required=True)
    value.add_argument("--sidecars", type=Path, required=True)
    value.add_argument("--mmap-sidecar", type=Path, required=True)
    value.add_argument("--reader-parity-manifest", type=Path, required=True)
    value.add_argument("--parent-model-bundle", type=Path, required=True)
    value.add_argument("--site-model-manifest", type=Path)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--executor-config", type=Path, default=DEFAULT_EXECUTOR_CONFIG)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--batch-size", type=int, default=262_144)
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
