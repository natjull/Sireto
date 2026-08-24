"""Spawn-safe zero-copy runtime and immutable functional bridge for v13.

The build and replay entrypoints retain the v12 scientific implementation and
only replace their physical sinks.  Three build actors exit before sequential
fits begin; three fresh replay actors then open only heldout fragments, labels,
models and the frozen config.  The final wrapper contains no merged Parquet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from .retrieval_fold_fused_fragment_dataset_v13 import (
    BufferedCompactTrainingStoreV13,
    XGBoostFragmentedSiteLambdaMARTBackendV13,
    build_logical_parity_digests_v13,
    fit_site_models_sequential_from_fragments_v13,
    load_build_fragment_dataset_v13,
    logical_heldout_view_from_parquet_v13,
    logical_train_view_from_parquet_v13,
    merkle_root_v13,
    seal_build_fragment_dataset_v13,
)
from .retrieval_fold_fused_fragment_outputs_v13 import (
    DEFAULT_ROW_GROUP_TARGET,
    load_compact_replay_dataset_v13,
    load_compact_replay_fragment_v13,
    logical_output_merkle_from_parquet_v13,
    prerank500_full_payload_merkle_from_parquet_v13,
    replay_fragment_v13,
    seal_compact_replay_dataset_v13,
)
from .retrieval_fold_fused_oof_v12 import (
    ExistingV12SelectionAdapter,
    FoldFusedInvariantError,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_row_group_ledger_v13 import (
    ActualRowGroupLedgerSetV13,
    ActualRowGroupLedgerSinkV13,
    ROW_GROUP_LEDGER_DIRNAME,
    ROW_GROUP_LEDGER_MANIFEST,
    actor_row_group_ledger_record_v13,
    load_actor_row_group_ledger_v13,
    validate_actual_row_group_ledger_set_v13,
)
from .retrieval_fold_fused_prefit_bridge_v12 import (
    PREFIT_BRIDGE_ROW_SCHEMA_VERSION,
    load_prefit_bridge_artifact_v12,
)
from .retrieval_fold_fused_physical_benchmark_v13 import (
    ALLOWED_FOLDS,
    CONFIG_SHA256,
    IMMUTABLE_FUNCTIONAL_PROJECTION_SCHEMA_VERSION,
    NATIONAL_QUERY_COUNT,
    REQUIRED_LOGICAL_BRIDGE_DIGESTS,
    SAMPLE_QUERY_COUNT,
    TRIAL_EVIDENCE_SCHEMA_VERSION,
    FragmentW3TrialProductV13,
    ImmutableW1AnchorV13,
    PhysicalBenchmarkInvariantErrorV13,
    _canonical_json,
    _sha256_json,
    _strict_sha_map,
    load_fragment_w3_trial_product_v13,
    load_physical_config_v13,
    v13_implementation_lineage,
)
from .retrieval_fold_fused_sharded_oof_v12 import (
    ShardBuildArtifactV12,
    ShardSpecV12,
    WorkerJobV12,
    fitted_site_model_records_v12,
    heldout_master_record_v12,
    load_fitted_site_model_records_v12,
    load_heldout_master_record_v12,
    spawn_manifest_workers_v12,
)
from .retrieval_fold_fused_sharded_runtime_v12 import (
    _ReplayFeatureContextV12,
    _ReplaySiteBackendV12,
    _canonical_json as _runtime_canonical_json,
    _caps,
    _load_xgb_handle,
    _require_common,
    _require_output,
    build_worker_entrypoint_v12,
)
from .retrieval_fold_fused_zero_copy_run_v13 import (
    CANONICAL_FAMILIES,
    load_zero_copy_canonical_run_v13,
    seal_zero_copy_canonical_run_v13,
)
from .retrieval_hierarchical_admission_v12 import site_feature_names


BUILD_ENTRYPOINT_V13 = (
    "src.xgb_matcher.retrieval_fold_fused_fragment_runtime_v13:"
    "build_worker_entrypoint_v13"
)
REPLAY_ENTRYPOINT_V13 = (
    "src.xgb_matcher.retrieval_fold_fused_fragment_runtime_v13:"
    "replay_worker_entrypoint_v13"
)
_SHA256_CHARS = frozenset("0123456789abcdef")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhysicalBenchmarkInvariantErrorV13(
            f"{label} is not readable JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise PhysicalBenchmarkInvariantErrorV13(f"{label} must be an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _relative(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class ImmutableW1FunctionalProjectionV13:
    manifest_path: Path
    manifest_sha256: str
    projection_id: str
    anchor_trial_sha256: str
    prefit_parity_manifest_sha256: str
    v13_implementation_sha256: str
    logical_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            not self.manifest_path.is_file()
            or sha256_file(self.manifest_path) != self.manifest_sha256
            or not all(
                _valid_sha256(value)
                for value in (
                    self.manifest_sha256,
                    self.projection_id,
                    self.anchor_trial_sha256,
                    self.prefit_parity_manifest_sha256,
                    self.v13_implementation_sha256,
                )
            )
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "immutable W1 functional projection is invalid"
            )
        _strict_sha_map(
            self.logical_digests,
            expected_keys=REQUIRED_LOGICAL_BRIDGE_DIGESTS,
            label="immutable W1 functional projection",
        )


def _w1_input_record(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    path = Path(str(value.get("path") or "")).resolve()
    digest = str(value.get("sha256") or "")
    if (
        not path.is_file()
        or not _valid_sha256(digest)
        or sha256_file(path) != digest
    ):
        raise PhysicalBenchmarkInvariantErrorV13(f"immutable W1 {label} drifted")
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


_PREFIT_BRIDGE_COLUMNS_V13 = (
    "schema_version",
    "query_id",
    "fold",
    "outer_fold",
    "fused_parent_sha256",
    "fused_master_sha256",
    "statistics_sha256",
    "fused_features_sha256",
    "master_count",
    "fused_fit_rows_sha256",
    "fit_sample_count",
)


def _prefit_bridge_logical_merkle_v13(
    paths: Sequence[Path],
    *,
    sample_query_ids: Sequence[str],
) -> str:
    """Hash the exact ordered 512 x 3 PREFIT bridge without a physical merge."""

    rows: list[Mapping[str, Any]] = []
    for path in paths:
        resolved = Path(path).resolve()
        table = pq.read_table(resolved, columns=list(_PREFIT_BRIDGE_COLUMNS_V13))
        if tuple(table.column_names) != _PREFIT_BRIDGE_COLUMNS_V13:
            raise PhysicalBenchmarkInvariantErrorV13(
                "PREFIT bridge logical schema drifted"
            )
        rows.extend(table.to_pylist())
    query_ids = tuple(str(value) for value in sample_query_ids)
    if (
        len(query_ids) != SAMPLE_QUERY_COUNT
        or len(set(query_ids)) != SAMPLE_QUERY_COUNT
        or len(rows) != SAMPLE_QUERY_COUNT * len(TRAIN_FOLDS)
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "PREFIT bridge logical population drifted"
        )
    for query_ordinal, query_id in enumerate(query_ids):
        group = rows[
            query_ordinal * len(TRAIN_FOLDS) :
            (query_ordinal + 1) * len(TRAIN_FOLDS)
        ]
        folds = {int(value.get("fold", -1)) for value in group}
        outer = tuple(int(value.get("outer_fold", -1)) for value in group)
        if (
            any(
                str(value.get("schema_version") or "")
                != PREFIT_BRIDGE_ROW_SCHEMA_VERSION
                or str(value.get("query_id") or "") != query_id
                or any(
                    not _valid_sha256(value.get(name))
                    for name in (
                        "fused_parent_sha256",
                        "fused_master_sha256",
                        "statistics_sha256",
                        "fused_features_sha256",
                        "fused_fit_rows_sha256",
                    )
                )
                or not 1 <= int(value.get("master_count", -1)) <= 2_000
                or not 0 <= int(value.get("fit_sample_count", -1)) <= 128
                for value in group
            )
            or len(folds) != 1
            or next(iter(folds)) not in TRAIN_FOLDS
            or outer != tuple(TRAIN_FOLDS)
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "PREFIT bridge logical query/order drifted"
            )
    return merkle_root_v13(rows)


def _prefit_site_feature_values_merkle_v13(
    *,
    prefit_parity_manifest_path: Path,
    expected_prefit_parity_manifest_sha256: str,
    expected_lineage: Mapping[str, str],
    sample_query_ids: Sequence[str],
) -> tuple[str, list[Mapping[str, Any]]]:
    """Project one heldout (outer==fold) feature digest per sample query."""

    from .retrieval_fold_fused_runtime_v12 import (
        load_prefit_parity_evidence_v12,
    )

    manifest_path = Path(prefit_parity_manifest_path).resolve()
    raw, observed_sha = load_prefit_parity_evidence_v12(
        manifest_path, expected_lineage=expected_lineage
    )
    artifacts = raw.get("artifacts") or {}
    rows_record = artifacts.get("parity_rows") or {}
    rows_path = (manifest_path.parent / str(rows_record.get("path") or "")).resolve()
    if (
        observed_sha != expected_prefit_parity_manifest_sha256
        or rows_record.get("path") != "parity_rows.parquet"
        or not rows_path.is_file()
        or rows_record.get("sha256") != sha256_file(rows_path)
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "PREFIT feature authority bytes drifted"
        )
    table = pq.read_table(
        rows_path,
        columns=["query_id", "fold", "outer_fold", "fused_features_sha256"],
    )
    heldout: dict[str, Mapping[str, Any]] = {}
    for value in table.to_pylist():
        query_id = str(value.get("query_id") or "")
        fold = int(value.get("fold", -1))
        outer = int(value.get("outer_fold", -1))
        digest = str(value.get("fused_features_sha256") or "")
        if outer != fold:
            continue
        if (
            not query_id
            or fold not in TRAIN_FOLDS
            or not _valid_sha256(digest)
            or query_id in heldout
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "PREFIT heldout feature projection drifted"
            )
        heldout[query_id] = {
            "kind": "site_feature_values",
            "query_id": query_id,
            "fold": fold,
            "fused_features_sha256": digest,
        }
    ordered_ids = tuple(str(value) for value in sample_query_ids)
    if (
        len(ordered_ids) != SAMPLE_QUERY_COUNT
        or len(set(ordered_ids)) != SAMPLE_QUERY_COUNT
        or set(ordered_ids) != set(heldout)
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "PREFIT feature projection differs from the sealed sample"
        )
    leaves = [heldout[query_id] for query_id in ordered_ids]
    return merkle_root_v13(leaves), [
        {
            "role": "prefit_parity_manifest",
            "path": str(manifest_path),
            "sha256": observed_sha,
            "size_bytes": manifest_path.stat().st_size,
        },
        {
            "role": "prefit_parity_rows",
            "path": str(rows_path),
            "sha256": str(rows_record["sha256"]),
            "size_bytes": rows_path.stat().st_size,
        },
    ]


def _canonical_query_sequence_from_output_v13(path: Path) -> tuple[str, ...]:
    """Read the immutable W1/SceneIR query order without using sample hash order."""

    parquet = pq.ParquetFile(Path(path).resolve())
    sequence: list[str] = []
    seen: set[str] = set()
    folds: list[int] = []
    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group, columns=["query_id", "fold"])
        query_values = table["query_id"].to_pylist()
        fold_values = table["fold"].to_pylist()
        start = 0
        while start < len(query_values):
            query_id = str(query_values[start])
            fold = int(fold_values[start])
            stop = start + 1
            while stop < len(query_values) and str(query_values[stop]) == query_id:
                if int(fold_values[stop]) != fold:
                    raise PhysicalBenchmarkInvariantErrorV13(
                        "immutable W1 query changes fold inside one output group"
                    )
                stop += 1
            if not query_id or query_id in seen or fold not in TRAIN_FOLDS:
                raise PhysicalBenchmarkInvariantErrorV13(
                    "immutable W1 output query sequence drifted"
                )
            seen.add(query_id)
            sequence.append(query_id)
            folds.append(fold)
            start = stop
    if (
        len(sequence) != SAMPLE_QUERY_COUNT
        or tuple(folds) != tuple(sorted(folds))
        or set(folds) != set(TRAIN_FOLDS)
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "immutable W1 output is not the canonical fold2/fold3/fold4 sequence"
        )
    return tuple(sequence)


def _compute_w1_functional_projection(
    anchor: ImmutableW1AnchorV13,
    *,
    prefit_parity_manifest_path: Path,
    expected_prefit_parity_manifest_sha256: str,
    expected_prefit_lineage: Mapping[str, str],
    sample_query_ids: Sequence[str],
) -> tuple[dict[str, str], list[Mapping[str, Any]]]:
    if sha256_file(anchor.trial_manifest_path) != anchor.trial_manifest_sha256:
        raise PhysicalBenchmarkInvariantErrorV13("immutable W1 trial bytes drifted")
    raw = _read_json(anchor.trial_manifest_path, label="immutable W1 trial")
    train = raw.get("train_matrices")
    heldout = raw.get("heldout_master")
    models = raw.get("site_models")
    outputs = raw.get("outputs")
    prefit_bridge_record = raw.get("prefit_bridge")
    if (
        raw.get("schema_version")
        != "sireto-retrieval-v12-sharded-oof-manifest-1"
        or raw.get("status") != "EXECUTED_NO_READY_CLAIM"
        or raw.get("worker_count") != 1
        or raw.get("query_count") != SAMPLE_QUERY_COUNT
        or raw.get("folds") != list(ALLOWED_FOLDS)
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or not isinstance(train, Mapping)
        or set(train) != {str(value) for value in TRAIN_FOLDS}
        or not isinstance(heldout, Mapping)
        or not isinstance(models, Mapping)
        or set(models) != {str(value) for value in TRAIN_FOLDS}
        or not isinstance(outputs, Mapping)
        or set(outputs) != {"top500", "top100", "outcomes"}
        or not isinstance(prefit_bridge_record, Mapping)
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "immutable W1 functional source contract drifted"
        )
    output_records = {
        name: _w1_input_record(outputs[name], label=name)
        for name in ("top500", "top100", "outcomes")
    }
    output_paths = {
        name: Path(str(record["path"]))
        for name, record in output_records.items()
    }
    canonical_query_sequence = _canonical_query_sequence_from_output_v13(
        output_paths["outcomes"]
    )
    heldout_record = _w1_input_record(heldout, label="heldout spool")
    if (
        _canonical_query_sequence_from_output_v13(output_paths["top500"])
        != canonical_query_sequence
        or _canonical_query_sequence_from_output_v13(output_paths["top100"])
        != canonical_query_sequence
        or _canonical_query_sequence_from_output_v13(
            Path(str(heldout_record["path"]))
        )
        != canonical_query_sequence
        or set(canonical_query_sequence)
        != {str(value) for value in sample_query_ids}
        or len(tuple(sample_query_ids)) != SAMPLE_QUERY_COUNT
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "immutable W1 output order/population differs from the sealed sample"
        )
    feature_merkle, prefit_inputs = _prefit_site_feature_values_merkle_v13(
        prefit_parity_manifest_path=prefit_parity_manifest_path,
        expected_prefit_parity_manifest_sha256=(
            expected_prefit_parity_manifest_sha256
        ),
        expected_lineage=expected_prefit_lineage,
        sample_query_ids=canonical_query_sequence,
    )
    inputs: list[Mapping[str, Any]] = list(prefit_inputs)
    w1_prefit_bridge = load_prefit_bridge_artifact_v12(
        dict(prefit_bridge_record), verify_hashes=True
    )
    if (
        w1_prefit_bridge.query_count != SAMPLE_QUERY_COUNT
        or w1_prefit_bridge.row_count != SAMPLE_QUERY_COUNT * len(TRAIN_FOLDS)
        or w1_prefit_bridge.prefit_manifest_sha256
        != expected_prefit_parity_manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "immutable W1 PREFIT bridge authority drifted"
        )
    inputs.extend(
        [
            {
                "role": "prefit_bridge_manifest",
                "path": str(w1_prefit_bridge.manifest_path),
                "sha256": w1_prefit_bridge.manifest_sha256,
                "size_bytes": w1_prefit_bridge.manifest_path.stat().st_size,
            },
            {
                "role": "prefit_bridge_rows",
                "path": str(w1_prefit_bridge.path),
                "sha256": w1_prefit_bridge.sha256,
                "size_bytes": w1_prefit_bridge.path.stat().st_size,
            },
        ]
    )
    prefit_bridge_logical = _prefit_bridge_logical_merkle_v13(
        [w1_prefit_bridge.path], sample_query_ids=canonical_query_sequence
    )
    train_views: dict[int, Mapping[str, Any]] = {}
    for outer in TRAIN_FOLDS:
        record = _w1_input_record(train[str(outer)], label=f"train outer{outer}")
        inputs.append({"role": f"train_outer{outer}", **record})
        train_views[outer] = logical_train_view_from_parquet_v13(
            [Path(str(record["path"]))], outer
        )
    inputs.append({"role": "heldout_spool", **heldout_record})
    heldout_view = logical_heldout_view_from_parquet_v13(
        [Path(str(heldout_record["path"]))]
    )
    model_sha: dict[int, str] = {}
    for outer in TRAIN_FOLDS:
        record = _w1_input_record(models[str(outer)], label=f"model outer{outer}")
        if int(models[str(outer)].get("outer_fold", -1)) != outer:
            raise PhysicalBenchmarkInvariantErrorV13("immutable W1 model fold drifted")
        inputs.append({"role": f"model_outer{outer}", **record})
        model_sha[outer] = str(record["sha256"])
    for name in ("top500", "top100", "outcomes"):
        record = output_records[name]
        inputs.append({"role": name, **record})
    logical = {
        "heldout_spool": str(heldout_view["logical_merkle_root"]),
        **{
            f"train_outer{outer}": str(
                train_views[outer]["logical_merkle_root"]
            )
            for outer in TRAIN_FOLDS
        },
        **{
            f"query_groups_outer{outer}": str(
                train_views[outer]["group_sequence_sha256"]
            )
            for outer in TRAIN_FOLDS
        },
        **{f"model_outer{outer}": model_sha[outer] for outer in TRAIN_FOLDS},
        "prefit_bridge_logical": prefit_bridge_logical,
        "site_feature_values": feature_merkle,
        "prerank500_common": logical_output_merkle_from_parquet_v13(
            [output_paths["top500"]], "top500"
        ),
        "prerank500_full_payload": (
            prerank500_full_payload_merkle_from_parquet_v13(
                [output_paths["top500"]]
            )
        ),
        "top100_full": logical_output_merkle_from_parquet_v13(
            [output_paths["top100"]], "top100"
        ),
        "outcomes": logical_output_merkle_from_parquet_v13(
            [output_paths["outcomes"]], "outcomes"
        ),
    }
    return (
        _strict_sha_map(
            logical,
            expected_keys=REQUIRED_LOGICAL_BRIDGE_DIGESTS,
            label="immutable W1 functional projection",
        ),
        inputs,
    )


def build_immutable_w1_functional_projection_v13(
    *,
    anchor: ImmutableW1AnchorV13,
    workspace_root: Path,
    output_root: Path,
    prefit_parity_manifest_path: Path,
    expected_prefit_parity_manifest_sha256: str,
    expected_prefit_lineage: Mapping[str, str],
    sample_query_ids: Sequence[str],
) -> ImmutableW1FunctionalProjectionV13:
    """Project, but never rerun, the exact sealed W1 scientific artifacts."""

    implementation_sha, implementation_files = v13_implementation_lineage(
        workspace_root
    )
    logical, inputs = _compute_w1_functional_projection(
        anchor,
        prefit_parity_manifest_path=prefit_parity_manifest_path,
        expected_prefit_parity_manifest_sha256=(
            expected_prefit_parity_manifest_sha256
        ),
        expected_prefit_lineage=expected_prefit_lineage,
        sample_query_ids=sample_query_ids,
    )
    identity = {
        "anchor_trial_sha256": anchor.trial_manifest_sha256,
        "prefit_parity_manifest_sha256": (
            expected_prefit_parity_manifest_sha256
        ),
        "v13_implementation_sha256": implementation_sha,
        "implementation_files": list(implementation_files),
        "inputs": inputs,
        "logical_digests": logical,
    }
    projection_id = _sha256_json(identity)
    root = Path(output_root).resolve()
    target = root / projection_id[:16]
    if target.exists():
        return load_immutable_w1_functional_projection_v13(
            target / "manifest.json",
            expected_manifest_sha256=sha256_file(target / "manifest.json"),
            expected_anchor_trial_sha256=anchor.trial_manifest_sha256,
            verify_inputs=True,
        )
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".w1-functional-v13-", dir=root))
    try:
        manifest = {
            "schema_version": IMMUTABLE_FUNCTIONAL_PROJECTION_SCHEMA_VERSION,
            "status": "SEALED_IMMUTABLE_W1_FUNCTIONAL_PROJECTION_V13",
            "projection_id": projection_id,
            "identity": identity,
            "folds": list(ALLOWED_FOLDS),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            **identity,
        }
        _atomic_json(stage / "manifest.json", manifest)
        os.rename(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return load_immutable_w1_functional_projection_v13(
        target / "manifest.json",
        expected_manifest_sha256=sha256_file(target / "manifest.json"),
        expected_anchor_trial_sha256=anchor.trial_manifest_sha256,
        verify_inputs=True,
    )


def load_immutable_w1_functional_projection_v13(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_anchor_trial_sha256: str | None,
    verify_inputs: bool = True,
) -> ImmutableW1FunctionalProjectionV13:
    path = Path(manifest_path).resolve()
    if (
        not path.is_file()
        or not _valid_sha256(expected_manifest_sha256)
        or sha256_file(path) != expected_manifest_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "immutable W1 functional manifest bytes drifted"
        )
    raw = _read_json(path, label="immutable W1 functional projection")
    identity = raw.get("identity")
    logical = raw.get("logical_digests")
    inputs = raw.get("inputs")
    implementation_files = raw.get("implementation_files")
    expected_keys = {
        "schema_version",
        "status",
        "projection_id",
        "identity",
        "folds",
        "fold0_opened",
        "fold1_opened",
        "positive_injection",
        "anchor_trial_sha256",
        "prefit_parity_manifest_sha256",
        "v13_implementation_sha256",
        "implementation_files",
        "inputs",
        "logical_digests",
    }
    if (
        set(raw) != expected_keys
        or raw.get("schema_version")
        != IMMUTABLE_FUNCTIONAL_PROJECTION_SCHEMA_VERSION
        or raw.get("status")
        != "SEALED_IMMUTABLE_W1_FUNCTIONAL_PROJECTION_V13"
        or not isinstance(identity, Mapping)
        or raw.get("projection_id") != _sha256_json(identity)
        or path.parent.name != str(raw.get("projection_id") or "")[:16]
        or raw.get("folds") != list(ALLOWED_FOLDS)
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or not _valid_sha256(raw.get("anchor_trial_sha256"))
        or not _valid_sha256(raw.get("prefit_parity_manifest_sha256"))
        or (
            expected_anchor_trial_sha256 is not None
            and raw.get("anchor_trial_sha256") != expected_anchor_trial_sha256
        )
        or not _valid_sha256(raw.get("v13_implementation_sha256"))
        or not isinstance(implementation_files, list)
        or not isinstance(inputs, list)
        or not isinstance(logical, Mapping)
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "immutable W1 functional projection contract drifted"
        )
    expected_identity = {
        "anchor_trial_sha256": raw["anchor_trial_sha256"],
        "prefit_parity_manifest_sha256": raw[
            "prefit_parity_manifest_sha256"
        ],
        "v13_implementation_sha256": raw["v13_implementation_sha256"],
        "implementation_files": implementation_files,
        "inputs": inputs,
        "logical_digests": logical,
    }
    if dict(identity) != expected_identity:
        raise PhysicalBenchmarkInvariantErrorV13(
            "immutable W1 functional projection identity drifted"
        )
    for record in inputs:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"role", "path", "sha256", "size_bytes"}
            or not isinstance(record.get("role"), str)
            or not record.get("role")
            or not _valid_sha256(record.get("sha256"))
            or type(record.get("size_bytes")) is not int
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "immutable W1 functional input record drifted"
            )
        source = Path(str(record.get("path") or "")).resolve()
        if (
            not source.is_file()
            or source.stat().st_size != int(record["size_bytes"])
            or (verify_inputs and sha256_file(source) != record["sha256"])
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "immutable W1 functional input bytes drifted"
            )
    return ImmutableW1FunctionalProjectionV13(
        manifest_path=path,
        manifest_sha256=expected_manifest_sha256,
        projection_id=str(raw["projection_id"]),
        anchor_trial_sha256=str(raw["anchor_trial_sha256"]),
        prefit_parity_manifest_sha256=str(
            raw["prefit_parity_manifest_sha256"]
        ),
        v13_implementation_sha256=str(raw["v13_implementation_sha256"]),
        logical_digests=_strict_sha_map(
            logical,
            expected_keys=REQUIRED_LOGICAL_BRIDGE_DIGESTS,
            label="immutable W1 functional projection",
        ),
    )


def build_worker_entrypoint_v13(job: Mapping[str, Any]) -> str:
    """Run v12 science and seal every actual source row-group cache miss."""

    if job.get("phase") != "BUILD" or not isinstance(job.get("common"), Mapping):
        raise FoldFusedInvariantError("invalid v13 build job")
    spec = ShardSpecV12.from_manifest(dict(job.get("spec") or {}))
    common = job["common"]
    paths = common.get("paths")
    if not isinstance(paths, Mapping):
        raise FoldFusedInvariantError("v13 build job lacks runtime paths")
    workspace_root = Path(str(paths.get("workspace_root") or "")).resolve()
    config = load_physical_config_v13(
        workspace_root / "config/retrieval_all_human_v13_physical_execution.json"
    )
    projection = config["cache_neutral_io_gate"]["source_extent"][
        "production_projection"
    ]
    sink = ActualRowGroupLedgerSinkV13(
        shard_spec=spec,
        execution_lineage={
            str(key): str(value)
            for key, value in dict(job.get("lineage") or {}).items()
        },
        sharded_execution_plan_sha256=str(
            common.get("sharded_execution_plan_sha256") or ""
        ),
        expected_projection_sha256=str(projection.get("sha256") or ""),
        expected_projection_column_count=int(
            projection.get("projected_column_count", -1)
        ),
    )
    manifest = build_worker_entrypoint_v12(
        job,
        train_store_factory=BufferedCompactTrainingStoreV13,
        row_group_observer=sink,
    )
    output = Path(str(job.get("output_dir") or "")).resolve()
    sink.seal(output / ROW_GROUP_LEDGER_DIRNAME)
    return manifest


def replay_worker_entrypoint_v13(job: Mapping[str, Any]) -> str:
    """Replay one fold-owned fragment without opening generation authorities."""

    if job.get("phase") != "REPLAY" or not isinstance(job.get("common"), Mapping):
        raise FoldFusedInvariantError("invalid v13 compact replay job")
    lineage = dict(job.get("lineage") or {})
    _paths_value, executor, admission, _cascade, ledger = _require_common(
        job["common"], lineage, open_generation_inputs=False
    )
    spec = ShardSpecV12.from_manifest(dict(job.get("spec") or {}))
    caps = _caps(dict(job.get("caps") or {}))
    output = _require_output(job.get("output_dir"), scratch_root=executor.scratch_root)
    relocation_root_value = job.get("relocation_root")
    relocation_root = (
        Path(str(relocation_root_value)).resolve()
        if relocation_root_value is not None
        else None
    )
    if relocation_root is not None:
        try:
            output.relative_to(relocation_root)
        except ValueError as error:
            raise FoldFusedInvariantError(
                "v13 replay output escaped its relocation root"
            ) from error
    build_manifest_path = Path(str(job.get("build_manifest_path") or "")).resolve()
    build_manifest_sha = str(job.get("build_manifest_sha256") or "")
    heldout_merkle = str(job.get("heldout_dataset_merkle_root") or "")
    if (
        not build_manifest_path.is_file()
        or not _valid_sha256(build_manifest_sha)
        or sha256_file(build_manifest_path) != build_manifest_sha
        or not _valid_sha256(heldout_merkle)
    ):
        raise FoldFusedInvariantError("v13 replay build/dataset lineage drifted")
    heldout = load_heldout_master_record_v12(
        dict(job.get("heldout") or {}), verify_hashes=True
    )
    if len(spec.folds) != 1:
        raise FoldFusedInvariantError(
            "v13 replay actor must own exactly one heldout fold/model"
        )
    models = load_fitted_site_model_records_v12(
        dict(job.get("models") or {}),
        handle_loader=_load_xgb_handle,
        verify_hashes=True,
        required_folds=spec.folds,
    )
    expected_features = tuple(site_feature_names(admission))
    expected_recipe = hashlib.sha256(
        _runtime_canonical_json(dict(admission.xgb_params("site"))).encode("utf-8")
    ).hexdigest()
    if any(
        model.feature_names != expected_features
        or model.recipe_sha256 != expected_recipe
        for model in models.values()
    ):
        raise FoldFusedInvariantError(
            "v13 replay model recipe/319-feature schema drifted"
        )
    artifact = replay_fragment_v13(
        spec,
        heldout=heldout,
        labels=ledger,
        contexts=_ReplayFeatureContextV12(admission),
        model_backend=_ReplaySiteBackendV12(),
        models=models,
        selector=ExistingV12SelectionAdapter(admission),
        output_dir=output,
        lineage=lineage,
        build_manifest_sha256=build_manifest_sha,
        heldout_dataset_merkle_root=heldout_merkle,
        caps=caps,
        row_group_target=int(job.get("row_group_target", DEFAULT_ROW_GROUP_TARGET)),
        relocation_root=relocation_root,
    )
    # The shared subprocess launcher requires a historical ``manifest.json``.
    # The compact manifest is a sibling and the coordinator loads it explicitly.
    return str(artifact.scientific_manifest_path)


def _replay_jobs_v13(
    *,
    prepared: Any,
    specs: Sequence[ShardSpecV12],
    builds: Sequence[ShardBuildArtifactV12],
    models: Mapping[int, Any],
    heldout_merkle: str,
    lineage: Mapping[str, str],
    output: Path,
    execution_phase: str = "PHYSICAL_V13_W3_COMPACT_REPLAY_512",
) -> tuple[WorkerJobV12, ...]:
    return tuple(
        WorkerJobV12(
            job_id=f"replay-{spec.shard_id}",
            payload={
                "phase": "REPLAY",
                "spec": spec.as_manifest(),
                "common": dict(prepared.runtime_payload),
                "lineage": dict(lineage),
                "caps": asdict(prepared.caps),
                "output_dir": str(output / "worker_replay" / spec.shard_id),
                "relocation_root": str(output),
                "build_manifest_path": str(build.manifest_path),
                "build_manifest_sha256": build.manifest_sha256,
                "heldout": heldout_master_record_v12(build.heldout_master),
                "models": fitted_site_model_records_v12(
                    {spec.folds[0]: models[spec.folds[0]]},
                    required_folds=spec.folds,
                ),
                "heldout_dataset_merkle_root": heldout_merkle,
                "row_group_target": DEFAULT_ROW_GROUP_TARGET,
                "execution_evidence": {
                    "phase": execution_phase,
                    "physical_config_sha256": CONFIG_SHA256,
                },
            },
        )
        for spec, build in zip(specs, builds, strict=True)
    )


def _w3_logical_digests(
    build: Any,
    replay: Any,
    models: Mapping[int, Any],
    build_artifacts: Sequence[ShardBuildArtifactV12],
    sample_query_ids: Sequence[str],
    expected_prefit_parity_manifest_sha256: str,
) -> dict[str, str]:
    parity = build_logical_parity_digests_v13(build)
    canonical_query_sequence = tuple(
        query_id for spec in build.specs for query_id in spec.query_ids
    )
    if (
        len(canonical_query_sequence) != SAMPLE_QUERY_COUNT
        or set(canonical_query_sequence)
        != {str(value) for value in sample_query_ids}
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "fresh W3 SceneIR order/population differs from the sealed sample"
        )
    bridges = tuple(value.prefit_bridge for value in build_artifacts)
    if (
        len(bridges) != 3
        or any(value is None for value in bridges)
        or any(
            value.prefit_manifest_sha256
            != expected_prefit_parity_manifest_sha256
            or value.row_count != value.query_count * len(TRAIN_FOLDS)
            for value in bridges
            if value is not None
        )
        or len(
            {
                (
                    value.prefit_manifest_sha256,
                    value.reference_manifest_sha256,
                    value.parity_rows_sha256,
                )
                for value in bridges
                if value is not None
            }
        )
        != 1
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "fresh W3 PREFIT bridge fragments drifted"
        )
    prefit_bridge_logical = _prefit_bridge_logical_merkle_v13(
        [value.path for value in bridges if value is not None],
        sample_query_ids=canonical_query_sequence,
    )
    logical = {
        "heldout_spool": str(parity["heldout_spool"]),
        **{
            f"train_outer{outer}": str(parity["train_matrix"][str(outer)])
            for outer in TRAIN_FOLDS
        },
        **{
            f"query_groups_outer{outer}": str(
                parity["group_sequence"][str(outer)]
            )
            for outer in TRAIN_FOLDS
        },
        **{f"model_outer{outer}": models[outer].sha256 for outer in TRAIN_FOLDS},
        "prefit_bridge_logical": prefit_bridge_logical,
        "site_feature_values": replay.feature_values_logical_merkle_root,
        "prerank500_common": replay.logical_merkle_root["top500"],
        "prerank500_full_payload": (
            replay.prerank500_full_payload_logical_merkle_root
        ),
        "top100_full": replay.logical_merkle_root["top100"],
        "outcomes": replay.logical_merkle_root["outcomes"],
    }
    return _strict_sha_map(
        logical,
        expected_keys=REQUIRED_LOGICAL_BRIDGE_DIGESTS,
        label="fresh W3 functional projection",
    )


def _write_trial_evidence_v13(
    *,
    output: Path,
    workspace_root: Path,
    lineage: Mapping[str, str],
    projection: ImmutableW1FunctionalProjectionV13,
    build: Any,
    replay: Any,
    canonical: Any,
    sharded_execution_plan_sha256: str,
) -> Path:
    implementation_sha, implementation_files = v13_implementation_lineage(
        workspace_root
    )
    if lineage.get("v13_implementation_sha256") != implementation_sha:
        raise PhysicalBenchmarkInvariantErrorV13(
            "trial evidence implementation differs from actor lineage"
        )
    path = output / "evidence" / "trial_evidence.json"
    _atomic_json(
        path,
        {
            "schema_version": TRIAL_EVIDENCE_SCHEMA_VERSION,
            "lineage": dict(lineage),
            "implementation_files": list(implementation_files),
            "immutable_w1_functional_projection": {
                "path": str(projection.manifest_path),
                "sha256": projection.manifest_sha256,
            },
            "prefit_parity_manifest_sha256": (
                projection.prefit_parity_manifest_sha256
            ),
            "sharded_execution_plan_sha256": sharded_execution_plan_sha256,
            "build_fragment_dataset": {
                "path": _relative(build.manifest_path, output),
                "sha256": build.manifest_sha256,
            },
            "compact_replay_dataset": {
                "path": _relative(replay.manifest_path, output),
                "sha256": replay.manifest_sha256,
            },
            "zero_copy_canonical_run": {
                "path": _relative(canonical.manifest_path, output),
                "sha256": canonical.manifest_sha256,
            },
            "compact_rehydration": {
                "status": "PENDING_ONE_INDEPENDENT_POST_TIMING_RECONSTRUCTION",
                "top500_row_count": sum(
                    value.row_count for value in replay.outputs["top500"]
                ),
                "feature_values_logical_merkle_root": (
                    replay.feature_values_logical_merkle_root
                ),
                "prerank500_full_payload_logical_merkle_root": (
                    replay.prerank500_full_payload_logical_merkle_root
                ),
            },
        },
    )
    return path


def _inventory_role_family(
    relative: str,
    *,
    family_paths: Mapping[str, str],
    special_paths: Mapping[str, tuple[str, str]],
) -> tuple[str, str]:
    if relative in family_paths:
        return "dataset_manifest", family_paths[relative]
    if relative in special_paths:
        return special_paths[relative]
    parts = Path(relative).parts
    if "site_models" in parts:
        return "model", "model_by_outer"
    if "worker_build" in parts:
        return "fragment_or_manifest", "build_fragments"
    if "worker_replay" in parts:
        return "fragment_or_manifest", "replay_fragments"
    return "support", "support"


def load_and_validate_build_row_group_ledgers_v13(
    *,
    prepared: Any,
    specs: Sequence[ShardSpecV12],
    build_artifacts: Sequence[ShardBuildArtifactV12],
    lineage: Mapping[str, str],
    expected_unique_row_groups: int | None = None,
    expected_logical_compressed_bytes: int | None = None,
    enforce_gate_totals: bool = True,
) -> ActualRowGroupLedgerSetV13:
    """Strictly reconcile actual build reads before fit/replay can continue."""

    if len(specs) != 3 or len(build_artifacts) != 3:
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 actual-read proof requires exactly three build actors"
        )
    config = load_physical_config_v13(
        Path(prepared.runtime_payload["paths"]["workspace_root"])
        / "config/retrieval_all_human_v13_physical_execution.json"
    )
    source_extent = config["cache_neutral_io_gate"]["source_extent"]
    projection = source_extent["production_projection"]
    records: list[tuple[Path, str]] = []
    for spec, build in zip(specs, build_artifacts, strict=True):
        manifest = (
            build.manifest_path.parent
            / ROW_GROUP_LEDGER_DIRNAME
            / ROW_GROUP_LEDGER_MANIFEST
        ).resolve()
        if not manifest.is_file():
            raise PhysicalBenchmarkInvariantErrorV13(
                f"v13 build actor {spec.shard_id} lacks actual-read evidence"
            )
        records.append((manifest, sha256_file(manifest)))
    return validate_actual_row_group_ledger_set_v13(
        records,
        expected_specs=specs,
        source_path=Path(prepared.scene_cache.source).resolve(),
        expected_source_candidate_sha256=str(
            prepared.scene_cache.manifest["source_candidate_sha256"]
        ),
        expected_projection_columns=tuple(projection["columns"]),
        expected_projection_sha256=str(projection["sha256"]),
        expected_sharded_execution_plan_sha256=str(
            prepared.sharded_execution_plan_sha256
        ),
        expected_execution_lineage=lineage,
        expected_unique_row_groups=(
            int(source_extent["sealed_sample_expected_unique_row_groups"])
            if expected_unique_row_groups is None
            else int(expected_unique_row_groups)
        ),
        expected_logical_compressed_bytes=(
            int(
                source_extent[
                    "sealed_sample_expected_logical_compressed_bytes"
                ]
            )
            if expected_logical_compressed_bytes is None
            else int(expected_logical_compressed_bytes)
        ),
        verify_source_hash=False,
        enforce_gate_totals=enforce_gate_totals,
    )


def _write_w3_wrapper_v13(
    *,
    output: Path,
    prepared: Any,
    specs: Sequence[ShardSpecV12],
    logical: Mapping[str, str],
    canonical: Any,
    build: Any,
    replay: Any,
    evidence_path: Path,
    row_group_ledgers: ActualRowGroupLedgerSetV13,
) -> Path:
    manifest_path = output / "manifest.json"
    family_paths = {
        _relative(value.path, output): family
        for family, value in canonical.canonical_datasets.items()
    }
    special_paths = {
        _relative(evidence_path, output): ("trial_evidence", "support"),
        _relative(canonical.manifest_path, output): (
            "zero_copy_canonical_run",
            "support",
        ),
        _relative(build.manifest_path, output): (
            "build_fragment_dataset",
            "support",
        ),
        _relative(replay.manifest_path, output): (
            "compact_replay_dataset",
            "support",
        ),
    }
    ledger_records = [
        actor_row_group_ledger_record_v13(value, root=output)
        for value in row_group_ledgers.artifacts
    ]
    for record in ledger_records:
        special_paths[str(record["manifest_path"])] = (
            "actual_row_group_ledger_manifest",
            "source_read_ledgers",
        )
        special_paths[str(record["parquet_path"])] = (
            "actual_row_group_ledger_rows",
            "source_read_ledgers",
        )
    inventory: list[dict[str, Any]] = []
    inodes: set[tuple[int, int]] = set()
    for child in sorted(output.rglob("*")):
        if child.is_symlink():
            raise PhysicalBenchmarkInvariantErrorV13(
                "v13 wrapper refuses symlinks"
            )
        if not child.is_file() or child == manifest_path:
            continue
        relative = child.relative_to(output).as_posix()
        if any(
            part.startswith(("merged_", "canonical_merged_"))
            for part in Path(relative).parts
        ):
            raise PhysicalBenchmarkInvariantErrorV13(
                "v13 wrapper refuses merged artifacts"
            )
        stat = child.stat()
        inode = (int(stat.st_dev), int(stat.st_ino))
        if inode in inodes or int(stat.st_nlink) != 1:
            raise PhysicalBenchmarkInvariantErrorV13(
                "v13 wrapper refuses hardlinked artifacts"
            )
        inodes.add(inode)
        role, family = _inventory_role_family(
            relative,
            family_paths=family_paths,
            special_paths=special_paths,
        )
        inventory.append(
            {
                "role": role,
                "family": family,
                "path": relative,
                "sha256": sha256_file(child),
                "size_bytes": stat.st_size,
            }
        )
    datasets = {
        family: {
            "path": _relative(value.path, output),
            "sha256": value.sha256,
            "size_bytes": value.size_bytes,
            "merkle_root": value.merkle_root,
            "rows": value.rows,
            "fragment_count": value.fragment_count,
        }
        for family, value in canonical.canonical_datasets.items()
    }
    if tuple(datasets) != CANONICAL_FAMILIES:
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 wrapper requires exactly seven canonical datasets"
        )
    config = load_physical_config_v13(
        Path(prepared.runtime_payload["paths"]["workspace_root"])
        / "config/retrieval_all_human_v13_physical_execution.json"
    )
    projection_sha = config["cache_neutral_io_gate"]["source_extent"][
        "production_projection"
    ]["sha256"]
    source_path = Path(prepared.scene_cache.source).resolve()
    wrapper: dict[str, Any] = {
        "schema_version": "sireto-retrieval-v13-sharded-run-1",
        "run_kind": "PHYSICAL_BENCHMARK_W3_V13",
        "query_count": SAMPLE_QUERY_COUNT,
        "worker_count": 3,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "source_candidate": {
            "path": str(source_path),
            "sha256": str(prepared.scene_cache.manifest["source_candidate_sha256"]),
            "projection_sha256": projection_sha,
        },
        "source_specs": [value.as_manifest() for value in specs],
        "source_row_group_ledgers": ledger_records,
        "canonical_datasets": datasets,
        "artifact_inventory": inventory,
        "materialized_output_bytes": 0,
        "logical_digests": dict(logical),
        "no_merged_files": True,
    }
    child_bytes = sum(int(value["size_bytes"]) for value in inventory)
    for _iteration in range(16):
        _atomic_json(manifest_path, wrapper)
        if manifest_path.stat().st_nlink != 1:
            raise PhysicalBenchmarkInvariantErrorV13(
                "v13 wrapper root manifest must not be hardlinked"
            )
        observed = child_bytes + manifest_path.stat().st_size
        if wrapper["materialized_output_bytes"] == observed:
            return manifest_path
        wrapper["materialized_output_bytes"] = observed
    raise PhysicalBenchmarkInvariantErrorV13(
        "v13 wrapper materialized byte fixed point did not converge"
    )


def finish_fragment_w3_trial_v13(
    *,
    prepared: Any,
    specs: Sequence[ShardSpecV12],
    build_artifacts: Sequence[ShardBuildArtifactV12],
    lineage: Mapping[str, str],
    output_dir: Path,
    anchor: ImmutableW1AnchorV13,
    workspace_root: Path,
    w1_functional_projection: ImmutableW1FunctionalProjectionV13,
) -> FragmentW3TrialProductV13:
    """Finish build -> sequential fit -> compact replay -> exact wrapper."""

    output = Path(output_dir).resolve()
    root = Path(workspace_root).resolve()
    if (
        len(specs) != 3
        or len(build_artifacts) != 3
        or w1_functional_projection.anchor_trial_sha256
        != anchor.trial_manifest_sha256
        or w1_functional_projection.prefit_parity_manifest_sha256
        != prepared.prefit_parity_manifest_sha256
        or lineage.get("v13_implementation_sha256")
        != w1_functional_projection.v13_implementation_sha256
    ):
        raise PhysicalBenchmarkInvariantErrorV13(
            "v13 finisher inputs differ from frozen W3/W1 lineage"
        )
    build = seal_build_fragment_dataset_v13(
        build_artifacts,
        output / "build_fragment_dataset",
        relocation_root=output,
    )
    build = load_build_fragment_dataset_v13(
        build.manifest_path, verify_hashes=True, verify_logical=True
    )
    row_group_ledgers = load_and_validate_build_row_group_ledgers_v13(
        prepared=prepared,
        specs=specs,
        build_artifacts=build_artifacts,
        lineage=lineage,
        enforce_gate_totals=False,
    )
    backend = XGBoostFragmentedSiteLambdaMARTBackendV13(
        params=prepared.admission.xgb_params("site"),
        output_dir=output / "site_models",
    )
    models = fit_site_models_sequential_from_fragments_v13(backend, build)
    replay_jobs = _replay_jobs_v13(
        prepared=prepared,
        specs=specs,
        builds=build_artifacts,
        models=models,
        heldout_merkle=build.heldout_logical_merkle_root,
        lineage=lineage,
        output=output,
    )
    scientific_manifests = spawn_manifest_workers_v12(
        REPLAY_ENTRYPOINT_V13, replay_jobs, max_workers=3
    )
    replay_fragments = tuple(
        load_compact_replay_fragment_v13(
            path.parent / "compact_fragment_manifest.json",
            verify_hashes=True,
            verify_logical=True,
        )
        for path in scientific_manifests
    )
    replay = seal_compact_replay_dataset_v13(
        replay_fragments,
        output / "compact_replay_dataset",
        relocation_root=output,
    )
    replay = load_compact_replay_dataset_v13(
        replay.manifest_path, verify_hashes=True, verify_logical=True
    )
    canonical = seal_zero_copy_canonical_run_v13(
        build,
        replay,
        models,
        output / "canonical_datasets",
        relocation_root=output,
    )
    canonical = load_zero_copy_canonical_run_v13(
        canonical.manifest_path, verify_hashes=True, verify_logical=True
    )
    logical = _w3_logical_digests(
        build,
        replay,
        models,
        build_artifacts,
        prepared.sample_query_ids,
        prepared.prefit_parity_manifest_sha256,
    )
    evidence_path = _write_trial_evidence_v13(
        output=output,
        workspace_root=root,
        lineage=lineage,
        projection=w1_functional_projection,
        build=build,
        replay=replay,
        canonical=canonical,
        sharded_execution_plan_sha256=(
            prepared.sharded_execution_plan_sha256
        ),
    )
    wrapper = _write_w3_wrapper_v13(
        output=output,
        prepared=prepared,
        specs=specs,
        logical=logical,
        canonical=canonical,
        build=build,
        replay=replay,
        evidence_path=evidence_path,
        row_group_ledgers=row_group_ledgers,
    )
    # The in-timer reload verifies every sealed fragment/inventory/Merkle, but
    # deliberately does not recompute 319 features.  The benchmark runner does
    # that independent reconstruction exactly once after its timer stops.
    return load_fragment_w3_trial_product_v13(
        wrapper,
        workspace_root=root,
        expected_manifest_sha256=sha256_file(wrapper),
        expected_anchor_trial_sha256=anchor.trial_manifest_sha256,
        verify_logical=True,
        verify_rehydration=False,
    )


__all__ = [
    "BUILD_ENTRYPOINT_V13",
    "ImmutableW1FunctionalProjectionV13",
    "REPLAY_ENTRYPOINT_V13",
    "build_immutable_w1_functional_projection_v13",
    "build_worker_entrypoint_v13",
    "finish_fragment_w3_trial_v13",
    "load_immutable_w1_functional_projection_v13",
    "load_and_validate_build_row_group_ledgers_v13",
    "replay_worker_entrypoint_v13",
]
