"""Seven-family canonical manifest for the zero-copy retrieval-v13 run.

Only small JSON manifests are created here.  Train matrices, heldout Masters,
models and replay outputs remain in their original worker/model locations and
are referenced once through a globally deduplicated artifact inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import os

from .retrieval_fold_fused_fragment_dataset_v13 import (
    BuildFragmentDatasetV13,
    build_logical_parity_digests_v13,
    load_build_fragment_dataset_v13,
    merkle_root_v13,
)
from .retrieval_fold_fused_fragment_outputs_v13 import (
    CompactReplayDatasetV13,
    load_compact_replay_dataset_v13,
    model_set_logical_sha256_v13,
)
from .retrieval_fold_fused_oof_v12 import (
    FittedSiteModel,
    FoldFusedInvariantError,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_relocatable_paths_v13 import (
    relocatable_record_path_v13,
)


SCHEMA_VERSION = "sireto-retrieval-v13-zero-copy-canonical-run-1"
FAMILY_SCHEMA_VERSION = "sireto-retrieval-v13-canonical-dataset-family-1"
ENGINE_ID = "FOLD_FUSED_ZERO_COPY_CANONICAL_RUN_V13_1"
CANONICAL_FAMILIES = (
    "train_outer2",
    "train_outer3",
    "train_outer4",
    "heldout_spool",
    "prerank500_compact",
    "top100_full",
    "outcomes",
)
ARTIFACT_FAMILIES = (*CANONICAL_FAMILIES, "model_by_outer")
_SHA256_CHARS = frozenset("0123456789abcdef")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _relative_or_absolute(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _record_path_v13(
    path: Path,
    root: Path,
    relocation_root: Path | None,
) -> str:
    if relocation_root is None:
        return _relative_or_absolute(path, root)
    return relocatable_record_path_v13(
        path, record_root=root, run_root=relocation_root
    )


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


@dataclass(frozen=True)
class ArtifactLeafV13:
    family: str
    fold_view: str
    fragment_ordinal: int
    query_start: int
    query_stop: int
    path: Path
    sha256: str
    size_bytes: int
    rows: int
    schema_sha256: str

    def __post_init__(self) -> None:
        if (
            self.family not in ARTIFACT_FAMILIES
            or not self.fold_view
            or self.fragment_ordinal < 0
            or self.query_start < 0
            or self.query_stop <= self.query_start
            or not _valid_sha256(self.sha256)
            or not _valid_sha256(self.schema_sha256)
            or self.size_bytes <= 0
            or self.rows <= 0
        ):
            raise FoldFusedInvariantError("invalid v13 canonical artifact leaf")


@dataclass(frozen=True)
class DatasetManifestV13:
    family: str
    path: Path
    sha256: str
    size_bytes: int
    merkle_root: str
    rows: int
    fragment_count: int
    materialized_bytes: int
    logical_digest: str
    leaf_files: tuple[ArtifactLeafV13, ...]

    def __post_init__(self) -> None:
        if (
            self.family not in CANONICAL_FAMILIES
            or not all(
                _valid_sha256(value)
                for value in (self.sha256, self.merkle_root, self.logical_digest)
            )
            or self.size_bytes <= 0
            or self.rows != sum(value.rows for value in self.leaf_files)
            or self.fragment_count != len(self.leaf_files)
            or self.materialized_bytes
            != sum(value.size_bytes for value in self.leaf_files)
            or self.fragment_count <= 0
        ):
            raise FoldFusedInvariantError("invalid v13 DatasetManifest")


@dataclass(frozen=True)
class ZeroCopyCanonicalRunV13:
    manifest_path: Path
    manifest_sha256: str
    run_id: str
    lineage: Mapping[str, str]
    canonical_datasets: Mapping[str, DatasetManifestV13]
    artifact_inventory: tuple[ArtifactLeafV13, ...]
    query_count: int
    query_sequence_sha256: str
    logical_parity_digests: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not _valid_sha256(self.manifest_sha256)
            or not _valid_sha256(self.run_id)
            or not _valid_sha256(self.query_sequence_sha256)
            or tuple(self.canonical_datasets) != CANONICAL_FAMILIES
            or self.query_count <= 0
            or not self.artifact_inventory
        ):
            raise FoldFusedInvariantError("invalid v13 canonical run manifest")


def _leaf_record(
    value: ArtifactLeafV13,
    root: Path,
    relocation_root: Path | None = None,
) -> dict[str, Any]:
    return {
        "family": value.family,
        "fold_view": value.fold_view,
        "fragment_ordinal": value.fragment_ordinal,
        "query_start": value.query_start,
        "query_stop": value.query_stop,
        "path": _record_path_v13(value.path, root, relocation_root),
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "rows": value.rows,
        "schema_sha256": value.schema_sha256,
    }


def _leaf_identity(value: ArtifactLeafV13) -> dict[str, Any]:
    """Path-independent Merkle/content identity; paths stay operational only."""

    return {
        "family": value.family,
        "fold_view": value.fold_view,
        "fragment_ordinal": value.fragment_ordinal,
        "query_start": value.query_start,
        "query_stop": value.query_stop,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "rows": value.rows,
        "schema_sha256": value.schema_sha256,
    }


def _family_leaves(
    build: BuildFragmentDatasetV13,
    replay: CompactReplayDatasetV13,
    models: Mapping[int, FittedSiteModel],
) -> Mapping[str, tuple[ArtifactLeafV13, ...]]:
    output: dict[str, list[ArtifactLeafV13]] = {
        family: [] for family in ARTIFACT_FAMILIES
    }
    specs = {value.shard_ordinal: value for value in build.specs}
    for outer in TRAIN_FOLDS:
        ordinal = 0
        for fragment in build.train[outer].fragments:
            spec = specs[fragment.shard_ordinal]
            train_family = f"train_outer{outer}"
            output[train_family].append(
                ArtifactLeafV13(
                    family=train_family,
                    fold_view=f"outer{outer}",
                    fragment_ordinal=ordinal,
                    query_start=spec.sequence_start,
                    query_stop=spec.sequence_stop,
                    path=fragment.path,
                    sha256=fragment.sha256,
                    size_bytes=fragment.size_bytes,
                    rows=fragment.row_count,
                    schema_sha256=fragment.schema_sha256,
                )
            )
            ordinal += 1
    for fragment, spec in zip(
        build.heldout_fragments, build.specs, strict=True
    ):
        output["heldout_spool"].append(
            ArtifactLeafV13(
                family="heldout_spool",
                fold_view="fold2_3_4",
                fragment_ordinal=fragment.shard_ordinal,
                query_start=spec.sequence_start,
                query_stop=spec.sequence_stop,
                path=fragment.path,
                sha256=fragment.sha256,
                size_bytes=fragment.size_bytes,
                rows=fragment.row_count,
                schema_sha256=fragment.schema_sha256,
            )
        )
    for ordinal, outer in enumerate(TRAIN_FOLDS):
        model = models[outer]
        path = Path(model.path).resolve()
        output["model_by_outer"].append(
            ArtifactLeafV13(
                family="model_by_outer",
                fold_view=f"outer{outer}",
                fragment_ordinal=ordinal,
                query_start=0,
                query_stop=build.query_count,
                path=path,
                sha256=model.sha256,
                size_bytes=path.stat().st_size,
                rows=1,
                schema_sha256=_sha256_json(list(model.feature_names)),
            )
        )
    family_map = {
        "top500": "prerank500_compact",
        "top100": "top100_full",
        "outcomes": "outcomes",
    }
    for source_name, family in family_map.items():
        for fragment, spec in zip(
            replay.outputs[source_name], replay.specs, strict=True
        ):
            output[family].append(
                ArtifactLeafV13(
                    family=family,
                    fold_view="fold2_3_4",
                    fragment_ordinal=fragment.shard_ordinal,
                    query_start=spec.sequence_start,
                    query_stop=spec.sequence_stop,
                    path=fragment.path,
                    sha256=fragment.sha256,
                    size_bytes=fragment.size_bytes,
                    rows=fragment.row_count,
                    schema_sha256=fragment.schema_sha256,
                )
            )
    return {key: tuple(value) for key, value in output.items()}


def _write_family_manifest(
    family: str,
    leaves: Sequence[ArtifactLeafV13],
    logical_digest: str,
    output: Path,
    relocation_root: Path | None = None,
) -> DatasetManifestV13:
    records = [
        _leaf_record(value, output, relocation_root) for value in leaves
    ]
    identity_records = [_leaf_identity(value) for value in leaves]
    merkle = merkle_root_v13(identity_records)
    identity = {
        "engine_id": ENGINE_ID,
        "family": family,
        "merkle_root": merkle,
        "logical_digest": logical_digest,
        "rows": sum(value.rows for value in leaves),
        "fragment_count": len(leaves),
        "materialized_bytes": sum(value.size_bytes for value in leaves),
        "leaf_files": identity_records,
    }
    manifest = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "status": "SEALED_CANONICAL_DATASET_FAMILY",
        "dataset_id": _sha256_json(identity),
        "identity": identity,
        "fold0_opened": False,
        "fold1_opened": False,
        "copy_materialized": False,
        **identity,
        "leaf_files": records,
    }
    path = output / f"{family}.manifest.json"
    _atomic_json(path, manifest)
    return DatasetManifestV13(
        family=family,
        path=path,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        merkle_root=merkle,
        rows=sum(value.rows for value in leaves),
        fragment_count=len(leaves),
        materialized_bytes=sum(value.size_bytes for value in leaves),
        logical_digest=logical_digest,
        leaf_files=tuple(leaves),
    )


def seal_zero_copy_canonical_run_v13(
    build: BuildFragmentDatasetV13,
    replay: CompactReplayDatasetV13,
    models: Mapping[int, FittedSiteModel],
    output_dir: Path,
    *,
    relocation_root: Path | None = None,
) -> ZeroCopyCanonicalRunV13:
    """Seal the exact seven canonical families; never create merged data files."""

    if (
        set(models) != set(TRAIN_FOLDS)
        or dict(build.lineage) != dict(replay.lineage)
        or build.specs != replay.specs
        or build.query_count != replay.query_count
        or build.query_sequence_sha256 != replay.query_sequence_sha256
        or tuple(build.build_manifest_sha256) != tuple(replay.build_manifest_sha256)
        or build.heldout_logical_merkle_root
        != replay.heldout_dataset_merkle_root
        or any(models[outer].sha256 != replay.model_sha256[outer] for outer in TRAIN_FOLDS)
    ):
        raise FoldFusedInvariantError("v13 build/replay/model lineage differs")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    leaves = _family_leaves(build, replay, models)
    parity = build_logical_parity_digests_v13(build)
    logical = {
        **{
            f"train_outer{outer}": _sha256_json(
                {
                    "train_matrix": parity["train_matrix"][str(outer)],
                    "group_sequence": parity["group_sequence"][str(outer)],
                }
            )
            for outer in TRAIN_FOLDS
        },
        "heldout_spool": str(parity["heldout_spool"]),
        "prerank500_compact": replay.logical_merkle_root["top500"],
        "top100_full": replay.logical_merkle_root["top100"],
        "outcomes": replay.logical_merkle_root["outcomes"],
    }
    datasets = {
        family: _write_family_manifest(
            family,
            leaves[family],
            logical[family],
            output,
            relocation_root,
        )
        for family in CANONICAL_FAMILIES
    }

    # The inventory counts every physical leaf once.  Repeated paths and
    # aliases through hardlinks would invalidate byte accounting.
    inventory: list[ArtifactLeafV13] = []
    paths: set[Path] = set()
    inodes: set[tuple[int, int]] = set()
    for family in ARTIFACT_FAMILIES:
        for leaf in leaves[family]:
            path = leaf.path.resolve()
            stat = path.stat()
            inode = (int(stat.st_dev), int(stat.st_ino))
            if (
                path in paths
                or inode in inodes
                or int(stat.st_nlink) != 1
                or sha256_file(path) != leaf.sha256
            ):
                raise FoldFusedInvariantError(
                    "v13 artifact inventory duplicates/tampers a physical leaf"
                )
            paths.add(path)
            inodes.add(inode)
            inventory.append(leaf)
    dataset_records = {
        family: {
            "path": _record_path_v13(
                value.path, output, relocation_root
            ),
            "sha256": value.sha256,
            "size_bytes": value.size_bytes,
            "merkle_root": value.merkle_root,
            "rows": value.rows,
            "fragment_count": value.fragment_count,
            "materialized_bytes": value.materialized_bytes,
            "logical_digest": value.logical_digest,
        }
        for family, value in datasets.items()
    }
    dataset_identity_records = {
        family: {
            "merkle_root": value.merkle_root,
            "rows": value.rows,
            "fragment_count": value.fragment_count,
            "materialized_bytes": value.materialized_bytes,
            "logical_digest": value.logical_digest,
        }
        for family, value in datasets.items()
    }
    model_identity_records = [
        {
            "outer_fold": outer,
            "sha256": models[outer].sha256,
            "size_bytes": Path(models[outer].path).stat().st_size,
            "recipe_sha256": models[outer].recipe_sha256,
            "feature_names": list(models[outer].feature_names),
        }
        for outer in TRAIN_FOLDS
    ]
    identity = {
        "engine_id": ENGINE_ID,
        "lineage": dict(build.lineage),
        "build_dataset_id": build.dataset_id,
        "replay_dataset_id": replay.dataset_id,
        "query_count": build.query_count,
        "query_sequence_sha256": build.query_sequence_sha256,
        "canonical_datasets": dataset_identity_records,
        "models": model_identity_records,
        "artifact_inventory": [_leaf_identity(value) for value in inventory],
        "logical_parity_digests": {
            **parity,
            "model_by_outer": model_set_logical_sha256_v13(models),
            "prerank500": logical["prerank500_compact"],
            "top100": logical["top100_full"],
            "outcomes": logical["outcomes"],
            "site_feature_values": replay.feature_values_logical_merkle_root,
        },
    }
    run_id = _sha256_json(identity)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED_ZERO_COPY_CANONICAL_RUN",
        "engine_id": ENGINE_ID,
        "run_id": run_id,
        "identity": identity,
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "merged_files_created": False,
        "source_manifests": {
            "build": {
                "path": _record_path_v13(
                    build.manifest_path, output, relocation_root
                ),
                "sha256": build.manifest_sha256,
            },
            "replay": {
                "path": _record_path_v13(
                    replay.manifest_path, output, relocation_root
                ),
                "sha256": replay.manifest_sha256,
            },
        },
        **identity,
        "canonical_datasets": dataset_records,
        "models": [
            {
                **record,
                "path": _record_path_v13(
                    models[record["outer_fold"]].path,
                    output,
                    relocation_root,
                ),
            }
            for record in model_identity_records
        ],
        "artifact_inventory": [
            _leaf_record(value, output, relocation_root)
            for value in inventory
        ],
    }
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return ZeroCopyCanonicalRunV13(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        run_id=run_id,
        lineage=dict(build.lineage),
        canonical_datasets=datasets,
        artifact_inventory=tuple(inventory),
        query_count=build.query_count,
        query_sequence_sha256=build.query_sequence_sha256,
        logical_parity_digests=identity["logical_parity_digests"],
    )


def _load_leaf(
    value: Mapping[str, Any],
    root: Path,
    *,
    verify_hashes: bool,
) -> ArtifactLeafV13:
    leaf = ArtifactLeafV13(
        family=str(value.get("family") or ""),
        fold_view=str(value.get("fold_view") or ""),
        fragment_ordinal=int(value.get("fragment_ordinal", -1)),
        query_start=int(value.get("query_start", -1)),
        query_stop=int(value.get("query_stop", -1)),
        path=_resolve_path(root, value.get("path")),
        sha256=str(value.get("sha256") or ""),
        size_bytes=int(value.get("size_bytes", -1)),
        rows=int(value.get("rows", -1)),
        schema_sha256=str(value.get("schema_sha256") or ""),
    )
    if (
        not leaf.path.is_file()
        or leaf.path.stat().st_size != leaf.size_bytes
        or (verify_hashes and sha256_file(leaf.path) != leaf.sha256)
    ):
        raise FoldFusedInvariantError("v13 canonical leaf hash/size drift")
    return leaf


def load_zero_copy_canonical_run_v13(
    path: Path,
    *,
    verify_hashes: bool = True,
    verify_logical: bool = True,
) -> ZeroCopyCanonicalRunV13:
    """Strictly reopen all seven datasets and the deduplicated inventory."""

    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("status") != "SEALED_ZERO_COPY_CANONICAL_RUN"
        or raw.get("engine_id") != ENGINE_ID
        or raw.get("folds") != list(TRAIN_FOLDS)
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or raw.get("merged_files_created") is not False
    ):
        raise FoldFusedInvariantError("invalid or unsafe v13 canonical run")
    identity = raw.get("identity")
    if (
        not isinstance(identity, Mapping)
        or raw.get("run_id") != _sha256_json(identity)
    ):
        raise FoldFusedInvariantError("v13 canonical run content address drift")
    root = manifest_path.parent
    sources = raw.get("source_manifests")
    if not isinstance(sources, Mapping):
        raise FoldFusedInvariantError("v13 canonical source manifests missing")
    loaded_sources: dict[str, Any] = {}
    for name, loader in (
        ("build", load_build_fragment_dataset_v13),
        ("replay", load_compact_replay_dataset_v13),
    ):
        record = sources.get(name)
        if not isinstance(record, Mapping):
            raise FoldFusedInvariantError("v13 canonical source record missing")
        source_path = _resolve_path(root, record.get("path"))
        digest = str(record.get("sha256") or "")
        if (
            not source_path.is_file()
            or not _valid_sha256(digest)
            or (verify_hashes and sha256_file(source_path) != digest)
        ):
            raise FoldFusedInvariantError("v13 canonical source manifest hash drift")
        loaded = loader(
            source_path,
            verify_hashes=verify_hashes,
            verify_logical=verify_logical,
        )
        if loaded.manifest_sha256 != digest:
            raise FoldFusedInvariantError("v13 canonical source identity drift")
        loaded_sources[name] = loaded
    build: BuildFragmentDatasetV13 = loaded_sources["build"]
    replay: CompactReplayDatasetV13 = loaded_sources["replay"]
    if (
        build.specs != replay.specs
        or build.heldout_logical_merkle_root
        != replay.heldout_dataset_merkle_root
    ):
        raise FoldFusedInvariantError("v13 canonical build/replay lineage drift")

    dataset_records = raw.get("canonical_datasets")
    if not isinstance(dataset_records, Mapping) or set(dataset_records) != set(CANONICAL_FAMILIES):
        raise FoldFusedInvariantError("v13 canonical datasets are not exactly seven")
    datasets: dict[str, DatasetManifestV13] = {}
    dataset_identity: dict[str, Mapping[str, Any]] = {}
    for family in CANONICAL_FAMILIES:
        record = dataset_records[family]
        if not isinstance(record, Mapping):
            raise FoldFusedInvariantError("v13 DatasetManifest record invalid")
        family_path = _resolve_path(root, record.get("path"))
        digest = str(record.get("sha256") or "")
        if (
            not family_path.is_file()
            or family_path.stat().st_size != int(record.get("size_bytes", -1))
            or not _valid_sha256(digest)
            or (verify_hashes and sha256_file(family_path) != digest)
        ):
            raise FoldFusedInvariantError("v13 DatasetManifest hash/size drift")
        family_raw = json.loads(family_path.read_text(encoding="utf-8"))
        family_identity = family_raw.get("identity")
        if (
            family_raw.get("schema_version") != FAMILY_SCHEMA_VERSION
            or family_raw.get("status") != "SEALED_CANONICAL_DATASET_FAMILY"
            or family_raw.get("fold0_opened") is not False
            or family_raw.get("fold1_opened") is not False
            or family_raw.get("copy_materialized") is not False
            or not isinstance(family_identity, Mapping)
            or family_raw.get("dataset_id") != _sha256_json(family_identity)
            or family_identity.get("family") != family
        ):
            raise FoldFusedInvariantError("invalid v13 DatasetManifest content")
        operational_leaves = family_raw.get("leaf_files")
        identity_leaves = family_identity.get("leaf_files")
        if (
            not isinstance(operational_leaves, list)
            or not isinstance(identity_leaves, list)
            or len(operational_leaves) != len(identity_leaves)
        ):
            raise FoldFusedInvariantError("v13 DatasetManifest leaf inventory drift")
        leaves = tuple(
            _load_leaf(value, root, verify_hashes=verify_hashes)
            for value in operational_leaves
        )
        if (
            tuple(value.fragment_ordinal for value in leaves)
            != tuple(range(len(leaves)))
            or [_leaf_identity(value) for value in leaves] != identity_leaves
        ):
            raise FoldFusedInvariantError("v13 DatasetManifest leaf identity drift")
        merkle = merkle_root_v13(identity_leaves)
        logical_digest = str(family_identity.get("logical_digest") or "")
        dataset = DatasetManifestV13(
            family=family,
            path=family_path,
            sha256=digest,
            size_bytes=family_path.stat().st_size,
            merkle_root=merkle,
            rows=int(family_identity.get("rows", -1)),
            fragment_count=int(family_identity.get("fragment_count", -1)),
            materialized_bytes=int(family_identity.get("materialized_bytes", -1)),
            logical_digest=logical_digest,
            leaf_files=leaves,
        )
        if (
            family_identity.get("merkle_root") != merkle
            or record.get("merkle_root") != merkle
            or record.get("logical_digest") != logical_digest
        ):
            raise FoldFusedInvariantError("v13 DatasetManifest Merkle/logical drift")
        datasets[family] = dataset
        dataset_identity[family] = {
            "merkle_root": dataset.merkle_root,
            "rows": dataset.rows,
            "fragment_count": dataset.fragment_count,
            "materialized_bytes": dataset.materialized_bytes,
            "logical_digest": dataset.logical_digest,
        }

    inventory_raw = raw.get("artifact_inventory")
    if not isinstance(inventory_raw, list):
        raise FoldFusedInvariantError("v13 global artifact inventory missing")
    inventory = tuple(
        _load_leaf(value, root, verify_hashes=verify_hashes)
        for value in inventory_raw
    )
    paths: set[Path] = set()
    inodes: set[tuple[int, int]] = set()
    for leaf in inventory:
        stat = leaf.path.stat()
        inode = (int(stat.st_dev), int(stat.st_ino))
        if leaf.path in paths or inode in inodes or int(stat.st_nlink) != 1:
            raise FoldFusedInvariantError("v13 global inventory duplicates/hardlinks a leaf")
        paths.add(leaf.path)
        inodes.add(inode)
    expected_family_leaves = [
        _leaf_identity(leaf)
        for family in CANONICAL_FAMILIES
        for leaf in datasets[family].leaf_files
    ]
    inventory_identity = [_leaf_identity(value) for value in inventory]

    model_records = raw.get("models")
    if not isinstance(model_records, list) or len(model_records) != len(TRAIN_FOLDS):
        raise FoldFusedInvariantError("v13 canonical model records missing")
    model_identity: list[Mapping[str, Any]] = []
    model_leaves: list[ArtifactLeafV13] = []
    for record, outer in zip(model_records, TRAIN_FOLDS, strict=True):
        model_path = _resolve_path(root, record.get("path"))
        value = {
            "outer_fold": int(record.get("outer_fold", -1)),
            "sha256": str(record.get("sha256") or ""),
            "size_bytes": int(record.get("size_bytes", -1)),
            "recipe_sha256": str(record.get("recipe_sha256") or ""),
            "feature_names": [str(item) for item in record.get("feature_names") or ()],
        }
        if (
            value["outer_fold"] != outer
            or not model_path.is_file()
            or model_path.stat().st_size != value["size_bytes"]
            or not _valid_sha256(value["sha256"])
            or not _valid_sha256(value["recipe_sha256"])
            or (verify_hashes and sha256_file(model_path) != value["sha256"])
        ):
            raise FoldFusedInvariantError("v13 canonical model record drift")
        model_identity.append(value)
        model_leaves.append(
            ArtifactLeafV13(
                family="model_by_outer",
                fold_view=f"outer{outer}",
                fragment_ordinal=len(model_leaves),
                query_start=0,
                query_stop=build.query_count,
                path=model_path,
                sha256=value["sha256"],
                size_bytes=value["size_bytes"],
                rows=1,
                schema_sha256=_sha256_json(value["feature_names"]),
            )
        )
    expected_inventory = [
        *expected_family_leaves,
        *[_leaf_identity(value) for value in model_leaves],
    ]
    expected_inventory_paths = [
        *[
            leaf.path.resolve()
            for family in CANONICAL_FAMILIES
            for leaf in datasets[family].leaf_files
        ],
        *[value.path.resolve() for value in model_leaves],
    ]
    if (
        inventory_identity != expected_inventory
        or [value.path.resolve() for value in inventory]
        != expected_inventory_paths
    ):
        raise FoldFusedInvariantError(
            "v13 global inventory is not exactly seven families plus three models"
        )
    parity = {
        **build_logical_parity_digests_v13(build),
        "model_by_outer": _sha256_json(
            [
                {
                    "outer_fold": value["outer_fold"],
                    "sha256": value["sha256"],
                    "recipe_sha256": value["recipe_sha256"],
                    "feature_names": value["feature_names"],
                }
                for value in model_identity
            ]
        ),
        "prerank500": replay.logical_merkle_root["top500"],
        "top100": replay.logical_merkle_root["top100"],
        "outcomes": replay.logical_merkle_root["outcomes"],
        "site_feature_values": replay.feature_values_logical_merkle_root,
    }
    expected_identity = {
        "engine_id": ENGINE_ID,
        "lineage": dict(build.lineage),
        "build_dataset_id": build.dataset_id,
        "replay_dataset_id": replay.dataset_id,
        "query_count": build.query_count,
        "query_sequence_sha256": build.query_sequence_sha256,
        "canonical_datasets": dataset_identity,
        "models": model_identity,
        "artifact_inventory": inventory_identity,
        "logical_parity_digests": parity,
    }
    if dict(identity) != expected_identity:
        raise FoldFusedInvariantError("v13 canonical run identity/parity drift")
    return ZeroCopyCanonicalRunV13(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        run_id=str(raw["run_id"]),
        lineage=dict(build.lineage),
        canonical_datasets=datasets,
        artifact_inventory=inventory,
        query_count=build.query_count,
        query_sequence_sha256=build.query_sequence_sha256,
        logical_parity_digests=parity,
    )


__all__ = [
    "ArtifactLeafV13",
    "CANONICAL_FAMILIES",
    "DatasetManifestV13",
    "ENGINE_ID",
    "FAMILY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ZeroCopyCanonicalRunV13",
    "load_zero_copy_canonical_run_v13",
    "seal_zero_copy_canonical_run_v13",
]
