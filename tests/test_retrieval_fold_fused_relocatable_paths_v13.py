from __future__ import annotations

from pathlib import Path

import pytest

from xgb_matcher.retrieval_fold_fused_relocatable_paths_v13 import (
    RelocatablePathInvariantErrorV13,
    relocatable_record_path_v13,
    resolve_relocatable_record_path_v13,
)


def test_relocatable_path_survives_atomic_run_root_rename(tmp_path: Path) -> None:
    stage = tmp_path / ".national-v13-stage"
    record_root = stage / "compact_replay" / "fold2"
    target = stage / "build" / "fold2" / "heldout.parquet"
    record_root.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"heldout")

    encoded = relocatable_record_path_v13(
        target, record_root=record_root, run_root=stage
    )
    assert not Path(encoded).is_absolute()
    assert (
        resolve_relocatable_record_path_v13(
            encoded, record_root=record_root, run_root=stage
        )
        == target.resolve()
    )

    canonical = tmp_path / "national-v13"
    stage.rename(canonical)
    promoted_record_root = canonical / "compact_replay" / "fold2"
    assert (
        resolve_relocatable_record_path_v13(
            encoded,
            record_root=promoted_record_root,
            run_root=canonical,
        )
        == (canonical / "build" / "fold2" / "heldout.parquet").resolve()
    )


def test_relocatable_path_rejects_external_or_absolute_targets(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    record_root = stage / "records"
    record_root.mkdir(parents=True)
    external = tmp_path / "external.bin"
    external.write_bytes(b"external")

    with pytest.raises(
        RelocatablePathInvariantErrorV13, match="escaped"
    ):
        relocatable_record_path_v13(
            external, record_root=record_root, run_root=stage
        )
    with pytest.raises(
        RelocatablePathInvariantErrorV13, match="relative"
    ):
        resolve_relocatable_record_path_v13(
            str(external), record_root=record_root, run_root=stage
        )
    with pytest.raises(
        RelocatablePathInvariantErrorV13, match="escaped"
    ):
        resolve_relocatable_record_path_v13(
            "../../external.bin", record_root=record_root, run_root=stage
        )
