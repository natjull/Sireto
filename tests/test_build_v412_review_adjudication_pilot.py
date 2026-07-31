from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_v412_review_adjudication_pilot as subject


def _frames() -> dict[str, pd.DataFrame]:
    query_rows = []
    guard_rows = []
    evidence_rows = []
    ranker_rows = []
    for stratum_index, stratum in enumerate(subject.STRATUM_ORDER):
        for row_index in range(3):
            query_id = f"q-{stratum_index}-{row_index}"
            query_rows.append(
                {
                    "query_id": query_id,
                    "crm_record_id": f"crm-{query_id}",
                    "crm_name": f"ENTITE {query_id}",
                    "crm_address": f"{row_index + 1} RUE EXEMPLE",
                    "crm_postcode": "75001",
                    "crm_city": "PARIS",
                    "crm_insee": "75101",
                }
            )
            guard_rows.append(
                {
                    "query_id": query_id,
                    "predicted_siret": f"12345678{stratum_index}{row_index:05d}",
                    "predicted_siren": f"12345678{stratum_index}",
                    "decision_v412": "REVIEW",
                    "review_reason_v412": "LOW_CONFIDENCE",
                }
            )
            evidence_rows.append(
                {
                    "query_id": query_id,
                    "direct_candidate_count": 2 if stratum_index < 2 else 1,
                    "direct_siren_count": 1 if stratum_index == 0 else 2,
                    "cross_siren_direct_collision": stratum_index == 1,
                    "same_siren_direct_multisite": stratum_index == 0,
                }
            )
            ranker_rows.extend(
                [
                    {
                        "query_id": query_id,
                        "candidate_siret": f"12345678{stratum_index}{row_index:05d}",
                        "candidate_siren": f"12345678{stratum_index}",
                        "retrieval_rank": 2,
                        "ranker_rank": 1,
                    },
                    {
                        "query_id": query_id,
                        "candidate_siret": f"98765432{stratum_index}{row_index:05d}",
                        "candidate_siren": f"98765432{stratum_index}",
                        "retrieval_rank": 1,
                        "ranker_rank": 2,
                    },
                ]
            )
    return {
        "queries": pd.DataFrame(query_rows, columns=subject.QUERY_COLUMNS),
        "guard": pd.DataFrame(guard_rows, columns=subject.GUARD_COLUMNS),
        "query_evidence": pd.DataFrame(
            evidence_rows, columns=subject.EVIDENCE_COLUMNS
        ),
        "partitions": pd.DataFrame(
            [
                {
                    "query_id": "safe-support",
                    "component_id": "component-safe",
                    "partition": "historical_fit",
                    "role": "historical_fit",
                }
            ],
            columns=subject.PARTITION_COLUMNS,
        ),
        "adjudicated_query_ids": pd.DataFrame(
            {"query_id": ["old-case"]}, columns=["query_id"]
        ),
        "ranker": pd.DataFrame(ranker_rows, columns=subject.RANKER_COLUMNS),
    }


def _build(frames: dict[str, pd.DataFrame], per_stratum: int = 2) -> dict:
    return subject.build_frames(
        **frames,
        per_stratum=per_stratum,
        enforce_canonical_counts=False,
        expected_selection_sha256=None,
    )


def _artifact_inputs(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, str], Path]:
    frames = _frames()
    files: dict[str, Path] = {}
    source_by_name = {
        "queries": frames["queries"],
        "guard": frames["guard"],
        "query_evidence": frames["query_evidence"],
        "partition_assignments": frames["partitions"],
        "adjudicated_query_ids": frames["adjudicated_query_ids"],
        "ranker": frames["ranker"],
    }
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    for key in subject.CANONICAL_INPUTS:
        path = input_root / f"{key}.parquet"
        if key in source_by_name:
            source_by_name[key].to_parquet(path, index=False)
        else:
            path.write_bytes(f"{key}\n".encode())
        files[key] = path
    contract = input_root / "contract.md"
    contract.write_text("synthetic contract\n", encoding="utf-8")
    hashes = {key: subject._file_sha256(path) for key, path in files.items()}
    return files, hashes, contract


def test_build_frames_selects_fixed_balanced_docket_without_identity_leak() -> None:
    result = _build(_frames())
    docket = result["docket"]
    assert docket["stratum"].tolist() == [
        "SAME_SIREN_MULTISITE",
        "SAME_SIREN_MULTISITE",
        "CROSS_SIREN_COLLISION",
        "CROSS_SIREN_COLLISION",
        "OTHER_REVIEW",
        "OTHER_REVIEW",
    ]
    assert len(docket) == 6
    assert result["candidate_context"].groupby("query_id").size().max() == 2
    assert len(result["collection_plan"]) == 18

    identity = result["identity_discovery"]
    forbidden = (
        "siret",
        "siren",
        "candidate",
        "rank",
        "score",
        "prediction",
        "top1",
        "label",
        "target",
    )
    assert not any(
        token in column.lower() for token in forbidden for column in identity.columns
    )


def test_collection_plan_is_exact_and_bounded() -> None:
    result = _build(_frames())
    row = result["identity_discovery"].iloc[0]
    plan = result["collection_plan"]
    observed = plan[plan["query_id"].eq(row["query_id"])].sort_values(
        "query_ordinal"
    )
    assert observed["search_query"].tolist() == [
        f'"{row["crm_name"]}" "{row["crm_postcode"]}"',
        f'"{row["crm_name"]}" "{row["crm_city"]}" "{row["crm_address"]}"',
        f'"{row["crm_name"]}" "{row["crm_city"]}" (SIRET OR établissement)',
    ]
    assert observed["max_results_logged"].eq(5).all()
    assert observed["max_admissible_pages_opened"].eq(2).all()
    assert observed["max_admissible_pages_total_for_dossier"].eq(6).all()


def test_selection_hash_uses_declared_stratum_order() -> None:
    result = _build(_frames())
    ids = result["docket"]["query_id"].astype(str).tolist()
    expected = hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert result["selection_sha256"] == expected


def test_component_closure_excludes_every_linked_query() -> None:
    frames = _frames()
    victim = frames["queries"].iloc[0]["query_id"]
    linked = frames["queries"].iloc[1]["query_id"]
    frames["partitions"] = pd.DataFrame(
        [
            {
                "query_id": victim,
                "component_id": "blocked-component",
                "partition": "random_sealed",
                "role": "random_sealed",
            },
            {
                "query_id": linked,
                "component_id": "blocked-component",
                "partition": "historical_fit",
                "role": "historical_fit",
            },
        ],
        columns=subject.PARTITION_COLUMNS,
    )
    result = _build(frames, per_stratum=1)
    assert victim not in set(result["docket"]["query_id"])
    assert linked not in set(result["docket"]["query_id"])


def test_old_adjudication_is_excluded_by_query_id_only() -> None:
    frames = _frames()
    victim = frames["queries"].iloc[0]["query_id"]
    frames["adjudicated_query_ids"] = pd.DataFrame({"query_id": [victim]})
    result = _build(frames, per_stratum=1)
    assert victim not in set(result["docket"]["query_id"])


def test_overlapping_direct_strata_are_rejected() -> None:
    frames = _frames()
    frames["query_evidence"].loc[0, "cross_siren_direct_collision"] = True
    with pytest.raises(ValueError, match="strata overlap"):
        _build(frames)


def test_string_boolean_cannot_change_a_stratum() -> None:
    frames = _frames()
    frames["query_evidence"]["cross_siren_direct_collision"] = (
        frames["query_evidence"]["cross_siren_direct_collision"].astype(object)
    )
    frames["query_evidence"].loc[0, "cross_siren_direct_collision"] = "False"
    with pytest.raises(ValueError, match="strict booleans"):
        _build(frames)


def test_candidate_cap_is_absolute() -> None:
    frames = _frames()
    query_id = frames["queries"].iloc[0]["query_id"]
    extra = []
    for index in range(99):
        extra.append(
            {
                "query_id": query_id,
                "candidate_siret": f"555555555{index:05d}",
                "candidate_siren": "555555555",
                "retrieval_rank": index + 3,
                "ranker_rank": index + 3,
            }
        )
    frames["ranker"] = pd.concat(
        [frames["ranker"], pd.DataFrame(extra)], ignore_index=True
    )
    with pytest.raises(ValueError, match="absolute cap"):
        _build(frames, per_stratum=3)


def test_guard_top1_must_equal_ranker_top1() -> None:
    frames = _frames()
    frames["guard"].loc[0, "predicted_siret"] = "00000000000000"
    with pytest.raises(ValueError, match="differs"):
        _build(frames, per_stratum=3)


def test_fractional_rank_is_rejected_without_truncation() -> None:
    frames = _frames()
    frames["ranker"]["ranker_rank"] = frames["ranker"]["ranker_rank"].astype(float)
    frames["ranker"].loc[0, "ranker_rank"] = 1.5
    with pytest.raises(ValueError, match="positive integer ranker_rank"):
        _build(frames, per_stratum=3)


def test_build_artifact_reads_only_closed_projections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, hashes, contract = _artifact_inputs(tmp_path)

    calls: list[tuple[str, ...]] = []
    original = subject._read_parquet

    def recording_read(descriptor: int, columns) -> pd.DataFrame:
        calls.append(tuple(columns))
        return original(descriptor, columns)

    monkeypatch.setattr(subject, "_read_parquet", recording_read)
    monkeypatch.setattr(subject, "EXPECTED_SELECTION_SHA256", None)
    target = subject.build_artifact(
        input_paths=files,
        input_hashes=hashes,
        contract_path=contract,
        output_root=tmp_path / "out",
        enforce_canonical=False,
        per_stratum=2,
    )
    assert ("query_id",) in calls
    assert {
        str(path.relative_to(target))
        for path in target.rglob("*")
        if path.is_file()
    } == {
        "identity/identity_discovery.parquet",
        "identity/collection_plan.parquet",
        "comparison/docket.parquet",
        "comparison/candidate_context.parquet",
        "summary.json",
        "manifest.json",
        "seal.json",
    }
    summary = json.loads((target / "summary.json").read_text())
    assert summary["label_payload_hashed"] is True
    assert summary["label_columns_deserialized"] == []
    assert summary["label_semantics_opened"] is False
    assert summary["public_adjudication_evidence_opened"] is False
    assert summary["network_access_performed"] is False
    assert summary["forbidden_population_opened"] is False
    assert summary["scope"] == "DOCKET_BUILDER_ONLY_NO_COLLECTION"
    assert (target.stat().st_mode & 0o777) == 0o555
    assert all(
        (path.stat().st_mode & 0o777) == 0o555
        for path in target.rglob("*")
        if path.is_dir()
    )
    assert all(
        (path.stat().st_mode & 0o777) == 0o444
        for path in target.rglob("*")
        if path.is_file()
    )
    with pytest.raises(PermissionError):
        subject.os.open(target / "summary.json", subject.os.O_WRONLY)


def test_concurrent_publication_has_exactly_one_winner(tmp_path: Path) -> None:
    files, hashes, contract = _artifact_inputs(tmp_path)
    output_root = tmp_path / "out"

    def launch() -> str:
        try:
            target = subject.build_artifact(
                input_paths=files,
                input_hashes=hashes,
                contract_path=contract,
                output_root=output_root,
                enforce_canonical=False,
                per_stratum=2,
            )
            return f"OK:{target.name}"
        except FileExistsError:
            return "EXISTS"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: launch(), range(2)))
    assert sum(result.startswith("OK:") for result in results) == 1
    assert results.count("EXISTS") == 1
    published = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert len(published) == 1
    assert (published[0] / "manifest.json").is_file()


def test_canonical_mode_rejects_an_arbitrary_output_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="preregistered output root"):
        subject._prepare_output_root(
            tmp_path / "safe-looking", enforce_canonical=True
        )


def test_input_symlink_and_hardlink_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    symlink = tmp_path / "symlink.bin"
    symlink.symlink_to(source)
    directory_fd = subject.os.open(
        tmp_path, subject.os.O_RDONLY | subject.os.O_DIRECTORY
    )
    try:
        with pytest.raises(ValueError, match="symlink"):
            subject._open_copy_input(symlink, directory_fd, "copy.bin")

        hardlink = tmp_path / "hardlink.bin"
        hardlink.hardlink_to(source)
        with pytest.raises(ValueError, match="multiply-linked"):
            subject._open_copy_input(source, directory_fd, "copy.bin")
    finally:
        subject.os.close(directory_fd)


def test_open_input_revalidation_detects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"before")
    directory_fd = subject.os.open(
        tmp_path, subject.os.O_RDONLY | subject.os.O_DIRECTORY
    )
    descriptor, private_fd, snapshot = subject._open_copy_input(
        source, directory_fd, "copy.bin"
    )
    try:
        source.write_bytes(b"after!")
        with pytest.raises(ValueError, match="changed|substituted"):
            subject._revalidate_open_input(source, descriptor, snapshot)
    finally:
        subject.os.close(private_fd)
        subject.os.close(descriptor)
        subject.os.close(directory_fd)
        for directory_fd in reversed(snapshot["_directory_fds"]):
            subject.os.close(directory_fd)


def test_open_input_revalidation_detects_parent_symlink_swap(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "inputs"
    parent.mkdir()
    source = parent / "source.bin"
    source.write_bytes(b"stable")
    destination_fd = subject.os.open(
        tmp_path, subject.os.O_RDONLY | subject.os.O_DIRECTORY
    )
    descriptor, private_fd, snapshot = subject._open_copy_input(
        source, destination_fd, "copy.bin"
    )
    moved = tmp_path / "inputs-real"
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="symlink|ancestor"):
            subject._revalidate_open_input(source, descriptor, snapshot)
    finally:
        subject.os.close(private_fd)
        subject.os.close(descriptor)
        subject.os.close(destination_fd)
        for directory_fd in reversed(snapshot["_directory_fds"]):
            subject.os.close(directory_fd)


def test_output_root_swap_during_private_read_stops_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, hashes, contract = _artifact_inputs(tmp_path)
    output_root = tmp_path / "out"
    original = subject._read_parquet
    swapped = False

    def swapping_read(descriptor: int, columns) -> pd.DataFrame:
        nonlocal swapped
        if not swapped:
            swapped = True
            moved = tmp_path / "out-real"
            output_root.rename(moved)
            output_root.symlink_to(moved, target_is_directory=True)
            output_root.unlink()
            moved.rename(output_root)
        return original(descriptor, columns)

    monkeypatch.setattr(subject, "_read_parquet", swapping_read)
    with pytest.raises(ValueError, match="output root parent changed"):
        subject.build_artifact(
            input_paths=files,
            input_hashes=hashes,
            contract_path=contract,
            output_root=output_root,
            enforce_canonical=False,
            per_stratum=2,
        )
    assert swapped is True
    assert not [
        path
        for path in output_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]


def test_post_promotion_verification_failure_removes_only_promoted_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, hashes, contract = _artifact_inputs(tmp_path)
    output_root = tmp_path / "out"
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    (sentinel / "keep.txt").write_text("keep", encoding="utf-8")

    def fail_after_promotion(*_args, **_kwargs) -> None:
        raise ValueError("injected post-promotion verification failure")

    monkeypatch.setattr(subject, "_verify_sealed_tree_fd", fail_after_promotion)
    with pytest.raises(ValueError, match="injected post-promotion"):
        subject.build_artifact(
            input_paths=files,
            input_hashes=hashes,
            contract_path=contract,
            output_root=output_root,
            enforce_canonical=False,
            per_stratum=2,
        )

    assert (sentinel / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not [
        path
        for path in output_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    quarantines = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith(".failed-")
    ]
    assert len(quarantines) == 1
    assert (quarantines[0].stat().st_mode & 0o777) == 0o555


def test_promotion_name_swap_before_quarantine_never_deletes_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, hashes, contract = _artifact_inputs(tmp_path)
    output_root = tmp_path / "out"
    original_rename_noreplace = subject._rename_noreplace_at
    swapped = False

    def swap_then_quarantine(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            subject.os.rename(
                source_name,
                ".saved-promoted",
                src_dir_fd=source_dir_fd,
                dst_dir_fd=source_dir_fd,
            )
            subject.os.mkdir(source_name, mode=0o700, dir_fd=source_dir_fd)
            replacement_fd = subject._open_directory_at(
                source_dir_fd, source_name
            )
            try:
                subject._write_bytes_at(
                    replacement_fd, "keep.txt", b"replacement survives\n"
                )
            finally:
                subject.os.close(replacement_fd)
        original_rename_noreplace(
            source_dir_fd,
            source_name,
            destination_dir_fd,
            destination_name,
        )

    def fail_after_promotion(*_args, **_kwargs) -> None:
        raise ValueError("injected post-promotion verification failure")

    monkeypatch.setattr(subject, "_verify_sealed_tree_fd", fail_after_promotion)
    monkeypatch.setattr(
        subject, "_rename_noreplace_at", swap_then_quarantine
    )
    with pytest.raises(ValueError, match="quarantined target differs"):
        subject.build_artifact(
            input_paths=files,
            input_hashes=hashes,
            contract_path=contract,
            output_root=output_root,
            enforce_canonical=False,
            per_stratum=2,
        )

    assert swapped is True
    assert (output_root / ".saved-promoted" / "seal.json").is_file()
    quarantines = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith(".failed-")
    ]
    assert len(quarantines) == 1
    assert (
        quarantines[0] / "keep.txt"
    ).read_text(encoding="utf-8") == "replacement survives\n"


def test_quarantine_rename_never_replaces_an_existing_destination(
    tmp_path: Path,
) -> None:
    (tmp_path / "source").mkdir()
    (tmp_path / "destination").mkdir()
    (tmp_path / "destination" / "keep.txt").write_text(
        "keep", encoding="utf-8"
    )
    directory_fd = subject.os.open(
        tmp_path, subject.os.O_RDONLY | subject.os.O_DIRECTORY
    )
    try:
        with pytest.raises(FileExistsError):
            subject._rename_noreplace_at(
                directory_fd,
                "source",
                directory_fd,
                "destination",
            )
    finally:
        subject.os.close(directory_fd)
    assert (tmp_path / "source").is_dir()
    assert (tmp_path / "destination" / "keep.txt").read_text() == "keep"


def test_hash_mismatch_stops_before_parquet_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {}
    hashes = {}
    for key in subject.CANONICAL_INPUTS:
        path = tmp_path / key
        path.write_text(key, encoding="utf-8")
        paths[key] = path
        hashes[key] = subject._file_sha256(path)
    hashes["guard"] = "0" * 64
    contract = tmp_path / "contract.md"
    contract.write_text("contract", encoding="utf-8")

    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("parquet was read before the hash gate")

    monkeypatch.setattr(subject, "_read_parquet", forbidden_read)
    with pytest.raises(ValueError, match="guard hash mismatch"):
        subject.build_artifact(
            input_paths=paths,
            input_hashes=hashes,
            contract_path=contract,
            output_root=tmp_path / "out",
            enforce_canonical=False,
        )
