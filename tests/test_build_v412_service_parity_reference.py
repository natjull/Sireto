from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_v412_service_parity_reference as subject


def _write_source(path: Path, frame: pd.DataFrame) -> subject.SourceFile:
    frame.to_parquet(path, index=False)
    return subject.SourceFile(path=path, sha256=subject.file_sha256(path))


def _reference_sources(tmp_path: Path) -> subject.ReferenceSources:
    query_ids = ["q0", "q1", "q2", "fit0"]
    queries = pd.DataFrame(
        [
            {
                column: (
                    query_id
                    if column == "query_id"
                    else f"crm-{query_id}"
                    if column == "crm_record_id"
                    else f"{column}-{query_id}"
                )
                for column in subject.QUERY_COLUMNS
            }
            for query_id in query_ids
        ]
    )
    splits = pd.DataFrame(
        {
            "query_id": query_ids,
            "split": ["dev", "dev", "dev", "fit"],
            "siren_component_id": ["forbidden"] * 4,
        }
    )

    candidate_rows: list[dict[str, object]] = []
    ranker_rows: list[dict[str, object]] = []
    for query_index, query_id in enumerate(query_ids):
        for rank in (1, 2):
            siren = f"{query_index + 1:09d}"
            siret = f"{siren}{rank:05d}"
            candidate = {
                column: 0.0 for column in subject.CANDIDATE_COLUMNS
            }
            candidate.update(
                {
                    "query_id": query_id,
                    "candidate_siret": siret,
                    "candidate_siren": siren,
                    "candidate_state": "A",
                    "retrieval_rank": rank,
                    "retrieval_source": "sparse",
                    "retrieval_channel_count": 1,
                    "retrieval_agreement": 1,
                    "enseigne1": "name",
                    "enseigne2": None,
                    "enseigne3": None,
                    "denomination_usuelle": "name",
                    "activity_code": "00.00Z",
                }
            )
            candidate["is_ground_truth"] = rank == 1
            candidate_rows.append(candidate)
            ranker_rows.append(
                {
                    "query_id": query_id,
                    "candidate_siret": siret,
                    "candidate_siren": siren,
                    "retrieval_rank": rank,
                    "is_ground_truth": rank == 1,
                    "ranker_score": 1.0 / rank,
                    "prediction_origin": "OOF",
                    "oof_fold": 0,
                    "ranker_rank": rank,
                }
            )
    candidates = pd.DataFrame(candidate_rows)[
        [*subject.CANDIDATE_COLUMNS[:4], "is_ground_truth", *subject.CANDIDATE_COLUMNS[4:]]
    ]
    ranker = pd.DataFrame(ranker_rows)[
        [*subject.RANKER_COLUMNS[:4], "is_ground_truth", *subject.RANKER_COLUMNS[4:]]
    ]

    scene_rows = []
    for query_index, query_id in enumerate(query_ids):
        siren = f"{query_index + 1:09d}"
        row: dict[str, object] = {
            "query_id": query_id,
            "crm_record_id": f"crm-{query_id}",
            "predicted_siret": f"{siren}00001",
            "predicted_siren": siren,
            "ground_truth_siret": f"{siren}00001",
            "acceptor_target": 1,
        }
        row.update({feature: 0.0 for feature in subject.SCENE_FEATURES})
        row["candidate_count"] = 2.0
        row["dev_partition"] = "threshold_dev"
        scene_rows.append(row)
    scenes = pd.DataFrame(scene_rows)

    acceptor_rows = []
    for family in (subject.COMPACT_LOGIT, "MONOTONIC_XGB"):
        for query_index, query_id in enumerate(query_ids):
            siren = f"{query_index + 1:09d}"
            acceptor_rows.append(
                {
                    "model_family": family,
                    "evaluation_partition": "threshold_dev",
                    "query_id": query_id,
                    "label_kind": "MATCH_EXACT",
                    "ground_truth_siret": f"{siren}00001",
                    "predicted_siret": f"{siren}00001",
                    "acceptor_target": 1,
                    "score": 0.95 if query_id != "q1" else 0.2,
                    "threshold": subject.FIXED_THRESHOLD,
                    "decision": "AUTO_MATCH" if query_id != "q1" else "REVIEW",
                }
            )
    acceptor = pd.DataFrame(acceptor_rows)[
        [
            "model_family",
            "evaluation_partition",
            "query_id",
            "label_kind",
            "ground_truth_siret",
            "predicted_siret",
            "acceptor_target",
            "score",
            "threshold",
            "decision",
        ]
    ]

    query_evidence_rows = []
    candidate_evidence_rows = []
    for query_index, query_id in enumerate(query_ids):
        siren = f"{query_index + 1:09d}"
        direct_sirets = (
            [f"{siren}00001"]
            if query_id in {"q0", "fit0"}
            else []
            if query_id == "q1"
            else [f"{siren}00001", f"{siren}00002"]
        )
        refs = [f"DIRECT:{query_id}:{siret}" for siret in direct_sirets]
        query_evidence_rows.append(
            {
                "query_id": query_id,
                "partition_key": f"partition-{query_id}",
                "active_universe_count": 2,
                "direct_candidate_count": len(direct_sirets),
                "direct_siren_count": 1 if direct_sirets else 0,
                "sole_direct_siret": direct_sirets[0]
                if len(direct_sirets) == 1
                else None,
                "sole_direct_siren": siren if len(direct_sirets) == 1 else None,
                "cross_siren_direct_collision": False,
                "same_siren_direct_multisite": len(direct_sirets) == 2,
                "evidence_refs_json": json.dumps(refs, separators=(",", ":")),
            }
        )
        for siret, ref in zip(direct_sirets, refs, strict=True):
            candidate_evidence_rows.append(
                {
                    "evidence_ref": ref,
                    "query_id": query_id,
                    "candidate_siret": siret,
                    "candidate_siren": siren,
                    "candidate_state": "A",
                    "exact_name_anchor": True,
                    "exact_address_anchor": True,
                    "strong_name_evidence": True,
                    "strong_address_evidence": True,
                    "direct_evidence_class": "NAME_AND_ADDRESS",
                    "direct_match_rule": "EXACT_NAME_AND_ADDRESS",
                }
            )
    query_evidence = pd.DataFrame(query_evidence_rows)[
        subject.QUERY_EVIDENCE_COLUMNS
    ]
    candidate_evidence = pd.DataFrame(candidate_evidence_rows)[
        subject.CANDIDATE_EVIDENCE_COLUMNS
    ]
    rebuilt_guard = subject._build_guard(
        acceptor.loc[
            acceptor["model_family"].eq(subject.COMPACT_LOGIT),
            subject.ACCEPTOR_COLUMNS,
        ],
        scenes[subject.SCENE_COLUMNS],
        query_evidence,
    )
    guard = rebuilt_guard[
        [
            "query_id",
            "predicted_siret",
            "acceptor_score",
            "decision_v411",
            "direct_candidate_count",
            "direct_siren_count",
            "sole_direct_siret",
            "sole_direct_siren",
            "decision_v412",
            "review_reason_v412",
        ]
    ].copy()
    guard.insert(
        guard.columns.get_loc("direct_candidate_count"),
        "review_reason_v411",
        guard["decision_v411"].map(
            {"AUTO_MATCH": None, "REVIEW": "LOW_CONFIDENCE"}
        ),
    )
    guard["label_kind"] = "MATCH_EXACT"
    guard["ground_truth_siret"] = guard["predicted_siret"]

    return subject.ReferenceSources(
        queries=_write_source(tmp_path / "queries.parquet", queries),
        splits=_write_source(tmp_path / "splits.parquet", splits),
        candidates=_write_source(tmp_path / "candidates.parquet", candidates),
        ranker=_write_source(tmp_path / "ranker.parquet", ranker),
        scenes=_write_source(tmp_path / "scenes.parquet", scenes),
        acceptor=_write_source(tmp_path / "acceptor.parquet", acceptor),
        guard=_write_source(tmp_path / "guard.parquet", guard),
        query_evidence=_write_source(
            tmp_path / "query_evidence.parquet", query_evidence
        ),
        candidate_evidence=_write_source(
            tmp_path / "candidate_evidence.parquet", candidate_evidence
        ),
    )


def _frames(
    sources: subject.ReferenceSources, reader=pd.read_parquet
) -> dict[str, pd.DataFrame]:
    return subject.build_frames(
        sources,
        expected_query_count=3,
        expected_candidate_count=6,
        reader=reader,
    )


def _reader_for_loaded(
    sources: subject.ReferenceSources,
    loaded: dict[str, pd.DataFrame],
):
    names = iter(field.name for field in subject.fields(sources))

    def reader(payload, *, columns: list[str]) -> pd.DataFrame:
        return loaded[next(names)][columns].copy()

    return reader


def test_build_publishes_closed_label_free_reference_atomically(
    tmp_path: Path,
) -> None:
    sources = _reference_sources(tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()

    target = subject._build_synthetic_reference(
        output_root,
        sources=sources,
        expected_query_count=3,
        expected_candidate_count=6,
    )

    assert target.parent == output_root.resolve()
    assert {path.name for path in target.iterdir()} == {
        *subject.OUTPUT_FILES,
        "manifest.json",
    }
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["invariants"]["labels_opened"] is False
    assert manifest["invariants"]["model_retrained"] is False
    assert manifest["invariants"]["candidate_maximum_absolute"] == 100
    for filename, expected_columns in subject.OUTPUT_FILES.items():
        frame = pd.read_parquet(target / filename)
        assert list(frame.columns) == expected_columns
        subject._validate_safe_columns(filename, frame.columns)
        assert subject.file_sha256(target / filename) == manifest["outputs"][
            filename
        ]["sha256"]
    assert len(pd.read_parquet(target / "queries.parquet")) == 3
    assert len(pd.read_parquet(target / "candidates_features.parquet")) == 6
    guard = pd.read_parquet(target / "guard_reference.parquet").set_index(
        "query_id"
    )
    assert guard.loc["q0", "decision_v412"] == "AUTO_MATCH"
    assert guard.loc["q1", "decision_v412"] == "REVIEW"
    assert guard.loc["q2", "decision_v412"] == "REVIEW"


def test_reader_receives_only_exact_safe_projections(tmp_path: Path) -> None:
    sources = _reference_sources(tmp_path)
    source_fields = list(subject.fields(sources))
    calls: list[list[str]] = []
    call_index = 0

    def spy(payload, *, columns: list[str]) -> pd.DataFrame:
        nonlocal call_index
        calls.append(columns)
        source = getattr(sources, source_fields[call_index].name)
        call_index += 1
        return pd.read_parquet(source.path, columns=columns)

    _frames(sources, reader=spy)

    for index, field in enumerate(source_fields):
        assert calls[index] == subject.SOURCE_PROJECTIONS[field.name]
        subject._validate_safe_columns(field.name, calls[index])


def test_production_api_has_no_source_or_reader_injection() -> None:
    assert list(inspect.signature(subject.build_reference).parameters) == [
        "output_root"
    ]


def test_projection_decodes_the_bytes_that_were_hashed(tmp_path: Path) -> None:
    sources = _reference_sources(tmp_path)
    source = sources.queries
    original = pd.read_parquet(source.path, columns=subject.QUERY_COLUMNS)
    replacement = original.copy()
    replacement.loc[0, "crm_name"] = "SUBSTITUTED"
    replacement_path = tmp_path / "replacement.parquet"
    replacement.to_parquet(replacement_path, index=False)

    def swap_after_capture(payload, *, columns: list[str]) -> pd.DataFrame:
        source.path.unlink()
        replacement_path.rename(source.path)
        return pd.read_parquet(payload, columns=columns)

    observed = subject.read_projection(
        source,
        subject.QUERY_COLUMNS,
        reader=swap_after_capture,
    )
    assert observed.equals(original)
    assert pd.read_parquet(source.path).loc[0, "crm_name"] == "SUBSTITUTED"


@pytest.mark.parametrize("rank_value", [1.0, 1.25, True, "1"])
def test_candidate_rank_requires_physical_integer_dtype(
    tmp_path: Path, rank_value
) -> None:
    sources = _reference_sources(tmp_path)
    loaded = {
        field.name: pd.read_parquet(getattr(sources, field.name).path)
        for field in subject.fields(sources)
    }
    loaded["candidates"]["retrieval_rank"] = rank_value
    with pytest.raises(subject.ReferenceBuildError, match="CANDIDATE_RANK"):
        _frames(sources, reader=_reader_for_loaded(sources, loaded))


@pytest.mark.parametrize(
    ("frame_name", "column", "stop"),
    [
        ("candidates", "name_jaro_max", "RANKER_FEATURES_FINITE"),
        ("ranker", "ranker_score", "RANKER_SCORE_FINITE"),
        ("scenes", "ranker_gap_fraction", "SCENE_FEATURES_FINITE"),
        ("acceptor", "score", "ACCEPTOR_FINITE"),
    ],
)
def test_non_finite_model_values_are_rejected(
    tmp_path: Path, frame_name: str, column: str, stop: str
) -> None:
    sources = _reference_sources(tmp_path)
    loaded = {
        field.name: pd.read_parquet(getattr(sources, field.name).path)
        for field in subject.fields(sources)
    }
    loaded[frame_name].loc[0, column] = np.nan
    with pytest.raises(subject.ReferenceBuildError, match=stop):
        _frames(sources, reader=_reader_for_loaded(sources, loaded))


def test_non_top1_ranker_identity_drift_is_rejected(tmp_path: Path) -> None:
    sources = _reference_sources(tmp_path)
    loaded = {
        field.name: pd.read_parquet(getattr(sources, field.name).path)
        for field in subject.fields(sources)
    }
    row = loaded["ranker"]["ranker_rank"].eq(2)
    loaded["ranker"].loc[row, "candidate_siren"] = "999999999"
    with pytest.raises(subject.ReferenceBuildError, match="RANKER_IDENTITY"):
        _frames(sources, reader=_reader_for_loaded(sources, loaded))


def test_acceptor_decision_must_follow_frozen_score_threshold(
    tmp_path: Path,
) -> None:
    sources = _reference_sources(tmp_path)
    loaded = {
        field.name: pd.read_parquet(getattr(sources, field.name).path)
        for field in subject.fields(sources)
    }
    mask = (
        loaded["acceptor"]["model_family"].eq(subject.COMPACT_LOGIT)
        & loaded["acceptor"]["query_id"].eq("q1")
    )
    loaded["acceptor"].loc[mask, "decision"] = "AUTO_MATCH"
    loaded["guard"].loc[
        loaded["guard"]["query_id"].eq("q1"), "decision_v411"
    ] = "AUTO_MATCH"
    with pytest.raises(subject.ReferenceBuildError, match="ACCEPTOR_POLICY"):
        _frames(sources, reader=_reader_for_loaded(sources, loaded))


def test_hash_drift_fails_before_any_projection_read(tmp_path: Path) -> None:
    sources = _reference_sources(tmp_path)
    sources.queries.path.write_bytes(sources.queries.path.read_bytes() + b"drift")
    calls = 0

    def forbidden_reader(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("reader must not be reached")

    with pytest.raises(subject.ReferenceBuildError, match="INPUT_HASH"):
        _frames(sources, reader=forbidden_reader)
    assert calls == 0


@pytest.mark.parametrize(
    ("mutator", "stop"),
    [
        (
            lambda frames: frames["splits"].__setitem__(
                "split", ["dev", "dev", "fit", "fit"]
            ),
            "DEV_COUNT",
        ),
        (
            lambda frames: frames["ranker"].__setitem__(
                "candidate_siret",
                ["99999999999999", *frames["ranker"]["candidate_siret"].iloc[1:]],
            ),
            "RANKER_IDENTITY",
        ),
        (
            lambda frames: frames["acceptor"].__setitem__(
                "threshold", subject.FIXED_THRESHOLD - 0.1
            ),
            "ACCEPTOR_POLICY",
        ),
        (
            lambda frames: frames["scenes"].__setitem__(
                "predicted_siret",
                ["99999999999999", *frames["scenes"]["predicted_siret"].iloc[1:]],
            ),
            "SCENE_PREDICTION_IDENTITY",
        ),
        (
            lambda frames: frames["query_evidence"].__setitem__(
                "evidence_refs_json", ["[]"] * len(frames["query_evidence"])
            ),
            "EVIDENCE_PARITY",
        ),
        (
            lambda frames: frames["guard"].__setitem__(
                "decision_v412",
                ["REVIEW"] * len(frames["guard"]),
            ),
            "GUARD_PARITY",
        ),
    ],
)
def test_contract_drift_is_fail_closed(
    tmp_path: Path, mutator, stop: str
) -> None:
    sources = _reference_sources(tmp_path)
    loaded = {
        field.name: pd.read_parquet(getattr(sources, field.name).path)
        for field in subject.fields(sources)
    }
    mutator(loaded)

    names = iter(field.name for field in subject.fields(sources))

    def reader(payload, *, columns: list[str]) -> pd.DataFrame:
        name = next(names)
        return loaded[name][columns].copy()

    with pytest.raises(subject.ReferenceBuildError, match=stop):
        _frames(sources, reader=reader)


def test_pool_cap_and_contiguous_ranks_are_enforced(tmp_path: Path) -> None:
    sources = _reference_sources(tmp_path)
    loaded = {
        field.name: pd.read_parquet(getattr(sources, field.name).path)
        for field in subject.fields(sources)
    }
    loaded["candidates"].loc[
        loaded["candidates"]["query_id"].eq("q0"), "retrieval_rank"
    ] = [1, 3]
    loaded["ranker"].loc[
        loaded["ranker"]["query_id"].eq("q0"), "retrieval_rank"
    ] = [1, 3]

    names = iter(field.name for field in subject.fields(sources))

    def reader(payload, *, columns: list[str]) -> pd.DataFrame:
        name = next(names)
        return loaded[name][columns].copy()

    with pytest.raises(subject.ReferenceBuildError, match="POOL_CAP_OR_RANKS"):
        _frames(sources, reader=reader)


def test_immutable_rerun_preserves_first_publication(tmp_path: Path) -> None:
    sources = _reference_sources(tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    target = subject._build_synthetic_reference(
        output_root,
        sources=sources,
        expected_query_count=3,
        expected_candidate_count=6,
    )
    before = {
        path.name: path.read_bytes() for path in target.iterdir() if path.is_file()
    }

    with pytest.raises(FileExistsError, match="Immutable reference exists"):
        subject._build_synthetic_reference(
            output_root,
            sources=sources,
            expected_query_count=3,
            expected_candidate_count=6,
        )

    assert before == {
        path.name: path.read_bytes() for path in target.iterdir() if path.is_file()
    }
    assert not list(output_root.glob(".*.tmp-*"))


def test_symlink_source_is_rejected(tmp_path: Path) -> None:
    sources = _reference_sources(tmp_path)
    link = tmp_path / "queries-link.parquet"
    link.symlink_to(sources.queries.path)
    linked_sources = subject.ReferenceSources(
        **{
            field.name: (
                subject.SourceFile(link, sources.queries.sha256)
                if field.name == "queries"
                else getattr(sources, field.name)
            )
            for field in subject.fields(sources)
        }
    )

    with pytest.raises(subject.ReferenceBuildError, match="INPUT_OPEN"):
        _frames(linked_sources)


def test_symlink_output_root_is_rejected(tmp_path: Path) -> None:
    sources = _reference_sources(tmp_path)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(subject.ReferenceBuildError, match="OUTPUT_ROOT"):
        subject._build_synthetic_reference(
            linked_output,
            sources=sources,
            expected_query_count=3,
            expected_candidate_count=6,
        )
    assert not list(real_output.iterdir())


def test_terminal_revalidation_rejects_post_manifest_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _reference_sources(tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    original = subject._promote_exclusive_at

    def attack(parent_fd, staging_name, target_name, **kwargs):
        staging_fd = kwargs["staging_fd"]
        filename = "candidates_features.parquet"
        os.chmod(filename, 0o600, dir_fd=staging_fd)
        fd = os.open(filename, os.O_WRONLY, dir_fd=staging_fd)
        try:
            os.pwrite(fd, b"X", 0)
        finally:
            os.close(fd)
        os.chmod(filename, 0o400, dir_fd=staging_fd)
        return original(parent_fd, staging_name, target_name, **kwargs)

    monkeypatch.setattr(subject, "_promote_exclusive_at", attack)
    with pytest.raises(
        subject.ReferenceBuildError,
        match="OUTPUT_(DRIFT|HASH|IDENTITY)",
    ):
        subject._build_synthetic_reference(
            output_root,
            sources=sources,
            expected_query_count=3,
            expected_candidate_count=6,
        )
    assert not [path for path in output_root.iterdir() if not path.name.startswith(".")]


def test_promotion_rejects_named_staging_directory_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _reference_sources(tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    original = subject._promote_exclusive_at

    def substitute(parent_fd, staging_name, target_name, **kwargs):
        parked_name = f"{staging_name}.parked"
        os.rename(
            staging_name,
            parked_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        try:
            return original(parent_fd, staging_name, target_name, **kwargs)
        finally:
            os.rmdir(staging_name, dir_fd=parent_fd)
            os.rename(
                parked_name,
                staging_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )

    monkeypatch.setattr(subject, "_promote_exclusive_at", substitute)
    with pytest.raises(
        subject.ReferenceBuildError,
        match="STAGING_IDENTITY",
    ):
        subject._build_synthetic_reference(
            output_root,
            sources=sources,
            expected_query_count=3,
            expected_candidate_count=6,
        )
    assert not list(output_root.iterdir())
