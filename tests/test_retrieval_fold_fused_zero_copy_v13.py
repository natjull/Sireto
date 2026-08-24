from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.test_retrieval_fold_fused_sharded_oof_v12 import (
    _Backend,
    _Contexts,
    _Ledger,
    _Selector,
    _build,
    _labels,
    _lineage,
    _replay,
    _scene_index,
    _Sampler,
    _Source,
)
from src.xgb_matcher.retrieval_fold_fused_fragment_dataset_v13 import (
    BufferedCompactTrainingStoreV13,
    XGBoostFragmentedSiteLambdaMARTBackendV13,
    build_logical_parity_digests_v13,
    fit_site_models_sequential_from_fragments_v13,
    load_build_fragment_dataset_v13,
    load_fragmented_train_arrays_v13,
    logical_heldout_view_from_parquet_v13,
    logical_train_view_from_parquet_v13,
    merkle_root_v13,
    seal_build_fragment_dataset_v13,
)
from src.xgb_matcher.retrieval_fold_fused_fragment_outputs_v13 import (
    TOP100_SCHEMA,
    TOP500_SCHEMA,
    iter_compact_output_query_tables_v13,
    load_compact_replay_dataset_v13,
    logical_output_merkle_from_parquet_v13,
    rehydrate_top500_row_v13,
    replay_fragment_v13,
    seal_compact_replay_dataset_v13,
)
from src.xgb_matcher import retrieval_fold_fused_fragment_outputs_v13 as compact_outputs
from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (
    CompactTrainBatch,
    ExecutionCaps,
    FoldFusedInvariantError,
    XGBoostSiteLambdaMARTBackend,
)
from src.xgb_matcher.retrieval_fold_fused_sharded_oof_v12 import (
    build_shard_v12,
    fit_site_models_sequential_v12,
    merge_heldout_shards_v12,
    merge_training_shards_v12,
    plan_query_shards,
)
from src.xgb_matcher.retrieval_fold_fused_zero_copy_run_v13 import (
    CANONICAL_FAMILIES,
    load_zero_copy_canonical_run_v13,
    seal_zero_copy_canonical_run_v13,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _build_w1_w3(tmp_path: Path):
    index = _scene_index()
    query_ids = tuple(index["query_id"].to_pylist())
    labels = _labels()
    w1_spec = plan_query_shards(index, query_ids, 1)[0]
    w1 = _build(w1_spec, tmp_path / "w1_build", labels, [])
    w3_specs = plan_query_shards(index, query_ids, 3)
    w3 = tuple(
        _build(spec, tmp_path / f"w3_build_{ordinal}", labels, [])
        for ordinal, spec in enumerate(w3_specs)
    )
    return index, query_ids, labels, w1_spec, w1, w3_specs, w3


def test_merkle_rule_is_domain_separated_duplicate_last() -> None:
    records = ({"value": 1}, {"value": 2}, {"value": 3})
    leaves = [
        hashlib.sha256(b"\x00" + _canonical_json(value).encode()).digest()
        for value in records
    ]
    leaves.append(leaves[-1])
    level = [
        hashlib.sha256(b"\x01" + leaves[index] + leaves[index + 1]).digest()
        for index in range(0, len(leaves), 2)
    ]
    expected = hashlib.sha256(b"\x01" + level[0] + level[1]).hexdigest()
    assert merkle_root_v13(records) == expected


def test_build_fragments_are_zero_copy_and_logically_equal_to_w1(
    tmp_path: Path,
) -> None:
    _, _, _, _, w1, _, w3 = _build_w1_w3(tmp_path)
    w1_dataset = seal_build_fragment_dataset_v13(
        (w1,), tmp_path / "w1_dataset"
    )
    w3_dataset = seal_build_fragment_dataset_v13(
        w3, tmp_path / "w3_dataset"
    )
    relocated_manifest = seal_build_fragment_dataset_v13(
        w3, tmp_path / "nested" / "w3_dataset"
    )
    loaded = load_build_fragment_dataset_v13(w3_dataset.manifest_path)

    assert list((tmp_path / "w3_dataset").iterdir()) == [
        w3_dataset.manifest_path
    ]
    assert w1_dataset.physical_merkle_root != w3_dataset.physical_merkle_root
    assert relocated_manifest.dataset_id == w3_dataset.dataset_id
    assert (
        relocated_manifest.physical_merkle_root
        == w3_dataset.physical_merkle_root
    )
    assert (
        w1_dataset.heldout_logical_merkle_root
        == loaded.heldout_logical_merkle_root
    )
    for outer in (2, 3, 4):
        assert (
            w1_dataset.train[outer].logical_merkle_root
            == loaded.train[outer].logical_merkle_root
        )
        assert (
            w1_dataset.train[outer].group_sequence_sha256
            == loaded.train[outer].group_sequence_sha256
        )

    merged_train = merge_training_shards_v12(w3, tmp_path / "merged_train")
    for outer in (2, 3, 4):
        fragmented = load_fragmented_train_arrays_v13(loaded, outer)
        expected_values, expected_labels, expected_qid = (
            XGBoostSiteLambdaMARTBackend._load(merged_train.matrices[outer])
        )
        assert np.array_equal(fragmented.values, expected_values)
        assert np.array_equal(fragmented.labels, expected_labels)
        assert np.array_equal(fragmented.qid, expected_qid)
        bridge = logical_train_view_from_parquet_v13(
            [merged_train.matrices[outer].path], outer
        )
        assert bridge["logical_merkle_root"] == loaded.train[outer].logical_merkle_root
        assert bridge["group_sequence_sha256"] == loaded.train[outer].group_sequence_sha256

    merged_heldout = merge_heldout_shards_v12(w3, tmp_path / "merged_heldout")
    heldout_bridge = logical_heldout_view_from_parquet_v13(
        [merged_heldout.artifact.path]
    )
    assert heldout_bridge["logical_merkle_root"] == loaded.heldout_logical_merkle_root
    parity = build_logical_parity_digests_v13(loaded)
    assert parity["heldout_spool"] == loaded.heldout_logical_merkle_root


def test_train_reader_accepts_buffered_multiquery_row_groups(tmp_path: Path) -> None:
    _, _, _, _, w1, _, _ = _build_w1_w3(tmp_path)
    source = w1.train_matrices[2].path
    parquet = pq.ParquetFile(source)
    tables = [
        parquet.read_row_group(index) for index in range(parquet.num_row_groups)
    ]
    buffered = tmp_path / "buffered_train.parquet"
    writer = pq.ParquetWriter(buffered, parquet.schema_arrow, compression="zstd")
    writer.write_table(pa.concat_tables(tables), row_group_size=sum(value.num_rows for value in tables))
    writer.close()

    source_view = logical_train_view_from_parquet_v13([source], 2)
    buffered_view = logical_train_view_from_parquet_v13([buffered], 2)
    assert source_view == buffered_view
    assert pq.ParquetFile(buffered).num_row_groups == 1
    assert source_view["query_count"] > 1

    store = BufferedCompactTrainingStoreV13(
        tmp_path / "buffered_store", row_group_target=65_536
    )
    names = tuple(json.loads(parquet.schema_arrow.metadata[b"feature_names_json"]))
    for table in tables:
        values = np.asarray(
            table["feature_values"].combine_chunks().values, dtype=np.float32
        ).reshape(table.num_rows, len(names))
        store.append(
            CompactTrainBatch(
                query_id=str(table["query_id"][0].as_py()),
                scene_fold=int(table["scene_fold"][0].as_py()),
                outer_fold=2,
                candidate_sirets=tuple(table["candidate_siret"].to_pylist()),
                candidate_parents=tuple(table["candidate_parent"].to_pylist()),
                feature_names=names,
                values=np.ascontiguousarray(values),
                labels=np.asarray(table["site_label"], dtype=np.int8),
                statistics_sha256=str(table["statistics_sha256"][0].as_py()),
            )
        )
    buffered_artifact = store.seal_outer(2)
    assert pq.ParquetFile(buffered_artifact.path).num_row_groups == 1
    assert logical_train_view_from_parquet_v13(
        [buffered_artifact.path], 2
    ) == source_view


def test_compact_output_writer_uses_frozen_parquet_encoding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    class Writer:
        def __init__(self, _path: Path, _schema: pa.Schema, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(compact_outputs.pq, "ParquetWriter", Writer)
    compact_outputs.BufferedCompactOOFOutputStoreV13(tmp_path / "compact")

    assert calls == [
        {
            "compression": "zstd",
            "compression_level": 3,
            "use_dictionary": True,
            "write_statistics": True,
            "data_page_version": "1.0",
        }
    ] * 3


def test_build_shard_injects_buffered_train_sink_without_science_change(
    tmp_path: Path,
) -> None:
    index = _scene_index()
    query_ids = tuple(index["query_id"].to_pylist())
    labels = _labels()
    spec = plan_query_shards(index, query_ids, 1)[0]
    legacy = _build(spec, tmp_path / "legacy", labels, [])
    truth = {query_id: value.human_siret for query_id, value in labels.items()}
    buffered = build_shard_v12(
        spec,
        source=_Source(spec, []),
        labels=_Ledger(labels, []),
        contexts=_Contexts(truth, []),
        sampler=_Sampler(),
        admission_config=SimpleNamespace(channels=()),
        output_dir=tmp_path / "buffered",
        lineage=_lineage(),
        caps=ExecutionCaps(fit_sample=4),
        train_store_factory=lambda path: BufferedCompactTrainingStoreV13(
            path, row_group_target=65_536
        ),
    )
    legacy_dataset = seal_build_fragment_dataset_v13(
        (legacy,), tmp_path / "legacy_dataset"
    )
    buffered_dataset = seal_build_fragment_dataset_v13(
        (buffered,), tmp_path / "buffered_dataset"
    )
    for outer in (2, 3, 4):
        assert pq.ParquetFile(buffered.train_matrices[outer].path).num_row_groups == 1
        assert (
            buffered_dataset.train[outer].logical_merkle_root
            == legacy_dataset.train[outer].logical_merkle_root
        )
        assert (
            buffered_dataset.train[outer].group_sequence_sha256
            == legacy_dataset.train[outer].group_sequence_sha256
        )


def test_fragment_fit_is_model_byte_exact_to_merged_v12(tmp_path: Path) -> None:
    _, _, _, _, _, _, w3 = _build_w1_w3(tmp_path)
    dataset = seal_build_fragment_dataset_v13(w3, tmp_path / "dataset")
    merged = merge_training_shards_v12(w3, tmp_path / "merged")
    params = {
        "objective": "rank:pairwise",
        "n_estimators": 6,
        "max_depth": 2,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "random_state": 1729,
        "n_jobs": 1,
    }
    legacy = fit_site_models_sequential_v12(
        XGBoostSiteLambdaMARTBackend(
            params=params, output_dir=tmp_path / "legacy_models"
        ),
        merged,
    )
    fragmented = fit_site_models_sequential_from_fragments_v13(
        XGBoostFragmentedSiteLambdaMARTBackendV13(
            params=params, output_dir=tmp_path / "fragment_models"
        ),
        dataset,
    )
    assert {outer: value.sha256 for outer, value in legacy.items()} == {
        outer: value.sha256 for outer, value in fragmented.items()
    }
    for outer in (2, 3, 4):
        arrays = load_fragmented_train_arrays_v13(dataset, outer)
        assert np.array_equal(
            legacy[outer].handle.predict(arrays.values),
            fragmented[outer].handle.predict(arrays.values),
        )


def test_compact_replay_matches_v12_and_natural_join_rehydrates(
    tmp_path: Path,
) -> None:
    _, _, labels, w1_spec, w1, w3_specs, w3 = _build_w1_w3(tmp_path)
    build_dataset = seal_build_fragment_dataset_v13(
        w3, tmp_path / "build_dataset"
    )
    merged_train = merge_training_shards_v12(w3, tmp_path / "train_for_models")
    models = fit_site_models_sequential_v12(
        _Backend(tmp_path / "models"), merged_train
    )
    first_spec = w3_specs[0]
    first_build = w3[0]
    truth = {query_id: value.human_siret for query_id, value in labels.items()}
    replay_common = {
        "heldout": first_build.heldout_master,
        "labels": _Ledger(labels, []),
        "contexts": _Contexts(truth, []),
        "model_backend": _Backend(tmp_path / "model_scope_score"),
        "selector": _Selector(),
        "lineage": _lineage(),
        "build_manifest_sha256": first_build.manifest_sha256,
        "heldout_dataset_merkle_root": build_dataset.heldout_logical_merkle_root,
        "caps": ExecutionCaps(fit_sample=4),
    }
    with pytest.raises(FoldFusedInvariantError, match="model|fold"):
        replay_fragment_v13(
            first_spec,
            models={2: models[2], 3: models[3]},
            output_dir=tmp_path / "extra_model_replay",
            **replay_common,
        )
    with pytest.raises(FoldFusedInvariantError, match="model|fold"):
        replay_fragment_v13(
            first_spec,
            models={3: models[3]},
            output_dir=tmp_path / "wrong_model_replay",
            **replay_common,
        )
    baseline = _replay(
        w1_spec, w1, tmp_path / "baseline_replay", labels, models
    )
    compact = tuple(
        replay_fragment_v13(
            spec,
            heldout=build.heldout_master,
            labels=_Ledger(labels, []),
            contexts=_Contexts(truth, []),
            model_backend=_Backend(tmp_path / f"score_{ordinal}"),
            models={spec.folds[0]: models[spec.folds[0]]},
            selector=_Selector(),
            output_dir=tmp_path / f"compact_replay_{ordinal}",
            lineage=_lineage(),
            build_manifest_sha256=build.manifest_sha256,
            heldout_dataset_merkle_root=build_dataset.heldout_logical_merkle_root,
            caps=ExecutionCaps(fit_sample=4),
            row_group_target=500,
        )
        for ordinal, (spec, build) in enumerate(zip(w3_specs, w3, strict=True))
    )
    replay_dataset = seal_compact_replay_dataset_v13(
        compact, tmp_path / "replay_dataset"
    )
    compact_w1 = replay_fragment_v13(
        w1_spec,
        heldout=w1.heldout_master,
        labels=_Ledger(labels, []),
        contexts=_Contexts(truth, []),
        model_backend=_Backend(tmp_path / "score_w1"),
        models=models,
        selector=_Selector(),
        output_dir=tmp_path / "compact_replay_w1",
        lineage=_lineage(),
        build_manifest_sha256=w1.manifest_sha256,
        heldout_dataset_merkle_root=build_dataset.heldout_logical_merkle_root,
        caps=ExecutionCaps(fit_sample=4),
        row_group_target=500,
    )
    relocated_replay = seal_compact_replay_dataset_v13(
        compact, tmp_path / "nested" / "replay_dataset"
    )
    loaded = load_compact_replay_dataset_v13(replay_dataset.manifest_path)
    assert relocated_replay.dataset_id == replay_dataset.dataset_id
    assert relocated_replay.physical_merkle_root == replay_dataset.physical_merkle_root
    assert (
        compact_w1.feature_values_logical_merkle_root
        == loaded.feature_values_logical_merkle_root
    )

    assert TOP500_SCHEMA.names == [
        "query_id",
        "fold",
        "candidate_siret",
        "candidate_parent",
        "rank",
        "site_score",
        "selection_reason",
    ]
    assert TOP500_SCHEMA.field("site_score").type == pa.float64()
    assert "payload_json" not in TOP500_SCHEMA.names
    assert TOP100_SCHEMA.names[-1] == "payload_json"
    for name, legacy_name in (
        ("top500", "top500"),
        ("top100", "top100"),
        ("outcomes", "outcomes"),
    ):
        compact_paths = [value.path for value in loaded.outputs[name]]
        assert logical_output_merkle_from_parquet_v13(
            [baseline.outputs[legacy_name].path], name
        ) == logical_output_merkle_from_parquet_v13(compact_paths, name)

    # Two fixture queries per shard are packed into one row group rather than
    # one tiny row group per query.
    assert all(value.row_group_count == 1 for value in loaded.outputs["outcomes"])
    first_top500 = next(
        iter(iter_compact_output_query_tables_v13(loaded, "top500"))
    )
    compact_row = first_top500.to_pylist()[0]
    hydrated = rehydrate_top500_row_v13(compact_row, build_dataset)
    assert hydrated["query_id"] == compact_row["query_id"]
    assert hydrated["siret"] == compact_row["candidate_siret"]
    with pytest.raises(FoldFusedInvariantError, match="rehydrated"):
        rehydrate_top500_row_v13(
            {**compact_row, "candidate_siret": "99999999999999"},
            build_dataset,
        )
    first_top100 = next(
        iter(iter_compact_output_query_tables_v13(loaded, "top100"))
    ).to_pylist()[0]
    proof_payload = json.loads(first_top100["payload_json"])
    assert proof_payload["official_evidence_sources"] == ["SIRENE"]

    canonical = seal_zero_copy_canonical_run_v13(
        build_dataset, loaded, models, tmp_path / "canonical_run"
    )
    relocated_canonical = seal_zero_copy_canonical_run_v13(
        build_dataset, loaded, models, tmp_path / "nested" / "canonical_run"
    )
    loaded_canonical = load_zero_copy_canonical_run_v13(
        canonical.manifest_path
    )
    assert tuple(canonical.canonical_datasets) == CANONICAL_FAMILIES
    assert loaded_canonical.run_id == canonical.run_id
    assert relocated_canonical.run_id == canonical.run_id
    assert all(value.fragment_count for value in canonical.canonical_datasets.values())
    assert not list((tmp_path / "canonical_run").glob("*.parquet"))
    assert len(list((tmp_path / "canonical_run").glob("*.json"))) == 8

    # Even a valid, uniquely copied and self-rehashed extra leaf is forbidden:
    # the inventory is exactly seven data families plus three model files.
    forged = json.loads(canonical.manifest_path.read_text(encoding="utf-8"))
    extra_path = tmp_path / "extra_model.json"
    shutil.copyfile(canonical.artifact_inventory[-1].path, extra_path)
    extra = dict(forged["artifact_inventory"][-1])
    extra["path"] = str(extra_path)
    extra["fragment_ordinal"] = 3
    forged["artifact_inventory"].append(extra)
    identity_extra = {key: value for key, value in extra.items() if key != "path"}
    forged["identity"]["artifact_inventory"].append(identity_extra)
    forged["run_id"] = _sha_json(forged["identity"])
    forged_path = tmp_path / "canonical_run" / "forged_extra_inventory.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(FoldFusedInvariantError, match="exactly"):
        load_zero_copy_canonical_run_v13(forged_path)


def test_complete_fragment_tree_reopens_after_atomic_root_rename(
    tmp_path: Path,
) -> None:
    stage = tmp_path / ".national-stage"
    _, _, labels, _w1_spec, _w1, w3_specs, w3 = _build_w1_w3(stage)
    build_dataset = seal_build_fragment_dataset_v13(
        w3,
        stage / "build_dataset",
        relocation_root=stage,
    )
    train = merge_training_shards_v12(
        w3, stage / "train_for_site_models"
    )
    models = fit_site_models_sequential_v12(
        _Backend(stage / "models"), train
    )
    truth = {
        query_id: value.human_siret for query_id, value in labels.items()
    }
    compact = tuple(
        replay_fragment_v13(
            spec,
            heldout=build.heldout_master,
            labels=_Ledger(labels, []),
            contexts=_Contexts(truth, []),
            model_backend=_Backend(stage / f"score_{ordinal}"),
            models={spec.folds[0]: models[spec.folds[0]]},
            selector=_Selector(),
            output_dir=stage / "replay_workers" / f"fold{spec.folds[0]}",
            lineage=_lineage(),
            build_manifest_sha256=build.manifest_sha256,
            heldout_dataset_merkle_root=(
                build_dataset.heldout_logical_merkle_root
            ),
            caps=ExecutionCaps(fit_sample=4),
            row_group_target=500,
            relocation_root=stage,
        )
        for ordinal, (spec, build) in enumerate(
            zip(w3_specs, w3, strict=True)
        )
    )
    replay_dataset = seal_compact_replay_dataset_v13(
        compact,
        stage / "replay_dataset",
        relocation_root=stage,
    )
    canonical = seal_zero_copy_canonical_run_v13(
        build_dataset,
        replay_dataset,
        models,
        stage / "canonical",
        relocation_root=stage,
    )
    expected = (
        build_dataset.dataset_id,
        replay_dataset.dataset_id,
        canonical.run_id,
    )

    promoted = tmp_path / "national"
    stage.rename(promoted)
    loaded_build = load_build_fragment_dataset_v13(
        promoted / "build_dataset" / "manifest.json"
    )
    loaded_replay = load_compact_replay_dataset_v13(
        promoted / "replay_dataset" / "manifest.json"
    )
    loaded_canonical = load_zero_copy_canonical_run_v13(
        promoted / "canonical" / "manifest.json"
    )
    assert (
        loaded_build.dataset_id,
        loaded_replay.dataset_id,
        loaded_canonical.run_id,
    ) == expected
    assert not stage.exists()


def test_manifest_tamper_order_and_fragment_hash_fail_closed(tmp_path: Path) -> None:
    _, _, _, _, _, _, w3 = _build_w1_w3(tmp_path)
    dataset = seal_build_fragment_dataset_v13(w3, tmp_path / "dataset")
    raw = json.loads(dataset.manifest_path.read_text(encoding="utf-8"))

    tampered_root = json.loads(json.dumps(raw))
    tampered_root["physical_merkle_root"] = "0" * 64
    tampered_root["identity"]["physical_merkle_root"] = "0" * 64
    tampered_root["dataset_id"] = _sha_json(tampered_root["identity"])
    root_path = tmp_path / "tampered_root.json"
    root_path.write_text(json.dumps(tampered_root), encoding="utf-8")
    with pytest.raises(FoldFusedInvariantError, match="Merkle root"):
        load_build_fragment_dataset_v13(root_path, verify_logical=False)

    tampered_order = json.loads(json.dumps(raw))
    tampered_order["specs"] = list(reversed(tampered_order["specs"]))
    order_path = tmp_path / "tampered_order.json"
    order_path.write_text(json.dumps(tampered_order), encoding="utf-8")
    with pytest.raises(FoldFusedInvariantError):
        load_build_fragment_dataset_v13(order_path, verify_logical=False)

    fragment = dataset.heldout_fragments[0].path
    with fragment.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(FoldFusedInvariantError, match="hash|corrupt"):
        load_build_fragment_dataset_v13(dataset.manifest_path)


def test_microbench_compaction_removes_payload_and_buffers_queries(
    tmp_path: Path,
) -> None:
    # This is a deterministic storage proxy rather than a fragile wall-clock
    # assertion: payload duplication and row-group explosion are the two
    # physical costs v13 removes.
    _, _, labels, w1_spec, w1, _, _ = _build_w1_w3(tmp_path)
    models = fit_site_models_sequential_v12(
        _Backend(tmp_path / "models"), w1.train_matrices
    )
    baseline = _replay(w1_spec, w1, tmp_path / "legacy", labels, models)
    truth = {query_id: value.human_siret for query_id, value in labels.items()}
    build_dataset = seal_build_fragment_dataset_v13(
        (w1,), tmp_path / "build_dataset"
    )
    compact = replay_fragment_v13(
        w1_spec,
        heldout=w1.heldout_master,
        labels=_Ledger(labels, []),
        contexts=_Contexts(truth, []),
        model_backend=_Backend(tmp_path / "score"),
        models=models,
        selector=_Selector(),
        output_dir=tmp_path / "compact",
        lineage=_lineage(),
        build_manifest_sha256=w1.manifest_sha256,
        heldout_dataset_merkle_root=build_dataset.heldout_logical_merkle_root,
        caps=ExecutionCaps(fit_sample=4),
    )
    legacy_file = pq.ParquetFile(baseline.outputs["top500"].path)
    compact_file = pq.ParquetFile(compact.outputs["top500"].path)
    assert compact.outputs["top500"].size_bytes < baseline.outputs["top500"].path.stat().st_size
    assert compact_file.num_row_groups < legacy_file.num_row_groups
    assert compact_file.schema_arrow.names == TOP500_SCHEMA.names
