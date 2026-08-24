from __future__ import annotations

import json

import numpy as np
import pytest

from src.xgb_matcher.retrieval_fold_fused_heldout_spool_v12 import (
    ParquetHeldoutMasterSpoolV12,
    projected_national_spool_bytes,
)
from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (
    FoldFusedInvariantError,
    MasterContext,
)
from src.xgb_matcher.retrieval_fold_fused_runtime_v12 import _MasterPayload
from src.xgb_matcher.retrieval_fold_fused_site_matrix_v12 import (
    compute_fold_fused_site_statistics_v12,
    materialize_fold_fused_site_matrix_v12,
)
from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
)
from tests.test_retrieval_fold_fused_runtime_v12 import _master_rows


def _context(query_id: str, fold: int, count: int = 140) -> MasterContext:
    rows = _master_rows(count)
    for row in rows:
        row["query_id"] = query_id
        row["fold"] = fold
    return MasterContext(
        query_id=query_id,
        scene_fold=fold,
        outer_fold=fold,
        candidate_sirets=tuple(str(row["siret"]) for row in rows),
        candidate_parents=tuple(str(row["siren"]) for row in rows),
        opaque=_MasterPayload(tuple(rows)),
    )


def _bits(context: MasterContext, config: AdmissionConfigV12) -> np.ndarray:
    statistics = compute_fold_fused_site_statistics_v12(
        context.opaque.raw_rows, config
    )
    return materialize_fold_fused_site_matrix_v12(
        statistics, config
    ).view(np.uint32)


def test_external_spool_replay_is_feature_bitwise_and_order_exact(tmp_path) -> None:
    config = AdmissionConfigV12.default()
    contexts = [_context(f"q-spool-{fold}", fold) for fold in (2, 3, 4)]
    expected = {context.query_id: _bits(context, config) for context in contexts}
    spool = ParquetHeldoutMasterSpoolV12(
        tmp_path / "spool",
        config=config,
        row_group_target=2_000,
    )
    for context in contexts:
        spool.append(context)
    artifact = spool.seal()
    replayed = list(spool.iter_contexts(artifact))

    assert [(value.query_id, value.scene_fold) for value in replayed] == [
        (value.query_id, value.scene_fold) for value in contexts
    ]
    assert artifact.query_count == 3
    assert artifact.row_count == 420
    assert artifact.size_bytes < 20 * 1024**3
    for context in replayed:
        assert np.array_equal(_bits(context, config), expected[context.query_id])
        assert context.outer_fold == context.scene_fold
        assert context.opaque.raw_rows[37]["signals"][0]["channel"] == "bodacc_relation"

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "SEALED_LABEL_BLIND"
    assert manifest["fold0_opened"] is False
    assert manifest["fold1_opened"] is False
    assert manifest["label_columns"] == []
    assert manifest["maximum_bytes"] == artifact.maximum_bytes
    assert manifest["artifact"]["size_bytes"] == artifact.size_bytes


def test_spool_rejects_labels_and_detects_post_seal_tampering(tmp_path) -> None:
    config = AdmissionConfigV12.default()
    labelled = _context("q-labelled", 2)
    labelled.opaque.raw_rows[0]["ground_truth_siret"] = "12345678900001"
    spool = ParquetHeldoutMasterSpoolV12(tmp_path / "labelled", config=config)
    with pytest.raises(FoldFusedInvariantError, match="forbidden label"):
        spool.append(labelled)

    derived_label = _context("q-derived-label", 2)
    derived_label.opaque.raw_rows[0]["label_confidence"] = 0.99
    derived_spool = ParquetHeldoutMasterSpoolV12(
        tmp_path / "derived-labelled", config=config
    )
    with pytest.raises(FoldFusedInvariantError, match="forbidden label"):
        derived_spool.append(derived_label)

    clean = _context("q-clean", 3)
    sealed = ParquetHeldoutMasterSpoolV12(tmp_path / "sealed", config=config)
    sealed.append(clean)
    artifact = sealed.seal()
    with artifact.path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(FoldFusedInvariantError, match="bytes/hash drift"):
        list(sealed.iter_contexts(artifact))


def test_national_size_projection_is_linear_and_conservative_integer() -> None:
    assert projected_national_spool_bytes(
        sample_bytes=1_001,
        sample_queries=512,
        national_queries=23_571,
    ) == 46_084
