from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (
    ExistingV12SelectionAdapter,
    FeatureBatch,
    MasterContext,
    RankingSelection,
)
from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    select_top100,
)


FEATURES = ("joint_name_geo", "parent_score_probability")


def _scene(
    size: int,
    *,
    seed: int,
    permutation: np.ndarray | None = None,
    joint_probability: float = 0.02,
) -> tuple[MasterContext, FeatureBatch, np.ndarray]:
    rng = np.random.default_rng(seed)
    parents = [f"{610_000_000 + index // 5:09d}" for index in range(size)]
    sirets = [f"{parent}{index % 5 + 1:05d}" for index, parent in enumerate(parents)]
    # Deliberate score ties exercise the SIRET tie-break.  Parent probabilities
    # also contain ties and exact zeros.
    scores = np.round(rng.normal(size=size), 1).astype(np.float64)
    parent_probability = np.round(rng.random(size), 2).astype(np.float32)
    joint = (rng.random(size) < joint_probability).astype(np.float32)
    matrix = np.column_stack((joint, parent_probability)).astype(np.float32)
    if permutation is None:
        permutation = np.arange(size, dtype=np.int64)
    permutation = np.asarray(permutation, dtype=np.int64)
    permuted_sirets = tuple(sirets[int(index)] for index in permutation)
    permuted_parents = tuple(parents[int(index)] for index in permutation)
    context = MasterContext(
        query_id=f"q-{seed}-{size}",
        scene_fold=2,
        outer_fold=3,
        candidate_sirets=permuted_sirets,
        candidate_parents=permuted_parents,
    )
    batch = FeatureBatch(
        query_id=context.query_id,
        scene_fold=context.scene_fold,
        outer_fold=context.outer_fold,
        candidate_indices=tuple(range(size)),
        candidate_sirets=permuted_sirets,
        candidate_parents=permuted_parents,
        feature_names=FEATURES,
        values=np.ascontiguousarray(matrix[permutation], dtype=np.float32),
        statistics_sha256="a" * 64,
    )
    return context, batch, np.ascontiguousarray(scores[permutation])


def _pandas_oracle(
    context: MasterContext,
    features: FeatureBatch,
    scores: np.ndarray,
    config: AdmissionConfigV12,
) -> RankingSelection:
    order = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), features.candidate_sirets[index]),
    )
    ranked_rank = np.empty(len(scores), dtype=np.int32)
    for rank, index in enumerate(order, start=1):
        ranked_rank[index] = rank
    frame = pd.DataFrame(
        {
            "query_id": [context.query_id] * len(scores),
            "candidate_siret": features.candidate_sirets,
            "candidate_parent": features.candidate_parents,
            "site_score": scores,
            "ranked_rank": ranked_rank,
            "prerank_rank": [
                int(rank) if int(rank) <= config.prerank_cap else math.nan
                for rank in ranked_rank
            ],
            "joint_name_geo": features.values[:, 0],
            "parent_score_probability": features.values[:, 1],
            "_context_index": np.arange(len(scores), dtype=np.int32),
        }
    )
    selected = select_top100(frame, config)
    return RankingSelection(
        tuple(order[: min(config.prerank_cap, len(order))]),
        tuple(int(value) for value in selected["_context_index"]),
        tuple(str(value) for value in selected["selection_reason"]),
    )


def _assert_exact(
    context: MasterContext,
    features: FeatureBatch,
    scores: np.ndarray,
    config: AdmissionConfigV12,
) -> RankingSelection:
    expected = _pandas_oracle(context, features, scores, config)
    actual = ExistingV12SelectionAdapter(config).select(
        context,
        features,
        scores,
        prerank_cap=500,
        product_cap=100,
    )
    assert actual == expected
    return actual


def test_numpy_selector_matches_pandas_oracle_on_128_varied_scenes() -> None:
    config = AdmissionConfigV12.default()
    sizes = (1, 2, 5, 17, 49, 99, 100, 101, 119)
    for seed in range(128):
        size = sizes[seed % len(sizes)]
        rng = np.random.default_rng(10_000 + seed)
        permutation = rng.permutation(size)
        context, features, scores = _scene(
            size,
            seed=seed,
            permutation=permutation,
            joint_probability=(seed % 7) / 100.0,
        )
        _assert_exact(context, features, scores, config)


def test_numpy_selector_matches_sparse_protection_at_master2000_permutations() -> None:
    config = AdmissionConfigV12.default()
    baseline_ids: tuple[str, ...] | None = None
    baseline_reasons: tuple[str, ...] | None = None
    for seed in range(12):
        permutation = np.random.default_rng(50_000 + seed).permutation(2_000)
        context, features, scores = _scene(
            2_000,
            seed=777,
            permutation=permutation,
            joint_probability=0.01,
        )
        selected = _assert_exact(context, features, scores, config)
        selected_ids = tuple(
            context.candidate_sirets[index] for index in selected.product_indices
        )
        if baseline_ids is None:
            baseline_ids = selected_ids
            baseline_reasons = selected.product_reasons
        else:
            assert selected_ids == baseline_ids
            assert selected.product_reasons == baseline_reasons


def test_numpy_selector_matches_when_protected_lane_alone_fills_top100() -> None:
    config = AdmissionConfigV12.default()
    context, features, scores = _scene(
        800,
        seed=991,
        permutation=np.random.default_rng(992).permutation(800),
        joint_probability=0.8,
    )
    selected = _assert_exact(context, features, scores, config)
    assert len(selected.product_indices) == 100
    assert set(selected.product_reasons) == {"protected_joint_name_geo"}
