from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any, Mapping

import numpy as np
import pytest

from src.xgb_matcher.retrieval_fold_fused_v12 import (
    LinearParentMatrixScorerV12,
    PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12,
    PARENT_MODEL_KEYS_V12,
    XGBoostParentMatrixScorerV12,
    assert_logical_result_parity_v12,
    compile_label_blind_scene_v12,
    materialize_outer_context_v12,
    prepare_fold_fused_scene_v12,
    score_three_outer_contexts_v12,
    validate_parent_scorer_catalog_v12,
)
from src.xgb_matcher.retrieval_sidecars_v11 import LocalSite, SiteQueryResult
from src.xgb_matcher.retrieval_top100_cascade_v12 import (
    CASCADE_PARENT_FEATURES,
    RetrievalTop100CascadeV12,
    RetrievalTop100CascadeV12Config,
    XGBoostParentScorerV12,
)


def _site(
    siret: str,
    *,
    insee: str = "75056",
    postcode: str = "75001",
    number: str = "10",
) -> LocalSite:
    return LocalSite(
        siret=siret,
        siren=siret[:9],
        insee=insee,
        postcode=postcode,
        street_number=number,
        street_number_suffix="",
        street_normalized="RUE DE LA PAIX",
        current_address_normalized=f"{number} RUE DE LA PAIX",
        administrative_state="A",
        is_headquarters=False,
    )


def _signal(channel: str, rank: int) -> dict[str, Any]:
    return {
        "source": "SIRENE_CURRENT:fixture",
        "channel": channel,
        "rank": rank,
        "score": 1.0 / rank,
        "parent_identifier": "",
        "parent_rank": None,
        "site_rank": None,
        "relation_direction": "",
    }


def _row(
    siret: str,
    channels: tuple[str, ...],
    *,
    rank: int,
    fold: int = 2,
) -> dict[str, Any]:
    return {
        "query_id": "q-fused",
        "fold": fold,
        "crm_name": "ALPHA",
        "crm_address": "10 RUE DE LA PAIX",
        "crm_insee": "75056",
        "crm_postcode": "75001",
        "union_rank": rank,
        "siret": siret,
        "siren": siret[:9],
        "channels": list(channels),
        "sources": ["SIRENE_CURRENT:fixture"],
        "signals": [_signal(channel, rank) for channel in channels],
        "candidate_names": ["ALPHA"],
        "candidate_addresses": ["10 RUE DE LA PAIX"],
        "candidate_insee": "75056",
        "candidate_postcode": "75001",
        "candidate_number": "10",
        "candidate_state": "A",
    }


@dataclass
class _CountingSidecars:
    sites: tuple[LocalSite, ...]
    commune_count: int = 50_000
    census_calls: int = 0
    expansion_calls: int = 0

    def all_sites_by_insee_if_fits(
        self, insee: str, *, max_sites: int
    ) -> SiteQueryResult:
        self.census_calls += 1
        matches = tuple(site for site in self.sites if site.insee == insee)
        count = max(self.commune_count, len(matches))
        if count > max_sites:
            return SiteQueryResult((), "INSEE_TOO_LARGE", count, max_sites, True)
        return SiteQueryResult(matches, "INSEE_ALL", count, max_sites, False)

    def sites_by_sirens_geo(self, **kwargs: Any) -> dict[str, SiteQueryResult]:
        self.expansion_calls += 1
        cap = int(kwargs["max_sites_per_siren"])
        insee = str(kwargs.get("insee") or "")
        postcode = str(kwargs.get("postcode") or "")
        output: dict[str, SiteQueryResult] = {}
        for siren in kwargs["sirens"]:
            matches = tuple(
                site
                for site in self.sites
                if site.siren == siren
                and (site.insee == insee if insee else site.postcode == postcode)
            )
            output[siren] = SiteQueryResult(
                matches[:cap],
                "SIREN_GEO_BATCH",
                len(matches),
                cap,
                len(matches) > cap,
            )
        return output


class _CountingMatrixScorer:
    def __init__(self, feature_index: int, multiplier: float = 1.0) -> None:
        self.feature_index = feature_index
        self.multiplier = np.float32(multiplier)
        self.calls = 0
        self.shapes: list[tuple[int, int]] = []

    def score_matrix(self, values: np.ndarray) -> np.ndarray:
        self.calls += 1
        matrix = np.asarray(values, dtype=np.float32)
        self.shapes.append(matrix.shape)
        return matrix[:, self.feature_index] * self.multiplier


def _config() -> RetrievalTop100CascadeV12Config:
    return RetrievalTop100CascadeV12Config(
        master_reference_cap=20,
        hydration_shortlist_cap=10,
        parent_shortlist_cap=1,
        small_insee_threshold=5,
        parent_expansion_budget=5,
        max_sites_per_parent=10,
        relation_seed_limit=4,
        relation_reference_budget=4,
        rescue_parent_budget=1,
        rescue_parent_reserve=1,
        rescue_min_name_parents=1,
        rescue_min_strong_parents=1,
    )


def _rows() -> list[dict[str, Any]]:
    return [
        _row(
            "11111111100001",
            ("legal_name_exact", "address_word"),
            rank=10,
        ),
        _row("22222222200001", ("name_word",), rank=1),
        _row("33333333300001", ("address_exact",), rank=2),
    ]


def _scorers() -> dict[str, _CountingMatrixScorer]:
    exact_name = CASCADE_PARENT_FEATURES.index("exact_name_evidence")
    exact_address = CASCADE_PARENT_FEATURES.index("exact_address_evidence")
    name = CASCADE_PARENT_FEATURES.index("name_evidence")
    return {
        "fit_2": _CountingMatrixScorer(name, 2.0),
        "fit_3": _CountingMatrixScorer(exact_address, 3.0),
        "fit_4": _CountingMatrixScorer(name, 4.0),
        "fit_23": _CountingMatrixScorer(exact_name, 5.0),
        "fit_24": _CountingMatrixScorer(exact_address, 6.0),
        "fit_34": _CountingMatrixScorer(exact_name, 7.0),
    }


def _singleton_for(scorer: _CountingMatrixScorer):
    def score(features: Mapping[str, float]) -> float:
        value = np.float32(features[CASCADE_PARENT_FEATURES[scorer.feature_index]])
        return float(value * scorer.multiplier)

    return score


def test_nested_route_table_is_exact_and_immutable() -> None:
    assert PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12 == {
        2: {2: "fit_34", 3: "fit_4", 4: "fit_3"},
        3: {2: "fit_4", 3: "fit_24", 4: "fit_2"},
        4: {2: "fit_3", 3: "fit_2", 4: "fit_23"},
    }
    with pytest.raises(TypeError):
        PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12[2][2] = "fit_2"  # type: ignore[index]


def test_scene_is_label_blind_immutable_and_refuses_closed_folds() -> None:
    sidecars = _CountingSidecars(())
    polluted = [
        {
            **row,
            "human_ground_truth_siret": "99999999999999",
            "ground_truth_siret": "99999999999999",
            "crm_alias": "PRIVATE",
        }
        for row in _rows()
    ]
    baseline = compile_label_blind_scene_v12(
        _rows(), query_id="q-fused", sidecars=sidecars, config=_config()
    )
    contaminated = compile_label_blind_scene_v12(
        polluted, query_id="q-fused", sidecars=sidecars, config=_config()
    )
    assert contaminated == baseline
    assert "99999999999999" not in repr(contaminated)
    assert "PRIVATE" not in repr(contaminated)
    with pytest.raises(FrozenInstanceError):
        contaminated.query_fold = 4  # type: ignore[misc]
    bad = [{**row, "fold": 1} for row in _rows()]
    with pytest.raises(ValueError, match="fold 0/1"):
        compile_label_blind_scene_v12(
            bad, query_id="q-fused", sidecars=sidecars, config=_config()
        )


def test_three_contexts_use_one_matrix_call_per_required_model() -> None:
    sidecars = _CountingSidecars(())
    scene = compile_label_blind_scene_v12(
        _rows(), query_id="q-fused", sidecars=sidecars, config=_config()
    )
    scorers = _scorers()
    contexts = score_three_outer_contexts_v12(
        scene, scorers=scorers, config=_config()
    )
    assert [(value.outer_fold, value.model_key) for value in contexts] == [
        (2, "fit_34"),
        (3, "fit_4"),
        (4, "fit_3"),
    ]
    for key in ("fit_34", "fit_4", "fit_3"):
        assert scorers[key].calls == 1
        assert scorers[key].shapes == [(3, len(CASCADE_PARENT_FEATURES))]
    for key in PARENT_MODEL_KEYS_V12 - {"fit_34", "fit_4", "fit_3"}:
        assert scorers[key].calls == 0


@pytest.mark.parametrize("query_fold", [2, 3, 4])
def test_fold_fused_results_match_three_legacy_cascades_and_share_lookup(
    query_fold: int,
) -> None:
    rows = [{**row, "fold": query_fold} for row in _rows()]
    sites = (
        _site("11111111100001"),
        _site("11111111100002"),
        _site("22222222200001"),
        _site("22222222200002"),
        _site("33333333300001"),
        _site("33333333300002"),
    )
    config = _config()
    fused_sidecars = _CountingSidecars(sites)
    scene = compile_label_blind_scene_v12(
        rows,
        query_id="q-fused",
        sidecars=fused_sidecars,
        config=config,
    )
    scorers = _scorers()
    prepared = prepare_fold_fused_scene_v12(
        scene,
        scorers=scorers,
        sidecars=fused_sidecars,
        config=config,
    )
    assert fused_sidecars.census_calls == 1
    assert fused_sidecars.expansion_calls == 1
    assert len(prepared.lookup_parent_sirens) >= 2

    selected_by_outer: dict[int, str] = {}
    for outer_fold in (2, 3, 4):
        model_key = PARENT_MODEL_BY_OUTER_AND_QUERY_FOLD_V12[outer_fold][
            query_fold
        ]
        legacy_sidecars = _CountingSidecars(sites)
        reference = RetrievalTop100CascadeV12(
            legacy_sidecars,
            config,
            parent_scorer=_singleton_for(scorers[model_key]),
        ).cascade(rows, query_id="q-fused")
        fused = materialize_outer_context_v12(
            prepared, outer_fold=outer_fold, config=config
        )
        assert_logical_result_parity_v12(
            reference, fused, query_id="q-fused"
        )
        assert legacy_sidecars.census_calls == 1
        assert legacy_sidecars.expansion_calls == 1
        selected_by_outer[outer_fold] = fused.parent_shortlist[0].siren

    # The union lookup may contain foreign-context parents, but the immutable
    # mask must prevent their expanded sites from leaking into another output.
    assert len(set(selected_by_outer.values())) >= 2
    for outer_fold, selected_parent in selected_by_outer.items():
        fused = materialize_outer_context_v12(
            prepared, outer_fold=outer_fold, config=config
        )
        expanded_children = {
            value.siren
            for value in fused.master_references
            if "targeted_parent_expansion" in value.lanes
        }
        assert expanded_children == {selected_parent}


def test_parent_score_ties_keep_input_rank_then_siren_tie_break() -> None:
    rows = [
        _row("33333333300001", ("name_word",), rank=2),
        _row("22222222200001", ("name_word",), rank=1),
        _row("11111111100001", ("name_word",), rank=1),
    ]
    sidecars = _CountingSidecars(())
    scene = compile_label_blind_scene_v12(
        rows, query_id="q-fused", sidecars=sidecars, config=_config()
    )
    zero = LinearParentMatrixScorerV12(
        tuple(0.0 for _ in CASCADE_PARENT_FEATURES)
    )
    contexts = score_three_outer_contexts_v12(
        scene,
        scorers={key: zero for key in PARENT_MODEL_KEYS_V12},
        config=_config(),
    )
    for context in contexts:
        assert [value.siren for value in context.parent_scores] == [
            "111111111",
            "222222222",
            "333333333",
        ]


def test_xgboost_batch_scores_are_bitwise_equal_to_singleton_adapter() -> None:
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(20260824)
    matrix = rng.random((64, len(CASCADE_PARENT_FEATURES)), dtype=np.float32)
    target = rng.random(64, dtype=np.float32)
    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": 3,
            "eta": 0.1,
            "nthread": 1,
            "seed": 20260824,
        },
        xgb.DMatrix(
            matrix,
            label=target,
            feature_names=list(CASCADE_PARENT_FEATURES),
        ),
        num_boost_round=8,
    )
    singleton = XGBoostParentScorerV12(booster)
    expected = np.asarray(
        [
            singleton.score(dict(zip(CASCADE_PARENT_FEATURES, row, strict=True)))
            for row in matrix
        ],
        dtype=np.float32,
    )
    observed = XGBoostParentMatrixScorerV12(booster).score_matrix(matrix)
    assert observed.dtype == expected.dtype
    assert np.array_equal(observed, expected)


def test_scorer_catalog_and_linear_feature_order_fail_closed() -> None:
    scorers = _scorers()
    validate_parent_scorer_catalog_v12(scorers)
    with pytest.raises(ValueError, match="six models"):
        validate_parent_scorer_catalog_v12({"fit_2": scorers["fit_2"]})
    with pytest.raises(ValueError, match="feature schema"):
        LinearParentMatrixScorerV12((1.0,))
