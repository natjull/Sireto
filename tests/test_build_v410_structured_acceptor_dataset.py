from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_v410_structured_acceptor_dataset as subject
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy


TAXONOMY = Path("config/v4_9_site_function_taxonomy.json")


def _candidate(
    *,
    query_id: str = "q1",
    siret: str = "11111111100011",
    rank: int = 1,
    score: float = 0.9,
) -> dict:
    row = {
        "query_id": query_id,
        "candidate_siret": siret,
        "candidate_siren": siret[:9],
        "candidate_state": "A",
        "retrieval_rank": rank,
        "retrieval_source": "frozen",
        "retrieval_channel_count": 2,
        "retrieval_agreement": 1,
        "rank": rank,
        "ranker_score": score,
    }
    row.update({name: 0.0 for name in subject.V41_CANDIDATE_FEATURES})
    row.update(
        {
            "name_jaro_max": 0.9,
            "name_token_overlap_max": 0.8,
            "addr_jaro": 0.8,
            "street_name_jaro": 0.8,
            "street_number_match": 1.0,
            "postcode_match": 1.0,
            "city_match": 1.0,
            "candidate_is_active": 1.0,
        }
    )
    return row


def _registry(*sirets: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "siret": siret,
                "siren": siret[:9],
                "registry_state": "A",
                "enseigne1": "ECOLE ALPHA",
                "enseigne2": None,
                "enseigne3": None,
                "denomination_usuelle": None,
                "activity_code": "85.20Z",
                "registry_postcode": "75001",
                "registry_city": "PARIS",
                "registry_street_number": "1",
            }
            for siret in sirets
        ]
    )


def _scene(query_id: str = "q1", predicted: str | None = "11111111100011") -> dict:
    row = {
        "query_id": query_id,
        "predicted_siret": predicted,
        "ranker_prediction_is_out_of_sample": True,
        "prediction_origin": "oof",
        "ranker_oof_fold": 0,
        **{name: 0.0 for name in subject.CURRENT80_FEATURES},
    }
    top1 = _candidate(query_id=query_id)
    if predicted is not None:
        for base in subject.SEMANTIC_ALIAS_BASE_FEATURES:
            row[f"top1_{base}"] = float(top1[base])
            row[f"top2_{base}"] = 0.0
            row[f"delta_{base}"] = float(top1[base])
        row["score_top1"] = float(top1["ranker_score"])
        row["score_top2"] = 0.0
        row["score_gap"] = float(top1["ranker_score"])
        row["top1_siren_candidate_count"] = 1.0
    return row


def _query(query_id: str = "q1") -> dict:
    return {
        "query_id": query_id,
        "crm_name": "Ecole Alpha Paris",
        "crm_address": "1 rue Alpha",
        "crm_city": "Paris",
        "input_siret": "11111111100011",
        "input_siren": "111111111",
    }


def _scene_for_candidates(candidates: pd.DataFrame) -> dict:
    ordered = candidates.sort_values(["rank", "ranker_score"], ascending=[True, False])
    top1 = ordered.iloc[0]
    top2 = ordered.iloc[1] if len(ordered) > 1 else None
    scene = _scene(predicted=str(top1["candidate_siret"]))
    for base in subject.SEMANTIC_ALIAS_BASE_FEATURES:
        first = pd.to_numeric(top1[base], errors="coerce")
        first = 0.0 if pd.isna(first) else float(first)
        second = 0.0
        if top2 is not None:
            second_value = pd.to_numeric(top2[base], errors="coerce")
            second = 0.0 if pd.isna(second_value) else float(second_value)
        scene[f"top1_{base}"] = first
        scene[f"top2_{base}"] = second
        scene[f"delta_{base}"] = first - second
    first_score = float(top1["ranker_score"])
    second_score = 0.0 if top2 is None else float(top2["ranker_score"])
    scene["score_top1"] = first_score
    scene["score_top2"] = second_score
    scene["score_gap"] = first_score - second_score
    scene["top1_siren_candidate_count"] = float(
        ordered["candidate_siren"].astype(str).eq(str(top1["candidate_siren"])).sum()
    )
    return scene


def test_scene_contract_keeps_71_and_excludes_nine_semantic_features() -> None:
    assert len(subject.SCENE_FEATURES) == 71
    assert len(subject.SEMANTIC_SCENE_FEATURES) == 9
    assert not any("semantic" in name for name in subject.SCENE_FEATURES)


def test_historical_candidate_join_uses_exact_v41_pairs_and_keeps_metadata() -> None:
    candidates = pd.DataFrame([_candidate()])
    predictions = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "candidate_siret": "11111111100011",
                "score": 0.92,
                "rank": 1,
                "prediction_origin": "oof",
                "fold": 2,
            },
            {
                "query_id": "sentinel",
                "candidate_siret": None,
                "score": 0.0,
                "rank": 1,
                "prediction_origin": "oof",
                "fold": 1,
            },
        ]
    )
    output, report = subject.prepare_historical_candidates(
        predictions, candidates, enforce_contract_counts=False
    )
    assert len(output) == 1
    assert output.iloc[0]["candidate_siren"] == "111111111"
    assert output.iloc[0]["retrieval_channel_count"] == 2
    assert report["candidate_prediction_join_rate"] == 1.0
    assert report["no_candidate_sentinel_count"] == 1
    assert "is_ground_truth" not in output.columns


def test_historical_join_rejects_a_missing_candidate_pair() -> None:
    predictions = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "candidate_siret": "11111111100011",
                "score": 0.9,
                "rank": 1,
                "prediction_origin": "oof",
                "fold": 0,
            }
        ]
    )
    with pytest.raises(ValueError, match="must join exactly"):
        subject.prepare_historical_candidates(
            predictions,
            pd.DataFrame([_candidate(siret="99999999900099")]),
            enforce_contract_counts=False,
        )


def test_targeted_scan_physically_filters_random(tmp_path: Path) -> None:
    path = tmp_path / "hard.parquet"
    pd.DataFrame(
        [
            {"audit_case_id": "allowed", "sampling_stratum": "TARGETED", "value": 1},
            {
                "audit_case_id": "forbidden",
                "sampling_stratum": "RANDOM_POPULATION",
                "value": 2,
            },
        ]
    ).to_parquet(path, index=False)
    output = subject.read_targeted_parquet(path)
    assert output["audit_case_id"].tolist() == ["allowed"]


def test_structured_scene_handles_sentinel_without_truth_or_injection() -> None:
    output = subject.build_structured_scenes(
        pd.DataFrame([_scene("sentinel", None)]),
        pd.DataFrame([_candidate()]).iloc[0:0],
        pd.DataFrame([_query("sentinel")]),
        _registry("11111111100011").iloc[0:0],
        SiteFunctionTaxonomy.load(TAXONOMY),
        population="historical_v41",
    )
    assert len(output) == 1
    assert output.iloc[0]["top1_siret"] is None
    assert output.iloc[0]["candidate_top1_ranker_score_missing"] == 1.0
    assert "acceptor_target" not in output.columns


def test_structured_features_are_invariant_to_forbidden_target_columns() -> None:
    candidates = pd.DataFrame([_candidate()])
    base_scene = _scene()
    leaked_scene = {**base_scene, "ground_truth_siret": "999", "is_ground_truth": 1}
    kwargs = {
        "candidates": candidates,
        "queries": pd.DataFrame([_query()]),
        "registry": _registry("11111111100011"),
        "taxonomy": SiteFunctionTaxonomy.load(TAXONOMY),
        "population": "historical_v41",
    }
    first = subject.build_structured_scenes(pd.DataFrame([base_scene]), **kwargs)
    second = subject.build_structured_scenes(pd.DataFrame([leaked_scene]), **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_current80_is_preserved_exactly_and_oos_proof_is_metadata() -> None:
    scene = _scene()
    scene["score_top1"] = 0.731
    scene["top1_name_semantic_max"] = 0.0
    output = subject.build_structured_scenes(
        pd.DataFrame([scene]),
        pd.DataFrame([_candidate(score=0.731)]),
        pd.DataFrame([_query()]),
        _registry("11111111100011"),
        SiteFunctionTaxonomy.load(TAXONOMY),
        population="historical_v41",
    )
    assert output.iloc[0]["score_top1"] == 0.731
    assert output.iloc[0]["scene_score_top1"] == 0.731
    assert output.iloc[0]["ranker_prediction_is_out_of_sample"] == 1
    catalog = subject.make_feature_catalog(list(output.columns))
    assert catalog["current80_feature_order"] == list(subject.CURRENT80_FEATURES)
    assert "top1_name_semantic_max" not in catalog["structured_feature_order"]
    assert "ranker_prediction_is_out_of_sample" not in catalog["feature_order"]


def test_query_coverage_is_exact_and_missing_crm_never_becomes_zero() -> None:
    report = subject.validate_query_coverage(
        pd.DataFrame({"query_id": ["q1"]}),
        pd.DataFrame({"query_id": ["q1"]}),
        name="unit",
    )
    assert report["join_rate"] == 1.0
    with pytest.raises(ValueError, match="coverage differs"):
        subject.validate_query_coverage(
            pd.DataFrame({"query_id": ["q1"]}),
            pd.DataFrame({"query_id": ["q2"]}),
            name="unit",
        )
    with pytest.raises(ValueError, match="CRM query missing"):
        subject.build_structured_scenes(
            pd.DataFrame([_scene()]),
            pd.DataFrame([_candidate()]),
            pd.DataFrame([_query("other")]),
            _registry("11111111100011"),
            SiteFunctionTaxonomy.load(TAXONOMY),
            population="historical_v41",
        )


def test_registry_requires_unique_and_complete_top_two() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(),
            _candidate(siret="11111111100022", rank=2, score=0.8),
        ]
    )
    with pytest.raises(ValueError, match="below 100%"):
        subject.validate_registry_coverage(
            candidates, _registry("11111111100011"), name="unit"
        )
    duplicate = pd.concat(
        [_registry("11111111100011"), _registry("11111111100011")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="not unique"):
        subject.validate_registry_coverage(candidates.iloc[:1], duplicate, name="unit")
    inconsistent = _registry("11111111100011", "11111111100022")
    inconsistent.loc[1, "siren"] = "999999999"
    with pytest.raises(ValueError, match="coherence"):
        subject.validate_registry_coverage(candidates, inconsistent, name="unit")


def test_naf_mapping_and_pinned_one_hot_encoding() -> None:
    assert subject._naf_parts("85.20Z") == ("P", "85")
    assert subject._naf_parts("47.73Z") == ("G", "47")
    output = subject.build_structured_scenes(
        pd.DataFrame([_scene()]),
        pd.DataFrame([_candidate()]),
        pd.DataFrame([_query()]),
        _registry("11111111100011"),
        SiteFunctionTaxonomy.load(TAXONOMY),
        population="historical_v41",
    )
    row = output.iloc[0]
    assert row["naf_top1_section__P"] == 1.0
    assert row["naf_top1_division__85"] == 1.0
    assert row["role_top1__EDU_PRIMAIRE"] == 1.0
    assert row["same_siren_distinct_division_count"] == 1.0
    assert row["same_siren_distinct_role_count"] == 1.0


def test_constellation_ties_use_score_rank_then_lexical_siret() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(siret="11111111100022", rank=1, score=0.9),
            _candidate(siret="11111111100011", rank=2, score=0.9),
            _candidate(siret="11111111100033", rank=3, score=0.8),
        ]
    )
    candidates["name_jaro_max"] = [0.8, 0.9, 0.7]
    candidates["addr_jaro"] = [0.7, 0.6, 0.9]
    output = subject.build_structured_scenes(
        pd.DataFrame([_scene_for_candidates(candidates)]),
        candidates,
        pd.DataFrame([_query()]),
        _registry("11111111100022", "11111111100011", "11111111100033"),
        SiteFunctionTaxonomy.load(TAXONOMY),
        population="historical_v41",
    )
    assert output.iloc[0]["same_siren_name_geo_best_disagreement"] == 1.0


def test_catalog_marks_retrieval_drift_features_audit_only() -> None:
    output = subject.build_structured_scenes(
        pd.DataFrame([_scene()]),
        pd.DataFrame([_candidate()]),
        pd.DataFrame([_query()]),
        _registry("11111111100011"),
        SiteFunctionTaxonomy.load(TAXONOMY),
        population="historical_v41",
    )
    names = list(output.columns)
    catalog = subject.make_feature_catalog(names)
    audit = {item["name"] for item in catalog["audit_only_features"]}
    assert {
        "candidate_top1_admission_fusion_score",
        "candidate_top1_admission_fusion_score_missing",
        "candidate_delta_candidate_from_sparse",
    }.issubset(audit)
    assert len(audit) == 91
    assert "candidate_top1_name_jaro_max" not in catalog["feature_order"]
    assert catalog["current80_feature_order"] == list(subject.CURRENT80_FEATURES)
    assert len(catalog["features"]) == catalog["output_feature_count"]

    with pytest.raises(ValueError, match="no explicit specification"):
        subject.make_feature_catalog(
            [*names, "unrelated_feature"]
        )
    assert len(catalog["features"]) == catalog["output_feature_count"]
    by_name = {item["name"]: item for item in catalog["features"]}
    assert by_name["candidate_top1_admission_fusion_score"]["model_allowed"] is False
    assert by_name["candidate_top1_name_jaro_max"]["source_block"] == "candidate_ranker"
    assert by_name["candidate_top1_name_jaro_max"]["structured_allowed"] is False
    assert by_name["candidate_top1_name_jaro_max"]["alias_of"] == {
        "kind": "column",
        "operands": ["scene_top1_name_jaro_max"],
    }
    assert len(catalog["feature_order_sha256"]) == 64


def test_v410b_catalog_has_exact_orders_aliases_and_scaler_partition() -> None:
    previous_catalog = Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
        "v4_10_structured_acceptor/0d6b87fd50fb550c/feature_catalog.json"
    )
    if not previous_catalog.exists():
        pytest.skip("Canonical V4.10 catalog is not mounted")
    names = json.loads(previous_catalog.read_text())["output_feature_order"]
    catalog = subject.make_feature_catalog(names)

    assert catalog["current80_feature_order"] == list(subject.CURRENT80_FEATURES)
    assert catalog["current80_feature_order_sha256"] == (
        subject.EXPECTED_CURRENT80_FEATURE_ORDER_SHA256
    )
    assert catalog["feature_count"] == 641
    assert catalog["structured_feature_order_sha256"] == (
        subject.EXPECTED_STRUCTURED_FEATURE_ORDER_SHA256
    )
    assert catalog["semantic_alias_feature_count"] == 58
    assert catalog["retrieval_audit_only_feature_count"] == 16
    assert catalog["legacy_audit_only_feature_count"] == 75
    assert catalog["structured_scaled_feature_count"] == 157
    assert catalog["structured_scaled_feature_order_sha256"] == (
        subject.EXPECTED_STRUCTURED_SCALED_FEATURE_ORDER_SHA256
    )
    assert catalog["structured_unscaled_feature_count"] == 484
    assert catalog["structured_unscaled_feature_order_sha256"] == (
        subject.EXPECTED_STRUCTURED_UNSCALED_FEATURE_ORDER_SHA256
    )
    scaled = set(catalog["structured_scaled_feature_order"])
    unscaled = set(catalog["structured_unscaled_feature_order"])
    assert scaled.isdisjoint(unscaled)
    reconstructed = [
        name
        for name in catalog["structured_feature_order"]
        if name in scaled or name in unscaled
    ]
    assert reconstructed == catalog["structured_feature_order"]


def test_v410b_alias_schema_is_typed_and_divergence_stops() -> None:
    assert len(subject.SEMANTIC_ALIAS_MAP) == 58
    kinds = [
        representation["kind"]
        for representation in subject.SEMANTIC_ALIAS_MAP.values()
    ]
    assert kinds.count("column") == 56
    assert kinds.count("subtract") == 1
    assert kinds.count("literal") == 1

    output = subject.build_structured_scenes(
        pd.DataFrame([_scene()]),
        pd.DataFrame([_candidate()]),
        pd.DataFrame([_query()]),
        _registry("11111111100011"),
        SiteFunctionTaxonomy.load(TAXONOMY),
        population="historical_v41",
    )
    row = output.iloc[0].to_dict()
    row["candidate_top1_name_jaro_max"] += 0.01
    with pytest.raises(ValueError, match="semantic alias divergence"):
        subject.assert_semantic_aliases(row, query_id="q1")


def test_feature_matrix_must_be_numeric_finite_and_non_nullable() -> None:
    subject.validate_feature_matrix(
        pd.DataFrame({"a": [0.0, 1.0], "b": [2, 3]}),
        ["a", "b"],
        name="valid",
    )
    with pytest.raises(ValueError, match="nullable"):
        subject.validate_feature_matrix(
            pd.DataFrame({"a": [float("nan")]}), ["a"], name="nan"
        )
    with pytest.raises(ValueError, match="non-finite"):
        subject.validate_feature_matrix(
            pd.DataFrame({"a": [float("inf")]}), ["a"], name="inf"
        )
    with pytest.raises(ValueError, match="non-numeric"):
        subject.validate_feature_matrix(
            pd.DataFrame({"a": ["text"]}), ["a"], name="text"
        )


def test_candidate_nonfinite_uses_explicit_missing_but_scene_nonfinite_stops() -> None:
    candidate = _candidate()
    candidate["name_jaro_max"] = float("nan")
    output = subject.build_structured_scenes(
        pd.DataFrame([_scene_for_candidates(pd.DataFrame([candidate]))]),
        pd.DataFrame([candidate]),
        pd.DataFrame([_query()]),
        _registry("11111111100011"),
        SiteFunctionTaxonomy.load(TAXONOMY),
        population="historical_v41",
    )
    assert output.iloc[0]["candidate_top1_name_jaro_max"] == 0.0
    assert output.iloc[0]["candidate_top1_name_jaro_max_missing"] == 1.0

    infinite_candidate = _candidate()
    infinite_candidate["name_jaro_max"] = float("inf")
    with pytest.raises(ValueError, match="infinite candidate"):
        subject.build_structured_scenes(
            pd.DataFrame([_scene()]),
            pd.DataFrame([infinite_candidate]),
            pd.DataFrame([_query()]),
            _registry("11111111100011"),
            SiteFunctionTaxonomy.load(TAXONOMY),
            population="historical_v41",
        )

    bad_scene = _scene()
    bad_scene["score_top1"] = float("inf")
    with pytest.raises(ValueError, match="not finite"):
        subject.build_structured_scenes(
            pd.DataFrame([bad_scene]),
            pd.DataFrame([_candidate()]),
            pd.DataFrame([_query()]),
            _registry("11111111100011"),
            SiteFunctionTaxonomy.load(TAXONOMY),
            population="historical_v41",
        )


def test_population_audit_records_gate_counts_and_hashes() -> None:
    historical = pd.DataFrame(
        {
            "query_id": [f"h{index}" for index in range(7003)],
            "split": ["fit"] * 7003,
            "role": ["historical_hard_support"] * 20
            + ["historical_fit"] * 6983,
            "hard_fold": [index % 5 for index in range(7003)],
            "acceptor_target": [1] * 7003,
            "adjudication_label": ["MATCH_EXACT"] * 7003,
            "ranker_prediction_is_out_of_sample": [True] * 7003,
            "prediction_origin": ["oof"] * 7003,
            "hard_component_id": [
                f"cc{index}" if index < 20 else f"hc{index}"
                for index in range(7003)
            ],
        }
    )
    labels = ["TOP1_CORRECT"] * 68 + ["TOP1_WRONG"] * 25 + ["AMBIGUOUS"]
    consumed = pd.DataFrame(
        {
            "query_id": [f"c{index}" for index in range(94)],
            "split": [None] * 94,
            "role": ["hard_oof"] * 94,
            "hard_fold": [index % 5 for index in range(94)],
            "acceptor_target": [1] * 68 + [0] * 26,
            "adjudication_label": labels,
            "ranker_prediction_is_out_of_sample": [True] * 94,
            "prediction_origin": ["hard_oos"] * 94,
            "hard_component_id": [f"cc{index}" for index in range(94)],
        }
    )
    locked = consumed.iloc[:4].copy()
    locked["query_id"] = [f"l{index}" for index in range(4)]
    locked["role"] = "hard_dev_locked"
    audit = subject.dataset_population_audit(historical, consumed, locked)
    assert audit["development_consumed"]["label_counts"] == {
        "AMBIGUOUS": 1,
        "TOP1_CORRECT": 68,
        "TOP1_WRONG": 25,
    }
    assert len(audit["historical"]["query_ids_sha256"]) == 64


def test_source_path_guard_rejects_test_holdout_random_and_fresh() -> None:
    for value in ("test.parquet", "random/data.parquet", "fresh-data", "holdout.csv"):
        with pytest.raises(ValueError, match="forbidden population"):
            subject._assert_authorized_path(Path(value), name="source")


def test_population_audit_rejects_component_crossing_folds() -> None:
    historical = pd.DataFrame(
        [
            {
                "query_id": f"h{index}",
                "role": (
                    "historical_hard_support" if index < 20 else "historical_fit"
                ),
                "hard_component_id": f"c{index}" if index < 20 else f"x{index}",
                "hard_fold": index % 5 if index < 20 else pd.NA,
                "split": "fit",
                "acceptor_target": 1,
                "adjudication_label": "MATCH_EXACT",
                "ranker_prediction_is_out_of_sample": True,
                "prediction_origin": "oof",
            }
            for index in range(7003)
        ]
    )
    labels = ["TOP1_CORRECT"] * 68 + ["TOP1_WRONG"] * 25 + ["AMBIGUOUS"]
    consumed = pd.DataFrame(
        [
            {
                "query_id": f"q{index}",
                "role": "hard_oof",
                "hard_component_id": f"c{index % 20}",
                "hard_fold": index % 5,
                "split": None,
                "acceptor_target": int(label == "TOP1_CORRECT"),
                "adjudication_label": label,
                "ranker_prediction_is_out_of_sample": True,
                "prediction_origin": "frozen_v41_ranker_on_v42b_experimental",
            }
            for index, label in enumerate(labels)
        ]
    )
    locked = consumed.iloc[:4].copy()
    locked["role"] = "hard_dev_locked"
    subject.dataset_population_audit(historical, consumed, locked)
    consumed.loc[1, "hard_component_id"] = consumed.loc[0, "hard_component_id"]
    consumed.loc[1, "hard_fold"] = (int(consumed.loc[0, "hard_fold"]) + 1) % 5
    with pytest.raises(ValueError, match="crosses OOF folds"):
        subject.dataset_population_audit(historical, consumed, locked)


def test_stable_rows_hash_supports_nullable_integer_columns() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["a", "b"],
            "hard_fold": pd.Series([1, pd.NA], dtype="Int64"),
        }
    )
    first = subject._stable_rows_sha256(frame, ["query_id", "hard_fold"])
    second = subject._stable_rows_sha256(
        frame.iloc[::-1], ["query_id", "hard_fold"]
    )
    assert first == second
    assert len(first) == 64
