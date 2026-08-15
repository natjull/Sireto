from scripts.build_synthetic_gt_oof_failure_catalog import failure_cell, scene_archetypes


def test_failure_cells_are_directional() -> None:
    assert failure_cell(False, False) == "BOTH_WRONG"
    assert failure_cell(True, False) == "BGE_ONLY_CORRECT"
    assert failure_cell(False, True) == "XGB_ONLY_CORRECT"
    assert failure_cell(True, True) == "BOTH_CORRECT"


def test_scene_archetypes_capture_matching_failure_shapes() -> None:
    tags = scene_archetypes({
        "top1_name_jaro_max": 0.4,
        "top1_addr_jaro": 0.98,
        "top1_top2_score_gap": 0.02,
        "top1_same_siren_count": 4,
        "top1_same_address_siren_count": 3,
        "distinct_siren_count": 80,
        "top1_business_role_conflict": 1,
        "ground_truth_state": "F",
    })
    assert {
        "WEAK_NAME_STRONG_ADDRESS", "LOW_XGB_MARGIN", "SAME_SIREN_COMPETITION",
        "SAME_ADDRESS_COMPETITION", "DENSE_CANDIDATE_SCENE",
        "BUSINESS_ROLE_CONFLICT", "CLOSED_TARGET",
    }.issubset(tags)
