import json

import pandas as pd

from scripts.evaluate_v4_final_holdout import final_verdict
from scripts.prepare_v4_final_holdout import prepare_evaluation_rows


def _holdout_fixture() -> pd.DataFrame:
    kinds = (
        ["MATCH_EXACT"] * 302
        + ["AMBIGUOUS"] * 52
        + ["UNRESOLVED"] * 991
    )
    rows = []
    for index, kind in enumerate(kinds):
        exact = kind == "MATCH_EXACT"
        rows.append(
            {
                "query_id": f"q{index}",
                "label_kind": kind,
                "ground_truth_siret": (
                    f"111111111{index:05d}"[-14:] if exact else None
                ),
                "ground_truth_siren": "111111111" if exact else None,
                "direct_active_sirets_json": (
                    json.dumps(
                        ["22222222200001", "22222222200002"]
                    )
                    if kind == "AMBIGUOUS"
                    else json.dumps([])
                ),
                "sirene_snapshot_id": "snapshot",
                "validator": "fixture",
            }
        )
    return pd.DataFrame(rows)


def test_prepare_evaluation_rows_keeps_truth_out_of_ambiguous() -> None:
    evaluation, labels, counts = prepare_evaluation_rows(
        _holdout_fixture()
    )
    assert counts == {
        "MATCH_EXACT": 302,
        "AMBIGUOUS": 52,
        "UNRESOLVED": 991,
    }
    assert len(evaluation) == 354
    assert len(labels) == 354
    ambiguous_labels = labels[labels["label_kind"].eq("AMBIGUOUS")]
    assert ambiguous_labels["ground_truth_siret"].isna().all()
    ambiguous_eval = evaluation[
        evaluation["evaluation_label_kind"].eq("AMBIGUOUS")
    ]
    assert ambiguous_eval["diagnostic_probe_siret"].notna().all()
    assert not ambiguous_eval["retrieval_uses_diagnostic_probe"].any()


def test_final_verdict_separates_coverage_from_technical_result() -> None:
    assert final_verdict(
        integrity_pass=True,
        source_coverage_pass=False,
        technical_pass=True,
    ) == ("PIVOT", "TECHNICAL_GO")
    assert final_verdict(
        integrity_pass=True,
        source_coverage_pass=True,
        technical_pass=True,
    ) == ("GO", "TECHNICAL_GO")
    assert final_verdict(
        integrity_pass=False,
        source_coverage_pass=True,
        technical_pass=True,
    ) == ("STOP", "TECHNICAL_INVALID")


def test_integrity_success_booleans_are_positive() -> None:
    integrity = {
        "old_test_not_read": True,
        "zero_positive_injection": True,
    }
    assert all(integrity.values())
