from pathlib import Path

import pandas as pd

from scripts.build_v412_corrected_review_overlay import _load_audit, _normalise_siret
from scripts.evaluate_v412_hard_label_ranker import load_hard_labels
from scripts.evaluate_v412_ranker_acceptor_stack import _load_adjudications


def test_normalise_siret_preserves_leading_zero() -> None:
    assert _normalise_siret("1234567890123") == "01234567890123"
    assert _normalise_siret(12345678901234) == "12345678901234"
    assert _normalise_siret("") is None


def test_load_audit_maps_exact_and_ambiguous(tmp_path: Path) -> None:
    source = tmp_path / "audit.csv"
    pd.DataFrame(
        [
            {
                "query_id": "1",
                "label": "MATCH_EXACT",
                "validated_siret": "12345678901234",
                "reliability": "HIGH",
                "legacy_label": "AMBIGUOUS",
                "pipeline_predicted_siret": "12345678901234",
                "pipeline_error": "FALSE_REVIEW",
            },
            {
                "query_id": "2",
                "label": "AMBIGUOUS",
                "validated_siret": "",
                "reliability": "MEDIUM",
                "legacy_label": "MATCH_EXACT",
                "pipeline_predicted_siret": "22345678901234",
                "pipeline_error": "CORRECT_REVIEW",
            },
        ]
    ).to_csv(source, index=False)

    result = _load_audit(source, "evidence.md")

    assert result["ground_truth_siret"].tolist() == ["12345678901234", None]
    assert result["evidence_reference"].tolist() == ["evidence.md", "evidence.md"]
    assert result["error_family"].tolist() == ["FALSE_REVIEW", "CORRECT_REVIEW"]


def test_corrected_overlay_extends_hard_labels() -> None:
    exact, all_ids, counts = load_hard_labels(
        Path("reports/v412_review_adjudication_labels.csv"),
        Path("reports/v412_review_rerank_counteraudit_53.csv"),
        Path("reports/v412_corrected_review_overlay_60.csv"),
    )

    assert len(exact) == 133
    assert len(all_ids) == 143
    assert counts["corrected_exact"] == 56
    assert counts["corrected_ambiguous"] == 4

    adjudications, adjudicated_ids = _load_adjudications(
        Path("reports/v412_review_adjudication_labels.csv"),
        Path("reports/v412_review_rerank_counteraudit_53.csv"),
        Path("reports/v412_corrected_review_overlay_60.csv"),
    )
    assert len(adjudications) == 143
    assert len(adjudicated_ids) == 143
    assert adjudications["label_kind"].value_counts().to_dict() == {
        "MATCH_EXACT": 133,
        "AMBIGUOUS": 10,
    }
