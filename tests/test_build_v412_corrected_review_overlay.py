from pathlib import Path

import pandas as pd

from scripts.build_v412_corrected_review_overlay import _load_audit, _normalise_siret


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
