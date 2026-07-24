import pandas as pd

from scripts.audit_benchmark_site_labels import (
    _candidate_hash,
    classify_site_label,
)


def test_closed_label_with_unique_active_exact_sibling_is_flagged() -> None:
    assert (
        classify_site_label(
            crm_address_usable=True,
            gt_state="F",
            date_reference_present=False,
            exact_sibling_count=1,
            active_exact_sibling_count=1,
        )
        == "CLOSED_GT_UNIQUE_ACTIVE_EXACT_SIBLING"
    )


def test_multiple_active_exact_siblings_are_ambiguous() -> None:
    assert (
        classify_site_label(
            crm_address_usable=True,
            gt_state="A",
            date_reference_present=False,
            exact_sibling_count=3,
            active_exact_sibling_count=2,
        )
        == "MULTIPLE_ACTIVE_EXACT_SIBLINGS"
    )


def test_reference_date_prevents_current_snapshot_reinterpretation() -> None:
    assert (
        classify_site_label(
            crm_address_usable=True,
            gt_state="F",
            date_reference_present=True,
            exact_sibling_count=1,
            active_exact_sibling_count=1,
        )
        == "HISTORICAL_REFERENCE_DATE_PRESENT"
    )


def test_candidate_hash_ignores_missing_street_type() -> None:
    row = pd.Series(
        {
            "numeroVoie": "12",
            "typeVoie": float("nan"),
            "libelleVoie": "Rue des Lilas",
        }
    )
    assert _candidate_hash(row) == "12|DES LILAS"
