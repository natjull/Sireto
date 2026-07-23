import pandas as pd

from scripts.build_v9_siren_dense_index import iter_entity_batches, siren_text
from src.xgb_matcher.dense_retrieval import GlobalDenseSirenIndex


def test_siren_text_deduplicates_names():
    text = siren_text(
        {
            "denominationUniteLegale": "ACME",
            "denominationUsuelle1UniteLegale": "ACME",
            "sigleUniteLegale": "A",
        }
    )
    assert text == "A | ACME"


def test_iter_entity_batches_filters_empty_names(tmp_path):
    source = tmp_path / "ul.parquet"
    pd.DataFrame(
        {
            "siren": ["123456789", "987654321"],
            "denominationUniteLegale": ["ACME", None],
        }
    ).to_parquet(source, index=False)

    batches = list(iter_entity_batches(source, batch_size=10))
    assert batches == [(["123456789"], ["ACME"])]


def test_global_dense_siren_rejects_model_fingerprint_before_loading_faiss(
    tmp_path,
):
    (tmp_path / "manifest.json").write_text(
        '{"semantic_model_fingerprint": "built-with-a"}',
        encoding="utf-8",
    )

    try:
        GlobalDenseSirenIndex(
            tmp_path,
            expected_model_fingerprint="running-with-b",
        )
    except ValueError as error:
        assert "fingerprint mismatch" in str(error)
    else:
        raise AssertionError("A mismatched global dense model must be rejected")
