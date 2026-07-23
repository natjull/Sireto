import pandas as pd

from scripts.build_v9_siren_dense_index import iter_entity_batches, siren_text


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
