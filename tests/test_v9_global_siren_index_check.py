from __future__ import annotations

import pandas as pd

from scripts.check_v9_global_siren_index import sample_entities


def test_sample_entities_spans_parquet_row_groups(tmp_path) -> None:
    source = tmp_path / "ul.parquet"
    frame = pd.DataFrame(
        {
            "siren": [f"{index:09d}" for index in range(12)],
            "denominationUniteLegale": [
                f"ENTITY {index}" for index in range(12)
            ],
        }
    )
    frame.to_parquet(source, index=False, row_group_size=3)

    samples = sample_entities(source, sample_count=4, seed=42)

    assert len(samples) == 4
    assert len({siren for siren, _ in samples}) == 4
    assert all(text.startswith("ENTITY") for _, text in samples)
