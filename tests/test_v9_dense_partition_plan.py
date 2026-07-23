from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.plan_v9_dense_partitions import build_partition_plan


def test_dense_partition_plan_prefers_exact_insee_then_postcode(
    tmp_path: Path,
) -> None:
    (tmp_path / "insee" / "insee=01001").mkdir(parents=True)
    (tmp_path / "cp" / "postcode=02000").mkdir(parents=True)
    benchmark = pd.DataFrame(
        [
            {
                "query_id": "a",
                "split": "dev",
                "insee": "01001",
                "postcode": "01000",
            },
            {
                "query_id": "b",
                "split": "dev",
                "insee": "99999",
                "postcode": "02000",
            },
            {
                "query_id": "c",
                "split": "test",
                "insee": "01001",
                "postcode": "01000",
            },
        ]
    )

    plan = build_partition_plan(
        benchmark,
        partitions_dir=tmp_path,
        splits={"dev"},
    )

    assert plan["insee_codes"] == ["01001"]
    assert plan["postcode_codes"] == ["02000"]
    assert plan["counts"]["missing_queries"] == 0
