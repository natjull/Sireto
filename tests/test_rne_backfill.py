from __future__ import annotations

import json
from pathlib import Path

from src.xgb_matcher.official_source_sync import canonical_json
from src.xgb_matcher.rne_backfill import (
    RneBackfillConfig,
    plan_rne_backfill,
    run_rne_backfill,
)


def _config() -> RneBackfillConfig:
    return RneBackfillConfig.from_dict(
        {
            "start_exclusive": "2026-08-01",
            "end_inclusive": "2026-08-06",
            "partition_days": 2,
            "sync": {
                "keychain": {"service": "fixture", "account": "fixture"},
                "api": {
                    "from": "2026-08-01",
                    "to": "2026-08-02",
                    "page_size": 100,
                },
            },
        }
    )


def test_plan_is_contiguous_and_closes_exact_end():
    assert [item.partition_id for item in plan_rne_backfill(_config())] == [
        "2026-08-01__2026-08-03",
        "2026-08-03__2026-08-05",
        "2026-08-05__2026-08-06",
    ]


def test_backfill_receipt_is_resumable_and_checks_manifests(tmp_path: Path):
    calls: list[str] = []

    def fake_sync(*, config, output_root):
        partition = f"{config.api.from_date}__{config.api.to_date}"
        calls.append(partition)
        output = output_root / "rne" / partition
        output.mkdir(parents=True)
        manifest = {
            "build_id": partition,
            "provenance": {
                "from_exclusive": config.api.from_date,
                "to_inclusive": config.api.to_date,
                "records": 3,
            },
        }
        (output / "manifest.json").write_bytes(canonical_json(manifest))
        return output

    receipt = tmp_path / "receipt.json"
    kwargs = {
        "output_root": tmp_path / "store",
        "receipt_path": receipt,
        "sync_function": fake_sync,
    }
    run_rne_backfill(_config(), **kwargs)
    run_rne_backfill(_config(), **kwargs)
    assert calls == [
        "2026-08-01__2026-08-03",
        "2026-08-03__2026-08-05",
        "2026-08-05__2026-08-06",
    ]
    value = json.loads(receipt.read_text())
    assert value["complete"] is True
    assert value["completed_partition_count"] == 3
    assert sum(item["records"] for item in value["partitions"]) == 9
