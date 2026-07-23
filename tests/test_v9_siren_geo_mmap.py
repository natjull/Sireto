from __future__ import annotations

import json

import numpy as np

from src.xgb_matcher.siren_retrieval import SirenToGeoIndex


def test_siren_geo_mmap_binary_lookup_preserves_count_order(tmp_path) -> None:
    np.save(
        tmp_path / "sirens.npy",
        np.asarray(["000000001", "000000001", "000000003"], dtype="S9"),
    )
    np.save(
        tmp_path / "insee.npy",
        np.asarray(["75056", "92050", "13055"], dtype="S5"),
    )
    np.save(
        tmp_path / "postcodes.npy",
        np.asarray(["75001", "92000", "13001"], dtype="S10"),
    )
    np.save(
        tmp_path / "siret_counts.npy",
        np.asarray([4, 1, 2], dtype=np.int32),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v9-siren-geo-mmap-1",
                "row_count": 3,
            }
        ),
        encoding="utf-8",
    )

    index = SirenToGeoIndex(tmp_path)

    assert index.get_locations("1") == [
        ("75056", "75001"),
        ("92050", "92000"),
    ]
    assert index.get_location_counts("000000001") == [
        ("75056", "75001", 4),
        ("92050", "92000", 1),
    ]
    assert index.get_locations("000000002") == []
