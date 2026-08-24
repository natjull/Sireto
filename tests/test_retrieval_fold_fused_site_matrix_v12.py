from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.xgb_matcher.retrieval_fold_fused_site_matrix_v12 import (
    compute_fold_fused_site_statistics_v12,
    materialize_fold_fused_site_matrix_v12,
)
from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    CASCADE_PARENT_FEATURES,
    canonicalize_union,
    site_feature_names,
)
from src.xgb_matcher.retrieval_v12_lazy_site_features import (
    compute_lazy_site_statistics,
)


def _random_master(seed: int, count: int = 257) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    config = AdmissionConfigV12.default()
    rows: list[dict[str, object]] = []
    signal_only_index = 37
    for index in range(count):
        parent_index = index // 3
        parent = f"{520_000_000 + parent_index:09d}"
        siret = f"{parent}{index % 3 + 1:05d}"
        chosen = [
            channel
            for channel in config.channels
            if rng.random() < 0.075 and channel != "bodacc_relation"
        ]
        # A channel listed without a rank must contribute to evidence/family
        # features, while its raw-channel ``present`` feature remains zero.
        if index == 11:
            chosen.append("rne_name")
        signals: list[dict[str, object]] = []
        if index == signal_only_index:
            signals.append(
                {
                    "channel": "bodacc_relation",
                    "rank": 2.0,
                    "score": -0.75,
                    "source": "BODACC",
                    "relation_direction": "PREDECESSOR",
                    "event_date": "2019-11-02",
                }
            )
        if index % 41 == 0:
            signals.append(
                {
                    "channel": "official_successor",
                    "rank": 1.0,
                    "score": 0.875,
                    "source": "SIRENE",
                    "relation_direction": "SUCCESSOR",
                    "relation_date": "2021-03-14",
                }
            )
        name = (
            "Établissement Démonstration"
            if index % 17 == 0
            else f"Concurrent aléatoire {index % 31}"
        )
        address = (
            "17 avenue des Tests 75001 Paris"
            if index % 19 == 0
            else f"{20 + index} rue secondaire 75001 Paris"
        )
        parent_values = [
            float(parent_index % (feature_index + 3)) / (feature_index + 1)
            for feature_index, _name in enumerate(CASCADE_PARENT_FEATURES)
        ]
        row: dict[str, object] = {
            "query_id": f"q-random-{seed}",
            "fold": 2 + seed % 3,
            "master_rank": index + 1,
            "hydration_rank": index + 1 if index < 149 else np.nan,
            "siret": siret,
            "siren": parent,
            "crm_name": "Établissement Démonstration",
            "crm_address": "17 avenue des Tests 75001 Paris",
            "crm_number": "17",
            "crm_postcode": "75001",
            "crm_insee": "75056",
            "candidate_names_json": json.dumps([name, f"Nom historique {index % 7}"]),
            "candidate_addresses": [address],
            "candidate_number": "17" if index % 19 == 0 else str(20 + index),
            "candidate_postcode": "75001" if index % 13 else "75002",
            "candidate_insee": "75056" if index % 23 else "75101",
            "candidate_state": "F" if index % 8 == 0 else "A",
            "candidate_is_headquarters": index % 3 == 0,
            "channels": sorted(set(chosen)),
            "sources": ["SIRENE_CURRENT:fixture"],
            "official_evidence_sources": [
                "SIRENE",
                *(["RNE"] if index % 13 == 0 else []),
            ],
            "signals": signals,
            "temporal_valid_from": "2012-06-01" if index % 2 else "",
            "temporal_valid_to": "2020-04-03" if index % 8 == 0 else "",
            "parent_rank": parent_index + 1,
            "parent_score": float((parent_index % 11) - 9.25),
            "parent_feature_names": list(CASCADE_PARENT_FEATURES),
            "parent_feature_values": parent_values,
            "official_catalog_present": True,
        }
        for position, channel in enumerate(chosen):
            # Preserve the rank-less RNE evidence case above.
            if index == 11 and channel == "rne_name":
                continue
            row[f"{channel}_rank"] = float(index % (position + 5) + 1)
            row[f"{channel}_score"] = float((index % 9) - 4) / 3.0
        rows.append(row)
    order = rng.permutation(len(rows))
    return [rows[int(index)] for index in order]


def _canonical_bits(
    rows: list[dict[str, object]], config: AdmissionConfigV12
) -> tuple[pd.DataFrame, np.ndarray]:
    canonical = canonicalize_union(
        pd.DataFrame(rows), config, labels_allowed=False
    )
    features = list(site_feature_names(config))
    matrix = np.ascontiguousarray(
        canonical[features].fillna(0.0).to_numpy(dtype=np.float32),
        dtype=np.float32,
    )
    return canonical, matrix.view(np.uint32)


@pytest.mark.parametrize("seed", [3, 17, 91])
def test_random_fold_fused_matrix_is_bitwise_canonical(seed: int) -> None:
    config = AdmissionConfigV12.default()
    rows = _random_master(seed)
    canonical, expected_bits = _canonical_bits(rows, config)
    statistics = compute_fold_fused_site_statistics_v12(rows, config)
    matrix = materialize_fold_fused_site_matrix_v12(statistics, config)

    assert tuple(statistics._base["candidate_siret"].astype(str)) == tuple(
        canonical["candidate_siret"].astype(str)
    )
    assert matrix.shape == (len(rows), 319)
    assert np.array_equal(matrix.view(np.uint32), expected_bits)


def test_signal_only_channel_and_fast_statistics_match_reference() -> None:
    config = AdmissionConfigV12.default()
    rows = _random_master(41)
    frame = pd.DataFrame(rows)
    reference = compute_lazy_site_statistics(frame, config)
    fast = compute_fold_fused_site_statistics_v12(rows, config)

    identity = ["query_id", "candidate_siret", "candidate_parent", "union_rank"]
    pd.testing.assert_frame_equal(
        fast._base[identity], reference._base[identity], check_exact=True
    )
    for channel in config.channels:
        for suffix in ("present",):
            column = f"channel__{channel}__{suffix}"
            np.testing.assert_array_equal(fast._base[column], reference._base[column])
        for prefix in ("__raw_rank__", "__raw_score__"):
            column = f"{prefix}{channel}"
            np.testing.assert_array_equal(fast._base[column], reference._base[column])

    signal_row = next(
        row
        for row in rows
        if any(
            signal.get("channel") == "bodacc_relation"
            for signal in row.get("signals", [])
        )
    )
    signal_siret = str(signal_row["siret"])
    observed = fast._base.loc[
        fast._base["candidate_siret"].eq(signal_siret)
    ].iloc[0]
    assert observed["channel__bodacc_relation__present"] == 1
    assert observed["__raw_rank__bodacc_relation"] == 2.0
    assert observed["__raw_score__bodacc_relation"] == -0.75
