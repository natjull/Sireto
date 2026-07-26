import json

import pandas as pd

from scripts.build_downstream_selective_dataset import selective_provenance
from src.xgb_matcher.v9_features import SELECTIVE_RETRIEVAL_CHANNELS


def _channel_row(values):
    payload = {}
    for channel in SELECTIVE_RETRIEVAL_CHANNELS:
        payload[f"{channel}_sirets_json"] = json.dumps(values.get(channel, []))
    return pd.Series(payload)


def test_selective_provenance_reconstructs_frozen_fusion_and_overlay():
    v7 = _channel_row(
        {
            "current_sparse": ["11111111100011", "22222222200022"],
            "name_word": ["11111111100011"],
            "name_char": ["22222222200022"],
        }
    )
    overlay = _channel_row(
        {
            "name_word": ["33333333300033"],
            "name_char": ["44444444400044"],
        }
    )

    result = selective_provenance(
        [
            "33333333300033",
            "44444444400044",
            "11111111100011",
            "22222222200022",
        ],
        v7,
        overlay,
    )

    assert result["33333333300033"]["admission_overlay_quota"] == 1.0
    assert result["11111111100011"]["admission_fusion_score"] == 3.0
    assert result["22222222200022"]["admission_fusion_score"] == 2.0
    assert result["11111111100011"]["admission_rank_recip"] == 1 / 3
    assert result["11111111100011"]["admission_channel_count"] == 2.0
