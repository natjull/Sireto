from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_v412_bge_xgb_stack import (
    STACK_SIGNAL_FEATURES,
    _add_scene_signals,
)


def test_stack_scene_signals_capture_agreement_ranks_and_margins() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q2", "q2"],
            "candidate_siret": ["1", "2", "3", "4", "5"],
            "retrieval_rank": [1, 2, 3, 1, 2],
            "ranker_rank": [1, 2, 3, 1, 2],
            "ranker_score": [0.9, 0.8, 0.1, 0.8, 0.7],
            "bge_score": [0.2, 0.9, 0.1, 0.8, 0.3],
        }
    )

    output = _add_scene_signals(frame).set_index(["query_id", "candidate_siret"])

    assert output.loc[("q1", "2"), "bge_rank"] == 1
    assert output.loc[("q1", "1"), "business_bge_top1_agreement"] == 0
    assert output.loc[("q2", "4"), "business_bge_top1_agreement"] == 1
    assert np.isclose(output.loc[("q1", "2"), "bge_top1_top2_gap"], 0.7)
    assert np.isclose(output.loc[("q1", "2"), "business_top1_top2_gap"], 0.1)
    assert np.isfinite(output[STACK_SIGNAL_FEATURES].to_numpy()).all()
