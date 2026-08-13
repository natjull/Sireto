from __future__ import annotations

import random

import pandas as pd

from scripts.train_v412_neural_groupwise_cross_encoder import _scene_batches


def test_scene_batches_never_split_a_scene() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["a"] * 16 + ["b"] * 16 + ["c"] * 16,
            "group_position": list(range(16)) * 3,
        }
    )

    batches = _scene_batches(frame, 2, random.Random(42))

    assert sorted(query for batch in batches for query in batch) == ["a", "b", "c"]
    assert all(len(batch) <= 2 for batch in batches)
