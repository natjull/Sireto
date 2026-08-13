from __future__ import annotations

import mlx.core as mx

from scripts.train_v412_neural_groupwise_qwen import _group_loss


class _Embedding:
    weight = mx.array([[0.0], [1.0], [-1.0]])


class _Inner:
    embed_tokens = _Embedding()

    def __call__(self, inputs: mx.array) -> mx.array:
        return inputs.astype(mx.float32)[..., None]


class _Model:
    model = _Inner()


def test_group_loss_prefers_positive_score() -> None:
    # true_id=1 and false_id=2 makes the score equal to 2 * final token.
    good = _group_loss(_Model(), mx.array([[2], [0]]), mx.array([1, 1]), 1, 2, 1, 2)
    bad = _group_loss(_Model(), mx.array([[0], [2]]), mx.array([1, 1]), 1, 2, 1, 2)

    assert float(good.item()) < float(bad.item())
