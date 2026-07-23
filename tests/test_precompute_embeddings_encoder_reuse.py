import numpy as np

from scripts.precompute_embeddings import encode_candidates


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(
        self,
        texts,
        batch_size,
        show_progress_bar,
        convert_to_numpy,
        normalize_embeddings,
    ):
        self.calls += 1
        assert show_progress_bar is False
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        assert batch_size == 32
        assert len(texts) == 2
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


def test_encode_candidates_reuses_provided_encoder() -> None:
    candidates = [
        {"name": "Alpha"},
        {"name": "Beta"},
    ]
    encoder = _FakeEncoder()

    embeddings = encode_candidates(candidates, encoder, batch_size=32)

    assert encoder.calls == 1
    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(embeddings, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
