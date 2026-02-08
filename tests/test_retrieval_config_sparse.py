from src.xgb_matcher.retrieval_config import RetrievalConfigV1


def test_sparse_retrieval_roundtrip_and_signature() -> None:
    base = RetrievalConfigV1()
    dense_only = RetrievalConfigV1(sparse_retrieval_enabled=False, dense_retrieval_enabled=True)

    payload = dense_only.to_dict()
    restored = RetrievalConfigV1.from_dict(payload)

    assert restored.sparse_retrieval_enabled is False
    assert restored.dense_retrieval_enabled is True
    assert base.signature().hash != dense_only.signature().hash
