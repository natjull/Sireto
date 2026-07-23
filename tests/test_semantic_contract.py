from pathlib import Path

import pytest

from src.xgb_matcher import features
from src.xgb_matcher.semantic import (
    assert_tokenizer_healthy,
    tokenizer_unknown_fraction,
)


class _Tokenizer:
    unk_token = "<unk>"
    all_special_tokens = ["<pad>", "<unk>"]

    def __init__(self, tokens):
        self._tokens = tokens

    def tokenize(self, _text):
        return list(self._tokens)


def test_tokenizer_healthcheck_rejects_unknown_heavy_tokenizer():
    tokenizer = _Tokenizer(["<unk>", "<unk>", "rue"])
    assert tokenizer_unknown_fraction(tokenizer, ["texte"]) == pytest.approx(2 / 3)
    with pytest.raises(RuntimeError, match="healthcheck failed"):
        assert_tokenizer_healthy(tokenizer, ["texte"], max_unknown_fraction=0.2)


def test_exported_fast_tokenizer_handles_french_text():
    tokenizer_path = Path("models/semantic/siret-bert-deploy/tokenizer.json")
    if not tokenizer_path.exists():
        pytest.skip("Local semantic export not available")

    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    texts = [
        "École Saint-Joseph",
        "12 rue de l'Église",
        "Société française de maintenance industrielle",
    ]
    tokens = [
        token
        for text in texts
        for token in tokenizer.encode(text).tokens
        if token not in {"<s>", "</s>", "<pad>", "<mask>"}
    ]
    fraction = sum(token == "<unk>" for token in tokens) / len(tokens)
    assert fraction <= 0.2


def test_semantic_batch_injection_applies_shared_gate(monkeypatch):
    monkeypatch.setattr(
        "src.xgb_matcher.semantic.top2_semantic_similarities_batch",
        lambda _crm, pools: [(0.9, 0.4, 0.5) for _ in pools],
    )
    monkeypatch.setattr(features, "SEMANTIC_GATE_ENABLED", True)
    rows = [
        {"name_jaro_max": 0.8, "name_token_overlap_max": 0.5},
        {"name_jaro_max": 0.1, "name_token_overlap_max": 0.0},
    ]

    features.inject_semantic_features_batch("École", rows, [["ECOLE"], ["GARAGE"]])

    assert rows[0]["name_semantic_max"] == pytest.approx(0.9)
    assert rows[0]["name_semantic_gap"] == pytest.approx(0.5)
    assert rows[1]["name_semantic_max"] == 0.0
    assert rows[1]["name_semantic_second"] == 0.0
    assert rows[1]["name_semantic_gap"] == 0.0
