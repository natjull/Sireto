import importlib
import os


def _reload_features(env: dict):
    original = {
        "XGB_SEMANTIC_GATE_ENABLED": os.getenv("XGB_SEMANTIC_GATE_ENABLED"),
        "XGB_SEMANTIC_GATE_JARO_MIN": os.getenv("XGB_SEMANTIC_GATE_JARO_MIN"),
        "XGB_SEMANTIC_GATE_TOKEN_MIN": os.getenv("XGB_SEMANTIC_GATE_TOKEN_MIN"),
    }
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)

    try:
        import src.xgb_matcher.features as features
        importlib.reload(features)
        return features
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_semantic_gate_blocks_low_lexical():
    features = _reload_features(
        {
            "XGB_SEMANTIC_GATE_ENABLED": "1",
            "XGB_SEMANTIC_GATE_JARO_MIN": "0.5",
            "XGB_SEMANTIC_GATE_TOKEN_MIN": "0.2",
        }
    )
    assert features.semantic_gate_allows(0.49, 0.19) is False


def test_semantic_gate_allows_if_one_signal_strong():
    features = _reload_features(
        {
            "XGB_SEMANTIC_GATE_ENABLED": "1",
            "XGB_SEMANTIC_GATE_JARO_MIN": "0.5",
            "XGB_SEMANTIC_GATE_TOKEN_MIN": "0.2",
        }
    )
    assert features.semantic_gate_allows(0.50, 0.10) is True
    assert features.semantic_gate_allows(0.10, 0.20) is True


def test_semantic_gate_disabled_allows_all():
    features = _reload_features(
        {
            "XGB_SEMANTIC_GATE_ENABLED": "0",
            "XGB_SEMANTIC_GATE_JARO_MIN": "0.5",
            "XGB_SEMANTIC_GATE_TOKEN_MIN": "0.2",
        }
    )
    assert features.semantic_gate_allows(0.0, 0.0) is True
