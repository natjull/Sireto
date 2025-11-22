"""Unit tests for the Ollama wrapper (task 3.1)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.config import PipelineConfig
from pipe_v6.llm_utils import LLMCallError, OllamaClient


class DummyResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None, text: str | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self) -> dict:
        if isinstance(self._payload, Exception):
            raise self._payload
        if self._payload is None:
            raise ValueError("No payload")
        return self._payload


class DummySession:
    def __init__(self, responses: list[DummyResponse]):
        self._responses = responses
        self.calls: list[dict] = []

    def post(
        self,
        url: str,
        json: dict | None = None,
        timeout: tuple[float, float] | None = None,
    ):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if not self._responses:
            raise RuntimeError("No more dummy responses")
        return self._responses.pop(0)

    def close(self) -> None:  # pragma: no cover - harmless for tests
        self._responses.clear()


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        crm_path=tmp_path / "crm.csv",
        output_path=tmp_path / "out.csv",
        sqlite_path=tmp_path / "cache.sqlite",
        log_path=tmp_path / "logs.log",
        sirene_api_url="https://sirene.example",
        sirene_token="token",
        rne_api_url="https://rne.example",
        rne_token_url="https://api.inpi.fr/api/sso/oauth2/token",
        rne_client_id="client",
        rne_client_secret="secret",
        qwant_base_url="https://qwant.example",
    )


def test_call_text_success(tmp_path, monkeypatch):
    monkeypatch.setattr("pipe_v6.llm_utils.time.sleep", lambda *_args, **_kwargs: None)
    config = _make_config(tmp_path)
    payload = {
        "model": config.model_name,
        "response": "Bonjour monde",
        "prompt_eval_count": 12,
        "eval_count": 24,
        "total_duration": 2_000_000,
        "load_duration": 1_000_000,
    }
    session = DummySession([DummyResponse(payload=payload)])
    client = OllamaClient(config, session=session)

    result = client.call_text("Salut")

    assert result.text == "Bonjour monde"
    assert result.metrics and result.metrics.prompt_eval_count == 12
    sent_payload = session.calls[0]["json"]
    assert sent_payload["model"] == config.model_name
    assert sent_payload["options"]["num_predict"] == config.max_tokens


def test_call_json_invalid_payload_raises(tmp_path):
    config = _make_config(tmp_path)
    payload = {
        "model": config.model_name,
        "response": "not-json",
    }
    session = DummySession([DummyResponse(payload=payload)])
    client = OllamaClient(config, session=session)

    with pytest.raises(LLMCallError):
        client.call_json("prompt")


def test_call_retries_on_server_error(tmp_path, monkeypatch):
    monkeypatch.setattr("pipe_v6.llm_utils.time.sleep", lambda *_args, **_kwargs: None)
    config = _make_config(tmp_path)
    bad = DummyResponse(status_code=500, payload={"response": ""}, text="boom")
    good_payload = {"model": config.model_name, "response": "ok"}
    good = DummyResponse(payload=good_payload)
    session = DummySession([bad, good])
    client = OllamaClient(config, session=session)

    result = client.call_text("hello")

    assert result.text == "ok"
    assert len(session.calls) == 2
