"""Ollama client helpers for Pipe V6.

Provides a thin wrapper around the local Ollama HTTP API with retries,
timeouts, concurrency limiting, and optional JSON-mode enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
import time
from typing import Any, Dict, Iterable

import requests
from requests import Session
from requests.exceptions import RequestException, Timeout

from .config import PipelineConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponseMetrics:
    """Subset of telemetry fields returned by Ollama."""

    prompt_eval_count: int | None = None
    eval_count: int | None = None
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Structured result returned by :class:`OllamaClient`."""

    text: str
    model: str
    raw: Dict[str, Any]
    metrics: LLMResponseMetrics | None = None
    parsed_json: Any | None = None


class LLMCallError(RuntimeError):
    """Raised when the LLM call fails or returns malformed data."""


def _truncate_for_log(value: str, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}… (total {len(value)} chars)"


def _extract_json_object(text: str) -> Any | None:
    """Best-effort extraction of the first JSON object from arbitrary text."""

    if not text:
        return None
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None

def _ns_to_ms(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 1_000_000.0
    except (TypeError, ValueError):
        return None


class OllamaClient:
    """High-level helper to call Ollama's /api/generate endpoint."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        logger: logging.Logger | None = None,
        session: Session | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or LOGGER
        self._session = session or requests.Session()
        self._owns_session = session is None
        concurrency = max(1, int(getattr(config, "llm_max_concurrency", 1) or 1))
        self._semaphore = threading.Semaphore(concurrency)

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def call_text(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        num_predict: int | None = None,
        stop: Iterable[str] | None = None,
        seed: int | None = None,
        timeout: float | None = None,
        json_mode: bool | None = None,
    ) -> LLMResponse:
        """Call Ollama and return the raw text response."""

        response_payload = self._invoke(
            prompt=str(prompt),
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            stop=stop,
            seed=seed,
            timeout=timeout,
            json_mode=json_mode,
        )

        text = response_payload.get("response", "")
        model = response_payload.get("model", self._config.model_name)
        metrics = LLMResponseMetrics(
            prompt_eval_count=response_payload.get("prompt_eval_count"),
            eval_count=response_payload.get("eval_count"),
            total_duration_ms=_ns_to_ms(response_payload.get("total_duration")),
            load_duration_ms=_ns_to_ms(response_payload.get("load_duration")),
        )

        parsed_json: Any | None = None
        json_mode_active = json_mode if json_mode is not None else self._config.llm_json_mode_default
        if json_mode_active:
            try:
                parsed_json = json.loads(text)
            except json.JSONDecodeError:
                # Lenient fallback: strip code fences / leading text and try to extract first JSON object.
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    # remove optional language label
                    if cleaned.lower().startswith("json"):
                        cleaned = cleaned[4:].strip()
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = cleaned[start : end + 1]
                    try:
                        parsed_json = json.loads(candidate)
                    except json.JSONDecodeError as exc:
                        preview = text.strip().replace("\n", " ")[:200]
                        raise LLMCallError(f"LLM output is not valid JSON (preview: {preview})") from exc
                else:
                    preview = text.strip().replace("\n", " ")[:200]
                    raise LLMCallError(f"LLM output is not valid JSON (preview: {preview})")

        return LLMResponse(
            text=text,
            model=model,
            raw=response_payload,
            metrics=metrics,
            parsed_json=parsed_json,
        )

    def call_json(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Call the LLM and parse the first JSON object from the response text.

        We avoid Ollama's `format=json` here because some models (e.g., gpt-oss)
        return an empty `response` field in that mode. Instead, we request plain
        text and extract the first JSON object.
        """

        kwargs.setdefault("json_mode", False)

        def _try_once() -> LLMResponse:
            resp = self.call_text(prompt, **kwargs)
            parsed_inner = _extract_json_object(resp.text)
            if parsed_inner is not None:
                return LLMResponse(
                    text=resp.text,
                    model=resp.model,
                    raw=resp.raw,
                    metrics=resp.metrics,
                    parsed_json=parsed_inner,
                )
            return LLMResponse(
                text=resp.text,
                model=resp.model,
                raw=resp.raw,
                metrics=resp.metrics,
                parsed_json=None,
            )

        first = _try_once()
        if first.parsed_json is not None:
            return first

        # Retry once on empty/invalid JSON output (common with some Ollama models).
        self._logger.warning(
            "LLM output not valid JSON, retrying once (preview: %s)",
            first.text.strip().replace("\n", " ")[:120],
        )

        second = _try_once()
        if second.parsed_json is not None:
            return second

        preview = second.text.strip().replace("\n", " ")[:200]
        raise LLMCallError(f"LLM output is not valid JSON (preview: {preview})")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _invoke(
        self,
        *,
        prompt: str,
        temperature: float | None,
        top_p: float | None,
        num_predict: int | None,
        stop: Iterable[str] | None,
        seed: int | None,
        timeout: float | None,
        json_mode: bool | None,
    ) -> Dict[str, Any]:
        payload = self._build_payload(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            stop=stop,
            seed=seed,
            json_mode=json_mode,
        )

        base_url = (self._config.ollama_base_url or "http://localhost:11434").rstrip("/")
        url = f"{base_url}/api/generate"
        connect_timeout = float(getattr(self._config, "llm_connect_timeout_sec", 5.0) or 5.0)
        read_timeout = float(timeout if timeout is not None else getattr(self._config, "llm_timeout_sec", 120.0) or 120.0)
        max_retries = max(1, int(getattr(self._config, "llm_max_retries", 3) or 1))
        backoff_base = float(getattr(self._config, "llm_retry_backoff_base_sec", 1.0) or 1.0)

        self._log_prompt(prompt)

        with self._semaphore:
            last_error: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = self._session.post(
                        url,
                        json=payload,
                        timeout=(connect_timeout, read_timeout),
                    )
                except (Timeout, RequestException) as exc:
                    last_error = exc
                    self._logger.warning(
                        "LLM call network error (attempt %s/%s): %s",
                        attempt,
                        max_retries,
                        exc,
                    )
                    self._sleep_backoff(backoff_base, attempt, attempt == max_retries)
                    continue

                if 500 <= response.status_code:
                    last_error = LLMCallError(
                        f"LLM server error HTTP {response.status_code}: {response.text[:500]}"
                    )
                    self._logger.warning(
                        "LLM server error %s (attempt %s/%s)",
                        response.status_code,
                        attempt,
                        max_retries,
                    )
                    self._sleep_backoff(backoff_base, attempt, attempt == max_retries)
                    continue

                if response.status_code >= 400:
                    raise LLMCallError(
                        f"LLM request rejected with HTTP {response.status_code}: {response.text[:500]}"
                    )

                try:
                    payload_json: Dict[str, Any] = response.json()
                except ValueError as exc:
                    raise LLMCallError("Ollama response is not valid JSON") from exc

                self._log_response(payload_json.get("response", ""))
                return payload_json

        assert last_error is not None
        raise LLMCallError(f"LLM call failed after {max_retries} attempts: {last_error}")

    def _sleep_backoff(self, base: float, attempt: int, is_last_attempt: bool) -> None:
        if is_last_attempt:
            return
        delay = base * (2 ** (attempt - 1))
        time.sleep(min(delay, 10.0))

    def _build_payload(
        self,
        *,
        prompt: str,
        temperature: float | None,
        top_p: float | None,
        num_predict: int | None,
        stop: Iterable[str] | None,
        seed: int | None,
        json_mode: bool | None,
    ) -> Dict[str, Any]:
        config = self._config
        resolved_temperature = temperature if temperature is not None else config.temperature
        resolved_top_p = top_p if top_p is not None else config.top_p
        resolved_num_predict = num_predict if num_predict is not None else config.max_tokens
        resolved_seed = seed if seed is not None else config.llm_seed

        options: Dict[str, Any] = {
            "temperature": resolved_temperature,
            "top_p": resolved_top_p,
            "num_predict": resolved_num_predict,
        }
        if resolved_seed is not None:
            options["seed"] = resolved_seed

        # Remove None values to rely on Ollama defaults when not specified.
        options = {k: v for k, v in options.items() if v is not None}

        payload: Dict[str, Any] = {
            "model": config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        if stop:
            payload["stop"] = list(stop)

        json_mode_active = json_mode if json_mode is not None else config.llm_json_mode_default
        if json_mode_active:
            payload["format"] = "json"

        return payload

    def _log_prompt(self, prompt: str) -> None:
        if getattr(self._config, "llm_log_prompts", False):
            self._logger.info("LLM prompt (%s chars): %s", len(prompt), _truncate_for_log(prompt))
        else:
            self._logger.debug("LLM prompt length=%s chars", len(prompt))

    def _log_response(self, response_text: str) -> None:
        if getattr(self._config, "llm_log_responses", False):
            self._logger.info(
                "LLM response (%s chars): %s",
                len(response_text),
                _truncate_for_log(response_text),
            )
        else:
            self._logger.debug("LLM response length=%s chars", len(response_text))


__all__ = [
    "LLMCallError",
    "LLMResponse",
    "LLMResponseMetrics",
    "OllamaClient",
]
