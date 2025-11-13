"""LLM-based normalization helpers for CRM entries (Task 3.2).

This module defines the JSON contract expected from the Ollama normalizer
model and provides strict parsing/validation utilities that downstream steps
can rely on.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import unicodedata
from typing import Mapping

from .config import PipelineConfig
from .llm_utils import LLMCallError, LLMResponse, OllamaClient


LOGGER = logging.getLogger(__name__)

CRM_CATEGORY_VALUES = ("PUBLIC", "PRIVE", "EQUIPEMENT_URBAIN", "INCONNU")

_NAME_ALLOWED_RE = re.compile(r"^[A-Z0-9 '&\-/]+$")
_ADDRESS_ALLOWED_RE = re.compile(r"^[A-Z0-9 '()\-/]+$")
_ADDRESS_RE = re.compile(
    r"^(?P<number>[0-9]{1,5}[A-Z0-9]{0,2})\s+(?P<street>[A-Z0-9' \-]+?)\s+(?P<postcode>\d{5})$"
)

_LEGAL_FORM_TOKENS = {
    "SAS",
    "SASU",
    "SARL",
    "SARLU",
    "SA",
    "SCOP",
    "SCI",
    "SCEA",
    "SNC",
    "SCM",
    "EARL",
    "EURL",
    "SELARL",
    "SELAS",
    "GIE",
    "ASSOCIATION",
    "ENTREPRISE",
    "SOCIETE",
}

_OPERATIONAL_TOKENS = {
    "AGENCE",
    "SITE",
    "BUREAU",
    "ANTENNE",
    "DELEGATION",
    "DIRECTION",
    "SERVICE",
}


class NormalizationParseError(ValueError):
    """Raised when the LLM output cannot be coerced into a valid structure."""


@dataclass(frozen=True)
class NormalizedCRMEntry:
    """Structured result of the LLM normalization step."""

    normalized_name: str
    normalized_address: str
    category: str

    def to_dict(self) -> dict[str, str]:
        return {
            "normalized_name": self.normalized_name,
            "normalized_address": self.normalized_address,
            "category": self.category,
        }


def normalize_with_llm(
    prompt: str,
    config: PipelineConfig,
    *,
    logger: logging.Logger | None = None,
    client: OllamaClient | None = None,
    expected_city: str | None = None,
) -> NormalizedCRMEntry:
    """Call the Ollama normalizer (JSON mode) and return a validated entry."""

    log = logger or LOGGER
    owns_client = False
    if client is None:
        client = OllamaClient(config, logger=log)
        owns_client = True

    try:
        response = client.call_json(prompt)
    finally:
        if owns_client:
            client.close()

    if response.parsed_json is None:
        raise LLMCallError("LLM response missing JSON payload")

    return parse_normalizer_output(response.parsed_json, expected_city=expected_city, logger=log)


def parse_normalizer_output(
    payload: Mapping[str, object],
    *,
    expected_city: str | None = None,
    logger: logging.Logger | None = None,
) -> NormalizedCRMEntry:
    """Validate and normalize the JSON payload produced by the LLM."""

    log = logger or LOGGER

    if not isinstance(payload, Mapping):
        raise NormalizationParseError("LLM payload must be a JSON object")

    try:
        raw_name = _ensure_str(payload["normalized_name"], "normalized_name")
        raw_address = _ensure_str(payload["normalized_address"], "normalized_address")
        raw_category = _ensure_str(payload["category"], "category")
    except KeyError as exc:  # pragma: no cover - trivial
        raise NormalizationParseError(f"Missing required field: {exc}") from None

    normalized_name = _prepare_name(raw_name, expected_city=expected_city)
    normalized_address = _prepare_address(raw_address)
    category = _prepare_category(raw_category)

    log.debug(
        "LLM normalization parsed successfully: %s",
        json.dumps(
            {
                "normalized_name": normalized_name,
                "normalized_address": normalized_address,
                "category": category,
            }
        ),
    )

    return NormalizedCRMEntry(
        normalized_name=normalized_name,
        normalized_address=normalized_address,
        category=category,
    )


def _ensure_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise NormalizationParseError(f"Field '{field}' must be a string")
    if not value.strip():
        raise NormalizationParseError(f"Field '{field}' cannot be empty")
    return value


def _ensure_uppercase(value: str, field: str) -> None:
    if value != value.upper():
        raise NormalizationParseError(f"Field '{field}' must be uppercase")


def _transliterate(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _filter_tokens(tokens: list[str], forbidden: set[str]) -> list[str]:
    return [token for token in tokens if token not in forbidden]


def _prepare_name(value: str, *, expected_city: str | None = None) -> str:
    _ensure_uppercase(value, "normalized_name")
    ascii_value = _transliterate(value)
    ascii_value = _collapse_spaces(ascii_value)

    tokens = ascii_value.split(" ")
    tokens = _filter_tokens(tokens, _LEGAL_FORM_TOKENS)
    tokens = _filter_tokens(tokens, _OPERATIONAL_TOKENS)

    if expected_city:
        city_ascii = _collapse_spaces(_transliterate(expected_city).upper())
        city_tokens = city_ascii.split(" ")
        tokens = [token for token in tokens if token not in city_tokens]

    if not tokens:
        raise NormalizationParseError("normalized_name became empty after cleaning")

    cleaned = " ".join(tokens)

    if not _NAME_ALLOWED_RE.fullmatch(cleaned):
        raise NormalizationParseError("normalized_name contains invalid characters")

    return cleaned


def _prepare_address(value: str) -> str:
    _ensure_uppercase(value, "normalized_address")
    ascii_value = _transliterate(value)
    ascii_value = _collapse_spaces(ascii_value)

    if not _ADDRESS_ALLOWED_RE.fullmatch(ascii_value):
        raise NormalizationParseError("normalized_address contains invalid characters")

    match = _ADDRESS_RE.match(ascii_value)
    if not match:
        raise NormalizationParseError(
            "normalized_address must follow '<numero> <voie> <code postal>'"
        )

    number = match.group("number")
    street = match.group("street").strip()
    postcode = match.group("postcode")

    if not street:
        raise NormalizationParseError("normalized_address street component is empty")

    return f"{number} {street} {postcode}"


def _prepare_category(value: str) -> str:
    upper_value = value.strip().upper()
    if upper_value not in CRM_CATEGORY_VALUES:
        raise NormalizationParseError(
            "category must be one of PUBLIC, PRIVE, EQUIPEMENT_URBAIN, INCONNU"
        )
    return upper_value


def parse_llm_response(
    response: LLMResponse,
    *,
    expected_city: str | None = None,
    logger: logging.Logger | None = None,
) -> NormalizedCRMEntry:
    """Helper that parses an :class:`LLMResponse` using this module's rules."""

    if response.parsed_json is None:
        raise NormalizationParseError("LLMResponse missing parsed_json payload")

    if not isinstance(response.parsed_json, Mapping):
        raise NormalizationParseError("LLMResponse parsed_json must be an object")

    return parse_normalizer_output(response.parsed_json, expected_city=expected_city, logger=logger)


__all__ = [
    "CRM_CATEGORY_VALUES",
    "NormalizedCRMEntry",
    "NormalizationParseError",
    "normalize_with_llm",
    "parse_llm_response",
    "parse_normalizer_output",
]
