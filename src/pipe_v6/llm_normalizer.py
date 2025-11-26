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
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .config import PipelineConfig
from .llm_utils import LLMCallError, LLMResponse, OllamaClient


LOGGER = logging.getLogger(__name__)

CRM_CATEGORY_VALUES = ("PUBLIC", "PRIVE", "EQUIPEMENT_URBAIN", "INCONNU")

_NAME_ALLOWED_RE = re.compile(r"^[A-Z0-9 '&\-/]+$")
_ADDRESS_ALLOWED_RE = re.compile(r"^[A-Z0-9 '()\-/]+$")
# Accept numeric street numbers with optional alphabetic suffixes (BIS/TER/QUATER) or single letters.
# Examples matched: "2B", "2 BIS", "2BIS", "12 TER", "4 QUATER".
_ADDRESS_RE = re.compile(
    r"^(?P<number>[0-9]{1,5}(?: ?[A-Z]{1,4})?)\s+(?P<street>[A-Z0-9' \-]+?)\s+(?P<postcode>\d{5})$"
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

_PROMPT_FEW_SHOT = """
1) INPUT
- Nom brut : ITAFRAN Site 2
- Adresse : 21 AVENUE DE L INDUSTRIE
- Code postal : 69960
- Commune : CORBAS
OUTPUT
{"normalized_name": "ITAFRAN", "normalized_address": "21 AVENUE DE L INDUSTRIE 69960", "category": "PRIVE"}

2) INPUT
- Nom brut : Groupe SIRENE (IPO)
- Adresse : 17 Rue Emile Zola
- Code postal : 69150
- Commune : DECINES-CHARPIEU
OUTPUT
{"normalized_name": "GROUPE SIRENE", "normalized_address": "17 RUE EMILE ZOLA 69150", "category": "PRIVE"}

3) INPUT
- Nom brut : Mairie de Corbas
- Adresse : 1 Place de la République
- Code postal : 69960
- Commune : CORBAS
OUTPUT
{"normalized_name": "MAIRIE", "normalized_address": "1 PLACE DE LA REPUBLIQUE 69960", "category": "PUBLIC"}

4) INPUT
- Nom brut : École Élémentaire Jules Ferry
- Adresse : 12 Rue de Paris
- Code postal : 75011
- Commune : PARIS
OUTPUT
{"normalized_name": "ECOLE ELEMENTAIRE JULES FERRY", "normalized_address": "12 RUE DE PARIS 75011", "category": "PUBLIC"}

5) INPUT
- Nom brut : Armoire Télécom NRO-CORBAS-01
- Adresse : 45 Rue Marcel Mérieux
- Code postal : 69960
- Commune : CORBAS
OUTPUT
{"normalized_name": "ARMOIRE TELECOM NRO CORBAS 01", "normalized_address": "45 RUE MARCEL MERIEUX 69960", "category": "EQUIPEMENT_URBAIN"}

6) INPUT
- Nom brut : Bâtiment non identifié
- Adresse : 99 Rue Inconnue
- Code postal : 69000
- Commune : LYON
OUTPUT
{"normalized_name": "BATIMENT NON IDENTIFIE", "normalized_address": "99 RUE INCONNUE 69000", "category": "INCONNU"}
"""

# Suffix to force JSON termination and help the model stop cleanly.
_PROMPT_STOP_SUFFIX = "Ajoute le token </END_JSON> juste après l'accolade fermante.\n</END_JSON>"

# Suffix pour forcer la clôture JSON et limiter le texte après l'objet
_PROMPT_STOP_SUFFIX = "\nAjoute le token </END_JSON> juste après l'accolade fermante.\n</END_JSON>"


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

    stop_tokens = ["</END_JSON>"]
    max_tokens = getattr(config, "max_tokens_normalizer", config.max_tokens)
    attempts = 2
    last_exc: Exception | None = None

    try:
        for attempt in range(1, attempts + 1):
            try:
                response = client.call_json(
                    prompt,
                    stop=stop_tokens,
                    num_predict=max_tokens if attempt == 1 else int(max_tokens * 1.5),
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt == attempts:
                    raise
                log.warning(
                    "Normalizer LLM retry (%s/%s) after error: %s",
                    attempt,
                    attempts,
                    exc,
                )
                continue
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


# ---------------------------------------------------------------------------
# Helper to access rows that may be Series, dict-like, or NamedTuple
# ---------------------------------------------------------------------------

def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Best-effort getter that works for pandas Series, dict-like objects, or itertuples rows."""

    if hasattr(row, "get"):
        try:
            return row.get(key, default)  # type: ignore[call-arg]
        except Exception:
            pass
    if hasattr(row, "_asdict"):
        try:
            return row._asdict().get(key, default)  # type: ignore[call-arg]
        except Exception:
            pass
    if hasattr(row, key):
        return getattr(row, key, default)
    try:
        return row[key]  # type: ignore[index]
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Public orchestrating helpers (task 3.3)
# ---------------------------------------------------------------------------


def build_normalizer_prompt(row: pd.Series) -> str:
    """Construct the French prompt for the CRM normalizer LLM.

    The prompt is strict about JSON-only output and encodes the business rules
    defined for task 3.3. It gracefully mentions missing address components so
    the model can still return a parseable result.
    """

    def _value(key: str) -> str:
        val = _row_get(row, key, "")
        if pd.isna(val):
            return ""
        return str(val)

    crm_name = _value("crm_name")
    street_number = _value("street_number")
    street_name = _value("street_name")
    postcode = _value("postcode")
    city = _value("city")

    missing_parts: list[str] = []
    if pd.isna(street_number) or not str(street_number).strip():
        street_number = "0"  # placeholder accepted by the address regex
        missing_parts.append("numéro de voie")
    if pd.isna(postcode) or not str(postcode).strip():
        postcode = "00000"  # placeholder to keep address parseable
        missing_parts.append("code postal")

    missing_note = "\n" + "\n".join(
        [
            "CHAMPS MANQUANTS : " + ", ".join(missing_parts),
            "Si un champ est manquant, garde la valeur fournie (ou le placeholder) et mets category=INCONNU si tu n'es pas sûr.",
        ]
    ) if missing_parts else ""

    examples = _PROMPT_FEW_SHOT.strip()

    prompt = f"""
Tu es un assistant chargé de normaliser des lignes CRM françaises.
Tu dois répondre UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, sans balises ```.

Format JSON OBLIGATOIRE :
{{
  "normalized_name": "NOM NORMALISE MAJUSCULES",
  "normalized_address": "NUMERO VOIE CODE_POSTAL",
  "category": "PUBLIC|PRIVE|EQUIPEMENT_URBAIN|INCONNU"
}}

Règles strictes pour normalized_name :
1. TOUT EN MAJUSCULES.
2. Supprime les formes juridiques : SAS, SARL, SASU, SA, ASSOCIATION, ENTREPRISE, etc.
3. Supprime les marqueurs opérationnels : SITE, AGENCE, BUREAU, DIRECTION, SERVICE, ANTENNE.
4. Supprime le nom de la commune si présent.
5. Translittère les accents (É→E, À→A...).

Règles strictes pour normalized_address :
- Format exact : "<numero> <voie> <code postal>" en MAJUSCULES, SANS ville.
- Si le numéro manque, utilise "0" (ou le numéro fourni).
- Si le code postal manque, utilise "00000".
- Pas de virgules, pas de ville, pas de pays.

Catégories (une seule) :
PUBLIC : collectivités, services de l'Etat, établissements publics (EPA/EPIC), hôpitaux publics, écoles publiques.
PRIVE : sociétés commerciales, artisans, professions libérales, associations privées, boutiques (Orange, banques...).
EQUIPEMENT_URBAIN : infrastructures sans personne morale claire (armoire télécom, NRO, antenne, abri bus, station vélo, transformateur, parking automatique anonyme).
INCONNU : informations insuffisantes.

Cas limites :
- Parking Vinci/Indigo -> PRIVE
- Parking sans marque -> EQUIPEMENT_URBAIN
- Gare SNCF ou station métro RATP -> PUBLIC (EPIC)
- Boutique SNCF ou Orange Boutique -> PRIVE
- La Poste bureau -> PUBLIC
- Association sportive locale -> PRIVE

Si tu hésites : choisis la catégorie la plus probable; si vraiment impossible, mets INCONNU.

Exemples :
{examples}

DONNEES CRM A NORMALISER :
- Nom brut : {crm_name}
- Adresse brute : {street_number} {street_name}
- Code postal : {postcode}
- Commune : {city}{missing_note}

Réponds maintenant UNIQUEMENT avec l'objet JSON demandé.
{_PROMPT_STOP_SUFFIX}
""".strip()

    return prompt


def normalize_crm_entry(
    row: pd.Series,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
    client: OllamaClient | None = None,
) -> NormalizedCRMEntry:
    """Normalize a CRM row with the LLM (task 3.3).

    Raises ``LLMCallError`` or ``NormalizationParseError`` on failure; callers
    (pipeline layer) decide how to handle errors.
    """

    log = logger or LOGGER

    crm_id = _row_get(row, "crm_id")
    if pd.isna(crm_id):
        raise ValueError("crm_id is mandatory for normalization")

    street_number = _row_get(row, "street_number")
    postcode = _row_get(row, "postcode")
    if pd.isna(street_number) or pd.isna(postcode):
        log.warning(
            "CRM %s: Missing address components (number=%s, postcode=%s)",
            crm_id,
            street_number,
            postcode,
        )

    prompt = build_normalizer_prompt(row)

    result = normalize_with_llm(
        prompt=prompt,
        config=config,
        logger=log,
        client=client,
        expected_city=_row_get(row, "city") if not pd.isna(_row_get(row, "city")) else None,
    )

    log.debug(
        "CRM %s normalized: name=%s, addr=%s, category=%s",
        crm_id,
        result.normalized_name,
        result.normalized_address,
        result.category,
    )

    return result


def dump_normalizer_prompt(row: pd.Series, output_path: str | None = None) -> str:
    """Utility to inspect the built prompt for a given CRM row.

    If ``output_path`` is provided, the prompt is written to disk for debugging.
    """

    prompt = build_normalizer_prompt(row)
    if output_path:
        Path(output_path).write_text(prompt, encoding="utf-8")
    return prompt


__all__ = [
    "CRM_CATEGORY_VALUES",
    "NormalizedCRMEntry",
    "NormalizationParseError",
    "normalize_with_llm",
    "parse_llm_response",
    "parse_normalizer_output",
    "normalize_crm_entry",
    "dump_normalizer_prompt",
    "build_normalizer_prompt",
]
