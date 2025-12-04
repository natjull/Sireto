"""Utilities for building and normalizing candidate names (bag-of-names strategy)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import pandas as pd


class NameSource(str, Enum):
    ETAB_ENSEIGNE = "ETAB_ENSEIGNE"
    ETAB_ENSEIGNE_X = "ETAB_ENSEIGNE_X"  # enseigne2/3
    ETAB_DENOM = "ETAB_DENOM"
    UL_SIGLE = "UL_SIGLE"
    UL_DENOM_USUELLE = "UL_DENOM_USUELLE"
    UL_DENOM = "UL_DENOM"
    PERSON_NAME = "PERSON_NAME"
    NONE = "NONE"


LEGAL_STOPWORDS = {
    "SAS",
    "SASU",
    "SARL",
    "EURL",
    "SCI",
    "SCIC",
    "SCOP",
    "SA",
    "SELARL",
    "SELAS",
    "SELASU",
    "SNC",
    "ASSOCIATION",
    "ASSOC",
    "ASL",
    "FONDATION",
    "SOCIETE",
    "ENTREPRISE",
    "ETABLISSEMENT",
}


@dataclass
class CandidateName:
    text: str
    source: NameSource
    is_ul_name: bool
    is_sigle: bool


def normalize_text(text: str | None) -> str:
    """Uppercase + accent stripping + whitespace collapse."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text).upper()
    replacements = {
        "É": "E",
        "È": "E",
        "Ê": "E",
        "Ë": "E",
        "À": "A",
        "Â": "A",
        "Ä": "A",
        "Ô": "O",
        "Ö": "O",
        "Û": "U",
        "Ü": "U",
        "Ù": "U",
        "Î": "I",
        "Ï": "I",
        "Ç": "C",
    }
    for old, new in replacements.items():
        t = t.replace(old, new)
    return " ".join(t.split())


def _strip_legal_terms(text: str) -> str:
    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in LEGAL_STOPWORDS]
    return " ".join(tokens)


def truncate_name(text: str, max_len: int = 100) -> str:
    if len(text) <= max_len:
        return text
    # try to cut on a space boundary
    cutoff = text[: max_len + 1]
    if " " in cutoff:
        cutoff = cutoff.rsplit(" ", 1)[0]
    return cutoff


def normalize_name(raw: str | None, *, max_len: int = 100) -> str:
    base = normalize_text(raw)
    if not base:
        return ""
    stripped = _strip_legal_terms(base)
    cleaned = stripped or base  # fallback if all tokens removed
    return truncate_name(cleaned, max_len=max_len)


def build_candidate_names(cand: dict) -> List[CandidateName]:
    """Return all available normalized names for a SIRET candidate."""
    names: List[CandidateName] = []

    def add(val: str | None, source: NameSource, *, is_ul: bool = False, is_sigle: bool = False):
        norm = normalize_name(val)
        if norm:
            names.append(CandidateName(text=norm, source=source, is_ul_name=is_ul, is_sigle=is_sigle))

    # Etablissement level
    add(cand.get("enseigne1"), NameSource.ETAB_ENSEIGNE, is_ul=False)
    add(cand.get("denomination"), NameSource.ETAB_DENOM, is_ul=False)
    add(cand.get("enseigne2"), NameSource.ETAB_ENSEIGNE_X, is_ul=False)
    add(cand.get("enseigne3"), NameSource.ETAB_ENSEIGNE_X, is_ul=False)

    # Unite Legale level
    add(cand.get("sigle_ul"), NameSource.UL_SIGLE, is_ul=True, is_sigle=True)
    add(cand.get("denomination_usuelle_ul"), NameSource.UL_DENOM_USUELLE, is_ul=True)
    add(cand.get("denomination_ul"), NameSource.UL_DENOM, is_ul=True)

    # Person name (optional, often noisy; kept for completeness)
    person_fullname = None
    if cand.get("prenom_usuel_ul") or cand.get("nom_ul"):
        person_fullname = " ".join(filter(None, [cand.get("prenom_usuel_ul"), cand.get("nom_ul")]))
    add(person_fullname, NameSource.PERSON_NAME, is_ul=True, is_sigle=False)

    return names


def primary_name(cand: dict) -> str:
    """Return the first available normalized name (for display), else empty."""
    names = build_candidate_names(cand)
    if not names:
        return ""
    return names[0].text


__all__ = [
    "NameSource",
    "CandidateName",
    "build_candidate_names",
    "normalize_name",
    "normalize_text",
    "primary_name",
]
