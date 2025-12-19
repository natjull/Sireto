"""Utilities for building and normalizing candidate names (bag-of-names strategy)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import List, Set

import pandas as pd


class NameSource(str, Enum):
    ETAB_ENSEIGNE = "ETAB_ENSEIGNE"
    ETAB_ENSEIGNE_X = "ETAB_ENSEIGNE_X"  # enseigne2/3
    ETAB_DENOM = "ETAB_DENOM"
    SIREN_ETAB_NAME = "SIREN_ETAB_NAME"
    PM_DIRIGEANT = "PM_DIRIGEANT"
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

# Minimal but high-coverage list of common French first names (uppercased, no accents)
# Used only for a lightweight heuristic; keep it short to avoid false positives.
COMMON_FIRST_NAMES: Set[str] = {
    "JEAN",
    "PIERRE",
    "PAUL",
    "JACQUES",
    "MICHEL",
    "ALAIN",
    "LOUIS",
    "CLAUDE",
    "GERARD",
    "FRANCOIS",
    "HENRI",
    "ANDRE",
    "RENE",
    "MARC",
    "DANIEL",
    "NICOLAS",
    "OLIVIER",
    "LAURENT",
    "ALEXANDRE",
    "THOMAS",
    "LUC",
    "ANTOINE",
    "SIMON",
    "VICTOR",
    "GABRIEL",
    "JULIEN",
    "YANN",
    "ARTHUR",
    "PAULINE",
    "MARIE",
    "ANNE",
    "NATHALIE",
    "CHRISTINE",
    "CATHERINE",
    "ISABELLE",
    "SOPHIE",
    "LAURENCE",
    "CELINE",
    "SANDRINE",
    "EMILIE",
    "JULIE",
    "ELISE",
    "LAURA",
    "ANNA",
    "SARAH",
    "CLARA",
    "EVA",
    "LENA",
    "LEA",
    "MANON",
    "MAELLE",
    "AMANDINE",
    "AUDREY",
    "MARION",
    "HELENE",
    "FLORENCE",
}

# Codes SIRENE (categorieJuridiqueUniteLegale) correspondant aux personnes physiques / EI
# Utilisés pour autoriser l'usage de noms de personne dans le bag-of-names.
PERSON_UL_CODES: Set[str] = {
    "1000",  # Entrepreneur individuel
    "1100",  # Entrepreneur individuel à responsabilité limitée
    "1200",
    "1300",
    "1400",
    "1500",
    "1600",
    "2110",  # Indivision entre personnes physiques
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
    # Harmonize separators (DECINES-CHARPIEU == DECINES CHARPIEU)
    t = t.replace("-", " ")
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


def looks_like_person_name(text: str | None) -> bool:
    """Heuristic detection of a person full name.

    Criteria (all must hold):
      - non-empty, normalized
      - 2 or 3 tokens
      - no digits
      - at least one token is a common French first name
      - none of the tokens are legal stopwords (SARL, SAS, ...)
    """
    if not text:
        return False
    norm = normalize_text(text)
    if not norm:
        return False
    tokens = norm.split()
    if not (2 <= len(tokens) <= 3):
        return False
    if any(any(ch.isdigit() for ch in tok) for tok in tokens):
        return False
    if any(tok in LEGAL_STOPWORDS for tok in tokens):
        return False
    return any(tok in COMMON_FIRST_NAMES for tok in tokens)


def build_candidate_names(cand: dict) -> List[CandidateName]:
    """Return all available normalized names for a SIRET candidate."""
    names: List[CandidateName] = []

    def add(val: str | None, source: NameSource, *, is_ul: bool = False, is_sigle: bool = False):
        norm = normalize_name(val)
        # Ignore pure numeric or ultra-short tokens (e.g., "38") which are usually noise
        if norm and not norm.isdigit() and len(norm) > 2:
            names.append(CandidateName(text=norm, source=source, is_ul_name=is_ul, is_sigle=is_sigle))

    # Etablissement level
    add(cand.get("enseigne1"), NameSource.ETAB_ENSEIGNE, is_ul=False)
    add(cand.get("denomination"), NameSource.ETAB_DENOM, is_ul=False)
    add(cand.get("enseigne2"), NameSource.ETAB_ENSEIGNE_X, is_ul=False)
    add(cand.get("enseigne3"), NameSource.ETAB_ENSEIGNE_X, is_ul=False)

    # Other SIRETs of same SIREN (establishment names)
    siren_etab_names = cand.get("siren_etab_names") or []
    if isinstance(siren_etab_names, str):
        siren_etab_names = [siren_etab_names]
    for val in siren_etab_names:
        add(val, NameSource.SIREN_ETAB_NAME, is_ul=False)

    # Dirigeant personne morale (name of controlling company)
    pm_dirigeant_names = cand.get("pm_dirigeant_names") or []
    if isinstance(pm_dirigeant_names, str):
        pm_dirigeant_names = [pm_dirigeant_names]
    for val in pm_dirigeant_names:
        add(val, NameSource.PM_DIRIGEANT, is_ul=False)

    # Unite Legale level
    add(cand.get("sigle_ul"), NameSource.UL_SIGLE, is_ul=True, is_sigle=True)
    add(cand.get("denomination_usuelle_ul"), NameSource.UL_DENOM_USUELLE, is_ul=True)
    add(cand.get("denomination_ul"), NameSource.UL_DENOM, is_ul=True)

    # Person name (optional, often noisy; kept for completeness)
    person_fullname = None
    if cand.get("prenom_usuel_ul") or cand.get("nom_ul"):
        person_fullname = " ".join(filter(None, [cand.get("prenom_usuel_ul"), cand.get("nom_ul")]))

    # On n'inclut les noms de personnes que si la CJ UL correspond à une personne physique / EI
    cj_ul = cand.get("cj_ul")
    if cj_ul in PERSON_UL_CODES and person_fullname:
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
    "looks_like_person_name",
]
