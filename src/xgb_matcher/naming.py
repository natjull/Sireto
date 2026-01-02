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
    "SNC",
    "SELARL",
    "SELAS",
    "SELASU",
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


def normalize_text(text: str | None, *, uppercase: bool = True) -> str:
    """Accent stripping + whitespace collapse. Uppercase is optional."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    t = str(text)
    if uppercase:
        t = t.upper()
        
    t = t.replace("-", " ")
    # Replace common french accents (both cases)
    replacements = {
        "É": "E", "È": "E", "Ê": "E", "Ë": "E",
        "à": "a", "â": "a", "ä": "a", "á": "a",
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "î": "i", "ï": "i", "í": "i",
        "ô": "o", "ö": "o", "ó": "o",
        "û": "u", "ü": "u", "ú": "u", "ù": "u",
        "ç": "c",
        "À": "A", "Â": "A", "Ä": "A", "Á": "A",
        "Î": "I", "Ï": "I", "Í": "I",
        "Ô": "O", "Ö": "O", "Ó": "O",
        "Û": "U", "Ü": "U", "Ú": "U", "Ù": "U",
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


def normalize_name(raw: str | None, *, max_len: int = 100, uppercase: bool = True) -> str:
    """Normalize name, optionally preserving case."""
    base = normalize_text(raw, uppercase=uppercase)
    if not base:
        return ""
    
    # Strip legal terms (case-insensitive if we chose not to uppercase)
    tokens = base.split()
    if uppercase:
        tokens = [tok for tok in tokens if tok not in LEGAL_STOPWORDS]
    else:
        tokens = [tok for tok in tokens if tok.upper() not in LEGAL_STOPWORDS]
        
    cleaned = " ".join(tokens) or base
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
    # Cache in the candidate dict to avoid repeated normalization across many queries.
    if isinstance(cand, dict):
        cached = cand.get("_xgb_cached_candidate_names")
        if cached is not None:
            return cached

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

    # Dirigeant personne morale (name of controlling company)
    pm_dirigeant_names = cand.get("pm_dirigeant_names") or []
    if isinstance(pm_dirigeant_names, str):
        # V4 partitions store PM names as pipe-separated string
        pm_dirigeant_names = [n.strip() for n in pm_dirigeant_names.split("|") if n.strip()]
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

    if isinstance(cand, dict):
        cand["_xgb_cached_candidate_names"] = names
    return names


def primary_name(cand: dict) -> str:
    """Return the best available normalized name for display.
    
    Priority order:
    1. Etablissement names (enseigne, denomination)
    2. Unite Legale names (sigle, denomination_usuelle, denomination)
    3. PM Dirigeant names (fallback only)
    4. Person name (last resort)
    """
    if isinstance(cand, dict):
        cached = cand.get("_xgb_cached_primary_name")
        if cached is not None:
            return cached

    names = build_candidate_names(cand)
    if not names:
        return ""
    
    # Priority: ETAB > UL > PM_DIRIGEANT > PERSON
    priority_sources = [
        NameSource.ETAB_ENSEIGNE,
        NameSource.ETAB_DENOM,
        NameSource.ETAB_ENSEIGNE_X,
        NameSource.UL_SIGLE,
        NameSource.UL_DENOM_USUELLE,
        NameSource.UL_DENOM,
        NameSource.PM_DIRIGEANT,
        NameSource.PERSON_NAME,
    ]
    
    for source in priority_sources:
        for nm in names:
            if nm.source == source:
                chosen = nm.text
                if isinstance(cand, dict):
                    cand["_xgb_cached_primary_name"] = chosen
                return chosen
    
    # Fallback to first available
    chosen = names[0].text
    if isinstance(cand, dict):
        cand["_xgb_cached_primary_name"] = chosen
    return chosen


__all__ = [
    "NameSource",
    "CandidateName",
    "build_candidate_names",
    "normalize_name",
    "normalize_text",
    "primary_name",
    "looks_like_person_name",
]
