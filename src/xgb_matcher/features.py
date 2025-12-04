"""
Shared feature engineering functions for XGBoost SIRET matcher.

This module is used by both training and inference scripts to ensure
consistency (no train/serve skew).

Features computed:
  - name_jaro: Jaro-Winkler similarity between normalized names
  - name_levenshtein: Normalized Levenshtein similarity between names
  - addr_jaro: Jaro-Winkler similarity between normalized addresses
  - addr_levenshtein: Normalized Levenshtein similarity between addresses
  - postcode_match: 1 if postcodes match exactly, 0 otherwise
  - city_match: 1 if normalized city names match, 0 otherwise
  - street_number_diff: Absolute difference between street numbers (9999 if unavailable)
  - name_token_overlap: Ratio of common words between names
  - name_first_word_match: 1 if first words match
  - name_contains_crm: 1 if candidate name contains CRM name
  - name_crm_contains_cand: 1 if CRM name contains candidate name
  - acronym_match: 1 if acronym matches expanded form
  - addr_token_overlap: Ratio of common words in addresses
  - street_name_jaro: Jaro similarity on street name only (no number)
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .naming import (
    CandidateName,
    NameSource,
    build_candidate_names,
    normalize_name,
    normalize_text,
)


# Feature names in order (used for training and inference)
FEATURE_NAMES: List[str] = [
    # Aggregated name similarities (bag of names)
    "has_any_name",
    "name_count",
    "name_jaro_max",
    "name_jaro_second",
    "name_jaro_gap",
    "name_levenshtein_max",
    "name_token_overlap_max",
    "name_first_word_match_max",
    "name_contains_crm_max",
    "name_crm_contains_cand_max",
    "acronym_match_max",
    "name_sim_max_etab",
    "name_sim_max_ul",
    "name_sim_max_sigle",
    "type_of_max_name",
    "is_ul_name_max",
    "is_sigle_max",
    "name_length_max",
    # Address / location features (unchanged)
    "addr_jaro",
    "addr_levenshtein",
    "postcode_match",
    "city_match",
    "street_number_diff",
    "addr_token_overlap",
    "street_name_jaro",
]


# --------------------------------------------------------------------------- #
# Similarity functions
# --------------------------------------------------------------------------- #


def jaro_sim(a: str, b: str) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.

    Returns 1.0 if both empty, 0.0 if only one is empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return JaroWinkler.similarity(a, b)


def levenshtein_norm(a: str, b: str) -> float:
    """
    Compute normalized Levenshtein similarity (1 - distance/max_len).

    Returns 1.0 if both empty, 0.0 if only one is empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


# --------------------------------------------------------------------------- #
# Address building
# --------------------------------------------------------------------------- #


def build_address(cand: dict) -> str:
    """
    Build normalized address from SIRENE candidate fields.

    Uses: numeroVoie + typeVoie + libelleVoie
    Falls back to complementAdresse if main fields are empty.

    Args:
        cand: Dictionary with SIRENE candidate data

    Returns:
        Normalized address string
    """
    parts = [
        cand.get("numeroVoie"),
        cand.get("typeVoie"),
        cand.get("libelleVoie"),
    ]
    addr = " ".join(str(x) for x in parts if x and str(x).strip())
    if not addr and cand.get("complementAdresse"):
        addr = str(cand["complementAdresse"])
    return normalize_text(addr)


def extract_street_number(addr: str | None) -> str | None:
    """
    Extract leading street number from address string.

    Uses regex to handle variations like "12", "12B", "12 BIS", etc.

    Args:
        addr: Address string

    Returns:
        Street number as string, or None if not found
    """
    if not addr:
        return None
    addr_str = str(addr).strip()
    if not addr_str:
        return None
    # Match leading digits (optionally followed by letter like "12B")
    match = re.match(r"^(\d+)", addr_str)
    if match:
        return match.group(1)
    return None


def get_street_name(cand: dict) -> str:
    """
    Get only the street name part (typeVoie + libelleVoie), excluding number.

    Used for street_name_jaro feature to compare street names independently.
    """
    parts = [
        cand.get("typeVoie"),
        cand.get("libelleVoie"),
    ]
    return normalize_text(" ".join(str(x) for x in parts if x and str(x).strip()))


def extract_street_name_from_address(addr: str | None) -> str:
    """
    Extract street name from full address string (remove leading number).

    Args:
        addr: Full address string

    Returns:
        Street name without the leading number
    """
    if not addr:
        return ""
    addr_str = normalize_text(addr)
    # Remove leading number and any bis/ter suffix
    cleaned = re.sub(r"^\d+\s*(BIS|TER|QUATER|B|T|Q)?\s*", "", addr_str, flags=re.IGNORECASE)
    return cleaned.strip()


# --------------------------------------------------------------------------- #
# Matching functions
# --------------------------------------------------------------------------- #


def postal_match(cp1: str | None, cp2: str | None) -> int:
    """Check if two postal codes match exactly."""
    if not cp1 or not cp2:
        return 0
    return int(str(cp1).strip() == str(cp2).strip())


def city_match(c1: str | None, c2: str | None) -> int:
    """Check if two city names match after normalization."""
    if not c1 or not c2:
        return 0
    return int(normalize_text(c1) == normalize_text(c2))


def street_number_diff(n1: str | None, n2: str | None) -> float:
    """
    Compute absolute difference between street numbers.

    Returns 9999.0 if either number is unavailable or invalid.
    """
    try:
        num1 = int(n1) if n1 else None
        num2 = int(n2) if n2 else None
        if num1 is not None and num2 is not None:
            return float(abs(num1 - num2))
        return 9999.0
    except (TypeError, ValueError):
        return 9999.0


# --------------------------------------------------------------------------- #
# Token-based features (new for improved ranking)
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> Set[str]:
    """Split normalized text into set of tokens (words)."""
    if not text:
        return set()
    return set(text.split())


def token_overlap(a: str, b: str) -> float:
    """
    Compute ratio of common tokens between two strings.

    Returns |A ∩ B| / |A ∪ B| (Jaccard similarity on tokens).
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def first_word_match(a: str, b: str) -> int:
    """
    Check if the first word of both strings matches.

    Useful for "SARL X" vs "X SARL" patterns.
    """
    if not a or not b:
        return 0
    words_a = a.split()
    words_b = b.split()
    if not words_a or not words_b:
        return 0
    return int(words_a[0] == words_b[0])


def contains_check(container: str, contained: str) -> int:
    """
    Check if container string contains the contained string.

    Useful for detecting "GE FRUITS" in "GE FRUITS DISTRIBUTION".
    """
    if not container or not contained:
        return 0
    return int(contained in container)


def _extract_acronym(text: str) -> str:
    """Extract initials from a multi-word string (e.g., 'SOCIETE NATIONALE' -> 'SN')."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= 1:
        return ""
    return "".join(w[0] for w in words if w)


def _is_acronym(text: str) -> bool:
    """Check if text looks like an acronym (2-6 uppercase letters, no spaces)."""
    if not text:
        return False
    # After normalization, acronyms are uppercase words with 2-6 chars
    return bool(re.match(r"^[A-Z]{2,6}$", text.replace(" ", "")))


def acronym_match(a: str, b: str) -> int:
    """
    Check if one string is the acronym of the other.

    Handles cases like "SNCF" ↔ "SOCIETE NATIONALE DES CHEMINS DE FER".
    """
    if not a or not b:
        return 0

    # Determine which might be the acronym
    a_is_acronym = _is_acronym(a.replace(" ", ""))
    b_is_acronym = _is_acronym(b.replace(" ", ""))

    if a_is_acronym and not b_is_acronym:
        # a is acronym, b is expanded form
        expanded_acronym = _extract_acronym(b)
        return int(a.replace(" ", "") == expanded_acronym)
    elif b_is_acronym and not a_is_acronym:
        # b is acronym, a is expanded form
        expanded_acronym = _extract_acronym(a)
        return int(b.replace(" ", "") == expanded_acronym)

    return 0


# --------------------------------------------------------------------------- #
# Aggregation helpers for bag-of-names strategy
# --------------------------------------------------------------------------- #


_SOURCE_ENCODING = {
    NameSource.NONE: 0,
    NameSource.ETAB_ENSEIGNE: 1,
    NameSource.ETAB_DENOM: 2,
    NameSource.ETAB_ENSEIGNE_X: 3,
    NameSource.UL_SIGLE: 4,
    NameSource.UL_DENOM_USUELLE: 5,
    NameSource.UL_DENOM: 6,
    NameSource.PERSON_NAME: 7,
}


def _encode_source(source: NameSource | str) -> int:
    """Map NameSource to an ordinal value for model consumption."""
    return _SOURCE_ENCODING.get(source, 0)


def _init_name_feature_defaults() -> Dict[str, float]:
    """Default values when no candidate names are available."""
    return {
        "has_any_name": 0.0,
        "name_count": 0.0,
        "name_jaro_max": 0.0,
        "name_jaro_second": 0.0,
        "name_jaro_gap": 0.0,
        "name_levenshtein_max": 0.0,
        "name_token_overlap_max": 0.0,
        "name_first_word_match_max": 0.0,
        "name_contains_crm_max": 0.0,
        "name_crm_contains_cand_max": 0.0,
        "acronym_match_max": 0.0,
        "name_sim_max_etab": 0.0,
        "name_sim_max_ul": 0.0,
        "name_sim_max_sigle": 0.0,
        "type_of_max_name": 0.0,
        "is_ul_name_max": 0.0,
        "is_sigle_max": 0.0,
        "name_length_max": 0.0,
    }


# --------------------------------------------------------------------------- #
# Main feature computation
# --------------------------------------------------------------------------- #


def make_features(crm_row: pd.Series, cand: dict) -> Dict[str, float]:
    """
    Compute all similarity features between a CRM row and a SIRENE candidate.

    This function is used by both training and inference to ensure consistency.

    Args:
        crm_row: Pandas Series with CRM data (crm_name, crm_address, postcode, etc.)
        cand: Dictionary with SIRENE candidate data

    Returns:
        Dictionary with all feature values
    """
    # Normalize inputs
    crm_name = normalize_name(crm_row.get("crm_name", ""))
    crm_addr = normalize_text(crm_row.get("crm_address", ""))
    cand_addr = build_address(cand)

    # Bag of names for the candidate
    candidate_names: List[CandidateName] = build_candidate_names(cand)

    # Street components
    crm_street_num = extract_street_number(crm_row.get("crm_address"))
    cand_street_num = cand.get("numeroVoie")
    if cand_street_num:
        cand_street_num = str(cand_street_num)

    crm_street_name = extract_street_name_from_address(crm_row.get("crm_address"))
    cand_street_name = get_street_name(cand)

    crm_city = crm_row.get("crm_city_addr") or crm_row.get("crm_city", "")

    # ---------------- Name similarities (aggregated) -----------------
    name_features = _init_name_feature_defaults()

    if candidate_names:
        sims = []
        for nm in candidate_names:
            sim_jaro = jaro_sim(crm_name, nm.text)
            sim_lev = levenshtein_norm(crm_name, nm.text)
            tok = token_overlap(crm_name, nm.text)
            fw = first_word_match(crm_name, nm.text)
            contains_crm = contains_check(nm.text, crm_name)
            crm_contains = contains_check(crm_name, nm.text)
            acr = acronym_match(crm_name, nm.text)

            sims.append(
                {
                    "nm": nm,
                    "jaro": sim_jaro,
                    "lev": sim_lev,
                    "tok": tok,
                    "fw": float(fw),
                    "contains_crm": float(contains_crm),
                    "crm_contains": float(crm_contains),
                    "acronym": float(acr),
                }
            )

        # Global maxima
        name_features["has_any_name"] = 1.0
        name_features["name_count"] = float(len(candidate_names))

        sims_sorted = sorted(sims, key=lambda x: x["jaro"], reverse=True)
        top = sims_sorted[0]
        second = sims_sorted[1]["jaro"] if len(sims_sorted) > 1 else 0.0

        name_features.update(
            {
                "name_jaro_max": top["jaro"],
                "name_jaro_second": second,
                "name_jaro_gap": top["jaro"] - second,
                "name_levenshtein_max": top["lev"],
                "name_token_overlap_max": max(s["tok"] for s in sims),
                "name_first_word_match_max": max(s["fw"] for s in sims),
                "name_contains_crm_max": max(s["contains_crm"] for s in sims),
                "name_crm_contains_cand_max": max(s["crm_contains"] for s in sims),
                "acronym_match_max": max(s["acronym"] for s in sims),
                "type_of_max_name": float(_encode_source(top["nm"].source)),
                "is_ul_name_max": float(top["nm"].is_ul_name),
                "is_sigle_max": float(top["nm"].is_sigle),
                "name_length_max": float(len(top["nm"].text)),
            }
        )

        # Source-specific maxima (based on Jaro)
        name_features["name_sim_max_etab"] = max(
            [s["jaro"] for s in sims if not s["nm"].is_ul_name] or [0.0]
        )
        name_features["name_sim_max_ul"] = max(
            [s["jaro"] for s in sims if s["nm"].is_ul_name] or [0.0]
        )
        name_features["name_sim_max_sigle"] = max(
            [s["jaro"] for s in sims if s["nm"].is_sigle] or [0.0]
        )

    # ---------------- Address & location features -----------------
    addr_features = {
        "addr_jaro": jaro_sim(crm_addr, cand_addr),
        "addr_levenshtein": levenshtein_norm(crm_addr, cand_addr),
        "postcode_match": float(postal_match(crm_row.get("postcode"), cand.get("postcode"))),
        "city_match": float(city_match(crm_city, cand.get("city"))),
        "street_number_diff": street_number_diff(crm_street_num, cand_street_num),
        "addr_token_overlap": token_overlap(crm_addr, cand_addr),
        "street_name_jaro": jaro_sim(crm_street_name, cand_street_name),
    }

    features = {**name_features, **addr_features}
    return features
