"""Common candidate structures for Pipe V6 (tasks 4.x, 5.x).

Defines the unified data model used to represent raw candidates coming from
RNE, DataGouv, and Qwant before SIRENE enrichment and LLM arbitrage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal


RAW_CANDIDATE_SOURCES = (
    "RNE",
    "DATAGOUV",
    "QWANT_PAPPERS",
    "QWANT_ANNUAIRE",
    "QWANT_SOCIETE",
)

CandidateSource = Literal[
    "RNE",
    "DATAGOUV",
    "QWANT_PAPPERS",
    "QWANT_ANNUAIRE",
    "QWANT_SOCIETE",
]


class InvalidCandidateError(ValueError):
    """Raised when a RawCandidate fails validation."""


LOGGER = logging.getLogger(__name__)


def _normalize_siren(siren: str | None) -> str | None:
    """Normalize SIREN: keep digits only, enforce length 9."""

    if siren is None:
        return None

    clean = "".join(c for c in siren.strip() if c.isdigit())
    if len(clean) != 9:
        raise ValueError(f"Invalid SIREN '{siren}': expected 9 digits, got {len(clean)}")
    return clean


def _normalize_siret(siret: str | None) -> str | None:
    """Normalize SIRET: keep digits only, enforce length 14."""

    if siret is None:
        return None

    clean = "".join(c for c in siret.strip() if c.isdigit())
    if len(clean) != 14:
        raise ValueError(f"Invalid SIRET '{siret}': expected 14 digits, got {len(clean)}")
    return clean


@dataclass(frozen=True)
class RawCandidate:
    """Common structure for candidates before SIRENE enrichment."""

    source: CandidateSource
    siren: str | None
    siret: str | None
    label: str | None
    url: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in RAW_CANDIDATE_SOURCES:
            raise InvalidCandidateError(
                f"Unknown source '{self.source}'. Expected one of {RAW_CANDIDATE_SOURCES}"
            )

        try:
            normalized_siren = _normalize_siren(self.siren)
            normalized_siret = _normalize_siret(self.siret)
        except ValueError as exc:
            raise InvalidCandidateError(f"Invalid identifier: {exc}") from exc

        object.__setattr__(self, "siren", normalized_siren)
        object.__setattr__(self, "siret", normalized_siret)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary for logging/export."""

        return {
            "source": self.source,
            "siren": self.siren,
            "siret": self.siret,
            "label": self.label,
            "url": self.url,
            "extra": self.extra,
        }


def create_raw_candidate(
    source: str,
    siren: str | None = None,
    siret: str | None = None,
    label: str | None = None,
    url: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RawCandidate:
    """Factory with automatic normalization and strict validation."""

    return RawCandidate(
        source=source,  # type: ignore[arg-type]  # runtime check in __post_init__
        siren=siren,
        siret=siret,
        label=label,
        url=url,
        extra=extra or {},
    )


def candidate_key(candidate: RawCandidate) -> tuple[str, str] | None:
    """Return a deduplication key or None when no identifier is present."""

    if candidate.siret:
        return ("siret", candidate.siret)
    if candidate.siren:
        return ("siren", candidate.siren)
    return None


def group_raw_candidates(
    candidates: list[RawCandidate],
) -> dict[tuple[str, str], list[RawCandidate]]:
    """Group candidates by SIRET/SIREN key, preserving order.

    - Uses ``candidate_key`` (SIRET priority over SIREN).
    - Excludes candidates without identifiers.

    Args:
        candidates: Flat list from multiple sources (RNE, DataGouv, Qwant).

    Returns:
        Dict mapping ("siret"|"siren", value) -> list of RawCandidate.
    """

    from collections import defaultdict

    groups: dict[tuple[str, str], list[RawCandidate]] = defaultdict(list)

    for candidate in candidates:
        key = candidate_key(candidate)
        if key is None:
            continue
        groups[key].append(candidate)

    return dict(groups)


@dataclass(frozen=True)
class NormalizedCandidate:
    """SIRENE-enriched candidate ready for LLM #2 arbitration.

    Fields mirror the contract defined in AGENTS (task 5.2).
    """

    siren: str
    siret: str | None
    name: str
    address: str
    postcode: str
    city: str
    insee_code: str | None
    legal_nature: str | None
    sources: list[str]
    raw_candidates: list[RawCandidate]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary for export/logging."""

        return {
            "siren": self.siren,
            "siret": self.siret,
            "name": self.name,
            "address": self.address,
            "postcode": self.postcode,
            "city": self.city,
            "insee_code": self.insee_code,
            "legal_nature": self.legal_nature,
            "sources": list(self.sources),
            "raw_candidates": [rc.to_dict() for rc in self.raw_candidates],
        }


__all__ = [
    "RawCandidate",
    "RAW_CANDIDATE_SOURCES",
    "CandidateSource",
    "InvalidCandidateError",
    "create_raw_candidate",
    "candidate_key",
    "group_raw_candidates",
    "NormalizedCandidate",
]
