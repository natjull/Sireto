from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.candidate_store import NormalizedCandidate
from pipe_v6.llm_matcher import build_matcher_prompt
from pipe_v6.llm_normalizer import NormalizedCRMEntry


def _cand(num: int, category: str, sources: list[str]) -> NormalizedCandidate:
    siret = f"1234567890{num:04d}"
    return NormalizedCandidate(
        siren=siret[:9],
        siret=siret,
        name=f"ACME-{num}",
        address=f"{num} RUE",
        postcode="75001",
        city="PARIS",
        insee_code="75056",
        legal_nature="5710",
        sources=sources,
        raw_candidates=[],
        category=category,
    )


def test_build_matcher_prompt_contains_sections():
    row = {
        "crm_name": "ACME HQ",
        "street_number": "1",
        "street_name": "RUE DE PARIS",
        "postcode": "75001",
        "city": "PARIS",
    }
    norm = NormalizedCRMEntry(
        normalized_name="ACME HQ",
        normalized_address="1 RUE DE PARIS 75001",
        category="PRIVE",
    )
    candidates = [
        _cand(1, "PRIVE", ["RNE", "DATAGOUV"]),
        _cand(2, "PUBLIC", ["RNE"]),
    ]

    prompt = build_matcher_prompt(row, norm, candidates)

    assert "DONNEES CRM ORIGINALES" in prompt
    assert "DONNEES CRM NORMALISEES" in prompt
    assert "CANDIDATS SIRENE" in prompt
    assert "## Candidat 1" in prompt
    assert "## Candidat 2" in prompt
    assert '"decision": "BEST_MATCH" ou "NO_MATCH"' in prompt
    assert "UNIQUEMENT en JSON" in prompt or "UNIQUEMENT en JSON valide" in prompt


def test_build_matcher_prompt_includes_multi_source_flag():
    row = {"crm_name": "ACME", "street_number": "", "street_name": "", "postcode": "", "city": ""}
    norm = NormalizedCRMEntry("ACME", "1 RUE 00000", "PRIVE")
    candidates = [_cand(1, "PRIVE", ["RNE", "DATAGOUV", "QWANT"])]

    prompt = build_matcher_prompt(row, norm, candidates)

    assert "Multi-source : Oui" in prompt
