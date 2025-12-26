import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.candidate_store import (  # noqa: E402
    create_raw_candidate,
    group_raw_candidates,
)


def test_group_by_siret_multiple_sources():
    candidates = [
        create_raw_candidate(source="RNE", siret="12345678901234", label="ACME"),
        create_raw_candidate(source="DATAGOUV", siret="12345678901234", label="ACME SA"),
        create_raw_candidate(source="WEB_PAPPERS", siret="12345678901234", label="Acme"),
    ]

    groups = group_raw_candidates(candidates)

    assert len(groups) == 1
    assert ("siret", "12345678901234") in groups
    assert len(groups[("siret", "12345678901234")]) == 3

    sources = [c.source for c in groups[("siret", "12345678901234")]]
    assert sources == ["RNE", "DATAGOUV", "WEB_PAPPERS"]


def test_group_by_siren_when_siret_absent():
    candidates = [
        create_raw_candidate(source="RNE", siren="123456789", label="BETA"),
        create_raw_candidate(source="DATAGOUV", siren="123456789", label="BETA CORP"),
    ]

    groups = group_raw_candidates(candidates)

    assert len(groups) == 1
    assert ("siren", "123456789") in groups
    assert len(groups[("siren", "123456789")]) == 2


def test_exclude_candidates_without_identifiers():
    candidates = [
        create_raw_candidate(source="WEB_ANNUAIRE", label="Unknown Company"),
        create_raw_candidate(source="RNE", siret="12345678901234", label="Valid"),
    ]

    groups = group_raw_candidates(candidates)

    assert len(groups) == 1
    assert ("siret", "12345678901234") in groups


def test_preserve_order_within_group():
    candidates = [
        create_raw_candidate(source="WEB_SOCIETE", siret="11111111111111"),
        create_raw_candidate(source="RNE", siret="11111111111111"),
        create_raw_candidate(source="DATAGOUV", siret="11111111111111"),
    ]

    groups = group_raw_candidates(candidates)

    sources = [c.source for c in groups[("siret", "11111111111111")]]
    assert sources == ["WEB_SOCIETE", "RNE", "DATAGOUV"]


def test_siret_and_siren_separate_groups():
    candidates = [
        create_raw_candidate(source="RNE", siret="12345678901234", siren="123456789"),
        create_raw_candidate(source="DATAGOUV", siren="123456789"),
    ]

    groups = group_raw_candidates(candidates)

    assert len(groups) == 2
    assert ("siret", "12345678901234") in groups
    assert ("siren", "123456789") in groups
    assert len(groups[("siret", "12345678901234")]) == 1
    assert len(groups[("siren", "123456789")]) == 1


def test_empty_input_returns_empty_dict():
    groups = group_raw_candidates([])
    assert groups == {}


def test_all_candidates_without_ids_returns_empty_dict():
    candidates = [
        create_raw_candidate(source="WEB_ANNUAIRE", label="A"),
        create_raw_candidate(source="WEB_SOCIETE", label="B"),
    ]

    groups = group_raw_candidates(candidates)
    assert groups == {}
