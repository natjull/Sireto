import pytest

from src.xgb_matcher.retrieval_source_coverage_v117 import (
    REQUIRED, field_coverage, temporal_coverage,
)


def test_legacy_complete_boolean_cannot_override_counts():
    result = temporal_coverage(dict(temporal_complete=True, official_source_counts={
        'SIRENE_CURRENT': 125551755, 'RNE': 81640996, 'BODACC': 98133358,
    }))
    assert result['status'] == 'DECLARATION_CONTRADICTED'
    assert result['missing'] == ['SIRENE_HISTORY', 'SIRENE_SUCCESSION']
    assert result['declaration_consistent'] is False


def test_presence_does_not_prove_exhaustive_history():
    result = temporal_coverage(dict(temporal_complete=True,
                                    official_source_counts=dict.fromkeys(REQUIRED, 1)))
    assert result['status'] == 'SOURCE_FAMILIES_PRESENT_NOT_FULL_COVERAGE'
    assert not result['exhaustive_period_coverage_proven']
    assert result['declaration_consistent'] is None


def test_absent_counts_are_unknown_not_empty():
    result = temporal_coverage(dict(temporal_complete=True))
    assert result['status'] == 'UNVERIFIED_SOURCE_FAMILIES'
    assert result['missing'] is None


def test_zero_source_is_not_presence():
    result = temporal_coverage(dict(official_source_counts=dict.fromkeys(REQUIRED, 0)))
    assert result['status'] == 'SOURCE_FAMILIES_MISSING'
    assert result['present'] == []


@pytest.mark.parametrize('count', [-1, True, 1.5, '1'])
def test_bad_counts_rejected(count):
    with pytest.raises(ValueError):
        temporal_coverage(dict(official_source_counts={'SIRENE_HISTORY': count}))


def test_available_ei_field_can_be_unprojected():
    result = field_coverage(['nomUniteLegale', 'prenomUsuelUniteLegale'],
                            ['nomUniteLegale'])
    assert result['available_not_projected'] == ['prenomUsuelUniteLegale']
    assert not result['alternative_source_coverage_proven']


def test_missing_address_field_is_not_manufactured():
    result = field_coverage(['enseigne1Etablissement', 'dateDebut', 'dateFin'],
                            ['libelleVoieEtablissement'])
    assert result['projected_not_available'] == ['libelleVoieEtablissement']
