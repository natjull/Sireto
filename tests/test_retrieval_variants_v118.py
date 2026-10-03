import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_variants_v118 import (
    OBS_SCHEMA, factorize, normalized, observations,
)


def ul(**values):
    return dict(siren='123456789', statutDiffusionUniteLegale='O', **values)


def test_ei_names_use_all_available_same_row_fields_and_both_orders():
    rows = list(observations(ul(nomUniteLegale='Durand', nomUsageUniteLegale='Martin',
        prenomUsuelUniteLegale='Éva', prenom1UniteLegale='Eva', prenom2UniteLegale='Jeanne'),
        'current_ul', 7))
    texts = {r['name_text'] for r in rows}
    assert {'DURAND EVA', 'EVA DURAND', 'MARTIN JEANNE', 'JEANNE MARTIN',
            'DURAND EVA JEANNE', 'EVA JEANNE DURAND'} <= texts
    assert len([r for r in rows if r['name_text'] == 'DURAND EVA']) == 2
    assert all(r['source_row_ordinal'] == 7 for r in rows)
    assert all(r['subject_kind'] == 'SIREN' and r['siret'] == '' for r in rows)


def test_no_current_prenom_or_address_joined_to_history():
    rows = list(observations(ul(nomUniteLegale='Ancien', prenom1UniteLegale='Eva',
        dateDebut='1999-01-01', dateFin='2001-12-31'), 'history_ul', 0))
    assert [r['name_text'] for r in rows] == ['ANCIEN']
    assert rows[0]['valid_from'] == '1999-01-01'
    assert rows[0]['valid_to'] == '2001-12-31'
    assert not rows[0]['historical_address_claimed']


@pytest.mark.parametrize('status', ['P', 'N', None, ''])
def test_partial_or_unknown_diffusion_never_generates_personal_combinations(status):
    row = ul(nomUniteLegale='Durand', prenom1UniteLegale='Eva')
    row['statutDiffusionUniteLegale'] = status
    rows = list(observations(row, 'current_ul', 0))
    assert all(r['derivation'] != 'SAME_ROW_EI_CONCAT' for r in rows)
    assert rows[0]['siren'] == row['siren']


@pytest.mark.parametrize('value', ['[ND]', '[nd] DURAND', 'EVA [ ND ]'])
def test_masks_have_no_lexical_evidence(value):
    assert normalized(value) == ''


def test_identifier_removed_only_from_lexical_signal_not_raw_provenance():
    row = list(observations(ul(denominationUniteLegale='Commune 123456789'), 'current_ul', 0))[0]
    assert row['name_text'] == 'COMMUNE'
    assert row['raw_value'] == 'Commune 123456789'


def test_no_token_bag_created_across_variants():
    rows = list(observations(ul(denominationUniteLegale='Alpha',
                               denominationUsuelle1UniteLegale='Beta'), 'current_ul', 0))
    assert {r['name_text'] for r in rows} == {'ALPHA', 'BETA'}
    assert all(not {'ALPHA', 'BETA'} <= set(r['name_text'].split()) for r in rows)


def test_current_period_start_not_claimed_as_name_start_or_closure():
    row = list(observations(ul(denominationUniteLegale='Alpha', dateDebut='2020-01-01',
        etatAdministratifUniteLegale='F'), 'current_ul', 0))[0]
    assert row['valid_from'] is None and row['valid_to'] is None
    assert row['source_period_start'] == '2020-01-01'
    assert row['administrative_state'] == 'F' and row['lexical_eligible']


def test_empty_site_retained_as_context_not_dropped():
    row = list(observations(dict(siren='123456789', siret='12345678900001',
        etatAdministratifEtablissement='F'), 'current_site', 0))[0]
    assert row['derivation'] == 'CONTEXT_ROW' and not row['lexical_eligible']
    assert row['siret'] == '12345678900001'


def test_invalid_identity_retained_but_not_lexically_admitted():
    row = list(observations(dict(siren='123456789', siret='98765432100001',
        enseigne1Etablissement='Alpha'), 'history_site', 0))[0]
    assert row['raw_value'] == 'Alpha' and not row['lexical_eligible']


def test_official_scope_and_metadata_not_transferred_to_sibling():
    row = dict(siren='123456789', siret='12345678900001', subject_kind='SIRET', source='BODACC',
        name_kind='SIGN', raw_value='Café', normalized_value='CAFE', valid_from='2000-01-01',
        valid_to='2005-01-01', evidence_id='e', source_record_id='r', observed_at='2006-01-01',
        source_priority=200, is_current=False)
    result = list(observations(row, 'official_names', 42))[0]
    assert result['lexical_scope'] == 'EXPLICIT_ENDPOINT_NAME'
    assert result['evidence_id'] == 'e' and result['source_record_id'] == 'r'
    assert result['observed_at'] == '2006-01-01' and result['source_priority'] == 200
    assert result['original_normalized_value'] == 'CAFE'
    assert not result['related_target_alias']


def test_empty_official_raw_name_uses_available_normalized_name_but_mask_never_does():
    row = dict(siren='123456789', subject_kind='SIREN', source='RNE', name_kind='LEGAL',
               raw_value='', normalized_value='ALPHA')
    result = list(observations(row, 'official_names', 0))[0]
    assert result['name_text'] == 'ALPHA' and result['original_raw_value'] == ''
    row['raw_value'] = '[ND]'
    result = list(observations(row, 'official_names', 0))[0]
    assert result['name_text'] == '' and result['original_normalized_value'] == 'ALPHA'


@pytest.mark.parametrize('ordinal', [-1, True, 1.2])
def test_bad_ordinals_rejected(ordinal):
    with pytest.raises(ValueError):
        list(observations(ul(), 'current_ul', ordinal))


def test_factorization_retains_every_period_role_source_and_mask(tmp_path):
    rows = []
    for ordinal, date in enumerate(['2001-01-01', '2005-01-01', '2010-01-01']):
        rows.extend(observations(ul(nomUniteLegale='Alpha', dateDebut=date), 'history_ul', ordinal))
    rows.extend(observations(ul(denominationUniteLegale='Alpha',
        denominationUsuelle1UniteLegale='Alpha'), 'current_ul', 0))
    rows.extend(observations(dict(siren='123456789', siret='12345678900001',
        enseigne1Etablissement='Alpha', enseigne2Etablissement='[ND]'), 'current_site', 0))
    path = tmp_path / 'raw.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=OBS_SCHEMA), path)
    c = duckdb.connect()
    stats = factorize(c, path, tmp_path)
    assert stats['observations'] == 7 and stats['names'] == 1 and stats['facts'] == 3
    assert stats['max_observations_per_fact'] == 4 and stats['full_observation_roundtrip']
    result = pq.read_table(tmp_path / 'observations.parquet').to_pylist()
    assert [r['valid_from'] for r in result[:3]] == ['2001-01-01', '2005-01-01', '2010-01-01']
    assert result[-1]['fact_id'] is None and result[-1]['name_id'] is None
    assert result[-1]['raw_value'] == '[ND]'
    assert {r['name_id'] for r in result[:-1]} == {0}
    c.close()


def test_no_name_or_parent_fanout_cap(tmp_path):
    rows = [r for i in range(350) for r in observations(dict(siren='123456789',
        siret='123456789' + str(i).zfill(5), enseigne1Etablissement='Alpha'), 'current_site', i)]
    rows += [r for i in range(250) for r in observations(ul(denominationUniteLegale=f'Name {i}'), 'current_ul', i)]
    path = tmp_path / 'raw.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=OBS_SCHEMA), path)
    c = duckdb.connect(); stats = factorize(c, path, tmp_path)
    assert stats['facts'] == stats['observations'] == 600
    assert stats['names'] == 251 and stats['name_or_period_cap'] is None
    c.close()


def test_all_context_only_factorizes_without_fake_name(tmp_path):
    rows = list(observations(ul(), 'current_ul', 0))
    path = tmp_path / 'raw.parquet'; pq.write_table(pa.Table.from_pylist(rows, schema=OBS_SCHEMA), path)
    c = duckdb.connect(); stats = factorize(c, path, tmp_path)
    assert stats['names'] == stats['facts'] == 0 and stats['observations'] == 1
    c.close()
