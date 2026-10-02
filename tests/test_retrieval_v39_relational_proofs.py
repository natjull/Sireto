import pytest

from scripts.retrieval_v39_relational_proofs import bodacc_transfer, deduplicate, membership, rne_role, site_name_binding_allowed, succession


def succession_row():
    return dict(relation_id='official-proof', predecessor_siret='11111111100001', predecessor_siren='111111111',
                successor_siret='22222222200002', successor_siren='222222222', economic_continuity=True,
                effective_date='2020-01-01', observed_at='2026-01-01')


def test_succession_preserves_both_directions_and_dates():
    edge = succession(succession_row()); paths = edge.bindings()
    assert len(paths) == 2 and paths[0]['anchor_id'] == paths[1]['target_id']
    assert edge.reported_date == '2020-01-01' and edge.observed_at == '2026-01-01'
    assert all('SITE_COMPATIBILITY' in p['required_evidence'] for p in paths)


@pytest.mark.parametrize('value', [False, None, '', 'true', 1])
def test_no_continuity_is_not_a_positive_relation(value):
    row = succession_row(); row['economic_continuity'] = value
    assert succession(row) is None


def test_asset_transfer_is_not_identity_or_automatic_all_site_expansion():
    row = dict(source='BODACC', relation_type='ASSET_TRANSFER', from_kind='SIREN', to_kind='SIREN',
               from_identifier='111111111', to_identifier='222222222', relation_id='b1', source_record_id='event:predecessor:0',
               effective_date='2021-01-01')
    edge = bodacc_transfer(row)
    assert edge.date_semantics == 'REPORTED_DATE_EFFECTIVITY_UNVERIFIED'
    assert all(p['target_kind'] == 'SIREN' and 'EVENT_SITE_BINDING' in p['required_evidence'] for p in edge.bindings())
    row['relation_type'] = 'UNKNOWN'
    with pytest.raises(ValueError): bodacc_transfer(row)


def test_rne_role_preserved_as_context_without_alias_propagation():
    row = dict(source='RNE', role_id='r1', source_record_id='official-rne', subject_siren='111111111',
               related_siren='222222222', role_code='99', role_label='Commissaire aux comptes')
    edge = rne_role(row)
    assert not edge.bindings() and edge.role_label == row['role_label']
    assert edge.relation_type == 'LEGAL_ENTITY_ROLE'


def test_membership_does_not_guess_parent_from_identifier_prefix():
    row = dict(siret='11111111100001', proof_id='s1', source_record_id='stock')
    with pytest.raises(KeyError): membership(row)
    row['siren'] = '111111111'
    assert len(membership(row).bindings()) == 1


def test_site_trade_name_does_not_spread_to_siblings_or_related_company():
    name = dict(subject_kind='SIRET', siren='111111111', siret='11111111100001')
    assert site_name_binding_allowed(name, '11111111100001', '111111111')
    assert not site_name_binding_allowed(name, '11111111100002', '111111111')
    name.update(subject_kind='SIREN')
    assert site_name_binding_allowed(name, '11111111100002', '111111111')
    assert not site_name_binding_allowed(name, '22222222200001', '222222222')


def test_missing_provenance_and_invalid_identifier_fail_closed():
    row = succession_row(); row['relation_id'] = ''
    with pytest.raises(ValueError, match='provenance'): succession(row)
    row = succession_row(); row['successor_siret'] = 'not-an-id'
    with pytest.raises(ValueError, match='identifier'): succession(row)


def test_duplicate_proofs_do_not_multiply_support_and_conflicts_fail():
    row = succession_row(); a = succession(row)
    assert len(deduplicate([a, a])) == 1
    row['effective_date'] = '2021-01-01'; b = succession(row)
    with pytest.raises(ValueError, match='Conflicting'): deduplicate([a, b])
