from scripts.audit_retrieval_v58_raw_proofs import compare,eligible,coherent_pairs

def proof(**kw):
    return dict(siren='100000000',siret='10000000000001',subject_kind='SIRET',
        normalized_value='ALPHA BETA',source='RNE',name_kind='USUAL',**kw)

def test_raw_legal_unit_name_is_identity_but_parent_address_is_not_site():
    p=proof(); p.update(siret='',subject_kind='SIREN')
    assert eligible(p,'10000000000001','names')
    assert not eligible(p,'10000000000001','addresses')
    p['resolved_siret']='10000000000001'
    assert eligible(p,'10000000000001','addresses')
    assert not eligible(p,'10000000000002','addresses')

def test_sibling_names_cannot_be_used_as_target_site_alias():
    p=proof(); p['siret']='10000000000002'
    assert not eligible(p,'10000000000001','names')
    assert compare('ALPHA BETA',[p],'10000000000001','names')['maximum_single_alias_word_coverage']==0.

def test_raw_alias_keeps_bound_source_and_does_not_union_unrelated_names():
    a=proof(); b=proof(); a['normalized_value']='ALPHA'; b['normalized_value']='BETA'
    result=compare('ALPHA BETA',[a,b],'10000000000001','names')
    assert result['maximum_single_alias_word_coverage']==.5 and not result['exact_any_source']
    result=compare('ALPHA BETA',[a,b,proof()],'10000000000001','names')
    assert result['exact_any_source'] and result['exact_sources']==['RNE']
    assert result['not_a_native_score_or_recall'] and result['date_reference_unknown']

def test_identifiers_masked_and_foreign_parent_not_eligible():
    p=proof(); p['normalized_value']='ALPHA 123456789'
    assert compare('ALPHA',[p],'10000000000001','names')['exact_any_source']
    assert not eligible(p,'20000000000001','names')

def test_coherence_requires_same_official_observation_at_same_site_and_source():
    n=proof(); n.update(evidence_id='event',source_record_id='record',valid_from='2020-01-01',valid_to=None,observed_at=None)
    a=dict(n,normalized_value='12 RUE BLEUE')
    assert len(coherent_pairs([n],[a],'10000000000001'))==1
    assert not coherent_pairs([dict(n,evidence_id='')],[dict(a,evidence_id='')],'10000000000001')
    assert not coherent_pairs([n],[dict(a,source='BODACC')],'10000000000001')
    assert not coherent_pairs([n],[dict(a,siret='10000000000002')],'10000000000001')
    assert not coherent_pairs([dict(n,siret='',subject_kind='SIREN')],[a],'10000000000001')
