import pytest

from src.xgb_matcher.retrieval_coupled_source_v122 import build_sites


def test_self_links_retained_in_ledger_not_turned_into_new_identity_or_alias():
    row=dict(siren='123456789',siret='12345678900001')
    names=[dict(name_id=0,name_text='ALPHA')]
    facts=[dict(fact_id=0,name_id=0,subject_kind='SIRET',siren=row['siren'],siret=row['siret'])]
    edge=dict(siretEtablissementPredecesseur=row['siret'],siretEtablissementSuccesseur=row['siret'],source_row_ordinal=44)
    sites,stats=build_sites([row],names,facts,[dict(fact_id=0,is_current=True)],edges=[edge])
    assert len(sites)==1 and len(sites[0].names)==1 and not sites[0].names[0].relation_hypothesis
    ledger=stats['succession_input_ledger']
    assert ledger['input_observations']==ledger['self_link_observations']==1
    assert ledger['self_link_context'][0]['source_row_ordinal']==44
    assert ledger['effective_observations']==0


def test_malformed_edge_still_stops_not_silently_repaired_or_dropped():
    with pytest.raises(ValueError,match='Invalid explicit'):
        build_sites([],[],[],[],edges=[dict(siretEtablissementPredecesseur='invalid',siretEtablissementSuccesseur='invalid')])
