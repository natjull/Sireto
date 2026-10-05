"""Existing V52 mask, raw input immutable and native score unchanged."""
from copy import deepcopy
import pytest
from src.xgb_matcher import retrieval_identifier_surface_v329 as work
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_variants_v118 import normalized


def query():
    return dict(query_id='CONTROL',fold=3,crm_name='MAIRIE DE TEST',crm_number='14',crm_suffix='BIS',
        crm_street='RUE CENTRALE',crm_postcode='75001',crm_insee='75056')


def test_valid_text_exact_same_original_query_semantics():
    raw=query();before=deepcopy(raw)
    expected=RetrievalQuery(name=normalized(raw['crm_name']),address=normalized(' '.join(raw[k] for k in ('crm_number','crm_suffix','crm_street'))),
        number=raw['crm_number'],postcode=raw['crm_postcode'],insee=raw['crm_insee'])
    assert work.feature_query(raw)==expected and raw==before


@pytest.mark.parametrize('field',['crm_name','crm_street','crm_number'])
@pytest.mark.parametrize('identifier',['100000000','10000000012345'])
def test_mask_before_native_query_no_filter_or_alias_lookup(field,identifier):
    raw=query();raw[field]=identifier+' 001';before=deepcopy(raw)
    q=work.feature_query(raw)
    assert all(identifier not in getattr(q,k) for k in ('name','address','number'))
    assert raw==before and q.postcode==raw['crm_postcode'] and q.insee==raw['crm_insee']
    if field=='crm_name':assert q.name=='001'
