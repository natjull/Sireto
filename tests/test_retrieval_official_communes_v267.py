"""Source schema/control fixtures only, NOT real HTTP/exit/completeness proof."""
from copy import deepcopy
import inspect
import json

import pytest
from scripts import retrieval_v267_official_communes_snapshot as snapshot
from src.xgb_matcher.retrieval_memory_policy_v214 import LIMITS


def rows():
    return [dict(code=f'{i:05d}',nom='SOURCE_FIXTURE_NOT_BUSINESS',codesPostaux=['11111'],
        siren='100000000' if i%2 else None,codeEpci=None) for i in range(30000)]


def test_all_rows_preserved_unknown_parent_not_silently_removed():
    original=rows();got,stats=snapshot.parse(json.dumps(original))
    assert got==original and stats==dict(rows=30000,missing_siren=15000,missing_epci=30000,postal_memberships=30000)


@pytest.mark.parametrize('bad',['duplicate','short','bad_name','bad_code','bad_postcode','duplicate_postcode','extra_fields','bad_siren','bad_epci'])
def test_invalid_listing_stops_instead_of_dropping_or_guessing(bad):
    value=rows();r=value[-1]
    if bad=='duplicate':r['code']=value[0]['code']
    elif bad=='short':value.pop()
    elif bad=='bad_name':r['nom']=None
    elif bad=='bad_code':r['code']='1'
    elif bad=='bad_postcode':r['codesPostaux']=['1111']
    elif bad=='duplicate_postcode':r['codesPostaux']*=2
    elif bad=='extra_fields':r['unrequested_siret']='10000000000001'
    elif bad=='bad_siren':r['siren']='10000000000001'
    else:r['codeEpci']=123456789
    with pytest.raises(ValueError):snapshot.parse(json.dumps(value))


def test_fixed_all_zone_all_type_request_not_client_filter_or_secret():
    assert 'type=commune-actuelle,arrondissement-municipal' in snapshot.URL
    assert 'zone=metro,drom,com' in snapshot.URL
    assert 'limit=' not in snapshot.URL and 'codePostal=' not in snapshot.URL
    code=inspect.getsource(snapshot)
    assert 'human_ground_truth' not in code and 'read_table' not in code and 'Authorization' not in code
    assert snapshot.LIMITS==LIMITS and snapshot.FALSE['truth_opened'] is False


@pytest.mark.parametrize('kind',['live','stop'])
def test_refuse_source_root_without_actual_admitted_producer(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(snapshot,'OUTPUT',tmp_path)
    root=tmp_path/'run';root.mkdir()
    if kind=='stop':(root/'status.json').write_text('{"status":"STOP"}')
    with pytest.raises((ValueError,FileNotFoundError)):snapshot.sealed(root)
