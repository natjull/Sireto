import pytest
from types import SimpleNamespace
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from scripts.retrieval_v279_municipal_structure import compile_db,validate_db,readonly,MunicipalIndex

def rows():
 return [
  dict(code='11111',nom='CITY ALPHA',codesPostaux=['10000','10001'],siren='100000001',codeEpci='200000001'),
  dict(code='11112',nom='CITY BETA',codesPostaux=['10000'],siren='100000002',codeEpci='200000001'),
  dict(code='22222',nom='CITY GAMMA',codesPostaux=[],siren=None,codeEpci=''),
 ]

def test_whole_roundtrip_unknowns_and_readonly(tmp_path):
 p=tmp_path/'source.sqlite';r=rows()
 assert compile_db(p,r)==dict(rows=3,missing_siren=1,missing_epci=1,postal_memberships=3)
 assert validate_db(p,r)['rows']==3
 with pytest.raises(ValueError):compile_db(p,r)
 c=readonly(p)
 with pytest.raises(Exception):c.execute('DELETE FROM communes')
 c.close()

def test_all_known_code_or_postal_joins_no_numeric_similarity_or_first(tmp_path):
 p=tmp_path/'source.sqlite';compile_db(p,rows())
 index=MunicipalIndex.__new__(MunicipalIndex);index.c=readonly(p);index.files=SimpleNamespace(guard=lambda:None,check_epochs=lambda:None)
 assert index.parents(RetrievalQuery(name='',postcode='10000'))==dict(commune={'100000001','100000002'},epci={'200000001'})
 assert index.parents(RetrievalQuery(name='',postcode='10001',insee='11112'))==dict(commune={'100000001','100000002'},epci={'200000001'})
 assert index.parents(RetrievalQuery(name='',postcode='10002',insee='22223'))==dict(commune=set(),epci=set())
 assert index.all_roles()==dict(commune={'100000001','100000002'},epci={'200000001'})
 index.close()

def test_any_dropped_changed_official_fact_fails_roundtrip(tmp_path):
 p=tmp_path/'source.sqlite';r=rows();compile_db(p,r)
 r[0]['nom']='RECONSTRUCTED'
 with pytest.raises(ValueError):validate_db(p,r)
