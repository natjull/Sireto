from types import SimpleNamespace
import numpy as np
import pytest,tantivy
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery,TantivyBackend
from src.xgb_matcher.retrieval_regional_abbreviation_v285 import residual,patterns,membership,component
from src.xgb_matcher.retrieval_proof_v56 import components
from src.xgb_matcher.retrieval_training_score_v231 import points
from test_retrieval_corroborated_initial_v217 import tiny_native

REGIONS=['ANCIENNE REGION','NOUVELLE REGION','REGION']

def test_all_overlaps_and_original_preserved():
 q='AZB ANCIENNE REGION';r,m=residual(q,REGIONS)
 assert r=='AZB' and m==['ANCIENNE REGION','REGION'] and q=='AZB ANCIENNE REGION'
 assert residual('AZB REGIONALE',REGIONS)==('AZB REGIONALE',[])

@pytest.mark.parametrize('q',['AZB','AB ANCIENNE REGION','AZB CONTEXTE ANCIENNE REGION','MAIRIE ANCIENNE REGION','123456789 ANCIENNE REGION','ABCDEFG ANCIENNE REGION'])
def test_no_embedded_or_old_stop_expansion(q):assert not patterns(q,REGIONS)

@pytest.mark.parametrize('alias',['AZIMUTH BRAVO','AZIMUTH BRAVO NOUVELLE REGION','NOUVELLE REGION AZIMUTH BRAVO','SA AZIMUTH DE BRAVO NOUVELLE REGION SARL'])
def test_source_only_whole_qualifier_boundaries(alias):assert membership('AZB ANCIENNE REGION',[alias],REGIONS)

def test_no_arbitrary_suffix_cross_alias_or_invented_short_regional_code():
 for aliases in (['AZIMUTH BRAVO AUTRE CHOSE'],['AZIMUTH BRAVO NOUVELLE REGION EXTRA'],['AZIMUTH','BRAVO NOUVELLE REGION'],['AZIMUTH BRAVO NR'],['AZIMUTH BRAVO NOUVELLE']):
  assert not membership('AZB ANCIENNE REGION',aliases,REGIONS)

def engine():
 b=tiny_native();w=b.index.writer()
 for i,name,place in [(11,'AZIMUTH BRAVO NOUVELLE REGION','11111'),(12,'AZIMUTH BRAVO EXTRA','11111'),(13,'AZIMUTH BRAVO NOUVELLE REGION','22222')]:
  w.add_document(tantivy.Document.from_dict(dict(document_type='siret',siret=f'100000000{i:05d}',siren='100000000',
   legal_names=name,legal_names_exact=name,insee=place,postcode=place,addresses='ROAD'),b.index.schema))
 w.commit();b.index.reload()
 return SimpleNamespace(_tantivy=tantivy,index=b.index,searcher=b.index.searcher(),_first=TantivyBackend._first)

def test_native_whole_alias_place_only_and_no_boost_without_known_place():
 b=engine();q=RetrievalQuery(name='AZB ANCIENNE REGION',postcode='11111');p=component(b,q,components(b,q),REGIONS)
 assert np.allclose(points(b,p,[f'100000000{i:05d}' for i in (11,12,13)]),[1.,0.,0.])
 q=RetrievalQuery(name='AZB ANCIENNE REGION');assert component(b,q,components(b,q),REGIONS) is None

def test_every_partition_retained_no_fanout_cap():
 assert len(patterns('AZBCDE ANCIENNE REGION',REGIONS))==31*3
