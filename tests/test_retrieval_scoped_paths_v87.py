import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.spatial import cKDTree

from tests.test_retrieval_name_facts_v85 import fixture as names_fixture
from tests.test_retrieval_succession_csr_v84 import fixture as graph_fixture
from tests.test_retrieval_proof_v56 import backend
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_name_facts_v85 import factorize
from src.xgb_matcher.retrieval_name_postings_v86 import WordPostings,build_postings,build_owners
from src.xgb_matcher.retrieval_succession_csr_v84 import build,SuccessionCSR,SITE_CLASSES
from src.xgb_matcher.retrieval_scoped_paths_v87 import OwnerFacts,ScopedPaths,NameStream,NativeStream,literal_scores,search
from src.xgb_matcher.retrieval_sparse_native_v60 import query
from src.xgb_matcher.retrieval_proof_v56 import signal


def sources(tmp_path):
    nroot=tmp_path/'names'; nroot.mkdir(); source,raw_names=names_fixture(nroot)
    facts=nroot/'facts'; facts.mkdir(); words=nroot/'words'; words.mkdir()
    c=duckdb.connect(); stats=factorize(c,source,facts); c.close()
    c=duckdb.connect(); posted=build_postings(c,stats['parts'],words); c.close(); build_owners(stats['parts'],words)
    grows=tmp_path/'graph'; grows.mkdir(); gsource,edges=graph_fixture(grows); build(gsource,grows)
    corpus=[]
    for part in stats['parts']: corpus.extend(pq.read_table(part['path']).to_pylist())
    return WordPostings(posted),OwnerFacts(words),SuccessionCSR(grows),corpus,edges


def test_sorted_and_all_incoming_scoped_paths_equal_independent_full_edge_oracle(tmp_path):
    index,owners,graph,facts,edges=sources(tmp_path)
    scores,_=literal_scores(index,'PHARMACIE CENTRE 00123456700001')
    # Independent expansion from original source tuples, not adapter callbacks.
    expected={f'{int(k):014d}':0. for k in graph.a['endpoint_keys']}
    for i,fact in enumerate(facts):
        if not scores[i]: continue
        if fact['subject_code']==1:
            anchors=[s for s in expected if s[:9]==f'{fact["official_parent"]:09d}']
        else: anchors=[f'{fact["owner_key"]:014d}']
        for anchor in anchors:
            expected[anchor]=max(expected[anchor],scores[i])
            for edge in edges:
                for source,target in (('predecessor_siret','successor_siret'),('successor_siret','predecessor_siret')):
                    if edge[source]==anchor:
                        grade={list(SITE_CLASSES)[0]:1.,list(SITE_CLASSES)[1]:.5,list(SITE_CLASSES)[2]:.25}[edge['site_binding']]
                        expected[edge[target]]=max(expected[edge[target]],scores[i]*grade)
    paths=ScopedPaths(owners,graph,scores); stream=NameStream(scores); emitted={s:0. for s in expected}
    while not stream.exhausted:
        for row in stream.fetch(2):
            for target in paths.targets(row): emitted[target['siret']]=max(emitted[target['siret']],target['path_name_score'])
    assert emitted==expected
    assert {s:paths.best(s)['score'] for s in expected}==expected
    assert paths.best('00123456700001')['score']==1.
    assert paths.best('00123456700002')['score']<1. # Own PHARMACIE does not become sibling alias.
    assert paths.best('00123456799999')['score']==0. # No hidden unexpanded sibling proof.
    assert paths.best('22222222200001')['historical_site_coexistence_proven'] is False


def test_context_only_words_and_identifier_only_input_produce_no_name_proof(tmp_path):
    index,owners,graph,_,_=sources(tmp_path)
    scores,_=literal_scores(index,'NEVER A SITE ALIAS')
    assert not scores.any()
    scores,_=literal_scores(index,'00123456700001 222222222')
    assert not scores.any() and NameStream(scores).exhausted


def test_native_sorted_random_scores_are_identical_with_pagination(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA' if i%3 else 'BETA',addresses='12 RUE BLEUE') for i in range(300)])
    q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',insee='75056'); w=np.arange(11)*.1
    parsed=query(b,q,w,[('commune',1.)],2.); stream=NativeStream(b,parsed)
    ids=[f'{100000000+i}00001' for i in range(300)]; scores=stream.complete(ids)
    seen=[]
    while not stream.exhausted:
        rows=stream.fetch(31); seen.extend(rows)
        assert all(r['score']==scores[r['siret']] for r in rows)
    assert len(seen)==300


@pytest.mark.parametrize('weak,available',[(False,True),(True,True),(False,False)])
def test_real_adapters_initial100_match_full_native_and_edge_oracle(tmp_path,weak,available):
    index,owners,graph,facts,edges=sources(tmp_path)
    ids=[f'{int(k):014d}' for k in graph.a['endpoint_keys']]+['00123456799999']+[f'{100000000+i}00001' for i in range(130)]
    rows=[dict(siret=s,siren=s[:9],legal_names='PHARMACIE' if i%3 else 'OTHER',addresses='12 RUE BLEUE') for i,s in enumerate(ids)]
    b=backend(tmp_path,rows); q=RetrievalQuery(name='PHARMACIE CENTRE',address='12 RUE BLEUE',insee='75056')
    keys=np.array(sorted(int(s) for s in ids),dtype=np.uint64); parents=(keys//100000).astype(np.uint32)
    xy=np.random.default_rng(87).uniform(0,1000,(len(ids),2)); tree=cKDTree(xy)
    g=dict(spatial_crs_compatible=available,ban_x_l93=20.,ban_y_l93=30.,ban_uncertainty_m=100.,ban_declared_weak=weak)
    w=np.ones(11); parsed=query(b,q,w,[],0.)
    native={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(score) for score,loc in b.searcher.search(parsed,limit=len(ids),count=False).hits}
    ns,_=literal_scores(index,q.name); expected_name={s:0. for s in ids}
    for i,f in enumerate(facts):
        if not ns[i]: continue
        anchors=[s for s in ids if s in {f'{int(k):014d}' for k in graph.a['endpoint_keys']} and (
                 s[:9]==f'{f["official_parent"]:09d}' if f['subject_code']==1 else s==f'{f["owner_key"]:014d}')]
        for anchor in anchors:
            expected_name[anchor]=max(expected_name[anchor],float(ns[i]))
            for e in edges:
                grade={list(SITE_CLASSES)[0]:1.,list(SITE_CLASSES)[1]:.5,list(SITE_CLASSES)[2]:.25}[e['site_binding']]
                for left,right in (('predecessor_siret','successor_siret'),('successor_siret','predecessor_siret')):
                    if e[left]==anchor: expected_name[e[right]]=max(expected_name[e[right]],float(ns[i])*grade)
    spatial={f'{int(k):014d}':(1./(1.+float(np.sqrt(np.sum((xy[i]-[20.,30.])**2)))/100.))*(.5 if weak else 1.) if available else 0.
             for i,k in enumerate(keys)}
    scores={s:max(native.get(s,0.),64.*expected_name[s]*spatial[s]) for s in ids}
    expected=sorted((s for s in ids if scores[s]>0),key=lambda s:(-scores[s],s))[:100]
    actual,trace=search(b,tree,keys,parents,index,owners,graph,q,g,w,[],0.)
    assert [r['siret'] for r in actual]==expected
    assert all(r['score']==scores[r['siret']] for r in actual)
    assert trace['certified'] and trace['fixed_reservoir'] is None and len(actual)<=100
