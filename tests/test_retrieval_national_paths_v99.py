import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial import cKDTree

from tests.test_retrieval_proof_v56 import backend
from tests.test_retrieval_succession_csr_v84 import fixture as relation_fixture
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_national_paths_v99 import CompactWordPostings,FullRegistryWithRelations,NationalScopedPaths,search
from src.xgb_matcher.retrieval_national_membership_v96 import build as membership_build,NationalMembership
from src.xgb_matcher.retrieval_succession_csr_v84 import build as relation_build
from src.xgb_matcher.retrieval_scoped_paths_v90 import OwnerFacts,SuccessionCSR,literal_scores,NameStream,native_query
from src.xgb_matcher.retrieval_name_postings_v86 import WordPostings,build_owners
from src.xgb_matcher.retrieval_national_postings_v98 import prepare,word_bucket


def sources(tmp_path):
    facts=[dict(subject_code=1,owner_key=1234567,official_parent=1234567,scope_code=1,name_text='PORT DU CENTRE'),
        dict(subject_code=2,owner_key=123456700001,official_parent=1234567,scope_code=3,name_text='PHARMACIE CENTRE'),
        dict(subject_code=2,owner_key=123456700002,official_parent=1234567,scope_code=3,name_text='OTHER'),
        dict(subject_code=2,owner_key=123456700003,official_parent=1234567,scope_code=3,name_text='PHARMACIE CENTRE'),
        dict(subject_code=1,owner_key=555555555,official_parent=555555555,scope_code=1,name_text='PHARMACIE CENTRE'),
        dict(subject_code=1,owner_key=1234567,official_parent=1234567,scope_code=0,name_text='FORBIDDEN CONTEXT'),
        dict(subject_code=2,owner_key=123456700004,official_parent=1234567,scope_code=0,name_text=None),
        dict(subject_code=2,owner_key=99999999900001,official_parent=999999999,scope_code=3,name_text='PHARMACIE CENTRE')]
    root=tmp_path/'words'; root.mkdir(); source=root/'facts.parquet'; pq.write_table(pa.Table.from_pylist(facts),source)
    parts=[dict(path=str(source),facts=len(facts),fact_start=0)]; c=duckdb.connect(); posted=prepare(c,parts,root)
    posted['buckets']=[s for b in range(32) if (s:=word_bucket(c,root,b,len(facts))) is not None]; c.close(); build_owners(parts,root)
    gdir=tmp_path/'relations'; gdir.mkdir(); gs,edges=relation_fixture(gdir); relation_build(gs,gdir)
    ids=sorted({e[k] for e in edges for k in ('predecessor_siret','successor_siret')}|
        {'00123456700003','00123456700004','55555555500001'}|{f'{100000000+i}00001' for i in range(125)})
    rdir=tmp_path/'registry'; rdir.mkdir(); rs=rdir/'source.parquet'
    pq.write_table(pa.Table.from_pylist([dict(site_key=int(s),parent_key=int(s[:9]),administrative_state='F') for s in ids],
        schema=pa.schema([('site_key',pa.uint64()),('parent_key',pa.uint32()),('administrative_state',pa.string())])),rs)
    membership_build(rs,rdir); graph=FullRegistryWithRelations(NationalMembership(rdir),SuccessionCSR(gdir)); graph.validate_relation_membership()
    return posted,OwnerFacts(root),graph,facts,edges,ids


def test_compact_word_lookup_same_full_vocabulary_tuples_and_scores(tmp_path):
    posted,_,_,_,_,_=sources(tmp_path); old=WordPostings(posted); new=CompactWordPostings(posted)
    assert set(old.terms)==set(new.terms)
    assert all(old.terms[t]==new.terms[t] for t in old.terms)
    for q in [{'PHARMACIE':.7,'CENTRE':.3},{'PORT':1.},{'FORBIDDEN':1.},{'ABSENT':1.}]:
        a,ta=old.scores(q); b,tb=new.scores(q); assert np.array_equal(a,b) and ta==tb
    a,ta=literal_scores(old,'PHARMACIE CENTRE 111111111'); b,tb=literal_scores(new,'PHARMACIE CENTRE 111111111')
    assert np.array_equal(a,b) and ta==tb


@pytest.mark.parametrize('weak,available',[(False,True),(True,True),(False,False)])
def test_initial100_full_site_and_one_hop_metric_equal_independent_whole_corpus_oracle(tmp_path,weak,available):
    posted,owners,graph,facts,edges,ids=sources(tmp_path); words=CompactWordPostings(posted)
    q=RetrievalQuery(name='PHARMACIE CENTRE',address='12 RUE BLEUE',insee='75056'); w=np.ones(11)
    scores,_=literal_scores(words,q.name); expected={s:0. for s in ids}
    for i,f in enumerate(facts):
        if not scores[i]: continue
        anchors=([s for s in ids if s[:9]==f'{f["official_parent"]:09d}'] if f['subject_code']==1
                 else [s for s in ids if int(s)==f['owner_key']])
        for anchor in anchors:
            expected[anchor]=max(expected[anchor],float(scores[i]))
            for e in edges:
                grade={'CURRENT_ADDRESS_GEO_CONCORDANT_NOT_HISTORICAL_PROOF':1.,'CURRENT_LOCATION_DIFFERS_OR_GEO_DISAGREES':.5,
                    'CURRENT_SITE_BINDING_INCOMPLETE':.25}[e['site_binding']]
                for left,right in (('predecessor_siret','successor_siret'),('successor_siret','predecessor_siret')):
                    if e[left]==anchor: expected[e[right]]=max(expected[e[right]],float(scores[i])*grade)
    paths=NationalScopedPaths(owners,graph,scores); stream=NameStream(scores); actual={s:0. for s in ids}
    while not stream.exhausted:
        for row in stream.fetch(2):
            for target in paths.targets(row): actual[target['siret']]=max(actual[target['siret']],target['path_name_score'])
    assert actual==expected and {s:paths.best(s)['score'] for s in ids}==expected
    assert paths.best('00123456700003')['score']==1. and paths.best('00123456700003')['outside_succession_endpoint_scope']
    assert paths.best('00123456700002')['score']<1. # No sibling sign transfer.
    assert paths.best('99999999900001')['score']==0. # Observation retained but not fabricated into a registered site.
    native_root=tmp_path/'native'; native_root.mkdir()
    b=backend(native_root,[dict(siret=s,siren=s[:9],legal_names='OTHER',addresses='12 RUE BLEUE') for s in ids])
    keys=np.array(sorted(int(s) for s in ids),dtype=np.uint64); parents=(keys//100000).astype(np.uint32)
    xy=np.random.default_rng(99).uniform(0,1000,(len(ids),2)); xy[np.where(keys==123456700003)[0][0]]=[20.,30.]; tree=cKDTree(xy)
    g=dict(spatial_crs_compatible=available,ban_x_l93=20.,ban_y_l93=30.,ban_uncertainty_m=100.,ban_declared_weak=weak)
    parsed=native_query(b,q,w,[],0.)
    native={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(score) for score,loc in b.searcher.search(parsed,limit=len(ids),count=False).hits}
    spatial={f'{int(k):014d}':(1./(1.+float(np.sqrt(np.sum((xy[i]-[20.,30.])**2)))/100.))*(.5 if weak else 1.) if available else 0. for i,k in enumerate(keys)}
    full={s:max(native.get(s,0.),64.*expected[s]*spatial[s]) for s in ids}
    wanted=sorted((s for s in ids if full[s]>0),key=lambda s:(-full[s],s))[:100]
    output,trace=search(b,tree,keys,parents,words,owners,graph,q,g,w,[],0.)
    assert [r['siret'] for r in output]==wanted and all(r['score']==full[r['siret']] for r in output)
    assert trace['certified'] and trace['fixed_reservoir'] is None and len(output)<=100
    if available: assert output[0]['siret']=='00123456700003'
