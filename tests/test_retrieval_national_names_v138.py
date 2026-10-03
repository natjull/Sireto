import json

import duckdb
import numpy as np
import pytest

from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_national_names_v138 import NationalNames,references,scatter_positive
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED,MISSING,scatter_name_facts
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_v20_workbench import record,write_json,ROOT


def offsets(values):
    return np.concatenate((np.zeros(1,dtype=np.uint64),np.cumsum(values,dtype=np.uint64)))


def setup(tmp_path):
    names=['ALPHA BETA','ALPHA BETA 123456789','ALPHA GAMMA','COMMUNE DE PARIS','[ND]'];n=len(names)
    source=tmp_path/'source.json';write_json(source,dict(names=n))
    stage=tmp_path/'stage.json';write_json(stage,dict(names=n,source=record(source),**{k:False for k in PROTECTED}))
    launch=tmp_path/'launch.json';write_json(launch,dict(config=dict(term_buckets=2,duckdb_version=duckdb.__version__,numpy_version=np.__version__),
        implementation={'core':record(ROOT/'src/xgb_matcher/retrieval_national_names_v138.py')}))
    c=duckdb.connect();access=[];total_posts=total_terms=0
    for channel in range(4):
        tokens={}
        for i,name in enumerate(names):
            ts=channels(name)[channel] if channel<3 else {normalized(name)}-{''}
            for token in ts:tokens.setdefault(token,[]).append(i)
        for bucket in range(2):
            rows={t:ids for t,ids in tokens.items() if c.execute('SELECT hash(?) % 2',[t]).fetchone()[0]==bucket}
            terms=sorted(rows);raw=[t.encode('ascii') for t in terms];ids=[rows[t] for t in terms]
            folder=tmp_path/f'c{channel}b{bucket}';folder.mkdir()
            arrays=dict(token_data=np.frombuffer(b''.join(raw),dtype=np.uint8),token_offsets=offsets([len(t) for t in raw]),
                post_ids=np.array([i for values in ids for i in values],dtype=np.uint32),post_offsets=offsets([len(values) for values in ids]))
            for k,a in arrays.items():np.save(folder/(k+'.npy'),a,allow_pickle=False)
            total_terms+=len(terms);total_posts+=len(arrays['post_ids'])
            access.append(dict(channel=channel,bucket=bucket,postings=len(arrays['post_ids']),terms=len(terms),
                artifacts={p.name:record(p) for p in folder.glob('*.npy')}))
    c.close();path=tmp_path/'manifest.json';write_json(path,dict(status='SEALED_V134_COMPLETE_ALL_OFFICIAL_NAME_CHANNEL_ACCESS',
        names=n,postings=total_posts,terms=total_terms,access=access,launch=record(launch),source_stage=record(stage),
        complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))
    return record(path),names


def test_all_three_lexical_channels_max_whole_alias_exact_collisions(tmp_path):
    ref,names=setup(tmp_path);index=NationalNames(ref)
    for query in ('ALPHA BETA','ALPHA DELTA','COMMUNE DE PARIS','BETA GAMMA','[ND]','123456789','FÖÖ'):
        cov,exact,stats=index.coverage(query,chunk=2);q=channels(query)
        expected=np.array([max(len(a&b)/len(a) if a else 0. for a,b in zip(q,channels(name))) for name in names])
        expected_exact=np.array([bool(normalized(query)) and normalized(query)==normalized(name) for name in names])
        assert np.array_equal(cov,expected) and np.array_equal(exact,expected_exact)
        assert stats['whole_alias_max'] and stats['postings_or_alias_cap'] is None
    assert index.posting(3,'ALPHA BETA').tolist()==[0,1] # all raw alias collisions, not first only
    assert index.posting(0,'ZZZZZZZ').tolist()==[]
    with pytest.raises(ValueError):index.posting(True,'ALPHA')
    index.close()


def test_any_source_array_drift_is_rejected_at_open(tmp_path):
    ref,_=setup(tmp_path);m=json.loads(open(ref['path']).read());p=m['access'][0]['artifacts']['post_ids.npy']['path']
    np.save(p,np.array([MISSING],dtype=np.uint32),allow_pickle=False)
    with pytest.raises(ValueError,match='Authority drift'):NationalNames(ref)


def test_fold_and_library_drift_guard(tmp_path,monkeypatch):
    ref,_=setup(tmp_path);m=json.loads(open(ref['path']).read());m['fold0_opened']=True
    p=tmp_path/'bad.json';write_json(p,m)
    with pytest.raises(ValueError,match='Protected'):NationalNames(record(p))
    monkeypatch.setattr(duckdb,'__version__','bad')
    with pytest.raises(ValueError,match='library drift'):NationalNames(ref)


def test_sparse_scatter_exact_full_scalar_oracle_and_site_scope():
    rng=np.random.default_rng(2026);n=101;counts=rng.integers(1,37,n);off=offsets(counts);nf=int(off[-1])
    coverage=rng.choice(np.array([0.,.25,1./3,1.]),n);exact=rng.choice([False,True],n)
    subject=rng.integers(1,3,nf,dtype=np.uint8);current=rng.choice([False,True],nf)
    pp=rng.integers(0,41,nf,dtype=np.uint32);sp=rng.integers(0,71,nf,dtype=np.uint32)
    pp[(subject==1)&(rng.random(nf)<.2)]=MISSING;sp[subject==1]=MISSING
    source=dict(name_fact_offsets=off,subject_code=subject,current_exact=current)
    bridge=dict(fact_parent_positions=pp,fact_site_positions=sp)
    def out():return np.zeros(41),np.zeros(41,dtype=bool),np.zeros(71),np.zeros(71,dtype=bool)
    expected=out();actual=out()
    scatter_name_facts(coverage,exact,off,subject,current,pp,sp,*expected,batch_rows=13)
    stats=scatter_positive(coverage,exact,source,bridge,*actual,chunk=17)
    assert all(np.array_equal(a,b) for a,b in zip(actual,expected))
    assert stats['no_positive_reference_cap'] and stats['references_processed']==int(np.sum(counts[(coverage>0)|exact]))


def test_single_popular_alias_all_references_without_fanout_cap():
    n=10001;off=np.array([0,n,n+2],dtype=np.uint64);selected=np.array([0,1],dtype=np.int64)
    batches=list(references(off,selected,chunk=17));r=np.concatenate([x for x,a in batches]);a=np.concatenate([a for x,a in batches])
    assert np.array_equal(r,np.arange(n+2)) and np.array_equal(a,np.r_[np.zeros(n),[1,1]])
    assert max(len(x) for x,a in batches)<=17


def test_sparse_only_zero_proofs_skipped_exact_flag_not_lost():
    off=np.array([0,1,2,3],dtype=np.uint64);source=dict(name_fact_offsets=off,
        subject_code=np.array([1,2,2],dtype=np.uint8),current_exact=np.array([True,True,False]))
    bridge=dict(fact_parent_positions=np.zeros(3,dtype=np.uint32),fact_site_positions=np.array([MISSING,0,1],dtype=np.uint32))
    pc=np.zeros(1);pe=np.zeros(1,dtype=bool);sc=np.zeros(2);se=np.zeros(2,dtype=bool)
    stats=scatter_positive(np.zeros(3),np.array([False,True,True]),source,bridge,pc,pe,sc,se,1)
    assert stats['references_processed']==2 and se.tolist()==[True,False] and sc.tolist()==[0.,0.]
