import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_address_dictionary_v145 import partition,dictionary
from src.xgb_matcher.retrieval_address_reverse_v147 import stage,pack,ReversePart,AliasSnapshot,STAGE_SCHEMA
from src.xgb_matcher.retrieval_address_projection_v142 import project
from test_retrieval_address_dictionary_v145 import projected
from test_retrieval_address_projection_v142 import Keys,canonical


def dictionary_source(tmp_path):
    p,t=projected(tmp_path);d=tmp_path/'source_stage';d.mkdir();c=duckdb.connect()
    s=partition(c,p,d,0,6,4,1,12,6);c.close()
    out=tmp_path/'source_dictionary';out.mkdir();c=duckdb.connect();dictionary(c,s['files'],out,0,1,4);c.close()
    return out/'facts.parquet'


def prepared(tmp_path):
    source=dictionary_source(tmp_path);out=tmp_path/'reverse_stage';out.mkdir();c=duckdb.connect()
    s=stage(c,source,out,Keys(),2,30,1,3,site_quantum=4);c.close()
    out=tmp_path/'packed';out.mkdir();c=duckdb.connect();p=pack(c,s['files'],out,8,12,12,31,3,batch_rows=1);c.close()
    return s,p,out


def snapshot(n,positive,exact=False):
    coverage=np.zeros(n);flags=np.zeros(n,dtype=bool)
    coverage[positive]=1.;flags[positive]=exact
    coverage.setflags(write=False);flags.setflags(write=False)
    return AliasSnapshot(coverage,flags)


def test_all_dictionary_facts_reachable_by_original_site_and_attributes(tmp_path):
    s,p,out=prepared(tmp_path);r=ReversePart(p)
    assert s['facts']==p['facts']==3 and p['every_original_field_and_site_csr_exact']
    assert r.a['site_offsets'].tolist()==[0,2,3,3,3]
    q=snapshot(31,[30],True);batches=list(r.batches(np.arange(8,12,dtype=np.uint32),q,chunk=1))
    assert sum(len(b.site_positions) for b in batches)==3
    assert [int(b.site_positions[0]) for b in batches]==[8,8,9]
    assert {b.number_texts[0] for b in batches}=={'1','9'}
    assert [bool(b.current_exact[0]) for b in batches].count(True)==1
    assert r.a['source_bucket'].tolist()==[2,2,2]
    assert sorted(r.a['source_fact_id'].tolist())==[0,1,2]
    assert r.a['alias_position'].tolist()==[30,30,30]
    assert not p['national_source_coverage_certified_here'] and not p['recall_measured']
    r.close()


def test_inverse_subinterval_no_scan_or_inheritance_to_empty_sibling_sites(tmp_path):
    s,p,out=prepared(tmp_path);r=ReversePart(p);q=snapshot(31,[30])
    b=list(r.batches(np.array([9],dtype=np.uint32),q))
    assert len(b)==1 and b[0].site_positions.tolist()==[9] and b[0].number_texts==('1',)
    assert list(r.batches(np.array([10,11],dtype=np.uint32),q))==[]
    for ids in (np.array([8,10],dtype=np.uint32),np.array([7],dtype=np.uint32),np.array([12],dtype=np.uint32)):
        with pytest.raises(ValueError,match='interval'):list(r.batches(ids,q))
    r.close()


def test_every_2501_facts_same_site_linked_no_fanout_cap(tmp_path):
    rows=[dict(site_position=8,parent_position=0,alias_position=i,postcode_code=0,insee_code=4294967295,
        number_text='',current_exact=False,source_bucket=0,source_fact_id=i) for i in range(2501)]
    source=tmp_path/'facts.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=STAGE_SCHEMA),source,row_group_size=23)
    out=tmp_path/'out';out.mkdir();c=duckdb.connect();p=pack(c,[source],out,8,12,12,2501,2501,batch_rows=37);c.close()
    r=ReversePart(p);q=snapshot(2501,list(range(2501)));b=list(r.batches(np.array([8],dtype=np.uint32),q,chunk=17))
    assert sum(len(v.site_positions) for v in b)==2501
    assert len(b)==148 and r.a['site_offsets'].tolist()==[0,2501,2501,2501,2501]
    assert r.a['source_fact_id'].tolist()==list(range(2501))
    assert p['site_or_proof_or_alias_cap'] is None
    r.close()


def test_number_dictionary_is_not_normalized_twice(tmp_path):
    source=tmp_path/'projection.parquet'
    row=dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',source='RNE',raw_value='RUE A',street_number='①②③④⑤⑥⑦⑧⑨')
    pq.write_table(project(canonical([row]),'official_addresses',Keys(),0,0),source)
    out=tmp_path/'part';out.mkdir();c=duckdb.connect();s=partition(c,source,out,0,1,1,1,12,6);c.close()
    out=tmp_path/'dict';out.mkdir();c=duckdb.connect();dictionary(c,s['files'],out,0,1,1);c.close()
    staged=tmp_path/'staged';staged.mkdir();c=duckdb.connect();s=stage(c,out/'facts.parquet',staged,Keys(),0,0,1,1);c.close()
    dest=tmp_path/'packed';dest.mkdir();c=duckdb.connect();p=pack(c,s['files'],dest,0,12,12,1,1);c.close()
    r=ReversePart(p);b=list(r.batches(np.array([8],dtype=np.uint32),snapshot(1,[0])))
    assert b[0].number_texts==('123456789',)
    r.close()


def test_empty_interval_has_complete_empty_csr(tmp_path):
    out=tmp_path/'out';out.mkdir();c=duckdb.connect();p=pack(c,[],out,8,12,12,31,0);c.close()
    r=ReversePart(p)
    assert r.a['site_offsets'].tolist()==[0]*5
    assert list(r.batches(np.arange(8,12,dtype=np.uint32),snapshot(31,[])))==[]
    r.close()


def test_original_site_parent_drift_is_stop_never_context_demoted(tmp_path):
    p=dictionary_source(tmp_path);t=pq.read_table(p);parents=t['parent_position'].to_pylist();parents[0]=1
    pq.write_table(t.set_column(t.schema.get_field_index('parent_position'),'parent_position',pa.array(parents,type=pa.uint32())),p)
    out=tmp_path/'out';out.mkdir();c=duckdb.connect()
    with pytest.raises(ValueError,match='site/parent'):stage(c,p,out,Keys(),0,0,1,3)
    c.close()


def test_duplicate_provenance_missing_part_or_wrong_alias_range_stops(tmp_path):
    source=dictionary_source(tmp_path);out=tmp_path/'stage';out.mkdir();c=duckdb.connect()
    s=stage(c,source,out,Keys(),0,0,1,3);c.close();destination=tmp_path/'pack';destination.mkdir()
    for files,aliases,expected in ((s['files']*2,1,6),(s['files'],0,3),(s['files'],1,2)):
        c=duckdb.connect()
        with pytest.raises(ValueError,match='exactly once'):pack(c,files,destination,0,12,12,aliases,expected)
        c.close()


def test_corrupted_artifact_hash_is_stop_not_silently_loaded(tmp_path):
    s,p,out=prepared(tmp_path)
    a=np.load(out/'site_position.npy');a[0]=9;np.save(out/'site_position.npy',a)
    with pytest.raises(ValueError,match='Authority drift'):ReversePart(p)


def test_alias_scores_checked_once_readonly_exact_coherence():
    c=np.array([.5,1.]);e=np.array([False,True])
    with pytest.raises(ValueError,match='Readonly'):AliasSnapshot(c,e)
    c.setflags(write=False);e.setflags(write=False);s=AliasSnapshot(c,e)
    assert s.coverage is c and s.exact is e
    c=np.array([.5]);e=np.array([True]);c.setflags(write=False);e.setflags(write=False)
    with pytest.raises(ValueError,match='Readonly'):AliasSnapshot(c,e)


def test_reverse_access_feeds_coherent_initial100_with_chunk_parity(tmp_path):
    from types import SimpleNamespace
    from src.xgb_matcher.retrieval_initial_collector_v146 import NameSnapshot,search
    from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
    from test_retrieval_initial_collector_v146 import Attributes
    source=dictionary_source(tmp_path);d=tmp_path/'staged';d.mkdir();c=duckdb.connect()
    s=stage(c,source,d,Keys(),0,0,1,3,site_quantum=16);c.close()
    out=tmp_path/'out';out.mkdir();c=duckdb.connect();p=pack(c,s['files'],out,0,12,12,1,3);c.close()
    r=ReversePart(p);scores=snapshot(1,[0],True);keys=Keys()
    cat=SimpleNamespace(sites=12,parents=6,keys=keys,previous=SimpleNamespace(sites=12,parents=6),
        new={'new_site_parent_positions':np.array([],dtype=np.uint32)})
    raw=(keys.reconstruct(np.arange(12,dtype=np.uint32),'site')//100000).astype(np.uint32)
    parent_pos=keys.parent_positions(raw)
    names=NameSnapshot(cat,parent_pos,np.zeros(6),np.zeros(6,dtype=bool),np.zeros(12),np.zeros(12,dtype=bool),np.zeros(12))
    class Addresses:
        sites=12
        def batches(self,ids,q):return r.batches(ids,scores,chunk=1)
    w=np.zeros(30);w[1]=1.;w[3]=5.;w[7]=1.;w[5]=1.;q=Query(address='RUE A',postcode='75001',number='1')
    observed=[]
    for size in (1,4,12):
        rows,stats=search(q,w,cat,names,Attributes(12),Addresses(),chunk=size)
        observed.append(rows)
        assert stats['sites_completed']==12 and stats['address_facts']==3
    assert observed[0]==observed[1]==observed[2]
    assert [r['siret'] for r in observed[0]]==['00000000100003','00000000200009']
    r.close()


def test_completed_reverse_interval_is_immutable(tmp_path):
    s,p,out=prepared(tmp_path);c=duckdb.connect()
    with pytest.raises(ValueError,match='already started'):pack(c,s['files'],out,8,12,12,31,3)
    c.close()
