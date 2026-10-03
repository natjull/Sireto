import json

import numpy as np
import pyarrow.parquet as pq
import pytest

import scripts.probe_retrieval_v140_canonical_inventory as inventory_workbench
import scripts.compile_retrieval_v141_append_catalog as workbench
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys,AppendCatalog
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_probe_retrieval_v140_canonical_inventory import setup as inventory_setup,finish
from test_retrieval_canonical_inventory_v140 import rows,table


def setup(tmp_path,monkeypatch,new_parent=False):
    inv=inventory_setup(tmp_path,monkeypatch)
    if new_parent:
        l=inventory_workbench.load(inv);r=rows();r['siren'][2]='000000004';r['siret'][2]='00000000400001'
        p=tmp_path/'addresses_new_parent.parquet';pq.write_table(table(r),p,row_group_size=2)
        l['config']['addresses']=record(p);write_json(inv/'launch.json',l)
    finish(inv);inventory_workbench.seal(inv)
    m=json.loads((inv/'manifest.json').read_text());cfg=tmp_path/'append_config.json'
    write_json(cfg,dict(previous_catalog=m['catalog'],inventory=record(inv/'manifest.json'),
        expected_previous_sites=8,expected_previous_parents=5,expected_previous_blocks=5,expected_additional_sites=3,
        block_size=2,numpy_version=np.__version__,resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),
        max_output_bytes=1048576));monkeypatch.setattr(workbench,'CONFIG',cfg)
    out=tmp_path/'append';workbench.init(out);return out


def test_complete_append_reader_retains_previous_sites_parents_blocks_and_unknowns(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);workbench.worker(out);ref=record(out/'manifest.json');m=json.loads((out/'manifest.json').read_text())
    assert m['stats']['sites']==11 and m['stats']['parents']==5 and m['stats']['blocks']==7
    assert all(m[k] is False for k in workbench.PROTECTED) and not m['ready_for_fold1']
    c=AppendCatalog(ref);k=AppendKeys(ref)
    assert [c.site_key(i) for i in range(c.sites)]==[100001,300001,500001,100002,200001,200002,900001,900002,100003,200009,900003]
    assert [c.parent_key(i) for i in range(c.parents)]==[1,3,5,2,9]
    assert c.members_for_parent(0).tolist()==[0,3,8] and c.blocks_for_parent(0).tolist()==[0,2,5]
    assert c.members_for_parent(3).tolist()==[4,5,9] and c.blocks_for_parent(3).tolist()==[2,3,5]
    assert c.members_for_parent(4).tolist()==[6,7,10] and c.blocks_for_parent(4).tolist()==[3,4,6]
    assert c.members_for_parent(1).tolist()==[1]
    assert c.sites_in_block(0).tolist()==[0,1] and c.sites_in_block(4).tolist()==[7]
    assert c.sites_in_block(5).tolist()==[8,9] and c.sites_in_block(6).tolist()==[10]
    assert c.endpoint_position('00000000100003')==8 and c.endpoint_position('00000000900002')==7
    assert c.endpoint_position('00000000400001') is None and c.parent_position('000000002')==3
    assert [c.site_parent_position(i) for i in (1,8,9,10)]==[1,0,3,4]
    assert c.state_code(1)==2 and all(c.state_code(i)==0 for i in range(3,11))
    assert k.reconstruct(np.array([7,8,9],dtype=np.uint32),'site').tolist()==[900002,100003,200009]
    with pytest.raises(ValueError,match='position'):c.site_key(True)
    with pytest.raises(ValueError,match='position'):c.sites_in_block(7)
    with pytest.raises(ValueError,match='grain'):k.reconstruct(np.array([],dtype=np.uint32),'wrong')
    with pytest.raises(ValueError,match='position'):k.reconstruct(np.array([MISSING],dtype=np.uint32),'site')
    with pytest.raises(ValueError,match='already started'):workbench.worker(out)
    with pytest.raises(FileExistsError):workbench.run(out)


def test_changed_additional_array_stops_and_never_retries_or_mutates_old_namespace(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);l=workbench.load(out);m=json.loads(workbench.verify(l['config']['inventory']).read_text())
    p=m['all_absent_sites']['path'];np.save(p,np.array([900003],dtype=np.uint64),allow_pickle=False)
    with pytest.raises(ValueError,match='Authority drift'):workbench.worker(out)
    assert json.loads((out/'worker_stop.json').read_text())['status']=='STOP'
    with pytest.raises(ValueError,match='already started'):workbench.worker(out)


def test_new_parent_appended_after_all_previous_parents_and_singleton_blocks(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch,new_parent=True);workbench.worker(out);c=AppendCatalog(record(out/'manifest.json'))
    assert c.parents==6 and c.parent_key(5)==4 and c.parent_position('000000004')==5
    assert c.endpoint_position('00000000400001')==10 and c.site_parent_position(10)==5
    assert c.members_for_parent(5).tolist()==[10] and c.blocks_for_parent(5).tolist()==[6]
    assert c.sites_in_block(6).tolist()==[10] and c.state_code(10)==0
    assert c.members_for_parent(4).tolist()==[6,7]
    assert c.keys.reconstruct(np.array([3,4,5],dtype=np.uint32),'parent').tolist()==[2,9,4]


def test_closed_fold_or_lexical_filtered_authority_rejected(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);cfg=workbench.load(out)['config'];m=json.loads(workbench.verify(cfg['inventory']).read_text())
    m['fold1_opened']=True;p=tmp_path/'bad.json';write_json(p,m);cfg['inventory']=record(p)
    with pytest.raises(ValueError,match='Protected'):workbench.sources(cfg)
    m['fold1_opened']=False;m['unique_additional_absent_sites']=2;write_json(p,m);cfg['inventory']=record(p)
    with pytest.raises(ValueError,match='Protected'):workbench.sources(cfg)


MISSING=np.iinfo(np.uint32).max
