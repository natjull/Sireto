import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.compile_retrieval_v136_extended_catalog as workbench
from src.xgb_matcher.retrieval_extended_catalog_v136 import VirtualCatalog
from src.xgb_matcher.retrieval_national_membership_v96 import build as build_current
from src.xgb_matcher.retrieval_owner_blocks_v127 import build as build_bridge
from src.xgb_matcher.retrieval_v20_workbench import record,write_json


def setup(tmp_path,monkeypatch):
    source=tmp_path/'current.parquet';pq.write_table(pa.table(dict(site_key=pa.array([100001,300001,500001],type=pa.uint64()),
        parent_key=pa.array([1,3,5],type=pa.uint32()),administrative_state=['A','F',None])),source)
    cur=tmp_path/'current';cur.mkdir();stats=build_current(source,cur)
    source_launch=tmp_path/'source_launch.json';write_json(source_launch,dict(implementation={'core':record(workbench.ROOT/'src/xgb_matcher/retrieval_extended_catalog_v136.py')}))
    common=dict(launch=record(source_launch),resources={'pass_resource_policy':True},**{k:False for k in workbench.PROTECTED})
    membership=tmp_path/'membership.json';write_json(membership,dict(status='SEALED_COMPLETE_NATIONAL_SITE_AND_PARENT_MEMBERSHIP',stats=stats,
        artifacts={p.name:record(p) for p in cur.glob('*.npy')},**common));mr=record(membership)
    layout=tmp_path/'layout';layout.mkdir();np.save(layout/'block_sites.npy',np.array([0,1,2],dtype=np.uint32),allow_pickle=False)
    np.save(layout/'block_offsets.npy',np.array([0,2,3],dtype=np.uint64),allow_pickle=False)
    lm=tmp_path/'layout.json';write_json(lm,dict(status='SEALED_ALL_NATIONAL_SITE_PHYSICAL_LAYOUT_ONLY',membership=mr,
        artifacts={p.name:record(p) for p in layout.glob('*.npy')},**common));lr=record(lm)
    bridge=tmp_path/'bridge';bridge.mkdir();bs=build_bridge(np.load(cur/'endpoint_parents.npy'),np.load(cur/'parent_keys.npy'),
        np.load(layout/'block_sites.npy'),np.load(layout/'block_offsets.npy'),bridge)
    bm=tmp_path/'bridge.json';write_json(bm,dict(status='SEALED_COMPLETE_NATIONAL_PARENT_BLOCK_BRIDGE_ONLY',membership=mr,layout=lr,stats=bs,
        artifacts={p.name:record(p) for p in bridge.glob('*.npy')},input_arrays={key:record(layout/(key+'.npy')) for key in ('block_sites','block_offsets')},**common));br=record(bm)
    extras=tmp_path/'all_absent_siret.npy';np.save(extras,np.array([100002,200001,200002,900001,900002],dtype=np.uint64),allow_pickle=False)
    inv=tmp_path/'inventory.json';write_json(inv,dict(status='SEALED_V135_COMPLETE_OFFICIAL_SITE_REFERENCE_INVENTORY',sources={'membership':mr},
        stats=dict(union_unique_absent_siret=5,candidate_catalog_changed=False,human_denominator_filtered=False),
        artifacts={'all_absent_siret.npy':record(extras)},**common))
    cfg=tmp_path/'config.json';write_json(cfg,dict(membership=mr,layout=lr,bridge=br,inventory=record(inv),expected_current_sites=3,
        expected_current_parents=3,expected_current_blocks=2,expected_extra_sites=5,block_size=2,numpy_version=np.__version__,
        resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),max_output_bytes=1048576))
    monkeypatch.setattr(workbench,'CONFIG',cfg);output=tmp_path/'catalog';workbench.init(output);return output,cur


def test_readonly_namespace_serves_all_old_and_extra_members_and_blocks(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);workbench.worker(output);m=json.loads((output/'manifest.json').read_text())
    assert m['stats']['sites']==8 and m['complete_candidate_namespace'] and not m['complete_retrieval_index']
    assert all(m[k] is False for k in workbench.PROTECTED) and not m['ready_for_fold1']
    catalog=VirtualCatalog(record(output/'manifest.json'))
    assert catalog.endpoint_position('00000000100001')==0 and catalog.endpoint_position('00000000900002')==7
    assert catalog.endpoint_position('00000000400001') is None and catalog.parent_position('000000002')==3
    assert catalog.members_for_parent(0).tolist()==[0,3] and catalog.blocks_for_parent(0).tolist()==[0,2]
    assert catalog.members_for_parent(3).tolist()==[4,5] and catalog.blocks_for_parent(3).tolist()==[2,3]
    assert catalog.members_for_parent(2).tolist()==[2] and catalog.blocks_for_parent(2).tolist()==[1]
    assert catalog.sites_in_block(0).tolist()==[0,1] and catalog.sites_in_block(2).tolist()==[3,4] and catalog.sites_in_block(4).tolist()==[7]
    assert [catalog.site_key(i) for i in range(8)]==[100001,300001,500001,100002,200001,200002,900001,900002]
    assert catalog.site_parent_position(1)==1 and catalog.site_parent_position(7)==4
    assert catalog.state_code(1)==2 and all(catalog.state_code(i)==0 for i in range(3,8))
    assert catalog.parent_key(3)==2
    with pytest.raises(ValueError,match='position'):catalog.site_key(True)
    with pytest.raises(ValueError,match='position'):catalog.sites_in_block(5)
    with pytest.raises(ValueError,match='grain'):catalog.endpoint_position('１２３４５６７８９００００１')
    with pytest.raises(ValueError,match='already started'):workbench.worker(output)
    with pytest.raises(FileExistsError):workbench.run(output)


def test_changed_current_array_stops_and_never_overwrites_namespace(tmp_path,monkeypatch):
    output,cur=setup(tmp_path,monkeypatch);np.save(cur/'endpoint_states.npy',np.zeros(3,dtype=np.uint8),allow_pickle=False)
    with pytest.raises(ValueError,match='Authority drift'):workbench.worker(output)
    assert json.loads((output/'worker_stop.json').read_text())['status']=='STOP'
    with pytest.raises(ValueError,match='already started'):workbench.worker(output)


def test_protected_input_fold_cannot_be_opened(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);l=json.loads((output/'launch.json').read_text());cfg=l['config']
    m=json.loads(workbench.verify(cfg['inventory']).read_text());m['fold1_opened']=True;p=tmp_path/'bad_inventory.json';write_json(p,m);cfg['inventory']=record(p)
    with pytest.raises(ValueError,match='Protected'):workbench.sources(cfg)
