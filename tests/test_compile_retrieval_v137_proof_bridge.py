import json

import numpy as np
import pytest

import scripts.compile_retrieval_v136_extended_catalog as catalog_workbench
import scripts.compile_retrieval_v137_proof_bridge as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_compile_retrieval_v136_extended_catalog import setup as catalog_setup


def setup(tmp_path,monkeypatch):
    cat,_=catalog_setup(tmp_path,monkeypatch);catalog_workbench.worker(cat)
    launch=tmp_path/'proof_source_launch.json';write_json(launch,dict(implementation={'core':record(workbench.ROOT/'src/xgb_matcher/retrieval_proof_bridge_v137.py')}))
    common=dict(launch=record(launch),complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in workbench.PROTECTED})
    owners=[]
    for i in range(2):
        folder=tmp_path/f'owners{i}';folder.mkdir()
        arrays=dict(parent_key=np.array([1,9,7],dtype=np.uint32),owner_key=np.array([1,900001,7],dtype=np.uint64),
            subject_code=np.array([1,2,1],dtype=np.uint8),scope_code=np.array([1,3,2],dtype=np.uint8),
            current_exact=np.array([True,False,True]),name_fact_offsets=np.array([0,2,3],dtype=np.uint64))
        for k,a in arrays.items():np.save(folder/(k+'.npy'),a,allow_pickle=False)
        owners.append(dict(bucket=i,name_start=2*i,fact_start=3*i,names=2,facts=3,artifacts={p.name:record(p) for p in folder.glob('*.npy')}))
    om=tmp_path/'owner_manifest.json';write_json(om,dict(status=workbench.STATUSES['owners'],names=4,facts=6,buckets=owners,**common))
    folder=tmp_path/'raw_graph';folder.mkdir()
    for k,a in dict(endpoint_keys=np.array([100001,200001,900001],dtype=np.uint64),
        predecessor_keys=np.array([100001,200001,900001],dtype=np.uint64),
        successor_keys=np.array([200001,900001,900001],dtype=np.uint64),continuity_codes=np.array([2,1,0],dtype=np.uint8)).items():
        np.save(folder/(k+'.npy'),a,allow_pickle=False)
    gm=tmp_path/'graph_manifest.json';write_json(gm,dict(status=workbench.STATUSES['graph'],stats=dict(endpoints=3,observations=3),
        artifacts={p.name:record(p) for p in folder.glob('*.npy')},**common))
    cfg=tmp_path/'bridge_config.json';write_json(cfg,dict(catalog=record(cat/'manifest.json'),owners=record(om),graph=record(gm),
        expected_sites=8,expected_parents=5,expected_names=4,expected_facts=6,expected_site_facts=2,expected_owner_buckets=2,
        expected_graph_endpoints=3,expected_graph_observations=3,owner_group_size=1,numpy_version=np.__version__,
        resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),controller_seconds=1200,
        max_phase_output_bytes=1048576,max_total_output_bytes=16777216))
    monkeypatch.setattr(workbench,'CONFIG',cfg);output=tmp_path/'proof_bridge';workbench.init(output);return output


def finish(output):
    for i in range(2):
        workbench.owner_phase(output,i).mkdir(parents=True);workbench.worker(output,'owners',i)
    (output/'graph').mkdir();workbench.worker(output,'graph',None)


def test_seal_requires_all_facts_and_graph_and_preserves_closed_folds(tmp_path,monkeypatch):
    output=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal(output)
    finish(output);workbench.seal(output);m=json.loads((output/'manifest.json').read_text())
    assert m['stats']['facts']==6 and m['stats']['unbound_parent_facts']==2
    assert m['stats']['site_facts']==2 and len(m['buckets'])==2
    assert all(m[k] is False for k in workbench.PROTECTED) and not m['complete_retrieval_index'] and not m['ready_for_fold1']
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(output)
    with pytest.raises(ValueError,match='already started'):workbench.worker(output,'owners',0)
    with pytest.raises(FileExistsError):workbench.init(output)


def test_any_phase_protected_fold_flag_stops_seal(tmp_path,monkeypatch):
    output=setup(tmp_path,monkeypatch);finish(output)
    p=output/'graph/manifest.json';m=json.loads(p.read_text());m['fold1_opened']=True;write_json(p,m)
    with pytest.raises(ValueError,match='Protected'):workbench.seal(output)


def test_any_output_array_drift_stops_seal(tmp_path,monkeypatch):
    output=setup(tmp_path,monkeypatch);finish(output)
    p=output/'owners/group000/bucket000/fact_site_positions.npy';np.save(p,np.zeros(3,dtype=np.uint32),allow_pickle=False)
    with pytest.raises(ValueError,match='Authority drift'):workbench.seal(output)


def test_version_and_invalid_group_guard(tmp_path,monkeypatch):
    output=setup(tmp_path,monkeypatch)
    for bad in (-1,True,2,None):
        with pytest.raises(ValueError,match='owner phase'):workbench.worker(output,'owners',bad)
    monkeypatch.setattr(workbench.np,'__version__','bad')
    with pytest.raises(ValueError,match='library drift'):workbench.load(output)
