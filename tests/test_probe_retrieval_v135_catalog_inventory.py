import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.probe_retrieval_v135_catalog_inventory as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json


def setup(tmp_path,monkeypatch):
    launch=tmp_path/'source_launch.json';write_json(launch,dict(implementation={'core':record(workbench.ROOT/'src/xgb_matcher/retrieval_catalog_inventory_v135.py')}))
    for name,values,dtype in (('current',[1,3,5],np.uint64),('graph',[1,9],np.uint64),
        ('owner_key',[1,7,123],np.uint64),('subject_code',[2,2,1],np.uint8)):
        np.save(tmp_path/(name+'.npy'),np.array(values,dtype=dtype),allow_pickle=False)
    common=dict(launch=record(launch),**{k:False for k in workbench.PROTECTED})
    files={}
    ms=dict(membership=dict(status='SEALED_COMPLETE_NATIONAL_SITE_AND_PARENT_MEMBERSHIP',stats={'sites':3},artifacts={'endpoint_keys.npy':record(tmp_path/'current.npy')},**common),
        graph=dict(status='SEALED_V133_COMPLETE_ALL_RAW_OFFICIAL_SUCCESSION',stats={'endpoints':2},artifacts={'endpoint_keys.npy':record(tmp_path/'graph.npy')},**common),
        owners=dict(status='SEALED_V132_COMPLETE_ALL_OFFICIAL_NAME_OWNER_CSR',facts=3,buckets=[dict(facts=3,artifacts={key+'.npy':record(tmp_path/(key+'.npy')) for key in ('owner_key','subject_code')})],**common))
    for key,m in ms.items():
        p=tmp_path/(key+'.json');write_json(p,m);files[key]=record(p)
    history=tmp_path/'history.parquet';pq.write_table(pa.table({'siret':[f'{i:014d}' for i in (1,3,7,7)]}),history)
    cfg=tmp_path/'config.json';write_json(cfg,dict(**files,history_sites=record(history),expected_sites=3,expected_facts=3,
        expected_site_facts=2,expected_graph_endpoints=2,expected_history_rows=4,
        resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),max_output_bytes=1048576))
    monkeypatch.setattr(workbench,'CONFIG',cfg);output=tmp_path/'inventory';workbench.init(output);return output,history


def test_whole_sources_inventory_no_catalog_or_truth_change(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);workbench.worker(output);m=json.loads((output/'manifest.json').read_text())
    assert m['stats']['sources']['history_sites']['references']==4
    assert m['stats']['sources']['official_name_sites']['references']==2
    assert m['stats']['union_unique_absent_siret']==2 and np.load(output/'all_absent_siret.npy').tolist()==[7,9]
    assert not m['stats']['candidate_catalog_changed'] and not m['stats']['human_denominator_filtered']
    assert all(m[k] is False for k in workbench.PROTECTED)
    with pytest.raises(ValueError,match='already started'):workbench.worker(output)
    with pytest.raises(FileExistsError):workbench.run(output)


def test_changed_historical_source_stops_without_retry(tmp_path,monkeypatch):
    output,history=setup(tmp_path,monkeypatch);pq.write_table(pa.table({'siret':['123']}),history)
    with pytest.raises(ValueError,match='Authority drift'):workbench.worker(output)
    assert json.loads((output/'worker_stop.json').read_text())['status']=='STOP'
    with pytest.raises(ValueError,match='already started'):workbench.worker(output)


def test_protected_source_cannot_open_fold(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);launch=json.loads((output/'launch.json').read_text());cfg=launch['config']
    p=tmp_path/'bad_graph.json';m=json.loads(workbench.verify(cfg['graph']).read_text());m['fold1_opened']=True;write_json(p,m)
    cfg['graph']=record(p)
    with pytest.raises(ValueError,match='Protected'):workbench.sources(cfg)
