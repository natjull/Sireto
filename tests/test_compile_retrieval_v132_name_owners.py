import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.compile_retrieval_v130_variant_dictionary as dictionary
import scripts.compile_retrieval_v132_name_owners as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_compile_retrieval_v130_variant_dictionary import setup as dictionary_setup,stage_all


def setup(tmp_path,monkeypatch):
    source,_=dictionary_setup(tmp_path,monkeypatch);stage_all(source)
    for b in range(2):
        dictionary.bucket_dir(source,b).mkdir(parents=True);dictionary.worker(source,'bucket',b)
    dictionary.seal_dictionary(source);m=json.loads((source/'manifest.json').read_text())
    cfg=tmp_path/'owner_config.json';write_json(cfg,dict(source_manifest=record(source/'manifest.json'),
        expected_names=m['names'],expected_facts=m['facts'],expected_observations=m['original_observations'],
        resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),
        max_bucket_output_bytes=1048576,max_total_output_bytes=16777216))
    monkeypatch.setattr(workbench,'CONFIG',cfg);output=tmp_path/'owners';workbench.init(output)
    return output,m


def test_only_complete_all_owner_buckets_can_be_sealed(tmp_path,monkeypatch):
    output,source=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='UNFINISHED_ALL_OWNER_BUCKETS'):workbench.seal(output)
    for b in range(2):
        workbench.bucket_dir(output,b).mkdir(parents=True);workbench.worker(output,b)
    workbench.seal(output);m=json.loads((output/'manifest.json').read_text())
    assert m['names']==source['names'] and m['facts']==source['facts'] and len(m['buckets'])==2
    assert all(m[k] is False for k in workbench.PROTECTED)
    assert m['complete_retrieval_index'] is False and m['ready_for_fold1'] is False
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(output)
    with pytest.raises(ValueError,match='already started'):workbench.worker(output,0)
    with pytest.raises(FileExistsError):workbench.execute(output,0)


def test_protected_fold_guard(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);workbench.bucket_dir(output,0).mkdir(parents=True);workbench.worker(output,0)
    path=workbench.bucket_dir(output,0)/'manifest.json';m=json.loads(path.read_text());m['fold0_opened']=True;write_json(path,m)
    with pytest.raises(ValueError,match='Protected'):workbench.phase_result(path,output)


def test_source_fact_drift_stops_without_retry(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);_,plan=workbench.load(output);path=plan['buckets'][0]['artifacts']['facts.parquet']['path']
    pq.write_table(pa.table({'bad':['changed']}),path);dest=workbench.bucket_dir(output,0);dest.mkdir(parents=True)
    with pytest.raises(ValueError,match='Authority drift'):workbench.worker(output,0)
    assert json.loads((dest/'worker_stop.json').read_text())['status']=='STOP'
    with pytest.raises(ValueError,match='already started'):workbench.worker(output,0)


def test_only_original_physical_owner_buckets(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch)
    for bad in (-1,True,9999):
        with pytest.raises(ValueError,match='Planned owner bucket'):workbench.execute(output,bad)
