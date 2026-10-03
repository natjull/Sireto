import json

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.compile_retrieval_v130_variant_dictionary as dictionary
import scripts.compile_retrieval_v131_name_access as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_compile_retrieval_v130_variant_dictionary import setup as dictionary_setup,stage_all


def setup(tmp_path,monkeypatch):
    source,_=dictionary_setup(tmp_path,monkeypatch);stage_all(source)
    for b in range(2):
        dictionary.bucket_dir(source,b).mkdir(parents=True);dictionary.worker(source,'bucket',b)
    dictionary.seal_dictionary(source);m=json.loads((source/'manifest.json').read_text())
    cfg=tmp_path/'access_config.json';write_json(cfg,dict(source_manifest=record(source/'manifest.json'),
        expected_names=m['names'],expected_facts=m['facts'],expected_observations=m['original_observations'],
        term_buckets=2,duckdb_version=duckdb.__version__,duckdb_memory='128MiB',spill_limit='32MiB',
        resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),
        max_phase_output_bytes=1048576,max_total_output_bytes=16777216))
    monkeypatch.setattr(workbench,'CONFIG',cfg);output=tmp_path/'access';workbench.init(output)
    return output,m


def parts(output):
    _,plan=workbench.load(output)
    for spec in plan['parts']:
        workbench.part_dir(output,spec['part_id']).mkdir(parents=True)
        workbench.worker(output,'part',spec['part_id'],None)
    workbench.seal_parts(output)


def test_every_source_name_and_four_channels_required_before_seal(tmp_path,monkeypatch):
    output,source=setup(tmp_path,monkeypatch);_,plan=workbench.load(output)
    assert plan['names']==source['names'] and sum(s['row_stop']-s['row_start'] for s in plan['parts'])==source['names']
    with pytest.raises(ValueError,match='UNFINISHED_ALL_NAME_PARTS'):workbench.seal_parts(output)
    with pytest.raises(FileNotFoundError):workbench.execute(output,'bucket',0,0)
    parts(output)
    with pytest.raises(ValueError,match='UNFINISHED_ALL_CHANNEL_ACCESS'):workbench.seal_access(output)
    for ch in range(4):
        for b in range(2):
            workbench.access_dir(output,ch,b).mkdir(parents=True);workbench.worker(output,'bucket',b,ch)
    workbench.seal_access(output);m=json.loads((output/'manifest.json').read_text())
    assert m['names']==source['names'] and len(m['access'])==8 and m['postings']>m['names']
    assert m['complete_retrieval_index'] is False and m['ready_for_fold1'] is False
    assert all(m[k] is False for k in workbench.PROTECTED)
    assert sum(a['postings'] for a in m['access'])==m['postings']
    with pytest.raises(ValueError,match='already sealed'):workbench.seal_parts(output)
    with pytest.raises(ValueError,match='already sealed'):workbench.seal_access(output)
    with pytest.raises(FileExistsError):workbench.execute(output,'part',0)
    with pytest.raises(ValueError,match='already started'):workbench.worker(output,'part',0,None)


def test_protected_fold_and_library_guards(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);parts(output)
    path=workbench.part_dir(output,0)/'manifest.json';m=json.loads(path.read_text());m['fold1_opened']=True;write_json(path,m)
    with pytest.raises(ValueError,match='Protected'):workbench.phase_result(path,output,'SEALED_V131_OFFICIAL_NAME_CHANNEL_PART')
    monkeypatch.setattr(duckdb,'__version__','not-pinned')
    with pytest.raises(ValueError,match='hash library'):workbench.load(output)


def test_changed_raw_name_source_stops_without_retry(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);_,plan=workbench.load(output)
    path=plan['parts'][0]['source']['path'];pq.write_table(pa.table({'bad':['changed']}),path)
    dest=workbench.part_dir(output,0);dest.mkdir(parents=True)
    with pytest.raises(ValueError,match='Authority drift'):workbench.worker(output,'part',0,None)
    stop=json.loads((dest/'worker_stop.json').read_text());assert stop['status']=='STOP'
    with pytest.raises(ValueError,match='already started'):workbench.worker(output,'part',0,None)


def test_only_planned_phase_and_actual_channel_are_accepted(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch)
    for bad in (-1,True,1000000):
        with pytest.raises(ValueError,match='Planned physical'):workbench.execute(output,'part',bad)
    for bad in (-1,True,4,None):
        with pytest.raises(ValueError,match='Four actual'):workbench.execute(output,'bucket',0,bad)
