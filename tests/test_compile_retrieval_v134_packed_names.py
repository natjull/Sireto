import json

import duckdb
import numpy as np
import pytest

import scripts.compile_retrieval_v134_packed_names as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_compile_retrieval_v131_name_access import setup as original_setup,parts


def setup(tmp_path,monkeypatch):
    source,_=original_setup(tmp_path,monkeypatch);parts(source);s=json.loads((source/'staging_manifest.json').read_text())
    cfg=tmp_path/'packed_config.json';write_json(cfg,dict(source_stage=record(source/'staging_manifest.json'),
        expected_names=s['names'],expected_postings=s['postings'],term_buckets=2,
        duckdb_version=duckdb.__version__,numpy_version=np.__version__,duckdb_memory='128MiB',spill_limit='32MiB',
        resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),
        max_phase_output_bytes=1048576,max_total_output_bytes_including_source_stage=16777216))
    monkeypatch.setattr(workbench,'CONFIG',cfg);output=tmp_path/'packed';workbench.init(output);return output,s


def test_all_original_channels_must_complete_before_seal(tmp_path,monkeypatch):
    output,source=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='UNFINISHED_ALL_CHANNEL_BUCKETS'):workbench.seal(output)
    for ch in range(4):
        for b in range(2):
            workbench.bucket_dir(output,ch,b).mkdir(parents=True);workbench.worker(output,ch,b)
    workbench.seal(output);m=json.loads((output/'manifest.json').read_text())
    assert m['names']==source['names'] and m['postings']==source['postings'] and len(m['access'])==8
    assert all(m[k] is False for k in workbench.PROTECTED)
    assert m['source_semantics_unchanged'] and not m['complete_retrieval_index'] and not m['ready_for_fold1']
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(output)
    with pytest.raises(FileExistsError):workbench.execute(output,0,0)
    with pytest.raises(ValueError,match='already started'):workbench.worker(output,0,0)


def test_original_protected_flags_and_numpy_pin(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);workbench.bucket_dir(output,0,0).mkdir(parents=True);workbench.worker(output,0,0)
    path=workbench.bucket_dir(output,0,0)/'manifest.json';m=json.loads(path.read_text());m['fold1_opened']=True;write_json(path,m)
    with pytest.raises(ValueError,match='Protected'):workbench.phase_result(path,output)
    monkeypatch.setattr(np,'__version__','not-pinned')
    with pytest.raises(ValueError,match='library drift'):workbench.load(output)


def test_only_actual_physical_channels_and_buckets(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch)
    for ch,b in ((True,0),(4,0),(0,True),(0,-1),(0,2)):
        with pytest.raises(ValueError,match='planned channel'):workbench.execute(output,ch,b)
