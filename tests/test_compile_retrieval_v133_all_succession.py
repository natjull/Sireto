import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.compile_retrieval_v133_all_succession as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_retrieval_all_succession_v133 import SCHEMA,row


def setup(tmp_path,monkeypatch):
    source=tmp_path/'source.parquet';pq.write_table(pa.Table.from_pylist([
        row('12345678900001','12345678900002',True),row('12345678900002','12345678900002')],schema=SCHEMA),source)
    cfg=tmp_path/'config.json';write_json(cfg,dict(source=record(source),expected_observations=2,
        resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),max_output_bytes=1048576))
    monkeypatch.setattr(workbench,'CONFIG',cfg);output=tmp_path/'mapped';workbench.init(output);return output,source


def test_full_source_manifest_and_immutable_phases(tmp_path,monkeypatch):
    output,_=setup(tmp_path,monkeypatch);workbench.worker(output);m=json.loads((output/'manifest.json').read_text())
    assert m['stats']['observations']==2 and m['stats']['self_context_links']==1
    assert m['stats']['every_original_field_reconstructed_exactly'] and all(m[k] is False for k in workbench.PROTECTED)
    assert m['complete_retrieval_index'] is False and m['ready_for_fold1'] is False
    with pytest.raises(ValueError,match='already started'):workbench.worker(output)
    with pytest.raises(FileExistsError):workbench.run(output)


def test_changed_official_source_stops_without_retry(tmp_path,monkeypatch):
    output,source=setup(tmp_path,monkeypatch);pq.write_table(pa.table({'bad':['changed']}),source)
    with pytest.raises(ValueError,match='Authority drift'):workbench.worker(output)
    assert json.loads((output/'worker_stop.json').read_text())['status']=='STOP'
    with pytest.raises(ValueError,match='already started'):workbench.worker(output)
