import json

import numpy as np
import pyarrow.parquet as pq
import pytest

import scripts.compile_retrieval_v136_extended_catalog as catalog_workbench
import scripts.probe_retrieval_v140_canonical_inventory as workbench
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_compile_retrieval_v136_extended_catalog import setup as catalog_setup
from test_retrieval_canonical_inventory_v140 import table,rows


def setup(tmp_path,monkeypatch):
    cat,_=catalog_setup(tmp_path,monkeypatch);catalog_workbench.worker(cat)
    refs={}
    for kind in ('names','addresses'):
        p=tmp_path/(kind+'.parquet');r=rows()
        if kind=='addresses':r['siren'][2]='000000009';r['siret'][2]='00000000900003'
        pq.write_table(table(r),p,row_group_size=2);refs[kind]=record(p)
    cfg=tmp_path/'inventory_config.json';write_json(cfg,dict(catalog=record(cat/'manifest.json'),**refs,
        expected_namespace_sites=8,expected_names_rows=8,expected_addresses_rows=8,numpy_version=np.__version__,
        resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),max_output_bytes=1048576))
    monkeypatch.setattr(workbench,'CONFIG',cfg);out=tmp_path/'canonical_inventory';workbench.init(out);return out


def finish(out):
    for kind in ('names','addresses'):
        (out/kind).mkdir();workbench.worker(out,kind)


def test_both_all_sources_required_then_exact_union_no_catalog_or_label_change(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal(out)
    finish(out);workbench.seal(out);m=json.loads((out/'manifest.json').read_text())
    assert m['unique_additional_absent_sites']==3 and all(m[k] is False for k in workbench.PROTECTED)
    assert np.load(m['all_absent_sites']['path']).tolist()==[100003,200009,900003]
    assert not m['candidate_catalog_changed'] and not m['complete_retrieval_index'] and not m['ready_for_fold1']
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(out)
    with pytest.raises(ValueError,match='Immutable'):workbench.worker(out,'names')


def test_any_closed_fold_phase_flag_stops_seal(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);finish(out)
    p=out/'addresses/manifest.json';m=json.loads(p.read_text());m['fold0_opened']=True;write_json(p,m)
    with pytest.raises(ValueError,match='Protected'):workbench.seal(out)


def test_source_drift_stops_worker_and_no_retry(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);l=workbench.load(out);p=l['config']['names']['path'];pq.write_table(table(rows()).slice(0,1),p)
    (out/'names').mkdir()
    with pytest.raises(ValueError,match='Authority drift'):workbench.worker(out,'names')
    assert json.loads((out/'names/worker_stop.json').read_text())['status']=='STOP'
    with pytest.raises(ValueError,match='Immutable'):workbench.worker(out,'names')
