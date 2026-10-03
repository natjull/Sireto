import json

import numpy as np
import pytest

import scripts.compile_retrieval_v136_extended_catalog as catalog_workbench
import scripts.compile_retrieval_v139_site_attributes as workbench
from src.xgb_matcher.retrieval_site_attributes_v139 import StructuredSites
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_compile_retrieval_v136_extended_catalog import setup as catalog_setup
from test_retrieval_site_attributes_v139 import source


def setup(tmp_path,monkeypatch):
    cat,_=catalog_setup(tmp_path,monkeypatch);catalog_workbench.worker(cat);cm=json.loads((cat/'manifest.json').read_text())
    stock,_,_,ctx,_=source(tmp_path)
    launch=tmp_path/'attribute_source_launch.json';write_json(launch,dict(implementation={'core':record(workbench.ROOT/'src/xgb_matcher/retrieval_site_attributes_v139.py')}))
    common=dict(launch=record(launch),resources={'pass_resource_policy':True},ready_for_fold1=False,**{k:False for k in workbench.PROTECTED})
    folder=tmp_path/'context139';folder.mkdir()
    for k,v in ctx.items():
        if isinstance(v,np.ndarray):np.save(folder/(k+'.npy'),v,allow_pickle=False)
    c=tmp_path/'context139.json';write_json(c,dict(status=workbench.STATUSES['context'],
        stats=dict(sites=3,full_registry_exact_alignment=True,codebooks={'administrative_state':ctx['state_book']}),
        inputs={'registry_arrays':{k:cm['input_arrays'][k] for k in ('endpoint_keys','endpoint_parents')}},
        artifacts={p.name:record(p) for p in folder.glob('*.npy')},**common))
    folder=tmp_path/'activity139';folder.mkdir()
    np.save(folder/'nomenclature_codes.npy',np.array([2,3,0],dtype=np.uint8),allow_pickle=False)
    np.save(folder/'original_activity_codes.npy',np.array([2,2,0],dtype=np.uint16),allow_pickle=False)
    labels=folder/'labels.json';write_json(labels,dict(rows=[dict(nomenclature='NAFRev2',code='47.11B',label='TEST')]))
    a=tmp_path/'activity139.json';write_json(a,dict(status=workbench.STATUSES['activity'],
        stats=dict(sites=3,all_sites_preserved=True,nomenclature_inferred=False,nomenclatures=[None,'','NAFRev2','NAP'],activity_book=[None,'','47.11B']),
        inputs=dict(stock=record(stock),keys=cm['input_arrays']['endpoint_keys'],parents=cm['input_arrays']['endpoint_parents']),
        artifacts={p.name:record(p) for p in folder.glob('*.npy')},official_labels=record(labels),**common))
    xy=tmp_path/'site_xy139.npy';np.save(xy,np.array([[650000.,6800000.],[700000.,6500000.],[np.nan,np.nan]]),allow_pickle=False)
    lay=tmp_path/'layout139.json';write_json(lay,dict(status=workbench.STATUSES['layout'],source=record(stock),
        stats=dict(sites=3,all_sites_exactly_once=True),artifacts={'site_xy.npy':record(xy)},**common))
    cfg=tmp_path/'attributes_config.json';write_json(cfg,dict(catalog=record(cat/'manifest.json'),context=record(c),activity=record(a),layout=record(lay),stock=record(stock),
        expected_current_sites=3,expected_namespace_sites=8,numpy_version=np.__version__,
        resources=dict(max_seconds=80,max_rss_bytes=3221225472,max_swapout_pages_delta=0),max_output_bytes=1048576))
    monkeypatch.setattr(workbench,'CONFIG',cfg);output=tmp_path/'attributes';workbench.init(output);return output


def finish(output):
    for phase in ('build','audit'):
        (output/phase).mkdir();workbench.worker(output,phase)


def test_no_seal_before_all_original_rows_audited_and_reader_extra_unknown(tmp_path,monkeypatch):
    output=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal(output)
    (output/'build').mkdir();workbench.worker(output,'build')
    with pytest.raises(ValueError,match='UNFINISHED_ALL_RAW'):workbench.seal(output)
    (output/'audit').mkdir();workbench.worker(output,'audit');workbench.seal(output)
    m=json.loads((output/'manifest.json').read_text())
    assert all(m[k] is False for k in workbench.PROTECTED) and m['namespace_sites']==8
    assert not m['complete_retrieval_index'] and not m['ready_for_fold1']
    reader=StructuredSites(record(output/'manifest.json'));ids=np.array([0,1,2,3,7],dtype=np.uint32)
    p,n,h,c=reader.flags(ids,insee='75056',number='07 bis')
    assert p.tolist()==[1.,0.,0.,0.,0.] and n.tolist()==[0.,1.,0.,0.,0.] and h.tolist()==[1.,0.,0.,0.,0.]
    assert c.tolist()==[0,-1,-1,-1,-1] # NAP explicitly UNKNOWN, no code-shape guess
    xy=reader.coordinates(ids)
    assert np.array_equal(xy[:2],np.array([[650000.,6800000.],[700000.,6500000.]])) and np.isnan(xy[2:]).all()
    for bad in (np.array([8],dtype=np.uint32),np.array([0.])):
        with pytest.raises(ValueError):reader.flags(bad)
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(output)
    with pytest.raises(ValueError,match='Immutable'):workbench.worker(output,'build')


def test_protected_fold_metadata_never_admitted(tmp_path,monkeypatch):
    output=setup(tmp_path,monkeypatch);l=workbench.load(output);cfg=l['config']
    m=json.loads(workbench.verify(cfg['context']).read_text());m['fold1_opened']=True
    p=tmp_path/'bad_context.json';write_json(p,m);cfg['context']=record(p)
    with pytest.raises(ValueError,match='Protected'):workbench.inputs(cfg)


def test_any_array_drift_stops_independent_audit_without_retry(tmp_path,monkeypatch):
    output=setup(tmp_path,monkeypatch);(output/'build').mkdir();workbench.worker(output,'build')
    np.save(output/'build/insee_codes.npy',np.zeros(3,dtype=np.uint32),allow_pickle=False)
    (output/'audit').mkdir()
    with pytest.raises(ValueError,match='Authority drift'):workbench.worker(output,'audit')
    assert json.loads((output/'audit/worker_stop.json').read_text())['status']=='STOP'
    with pytest.raises(ValueError,match='Immutable'):workbench.worker(output,'audit')


def test_original_context_mismatch_stops_not_fabricated_or_overwritten(tmp_path,monkeypatch):
    output=setup(tmp_path,monkeypatch);finish(output)
    p=output/'audit/manifest.json';m=json.loads(p.read_text());m['stats']['every_reused_headquarters_state_and_date_matches_raw_stock']=False;write_json(p,m)
    with pytest.raises(ValueError,match='Protected'):workbench.seal(output)
