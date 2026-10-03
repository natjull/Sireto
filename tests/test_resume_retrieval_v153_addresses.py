import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.resume_retrieval_v143_addresses as legacy
import scripts.resume_retrieval_v153_addresses as workbench
from src.xgb_matcher import retrieval_memory_policy_v153 as policy
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_resume_retrieval_v143_addresses import setup as legacy_setup


def gate(*args):
    return dict(policy_id=policy.POLICY,pass_system_quiet_policy=True,
                worker_swap_budget_changed=True,final_snapshot=dict(swapouts=42,free_percent=73,pressure_level=1))


def setup(tmp_path,monkeypatch):
    old,_=legacy_setup(tmp_path,monkeypatch)
    stop=legacy.directory(old,'official_addresses',1)/'worker_stop.json';stop.parent.mkdir(parents=True)
    write_json(stop,dict(status='STOP',resources=dict(pass_resource_policy=False,swapout_pages_delta=376),
                         **{k:False for k in workbench.PROTECTED}))
    cfg=tmp_path/'v153_config.json'
    write_json(cfg,dict(original_launch=record(old/'launch.json'),original_stop=record(stop),expected_reused_parts=2,
        batch_rows=1,initial_quiet_seconds=30,phase_quiet_seconds=1,min_system_free_percent=50,max_quiet_wait_seconds=55,
        authorization='HUMAN_OUI_2026_10_04_1GIB_PRESSURE_SWAP_TELEMETRY'))
    monkeypatch.setattr(workbench,'CONFIG',cfg);monkeypatch.setattr(workbench,'quiet',gate)
    swaps=iter(range(42,1000000))
    monkeypatch.setattr(policy,'snapshot',lambda:dict(swapouts=next(swaps),free_percent=73,pressure_level=1))
    out=tmp_path/'v153';workbench.init(out);return out,old


def finish(out):
    _,plan=workbench.load(out)
    for role in plan['role_order']:
        for spec in plan['sources'][role]['parts']:
            part=spec['part_id']
            if f'{role}/{part}' in plan['imported_parts']:continue
            dest=workbench.directory(out,role,part);dest.mkdir(parents=True)
            write_json(dest/'preflight.json',gate());workbench.worker(out,role,part)


def test_all_parts_preserve_bit_exact_proofs_despite_nonzero_global_swap(tmp_path,monkeypatch):
    out,old=setup(tmp_path,monkeypatch);stop=record(legacy.directory(old,'official_addresses',1)/'worker_stop.json')
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal(out)
    finish(out);workbench.seal(out);m=json.loads((out/'manifest.json').read_text())
    assert m['source_rows']==6 and m['observations']==12 and len(m['parts'])==4
    assert m['status']=='SEALED_V153_ALL_ORIGINAL_OFFICIAL_AND_RAW_CURRENT_ADDRESS_PROJECTION'
    assert m['seal_resources']['swapout_pages_delta']>0
    assert all(m[k] is False for k in workbench.PROTECTED) and not m['complete_retrieval_index']
    l,p=workbench.load(out);keys=AppendKeys(l['config']['catalog']);actual=[];expected=[]
    for item in m['parts']:
        part=json.loads(workbench.verify(item['manifest']).read_text());actual.append(pq.read_table(part['artifact']['path']))
        if not item['imported']:assert part['resources']['policy_id']==policy.POLICY and part['resources']['swapout_pages_delta']>0
        else:assert part['resources']['swapout_pages_delta']==0
    for role in p['role_order']:
        source=p['sources'][role];original=pq.read_table(source['reference']['path'])
        expected.append(workbench.project(original,role,keys,0,source['observation_base']))
    assert pa.concat_tables(actual).equals(pa.concat_tables(expected))
    assert stop==record(legacy.directory(old,'official_addresses',1)/'worker_stop.json')
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(out)
    with pytest.raises(ValueError,match='overwrite/retry'):workbench.worker(out,'official_addresses',1)


def test_import_every_nested_and_recent_success(tmp_path,monkeypatch):
    out,old=setup(tmp_path,monkeypatch)
    dest=legacy.directory(old,'current_addresses',1);dest.mkdir(parents=True)
    from test_resume_retrieval_v143_addresses import fake_gate
    write_json(dest/'preflight.json',fake_gate(1));legacy.worker(old,'current_addresses',1)
    cfg=json.loads(workbench.CONFIG.read_text());cfg['expected_reused_parts']=3;write_json(workbench.CONFIG,cfg)
    workbench.init(tmp_path/'v153_more');_,p=workbench.load(tmp_path/'v153_more')
    assert p['imported_part_count']==3
    assert p['imported_parts']['current_addresses/1']['launch']==cfg['original_launch']


def test_pressure_failure_no_launch_no_retry_no_seal(tmp_path,monkeypatch):
    out,_=setup(tmp_path,monkeypatch)
    def fail(*args):raise RuntimeError('STOP_ACTUAL_MEMORY_PRESSURE')
    monkeypatch.setattr(workbench,'quiet',fail)
    with pytest.raises(RuntimeError,match='PRESSURE'):workbench.execute(out,'official_addresses',1)
    dest=workbench.directory(out,'official_addresses',1)
    assert (dest/'status.json').exists() and not (dest/'observations.parquet').exists()
    with pytest.raises(ValueError,match='DO_NOT_RESTART'):workbench.remaining(out)
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal(out)


@pytest.mark.parametrize('key,value',[('fold1_opened',True),('parent_addresses_spread_to_sites',True),
    ('every_exported_field_matches_second_source_pass',False),('context_or_masked_rows_removed',True)])
def test_new_projection_guard_drift_stops(tmp_path,monkeypatch,key,value):
    out,_=setup(tmp_path,monkeypatch);finish(out)
    path=workbench.directory(out,'official_addresses',1)/'manifest.json'
    m=json.loads(path.read_text());m[key]=value;write_json(path,m)
    with pytest.raises(ValueError,match='independently complete'):workbench.seal(out)


def test_authorization_or_memory_limit_drift_stops(tmp_path,monkeypatch):
    out,_=setup(tmp_path,monkeypatch);path=out/'launch.json';l=json.loads(path.read_text())
    l['config']['resources']['max_rss_bytes']=3221225472;write_json(path,l)
    with pytest.raises(ValueError,match='authorized memory policy'):workbench.load(out)
