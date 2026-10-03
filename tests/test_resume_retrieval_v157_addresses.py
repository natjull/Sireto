import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.resume_retrieval_v153_addresses as old
import scripts.resume_retrieval_v157_addresses as workbench
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from test_resume_retrieval_v153_addresses import setup as source_setup,finish as source_finish,gate


def setup(tmp_path,monkeypatch):
    source,_=source_setup(tmp_path,monkeypatch);source_finish(source)
    stop=old.directory(source,'current_addresses',1)/'status.json'
    write_json(stop,dict(status='STOP',reason='process no longer exists (pid=12345)',
                         **{k:False for k in workbench.PROTECTED}))
    cfg=tmp_path/'race_config.json'
    write_json(cfg,dict(original_launch=record(source/'launch.json'),original_stop=record(stop),expected_reused_parts=3,
        batch_rows=1,initial_quiet_seconds=30,phase_quiet_seconds=1,min_system_free_percent=50,max_quiet_wait_seconds=55,
        authorization='HUMAN_OUI_2026_10_04_1GIB_PRESSURE_SWAP_TELEMETRY'))
    monkeypatch.setattr(workbench,'CONFIG',cfg);monkeypatch.setattr(workbench,'quiet',gate)
    out=tmp_path/'race_resume';workbench.init(out);return out,source


def finish(out):
    _,plan=workbench.load(out)
    for role in plan['role_order']:
        for spec in plan['sources'][role]['parts']:
            part=spec['part_id']
            if f'{role}/{part}' in plan['imported_parts']:continue
            dest=workbench.directory(out,role,part);dest.mkdir(parents=True)
            write_json(dest/'preflight.json',gate());workbench.worker(out,role,part)


def test_worker_success_with_parent_stop_is_excluded_then_recomputed_same_proofs(tmp_path,monkeypatch):
    out,source=setup(tmp_path,monkeypatch)
    stop=record(old.directory(source,'current_addresses',1)/'status.json')
    orphan=record(old.directory(source,'current_addresses',1)/'manifest.json')
    l,p=workbench.load(out)
    assert p['imported_part_count']==3 and 'current_addresses/1' not in p['imported_parts']
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal(out)
    finish(out);workbench.seal(out);m=json.loads((out/'manifest.json').read_text())
    assert m['observations']==12 and m['source_rows']==6 and len(m['parts'])==4
    assert m['status']=='SEALED_V157_ALL_ORIGINAL_OFFICIAL_AND_RAW_CURRENT_ADDRESS_PROJECTION'
    assert stop==record(old.directory(source,'current_addresses',1)/'status.json')
    assert orphan==record(old.directory(source,'current_addresses',1)/'manifest.json')
    tables=[pq.read_table(json.loads(workbench.verify(i['manifest']).read_text())['artifact']['path']) for i in m['parts']]
    keys=AppendKeys(l['config']['catalog']);expected=[]
    for role in p['role_order']:
        s=p['sources'][role];expected.append(workbench.project(pq.read_table(s['reference']['path']),role,keys,0,s['observation_base']))
    assert pa.concat_tables(tables).equals(pa.concat_tables(expected))
    assert all(m[k] is False for k in workbench.PROTECTED) and not m['complete_retrieval_index']
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(out)


def test_new_parent_stop_blocks_seal_even_when_manifest_present(tmp_path,monkeypatch):
    out,_=setup(tmp_path,monkeypatch);finish(out)
    write_json(workbench.directory(out,'current_addresses',1)/'status.json',dict(status='STOP'))
    with pytest.raises(ValueError,match='STOP_SOURCE_SUPERVISION'):workbench.seal(out)


def test_global_recovery_store_accepts_only_every_original_part(tmp_path,monkeypatch):
    out,_=setup(tmp_path,monkeypatch);finish(out);workbench.seal(out)
    import scripts.compile_retrieval_v154_address_store as storage
    cfg=json.loads(storage.CONFIG.read_text());cfg.update(expected_observations=12,expected_source_rows=6,expected_sites=11)
    p=storage.projection(record(out/'manifest.json'),cfg)
    assert p['source_rows']==6 and p['observations']==12 and len(p['parts'])==4
    m=json.loads((out/'manifest.json').read_text());m['parts'].pop();write_json(out/'manifest.json',m)
    with pytest.raises(ValueError,match='UNFINISHED'):storage.projection(record(out/'manifest.json'),cfg)
