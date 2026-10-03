import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.project_retrieval_v142_addresses as old
import scripts.resume_retrieval_v143_addresses as workbench
from src.xgb_matcher.retrieval_address_projection_v142 import project
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,write_json
from test_project_retrieval_v142_addresses import setup as old_setup
from test_retrieval_address_projection_v142 import canonical,current,Keys


def fake_gate(*args):
    return dict(pass_system_quiet_policy=True,worker_swap_budget_changed=False,
        final_snapshot={'swapouts':42,'free_percent':73},quiet_seconds=args[0])


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(Budget,'swap',staticmethod(lambda:42))
    root=old_setup(tmp_path,monkeypatch);l,_=old.load(root);l['config']['batch_rows']=4
    cfg_path=old.CONFIG;write_json(cfg_path,l['config']);l['implementation'][str(cfg_path)]=record(cfg_path)
    write_json(root/'launch.json',l)
    for role in ('official_addresses','current_addresses'):
        old.directory(root,role,0).mkdir(parents=True);old.worker(root,role,0)
    stop=old.directory(root,'official_addresses',1)/'worker_stop.json';stop.parent.mkdir()
    write_json(stop,dict(status='STOP',reason='STOP_RESOURCE_POLICY',resources=dict(pass_resource_policy=False,swapout_pages_delta=64),
        **{k:False for k in old.PROTECTED}))
    cfg=tmp_path/'resume_config.json';write_json(cfg,dict(original_launch=record(root/'launch.json'),original_stop=record(stop),
        expected_reused_parts=2,batch_rows=1,initial_quiet_seconds=30,phase_quiet_seconds=1,min_system_free_percent=50,max_quiet_wait_seconds=55))
    monkeypatch.setattr(workbench,'CONFIG',cfg);monkeypatch.setattr(workbench,'quiet',fake_gate)
    out=tmp_path/'resume';workbench.init(out);return out,root


def finish(out):
    _,plan=workbench.load(out)
    for role in plan['role_order']:
        for spec in plan['sources'][role]['parts']:
            part=spec['part_id']
            if f'{role}/{part}' in plan['imported_parts']:continue
            dest=workbench.directory(out,role,part);dest.mkdir(parents=True)
            write_json(dest/'preflight.json',fake_gate(1));workbench.worker(out,role,part)


def test_partial_stop_not_imported_all_proofs_bit_exact_across_smaller_batches(tmp_path,monkeypatch):
    out,root=setup(tmp_path,monkeypatch);before=record(old.directory(root,'official_addresses',1)/'worker_stop.json')
    l,plan=workbench.load(out);assert plan['imported_part_count']==2 and l['config']['batch_rows']==1
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal(out)
    with pytest.raises(ValueError,match='imported'):workbench.execute(out,'official_addresses',0)
    finish(out);workbench.seal(out);m=json.loads((out/'manifest.json').read_text())
    assert m['observations']==12 and m['source_rows']==6 and len(m['parts'])==4
    assert sum(r['imported'] for r in m['parts'])==2 and m['lexical_observations']==4
    assert m['output_bytes']==m['new_output_bytes']+m['reused_output_bytes'] and m['seal_resources']['pass_resource_policy']
    assert all(m[k] is False for k in workbench.PROTECTED) and not m['complete_retrieval_index'] and not m['ready_for_fold1']
    assert before==record(old.directory(root,'official_addresses',1)/'worker_stop.json')
    tables=[]
    for item in m['parts']:
        part=json.loads(workbench.verify(item['manifest']).read_text());tables.append(pq.read_table(part['artifact']['path']))
    actual=pa.concat_tables(tables);expected=[];k=AppendKeys(l['config']['catalog'])
    for role in plan['role_order']:
        s=plan['sources'][role];t=pq.read_table(s['reference']['path']);expected.append(project(t,role,k,0,s['observation_base']))
    assert actual.equals(pa.concat_tables(expected))
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(out)
    with pytest.raises(ValueError,match='overwrite/retry'):workbench.worker(out,'official_addresses',1)


def test_memory_batch_partition_is_not_business_or_proof_cap():
    t=canonical([dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',raw_value='① rue côté',source='RNE',is_current=True)]*2051)
    k=Keys();whole=project(t,'official_addresses',k,0,0)
    small=pa.concat_tables([project(t.slice(i,17),'official_addresses',k,i,0) for i in range(0,len(t),17)])
    assert whole.equals(small) and len(small)==2051
    t=current([dict(siren='000000001',siret='00000000100001',complementAdresseEtablissement='[\u00a0ND\u00a0]',
        libelleVoieEtablissement='École',numeroVoie2Etablissement='①②③④⑤⑥⑦⑧⑨')]*19)
    whole=project(t,'current_addresses',k,0,100)
    small=pa.concat_tables([project(t.slice(i,3),'current_addresses',k,i,100) for i in range(0,len(t),3)])
    assert whole.equals(small) and len(small)==76


def test_any_imported_authority_closed_fold_or_export_drift_stops(tmp_path,monkeypatch):
    out,root=setup(tmp_path,monkeypatch);l,p=workbench.load(out)
    ref=p['imported_parts']['official_addresses/0'];m=json.loads(workbench.verify(ref).read_text());m['fold1_opened']=True
    bad=tmp_path/'bad_part.json';write_json(bad,m)
    with pytest.raises(ValueError,match='independently complete'):
        workbench.validate(record(bad),'official_addresses',m['part'],p['sources']['official_addresses'],l['resumption']['original_launch'],l['config']['catalog'],1)
    m=json.loads(workbench.verify(ref).read_text());path=m['artifact']['path'];pq.write_table(canonical([]),path)
    with pytest.raises(ValueError,match='Authority drift'):workbench.seal(out)


def test_preflight_failure_or_changed_counter_does_not_launch_or_retry(tmp_path,monkeypatch):
    out,_=setup(tmp_path,monkeypatch)
    def fail(*args):raise RuntimeError('STOP_SYSTEM_NOT_QUIET_NO_WORKER_LAUNCHED')
    monkeypatch.setattr(workbench,'quiet',fail)
    with pytest.raises(RuntimeError,match='NO_WORKER'):workbench.execute(out,'official_addresses',1)
    dest=workbench.directory(out,'official_addresses',1)
    assert (dest/'status.json').exists() and not (dest/'observations.parquet').exists()
    with pytest.raises(ValueError,match='DO_NOT_RESTART'):workbench.remaining(out)
    dest=workbench.directory(out,'current_addresses',1);dest.mkdir(parents=True)
    g=fake_gate(1);g['final_snapshot']['swapouts']=41;write_json(dest/'preflight.json',g)
    with pytest.raises(RuntimeError,match='CHANGED_SINCE_QUIET_GATE'):workbench.worker(out,'current_addresses',1)
    assert not (dest/'observations.parquet').exists()
    with pytest.raises(ValueError,match='overwrite/retry'):workbench.worker(out,'current_addresses',1)
