import json

import numpy as np
import pyarrow.parquet as pq
import pytest

import scripts.compile_retrieval_v155_address_access as workbench
import scripts.compile_retrieval_v154_address_store as store
from src.xgb_matcher.retrieval_address_access_v155 import NationalAddresses
from src.xgb_matcher.retrieval_address_reverse_v147 import ReversePart
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget,POLICY
from test_compile_retrieval_v154_address_store import setup as store_setup,finish as store_finish
from test_resume_retrieval_v153_addresses import gate as fake_gate


def setup(tmp_path,monkeypatch,complete=True):
    projection,source=store_setup(tmp_path,monkeypatch)
    if complete:
        store.init(source,record(projection/'manifest.json'));store_finish(source)
    cfg=json.loads(workbench.CONFIG.read_text())
    cfg.update(expected_sites=11,expected_original_observations=12,expected_source_rows=6,
        output_root=str(tmp_path),term_buckets=2,batch_rows=1)
    path=tmp_path/'access_config.json';write_json(path,cfg);monkeypatch.setattr(workbench,'CONFIG',path)
    monkeypatch.setattr(workbench,'quiet',fake_gate)
    return source,tmp_path/'access'


def finish_phase(output,phase):
    l,p=workbench.load(output)
    pairs=[(i,None) for i in range(len(p['parts']))] if phase=='part' else [(b,ch) for ch in range(4) for b in range(l['config']['term_buckets'])]
    for index,channel in pairs:
        dest=workbench.directory(output,phase,index,channel);dest.mkdir(parents=True)
        write_json(dest/'preflight.json',fake_gate(1));workbench.worker(output,phase,index,channel)


def finish(output):
    finish_phase(output,'part');workbench.seal_parts(output)
    finish_phase(output,'bucket');workbench.seal(output)


def test_full_store_compile_query_snapshot_and_original_site_facts_end_to_end(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'));finish(out)
    m=json.loads((out/'manifest.json').read_text());l,p=workbench.load(out)
    assert m['status']==workbench.STATUS and m['aliases']==2 and len(m['access'])==8
    assert all(m[k] is False for k in workbench.PROTECTED)
    assert m['lexical_address_access_built'] and not m['complete_retrieval_index'] and not m['ready_for_fold1']
    d=json.loads((source/'dictionary.json').read_text());texts=[]
    for b in d['buckets']:texts.extend(pq.ParquetFile(b['artifacts']['addresses.parquet']['path']).read()['address_text'].to_pylist())
    reader=NationalAddresses(record(out/'manifest.json'))
    try:
        for text in [*texts,'RUE TEST','123456789','AB','']:
            snapshot,stats=reader.snapshot(text,chunk=1);query=channels(text)
            expected=[max((len(q&s)/len(q) if q else 0.) for q,s in zip(query,channels(a))) for a in texts]
            assert np.array_equal(snapshot.coverage,np.array(expected))
            assert np.array_equal(snapshot.exact,np.array([bool(normalized(text)) and normalized(text)==normalized(a) for a in texts]))
            assert not snapshot.coverage.flags.writeable and not snapshot.exact.flags.writeable
            facts=0
            for interval in json.loads((source/'manifest.json').read_text())['intervals']:
                reverse=ReversePart(interval)
                try:
                    ids=np.arange(interval['start'],interval['stop'],dtype=np.uint32)
                    for batch in reverse.batches(ids,snapshot,chunk=1):
                        facts+=len(batch.site_positions)
                finally:reverse.close()
            assert facts==p['facts'] and not stats['candidates_emitted']
        borrowed=reader.posting(3,normalized(texts[0]));reader.verify_after()
    finally:reader.close()
    reader.close();assert borrowed.tolist()
    with pytest.raises(ValueError,match='live whole-address'):reader.snapshot('RUE TEST')
    with pytest.raises(ValueError,match='Closed'):reader.posting(0,'RUE')
    before=record(out/'manifest.json')
    monkeypatch.setattr(workbench,'execute',lambda *a:pytest.fail('Already sealed phases must not rerun'))
    workbench.remaining(out);assert before==record(out/'manifest.json')


def test_missing_complete_store_fails_before_output_or_quiet_worker(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch,complete=False)
    monkeypatch.setattr(workbench,'quiet',lambda *a:pytest.fail('No worker or wait without complete source'))
    with pytest.raises(ValueError,match='NO_WORKER_LAUNCHED'):
        workbench.init(out,dict(path=str(source/'manifest.json'),sha256='0'*64))
    assert not out.exists()


@pytest.mark.parametrize('field',['fold0_opened','fold1_opened','truth_opened','crm_opened','complete_retrieval_index'])
def test_closed_source_or_scope_cannot_be_reinterpreted(tmp_path,monkeypatch,field):
    source,out=setup(tmp_path,monkeypatch);path=source/'manifest.json';m=json.loads(path.read_text());m[field]=True;write_json(path,m)
    with pytest.raises(ValueError,match='protected original address store'):workbench.init(out,record(path))
    assert not out.exists()


def test_wrong_denominator_and_missing_site_interval_stop(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);path=source/'manifest.json';m=json.loads(path.read_text());m['intervals'].pop();write_json(path,m)
    with pytest.raises(ValueError,match='every physical site interval'):workbench.init(out,record(path))
    assert not out.exists()


def test_partial_stage_and_existing_phase_never_restarted(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'))
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal_parts(out)
    dest=workbench.directory(out,'part',0);dest.mkdir(parents=True)
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.remaining(out)
    with pytest.raises(FileExistsError):workbench.execute(out,'part',0)


def test_hash_drift_and_namespace_drift_rejected(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'));finish_phase(out,'part');workbench.seal_parts(out)
    path=out/'staging_manifest.json';s=json.loads(path.read_text());item=next(i for items in s['buckets'].values() for i in items)
    item['channel_counts']['0']+=1;write_json(path,s)
    l,p=workbench.load(out)
    with pytest.raises(ValueError,match='exactly once'):workbench.complete_stage(out,l,p)
    ref=next(iter(workbench.phase_record(out,'part',0)['artifacts'].values()))
    with open(ref['path'],'ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='Authority drift'):workbench.remaining(out)


def test_seal_resource_stop_no_success_and_no_retry(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'));finish_phase(out,'part')
    previous=Budget.__exit__
    def fail(self,*args):previous(self,*args);raise RuntimeError('STOP_RESOURCE_POLICY')
    monkeypatch.setattr(Budget,'__exit__',fail)
    with pytest.raises(RuntimeError,match='STOP_RESOURCE_POLICY'):workbench.seal_parts(out)
    assert not (out/'staging_manifest.json').exists() and (out/'seal_seal_parts_stop.json').exists()
    with pytest.raises(ValueError,match='DO_NOT_RESTART'):workbench.remaining(out)


def test_relaxed_policy_or_wrong_output_root_refused(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);cfg=json.loads(workbench.CONFIG.read_text())
    cfg['resources']['max_swapout_pages_delta']=1;write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='authorized pressure/RSS'):workbench.init(out,record(source/'manifest.json'))
    cfg['resources'].pop('max_swapout_pages_delta');cfg['output_root']=str(tmp_path/'other');write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='CATNAT-only'):workbench.init(out,record(source/'manifest.json'))
    assert not out.exists()


def test_hot_load_no_repeated_global_hash_but_source_epoch_drift_stops(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'))
    monkeypatch.setattr(workbench,'store',lambda *a:pytest.fail('No multi-GB whole-source rehash per slice'))
    l,p=workbench.load(out)
    assert p['full_source_hashes_at_init_and_seal_not_each_phase'] and p['source_epochs']
    ref=next(item['reference'] for item in p['source_epochs'] if item['reference']['path'].endswith('addresses.parquet'))
    with open(ref['path'],'ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='epoch drift'):workbench.load(out)

def test_authorized_policy_is_explicit_in_full_lexical_seal(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'));finish(out)
    m=json.loads((out/'manifest.json').read_text())
    assert m['seal_resources']['policy_id']==POLICY and m['seal_resources']['swapout_pages_delta']>0
    assert m['seal_resources']['global_swap_is_worker_attributed'] is False
    assert not m['complete_retrieval_index'] and not m['ready_for_fold1']
    assert m['status'].startswith('SEALED_V155')


def test_scalar_channels_and_full_physical_plan_not_replaced():
    import scripts.compile_retrieval_v152_address_access as old
    assert workbench.physical_plan is old.physical_plan
    assert workbench.stage_addresses is old.stage_addresses
    assert workbench.pack is old.pack
