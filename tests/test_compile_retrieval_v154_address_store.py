import json

import pytest

import scripts.compile_retrieval_v154_address_store as workbench
import scripts.resume_retrieval_v153_addresses as projection
from src.xgb_matcher.retrieval_address_reverse_v147 import ReversePart,AliasSnapshot
from src.xgb_matcher.retrieval_v20_workbench import record,write_json
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget,LIMITS,POLICY
from test_resume_retrieval_v153_addresses import setup as source_setup,finish as source_finish,gate as fake_gate


def setup(tmp_path,monkeypatch,complete=True):
    source,_=source_setup(tmp_path,monkeypatch)
    cfg=json.loads(workbench.CONFIG.read_text())
    cfg.update(expected_observations=12,expected_source_rows=6,expected_sites=11,dictionary_buckets=2,site_quantum=4,output_root=str(tmp_path))
    path=tmp_path/'store_config.json';write_json(path,cfg);monkeypatch.setattr(workbench,'CONFIG',path)
    monkeypatch.setattr(workbench,'quiet',fake_gate)
    if complete:source_finish(source);projection.seal(source)
    return source,tmp_path/'store'


def finish_phase(output,phase):
    for index in range(workbench.maximum(output,phase)):
        dest=workbench.directory(output,phase,index);dest.mkdir(parents=True)
        write_json(dest/'preflight.json',fake_gate(1));workbench.worker(output,phase,index)


def finish(output):
    finish_phase(output,'part');workbench.seal_staging(output)
    finish_phase(output,'dictionary');workbench.seal_dictionary(output)
    finish_phase(output,'reverse-stage');workbench.seal_staging(output,True)
    finish_phase(output,'reverse-pack');workbench.seal(output)


def test_missing_national_manifest_never_starts_worker_or_creates_root(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch,complete=False)
    with pytest.raises(ValueError,match='NO_WORKER_LAUNCHED'):
        workbench.init(out,dict(path=str(source/'manifest.json'),sha256='0'*64))
    assert not out.exists()


def test_full_original_plan_required_before_dictionary_or_reverse_seal(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'))
    l,p=workbench.load(out)
    assert p['observations']==12 and len(p['parts'])==4 and p['sites']==11
    assert [s['observation_start'] for s in p['parts']]==[0,2,4,8]
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal_staging(out)
    finish(out);m=json.loads((out/'manifest.json').read_text())
    assert m['status']==workbench.GLOBAL and m['original_observations']==12 and m['lexical_observations']==4
    assert m['sites']==11 and len(m['intervals'])==3
    assert m['facts']==2 and m['aliases']==2
    assert sum(v['facts'] for v in m['intervals'])==m['facts']
    assert all(m[k] is False for k in workbench.PROTECTED)
    assert not m['lexical_address_access_built'] and not m['complete_retrieval_index'] and not m['ready_for_fold1']
    for interval in m['intervals']:
        r=ReversePart(interval);r.close()
    with pytest.raises(ValueError,match='already sealed'):workbench.seal(out)
    with pytest.raises(ValueError,match='no retry'):workbench.worker(out,'part',0)
    with pytest.raises(FileExistsError):workbench.init(out,record(source/'manifest.json'))


@pytest.mark.parametrize('field',['fold0_opened','fold1_opened','truth_opened','crm_opened'])
def test_source_closed_fold_or_truth_flag_is_stop_not_reinterpreted(tmp_path,monkeypatch,field):
    source,out=setup(tmp_path,monkeypatch);p=source/'manifest.json';m=json.loads(p.read_text());m[field]=True;write_json(p,m)
    with pytest.raises(ValueError,match='protected all-source'):workbench.init(out,record(p))
    assert not out.exists()


def test_missing_original_part_or_lie_in_observation_count_stops(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);p=source/'manifest.json';m=json.loads(p.read_text());m['parts'].pop();write_json(p,m)
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.init(out,record(p))
    assert not out.exists()


def test_phase_live_or_partial_is_not_restarted_or_counted_complete(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'))
    dest=workbench.directory(out,'part',0);dest.mkdir(parents=True)
    with pytest.raises(ValueError,match='DO_NOT_RESTART'):workbench.remaining(out)
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal_staging(out)


def test_existing_success_hash_drift_blocks_remaining(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'))
    finish_phase(out,'part');m=workbench.phase_record(out,'part',0)
    ref=next(iter(m['artifacts'].values()));p=workbench.verify(ref)
    with p.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='Authority drift'):workbench.remaining(out)


def test_resource_policy_relaxation_is_not_implicit_in_this_driver(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);cfg=json.loads(workbench.CONFIG.read_text())
    cfg['resources']['max_swapout_pages_delta']=1;write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='authorized pressure/RSS'):workbench.init(out,record(source/'manifest.json'))
    assert not out.exists()


def test_completed_pipeline_is_revalidated_not_executed_again(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'));finish(out)
    before=record(out/'manifest.json')
    def forbidden(*args):raise AssertionError('Never rebuild an already successful phase')
    monkeypatch.setattr(workbench,'execute',forbidden);workbench.remaining(out)
    assert before==record(out/'manifest.json')
    m=json.loads((out/'manifest.json').read_text());m['fold1_opened']=True;write_json(out/'manifest.json',m)
    with pytest.raises(ValueError,match='Protected complete'):workbench.remaining(out)


def test_seal_resource_stop_never_publishes_success_or_allows_retry(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'));finish_phase(out,'part')
    previous_exit=Budget.__exit__
    def fail(self,*args):
        previous_exit(self,*args)
        raise RuntimeError('STOP_RESOURCE_POLICY')
    monkeypatch.setattr(Budget,'__exit__',fail)
    with pytest.raises(RuntimeError,match='STOP_RESOURCE_POLICY'):workbench.seal_staging(out)
    assert not (out/'staged.json').exists() and (out/'seal_seal_staging_stop.json').exists()
    with pytest.raises(ValueError,match='DO_NOT_RESTART'):workbench.remaining(out)


def test_wrong_output_root_is_refused_without_creating_files(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);cfg=json.loads(workbench.CONFIG.read_text())
    cfg['output_root']=str(tmp_path/'only_this_root');write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='CATNAT-only'):workbench.init(out,record(source/'manifest.json'))
    assert not out.exists()

def test_new_certificates_keep_nonzero_global_swap_telemetry(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);workbench.init(out,record(source/'manifest.json'));finish(out)
    m=json.loads((out/'manifest.json').read_text())
    assert m['seal_resources']['policy_id']==POLICY
    assert m['seal_resources']['swapout_pages_delta']>0
    assert m['seal_resources']['global_swap_is_worker_attributed'] is False
    assert m['status'].startswith('SEALED_V154')
    assert workbench.config()['resources']==LIMITS


def test_business_compilers_are_identical_not_a_new_candidate_policy():
    import scripts.compile_retrieval_v148_address_store as old
    assert all(getattr(workbench,k) is getattr(old,k) for k in ('partition','dictionary','stage','pack'))
    assert workbench.PHASES==old.PHASES


def test_reordered_or_duplicate_source_parts_cannot_be_accepted(tmp_path,monkeypatch):
    source,out=setup(tmp_path,monkeypatch);path=source/'manifest.json';m=json.loads(path.read_text())
    m['parts'][0],m['parts'][1]=m['parts'][1],m['parts'][0];write_json(path,m)
    with pytest.raises(ValueError,match='physical part/order'):workbench.init(out,record(path))
    assert not out.exists()
