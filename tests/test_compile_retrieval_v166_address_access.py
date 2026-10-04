"""Tiny fixtures and fake process/pressure reports; no national cost claims."""
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.compile_retrieval_v163_address_access as source
import scripts.compile_retrieval_v166_address_access as access
from src.xgb_matcher.retrieval_address_access_v166 import NationalAddresses
from src.xgb_matcher.retrieval_memory_policy_v153 import LIMITS, POLICY
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_compile_retrieval_v163_address_access import setup as source_setup
from test_resume_retrieval_v153_addresses import gate


def supervise_fixture(args,log,env,stop_path):
    root=Path(args[args.index('--output')+1])
    if 'audit-worker' in args:
        access.audit_worker(root,args[args.index('--audit-tag')+1],args[args.index('--audit-kind')+1])
    else:
        channel=int(args[args.index('--channel')+1]) if '--channel' in args else None
        access.worker(root,args[args.index('--phase')+1],int(args[args.index('--index')+1]),channel)
    return dict(policy_id=POLICY,limits=LIMITS,child_pid=1,exit_code=0,
        own_child_exit_confirmed=True,peak_rss_bytes=1,minimum_system_free_percent=73,
        sample_count=1,wall_seconds=0.,swapout_pages_delta=0,global_swap_is_worker_attributed=False)


def setup(tmp_path,monkeypatch,complete=True):
    raw,old,oldcfg=source_setup(tmp_path,monkeypatch)
    if complete:source.init(old,record(raw/'manifest.json'))
    cfg=json.loads(access.CONFIG.read_text())
    cfg.update({k:oldcfg[k] for k in ('expected_sites','expected_original_observations','expected_source_rows',
        'output_root','term_buckets','batch_rows')})
    path=tmp_path/'v166_config.json';write_json(path,cfg)
    monkeypatch.setattr(access,'CONFIG',path);monkeypatch.setattr(access,'quiet',gate)
    monkeypatch.setattr(access,'supervise',supervise_fixture)
    return old,tmp_path/'v166_access'


def test_complete_source_init_reused_but_stopped_lexical_and_probe_never_imported(tmp_path,monkeypatch):
    old,out=setup(tmp_path,monkeypatch)
    dest=source.directory(old,'part',0);dest.mkdir(parents=True)
    write_json(dest/'status.json',dict(status='STOP',reason='original native lexical'))
    original=record(old/'launch.json');stopped=record(dest/'status.json')
    access.init(out,original)
    l,p=access.load(out)
    _,oldp=source.load(old)
    assert p['parts']==oldp['parts'] and p['source_epochs']==oldp['source_epochs']
    assert p['sites']==11 and p['aliases']==2 and p['facts']==2
    assert not p['stopped_lexical_phase_or_probe_imported']
    assert not (out/'parts').exists() and not (out/'source_audits'/'init').exists()
    assert p['source_audits']==oldp['source_audits']  # Original V163 titles/hashes, not relabeled.
    access.remaining(out)
    m=json.loads((out/'manifest.json').read_text())
    assert m['lexical_address_access_built'] and not m['complete_retrieval_index'] and not m['ready_for_fold1']
    assert all(m[k] is False for k in access.PROTECTED)
    assert original==record(old/'launch.json') and stopped==record(dest/'status.json')
    own=access.phase_record(out,'part',0)
    assert own['stats']['all_original_channels_and_tokens_unchanged']
    assert own['stats']['maximum_physical_writers_per_copy']<=4
    reader=NationalAddresses(record(out/'manifest.json'))
    try:
        snap,stats=reader.snapshot('RUE TEST',chunk=1)
        assert snap.coverage.shape==(2,) and not snap.coverage.flags.writeable
        assert np.all((snap.coverage>=0)&(snap.coverage<=1))
        assert not stats['candidates_emitted'];reader.verify_after()
    finally:reader.close()
    final=record(out/'manifest.json')
    monkeypatch.setattr(access,'supervise',lambda *a:pytest.fail('No completed phases rerun'))
    access.remaining(out);assert final==record(out/'manifest.json')


def test_missing_source_init_no_quiet_no_output(tmp_path,monkeypatch):
    old,out=setup(tmp_path,monkeypatch,complete=False)
    monkeypatch.setattr(access,'quiet',lambda *a:pytest.fail('Missing source before quiet'))
    with pytest.raises(ValueError,match='NO_WORKER_LAUNCHED'):
        access.init(out,dict(path=str(old/'launch.json'),sha256='0'*64))
    assert not out.exists()


def test_source_scope_and_truth_closed(tmp_path,monkeypatch):
    old,out=setup(tmp_path,monkeypatch)
    path=old/'launch.json';m=json.loads(path.read_text());m['truth_opened']=True;write_json(path,m)
    with pytest.raises(ValueError,match='immutable address access launch'):
        access.init(out,record(path))
    assert (out/'init_stop.json').exists() and not (out/'plan.json').exists()


def test_unknown_manifest_and_original_stopped_lexical_cannot_supply_sources(tmp_path,monkeypatch):
    old,out=setup(tmp_path,monkeypatch)
    stopped=tmp_path/'status.json';write_json(stopped,dict(status='STOP'))
    with pytest.raises(ValueError,match='source INIT launch'):
        access.init(out,record(stopped))
    assert not (out/'launch.json').exists()


def test_source_epoch_drift_cannot_reuse_old_successful_audit(tmp_path,monkeypatch):
    old,out=setup(tmp_path,monkeypatch)
    _,p=source.load(old)
    ref=next(e['reference'] for e in p['source_epochs'] if e['reference']['path'].endswith('addresses.parquet'))
    with open(ref['path'],'ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='epoch drift'):
        access.init(out,record(old/'launch.json'))
    assert not (out/'launch.json').exists()


def test_old_audit_actual_exit_required_even_with_complete_plan(tmp_path,monkeypatch):
    old,out=setup(tmp_path,monkeypatch)
    path=old/'source_audits'/'init'/'reverse'/'supervision.json'
    m=json.loads(path.read_text());m['own_child_exit_confirmed']=False;write_json(path,m)
    with pytest.raises(ValueError,match='UNCONFIRMED_PARENT_EXIT'):
        access.init(out,record(old/'launch.json'))
    assert not (out/'plan.json').exists()


def test_new_phase_worker_success_cannot_hide_supervisor_stop(tmp_path,monkeypatch):
    old,out=setup(tmp_path,monkeypatch);access.init(out,record(old/'launch.json'))
    access.execute(out,'part',0)
    path=access.directory(out,'part',0)/'supervisor_stop.json'
    write_json(path,dict(status='STOP',reason='RSS'))
    with pytest.raises(ValueError,match='STOP_PHASE_SUPERVISION'):
        access.remaining(out)
    assert not (out/'staging_manifest.json').exists()


@pytest.mark.parametrize('field,value',[('all_original_channels_and_tokens_unchanged',False),
    ('physical_writer_group_is_not_a_candidate_cap',False),('maximum_physical_writers_per_copy',9)])
def test_physical_writer_certificate_not_business_filter(tmp_path,monkeypatch,field,value):
    old,out=setup(tmp_path,monkeypatch);access.init(out,record(old/'launch.json'));access.execute(out,'part',0)
    path=access.directory(out,'part',0)/'manifest.json';m=json.loads(path.read_text());m['stats'][field]=value;write_json(path,m)
    with pytest.raises(ValueError,match='unchanged channels'):
        access.phase_record(out,'part',0)


def test_same_bounded_kernel_pack_and_pressure_limits():
    from src.xgb_matcher.retrieval_address_channels_v166 import stage_addresses
    import scripts.compile_retrieval_v163_address_access as old
    assert access.stage_addresses is stage_addresses
    assert access.physical_plan is old.physical_plan and access.pack is old.pack
    assert access.config()['resources']==LIMITS
    assert access.config()['writer_group']==4


def test_imported_source_metadata_and_config_cannot_implicitly_change(tmp_path,monkeypatch):
    old,out=setup(tmp_path,monkeypatch)
    cfg=json.loads(access.CONFIG.read_text());cfg['batch_rows']+=1;write_json(access.CONFIG,cfg)
    with pytest.raises(ValueError,match='Same entire original source'):
        access.init(out,record(old/'launch.json'))
    assert not (out/'launch.json').exists()
