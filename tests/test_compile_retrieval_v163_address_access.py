"""Split audit fixtures: fake supervision/pressure, no national claims."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

import scripts.compile_retrieval_v163_address_access as access
import scripts.resume_retrieval_v160_reverse_store as resume
from src.xgb_matcher.retrieval_address_access_v163 import NationalAddresses
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, POLICY
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_compile_retrieval_v162_address_access import setup as previous_setup
from test_resume_retrieval_v153_addresses import gate


def supervisor_fixture(args, log, env):
    """Explicit in-process fixture, NOT a measured own-child exit."""
    output = Path(args[args.index('--output')+1])
    if 'audit-worker' in args:
        access.audit_worker(output, args[args.index('--audit-tag')+1], args[args.index('--audit-kind')+1])
    else:
        channel = int(args[args.index('--channel')+1]) if '--channel' in args else None
        access.worker(output, args[args.index('--phase')+1], int(args[args.index('--index')+1]), channel)
    return dict(policy_id=POLICY, limits=LIMITS, child_pid=1, exit_code=0,
        own_child_exit_confirmed=True, peak_rss_bytes=1, minimum_system_free_percent=73,
        sample_count=1, wall_seconds=0., swapout_pages_delta=0, global_swap_is_worker_attributed=False)


def setup(tmp_path, monkeypatch, complete=True):
    root, _, cfg = previous_setup(tmp_path, monkeypatch, complete)
    cfg['scope'] = json.loads(access.CONFIG.read_text())['scope']
    path = tmp_path/'v163_config.json'
    write_json(path, cfg)
    monkeypatch.setattr(access, 'CONFIG', path)
    monkeypatch.setattr(access, 'quiet', gate)
    monkeypatch.setattr(access, 'supervise', supervisor_fixture)
    return root, tmp_path/'v163_access', cfg


def test_split_audits_all_raw_sources_then_reverse_then_same_lexical_channels(tmp_path, monkeypatch):
    root, out, cfg = setup(tmp_path, monkeypatch)
    before = record(root/'manifest.json')
    calls = []
    previous = resume.prefix
    def audit(*args):
        result = previous(*args)
        calls.append('full_prefix')
        return result
    monkeypatch.setattr(resume, 'prefix', audit)
    access.init(out, before)
    assert calls == ['full_prefix']
    l, p = access.load(out)
    assert len(p['parts']) == 2 and p['sites'] == 11 and p['facts'] == 2
    assert p['source_store'] == before
    assert all(p[k] is False for k in access.PROTECTED)
    assert p['prefix_audit_resources']['limits'] == p['source_audit_resources']['limits'] == LIMITS
    access.remaining(out)
    assert calls == ['full_prefix', 'full_prefix']  # Complete rehash again at seal.
    m = json.loads((out/'manifest.json').read_text())
    assert m['lexical_address_access_built'] and not m['complete_retrieval_index']
    assert not m['ready_for_fold1'] and all(m[k] is False for k in access.PROTECTED)
    reader = NationalAddresses(record(out/'manifest.json'))
    try:
        snap, stats = reader.snapshot('RUE TEST', chunk=1)
        d = json.loads(Path(p['dictionary']['path']).read_text())
        texts = [text for b in d['buckets'] for text in pq.read_table(b['artifacts']['addresses.parquet']['path'])['address_text'].to_pylist()]
        query = channels('RUE TEST')
        expected = [max(len(q&s)/len(q) if q else 0. for q,s in zip(query,channels(text))) for text in texts]
        assert np.array_equal(snap.coverage, np.array(expected))
        assert np.array_equal(snap.exact, np.array([normalized(text)==normalized('RUE TEST') for text in texts]))
        assert not snap.coverage.flags.writeable and not stats['candidates_emitted']
        reader.verify_after()
    finally:
        reader.close()
    assert before == record(root/'manifest.json')
    final = record(out/'manifest.json')
    monkeypatch.setattr(access, 'supervise', lambda *a: pytest.fail('No completed phases rerun'))
    access.remaining(out)
    assert final == record(out/'manifest.json')


def test_missing_store_no_root_or_quiet(tmp_path, monkeypatch):
    root, out, cfg = setup(tmp_path, monkeypatch, complete=False)
    monkeypatch.setattr(access, 'quiet', lambda *a: pytest.fail('Missing source must not wait'))
    with pytest.raises(ValueError, match='NO_WORKER_LAUNCHED'):
        access.init(out, dict(path=str(root/'manifest.json'), sha256='0'*64))
    assert not out.exists()


@pytest.mark.parametrize('kind', ['prefix', 'reverse'])
def test_each_resource_stop_is_retained_without_plan_or_retry(tmp_path, monkeypatch, kind):
    root, out, cfg = setup(tmp_path, monkeypatch)
    previous = access.audit_worker
    def audit(output, tag, role):
        if role != kind:
            return previous(output, tag, role)
        # Exercise actual worker exception/export path via Budget failure.
        exit_method = Budget.__exit__
        def fail(self, *args):
            exit_method(self, *args)
            self.failure = 'STOP_RSS_OR_WALL_RESOURCE_POLICY'
            raise RuntimeError(self.failure)
        monkeypatch.setattr(Budget, '__exit__', fail)
        return previous(output, tag, role)
    monkeypatch.setattr(access, 'audit_worker', audit)
    with pytest.raises(RuntimeError, match='RESOURCE_POLICY'):
        access.init(out, record(root/'manifest.json'))
    stop = json.loads((out/'source_audits'/'init'/kind/'worker_stop.json').read_text())
    assert stop['resources']['pass_resource_policy'] is False
    assert stop['resources']['limits'] == LIMITS
    assert (out/'init_stop.json').exists() and not (out/'plan.json').exists()
    assert not (out/'launch.json').exists()
    with pytest.raises(FileExistsError):
        access.init(out, record(root/'manifest.json'))


def test_drift_between_prefix_and_reverse_cannot_pass(tmp_path, monkeypatch):
    root, out, cfg = setup(tmp_path, monkeypatch)
    previous = access.audit_worker
    def audit(output, tag, kind):
        if kind == 'reverse':
            a = json.loads((output/'source_audits'/tag/'prefix'/'manifest.json').read_text())
            ref = next(e['reference'] for e in a['source_epochs'] if e['reference']['path'].endswith('addresses.parquet'))
            with open(ref['path'], 'ab') as f:
                f.write(b'changed')
        return previous(output, tag, kind)
    monkeypatch.setattr(access, 'audit_worker', audit)
    with pytest.raises(ValueError, match='epoch drift'):
        access.init(out, record(root/'manifest.json'))
    assert not (out/'plan.json').exists() and not (out/'launch.json').exists()


@pytest.mark.parametrize('field,value', [('exit_code', 1), ('exit_code', False),
    ('own_child_exit_confirmed', False), ('peak_rss_bytes', 1073741825)])
def test_supervised_success_required_for_each_audit_before_plan(tmp_path, monkeypatch, field, value):
    root, out, cfg = setup(tmp_path, monkeypatch)
    def invalid(*args):
        report = supervisor_fixture(*args)
        report[field] = value
        return report
    monkeypatch.setattr(access, 'supervise', invalid)
    with pytest.raises(ValueError, match='UNCONFIRMED_PARENT_EXIT'):
        access.init(out, record(root/'manifest.json'))
    assert not (out/'plan.json').exists() and (out/'init_stop.json').exists()


@pytest.mark.parametrize('field', ['fold0_opened', 'fold1_opened', 'truth_opened', 'complete_retrieval_index'])
def test_closed_full_source_scope_still_required(tmp_path, monkeypatch, field):
    root, out, cfg = setup(tmp_path, monkeypatch)
    path = root/'manifest.json'
    m = json.loads(path.read_text())
    m[field] = True
    write_json(path, m)
    with pytest.raises(ValueError, match='Complete protected V160'):
        access.init(out, record(path))
    assert not (out/'plan.json').exists() and not (out/'launch.json').exists()


def test_hot_load_and_final_seal_audit_supervision_pinned(tmp_path, monkeypatch):
    root, out, cfg = setup(tmp_path, monkeypatch)
    access.init(out, record(root/'manifest.json'))
    path = out/'source_audits'/'init'/'prefix'/'supervision.json'
    report = json.loads(path.read_text())
    report['child_pid'] = 2  # Still well typed but not the originally sealed report.
    write_json(path, report)
    with pytest.raises(ValueError, match='Same actual source audit supervision'):
        access.load(out)


def test_same_kernels_and_limits_no_candidate_reservoir():
    import scripts.compile_retrieval_v162_address_access as old
    assert access.physical_plan is old.physical_plan
    assert access.stage_addresses is old.stage_addresses
    assert access.pack is old.pack
    assert access.config()['resources'] == old.config()['resources'] == LIMITS


def test_lexical_worker_success_is_not_a_confirmed_parent_exit(tmp_path, monkeypatch):
    root, out, cfg = setup(tmp_path, monkeypatch)
    access.init(out, record(root/'manifest.json'))
    access.execute(out, 'part', 0)
    dest = access.directory(out, 'part', 0)
    path = dest/'supervision.json'
    report = json.loads(path.read_text())
    report['own_child_exit_confirmed'] = False
    write_json(path, report)
    with pytest.raises(ValueError, match='UNCONFIRMED_PARENT_EXIT'):
        access.phase_record(out, 'part', 0)
    with pytest.raises(ValueError, match='UNCONFIRMED_PARENT_EXIT'):
        access.remaining(out)
    assert not (out/'staging_manifest.json').exists()
