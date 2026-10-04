"""Toy source proofs/pressure fixtures; NOT national resource measurements."""
import json

import numpy as np
import pytest

import scripts.compile_retrieval_v162_address_access as access
import scripts.resume_retrieval_v160_reverse_store as resume
from src.xgb_matcher.retrieval_address_access_v162 import NationalAddresses
from src.xgb_matcher.retrieval_address_authority_v162 import store, epochs
from src.xgb_matcher.retrieval_address_reverse_v147 import ReversePart
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_retrieval_address_authority_v161 import setup as source_setup
from test_resume_retrieval_v153_addresses import gate


def setup(tmp_path, monkeypatch, complete=True):
    root, _, cfg = source_setup(tmp_path, monkeypatch, complete)
    cfg['scope'] = json.loads(access.CONFIG.read_text())['scope']
    path = tmp_path/'v162_config.json'
    write_json(path, cfg)
    monkeypatch.setattr(access, 'CONFIG', path)
    monkeypatch.setattr(access, 'quiet', gate)
    return root, tmp_path/'v162_access', cfg


def finish(output):
    l, p = access.load(output)
    for index in range(len(p['parts'])):
        dest = access.directory(output, 'part', index)
        dest.mkdir(parents=True)
        write_json(dest/'preflight.json', gate())
        access.worker(output, 'part', index)
    access.seal_parts(output)
    for ch in range(4):
        for index in range(l['config']['term_buckets']):
            dest = access.directory(output, 'bucket', index, ch)
            dest.mkdir(parents=True)
            write_json(dest/'preflight.json', gate())
            access.worker(output, 'bucket', index, ch)
    access.seal(output)


def test_same_full_source_native_postings_and_all_original_facts(tmp_path, monkeypatch):
    root, out, cfg = setup(tmp_path, monkeypatch)
    ref = record(root/'manifest.json')
    m, d = store(ref, cfg)
    assert m['status'] == resume.GLOBAL and m['aliases'] == d['aliases'] == 2
    access.init(out, ref)
    finish(out)
    _, p = access.load(out)
    assert p['source_epochs'] == epochs(ref, m, d)
    reader = NationalAddresses(record(out/'manifest.json'))
    try:
        for text in ('RUE TEST', 'AUTRE', '123456789', ''):
            snap, stats = reader.snapshot(text, chunk=1)
            assert not snap.coverage.flags.writeable and not snap.exact.flags.writeable
            assert not stats['candidates_emitted']
            facts = 0
            for interval in m['intervals']:
                part = ReversePart(interval)
                try:
                    ids = np.arange(interval['start'], interval['stop'], dtype=np.uint32)
                    facts += sum(len(b.site_positions) for b in part.batches(ids, snap, chunk=1))
                finally:
                    part.close()
            assert facts == p['facts'] == d['facts']
        reader.verify_after()
    finally:
        reader.close()
    assert ref == record(root/'manifest.json')
    cert = json.loads((out/'manifest.json').read_text())
    assert cert['lexical_address_access_built']
    assert not cert['complete_retrieval_index'] and not cert['ready_for_fold1']
    assert all(cert[k] is False for k in access.PROTECTED)


def test_prefix_fully_audited_before_global_parse(tmp_path, monkeypatch):
    root, _, cfg = setup(tmp_path, monkeypatch)
    ref = record(root/'manifest.json')
    full = (root/'manifest.json').read_text()
    previous_prefix, previous_parse = resume.prefix, json.loads
    calls = []
    def audit(*args):
        result = previous_prefix(*args)
        calls.append('prefix_complete')
        return result
    def parse(data, *args, **kwargs):
        if data == full:
            assert calls == ['prefix_complete']
            calls.append('global_parse')
        return previous_parse(data, *args, **kwargs)
    monkeypatch.setattr(resume, 'prefix', audit)
    monkeypatch.setattr(json, 'loads', parse)
    store(ref, cfg)
    assert calls == ['prefix_complete', 'global_parse']


@pytest.mark.parametrize('field', ['fold0_opened', 'fold1_opened', 'truth_opened',
                                  'complete_retrieval_index', 'stopped_reverse_phase_imported'])
def test_scope_stop_diagnostic_not_launch(tmp_path, monkeypatch, field):
    root, out, cfg = setup(tmp_path, monkeypatch)
    path = root/'manifest.json'
    m = json.loads(path.read_text())
    m[field] = True
    write_json(path, m)
    with pytest.raises(ValueError, match='Complete protected V160'):
        access.init(out, record(path))
    stopped = json.loads((out/'init_stop.json').read_text())
    assert stopped['status'] == 'STOP' and stopped['resources'] is not None
    assert not (out/'plan.json').exists() and not (out/'launch.json').exists()
    with pytest.raises(ValueError, match='DO_NOT_RESTART'):
        access.load(out)
    with pytest.raises(FileExistsError):
        access.init(out, record(path))


def test_absent_global_still_no_quiet_or_output(tmp_path, monkeypatch):
    root, out, cfg = setup(tmp_path, monkeypatch, complete=False)
    monkeypatch.setattr(access, 'quiet', lambda *a: pytest.fail('Missing source must not wait'))
    with pytest.raises(ValueError, match='NO_WORKER_LAUNCHED'):
        access.init(out, dict(path=str(root/'manifest.json'), sha256='0'*64))
    assert not out.exists()


def test_init_resource_failure_retains_actual_report_and_never_plan(tmp_path, monkeypatch):
    root, out, cfg = setup(tmp_path, monkeypatch)
    previous = Budget.__exit__
    def fail(self, *args):
        previous(self, *args)
        self.failure = 'STOP_RSS_OR_WALL_RESOURCE_POLICY'
        raise RuntimeError(self.failure)
    monkeypatch.setattr(Budget, '__exit__', fail)
    with pytest.raises(RuntimeError, match='RESOURCE_POLICY'):
        access.init(out, record(root/'manifest.json'))
    stopped = json.loads((out/'init_stop.json').read_text())
    assert stopped['resources']['pass_resource_policy'] is False
    assert stopped['resources']['peak_rss_bytes'] >= 0
    assert stopped['resources']['wall_seconds'] >= 0
    assert stopped['resources']['limits'] == cfg['resources']
    assert all(stopped[k] is False for k in access.PROTECTED)
    assert not (out/'plan.json').exists() and not (out/'launch.json').exists()


@pytest.mark.parametrize('field,value', [('exit_code', 1), ('exit_code', False),
    ('own_child_exit_confirmed', False), ('minimum_system_free_percent', 49),
    ('peak_rss_bytes', 1073741825), ('wall_seconds', 81), ('swapout_pages_delta', -1)])
def test_full_worker_success_needs_each_confirmed_source_parent(tmp_path, monkeypatch, field, value):
    root, _, cfg = setup(tmp_path, monkeypatch)
    path = resume.directory(root, 'reverse-pack', 0)/'supervision.json'
    m = json.loads(path.read_text())
    m[field] = value
    write_json(path, m)
    with pytest.raises(ValueError, match='UNCONFIRMED_PARENT_EXIT'):
        store(record(root/'manifest.json'), cfg)


def test_removed_whole_interval_and_source_hash_drift_refused(tmp_path, monkeypatch):
    root, _, cfg = setup(tmp_path, monkeypatch)
    path = root/'manifest.json'
    m = json.loads(path.read_text())
    m['intervals'].pop()
    write_json(path, m)
    ref = record(path)
    with pytest.raises(ValueError, match='Every whole-site'):
        store(ref, cfg)
    with path.open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='Authority drift'):
        store(ref, cfg)


def test_kernels_and_policy_unchanged():
    import scripts.compile_retrieval_v155_address_access as old
    assert access.physical_plan is old.physical_plan
    assert access.stage_addresses is old.stage_addresses
    assert access.pack is old.pack
    cfg = access.config()
    assert cfg['resources'] == old.config()['resources']
    assert cfg['scope'].startswith('ALL_COMPLETE_V160_')
