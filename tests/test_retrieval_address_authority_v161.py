import json

import numpy as np
import pytest

import scripts.compile_retrieval_v155_address_access as access
import scripts.resume_retrieval_v160_reverse_store as resume
from src.xgb_matcher.retrieval_address_access_v155 import NationalAddresses
from src.xgb_matcher.retrieval_address_authority_v161 import store, epochs
from src.xgb_matcher.retrieval_initial_engine_v158 import CompleteAddressFacts
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
from src.xgb_matcher.retrieval_memory_policy_v153 import LIMITS, POLICY
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_resume_retrieval_v160_reverse_store import setup as source_setup, finish as source_finish
from test_compile_retrieval_v155_address_access import finish as access_finish
from test_resume_retrieval_v153_addresses import gate


def setup(tmp_path, monkeypatch, complete=True):
    prefix, root, _, _ = source_setup(tmp_path, monkeypatch)
    resume.init(root)
    if complete:
        source_finish(root)
        # Toy supervisor fixtures are not real process/pressure measurements.
        for phase in resume.PHASES:
            for i in range(resume.maximum(root, phase)):
                write_json(resume.directory(root, phase, i)/'supervision.json', dict(
                    policy_id=POLICY, limits=LIMITS, child_pid=1, exit_code=0,
                    own_child_exit_confirmed=True, peak_rss_bytes=1,
                    minimum_system_free_percent=73, sample_count=1,
                    wall_seconds=0., swapout_pages_delta=0,
                    global_swap_is_worker_attributed=False))
    cfg = json.loads(access.CONFIG.read_text())
    cfg.update(expected_sites=11, expected_original_observations=12, expected_source_rows=6,
               output_root=str(tmp_path), term_buckets=2, batch_rows=1)
    config = tmp_path/'v161_access_config.json'
    write_json(config, cfg)
    monkeypatch.setattr(access, 'CONFIG', config)
    monkeypatch.setattr(access, 'quiet', gate)
    return root, tmp_path/'v161_access', cfg


def test_explicit_full_v160_to_original_lexical_kernels_and_complete_site_facts(tmp_path, monkeypatch):
    root, output, cfg = setup(tmp_path, monkeypatch)
    ref = record(root/'manifest.json')
    m, d = store(ref, cfg)
    assert m['status'] == resume.GLOBAL and m['aliases'] == d['aliases'] == 2
    assert access.store(ref, cfg) == (m, d)
    original = record(root/'manifest.json')
    access.init(output, ref)
    access_finish(output)
    reader = NationalAddresses(record(output/'manifest.json'))
    facts = CompleteAddressFacts(ref)
    try:
        q = Query(address='RUE TEST')
        snapshot, _ = reader.snapshot(q.address)
        bound = facts.bind(snapshot, q, 1)
        ids = np.arange(11, dtype=np.uint32)
        assert sum(len(b.site_positions) for b in bound.batches(ids, q)) == 2
        bound.finish()
        facts.verify_after()
        reader.verify_after()
    finally:
        reader.close()
        facts.close()
    assert original == record(root/'manifest.json')  # Never rewrite/relabel source.
    l, p = access.load(output)
    assert p['source_store'] == ref and p['aliases'] == 2 and p['sites'] == 11
    assert p['dictionary']['path'] != str(root/'dictionary.json')
    assert p['source_epochs'] == epochs(ref, m, d)
    cert = json.loads((output/'manifest.json').read_text())
    assert not cert['complete_retrieval_index'] and not cert['ready_for_fold1']
    assert all(cert[k] is False for k in access.PROTECTED)


def test_partial_successes_without_full_v160_global_cannot_launch_lexical_worker(tmp_path, monkeypatch):
    root, output, cfg = setup(tmp_path, monkeypatch, complete=False)
    monkeypatch.setattr(access, 'quiet', lambda *a: pytest.fail('No worker/quiet without complete store'))
    with pytest.raises(ValueError, match='NO_WORKER_LAUNCHED'):
        access.init(output, {'path': str(root/'manifest.json'), 'sha256': '0'*64})
    assert not output.exists()


@pytest.mark.parametrize('field', ['fold0_opened', 'fold1_opened', 'truth_opened',
                                  'complete_retrieval_index', 'stopped_reverse_phase_imported'])
def test_bad_scope_cannot_be_accepted_under_another_backend(tmp_path, monkeypatch, field):
    root, output, cfg = setup(tmp_path, monkeypatch)
    path = root/'manifest.json'
    m = json.loads(path.read_text())
    m[field] = True
    write_json(path, m)
    with pytest.raises(ValueError, match='Complete protected V160'):
        access.init(output, record(path))
    assert not output.exists()


def test_removed_site_interval_stops_even_with_full_store_title(tmp_path, monkeypatch):
    root, output, cfg = setup(tmp_path, monkeypatch)
    path = root/'manifest.json'
    m = json.loads(path.read_text())
    m['intervals'].pop()
    write_json(path, m)
    with pytest.raises(ValueError, match='Every whole-site'):
        store(record(path), cfg)


def test_native_kernels_are_not_replaced_by_backend_adapter():
    import scripts.compile_retrieval_v152_address_access as old
    assert access.physical_plan is old.physical_plan
    assert access.stage_addresses is old.stage_addresses
    assert access.pack is old.pack


@pytest.mark.parametrize('field,value', [('exit_code', 1), ('exit_code', False),
    ('own_child_exit_confirmed', False), ('minimum_system_free_percent', 49),
    ('peak_rss_bytes', 1073741825), ('wall_seconds', 81), ('swapout_pages_delta', -1)])
def test_success_manifest_cannot_hide_unconfirmed_parent_exit(tmp_path, monkeypatch, field, value):
    root, _, cfg = setup(tmp_path, monkeypatch)
    path = resume.directory(root, 'reverse-pack', 0)/'supervision.json'
    m = json.loads(path.read_text())
    m[field] = value
    write_json(path, m)
    with pytest.raises(ValueError, match='UNCONFIRMED_PARENT_EXIT'):
        store(record(root/'manifest.json'), cfg)
