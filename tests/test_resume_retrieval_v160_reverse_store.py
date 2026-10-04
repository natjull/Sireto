import json

import numpy as np
import pytest

import scripts.compile_retrieval_v154_address_store as old
import scripts.resume_retrieval_v160_reverse_store as workbench
from src.xgb_matcher.retrieval_address_reverse_v147 import ReversePart, AliasSnapshot
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_compile_retrieval_v154_address_store import setup as source_setup, finish_phase
from test_resume_retrieval_v153_addresses import gate


def setup(tmp_path, monkeypatch):
    source, prefix = source_setup(tmp_path, monkeypatch)
    old.init(prefix, record(source/'manifest.json'))
    finish_phase(prefix, 'part')
    old.seal_staging(prefix)
    finish_phase(prefix, 'dictionary')
    old.seal_dictionary(prefix)
    stop = old.directory(prefix, 'reverse-stage', 0)/'status.json'
    stop.parent.mkdir(parents=True)
    write_json(stop, dict(status='STOP', reason='STOP_RSS_OR_WALL_RESOURCE_POLICY',
                         **{k: False for k in old.PROTECTED}))
    # Orphan bytes must stay intact and must never be imported as success.
    orphan = stop.parent/'orphan.parquet'
    orphan.write_bytes(b'not an authority')
    cfg = json.loads(workbench.CONFIG.read_text())
    cfg.update(dictionary=record(prefix/'dictionary.json'), original_stop=record(stop),
        expected_sites=11, expected_source_rows=6, expected_original_observations=12,
        expected_lexical_observations=4, expected_aliases=2, expected_facts=2,
        dictionary_buckets=2, site_quantum=4, writer_group_parts=1, output_root=str(tmp_path))
    path = tmp_path/'v160_config.json'
    write_json(path, cfg)
    monkeypatch.setattr(workbench, 'CONFIG', path)
    monkeypatch.setattr(workbench, 'quiet', gate)
    return prefix, tmp_path/'v160', stop, orphan


def finish_reverse_phase(output, phase):
    for i in range(workbench.maximum(output, phase)):
        dest = workbench.directory(output, phase, i)
        dest.mkdir(parents=True)
        write_json(dest/'preflight.json', gate())
        workbench.worker(output, phase, i)


def finish(output):
    finish_reverse_phase(output, 'reverse-stage')
    workbench.seal_staging(output)
    finish_reverse_phase(output, 'reverse-pack')
    workbench.seal(output)


def test_complete_original_prefix_and_all_sites_facts_without_importing_stop(tmp_path, monkeypatch):
    prefix, output, stop, orphan = setup(tmp_path, monkeypatch)
    before = record(stop), record(orphan), record(prefix/'dictionary.json')
    workbench.init(output)
    l, p = workbench.load(output)
    assert p['dictionary'] == before[2] and p['facts'] == 2 and p['sites'] == 11
    assert p['prefix_audit_resources']['swapout_pages_delta'] > 0
    assert p['prefix_audit_resources']['global_swap_is_worker_attributed'] is False
    with pytest.raises(ValueError, match='UNFINISHED'):
        workbench.seal_staging(output)
    finish(output)
    m = workbench.authority(output, 'manifest', workbench.GLOBAL)
    assert m['sites'] == 11 and m['aliases'] == 2 and m['facts'] == 2
    assert m['original_observations'] == 12 and m['lexical_observations'] == 4
    assert len(m['intervals']) == 3
    assert m['every_original_site_and_fact_preserved'] and m['all_original_observations_and_provenance_retained']
    assert not m['stopped_reverse_phase_imported'] and not m['complete_retrieval_index'] and not m['ready_for_fold1']
    assert all(m[k] is False for k in workbench.PROTECTED)
    assert before == (record(stop), record(orphan), record(prefix/'dictionary.json'))
    assert not (output/'part').exists() and not (output/'dictionary').exists()
    assert m['dictionary']['path'] == str(prefix/'dictionary.json')
    cov = np.zeros(2)
    exact = np.zeros(2, dtype=bool)
    cov.flags.writeable = exact.flags.writeable = False
    snapshot = AliasSnapshot(cov, exact)
    total = sites = 0
    for stats in m['intervals']:
        reader = ReversePart(stats)
        try:
            ids = np.arange(stats['start'], stats['stop'], dtype=np.uint32)
            sites += len(ids)
            total += sum(len(batch.site_positions) for batch in reader.batches(ids, snapshot, chunk=1))
        finally:
            reader.close()
    assert sites == 11 and total == 2
    for i in range(2):
        st = workbench.phase_record(output, 'reverse-stage', i)['stats']
        assert st['maximum_physical_writers_per_copy'] <= 1 and st['site_or_proof_or_alias_cap'] is None
    monkeypatch.setattr(workbench, 'execute', lambda *a: pytest.fail('Sealed phases must not rerun'))
    cert = record(output/'manifest.json')
    workbench.remaining(output)
    assert cert == record(output/'manifest.json')
    with pytest.raises(FileExistsError):
        workbench.init(output)


@pytest.mark.parametrize('field', ['truth_opened', 'fold0_opened', 'fold1_opened', 'crm_opened', 'complete_retrieval_index'])
def test_protected_prefix_cannot_be_reinterpreted(tmp_path, monkeypatch, field):
    prefix, output, _, _ = setup(tmp_path, monkeypatch)
    path = prefix/'dictionary.json'
    m = json.loads(path.read_text())
    m[field] = True
    write_json(path, m)
    cfg = json.loads(workbench.CONFIG.read_text())
    cfg['dictionary'] = record(path)
    write_json(workbench.CONFIG, cfg)
    with pytest.raises(ValueError, match='Protected complete'):
        workbench.init(output)
    assert not output.exists()


def test_missing_or_stopped_original_prefix_phase_blocks_init(tmp_path, monkeypatch):
    prefix, output, _, _ = setup(tmp_path, monkeypatch)
    write_json(old.directory(prefix, 'dictionary', 1)/'status.json', {'status': 'STOP'})
    with pytest.raises(ValueError, match='STOP_PHASE'):
        workbench.init(output)
    assert not output.exists()


def test_no_claim_based_on_subset_of_dictionary_buckets(tmp_path, monkeypatch):
    prefix, output, _, _ = setup(tmp_path, monkeypatch)
    path = prefix/'dictionary.json'
    m = json.loads(path.read_text())
    m['buckets'].pop()
    write_json(path, m)
    cfg = json.loads(workbench.CONFIG.read_text())
    cfg['dictionary'] = record(path)
    write_json(workbench.CONFIG, cfg)
    with pytest.raises(ValueError, match='Every original dictionary bucket'):
        workbench.init(output)


def test_namespace_and_denominator_changes_stop(tmp_path, monkeypatch):
    _, output, _, _ = setup(tmp_path, monkeypatch)
    cfg = json.loads(workbench.CONFIG.read_text())
    cfg['expected_facts'] += 1
    write_json(workbench.CONFIG, cfg)
    with pytest.raises(ValueError, match='Entire original'):
        workbench.init(output)


def test_hot_load_epoch_checks_not_whole_prefix_hash_per_physical_chunk(tmp_path, monkeypatch):
    _, output, _, _ = setup(tmp_path, monkeypatch)
    workbench.init(output)
    monkeypatch.setattr(workbench, 'prefix', lambda *a: pytest.fail('No repeated global audit per chunk'))
    _, p = workbench.load(output)
    item = next(e for e in p['prefix_epochs'] if e['reference']['path'].endswith('facts.parquet'))
    with open(item['reference']['path'], 'ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='epoch drift'):
        workbench.load(output)


def test_partial_or_stopped_new_phase_is_never_retried(tmp_path, monkeypatch):
    _, output, _, _ = setup(tmp_path, monkeypatch)
    workbench.init(output)
    dest = workbench.directory(output, 'reverse-stage', 0)
    dest.mkdir(parents=True)
    write_json(dest/'preflight.json', gate())
    workbench.worker(output, 'reverse-stage', 0)
    write_json(dest/'status.json', dict(status='STOP', reason='supervisor'))
    with pytest.raises(ValueError, match='STOP_PHASE'):
        workbench.remaining(output)
    with pytest.raises(ValueError, match='STOP_PHASE'):
        workbench.seal_staging(output)
    with pytest.raises(FileExistsError):
        workbench.execute(output, 'reverse-stage', 0)


def test_new_seal_resource_failure_publishes_stop_not_success(tmp_path, monkeypatch):
    _, output, _, _ = setup(tmp_path, monkeypatch)
    workbench.init(output)
    finish_reverse_phase(output, 'reverse-stage')
    before = Budget.__exit__
    def fail(self, *args):
        before(self, *args)
        raise RuntimeError('STOP_RESOURCE_POLICY')
    monkeypatch.setattr(Budget, '__exit__', fail)
    with pytest.raises(RuntimeError, match='RESOURCE_POLICY'):
        workbench.seal_staging(output)
    assert (output/'seal_seal_staging_stop.json').exists()
    assert not (output/'reverse_staged.json').exists()
    with pytest.raises(ValueError, match='STOP_SEAL'):
        workbench.remaining(output)


def test_resource_policy_cannot_relax_or_change_output_root(tmp_path, monkeypatch):
    _, output, _, _ = setup(tmp_path, monkeypatch)
    cfg = json.loads(workbench.CONFIG.read_text())
    cfg['resources']['max_rss_bytes'] *= 2
    write_json(workbench.CONFIG, cfg)
    with pytest.raises(ValueError, match='unchanged resource'):
        workbench.init(output)
    cfg['resources']['max_rss_bytes'] //= 2
    cfg['output_root'] = str(tmp_path/'different')
    write_json(workbench.CONFIG, cfg)
    with pytest.raises(ValueError, match='CATNAT-only'):
        workbench.init(output)


def test_native_storage_kernels_are_the_audited_versions():
    from src.xgb_matcher.retrieval_address_stage_v159 import stage
    assert workbench.stage is stage
    assert workbench.pack is old.pack
