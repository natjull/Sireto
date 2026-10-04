"""Every tiny file actually hashed; no process/resource/national simulation here."""
from copy import deepcopy
import os

import pytest

from src.xgb_matcher import retrieval_postscore_audit_v187 as workbench
from src.xgb_matcher.retrieval_v20_workbench import record, write_json


def fixture(tmp_path):
    refs = {}
    for group in workbench.GROUPS:
        p = tmp_path / (group+'.json')
        write_json(p, {'official_style_fixture': group})
        refs[group] = record(p)
    epochs = [dict(reference=r, fingerprint=list(workbench.fingerprint(r['path']))) for r in refs.values()]
    groups = {k: [r] for k, r in refs.items()}
    spans = [dict(start=0, stop=3, facts=7), dict(start=3, stop=9, facts=10)]
    packets = [workbench.packet(groups, epochs, 'a'*64, span, 9, 17) for span in spans]
    return refs, epochs, packets


def test_all_original_files_full_hash_targets_exact_after_all_sites_and_facts(tmp_path):
    refs, epochs, packets = fixture(tmp_path)
    plan = workbench.plan(packets, 9, 17)
    progress = []
    receipts = [workbench.audit_group(plan, group, progress.append) for group in workbench.GROUPS]
    workbench.require_receipts(plan, receipts)
    assert {r['group'] for r in receipts} == set(workbench.GROUPS)
    assert all(r['files'] == 1 and r['every_expected_source_full_hash_verified'] for r in receipts)
    assert len(progress) == 3
    assert all(not r['initial100_emitted'] and not r['truth_opened'] for r in receipts)
    # Captured epochs are canonical but no reference checksum was dropped.
    assert len(plan['source_epochs']) == 3 and sum(len(v) for v in plan['groups'].values()) == 3


def test_optional_size_legacy_and_current_refs_same_bytes_but_not_different_hash(tmp_path):
    refs, epochs, packets = fixture(tmp_path)
    ref = refs['facts']
    legacy = {k: ref[k] for k in ('path', 'sha256')}
    assert workbench.references([ref, legacy]) == [legacy]
    assert len(workbench.canonical_epochs([epochs[0], {**epochs[0], 'reference': legacy}])) == 1
    with pytest.raises(ValueError, match='Contradictory source hashes'):
        workbench.references([ref, {**legacy, 'sha256': 'f'*64}])
    with pytest.raises(ValueError, match='sizes'):
        workbench.references([ref, {**ref, 'size_bytes': ref['size_bytes']+1}])


@pytest.mark.parametrize('change', ['missing_tail', 'overlap', 'signature', 'wrong_global_sites',
                                   'missing_facts', 'false_epoch', 'changed_hash', 'truth', 'unknown_group'])
def test_incomplete_or_different_physical_world_never_creates_audit_plan(tmp_path, change):
    refs, epochs, packets = fixture(tmp_path)
    altered = deepcopy(packets)
    if change == 'missing_tail':
        altered.pop()
    elif change == 'overlap':
        altered[1]['span']['start'] = 2
    elif change == 'signature':
        altered[1]['signature'] = 'b'*64
    elif change == 'wrong_global_sites':
        altered[1]['sites'] = 10
    elif change == 'missing_facts':
        altered[1]['span']['facts'] -= 1
    elif change == 'false_epoch':
        altered[1]['source_epochs'][0]['fingerprint'][3] += 1
    elif change == 'changed_hash':
        altered[1]['groups']['facts'][0]['sha256'] = 'b'*64
    elif change == 'truth':
        altered[1]['truth_opened'] = True
    else:
        altered[1]['groups']['label_side'] = altered[1]['groups'].pop('facts')
    with pytest.raises(ValueError):
        workbench.plan(altered, 9, 17)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_signature', 'wrong_files',
                                   'wrong_target', 'unverified', 'truth', 'wrong_bytes'])
def test_partial_or_wrong_receipts_never_release_pending_collectors(tmp_path, change):
    refs, epochs, packets = fixture(tmp_path)
    plan = workbench.plan(packets, 9, 17)
    receipts = [workbench.audit_group(plan, g) for g in workbench.GROUPS]
    if change == 'missing':
        receipts.pop()
    elif change == 'duplicate':
        receipts[1] = receipts[0]
    elif change == 'wrong_signature':
        receipts[0]['signature'] = 'b'*64
    elif change == 'wrong_files':
        receipts[0]['files'] = 0
    elif change == 'wrong_target':
        receipts[0]['exact_target_digest'] = 'b'*64
    elif change == 'unverified':
        receipts[0]['every_expected_source_full_hash_verified'] = False
    elif change == 'truth':
        receipts[0]['truth_opened'] = True
    else:
        receipts[0]['full_bytes_hashed'] += 1
    with pytest.raises(ValueError):
        workbench.require_receipts(plan, receipts)


def test_same_inode_size_mtime_mutation_still_refused_by_full_file_hash(tmp_path):
    refs, epochs, packets = fixture(tmp_path)
    plan = workbench.plan(packets, 9, 17)
    p = __import__('pathlib').Path(refs['facts']['path'])
    old = p.stat()
    data = p.read_bytes()
    p.write_bytes(data.replace(b'facts', b'FACTS'))  # Own fixture only, same width.
    os.utime(p, ns=(old.st_atime_ns, old.st_mtime_ns))
    workbench.check_epochs(plan['source_epochs'])  # Demonstrates why epoch alone is insufficient.
    with pytest.raises(ValueError):
        workbench.audit_group(plan, 'facts')


def test_late_drift_in_other_group_stops_the_current_group_audit(tmp_path):
    refs, epochs, packets = fixture(tmp_path)
    plan = workbench.plan(packets, 9, 17)
    def changed(unused):
        write_json(refs['context']['path'], {'different': True})
    with pytest.raises(ValueError, match='epoch drift'):
        workbench.audit_group(plan, 'facts', changed)
