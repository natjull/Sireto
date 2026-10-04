"""Real tiny file/hash fixtures; no national data or recall claims."""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest

from src.xgb_matcher import retrieval_population_audit_v197 as whole
from src.xgb_matcher import retrieval_postscore_audit_v187 as hashes
from src.xgb_matcher.retrieval_query_population_v192 import Population
from src.xgb_matcher.retrieval_v20_workbench import record, write_json, verify
from test_retrieval_postscore_audit_v187 import fixture


def controls():
    return [dict(position=i, query_id='ONLY_UNIT_CONTROL_'+str(i), fold=2+i%3,
                 signature=whole.digest({'own_position_control': i})) for i in range(128)]


def population():
    rows = tuple(dict(query_id=r['query_id'], fold=r['fold'], crm_name='', crm_number='',
                      crm_suffix='', crm_street='', crm_postcode='', crm_insee='') for r in controls())
    return Population(rows, np.eye(128, 384, dtype=np.float32))


def outputs():
    return [dict(position=r['position'], query_id=r['query_id'], fold=r['fold'], candidates=[],
        trace=dict(signature=r['signature'], sites_completed=9, address_facts=17, physical_partitions=4,
            initial_cap=100, fixed_reservoir=None, score_after_fixed_pool=False,
            every_site_scored_once=True, same_original_score_never_rescored=True)) for r in controls()]


def test_all_original128_once_own_folds_and_control_recipes():
    assert [p for b in range(64) for p in whole.positions(b)] == list(range(128))
    assert whole.controls(controls(), list(range(128))) == controls()
    assert whole.outputs(outputs(), population(), controls(), 9, 17, 4) == outputs()


@pytest.mark.parametrize('batch', [-1, 64, True, '1', 1.])
def test_batch_is_never_coerced_or_clipped(batch):
    with pytest.raises(ValueError): whole.positions(batch)


@pytest.mark.parametrize('change', ['partial', 'duplicate', 'fold0', 'fold1', 'boolfold',
                                    'swap', 'foreign_recipe', 'query_mixed', 'cap101', 'source', 'facts', 'rerank'])
def test_full128_or_nothing_never_selects_an_easier_denominator(change):
    result, c = outputs(), controls()
    if change == 'partial': result.pop()
    elif change == 'duplicate': result[-1] = deepcopy(result[-2])
    elif change in ('fold0', 'fold1', 'boolfold'): c[-1]['fold'] = {'fold0': 0, 'fold1': 1, 'boolfold': True}[change]
    elif change == 'swap': result[-2:] = result[-2:][::-1]
    elif change == 'foreign_recipe': result[-1]['trace']['signature'] = c[0]['signature']
    elif change == 'query_mixed': result[-1]['query_id'] = result[0]['query_id']
    elif change == 'cap101': result[-1]['candidates'] = [{'siret': f'{i:014d}'} for i in range(101)]
    elif change == 'source': result[-1]['trace']['sites_completed'] -= 1
    elif change == 'facts': result[-1]['trace']['address_facts'] -= 1
    else: result[-1]['trace']['score_after_fixed_pool'] = True
    with pytest.raises(ValueError): whole.outputs(result, population(), c, 9, 17, 4)


def test_source_capsule_keeps_all_external_and_owned_files_remain_separate(tmp_path):
    refs, epochs, unused = fixture(tmp_path)
    root = tmp_path / 'own'; root.mkdir()
    own = root / 'own_snapshot.json'; write_json(own, {'not_a_source': True})
    all_refs = [*refs.values(), record(own)]
    all_epochs = hashes.canonical_epochs([*epochs, dict(reference=record(own), fingerprint=list(hashes.fingerprint(own)))])
    groups = {k: [r] for k, r in refs.items()}; groups['context'].append(record(own))
    value = whole.capsule(groups, all_epochs, root)
    assert len(value['references']) == 3 and value['pending_full_hash']
    assert whole.require_sources(value, all_refs, all_epochs, root, exact=True)['exact_full_source_set_checked']
    # The global audit MUST keep the OWN feature bytes too, not just external sources.
    parts = whole.chunks(all_refs, max_files=2)
    assert hashes.references(r for p in parts for r in p) == hashes.references(all_refs)
    assert len(parts) == 2


@pytest.mark.parametrize('change', ['source_missing', 'hash_changed', 'epoch_changed', 'foreign', 'truth', 'empty'])
def test_every_external_source_hash_and_original_epoch_required(tmp_path, change):
    refs, epochs, unused = fixture(tmp_path); root = tmp_path / 'own'; root.mkdir()
    value = whole.capsule({k: [r] for k, r in refs.items()}, epochs, root)
    actual, captured = list(refs.values()), deepcopy(epochs)
    if change == 'source_missing': actual.pop()
    elif change == 'hash_changed': actual[0] = {**actual[0], 'sha256': 'f'*64}
    elif change == 'epoch_changed': captured[0]['fingerprint'][3] += 1
    elif change == 'foreign':
        p = tmp_path / 'foreign.json'; write_json(p, {'foreign': True}); actual.append(record(p))
    elif change == 'truth': value['truth_opened'] = True
    else: value['references'] = []; value['source_epochs'] = []
    with pytest.raises(ValueError): whole.require_sources(value, actual, captured, root, exact=True)


def test_same_epoch_mutation_is_still_rejected_by_actual_full_hash(tmp_path):
    refs, epochs, unused = fixture(tmp_path); root = tmp_path / 'own'; root.mkdir()
    value = whole.capsule({k: [r] for k, r in refs.items()}, epochs, root)
    p = Path(refs['facts']['path']); before = p.stat()
    p.write_bytes(p.read_bytes().replace(b'facts', b'FACTS'))
    os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
    whole.require_sources(value, list(refs.values()), epochs, root, exact=True)  # Still QUARANTINED, not an admission.
    with pytest.raises(ValueError): verify(refs['facts'])


@pytest.mark.parametrize('change', ['missing', 'duplicated', 'wrong_bytes', 'false_full', 'truth', 'foreign_plan'])
def test_partial_audit_never_releases_128_collectors(tmp_path, change):
    refs, unused, unused_p = fixture(tmp_path)
    pieces = whole.chunks(refs.values(), max_files=1)
    p = dict(chunks=pieces, signature='a'*64)
    receipts = [dict(chunk=i, signature=p['signature'], files=len(part),
        full_bytes_hashed=sum(verify(r).stat().st_size for r in part),
        exact_target_digest=hashes.targets_digest(part), every_expected_source_full_hash_verified=True,
        truth_opened=False, recall_measured=False, initial100_emitted=False) for i, part in enumerate(pieces)]
    whole.require_receipts(p, receipts)
    if change == 'missing': receipts.pop()
    elif change == 'duplicated': receipts[-1] = receipts[0]
    elif change == 'wrong_bytes': receipts[0]['full_bytes_hashed'] -= 1
    elif change == 'false_full': receipts[0]['every_expected_source_full_hash_verified'] = False
    elif change == 'truth': receipts[0]['truth_opened'] = True
    else: receipts[0]['signature'] = 'b'*64
    with pytest.raises(ValueError): whole.require_receipts(p, receipts)


def test_oversize_file_is_stop_not_dropped_or_sampled(tmp_path):
    refs, unused, unused_p = fixture(tmp_path)
    with pytest.raises(ValueError, match='single full file'): whole.chunks(refs.values(), max_bytes=1)
