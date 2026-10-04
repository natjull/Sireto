"""Real tiny hashes with SIMULATED admitted processes, no national metric."""
from copy import deepcopy
import json

import pytest

from scripts import audit_retrieval_pair_semantic_parity as p
from src.xgb_matcher.retrieval_v20_workbench import record, write_json


def fixture(tmp_path, monkeypatch):
    refs = []
    calls = []
    for name, module in (('old', p.baseline), ('new', p.current)):
        root = tmp_path / name; root.mkdir()
        c = dict(output_root=str(root), sources='SIMULATED_IDENTICAL')
        write_json(root / 'launch.json', {'config': c})
        rows = [dict(position=i, query_id='Q'+str(i), fold=3-i,
            candidates=[dict(siret='0000000000000'+str(i), siren='000000000', score=1.+i)],
            trace=dict(signature=str(i)*64, sites_completed=9, address_facts=17)) for i in (0, 1)]
        write_json(root / 'outputs.json', dict(rows=rows, truth_opened=False, recall_measured=False,
                                             population=128, full_query_population=False))
        write_json(root / 'manifest.json', dict(status=module.STATUS, launch=record(root / 'launch.json'),
            outputs=record(root / 'outputs.json'), queries=2, population=128, initial100_emitted=True,
            complete_national_source=True, truth_opened=False, old_candidates_opened=False,
            fold0_opened=False, fold1_opened=False, recall_measured=False, full_query_population=False,
            ready_for_fold1=False, slo_validated=False))
        monkeypatch.setattr(module, 'admission', lambda root: calls.append(str(root)))
        refs.append(record(root / 'manifest.json'))
    return refs, calls


def test_only_exact_postseal_parity_with_no_gt_or_generation_feature(tmp_path, monkeypatch):
    refs, calls = fixture(tmp_path, monkeypatch)
    result = p.compare(*refs)
    assert len(calls) == 2 and result['counts'] == [1, 1]
    assert result['old_candidates_opened'] and not result['candidates_used_as_features']
    assert not result['generation_influenced_by_parity'] and not result['truth_opened']
    assert not result['recall_measured'] and not result['full_query_population']


@pytest.mark.parametrize('change', ['partial', 'truth', 'fold1', 'source_changed', 'score', 'candidate', 'query', 'trace', 'population'])
def test_no_partial_or_different_semantics_or_labels_accepted(tmp_path, monkeypatch, change):
    refs, calls = fixture(tmp_path, monkeypatch)
    root = __import__('pathlib').Path(refs[1]['path']).parent
    m = json.loads((root / 'manifest.json').read_text())
    if change == 'partial': m['status'] = 'QUARANTINED'
    elif change == 'truth': m['truth_opened'] = True
    elif change == 'fold1': m['fold1_opened'] = True
    elif change == 'source_changed':
        write_json(root / 'launch.json', {'config': {'output_root': str(root), 'sources': 'DIFFERENT'}})
        m['launch'] = record(root / 'launch.json')
    else:
        rows = json.loads((root / 'outputs.json').read_text())
        if change == 'score': rows['rows'][1]['candidates'][0]['score'] += .00000001
        elif change == 'candidate': rows['rows'][1]['candidates'][0]['siret'] = '00000000000099'
        elif change == 'query': rows['rows'][1]['query_id'] = 'Q0'
        elif change == 'trace': rows['rows'][1]['trace']['sites_completed'] -= 1
        else: rows['population'] = 2
        write_json(root / 'outputs.json', rows); m['outputs'] = record(root / 'outputs.json')
    write_json(root / 'manifest.json', m); refs[1] = record(root / 'manifest.json')
    with pytest.raises(ValueError): p.compare(*refs)


def test_actual_admission_failure_blocks_even_when_lists_match(tmp_path, monkeypatch):
    refs, calls = fixture(tmp_path, monkeypatch)
    def fail(root): raise ValueError('SIMULATED actual source/audit/exit failure')
    monkeypatch.setattr(p.current, 'admission', fail)
    with pytest.raises(ValueError): p.compare(*refs)
