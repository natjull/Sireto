"""Real tiny-file hashes, simulated parent/resource reports: not a national run."""
from copy import deepcopy
import json

import pytest

import scripts.probe_retrieval_v186_staged_initial as original
import scripts.probe_retrieval_v187_staged_initial as workbench
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_retrieval_postscore_audit_v187 import fixture
from test_seal_retrieval_v175_address_parts import gate, supervision, resource_report


def stages_fixture(tmp_path):
    refs, epochs, packets = fixture(tmp_path)
    output = tmp_path / 'run'
    output.mkdir()
    write_json(output / 'launch.json', {'fixture': 'SIMULATED_CONTROL_NOT_NATIONAL'})
    cfg = dict(partitions=2, sites=9, facts=17)
    def put(name, artifacts, captured, trace):
        dest = output / name
        dest.mkdir()
        write_json(dest / 'preflight.json', gate())
        write_json(dest / 'supervision.json', supervision())
        stored = {}
        for key, value in artifacts.items():
            path = dest / (key+'.json')
            write_json(path, value)
            stored[key] = record(path)
        m = dict(status=('QUARANTINED_V187_' if name.startswith('score') else 'SEALED_V187_')+name.upper(),
                 source_hash_audit_pending=name.startswith('score'), launch=record(output / 'launch.json'),
                 preflight=record(dest / 'preflight.json'), artifacts=stored, source_epochs=captured,
                 resources=resource_report(), trace=trace, old_candidates_opened=False,
                 initial100_emitted=False, **workbench.FLAGS)
        write_json(dest / 'manifest.json', m)
    for i, packet in enumerate(packets):
        span = packet['span']
        part = dict(span=span, signature=packet['signature'], candidates=[],
                    partition_only_not_initial_output=True, same_scores_never_rescored=True,
                    stats=dict(sites_completed=span['stop']-span['start'], address_facts=span['facts'],
                               every_site_scored_once=True, fixed_reservoir=None,
                               score_after_fixed_pool=False, initial_cap=100))
        put(f'score{i:03d}', {'partition': part, 'audit_packet': packet}, packet['source_epochs'],
            {'signature': packet['signature']})
    plan, score_refs, packet_refs = workbench.audit_plan(output, cfg)
    for group in workbench.post_audit.GROUPS:
        receipt = workbench.post_audit.audit_group(plan, group)
        receipt.update(score_stages=score_refs, audit_packets=packet_refs)
        put('audit-'+group, {'receipt': receipt}, plan['source_epochs'], {'group': group})
    return output, cfg


def replace_artifact(output, name, key, change):
    dest = output / name
    path = dest / (key+'.json')
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)
    m = json.loads((dest / 'manifest.json').read_text())
    m['artifacts'][key] = record(path)
    write_json(dest / 'manifest.json', m)


def test_all_pending_scores_then_every_actual_control_and_entire_byte_receipt(tmp_path):
    output, cfg = stages_fixture(tmp_path)
    plan = workbench.audited_plan(output, cfg)
    assert plan['sites'] == 9 and plan['facts'] == 17
    score, ref = workbench.stage(output, 'score000')
    assert score['status'] == 'QUARANTINED_V187_SCORE000'
    assert not score['initial100_emitted'] and not score['truth_opened']
    assert all(len(v) == 1 for v in plan['groups'].values())


@pytest.mark.parametrize('change', ['missing_receipt', 'parent_nonzero', 'parent_unconfirmed', 'rss',
                                   'truth', 'stop', 'wrong_group', 'wrong_scores', 'wrong_packets',
                                   'wrong_target', 'source_changed', 'wrong_partition_span',
                                   'different_signature', 'missing_site', 'pool_rescored', 'emitted_score'])
def test_nothing_released_if_any_score_or_source_audit_is_incomplete(tmp_path, change):
    output, cfg = stages_fixture(tmp_path)
    audit = output / 'audit-facts'
    if change == 'missing_receipt':
        (audit / 'receipt.json').unlink()  # Own test fixture only.
    elif change in ('parent_nonzero', 'parent_unconfirmed'):
        s = supervision()
        s.update(exit_code=1 if change == 'parent_nonzero' else 0,
                 own_child_exit_confirmed=change != 'parent_unconfirmed')
        write_json(audit / 'supervision.json', s)
    elif change in ('rss', 'truth', 'emitted_score'):
        path = (output / 'score000' if change == 'emitted_score' else audit) / 'manifest.json'
        m = json.loads(path.read_text())
        if change == 'rss':
            m['resources']['peak_rss_bytes'] = 2**30+1
        elif change == 'truth':
            m['truth_opened'] = True
        else:
            m['initial100_emitted'] = True
        write_json(path, m)
    elif change == 'stop':
        write_json(audit / 'supervisor_stop.json', {'status': 'STOP'})
    elif change in ('wrong_group', 'wrong_scores', 'wrong_packets', 'wrong_target'):
        def mutate(r):
            if change == 'wrong_group':
                r['group'] = 'postings'
            elif change == 'wrong_scores':
                r['score_stages'][0]['sha256'] = 'b'*64
            elif change == 'wrong_packets':
                r['audit_packets'][0]['sha256'] = 'b'*64
            else:
                r['exact_target_digest'] = 'b'*64
        replace_artifact(output, 'audit-facts', 'receipt', mutate)
    elif change == 'source_changed':
        write_json(tmp_path / 'facts.json', {'changed': True})
    else:
        def mutate(p):
            if change == 'wrong_partition_span':
                p['span']['stop'] += 1
            elif change == 'different_signature':
                p['signature'] = 'b'*64
            elif change == 'missing_site':
                p['stats']['sites_completed'] -= 1
            else:
                p['stats']['score_after_fixed_pool'] = True
        replace_artifact(output, 'score000', 'partition', mutate)
    with pytest.raises((ValueError, FileNotFoundError)):
        workbench.audited_plan(output, cfg)


def test_unchanged_initial_score_kernel_and_frozen_budget():
    assert workbench.PartitionFacts is original.PartitionFacts
    assert workbench.partition_search is original.partition_search
    assert workbench.merge_partitions is original.merge_partitions
    assert workbench.activity.__code__.co_code == original.activity.__code__.co_code
    assert workbench.inputs.__code__.co_code == original.inputs.__code__.co_code
    assert workbench.LIMITS == original.LIMITS
    assert workbench.FLAGS == original.FLAGS
    a, b = deepcopy(workbench.config()), deepcopy(original.config())
    a.pop('output_root'); b.pop('output_root')
    assert a == b
