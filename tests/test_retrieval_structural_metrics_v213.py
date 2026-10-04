"""Synthetic metric fixtures only, NOT recall of national/CRM data."""
from copy import deepcopy
import inspect

import pytest

from src.xgb_matcher import retrieval_structural_metrics_v213 as work
from test_retrieval_pilot_metrics_v200 import tables


def structural_tables():
    generated, controls, labels, official, acceptable, native = tables()
    for i, row in enumerate(generated):
        candidates = row.pop('candidates')
        for rank, candidate in enumerate(candidates, 1):
            candidate.update(siren=candidate['siret'][:9], retrieval_rank=rank,
                             source_score=1., channel=work.CHANNELS[i%4])
        row.update(results={'primary': candidates}, seconds={'primary': i+1.})
    available = {r['query_id']: i%16 != 0 for i, r in enumerate(controls)}
    return [generated, controls, labels, official, acceptable, native, available]


def test_allhuman_exact_operational_and_three_loss_classes_without_hiding_missing_sites():
    values = structural_tables()
    frozen = deepcopy(values)
    result = work.evaluate(*values)
    assert values == frozen
    assert result['primary']['queries'] == 128
    assert result['primary']['exact_hits'] == 96
    assert result['primary']['operational_hits'] == 128
    assert result['structural_loss_counts'] == dict(HIT=96, AVAILABLE_OFFICIAL_CATALOG_ABSENT=8,
        AVAILABLE_BUT_NOT_IN_NATIVE_LEAF_UNIVERSE=8, NOT_RETRIEVED_AT100=16)
    assert result['loss_counts']['CURRENT_INDEX_CATALOG_ABSENT'] == 16
    assert result['segments']['qualification_v3']['AMBIGUOUS']['queries'] == 128
    assert result['segments']['qualification_v2']['UNRESOLVED']['queries'] == 64
    assert all(result[k] is False for k in ('ready_for_fold1', 'full_oof_population', 'slo_validated',
        'fine_loss_attribution_measured', 'target_io_authorized_by_this_kernel',
        'generation_admission_verified_by_this_kernel'))
    row = result['outcomes'][8]
    assert row['available_official_catalog_present'] and not row['native_index_catalog_present']


def test_channel_assignment_is_not_standalone_channel_recall():
    result = work.evaluate(*structural_tables())
    stats = result['emitted_channel_statistics']
    assert sum(s['candidates_admitted'] for s in stats.values()) == 128
    assert sum(s['exact_hits_attributed_to_emission'] for s in stats.values()) == 96
    assert stats['text']['exact_hits_attributed_to_emission'] == 0
    assert stats['geo_text']['exact_hits_attributed_to_emission'] == 32
    assert all(s['all_human_denominator'] == 128 and s['queries_with_admission'] == 32 for s in stats.values())
    assert 'NOT_STANDALONE_CHANNEL_RECALL' in result['channel_statistics_scope']


def test_descriptive_latency_does_not_claim_slo():
    latency = work.evaluate(*structural_tables())['latency']
    assert latency['p50_seconds'] == 64.5
    assert latency['p95_seconds'] == pytest.approx(121.65)
    assert latency['p99_seconds'] == pytest.approx(126.73)
    assert latency['total_query_seconds'] == 128*129/2
    assert latency['cache_state'] == 'UNCONTROLLED' and not latency['production_slo_validated']


@pytest.mark.parametrize('change', ['missing_available', 'nonbool_available', 'missing_native', 'nonbool_native',
    'native_outside_available', 'duplicate', 'cap101', 'rank', 'parent', 'channel', 'score_nan', 'score_bool',
    'seconds_nan', 'seconds_bool', 'seconds_negative', 'fold0', 'fold1', 'partial_outputs', 'partial_labels'])
def test_no_partial_labels_false_presence_recapping_or_source_repair(change):
    values = structural_tables()
    generated, controls, labels, official, acceptable, native, available = values
    candidate = generated[0]['results']['primary'][0]
    if change == 'missing_available': available.pop('Q0')
    elif change == 'nonbool_available': available['Q0'] = 0
    elif change == 'missing_native': native.pop('Q0')
    elif change == 'nonbool_native': native['Q0'] = 0
    elif change == 'native_outside_available': available['Q1'] = False
    elif change == 'duplicate': generated[0]['results']['primary'] *= 2
    elif change == 'cap101': generated[0]['results']['primary'] *= 101
    elif change == 'rank': candidate['retrieval_rank'] = True
    elif change == 'parent': candidate['siren'] = '999999999'
    elif change == 'channel': candidate['channel'] = 'downstream'
    elif change == 'score_nan': candidate['source_score'] = float('nan')
    elif change == 'score_bool': candidate['source_score'] = True
    elif change == 'seconds_nan': generated[0]['seconds']['primary'] = float('nan')
    elif change == 'seconds_bool': generated[0]['seconds']['primary'] = True
    elif change == 'seconds_negative': generated[0]['seconds']['primary'] = -1
    elif change == 'fold0': controls[0]['fold'] = generated[0]['fold'] = 0
    elif change == 'fold1': controls[0]['fold'] = generated[0]['fold'] = 1
    elif change == 'partial_outputs': generated.pop()
    else: labels.pop()
    with pytest.raises(ValueError):
        work.evaluate(*values)


def test_even_perfect_pilot_has_no_production_or_permission_claim():
    values = structural_tables()
    for row, label in zip(values[0], values[2]):
        row['results']['primary'][0]['siret'] = label['human_ground_truth_siret']
    values[5] = values[6] = {r['query_id']: True for r in values[1]}
    result = work.evaluate(*values)
    assert result['primary']['exact_recall'] == 1.
    assert result['primary']['wilson95_lower'] < .99
    assert not result['ready_for_fold1'] and not result['full_oof_population']
    assert not result['target_io_authorized_by_this_kernel']
    code = inspect.getsource(work)
    assert all(token not in code for token in ('read_table(', 'verify(', 'open(', 'fit(', 'search('))
