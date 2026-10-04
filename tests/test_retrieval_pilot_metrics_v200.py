"""Synthetic metric tables in tests only; no production labels or sources read."""
from copy import deepcopy
import math

import pytest

from src.xgb_matcher import retrieval_pilot_metrics_v200 as metrics


def tables():
    controls, outputs, labels, official, acceptable, present = [], [], [], [], [], {}
    for i in range(128):
        qid, fold = 'Q'+str(i), 2+i%3
        siren = str(100000000+i); truth = siren+'12345'; sibling = siren+'99999'
        control = dict(position=i, query_id=qid, fold=fold); controls.append(control)
        outputs.append(dict(control, candidates=[{'siret': sibling if i%4 == 0 else truth}]))
        labels.append(dict(query_id=qid, fold=fold, human_ground_truth_siret=truth, human_ground_truth_siren=siren,
            label_kind='HUMAN', qualification_v2='UNRESOLVED' if i%2 else 'MATCH_EXACT',
            qualification_v3='AMBIGUOUS', data_origin=None, ground_truth_state='F' if i%4 == 0 else 'A'))
        historical = i%12 != 0
        official.append(dict(query_id=qid, fold=fold, gt_siret=truth, gt_siren=siren,
            official_catalog_present=historical, exact_state=None if not historical else 'F' if i%4 == 0 else 'A'))
        acceptable.append(dict(query_id=qid, fold=fold, siret=truth, exact_truth=True,
            administrative_state='F' if i%4 == 0 else 'A', same_site_evidence=True))
        if i%4 == 0:
            acceptable.append(dict(query_id=qid, fold=fold, siret=sibling, exact_truth=False,
                administrative_state='A', same_site_evidence=True))
        present[qid] = i%8 != 0
    return [outputs, controls, labels, official, acceptable, present]


def test_all_human_denominator_status_segments_and_exact_operational_are_distinct():
    result = metrics.evaluate(*tables())
    assert result['primary']['queries'] == 128
    assert result['primary']['exact_hits'] == 96 and result['primary']['exact_recall'] == .75
    assert result['primary']['operational_hits'] == 128 and result['primary']['operational_only_hits'] == 32
    assert result['historical_all_human'] == result['primary']
    assert result['segments']['qualification_v2']['UNRESOLVED']['queries'] == 64
    assert result['segments']['qualification_v3']['AMBIGUOUS']['queries'] == 128
    assert result['segments']['data_origin']['UNKNOWN']['queries'] == 128
    assert sum(r['queries'] for r in result['segments']['fold'].values()) == 128
    assert result['decision'] == 'PIVOT_GENERATION' and not result['ready_for_fold1']
    assert result['loss_counts'] == {'CURRENT_INDEX_CATALOG_ABSENT': 16, 'NOT_RETRIEVED_AT100': 16, 'HIT': 96}
    row = next(r for r in result['outcomes'] if r['position'] == 12)
    assert not row['historical_official_catalog_present'] and row['current_catalog_present']
    assert row['loss'] == 'NOT_RETRIEVED_AT100'
    assert not result['fine_loss_attribution_measured'] and not result['operational_current_catalog_completeness_measured']


def test_perfect_reused_pilot_is_not_full_oof_or_ready():
    values = tables()
    for output, label in zip(values[0], values[2]): output['candidates'] = [{'siret': label['human_ground_truth_siret']}]
    values[5] = {r['query_id']: True for r in values[1]}
    result = metrics.evaluate(*values)
    assert result['primary']['exact_recall'] == 1. and result['primary']['wilson95_lower'] < .99
    assert result['decision'] == 'STOP' and not result['ready_for_fold1'] and not result['full_oof_population']
    assert not result['slo_validated'] and not result['fold0_opened'] and not result['fold1_opened']


@pytest.mark.parametrize('change', ['label_missing', 'label_duplicate', 'foreign_label', 'contaminated_fold', 'missing_output',
    'duplicate_output', 'mixed_position', 'cap101', 'duplicate_candidate', 'numeric_truth', 'wrong_siren', 'side_truth',
    'missing_acceptable', 'missing_exact_target', 'fuzzy_equivalent', 'foreign_siren', 'false_exact', 'duplicate_acceptable',
    'missing_current', 'nonbool_current', 'hit_absent_current', 'partial_controls', 'fold1_controls', 'bool_position'])
def test_no_partial_denominator_label_repair_false_equivalence_or_cap_drift(change):
    output, control, labels, side, acceptable, present = tables()
    if change == 'label_missing': labels.pop()
    elif change == 'label_duplicate': labels.append(deepcopy(labels[0]))
    elif change == 'foreign_label': labels[0]['query_id'] = 'FOREIGN'
    elif change == 'contaminated_fold': labels[0]['fold'] = 0
    elif change == 'missing_output': output.pop()
    elif change == 'duplicate_output': output.append(deepcopy(output[0]))
    elif change == 'mixed_position': output[0]['position'] = 1
    elif change == 'cap101': output[0]['candidates'] = [{'siret': f'{i:014d}'} for i in range(101)]
    elif change == 'duplicate_candidate': output[0]['candidates'] *= 2
    elif change == 'numeric_truth': labels[0]['human_ground_truth_siret'] = int(labels[0]['human_ground_truth_siret'])
    elif change == 'wrong_siren': labels[0]['human_ground_truth_siren'] = '999999999'
    elif change == 'side_truth': side[0]['gt_siret'] = '99999999912345'
    elif change == 'missing_acceptable': acceptable = [r for r in acceptable if r['query_id'] != 'Q0']
    elif change == 'missing_exact_target': acceptable = [r for r in acceptable if not (r['query_id'] == 'Q0' and r['exact_truth'])]
    elif change == 'fuzzy_equivalent': acceptable[1]['same_site_evidence'] = False
    elif change == 'foreign_siren': acceptable[1]['siret'] = '99999999999999'
    elif change == 'false_exact': acceptable[1]['exact_truth'] = True
    elif change == 'duplicate_acceptable': acceptable.append(deepcopy(acceptable[0]))
    elif change == 'missing_current': present.pop('Q0')
    elif change == 'nonbool_current': present['Q0'] = 0
    elif change == 'hit_absent_current': present['Q1'] = False
    elif change == 'partial_controls': control.pop()
    elif change == 'fold1_controls': control[0]['fold'] = 1
    else: control[1]['position'] = True
    with pytest.raises(ValueError): metrics.evaluate(output, control, labels, side, acceptable, present)


def test_no_inplace_change_of_original_truth_or_outputs():
    values = tables(); frozen = deepcopy(values)
    metrics.evaluate(*values)
    assert values == frozen


def test_wilson_two_sided95_is_exact_and_denominator_never_filtered():
    rows = [{'exact_hit': i < 96, 'operational_hit': True, 'active_successor_same_site': False,
             'multiple_active_equivalents': False} for i in range(128)]
    result = metrics.summary(rows)
    z = 1.959963984540054; p = .75; n = 128
    expected = (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    assert abs(result['wilson95_lower']-expected) < 1e-15
    with pytest.raises(ValueError): metrics.summary([])
