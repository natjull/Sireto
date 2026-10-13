"""Describe sealed pilot density weights, not certify national confidence.

Kish ESS is a weight diagnostic, not independent groups or a confidence bound.
The recipe was selected using this pilot; don't reset it into a pristine test.
"""
from collections import defaultdict
from pathlib import Path
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME/'v110/pilot_uncertainty_20261013_a'
PIPELINE = io.HOME/'v99/initial_alignment_native_20261013_a'
SAMPLING = io.HOME/'v14/representative_20261008_b/sampling.json'


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():
        raise ValueError('Fresh diagnostic namespace')
    generation = io.read(PIPELINE/'generation.json')
    owned = io.read(PIPELINE/'generate_exit.json')
    if generation['status']!='COMPLETE_SEALED_ALL526' or owned['exit_code'] or not owned['own_child_exit_confirmed']:
        raise ValueError('Owned whole sealed pipeline')
    verify(generation['outputs'])
    rows = io.read(PIPELINE/'outcomes.json')['learned_native_ooc']
    if len(rows)!=526:
        raise ValueError('All526 retained')
    sampling = io.read(SAMPLING)
    total = sum(r['calibrated_query_weight'] for r in rows)
    weights = [r['calibrated_query_weight'] for r in rows]
    ess = total**2/sum(w*w for w in weights)
    if abs(ess-sampling['kish_query_weight_ess'])>1e-8 or total!=sampling['source_queries']:
        raise ValueError('Sampling weight authority mismatch')
    groups = defaultdict(list)
    for r in rows:
        groups[r['stratum']].append(r)
    strata = {k:dict(queries=len(rs),exact=sum(r['exact'] for r in rs),
        represented_population_fraction=sum(r['calibrated_query_weight'] for r in rs)/total,
        exact_error_population_fraction=sum(r['calibrated_query_weight'] for r in rs if not r['exact'])/total)
        for k,rs in groups.items()}
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_CONSUMED_PILOT_WEIGHT_DIAGNOSIS_NOT_CERTIFICATION',
        implementation=record(Path(__file__)),generation=record(PIPELINE/'generation.json'),
        owned_exit=record(PIPELINE/'generate_exit.json'),outcomes=record(PIPELINE/'outcomes.json'),
        sampling=record(SAMPLING),queries=len(rows),communes=sampling['communes'],
        reference_queries=sampling['source_queries'],sampling_method=sampling['selection'],
        sum_weights=total,minimum_weight=min(weights),maximum_weight=max(weights),
        kish_query_weight_ess=ess,strata=strata,
        no_independent_identical_distribution_claim=True,no_new_confidence_interval_or_recall=True,
        multiple_recipe_selection_not_corrected=True,no_model_or_prediction_changed=True,
        no_reserved_test_opened=True,observed_unix=time.time()))
    print(ess,strata,flush=True)


if __name__=='__main__':
    main()
