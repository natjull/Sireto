"""Reconcile completed BODACC diagnostics with sealed V99 misses.

No new source pull, encoding, features, alias, label edit or retrieval result.
The small passage bank was label-selected: its similarities are diagnostic.
"""
from pathlib import Path
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v106/remaining_bodacc_probe_20261013_a'
PIPELINE = io.HOME / 'v99/initial_alignment_native_20261013_a'
PROBE = io.HOME / 'v54/postseal_bodacc_embedding_probe_20261012_a'
STATUS = io.HOME / 'v103/raw_alias_coverage_20261013_a'


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():
        raise ValueError('Fresh postseal audit namespace')
    generation = io.read(PIPELINE / 'generation.json')
    owned = io.read(PIPELINE / 'generate_exit.json')
    if generation['status'] != 'COMPLETE_SEALED_ALL526' or owned['exit_code'] or not owned['own_child_exit_confirmed']:
        raise ValueError('Complete owned sealed pipeline required')
    verify(generation['outputs'])
    outcomes = io.read(PIPELINE / 'outcomes.json')['learned_native_ooc']
    misses = {r['query_id'] for r in outcomes if not r['exact']}
    if len(outcomes) != 526 or len(misses) != 23:
        raise ValueError('Same whole V99 population')
    probe = io.read(PROBE / 'summary.json')
    if probe['status'] != 'COMPLETE_FROZEN_GEMMA_BODACC_PASSAGE_DIAGNOSIS_NOT_RETRIEVAL':
        raise ValueError('Completed frozen diagnostic required')
    source_cases = io.read(verify(probe['cases']))
    status = io.read(STATUS / 'report.json')
    statuses = {r['qid']: r for r in io.read(verify(status['statuses']))}
    selected = []
    for row in source_cases:
        if row['query_id'] not in misses:
            continue
        diffusion = statuses[row['query_id']]
        protected = diffusion['site_diffusion'] != 'O' or diffusion['owner_diffusion'] != 'O'
        own = row['best_own_without_administration'] if not protected else None
        other = row['best_other_diagnostic_passage'] if not protected else None
        selected.append(dict(query_id=row['query_id'], protected_not_backfilled=protected,
            own_nonadministration_passage=bool(own),
            own_nonadministration_best_cosine=own['cosine'] if own else None,
            other_diagnostic_best_cosine=other['cosine'] if other else None,
            margin=own['cosine']-other['cosine'] if own and other else None,
            own_source_fields=sorted({r['field'] for r in own['refs']}) if own else [],
            diagnostic_bank_is_label_selected=True,
            other_bank_may_contain_administration_not_controlled_ablations=True))
    if {r['query_id'] for r in selected} != misses:
        raise ValueError('Every miss retained, no denominator filter')
    public = [r for r in selected if not r['protected_not_backfilled']]
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT / 'cases.json', selected)
    io.write(OUTPUT / 'report.json', dict(
        status='COMPLETE_REUSED_BODACC_PROBE_REMAINING_MISSES_DIAGNOSTIC_ONLY',
        implementation=record(Path(__file__)), pipeline_generation=record(PIPELINE / 'generation.json'),
        pipeline_exit=record(PIPELINE / 'generate_exit.json'), pipeline_outcomes=record(PIPELINE / 'outcomes.json'),
        frozen_probe=record(PROBE / 'summary.json'), statuses=status['statuses'],
        cases=record(OUTPUT / 'cases.json'), all_misses=len(selected), public_misses=len(public),
        protected_not_backfilled=sum(r['protected_not_backfilled'] for r in selected),
        public_with_own_nonadministration_passage=sum(r['own_nonadministration_passage'] for r in public),
        public_with_positive_diagnostic_margin=sum(r['margin'] is not None and r['margin'] > 0 for r in public),
        no_global_recall_rank_or_precision_claim=True,
        no_new_source_encoding_training_or_retrieval=True, no_prediction_or_label_changed=True,
        source_presence_similarity_not_identity_or_exact_site_proof=True, observed_unix=time.time()))
    print(len(public), sum(r['own_nonadministration_passage'] for r in public),
          sum(r['margin'] is not None and r['margin'] > 0 for r in public), flush=True)


if __name__ == '__main__':
    main()
