"""Audit frozen Gemma TRAIN input ambiguity, no gradient/test/alias mapping.

Repeated query texts with distinct target parents can yield conflicting
contrastive anchors. Count actual unmasked collisions in original seeded
epoch0 batches; this is not a retrieval metric or proof every label is wrong.
"""
from collections import defaultdict,Counter
from pathlib import Path
import random
import time
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v3_fit as fit
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v104/identity_ambiguity_20261013_a'
LAUNCH=io.HOME/'v6/deadline_recovery_20261008_a/fit/launch.json'


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh train-only audit')
    launch=io.read(LAUNCH);manifest=io.read(verify(launch['data']));cfg=io.read(verify(manifest['config']))
    mining=io.read(verify(launch['mining']));rows=io.read(verify(mining['train']))
    docs=io.read(verify(manifest['artifacts']['documents.json']));parents=io.read(verify(manifest['artifacts']['parent_views.json']))
    summaries={}
    for task in ('identity','site'):
        groups=defaultdict(set);counts=Counter()
        for r in rows:groups[r[task+'_query']].add(r['siren']);counts[r[task+'_query']]+=1
        ambiguous={q:ps for q,ps in groups.items() if len(ps)>1}
        summaries[task]=dict(unique_exact_query_texts=len(groups),ambiguous_exact_query_texts=len(ambiguous),
            rows_with_identical_input_but_distinct_target_sirens=sum(counts[q] for q in ambiguous),
            maximum_distinct_target_sirens_same_exact_input=max(map(len,groups.values())),
            target_multiplicity_histogram=dict(Counter(len(ps) for ps in ambiguous.values())),
            no_text_to_siren_mapping_exported=True)
    plan=fit.plan_rows(rows,docs,parents);order=list(range(len(plan)));random.Random(cfg['seed']).shuffle(order)
    step=0;collisions=0;batches=0;conflicting_batches=0;related_excluded=0;source_pairs=0;anchor_positions=0
    details=[]
    for start in range(0,len(order),cfg['batch_size']):
        mandate.guard(mandate.DEADLINE)
        batch=[(rows[plan[i][0]],plan[i][1],plan[i][2]) for i in order[start:start+cfg['batch_size']]]
        step+=1;batches+=1;hit=set()
        for i,(a,_,_) in enumerate(batch):
            for j,(b,_,_) in enumerate(batch):
                if i==j or a['identity_query']!=b['identity_query'] or a['siren']==b['siren']:continue
                source_pairs+=1;positive=docs[a['positive']]
                blocked=b['siren'] in positive['related_parents'] or b['positive']==a['positive']
                if blocked:related_excluded+=1;continue
                collisions+=1;hit.add(i)
        anchor_positions+=len(hit);conflicting_batches+=bool(hit)
        if hit:details.append(dict(step=step,unmasked_conflicting_anchor_positions=len(hit),
            train_only_no_target_ids_or_query_text_exported=True))
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'epoch0_collisions.json',details)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_TRAIN_ONLY_IDENTITY_INPUT_AMBIGUITY_AUDIT',
        implementation=record(Path(__file__)),frozen_training_implementation=launch['implementation'],
        fit_launch=record(LAUNCH),data=launch['data'],mining=launch['mining'],config=manifest['config'],
        rows=len(rows),queries=summaries,original_epoch0_plan_examples=len(plan),original_batch_size=cfg['batch_size'],
        epoch0_batches=batches,epoch0_batches_with_unmasked_identical_input_distinct_parent_anchors=conflicting_batches,
        epoch0_unmasked_conflicting_anchor_positions=anchor_positions,epoch0_directed_conflicting_anchor_pairs=collisions,
        related_parent_masks_protecting_identical_inputs=related_excluded,
        plan_steps_to_best_checkpoint=1000,plan_steps_to_last_checkpoint=1340,
        exact_text_ambiguity_not_all_tokenizer_equivalences=True,
        no_gradient_or_model_changed=True,no_new_recall_measured=True,no_validation_or_test_opened=True,
        collisions=record(OUTPUT/'epoch0_collisions.json'),observed_unix=time.time()))
    print(summaries,'unmasked anchors',anchor_positions,'batches',conflicting_batches,flush=True)


if __name__=='__main__':main()
