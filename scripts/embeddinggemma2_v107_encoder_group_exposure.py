"""Audit consumed pilot group exposure in frozen V1/V6 TRAIN only.

No reserved test or validation membership opened. This checks supervised
positive/explicit negative exposure, not all optimizer/mining internals or
pretraining, and does not make a reused pilot an independent evaluation.
"""
from pathlib import Path
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v107/encoder_group_exposure_20261013_a'
FIT = io.HOME / 'v6/deadline_recovery_20261008_a/fit/launch.json'
SOURCE = io.HOME / 'night_20261006_a'
CACHED = io.HOME / 'cached_20261007_a/launch.json'
PILOT = io.HOME / 'v20/deep_audit_20261011_c/rows.json'
PIPELINE = io.HOME / 'v99/initial_alignment_native_20261013_a'


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():
        raise ValueError('Fresh audit namespace')
    generation = io.read(PIPELINE / 'generation.json')
    owned = io.read(PIPELINE / 'generate_exit.json')
    if generation['status'] != 'COMPLETE_SEALED_ALL526' or owned['exit_code'] or not owned['own_child_exit_confirmed']:
        raise ValueError('Whole sealed owned pipeline before labels')
    verify(generation['outputs'])
    pilot = io.read(PILOT)
    if len(pilot) != 526 or any(r['label']['fold'] not in (2, 3, 4) for r in pilot):
        raise ValueError('Only consumed human folds2/3/4')
    qs = {r['query_id'] for r in pilot}
    ps = {r['label']['siren'] for r in pilot}
    original = io.read(SOURCE / 'manifest.json')
    cached = io.read(CACHED)
    for name, expected in [('train_membership.json', original['split_hashes']['train']),
                           ('train.json', original['training_hashes']['train'])]:
        if io.digest(SOURCE / name) != expected:
            raise ValueError('Original TRAIN source hash drift')
    if cached['source'] != str(SOURCE) or cached['inputs']['train.json'] != original['training_hashes']['train']:
        raise ValueError('Initial adapter provenance drift')
    fit = io.read(FIT)
    verify(fit['initial'])
    if fit['initial']['path'] != str(io.HOME / 'cached_20261007_a/best_adapter.pt'):
        raise ValueError('Different initial adapter')
    mining = io.read(verify(fit['mining']))
    data = io.read(verify(fit['data']))
    docs = io.read(verify(data['artifacts']['documents.json']))
    train = io.read(verify(mining['train']))
    first = io.read(SOURCE / 'train.json')
    membership = io.read(SOURCE / 'train_membership.json')
    sources = {}
    for name, rows, positives, negatives in [
        ('V1', first, {r['siren'] for r in membership},
         {n['siren'] for r in first for n in r['negatives']}),
        ('V6', train, {r['siren'] for r in train},
         {docs[n['key']]['siren'] for r in train for n in r['negatives']}),
    ]:
        sources[name] = dict(query_overlap=len(qs & {r['query_id'] for r in rows}),
            positive_parent_overlap=len(ps & positives), explicit_negative_parent_overlap=len(ps & negatives))
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT / 'report.json', dict(
        status='COMPLETE_FROZEN_ENCODER_TRAIN_SUPERVISED_EXPOSURE_AUDIT',
        implementation=record(Path(__file__)), fit_launch=record(FIT), initial=fit['initial'],
        original_manifest=record(SOURCE / 'manifest.json'), cached_launch=record(CACHED),
        original_train=record(SOURCE / 'train.json'), original_train_membership=record(SOURCE / 'train_membership.json'),
        v6_mining=fit['mining'], v6_train=mining['train'], pilot=record(PILOT),
        pipeline_generation=record(PIPELINE / 'generation.json'), pipeline_exit=record(PIPELINE / 'generate_exit.json'),
        pilot_queries=len(qs), pilot_parent_groups=len(ps), supervised_exposure=sources,
        pilot_reused_for_source_and_recipe_selection=True, not_pristine_or_full_encoder_oof=True,
        no_claim_all_pretraining_or_intermediate_optimizer_states_audited=True,
        no_reserved_test_or_validation_membership_opened=True, no_query_to_parent_mapping_exported=True,
        no_model_or_prediction_changed=True, observed_unix=time.time()))
    print(sources, flush=True)


if __name__ == '__main__':
    main()
