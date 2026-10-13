"""TRAIN-only information and exact-text false-negative audit, no gradients.

Never export CRM-to-parent aliases or declare homonymous companies equivalent.
Count exact texts in the original seeded epoch0 plan and its actual masks.
This is a data/objective diagnosis, not a counterfactual retrieval result.
"""
from collections import Counter, defaultdict
from pathlib import Path
import random
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v3_fit as fit
from scripts import embeddinggemma2_v56_corporate_site as mandate
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v105/train_information_audit_20261013_a'
LAUNCH = io.HOME / 'v6/deadline_recovery_20261008_a/fit/launch.json'
AUDIT = io.HOME / 'v104/identity_ambiguity_20261013_a/report.json'


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():
        raise ValueError('Fresh audit namespace required')
    launch = io.read(LAUNCH)
    manifest = io.read(verify(launch['data']))
    cfg = io.read(verify(manifest['config']))
    mining = io.read(verify(launch['mining']))
    rows = io.read(verify(mining['train']))
    docs = io.read(verify(manifest['artifacts']['documents.json']))
    parents = io.read(verify(manifest['artifacts']['parent_views.json']))
    original = io.read(AUDIT)
    collisions = io.read(verify(original['collisions']))
    information = {}
    fields = {
        'name': ('crm_name',),
        'name_postcode': ('crm_name', 'crm_postcode'),
        'name_commune_lookup_diagnostic_only': ('crm_name', 'crm_insee'),
        'name_postcode_street': ('crm_name', 'crm_postcode', 'crm_street'),
        'name_full_address': ('crm_name', 'crm_postcode', 'crm_street', 'crm_number', 'crm_suffix'),
    }
    for name, keys in fields.items():
        groups = defaultdict(set)
        counts = Counter()
        for row in rows:
            value = tuple(row['crm'].get(k, '') for k in keys)
            groups[value].add(row['siren'])
            counts[value] += 1
        ambiguous = [q for q, ps in groups.items() if len(ps) > 1]
        information[name] = dict(exact_inputs=len(groups), ambiguous_inputs=len(ambiguous),
            ambiguous_rows=sum(counts[q] for q in ambiguous),
            maximum_distinct_target_parents=max(map(len, groups.values())))

    plan = fit.plan_rows(rows, docs, parents)
    order = list(range(len(plan)))
    random.Random(cfg['seed']).shuffle(order)
    by_step = []
    for start in range(0, len(order), cfg['batch_size']):
        mandate.guard(mandate.DEADLINE)
        batch = [(rows[plan[i][0]], plan[i][1], plan[i][2])
                 for i in order[start:start + cfg['batch_size']]]
        step = start // cfg['batch_size'] + 1
        tasks = {}
        for task in ('site', 'identity'):
            _, texts, masks, _ = fit.training_texts(batch, docs, parents, task, 0)
            affected = 0
            exact_duplicates = 0
            for i in range(len(batch)):
                number = sum(j != i and not masks[i][j] and text == texts[i]
                             for j, text in enumerate(texts))
                affected += bool(number)
                exact_duplicates += number
            tasks[task] = dict(affected_anchor_positions=affected,
                               unmasked_identical_positive_text_negative_pairs=exact_duplicates)
        if any(t['affected_anchor_positions'] for t in tasks.values()):
            by_step.append(dict(step=step, tasks=tasks))

    checkpoints = {}
    for stop in (1000, 1340, len(order) // cfg['batch_size'] + bool(len(order) % cfg['batch_size'])):
        selected = [r for r in collisions if r['step'] <= stop]
        tasks = {}
        for task in ('site', 'identity'):
            counts = Counter()
            for row in by_step:
                if row['step'] <= stop:
                    counts.update(row['tasks'][task])
            tasks[task] = dict(counts)
        checkpoints[str(stop)] = dict(
            identical_query_conflicting_parent_batches=len(selected),
            identical_query_conflicting_parent_anchor_positions=sum(
                r['unmasked_conflicting_anchor_positions'] for r in selected),
            exact_document_text_false_negatives=tasks)
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT / 'report.json', dict(
        status='COMPLETE_TRAIN_ONLY_INFORMATION_AND_OBJECTIVE_AUDIT',
        implementation=record(Path(__file__)), original_audit=record(AUDIT),
        fit_launch=record(LAUNCH), frozen_fit=launch['implementation'],
        training_helper=record(Path(fit.__file__)), data=launch['data'],
        mining=launch['mining'], config=manifest['config'], rows=len(rows),
        query_information=information, epoch0_checkpoint_counts=checkpoints,
        comparison_is_exact_input_partition_not_model_performance=True,
        no_text_or_query_to_parent_mapping_exported=True,
        commune_is_lookup_diagnostic_not_new_feature=True,
        identical_text_does_not_prove_entity_equivalence=True,
        no_tokenizer_equivalence_or_full_corpus_false_negative_claim=True,
        no_gradient_or_model_changed=True, no_validation_or_test_opened=True,
        no_new_recall_measured=True, observed_unix=time.time()))
    print(information, checkpoints, flush=True)


if __name__ == '__main__':
    main()
