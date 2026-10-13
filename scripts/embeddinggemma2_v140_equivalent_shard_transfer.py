"""Input-proof-authorized byte-identical native shards, no fake new exits.

Only five isolated recipes are eligible. Their original generation/exits and
timings remain provenance; destinations aggregate shards with actual new exits.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v143_empty_reference_continuation as owner_target
from scripts import embeddinggemma2_v134_supplier_native_ablation as leaf_target
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = io.HOME/'v140/structural_native_disk_20261013_night'
PROOF = io.HOME/'v140/isolated_equivalence_20261013_night'


def main(kind):
    root = owner_target.ROOT if kind == 'owner' else leaf_target.ROOT
    variants = ('rna_only', 'banatic_only') if kind == 'owner' else (
        'supplier_own_site_only', 'supplier_temporal_only', 'continuity_address_only')
    module = owner_target if kind == 'owner' else leaf_target
    launch = module.authority(root)
    proof = io.read(PROOF/'report.json')
    px = io.read(PROOF/'owned_tool_exit.json')
    if proof['status'] != 'STATIC_INPUT_EQUIVALENCE_FIVE_ISOLATED_RECIPES_NOT_CANDIDATE_MATCHING' or px['exit_code'] or not px['own_tool_exit_confirmed']:
        raise ValueError('Completed owned static/input proof before copying')
    old = io.read(verify(proof['source_launch']))
    verify(proof['target_launch'])
    verify(proof['source_owned_exit'])
    verify(proof['scoring_ast'])
    for key in ('panel', 'baseline_features', 'baseline_calibration', 'baseline_generation', 'index_meta', 'index_manifest'):
        if old[key] != launch[key]:
            raise ValueError('Exact initialized destination scientific binding: '+key)
    if kind == 'owner':
        if record(root/'launch.json') != proof['target_launch']:
            raise ValueError('Exact owner destination already proved')
    else:
        for name, field in (('supplier', 'supplier_generation'), ('address', 'address_generation')):
            if old['structural_sources'][name]['generation'] != launch[field]:
                raise ValueError('Same prepared isolated context inputs at destination')
    cal = io.read(root/'calibration/generation.json')
    if cal != io.read(SOURCE/'calibration/generation.json'):
        raise ValueError('Identical full model/assignment authority')
    assignment = io.read(verify(cal['assignment']))
    panel = {q['query_id']: q for q in io.read(verify(launch['panel']))}
    if len(panel) != 526:
        raise ValueError('Whole526 retained')
    receipts = {}
    for variant in variants:
        source = SOURCE/variant
        gen, ex = io.read(source/'generation.json'), io.read(source/'generate_exit.json')
        if gen['status'] != 'COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:
            raise ValueError('Original complete real observed generation')
        dest = root/variant/'native'
        dest.mkdir(parents=True, exist_ok=True)
        copied, seen = [], set()
        for path in sorted((source/'native').glob('*.json')):
            sealed = engine.sealed(path)
            if not sealed:
                continue
            part = io.read(sealed)
            qid = part['query']['query_id']
            if (qid in seen or part['query'] != panel[qid] or part['model'] != cal['models'][str(assignment[qid])]
                    or path.stem != hashlib.sha256(qid.encode()).hexdigest() or part['labels_read'] is not False
                    or len(part['candidates']) > 100):
                raise ValueError('Exact unique blind query/model initial100 shard')
            target = dest/path.name
            if target.exists():
                if record(target)['sha256'] != record(path)['sha256']:
                    raise ValueError('Never overwrite a different destination artifact')
            else:
                shutil.copy2(path, target)
            if record(target)['sha256'] != record(path)['sha256']:
                raise ValueError('Byte-identical shard required')
            engine.seal(target)
            copied.append(dict(query_id=qid, source=record(path), destination=record(target)))
            seen.add(qid)
        if seen != set(panel):
            raise ValueError('All526 shards copied, not selected cases')
        receipts[variant] = dict(original_generation=record(source/'generation.json'),
            original_actual_exit=record(source/'generate_exit.json'), shards=copied)
    io.write(root/('equivalent_native_reuse_'+kind+'.json'), dict(
        status='BYTE_IDENTICAL_ISOLATED_NATIVE_SHARDS_INPUT_PROOF_NOT_OUTCOME_SELECTION',
        implementation=record(Path(__file__)), proof=record(PROOF/'report.json'),
        proof_owned_exit=record(PROOF/'owned_tool_exit.json'), original_parent_exit=proof['source_owned_exit'],
        initialized_destination_launch=record(root/'launch.json'), variants=receipts,
        generations_and_exits_not_copied=True, destination_must_aggregate_and_observe_actual_exit=True,
        no_outputs_or_outcomes_compared=True, all526_retained=True, no_composite_reuse=True,
        historical_timings_reused_not_new_full_cold_warm_slo=True, runtime_contains_equivalent_shard_reuse=True,
        created_unix=time.time()))
    print('EXACT_ISOLATED_SHARDS_READY', kind, {v: len(r['shards']) for v, r in receipts.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['owner', 'leaf'])
    main(parser.parse_args().kind)
