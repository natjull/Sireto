"""Execution-only V139 continuation, admitted only after its real disk yield.

Reserve80->40GiB; same RSS10GiB/free25/pressure012, source/model/score,
query population and deadline. Historical modules/outputs are untouched.
"""
import argparse
import json
from pathlib import Path
import shutil
import signal
import time

from scripts import embeddinggemma2_v138_structural_native_ablation as bound
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = Path(bound.__file__)
OLD = bound.io.HOME/'v139/structural_native_resource_20261013_night'
NEW = bound.io.HOME/'v140/structural_native_disk_20261013_night'


def bind_original():
    old = bound.io.read(OLD/'launch.json')
    verify(old['implementation'])
    if record(SOURCE) not in old['helpers']:
        raise ValueError('Original frozen scientific module binding required')
    text = SOURCE.read_text()
    changes = {
        "ROOT=io.HOME/'v138/structural_native_20261013_night'":
            "ROOT=io.HOME/'v140/structural_native_disk_20261013_night'",
        'rss>7*1024**3': 'rss>10*1024**3',
        'max_rss_gib=7': 'max_rss_gib=10',
        'reserve_catnat_gib=80': 'reserve_catnat_gib=40',
        '.free<80*1024**3': '.free<40*1024**3',
        'CATNAT reserve below80GiB': 'CATNAT reserve below40GiB',
        "'scripts.embeddinggemma2_v138_structural_native_ablation'":
            "'scripts.embeddinggemma2_v140_disk_continuation'",
    }
    for (before, after), expected in zip(changes.items(), (1, 2, 1, 1, 1, 1, 1)):
        if text.count(before) != expected:
            raise ValueError('Exact execution-only transformation count')
        text = text.replace(before, after)
    bound.__dict__['__file__'] = str(Path(__file__).resolve())
    exec(compile(text, str(SOURCE), 'exec'), bound.__dict__)
    # Historical helper files remain immutable, but their process-local calls
    # must respect the same declared40GiB execution policy, not stale80GiB.
    bound.engine.guard = bound.guard
    previous = bound.io.read(OLD/'recipe.json')
    current = json.loads(json.dumps(bound.RECIPE))
    previous_resources = previous.pop('resources')
    current_resources = current.pop('resources')
    if previous != current:
        raise ValueError('No scientific recipe change permitted')
    previous_resources['reserve_catnat_gib'] = 40
    if previous_resources != current_resources:
        raise ValueError('Only disk reserve80->40 differs from V139')


def migrate():
    bind_original()
    bound.guard()
    observed = bound.io.read(OLD/'owned_tool_exit.json')
    if (observed['exit_code'] != 1 or not observed['own_tool_exit_confirmed']
            or not observed['voluntary_stop'] or not observed['real_disk80_block_confirmed']):
        raise ValueError('Old real disk block and observed voluntary parent exit required')
    if NEW.exists():
        raise ValueError('Fresh execution-only namespace')
    old = bound.authority(OLD)
    panel = bound.io.read(verify(old['panel']))
    queries = {q['query_id']: q for q in panel}
    cal = bound.io.read(verify(old['baseline_calibration']))
    assignment = bound.io.read(verify(cal['assignment']))
    NEW.mkdir(parents=True)
    (NEW/'spill').mkdir()
    (NEW/'calibration').mkdir()
    bound.io.write(NEW/'recipe.json', bound.RECIPE)
    launch = dict(old, implementation=record(Path(__file__)),
                  recipe=record(NEW/'recipe.json'), created_unix=time.time())
    launch['helpers'] = [*old['helpers'], old['implementation']]
    bound.io.write(NEW/'launch.json', launch)
    for name in ('features_generation.json', 'calibration/generation.json'):
        shutil.copy2(OLD/name, NEW/name)
    copied, completed = {}, []
    for variant in bound.VARIANTS:
        root = NEW/variant
        (root/'native').mkdir(parents=True)
        count = 0
        for source in sorted((OLD/variant/'native').glob('*.json')):
            sealed = bound.engine.sealed(source)
            if not sealed:
                continue
            part = bound.io.read(sealed)
            qid = part['query']['query_id']
            if part['query'] != queries[qid] or part['model'] != cal['models'][str(assignment[qid])]:
                raise ValueError('Identical query/model binding before shard migration')
            if len(part['candidates']) > 100 or part['labels_read'] is not False:
                raise ValueError('Blind initial100 only')
            destination = root/'native'/source.name
            shutil.copy2(source, destination)
            if record(destination)['sha256'] != record(source)['sha256']:
                raise ValueError('Byte-identical native shard')
            bound.engine.seal(destination)
            count += 1
        copied[variant] = count
        ex_path, gen_path = OLD/variant/'generate_exit.json', OLD/variant/'generation.json'
        if ex_path.exists() and gen_path.exists():
            ex, gen = bound.io.read(ex_path), bound.io.read(gen_path)
            if ex['exit_code'] == 0 and ex['own_child_exit_confirmed']:
                if gen['status'] != 'COMPLETE_SEALED_ALL526' or count != 526:
                    raise ValueError('Whole original generation before complete reuse')
                for field, name in (('outputs', 'outputs.json'), ('timings', 'native_timings.json')):
                    source = verify(gen[field])
                    shutil.copy2(source, root/name)
                    if record(source)['sha256'] != record(root/name)['sha256']:
                        raise ValueError('Byte-identical complete output/timings')
                    gen[field] = record(root/name)
                bound.io.write(root/'generation.json', gen)
                bound.io.write(root/'generate_exit.json', ex)
                completed.append(variant)
    bound.io.write(NEW/'migration.json', dict(
        status='EXACT_SEALED_OUTPUT_DISK_RESERVE_ONLY_CONTINUATION',
        implementation=record(Path(__file__)), original_code=record(SOURCE),
        original_launch=record(OLD/'launch.json'), original_recipe=record(OLD/'recipe.json'),
        old_actual_exit=record(OLD/'owned_tool_exit.json'), copied_native_shards=copied,
        complete_generations_reused=completed, only_execution_change='DISK_RESERVE80_TO40GiB',
        rss10_free25_pressure012_and_deadline_unchanged=True,
        all_source_model_score_and_quantization_bindings_identical=True, no_labels_read=True))
    print('MIGRATED_EXACT_OUTPUTS', copied, completed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['migrate', 'generate', 'run', 'evaluate'])
    parser.add_argument('--output', type=Path, default=NEW)
    parser.add_argument('--variant', choices=bound.VARIANTS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, bound.engine.interrupted)
    signal.signal(signal.SIGINT, bound.engine.interrupted)
    try:
        if args.action == 'migrate':
            migrate()
        else:
            bind_original()
            if args.action == 'generate':
                bound.generate(args.output, args.variant)
            else:
                getattr(bound, args.action)(args.output)
    except bound.engine.ResourceYield as error:
        print(str(error), flush=True)
        raise SystemExit(75)
    except Exception as error:
        if args.output.exists():
            bound.io.write(args.output/(args.action+'_error.json'),
                           dict(error=repr(error), updated_unix=time.time()))
        raise
