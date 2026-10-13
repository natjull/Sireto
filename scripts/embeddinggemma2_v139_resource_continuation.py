"""Transparent V138 continuation: ONLY RSS admission7→10GiB, same science.

The original code, roots and actual exits remain preserved. Resume copies
only exact query/model-bound sealed shards; no predictions or labels opened.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import signal
import time

from scripts import embeddinggemma2_v138_structural_native_ablation as bound
from src.xgb_matcher.retrieval_v20_workbench import record,verify

SOURCE=Path(bound.__file__)
OLD=bound.ROOT
NEW=bound.io.HOME/'v139/structural_native_resource_20261013_night'


def bind_original():
    launch=bound.io.read(OLD/'launch.json')
    if verify(launch['implementation'])!=SOURCE:raise ValueError('Exact frozen V138 source code required')
    text=SOURCE.read_text()
    changes={
        "ROOT=io.HOME/'v138/structural_native_20261013_night'":
            "ROOT=io.HOME/'v139/structural_native_resource_20261013_night'",
        'rss>7*1024**3':'rss>10*1024**3',
        'max_rss_gib=7':'max_rss_gib=10',
        "'scripts.embeddinggemma2_v138_structural_native_ablation'":
            "'scripts.embeddinggemma2_v139_resource_continuation'",
    }
    expected=[1,2,1,1]
    for (before,after),count in zip(changes.items(),expected):
        if text.count(before)!=count:raise ValueError('Bound execution-only transformation drift')
        text=text.replace(before,after)
    # The namespace remains the registered original module, so its evaluator
    # sees the same authority/recipe interfaces. No historical file changes.
    bound.__dict__['__file__']=str(Path(__file__).resolve())
    exec(compile(text,str(SOURCE),'exec'),bound.__dict__)
    original=bound.io.read(OLD/'recipe.json');new=json.loads(json.dumps(bound.RECIPE))
    original.pop('resources');new.pop('resources')
    if original!=new:raise ValueError('Scientific recipe unchanged, resource policy only')


def migrate():
    bind_original();bound.guard()
    ex=bound.io.read(OLD/'owned_tool_exit.json')
    if ex['exit_code']!=1 or not ex['own_tool_exit_confirmed'] or not ex['voluntary_stop']:
        raise ValueError('Old parent real observed stop before migration')
    if NEW.exists():raise ValueError('Fresh execution continuation namespace')
    old=bound.io.read(OLD/'launch.json');panel=bound.io.read(verify(old['panel']))
    queries={q['query_id']:q for q in panel};cal=bound.io.read(verify(old['baseline_calibration']))
    assignment=bound.io.read(verify(cal['assignment']))
    NEW.mkdir(parents=True);(NEW/'spill').mkdir();(NEW/'calibration').mkdir()
    bound.io.write(NEW/'recipe.json',bound.RECIPE)
    launch=dict(old,implementation=record(Path(__file__)),recipe=record(NEW/'recipe.json'),created_unix=time.time())
    launch['helpers']=[*old['helpers'],record(SOURCE)]
    bound.io.write(NEW/'launch.json',launch)
    bound.io.write(NEW/'features_generation.json',bound.io.read(OLD/'features_generation.json'))
    bound.io.write(NEW/'calibration/generation.json',bound.io.read(OLD/'calibration/generation.json'))
    copied={}
    for variant in bound.VARIANTS:
        root=NEW/variant;(root/'native').mkdir(parents=True);count=0
        for source in sorted((OLD/variant/'native').glob('*.json')):
            sealed=bound.engine.sealed(source)
            if not sealed:continue
            part=bound.io.read(sealed);qid=part['query']['query_id']
            if part['query']!=queries[qid] or part['model']!=cal['models'][str(assignment[qid])]:
                raise ValueError('Exact query/model/source recipe before shard reuse')
            if len(part['candidates'])>100 or part['labels_read'] is not False:raise ValueError('Original blind cap100')
            destination=root/'native'/source.name;shutil.copy2(source,destination)
            if record(destination)['sha256']!=record(source)['sha256']:raise ValueError('Byte-identical sealed shard')
            bound.engine.seal(destination);count+=1
        copied[variant]=count
    bound.io.write(NEW/'migration.json',dict(status='EXACT_SEALED_SHARD_RESOURCE_ONLY_CONTINUATION',
        implementation=record(Path(__file__)),original_code=record(SOURCE),original_launch=record(OLD/'launch.json'),
        original_recipe=record(OLD/'recipe.json'),old_actual_exit=record(OLD/'owned_tool_exit.json'),
        copied_native_shards=copied,only_execution_change='RSS_ADMISSION_AND_SUPERVISION7_TO10GiB',
        all_source_model_score_and_quantization_bindings_identical=True,
        memory_free25_percent_and_pressure012_and_deadline_unchanged=True,no_labels_read=True))
    print('MIGRATED_EXACT_NATIVE_SHARDS',copied,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['migrate','generate','run','evaluate'])
    p.add_argument('--output',type=Path,default=NEW);p.add_argument('--variant',choices=bound.VARIANTS);a=p.parse_args()
    signal.signal(signal.SIGTERM,bound.engine.interrupted);signal.signal(signal.SIGINT,bound.engine.interrupted)
    try:
        if a.action=='migrate':migrate()
        else:
            bind_original()
            if a.action=='generate':bound.generate(a.output,a.variant)
            else:getattr(bound,a.action)(a.output)
    except bound.engine.ResourceYield as error:print(str(error),flush=True);raise SystemExit(75)
    except Exception as error:
        if a.output.exists():bound.io.write(a.output/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
