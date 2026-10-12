"""V24: unchanged sealed V23 full proofs; only training objective/mining changes."""
from pathlib import Path
import argparse
from scripts import embeddinggemma2_v23_native_features as source
from scripts.embeddinggemma2_v23_native_features import *
from src.xgb_matcher.retrieval_v20_workbench import record, verify

PREPARED=io.HOME/'v23/hierarchical_native_20261012_a'
OUTPUT=io.HOME/'v24/native_top100_objective_20261012_a'
RECIPE=dict(source.RECIPE, feature_reuse='EXACT_SEALED_V23_ALL526', feature_values_changed=False)

def initialize(out,deadline):
    source.initialize(out,deadline)
    previous=source.authority(PREPARED)
    generation=io.read(PREPARED/'features_generation.json')
    ex=io.read(PREPARED/'prepare_exit.json')
    if ex['exit_code']!=0 or not ex['own_child_exit_confirmed'] or generation['queries']!=526:
        raise ValueError('Owned complete original features only')
    launch=io.read(out/'launch.json')
    if previous['panel']!=launch['panel']:raise ValueError('Same complete query panel')
    io.write(out/'recipe.json',RECIPE)
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prepared_launch=record(PREPARED/'launch.json'),prepared_generation=record(PREPARED/'features_generation.json'),
        prepared_exit=record(PREPARED/'prepare_exit.json'))
    launch['helpers'].append(record(Path(source.__file__)))
    io.write(out/'launch.json',launch)

def authority(out):
    launch=source.authority(out)
    for name in ('prepared_launch','prepared_generation','prepared_exit'):verify(launch[name])
    return launch

def prepare(out):
    launch=authority(out)
    generation=io.read(verify(launch['prepared_generation']))
    if generation['status']!='COMPLETE_SEALED_QUERY_ONLY_ALL526_FULL_PROOFS':
        raise ValueError('Complete query-only corpus required')
    io.write(out/'features_generation.json',dict(generation,
        immutable_shared_source=launch['prepared_generation'],labels_used=False))
    io.write(out/'worker_state.json',dict(status='REUSED_SEALED_IDENTICAL_V23_PROOFS',full_feature_rows=generation['full_feature_rows']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--deadline-unix',type=float);a=p.parse_args()
    if a.action=='initialize':initialize(a.output,a.deadline_unix)
    else:prepare(a.output)

