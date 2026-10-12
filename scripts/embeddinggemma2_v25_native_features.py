"""V25 soft query-conditioned evidence fusion, without aliases or fixed prepool."""
from pathlib import Path
import argparse
import numpy as np
from scripts import embeddinggemma2_v23_native_features as source
from scripts.embeddinggemma2_v23_native_features import *
from src.xgb_matcher.retrieval_v20_workbench import record,verify
PREPARED=io.HOME/'v23/hierarchical_native_20261012_a'
OUTPUT=io.HOME/'v25/coherent_query_experts_20261012_a'
BASE_FEATURE_FIELDS=list(source.FEATURE_FIELDS)
FEATURE_FIELDS=['COHERENT_QUERY__'+f for f in BASE_FEATURE_FIELDS]+['UNCERTAIN_QUERY__'+f for f in BASE_FEATURE_FIELDS]
RECIPE=dict(source.RECIPE,feature_fields=FEATURE_FIELDS,
    soft_query_gate='MAX_ALL_OWN_NAME_SIMILARITY_TIMES_OWN_GEOGRAPHIC_QUALITY',
    gate_is_query_only_full_stream_no_labels=True,
    query_experts='TWO_NONNEGATIVE_LINEAR_HEADS_WEIGHTED_R_AND_1_MINUS_R',
    all_sources_retained_no_hard_routing=True,feature_reuse='EXACT_SEALED_V23_ALL526')

def coherence(features):
    if features.shape[1]!=len(source.PROOF_FIELDS):raise ValueError('Original full role features')
    value=float(features[:,source.PROOF_FIELDS.index('name_x_geo')].max()) if len(features) else 0.
    if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Query-only coherence bounded')
    return value

def initialize(out,deadline):
    source.initialize(out,deadline)
    previous=source.authority(PREPARED);gen=io.read(PREPARED/'features_generation.json');ex=io.read(PREPARED/'prepare_exit.json')
    if ex['exit_code']!=0 or not ex['own_child_exit_confirmed'] or gen['queries']!=526:raise ValueError('Complete prepared source')
    launch=io.read(out/'launch.json')
    if previous['panel']!=launch['panel']:raise ValueError('Same526')
    io.write(out/'recipe.json',RECIPE)
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prepared_launch=record(PREPARED/'launch.json'),prepared_generation=record(PREPARED/'features_generation.json'),
        prepared_exit=record(PREPARED/'prepare_exit.json'))
    launch['helpers'].append(record(Path(source.__file__)));io.write(out/'launch.json',launch)

def authority(out):
    launch=source.authority(out)
    for name in ('prepared_launch','prepared_generation','prepared_exit'):verify(launch[name])
    return launch

def prepare(out):
    launch=authority(out);gen=io.read(verify(launch['prepared_generation']))
    if gen['status']!='COMPLETE_SEALED_QUERY_ONLY_ALL526_FULL_PROOFS':raise ValueError('Complete features first')
    io.write(out/'features_generation.json',dict(gen,immutable_shared_source=launch['prepared_generation'],labels_used=False))

def native_search(backend,q,keys,features,coefficients):
    w=np.asarray(coefficients,dtype=float);n=len(BASE_FEATURE_FIELDS)
    if w.shape!=(2*n,) or not np.isfinite(w).all() or (w<0).any():raise ValueError('Nonnegative typed expert heads')
    r=coherence(features);effective=r*w[:n]+(1-r)*w[n:]
    rows,cost=source.native_search(backend,q,keys,features,effective)
    cost.update(query_only_coherence=r,learned_two_soft_experts=True,no_hard_filter=True)
    return rows,cost

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--deadline-unix',type=float);a=p.parse_args()
    if a.action=='initialize':initialize(a.output,a.deadline_unix)
    else:prepare(a.output)

