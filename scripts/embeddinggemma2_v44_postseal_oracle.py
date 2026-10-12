"""Diagnostic oracle over completed outputs, NOT a retrieval or product score."""
from pathlib import Path
import numpy as np
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record,verify

ROOTS=['v22/learned_native_20261012_b','v23/hierarchical_native_20261012_a',
    'v24/native_top100_objective_20261012_a','v25/coherent_query_experts_20261012_a',
    'v26/official_site_priors_20261012_a','v27/nonlinear_native_20261012_a',
    'v28/official_sector_native_20261012_a','v29/source_robust_native_20261012_a',
    'v30/legal_role_context_20261012_a','v31/local_lifecycle_views_20261012_a',
    'v32/global_public_parent_names_20261012_a','v35/coherence_sector_recovery_20261012_a']
OUT=io.HOME/'v44/completed_output_oracle_20261012_a'

def main():
    if OUT.exists():raise ValueError('Fresh immutable audit namespace')
    variants={};refs=[];truth=None
    for name in ROOTS:
        root=io.HOME/name;gen=io.read(root/'generation.json');ex=io.read(root/'generate_exit.json')
        if gen['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code']!=0 or not ex['own_child_exit_confirmed']:
            raise ValueError('Only complete owned outputs')
        verify(gen['outputs']);refs.extend([record(root/'generation.json'),record(root/'generate_exit.json'),record(root/'outcomes.json')])
        data=io.read(root/'outcomes.json')
        for key,rows in data.items():
            mapped={r['query_id']:r for r in rows}
            if len(mapped)!=526:raise ValueError('All526')
            signature={k:(r['calibrated_query_weight'],r['siren_group'],r['fold']) for k,r in mapped.items()}
            if any(r['fold'] not in (2,3,4) for r in rows):raise ValueError('Protected folds')
            if truth is None:truth=signature
            elif signature!=truth:raise ValueError('Identical human denominator and weights')
            variants[name+'/'+key]=mapped
    union={qid:any(data[qid]['exact'] for data in variants.values()) for qid in truth}
    weights={qid:truth[qid][0] for qid in truth}
    result=dict(status='POSTSEAL_FINITE_OUTPUT_ORACLE_NOT_PRODUCT_RECALL',queries=526,
        diagnostic_exact_hits=sum(union.values()),diagnostic_raw_fraction=sum(union.values())/526,
        diagnostic_weighted_fraction=sum(weights[q]*union[q] for q in union)/sum(weights.values()),
        absent_from_all_completed_outputs=[q for q in union if not union[q]],
        no_candidate_generation=True,no_feature_created=True,no_label_correction=True,
        not_a_full_proof_stream_ceiling=True,not_retrieval100=True,not_holdout=True,
        inputs=refs,implementation=record(Path(__file__)))
    OUT.mkdir(parents=True);io.write(OUT/'summary.json',result)
    print({k:v for k,v in result.items() if k!='inputs'},flush=True)

if __name__=='__main__':main()
