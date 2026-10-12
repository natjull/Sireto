"""Post-seal fixed-score uncertainty sensitivity, not fresh validation or refit."""
import argparse
from pathlib import Path
import numpy as np
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record,verify

def main(source):
    gen=io.read(source/'generation.json');ex=io.read(source/'generate_exit.json')
    if gen['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code']!=0 or not ex['own_child_exit_confirmed']:
        raise ValueError('Only complete owned generation before labels')
    verify(gen['outputs']);data=io.read(source/'outcomes.json');report=io.read(source/'report.json')
    rows=sorted(data['learned_native_ooc'],key=lambda r:r['query_id']);old={r['query_id']:r for r in data['v18_reference']}
    if len(rows)!=526 or len(old)!=526 or any(r['fold'] not in (2,3,4) for r in rows):raise ValueError('Safe all526')
    w=np.array([r['calibrated_query_weight'] for r in rows]);y=np.array([r['exact'] for r in rows],dtype=float)
    before=np.array([old[r['query_id']]['exact'] for r in rows],dtype=float)
    if abs(float(w@y/w.sum())-report['variants']['learned_native_ooc']['weighted']['exact']['poststratified_hajek_estimate'])>1e-12:
        raise ValueError('Actual weighted point estimate must reproduce sealed report')
    zones,zi=np.unique([r['zone'] for r in rows],return_inverse=True)
    sirens,si=np.unique([r['siren_group'] for r in rows],return_inverse=True)
    rng=np.random.default_rng(20261012);reps=4000
    zw=rng.exponential(size=(reps,len(zones)))[:,zi];sw=rng.exponential(size=(reps,len(sirens)))[:,si]
    strata={s:np.array([r['stratum']==s for r in rows]) for s in sorted({r['stratum'] for r in rows})}
    shares=report['variants']['learned_native_ooc']['weighted']['exact']['strata']
    results={}
    for name,cluster in [('commune_only',zw),('siren_only',sw),('crossed_commune_and_siren',zw*sw)]:
        fresh=np.zeros(reps);prior=np.zeros(reps)
        for s,mask in strata.items():
            weights=cluster[:,mask]*w[mask];den=weights.sum(axis=1);share=shares[s]['population_share']
            fresh+=share*(weights@y[mask])/den;prior+=share*(weights@before[mask])/den
        results[name]=dict(approximate_fixed_score_sensitivity95=np.quantile(fresh,[.025,.975]).tolist(),
            paired_gain_vs_v18_sensitivity95=np.quantile(fresh-prior,[.025,.975]).tolist())
    out=io.HOME/'v46'/('cluster_sensitivity_'+source.parent.name+'_20261012_a')
    if out.exists():raise ValueError('Immutable diagnostic namespace')
    out.mkdir(parents=True)
    result=dict(status='POSTSEAL_CLUSTER_SENSITIVITY_NOT_CERTIFICATION',queries=526,commune_groups=len(zones),siren_groups=len(sirens),
        weighted_point=float(w@y/w.sum()),replicates=reps,seed=20261012,
        method='INDEPENDENT_EXPONENTIAL_CLUSTER_WEIGHTS_PRODUCT_WITH_FIXED_GEOGRAPHIC_STRATUM_SHARES',
        method_primary_reference='https://arxiv.org/abs/1106.2125',results=results,
        not_exact_sampling_design_coverage=True,model_selection_uncertainty_not_accounted=True,
        already_consumed_development=True,not_holdout=True,no_refit=True,no_features_changed=True,
        source_generation=record(source/'generation.json'),source_outcomes=record(source/'outcomes.json'),implementation=record(Path(__file__)))
    io.write(out/'summary.json',result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();main(a.source)
