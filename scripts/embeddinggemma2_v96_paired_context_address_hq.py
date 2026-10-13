"""Complete post-seal paired source/learner comparisons, never a candidate union.

Same consumed526-query pilot; descriptive paired uncertainty, not pristine
confirmation, national99.2, full encoder OOF, source-policy or serving approval.
"""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v96/paired_context_address_hq_20261013_a'
ROOTS={'v81':io.HOME/'v81/indirect_lexical_query_20261013_a',
    'v88':io.HOME/'v88/typed_context_native_20261013_a',
    'v94':io.HOME/'v94/coherent_address_recovery_20261013_a',
    'v95':io.HOME/'v95/parent_headquarters_context_20261013_a'}
COMPARISONS=[('v81','v88','FROZEN_V81_PLUS_TYPED_WHOLE_SOURCE_CONTEXT_SCORE_NOT_IDENTITY'),
    ('v81','v94','COHERENT_OWN_ADDRESS_CONTENT_PLUS_NEW_TRAIN_NEGATIVES_AND_NESTED_OFFSETS'),
    ('v81','v95','PUBLIC_HEADQUARTERS_PARENT_CONTEXT_PLUS_NEW_TRAIN_NEGATIVES_AND_NESTED_OFFSETS')]



def paired_uncertainty(rows,sampling):
    groups=defaultdict(list)
    for r in rows:groups[r['zone']].append(r)
    rng=np.random.default_rng(20261013);replicas=np.zeros(2000);point=0.;specs={}
    for spec in sampling['strata']:
        name=spec['name'];zones=[z for z,w in sampling['zone_weights'].items() if w['stratum']==name]
        counts=np.asarray([(sum(r['difference'] for r in groups[z]),len(groups[z])) for z in zones],dtype=float)
        n=len(zones);M=spec['population_zones'];rate=counts[:,0].sum()/counts[:,1].sum()
        share=spec['population_queries']/sampling['source_queries'];point+=share*rate
        if n==1 and n<M:return dict(point=point,interval=None,reason='Noncertainty stratum has one PSU')
        fpc=np.sqrt((1-n/M)*n/(n-1)) if n>1 else 0.
        selected=counts[rng.integers(0,n,(2000,n))].sum(axis=1)
        replicas+=share*(rate+fpc*(selected[:,0]/selected[:,1]-rate))
        specs[name]=dict(sampled_communes=n,population_communes=M,paired_rate_difference=float(rate))
    return dict(poststratified_paired_difference=float(point),
        approximate_stratified_commune_bootstrap95=np.clip(np.quantile(replicas,[.025,.975]),-1,1).tolist(),
        strata=specs,descriptive_not_confirmatory=True,selection_of_best_consumed_recipe_not_corrected=True,
        not_future_crm_or_national_generalization_interval=True)


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh complete paired comparison')
    refs={};generations={}
    for name,root in ROOTS.items():
        generation=io.read(root/'generation.json');ex=io.read(root/'generate_exit.json')
        if generation['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('EVERY compared generation must be owned complete first')
        verify(generation['outputs']);generations[name]=generation
        refs[name]=dict(generation=record(root/'generation.json'),owned_exit=record(root/'generate_exit.json'),
            report=record(root/'report.json'),outcomes=record(root/'outcomes.json'))
    # Only after every above generation/exit seal is checked.
    outcomes={k:{r['query_id']:r for r in io.read(verify(refs[k]['outcomes']))['learned_native_ooc']} for k in ROOTS}
    scope=set(outcomes['v81'])
    if len(scope)!=526 or any(set(v)!=scope for v in outcomes.values()):raise ValueError('All same526 no filtered denominator')
    launch=io.read(ROOTS['v81']/'launch.json');source=io.read(verify(launch['source_launch']));sampling=io.read(verify(source['sampling']))
    summaries={}
    for k,rows in outcomes.items():
        weight=sum(r['calibrated_query_weight'] for r in rows.values())
        cap_rows=io.read(verify(generations[k]['outputs']))['learned_native_ooc']
        if set(cap_rows)!=scope or any(len(cs)>100 or len({c['siret'] for c in cs})!=len(cs) for cs in cap_rows.values()):raise ValueError('Absolute unique100 all526')
        measured=io.read(verify(refs[k]['report']))
        summaries[k]=dict(runtime=measured.get('runtime'),absolute_cap100_verified=True,queries=len(rows),siren_groups=len({r['siren_group'] for r in rows.values()}),
            exact=sum(r['exact'] for r in rows.values()),siren=sum(r['siren'] for r in rows.values()),
            operational=sum(r['operational'] for r in rows.values()),
            weighted={field:sum(r['calibrated_query_weight']*r[field] for r in rows.values())/weight for field in ('exact','siren','operational')})
    comparisons={}
    for old,new,hypothesis in COMPARISONS:
        rows=[]
        for qid in sorted(scope):
            a=outcomes[old][qid];b=outcomes[new][qid]
            if any(a[f]!=b[f] for f in ('zone','siren_group','fold','calibrated_query_weight')):raise ValueError('Paired unit authority')
            rows.append(dict(a,difference=int(b['exact'])-int(a['exact'])))
        comparisons[old+'_to_'+new]=dict(hypothesis=hypothesis,
            gains=[r['query_id'] for r in rows if r['difference']==1],losses=[r['query_id'] for r in rows if r['difference']==-1],
            paired_weighted=paired_uncertainty(rows,sampling),
            gained_siren_groups=sorted({r['siren_group'] for r in rows if r['difference']==1}),
            lost_siren_groups=sorted({r['siren_group'] for r in rows if r['difference']==-1}),
            source_recipe_is_diagnostic_only=True,no_candidate_union_or_oracle_lookup=True)
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_POSTSEAL_PAIRED_NIGHT_COMPARISONS',
        implementation=record(Path(__file__)),source_refs=refs,sampling=source['sampling'],
        summaries=summaries,comparisons=comparisons,all526_retained=True,
        no_partial_recall=True,no_positive_injected=True,no_closed_fold_or_test_opened=True,
        population_is_consumed_development_not_pristine_confirmation=True,full_encoder_oof=False,
        production_slo_measured=False,source_policy_promotion_allowed=False,no_99_2_certificate=True,
        multiple_development_experiments_not_confirmatory_inference=True,observed_unix=time.time()))
    print(summaries,flush=True)


if __name__=='__main__':main()

