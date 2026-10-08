"""Outcome-blind stratified commune pilot of the frozen Gemma2 pipeline.

Delegates preparation/encoding to immutable V14 workers. Reuses all full-query
lexical/geo outputs and owner lookup. Sampling changes scope, not corpus/score.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v10_representative as population
from src.xgb_matcher.retrieval_v20_workbench import record,verify

SOURCE=io.HOME/'v14/full_validation_20261008_a'
# Dedicated NEW namespace within V14's permitted worker root; not the old run.
OUTPUT=io.HOME/'v14/representative_20261008_b'
SEED='gemma2-probability-commune-pilot-20261008-v1'
STRATA=(('SMALL',0,1000,None),('MEDIUM',1000,10000,100),
    ('DENSE',10000,50000,40),('LARGE',50000,200000,8),('VERY_LARGE',200000,float('inf'),2))


def sample(census,panel):
    """Uniform hash permutation per cost stratum; NEVER reject a costly draw."""
    if len(panel)!=2212 or len({q['query_id'] for q in panel})!=2212:
        raise ValueError('Entire source query census required')
    zones=[]; strata=[]; weights={}
    for name,lo,hi,take in STRATA:
        candidates=[z for z in census['zones'] if lo<=z['indexed_sites']<hi]
        ordered=sorted(candidates,key=lambda z:hashlib.sha256((SEED+'|'+name+'|'+z['zone']).encode()).digest())
        chosen=ordered if take is None else ordered[:take]
        if len(chosen)<2 and len(chosen)!=len(candidates): raise ValueError('Noncertainty stratum requires two PSUs')
        total_queries=sum(z['queries'] for z in candidates); sampled_queries=sum(z['queries'] for z in chosen)
        probability=len(chosen)/len(candidates)
        for z in chosen:
            zones.append({**z,'stratum':name,'zone_inclusion_probability':probability})
            weights[z['zone']]=dict(stratum=name,zone_inclusion_probability=probability,
                inverse_probability_weight=1/probability,calibrated_query_weight=total_queries/sampled_queries)
        strata.append(dict(name=name,population_zones=len(candidates),sampled_zones=len(chosen),
            population_queries=total_queries,sampled_queries=sampled_queries,
            population_sites=sum(z['indexed_sites'] for z in candidates),
            sampled_sites=sum(z['indexed_sites'] for z in chosen),zone_inclusion_probability=probability))
    selected={z['zone'] for z in zones}; queries=[q for q in panel if q['crm_insee'] in selected]
    if sum(z['queries'] for z in zones)!=len(queries) or any(q['fold'] not in (2,3,4) for q in queries):
        raise ValueError('Entire selected communes/query-only folds required')
    return zones,queries,strata,weights


def projected_rows(ref,panel):
    rows=io.read(verify(ref)); scope={q['query_id'] for q in panel}
    projected=[r for r in rows if r['query_id'] in scope]
    if len(projected)!=len(scope) or {r['query_id'] for r in projected}!=scope:
        raise ValueError('Full selected channel scope')
    return projected


def initialize(out):
    if out.exists(): engine.authority(out); return
    source=engine.authority(SOURCE); census=io.read(verify(source['census']))
    panel=io.read(verify(source['panel'])); zones,selected,strata,weights=sample(census,panel)
    out.mkdir(parents=True); (out/'tmp').mkdir(); (out/'names').mkdir()
    io.write(out/'panel.json',selected); io.write(out/'census.json',{**census,'zones':zones,'queries':len(selected),
        'communes':len(zones),'total_indexed_commune_sites':sum(z['indexed_sites'] for z in zones),
        'population_census':source['census']})
    io.write(out/'sampling.json',dict(seed=SEED,selection='SRS_WITHOUT_REPLACEMENT_COMMUNES_WITHIN_DENSITY_STRATA',
        strata=strata,zone_weights=weights,all_queries_in_selected_communes=True,
        strata_allocations_chosen_on_query_only_cost_and_precision=True,hash_order_not_rerolled=True,
        source_queries=len(panel),queries=len(selected),communes=len(zones),
        sites=sum(z['indexed_sites'] for z in zones),
        kish_query_weight_ess=len(panel)**2/sum(s['population_queries']**2/s['sampled_queries'] for s in strata),
        folds=dict(Counter(str(q['fold']) for q in selected)),departments=dict(Counter(q['crm_insee'][:2] or 'UNKNOWN' for q in selected)),
        missing_number_queries=sum(not q['crm_number'] for q in selected),
        source_folds=dict(Counter(str(q['fold']) for q in panel)),
        source_departments=dict(Counter(q['crm_insee'][:2] or 'UNKNOWN' for q in panel)),
        source_missing_number_queries=sum(not q['crm_number'] for q in panel),truth_opened=False,test_opened=False))
    io.write(out/'lexical.json',projected_rows(source['lexical'],selected))
    geo_exit=io.read(SOURCE/'geo_exit.json')
    if geo_exit.get('exit_code')!=0 or not geo_exit.get('own_child_exit_confirmed'): raise ValueError('Original complete geo owned exit')
    geo_ref=io.read((SOURCE/'geo/outputs.json').with_suffix('.seal.json'))
    io.write(out/'geo.json',projected_rows(geo_ref,selected))
    names=io.read(SOURCE/'names/generation.json'); verify(names['database'])
    io.write(out/'names/generation.json',names); io.write(out/'names/owners.seal.json',names['database'])
    launch={**source,'panel':record(out/'panel.json'),'census':record(out/'census.json'),
        'lexical':record(out/'lexical.json'),'projected_geo':record(out/'geo.json'),
        'sampling':record(out/'sampling.json'),'reused_source_launch':record(SOURCE/'launch.json'),
        'reused_geo_source':geo_ref,'reused_geo_owned_exit':record(SOURCE/'geo_exit.json'),
        'implementations':[*source['implementations'],record(Path(__file__))],
        'measured_dense_variants':['v3_official512'],'old_encoder_not_recomputed':True,
        'scope':'STRATIFIED_COMMUNE_PROBABILITY_PILOT_NOT_CENSUS_NOT_CERTIFICATION'}
    io.write(out/'launch.json',launch)
    # Reuse complete exact zones without opening labels. Retain source receipts.
    reused=[]
    for z in zones:
        zone=z['zone']; src=engine.zone_path(SOURCE,zone); dst=engine.zone_path(out,zone)
        dst.mkdir(parents=True,exist_ok=True)
        for artifact,receipt in (('documents','prepare'),('v3_official512','v3_official512')):
            seal_path=src/(artifact+'.seal.json'); exit_path=src/(receipt+'_exit.json')
            if seal_path.exists() and exit_path.exists():
                exited=io.read(exit_path)
                if exited.get('exit_code')!=0 or not exited.get('own_child_exit_confirmed'): continue
                ref=io.read(seal_path); verify(ref)
                if artifact=='v3_official512':
                    rows=io.read(ref['path']); expected={q['query_id'] for q in selected if q['crm_insee']==zone}
                    if len(rows)!=len(expected) or {r['query_id'] for r in rows}!=expected: raise ValueError('Reused dense scope')
                io.write(dst/(artifact+'.seal.json'),ref)
                io.write(dst/(receipt+'_exit.json'),{**exited,'reused_owned_receipt':record(exit_path)})
                reused.append(dict(zone=zone,artifact=artifact,file=ref,owned_exit=record(exit_path)))
    io.write(out/'reuse.json',dict(source_launch=record(SOURCE/'launch.json'),artefacts=reused,
        names=names['database'],geo=geo_ref,lexical=source['lexical']))


def generation(out):
    from scripts.embeddinggemma2_v5_geo import weighted_fusion
    launch=engine.authority(out); panel=io.read(verify(launch['panel'])); scope={q['query_id'] for q in panel}
    sampling=io.read(verify(launch['sampling'])); census=io.read(verify(launch['census']))
    values=[]
    for zone in census['zones']:
        directory=engine.zone_path(out,zone['zone']); path=directory/'v3_official512.json'
        for tag in ('prepare','v3_official512'):
            exited=io.read(directory/(tag+'_exit.json'))
            if exited.get('exit_code')!=0 or not exited.get('own_child_exit_confirmed'): raise ValueError('Owned zone exit0 required')
        ref=engine.sealed(path)
        if not ref: raise ValueError('Complete dense zone seal')
        values.extend(io.read(ref))
    if len(values)!=len(scope) or {r['query_id'] for r in values}!=scope: raise ValueError('Complete pilot before labels')
    lexical=projected_rows(launch['lexical'],panel); geographic=projected_rows(launch['projected_geo'],panel)
    geo={r['query_id']:r for r in geographic}
    channels={'gemma2':{r['query_id']:r['candidates'] for r in values},
        'lexical':{r['query_id']:r['candidates'] for r in lexical},'geo':{q:r['candidates'] for q,r in geo.items()}}
    channels['gemma2_lexical_geo']={q:weighted_fusion([(channels['gemma2'][q],1.),(channels['lexical'][q],1.),
        (channels['geo'][q],geo[q]['rrf_geo_weight'])]) for q in scope}
    for channel in channels.values():
        for cs in channel.values():
            if len(cs)>100 or len({c['siret'] for c in cs})!=len(cs): raise ValueError('Initial100 cap')
    verify(io.read(out/'names/generation.json')['database'])
    engine.authority(out); io.write(out/'all_outputs.json',channels); ref=engine.seal(out/'all_outputs.json')
    io.write(out/'generation.json',dict(status='SEALED_COMPLETE_PROBABILITY_PILOT',outputs=ref,
        sampling=launch['sampling'],queries=len(scope),truth_opened=False,test_opened=False))
    return channels,scope,launch,sampling


def weighted_report(rows,sampling,field,draws=2000):
    """Poststratified Hajek point; APPROXIMATE stratified-PSU bootstrap only.

    Wilson on raw sample is separate and NOT a CI for this survey estimate.
    Bootstrap cannot reveal unobserved failure classes, especially zero misses.
    """
    import numpy as np
    groups=defaultdict(list)
    for r in rows: groups[r['zone']].append(r)
    rng=np.random.default_rng(20261008); replicas=np.zeros(draws); point=0.; strata={}
    for spec in sampling['strata']:
        name=spec['name']; zs=[z for z,w in sampling['zone_weights'].items() if w['stratum']==name]
        counts=np.asarray([(sum(r[field] for r in groups[z]),len(groups[z])) for z in zs],dtype=float)
        rate=counts[:,0].sum()/counts[:,1].sum(); weight=spec['population_queries']/sampling['source_queries']
        point+=weight*rate; M=spec['population_zones']; n=len(zs)
        # Ordinary n-draw bootstrap underestimates mean variance by (n-1)/n.
        # Correct that AND the without-replacement population fraction.
        fpc=np.sqrt((1-n/M)*n/(n-1)) if n>1 else 0.
        for i in range(draws):
            sampled=counts[rng.integers(0,n,n)].sum(axis=0)
            replicas[i]+=weight*np.clip(rate+fpc*(sampled[0]/sampled[1]-rate),0.,1.)
        strata[name]=dict(sample_hits=int(counts[:,0].sum()),sample_queries=int(counts[:,1].sum()),
            raw_rate=rate,population_share=weight,sampled_communes=n,population_communes=M)
    return dict(poststratified_hajek_estimate=point,approximate_commune_bootstrap95=np.quantile(replicas,[.025,.975]).tolist(),
        strata=strata,confidence_interval_is_approximate=True,not_a_99_2pct_certification=True,
        bootstrap_can_be_degenerate_when_no_observed_misses=True)


def evaluate(out):
    import numpy as np
    import pyarrow.parquet as pq
    from scripts.embeddinggemma2_v3_compare import summarize
    channels,scope,launch,sampling=generation(out)
    manifest=io.read(verify(launch['training_manifest'])); membership=population.SOURCE/'validation_membership.json'
    if io.digest(membership)!=manifest['split_hashes']['validation']: raise ValueError('Label authority drift')
    labels={r['query_id']:r for r in io.read(membership) if r['query_id'] in scope}
    if set(labels)!=scope: raise ValueError('Every selected human query required')
    cfg=io.read(io.ROOT/'config/retrieval_v358_postseal_evaluation.json')
    ev=io.read(verify(cfg['evaluation'])); side=io.read(verify(ev['label_side']))
    if side['never_used_in_retrieval'] is not True or side['ledger']!=cfg['ledger']: raise ValueError('Official sidecar authority')
    for ref in (side['policy'],*side['files'].values()): verify(ref)
    filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))]
    accepted={q:{labels[q]['siret']} for q in scope}
    for r in pq.read_table(side['files']['acceptable_operational.parquet']['path'],filters=filters,use_threads=False).to_pylist(): accepted[r['query_id']].add(r['siret'])
    meta={r['query_id']:r for r in pq.read_table(verify(cfg['ledger']),filters=filters,use_threads=False).to_pylist()}
    roles=io.read(verify(launch['roles'])); outcomes={}; reports={}
    for variant,values in channels.items():
        rows=[]
        for q in sorted(scope):
            gt=labels[q]; cs=values[q]; keys={c['siret'] for c in cs}; w=sampling['zone_weights'][gt['crm_insee']]
            rows.append(dict(query_id=q,zone=gt['crm_insee'],siren_group=gt['siren'],fold=gt['fold'],role=roles[q],
                stratum=w['stratum'],calibrated_query_weight=w['calibrated_query_weight'],
                exact=gt['siret'] in keys,siren=gt['siren'] in {c['siren'] for c in cs},
                operational=bool(keys&accepted[q])))
        segments={}
        for field in ('fold','role','stratum','label_kind','qualification_v2','qualification_v3','data_origin','ground_truth_state'):
            groups=defaultdict(list)
            for r in rows: groups[str(r.get(field,meta.get(r['query_id'],{}).get(field,'UNKNOWN')))].append(r)
            segments[field]={k:summarize(v) for k,v in groups.items()}
        groups=defaultdict(list)
        for r in rows: groups[r['siren_group']].append(r)
        counts=np.asarray([(sum(r['calibrated_query_weight']*r['exact'] for r in v),sum(r['calibrated_query_weight'] for r in v)) for v in groups.values()])
        rng=np.random.default_rng(20261008); draws=[]
        for _ in range(2000):
            sampled=counts[rng.integers(0,len(counts),len(counts))].sum(axis=0); draws.append(sampled[0]/sampled[1])
        reports[variant]=dict(raw_sample=summarize(rows),segments=segments,
            weighted={field:weighted_report(rows,sampling,field) for field in ('exact','siren','operational')},
            approximate_weighted_siren_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),
            siren_groups=len(groups),no_distribution_free_99percent_guarantee=True)
        outcomes[variant]=rows
    lexical={r['query_id']:r for r in outcomes['lexical']}; fused={r['query_id']:r for r in outcomes['gemma2_lexical_geo']}
    io.write(out/'outcomes.json',outcomes)
    io.write(out/'report.json',dict(status='MEASURED_STRATIFIED_COMMUNE_PILOT_FULL_GEMMA_PIPELINE',
        variants=reports,sampling=launch['sampling'],gained=sum(fused[q]['exact'] and not lexical[q]['exact'] for q in scope),
        lost=sum(lexical[q]['exact'] and not fused[q]['exact'] for q in scope),
        all_sample_queries_in_denominator=True,full_selected_commune_corpora=True,
        raw_rate_not_representative_without_weights=True,validation_used_for_checkpoint_selection=True,
        raw_wilson_not_sampling_design_interval=True,
        old_encoder_recomputed=False,test_opened=False,fold0_opened=False,fold1_opened=False,
        full_oof_measured=False,production_slo_measured=False,downstream_models_changed=False,
        no_99_2pct_certification=True,labels_not_used_to_select_cohort=True))


def run(out):
    import psutil
    for p in psutil.process_iter(['pid','cmdline']):
        command=p.info.get('cmdline') or []
        if 'scripts.embeddinggemma2_v14_full_pipeline' in command and str(SOURCE) in command:
            raise ValueError('Exhaustive source worker/controller still alive; do not duplicate')
    if not (SOURCE/'suspension.json').exists():
        io.write(SOURCE/'suspension.json',dict(status='SUSPENDED_BY_EXPLICIT_USER_SCOPE_CHANGE',
            stopped_controller=12216,cooperatively_stopped_worker=24904,controller_session_exit=143,
            artifacts_preserved=True,labels_not_evaluated=True,recorded_unix=time.time()))
    initialize(out); engine.authority(out)
    lock=out/'controller.json'
    if lock.exists():
        owner=io.read(lock)
        if psutil.pid_exists(owner['pid']) and abs(psutil.Process(owner['pid']).create_time()-owner['created'])<1:
            raise ValueError('Active pilot controller')
    io.write(lock,dict(pid=os.getpid(),created=psutil.Process().create_time()))
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        census=io.read(verify(io.read(out/'launch.json')['census']))
        for spec in sorted(census['zones'],key=lambda z:(z['indexed_sites'],z['zone'])):
            engine.owned(out,'prepare',spec['zone'])
            engine.owned(out,'dense',spec['zone'],'v3_official512')
        evaluate(out)
        io.write(out/'state.json',dict(status='COMPLETE_PROBABILITY_PILOT',report=record(out/'report.json'),test_opened=False))
    finally:
        if awake.poll() is None: awake.terminate(); awake.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['run']); p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args()
    if args.output.resolve()!=OUTPUT.resolve(): raise ValueError('Dedicated V15 pilot namespace')
    try: run(args.output)
    except Exception as error:
        if args.output.exists(): io.write(args.output/'execution_error.json',dict(reason=str(error),pid=os.getpid()))
        raise
