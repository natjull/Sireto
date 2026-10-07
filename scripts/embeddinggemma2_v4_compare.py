"""Paired initial100 searches, full official communes, global lexical channel.

Selection is CRM/hash-only. Labels are read only by evaluate after owned exits
and complete seals. This development diagnostic is NOT a national/OOF gate.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v2_data as data
from scripts import embeddinggemma2_v4_fit as fit
from scripts import embeddinggemma2_compare as old
from scripts import retrieval_v38_initial_top100 as lexical
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend,RetrievalQuery
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v4'
SEED='gemma2-v3-paired-complete-zones-before-outcomes-20261008'
VARIANTS=('v1_native160','v1_official512','v3_official512')


def fused100(channels,constant=60):
    """Fixed RRF of independently emitted initial100 channels, no trained ranker."""
    scores=Counter(); parents={}
    for rows in channels:
        if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows): raise ValueError('Invalid channel cap')
        for rank,row in enumerate(rows,1):
            scores[row['siret']]+=1/(constant+rank); parents[row['siret']]=row['siren']
    return [dict(siret=key,siren=parents[key],score=float(scores[key]))
        for key in sorted(scores,key=lambda key:(-scores[key],key))[:100]]


def views_for_site(doc,extra,tokenizer):
    anchor=io.clean_text(doc['own_names'][:1],doc['all_addresses'][:1],doc['postcode'])
    return data.pack_atoms(data.site_atoms(doc,extra),tokenizer,512,anchor)[0]


def prepare(night,out):
    import pyarrow.dataset as ds
    from transformers import AutoTokenizer
    out.mkdir(parents=True,exist_ok=False)
    source=io.read(io.HOME/'night_20261006_a/manifest.json')
    membership=io.HOME/'night_20261006_a/validation_membership.json'
    if io.digest(membership)!=source['split_hashes']['validation']: raise ValueError('Validation split drift')
    queries=[{k:r[k] for k in old.CRM} for r in io.read(membership)]
    if len(queries)!=2212 or any(q['fold'] not in (2,3,4) for q in queries): raise ValueError('Safe validation only')
    counts=Counter(q['crm_insee'] for q in queries if q['crm_insee'])
    backend=TantivyBackend(io.INDEX); before=old.inventory()
    if backend.searcher.num_docs!=82772691: raise ValueError('Full national official index drift')
    selected=[]; inspected=[]; total_sites=0; query_count=0
    # Geographic development cohort, predeclared without labels or outcome selection.
    for zone in sorted((z for z in counts if counts[z]>=3),key=lambda z:hashlib.sha256((SEED+'|'+z).encode()).digest()):
        n=backend.searcher.search(old.zone_query(backend,zone),limit=1,count=True).count
        inspected.append(dict(zone=zone,official_sites=n,queries=counts[zone]))
        if n<100 or n>20000 or total_sites+n>120000: continue
        selected.append(dict(zone=zone,official_sites=n,queries=counts[zone])); total_sites+=n; query_count+=counts[zone]
        if query_count>=128: break
    if not selected: raise ValueError('No complete official commune')
    chosen={z['zone'] for z in selected}; panel=[q for q in queries if q['crm_insee'] in chosen]
    io.write(out/'panel.json',panel)
    zones={}; parents=set()
    for zone in selected:
        hits=backend.searcher.search(old.zone_query(backend,zone['zone']),limit=zone['official_sites']+1,count=True)
        if hits.count!=zone['official_sites'] or len(hits.hits)!=hits.count: raise ValueError('Incomplete official commune')
        docs=[]
        for _,loc in hits.hits:
            d=backend.searcher.doc(loc).to_dict(); native=data.native_document(backend,d)
            native['native_text']=old.document_text(backend,d); docs.append(native); parents.add(native['siren'])
        docs.sort(key=lambda d:d['key'])
        if len({d['key'] for d in docs})!=len(docs): raise ValueError('Duplicate site')
        zones[zone['zone']]=docs
        io.write(out/'state.json',dict(status='PREPARING_FULL_OFFICIAL_COMMUNES',zones=len(zones),sites=sum(map(len,zones.values()))))
    # Equal official enrichment for EVERY corpus row, never truth-only hydration.
    dictionary=io.read(verify(data.DICTIONARY)); names=defaultdict(lambda:dict(legal=set(),sites=set())); own=defaultdict(set); refs=[]
    if dictionary['status']!='SEALED_V130_COMPLETE_ALL_OFFICIAL_NAME_OWNER_DICTIONARY': raise ValueError('Name dictionary authority')
    for i,bucket in enumerate(dictionary['buckets']):
        f=bucket['artifacts']['facts.parquet']; n=bucket['artifacts']['names.parquet']; refs.extend([f,n])
        facts=ds.dataset(verify(f)).to_table(filter=ds.field('parent_key').isin([int(p) for p in parents]),
            columns=['name_id','parent_key','owner_key','subject_code','scope_code']).to_pylist()
        npth=verify(n)
        if facts:
            values={r['name_id']:data.safe(r['name_text']) for r in ds.dataset(npth).to_table(
                filter=ds.field('name_id').isin(sorted({r['name_id'] for r in facts})),columns=['name_id','name_text']).to_pylist()}
            for r in facts:
                text=values[r['name_id']]
                if not text: continue
                parent=f"{r['parent_key']:09d}"
                if r['subject_code']==1 and r['scope_code'] in (1,2): names[parent]['legal'].add(text)
                elif r['subject_code']==2 and r['scope_code']==3:
                    names[parent]['sites'].add(text); own[f"{r['owner_key']:014d}"].add(text)
                else: raise ValueError('Official name provenance drift')
        io.write(out/'state.json',dict(status='OFFICIAL_NAMES_ALL_DISTRACTORS',buckets=i+1,total=64))
    tokenizer=AutoTokenizer.from_pretrained(io.HOME/'base',local_files_only=True); parent_views={}
    for parent in sorted(parents):
        ns={k:sorted(v) for k,v in names[parent].items()}
        parent_views[parent]=data.pack_atoms(data.identity_atoms(ns),tokenizer,512)[0]
    corpora=[]
    for zone in selected:
        docs=zones[zone['zone']]
        for d in docs:
            extra=dict(legal=sorted(names[d['siren']]['legal']),own_site=sorted(own[d['key']]))
            d['site_views']=views_for_site(d,extra,tokenizer)
        path=out/(zone['zone']+'.json'); io.write(path,docs); corpora.append({**zone,'file':record(path)})
    io.write(out/'parent_views.json',parent_views)
    lexical_rows=[]
    for i,q in enumerate(panel):
        rq=RetrievalQuery(name=data.safe(q['crm_name']),address=' '.join(data.safe(q[k]) for k in ('crm_number','crm_suffix','crm_street')),
            number=q['crm_number'],insee=q['crm_insee'],postcode=q['crm_postcode'])
        t=time.monotonic(); rows=lexical.search(backend,rq,'primary')
        lexical_rows.append(dict(query_id=q['query_id'],fold=q['fold'],zone=q['crm_insee'],candidates=rows,seconds=time.monotonic()-t))
        if i%16==0: io.write(out/'state.json',dict(status='NATIONAL_LEXICAL_INITIAL100',queries=i+1,total=len(panel)))
    io.write(out/'lexical.json',lexical_rows)
    if old.inventory()!=before: raise ValueError('Official index changed')
    io.write(out/'launch.json',dict(panel=record(out/'panel.json'),corpora=corpora,parent_views=record(out/'parent_views.json'),
        lexical=record(out/'lexical.json'),selection_seed=SEED,selection_inspected=inspected,requested_queries=128,actual_queries=len(panel),
        corpus_total_sites=total_sites,selection_query_only=True,complete_all_indexed_sites_per_selected_commune=True,
        corpus_site_budget120000=True,gt_outside_partition_not_filtered=True,dictionary=data.DICTIONARY,consumed_official_name_refs=refs,
        base_revision=io.REVISION,v1_adapter=record(fit.INITIAL),night=record(night/'night.json'),night_config=record(night/'config.json'),
        requested_v3_adapter_path=str(night/'fit/best_adapter.pt'),training_manifest=record(io.HOME/'night_20261006_a/manifest.json'),
        index_manifest=record(io.INDEX/'manifest.json'),index_meta=record(io.INDEX/'tantivy/meta.json'),index_inventory=before,
        implementation={str(p):record(p) for p in (Path(__file__),Path(fit.__file__),Path(data.__file__),Path(old.__file__),Path(lexical.__file__))},
        variants=VARIANTS,dual_dense_weights=dict(site=.75,identity=.25),rrf_constant=60,
        scope='DEVELOPMENT_VALIDATION_GEOGRAPHIC_COHORT_NOT_NATIONAL_NOT_OOF_NOT_UNTOUCHED_TEST',
        test_opened=False,fold0_opened=False,fold1_opened=False,downstream_models_changed=False))
    io.write(out/'state.json',dict(status='PREPARED_EVALUATION_WAITING_TRAINING',queries=len(panel),sites=total_sites))


def verify_launch(launch):
    for ref in launch['implementation'].values(): verify(ref)
    verify(launch['index_manifest']); verify(launch['index_meta']); verify(launch['night']); verify(launch['night_config'])
    if old.inventory()!=[tuple(r) for r in launch['index_inventory']]: raise ValueError('Official index inventory drift')


def generate(out,variant):
    if variant not in VARIANTS: raise ValueError('Predeclared variant required')
    launch=io.read(out/'launch.json'); verify_launch(launch)
    night=io.read(verify(launch['night'])); deadline=night['deadline_unix']
    panel=io.read(verify(launch['panel'])); parents=io.read(verify(launch['parent_views'])); cfg=io.read(verify(launch['night_config']))
    adapter=launch['v1_adapter'] if variant!='v3_official512' else record(Path(launch['requested_v3_adapter_path']))
    if variant=='v3_official512':
        import torch
        fit_root=Path(launch['requested_v3_adapter_path']).parent
        training_state=io.read(fit_root/'state.json')
        if training_state['status'] not in ('FIT_COMPLETE','EARLY_STOP','TIME_BUDGET_CHECKPOINTED'):
            raise ValueError('Training still running or interrupted')
        checkpoint=torch.load(verify(adapter),map_location='cpu',weights_only=True)
        if checkpoint['launch_sha256']!=io.digest(fit_root/'launch.json'): raise ValueError('Best adapter training binding')
        if checkpoint['step']>training_state['step']: raise ValueError('Impossible adapter step')
    if variant=='v1_native160':
        mode,inference,tokenizer,step=old.load_encoder(verify(adapter)); mode(True)
    else:
        model,params,encode,inference,tokenizer=fit.load_model(cfg,verify(adapter))
        import torch
        step=torch.load(verify(adapter),map_location='cpu',weights_only=True)['step']
    binding=dict(adapter=adapter,launch=record(out/'launch.json'),variant=variant,step=step)
    if variant=='v3_official512': binding['training_state']=record(fit_root/'state.json')
    bindpath=out/(variant+'_binding.json')
    if bindpath.exists():
        if io.read(bindpath)!=binding: raise ValueError('Comparison resume binding mismatch')
    else: io.write(bindpath,binding)
    results=[]; encoded=0; truncated=0; started=time.monotonic()
    for zone in launch['corpora']:
        path=out/(variant+'_'+zone['zone']+'.json'); refpath=path.with_suffix('.seal.json')
        if refpath.exists():
            results.extend(io.read(verify(io.read(refpath)))); continue
        queries=[q for q in panel if q['crm_insee']==zone['zone']]; docs=io.read(verify(zone['file']))
        best=[[] for _ in queries]
        if variant=='v1_native160':
            qe=inference([io.QUERY_PROMPT+old.query_text(q) for q in queries])
            texts=list(dict.fromkeys(io.DOC_PROMPT+d['native_text'] for d in docs))
        else:
            qe=inference([io.QUERY_PROMPT+old.query_text(q) for q in queries])
            qi=inference([io.QUERY_PROMPT+'noms: '+data.safe(q['crm_name']) for q in queries])
            unique_parents=sorted({d['siren'] for d in docs}); name_texts=list(dict.fromkeys(io.DOC_PROMPT+v for p in unique_parents for v in parents[p]))
            pv={}
            for i in range(0,len(name_texts),128):
                fit.resource_guard(deadline,cfg['max_rss_bytes'])
                pv.update(zip(name_texts[i:i+128],inference(name_texts[i:i+128]))); encoded+=len(name_texts[i:i+128])
            parent_scores={p:(qi@np.stack([pv[io.DOC_PROMPT+v] for v in parents[p]]).T).max(axis=1) for p in unique_parents}
            texts=list(dict.fromkeys(io.DOC_PROMPT+v for d in docs for v in d['site_views']))
        vectors={}
        for i in range(0,len(texts),128):
            fit.resource_guard(deadline,cfg['max_rss_bytes'])
            batch=texts[i:i+128]; vectors.update(zip(batch,inference(batch))); encoded+=len(batch)
            if variant=='v1_native160': truncated+=sum(len(x)>160 for x in tokenizer(batch,truncation=False)['input_ids'])
            io.write(out/'state.json',dict(status='INITIAL100_GENERATING',variant=variant,zone=zone['zone'],encoded_texts=encoded,
                zone_encoded=min(i+128,len(texts)),zone_total=len(texts),seconds=time.monotonic()-started))
        for i in range(0,len(docs),128):
            chunk=docs[i:i+128]
            if variant=='v1_native160': similarities=qe@np.stack([vectors[io.DOC_PROMPT+d['native_text']] for d in chunk]).T
            else:
                similarities=np.column_stack([.75*(qe@np.stack([vectors[io.DOC_PROMPT+v] for v in d['site_views']]).T).max(axis=1)
                    +.25*parent_scores[d['siren']] for d in chunk])
            sites=[dict(siret=d['key'],siren=d['siren']) for d in chunk]
            for j in range(len(queries)): best[j]=old.top100(best[j],similarities[j],sites)
        rows=[dict(query_id=q['query_id'],fold=q['fold'],zone=zone['zone'],variant=variant,
            candidates=[dict(score=s,siret=t,siren=n) for s,t,n in top]) for q,top in zip(queries,best)]
        if any(len(r['candidates'])!=min(100,len(docs)) or len({d['siret'] for d in r['candidates']})!=len(r['candidates']) for r in rows): raise ValueError('Initial100 cap/uniqueness')
        io.write(path,rows); io.write(refpath,record(path)); results.extend(rows)
    if {r['query_id'] for r in results}!={q['query_id'] for q in panel} or len(results)!=len(panel): raise ValueError('Complete panel')
    verify_launch(launch); verify(adapter)
    output=out/(variant+'_outputs.json'); io.write(output,results)
    io.write(out/(variant+'_generation.json'),dict(status='SEALED_COMPLETE_PAIRED_INITIAL100',outputs=record(output),binding=record(bindpath),
        queries=len(panel),encoded_texts_this_attempt=encoded,truncated_texts_this_attempt=truncated,seconds_this_attempt=time.monotonic()-started,
        truth_opened=False,test_opened=False,national_dense_retrieval=False))


def summarize(rows):
    n=len(rows); hits=sum(r['exact'] for r in rows); p=hits/n; z=1.959963984540054; d=1+z*z/n
    center=(p+z*z/(2*n))/d; width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return dict(queries=n,exact_hits=hits,exact_recall=p,siren_hits=sum(r['siren'] for r in rows),
        operational_hits=sum(r['operational'] for r in rows),wilson95_lower=center-width,wilson95_upper=center+width)


def evaluate(out):
    import pyarrow.parquet as pq
    launch=io.read(out/'launch.json'); verify_launch(launch); panel=io.read(verify(launch['panel'])); scope={q['query_id'] for q in panel}
    candidates={'national_lexical':{r['query_id']:r['candidates'] for r in io.read(verify(launch['lexical']))}}
    generation_refs=[]
    # All predeclared outputs and actual owned exits required BEFORE opening labels.
    for variant in VARIANTS:
        gen=io.read(out/(variant+'_generation.json')); ex=io.read(out/(variant+'_exit.json'))
        if gen['status']!='SEALED_COMPLETE_PAIRED_INITIAL100' or ex.get('exit_code')!=0 or not ex.get('own_child_exit_confirmed'): raise ValueError('Unsealed/incomplete variant')
        rows=io.read(verify(gen['outputs']))
        if len(rows)!=len(scope) or {r['query_id'] for r in rows}!=scope: raise ValueError('Missing variant queries')
        candidates[variant]={r['query_id']:r['candidates'] for r in rows}; generation_refs.append(record(out/(variant+'_generation.json')))
        candidates[variant+'_rrf']={q:fused100([candidates['national_lexical'][q],candidates[variant][q]]) for q in scope}
    io.write(out/'all_outputs_sealed.json',dict(candidates=candidates,generations=generation_refs,truth_opened=False))
    # Only here are exact and acceptable operational labels read.
    manifest=io.read(verify(launch['training_manifest'])); membership=io.HOME/'night_20261006_a/validation_membership.json'
    if io.digest(membership)!=manifest['split_hashes']['validation']: raise ValueError('Label membership drift')
    labels={r['query_id']:r for r in io.read(membership) if r['query_id'] in scope}
    if set(labels)!=scope: raise ValueError('Missing human labels')
    cfg=io.read(io.ROOT/'config/retrieval_v358_postseal_evaluation.json'); ev=io.read(verify(cfg['evaluation'])); side=io.read(verify(ev['label_side']))
    if side['never_used_in_retrieval'] is not True or side['ledger']!=cfg['ledger']: raise ValueError('Official sealed sidecar authority')
    for ref in (side['policy'],*side['files'].values()): verify(ref)
    accepts=pq.read_table(side['files']['acceptable_operational.parquet']['path'],filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))],use_threads=False).to_pylist()
    acceptable={q:{labels[q]['siret']} for q in scope}
    for r in accepts: acceptable[r['query_id']].add(r['siret'])
    ledger=pq.read_table(verify(cfg['ledger']),filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))],use_threads=False).to_pylist()
    meta={r['query_id']:r for r in ledger}; outcomes={}; reports={}
    for variant,byid in candidates.items():
        rows=[]
        for q in sorted(scope):
            cs=byid[q]
            if len(cs)>100 or len({d['siret'] for d in cs})!=len(cs): raise ValueError('Final initial100 cap')
            ids={d['siret'] for d in cs}; gt=labels[q]
            rows.append(dict(query_id=q,fold=gt['fold'],zone=gt['crm_insee'],exact=gt['siret'] in ids,
                siren=gt['siren'] in {d['siren'] for d in cs},operational=bool(ids&acceptable[q]),
                exact_rank=next((i+1 for i,d in enumerate(cs) if d['siret']==gt['siret']),None)))
        outcomes[variant]=rows; segments={}
        for field in ('fold','label_kind','qualification_v2','qualification_v3','data_origin','ground_truth_state'):
            groups=defaultdict(list)
            for r in rows: groups[str(meta.get(r['query_id'],{}).get(field,'UNKNOWN'))].append(r)
            segments[field]={key:summarize(values) for key,values in groups.items()}
        reports[variant]={**summarize(rows),'segments':segments}
    baseline={r['query_id']:r for r in outcomes['v1_official512']}; new={r['query_id']:r for r in outcomes['v3_official512']}
    corpus={z['zone']:{d['key'] for d in io.read(verify(z['file']))} for z in launch['corpora']}
    outside=[q for q in sorted(scope) if labels[q]['siret'] not in corpus[labels[q]['crm_insee']]]
    io.write(out/'outcomes.json',outcomes)
    io.write(out/'report.json',dict(status='MEASURED_DEVELOPMENT_PAIRED_INITIAL100_NOT_NATIONAL_GATE',variants=reports,
        v3_vs_v1_same_official_texts_gained=sum(new[q]['exact'] and not baseline[q]['exact'] for q in scope),
        v3_vs_v1_same_official_texts_lost=sum(baseline[q]['exact'] and not new[q]['exact'] for q in scope),
        gt_outside_commune_partition=outside,all_queries_in_denominator=True,all_outputs=record(out/'all_outputs_sealed.json'),
        operational_sidecar=ev['label_side'],national_dense_recall_measured=False,full_oof_measured=False,production_slo_measured=False,
        test_opened=False,fold0_opened=False,fold1_opened=False,downstream_models_changed=False))


def run(out):
    launch=io.read(out/'launch.json'); deadline=io.read(verify(launch['night']))['deadline_unix']
    for variant in VARIANTS:
        if (out/(variant+'_generation.json')).exists() and (out/(variant+'_exit.json')).exists():
            ex=io.read(out/(variant+'_exit.json'))
            if ex.get('exit_code')==0: continue
        while time.time()<deadline:
            with (out/(variant+'.log')).open('a') as log:
                child=subprocess.run([sys.executable,'-u','-m','scripts.embeddinggemma2_v4_compare','generate',
                    '--output',str(out),'--variant',variant],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
            io.write(out/(variant+'_exit.json'),dict(own_child_exit_confirmed=True,exit_code=child.returncode))
            if child.returncode==0: break
            if child.returncode!=75: raise RuntimeError('Generation failed '+variant)
            time.sleep(30)
        else: raise TimeoutError('Comparison night deadline')
    evaluate(out); io.write(out/'state.json',dict(status='COMPLETE_PAIRED_INITIAL100_COMPARISON',test_opened=False))


def night_run(night,out):
    """Prepare while GPU mines, then launch the paired searches after actual fit exit."""
    deadline=io.read(night/'night.json')['deadline_unix']
    try:
        if not (out/'launch.json').exists(): prepare(night,out)
        while time.time()<deadline:
            state=io.read(night/'state.json')
            if state['status']=='FIT_DONE_EVALUATION_PENDING': break
            if state['status']=='NIGHT_INTERRUPTED': raise RuntimeError('Training pipeline interrupted: '+state['reason'])
            io.write(out/'night_watch.json',dict(status='WAITING_FIT',training_status=state['status'],deadline_unix=deadline))
            time.sleep(30)
        else: raise TimeoutError('Night ended before paired evaluation')
        io.write(out/'night_watch.json',dict(status='PAIRED_SEARCH_RUNNING',deadline_unix=deadline))
        run(out)
        io.write(out/'night_watch.json',dict(status='COMPLETE_MEASURED',report=record(out/'report.json'),ended_unix=time.time()))
    except BaseException as exc:
        if out.exists(): io.write(out/'night_watch.json',dict(status='INTERRUPTED',reason=str(exc),ended_unix=time.time()))
        raise


def prepared_continuation(source,night,out):
    """Reuse exactly the preselected 80 CRM queries and full official corpora."""
    source_launch=io.read(source/'launch.json'); verify_launch(source_launch)
    out.mkdir(parents=True,exist_ok=False)
    launch={**source_launch,
        'night':record(night/'night.json'),'night_config':record(night/'config.json'),
        'requested_v3_adapter_path':str(night/'fit/best_adapter.pt'),
        'implementation':{str(p):record(p) for p in (Path(__file__),Path(fit.__file__),Path(data.__file__),Path(old.__file__),Path(lexical.__file__))},
        'resource_only_source_comparison':record(source/'launch.json')}
    io.write(out/'launch.json',launch)
    io.write(out/'state.json',dict(status='PREPARED_EVALUATION_WAITING_TRAINING',queries=launch['actual_queries'],
        sites=launch['corpus_total_sites'],cohort_and_official_texts_unchanged=True))
    night_run(night,out)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['continue-prepared','run','generate','evaluate'])
    p.add_argument('--source',type=Path); p.add_argument('--night',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS); a=p.parse_args()
    if not a.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated V4 comparison namespace')
    try:
        if a.action=='continue-prepared': prepared_continuation(a.source,a.night,a.output)
        elif a.action=='run': run(a.output)
        elif a.action=='generate': generate(a.output,a.variant)
        else: evaluate(a.output)
    except fit.ResourceYield as exc:
        io.write(a.output/'resource_yield.json',dict(reason=str(exc),when_unix=time.time())); sys.exit(75)

