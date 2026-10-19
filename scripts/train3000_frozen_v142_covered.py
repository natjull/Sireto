"""Lot A: cache-covered, budgeted subset of the immutable3000; never full.

Only query-conditioned evidence/scores are computed. Complete commune, SITE
and PARENT banks must pre-exist. Labels open only in a separate sealed-lot audit.
"""
from collections import OrderedDict
import datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

import numpy as np
from scripts import train3000_frozen_v142_pipeline as pipeline
from scripts import train3000_frozen_v142_numeric as numeric
mini=numeric.mini
OUT=pipeline.control.OUT/'covered'
BUDGET=6*3600


def prior(refs,directory):
    for ref in refs:
        # The supervisor gains a mode; all scientific/data worker bytes stay fixed.
        path=directory/Path(ref['path']).name if Path(ref['path']).name=='train3000_frozen_v142_supervisor.py' else Path(ref['path'])
        if Path(ref['path']).name=='train3000_frozen_v142_covered.py' and mini.record(path)['sha256']!=ref['sha256']:
            import ast
            archived=mini.record(directory/path.name,ref['sha256'])
            def bodies(p):
                tree=ast.parse(Path(p).read_text())
                return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('ExistingIdentityBank','prepare','evaluate')}
            if bodies(archived['path'])!=bodies(path):raise ValueError('Scientific worker code changed; reuse refused')
        else:mini.record(path,ref['sha256'])


def adopt(supervisor,phase,args):
    # Selection-only coverage correction: preserve successful owned work when
    # the worker bodies, exact inputs and all data dependencies are unchanged.
    for path in sorted((supervisor.OUT/'completed').glob(phase+'_*.json')):
        saved=mini.read(path);d=Path(saved['result']['path']).parent
        old=mini.read(d/'input.json');old.pop('selection',None)
        wanted=dict(args);wanted.pop('selection',None)
        if old!=wanted:continue
        prior(saved['sources'],d)
        ex=mini.read(mini.verified(saved['receipt']))
        if ex['exit_code'] or not ex['owned_child_exit_confirmed']:continue
        mini.verified(saved['result'])
        if phase=='audit':
            for p,expected in mini.read(saved['result']['path'])['signatures'].items():
                stat=Path(p).stat()
                if [stat.st_size,stat.st_mtime_ns,stat.st_ino]!=expected:raise ValueError('Frozen authority changed')
        return dict(saved,compatible_selection_only_resume=True,original_completion=mini.record(path))
    return supervisor.task(phase,args)


def inventory():
    ready=mini.read(pipeline.control.OUT/'ready.json')
    for family in ready['families'].values():
        result=mini.read(mini.verified(family['result_ref']))
        receipt=mini.read(mini.verified(family['receipt_ref']))
        if result['queries']!=20 or result['max_absolute_error']!=0 or receipt['exit_code'] or not receipt['owned_child_exit_confirmed']:
            raise ValueError('Four-family fidelity prerequisite')
    if ready['bano']['context_matches']!=20 or ready['bano']['weighted_delta_matches']!=20:raise ValueError('BANO prerequisite')
    launch=mini.read(numeric.supplier.OUTPUT/'launch.json')
    cache=mini.read(mini.verified(launch['feature_launch']))
    counts=mini.read(mini.bano.cohort.OUT/'commune_counts.json');zones={}
    for zone,ref in cache['corpora'].items():
        base=numeric.engine.zone_path(numeric.own.SOURCE,zone);bank=base/'v3_official512'
        if not bank.exists() and (base/'v3_official512.seal.json').exists():
            bank=Path(mini.read(base/'v3_official512.seal.json')['path']).parent/'v3_official512'
        if (bank/'texts_binding.json').exists() or counts.get(zone)!=None and counts[zone]==0:
            zones[zone]=dict(corpus=ref,site_bank=str(bank),parent_extra=None,kind='frozen',empty=counts.get(zone)==0)
    preparations=[];outputs={}
    for path in sorted((pipeline.control.OUT/'completed').glob('*.json')):
        item=mini.read(path);d=Path(item['result']['path']).parent
        if not (d/'controller.json').exists():continue
        receipt=mini.read(mini.verified(item['receipt']))
        if receipt['exit_code'] or not receipt['owned_child_exit_confirmed']:continue
        result=mini.read(mini.verified(item['result']))
        if result['status'] not in ('COMPLETE_LABEL_FREE_PREPARATION','COMPLETE_NATIVE100') or d.name.startswith('covered-prepare_'):continue
        prior(item['sources'],d)
        if result['status']=='COMPLETE_LABEL_FREE_PREPARATION':preparations.append((item,result,d))
        elif result['status']=='COMPLETE_NATIVE100':outputs[result['query']['query_id']]=item
    if len(preparations)!=1:raise ValueError('One owned, source-compatible complete smoke preparation required')
    proof,prepared,root=preparations[0]
    if prepared['queries']!=20 or prepared['raw_labels_read']:raise ValueError('Complete label-free smoke')
    construction=mini.read(root/'numeric/construction.json')
    if construction['status']!='COMPLETE_QUERY_ONLY_NUMERIC51':raise ValueError('Complete bank provenance')
    census=mini.read(root/'numeric/v14/census.json')['zones']
    for row in census:
        zone=row['zone']
        if zone in zones:continue
        base=numeric.engine.zone_path(root/'numeric/v14',zone)
        docs=numeric.engine.sealed(base/'documents.json')
        complete=numeric.engine.sealed(base/'v3_official512.json')
        bank=base/'v3_official512';parent=root/'numeric/v19/zones'/(zone or 'UNKNOWN')
        if not docs or not complete or not (bank/'texts_binding.json').exists() or not (parent/'new_identities.sqlite').exists():continue
        # Owned exit0 plus unchanged pre-completion files; seal hashes now for reuse.
        paths=[parent/'new_identities.sqlite',*sorted(parent.glob('new_identity_vectors_*.npy'))]
        if any(p.stat().st_mtime_ns>Path(proof['receipt']['path']).stat().st_mtime_ns for p in paths):raise ValueError('PARENT bank modified after owned completion')
        zones[zone]=dict(corpus=mini.record(docs),site_bank=str(bank),kind='owned_complete',
            parent_extra=dict(database=mini.record(paths[0]),vectors=[mini.record(p) for p in paths[1:]]))
    for zone,value in zones.items():
        if value.get('empty'):
            value['site_binding']=None;continue
        binding=mini.read(Path(value['site_bank'])/'binding.json')
        if binding['adapter']['sha256']!=cache['adapter']['sha256']:raise ValueError('Same original SITE encoder')
        value['site_binding']=mini.record(Path(value['site_bank'])/'texts_binding.json')
    # File existence is insufficient: the frozen UNKNOWN cache, for example,
    # may be empty despite non-empty current index scope. Never force its count.
    import ijson
    from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
    wanted={q['crm_insee'] for q in pipeline.selected(mini.read(mini.bano.cohort.OUT/'sample.json')['selected_query_ids'])}
    backend=TantivyBackend(numeric.engine.io.INDEX);rejected=[]
    if backend.searcher.num_docs!=82772691:raise ValueError('Original national index')
    for zone in list(zones):
        if zone not in wanted:continue
        indexed=backend.searcher.search(numeric.engine.old.zone_query(backend,zone),limit=1,count=True).count
        cached=0
        with mini.verified(zones[zone]['corpus']).open('rb') as stream:
            for prefix,event,value in ijson.parse(stream,use_float=True):
                if prefix=='item' and event=='end_map':cached+=1
        if cached!=indexed:
            rejected.append(dict(zone=zone,cached_sites=cached,indexed_sites=indexed,reason='incomplete_prepared_corpus'))
            del zones[zone]
    return dict(zones=zones,rejected_incomplete_zones=rejected,reference_preparation=proof,reference_root=str(root),reference_bindings=prepared['bindings'],
        existing_outputs=outputs,ready_ref=mini.record(pipeline.control.OUT/'ready.json'),
        family_sources='Original full official activity/BANO/OSM and collector authorities; query-conditioned projections are recomputed, never borrowed from another query.')


def plan():
    inv=inventory();sample=mini.read(mini.bano.cohort.OUT/'sample.json')
    panel=pipeline.selected(sample['selected_query_ids']);by_id={q['query_id']:q for q in panel}
    eligible={q['query_id'] for q in panel if q['crm_insee'] in inv['zones']}
    smoke=sample['smoke_query_ids']
    if not set(smoke)<=eligible:raise ValueError('Full smoke banks must remain covered')
    remaining=[qid for qid in sample['selected_query_ids'] if qid not in set(smoke)]
    batches=[remaining[i:i+20] for i in range(0,len(remaining),20)]
    # Keep the canonical full-run encoder companions/padding. Only the choice
    # of batches is cost-based (covered count), never the queries' labels.
    candidates=[dict(companions=ids,ids=[qid for qid in ids if qid in eligible],ordinal=i) for i,ids in enumerate(batches)]
    candidates=[b for b in candidates if b['ids']]
    candidates.sort(key=lambda b:(-len(b['ids']),b['ordinal']))
    ready=mini.read(inv['ready_ref']['path']);reference=mini.read(mini.verified(inv['reference_preparation']['result']))
    # Conservative frozen numeric replay plus the observed smoke residual,
    # deducting only the low end of the documented PARENT-encoding proxy.
    dense=sum(r['seconds'] for r in reference['commune_work']['dense'])
    new_sites=ready['smoke']['preparation_eta_detail']['measured_units']['dense']
    numeric_seconds=max(ready['families']['numeric51']['result']['elapsed_seconds'],reference['stage_seconds']['numeric51']-dense-.009*new_sites)
    common=sum(reference['stage_seconds'][k] for k in ('activity','osm','bano','other_original_collector_contexts'))
    per_batch=1.25*(numeric_seconds+common)
    gen_per=mini.read(mini.ROOT/'docs/train3000_frozen_v142_brief13_result.json')['generation_including_bounded_retry_seconds_per_query']*1.25
    reserve=900.;estimate=reserve+per_batch # actual20-query cache-only fidelity replay
    chosen=[]
    for batch in candidates:
        cost=per_batch+len(batch['ids'])*gen_per
        if estimate+cost>BUDGET:continue
        chosen.append(batch);estimate+=cost
    selected=smoke+[qid for batch in chosen for qid in batch['ids']]
    counts=dict(total=3000,cache_covered=len(eligible),missing_preparation=3000-len(eligible),
        excluded_budget=len(eligible)-len(selected),excluded_total=3000-len(selected),retained=len(selected),already_generated=len(smoke))
    return dict(kind='COVERED_SELECTION_V1',cohort_commit='d54cec2c',sample=mini.record(mini.bano.cohort.OUT/'sample.json'),
        counts=counts,eligible_query_ids=[q['query_id'] for q in panel if q['query_id'] in eligible],selected_query_ids=selected,
        batches=chosen,reference_query_ids=smoke,inventory=inv,estimated_active_seconds=estimate,
        estimated_all_covered_active_seconds=reserve+per_batch*(1+len(candidates))+(len(eligible)-20)*gen_per,
        budget_seconds=BUDGET,estimate_detail=dict(per_batch_seconds=per_batch,generation_seconds_per_query=gen_per,
            reserve_seconds=reserve,numeric_seconds=numeric_seconds,margin=1.25,
            method='Observed cached numeric replay and smoke residual minus documented low-end PARENT proxy;25% margin, canonical batches, no wall-time guarantee.'),
        exclusions=[dict(query_id=q['query_id'],reason='missing_complete_commune_SITE_PARENT' if q['query_id'] not in eligible else 'six_hour_estimated_compute_budget')
            for q in panel if q['query_id'] not in set(selected)],
        label_blind_selection=True,no_new_commune_extraction=True,coverage_biased_not_representative=True)


class ExistingIdentityBank:
    def __init__(self,launch,directory,inference,tokenizer,parents,zones):
        self.old=sqlite3.connect(f"file:{mini.verified(launch['identity_database'])}?mode=ro&immutable=1",uri=True)
        self.cache=numeric.identity.VectorCache(numeric.identity.OUTPUT/'identity_vectors')
        self.offset=self.old.execute('SELECT max(id) FROM texts').fetchone()[0]
        zone='' if directory.name=='UNKNOWN' else directory.name;extra=zones[zone]['parent_extra']
        self.new=None;self.ids={};self.fresh={};self.loaded=OrderedDict()
        if extra:
            self.new=sqlite3.connect(f"file:{mini.verified(extra['database'])}?mode=ro&immutable=1",uri=True)
            files={Path(ref['path']).name:ref for ref in extra['vectors']};encoded=0
            for idx,text in self.new.execute('SELECT id,text FROM texts ORDER BY id'):
                old=self.old.execute('SELECT id FROM texts WHERE text=?',(text,)).fetchone()
                self.ids[idx]=old[0] if old else self.offset+idx
                if not old:
                    path=Path(mini.verified(files[f'new_identity_vectors_{encoded//128*128:08d}.npy'])) if encoded%128==0 else path
                    self.fresh[self.offset+idx-1]=(path,encoded%128);encoded+=1
        for i,parent in enumerate(sorted(parents)):
            if i%256==0:mini.guard()
            if not self.execute('SELECT text_id FROM views WHERE parent=? ORDER BY ordinal',(parent,)):
                raise ValueError('Missing prepared PARENT context; encoding forbidden')
    def execute(self,sql,args):
        rows=self.old.execute(sql,args).fetchall()
        if rows:return rows
        return [(self.ids[r[0]],) for r in self.new.execute(sql,args)] if self.new else []
    def get(self,idx):
        if idx<self.offset:return self.cache.get(idx)
        path,row=self.fresh[idx]
        if path not in self.loaded:
            self.loaded[path]=np.load(path,mmap_mode='r',allow_pickle=False)
            if len(self.loaded)>32:
                _,matrix=self.loaded.popitem(last=False);matrix._mmap.close()
        self.loaded.move_to_end(path)
        return self.loaded[path][row].copy()
    def close(self):
        self.old.close();self.cache.close()
        if self.new:self.new.close()
        for matrix in self.loaded.values():matrix._mmap.close()


def prepare(directory,args):
    selection=mini.read(mini.verified(args['selection']));zones=selection['inventory']['zones']
    companions=pipeline.selected(args['companions']);original=numeric.corpora;reconstruct=numeric.reconstruct
    bank=numeric.IdentityBank;dense=numeric.engine.dense_zone;prepare_zone=numeric.engine.prepare_zone
    def corpora(panel,out,source,cache):
        if any(q['crm_insee'] not in zones for q in panel):raise ValueError('Uncovered commune; extraction forbidden')
        return original(panel,out,source,dict(cache,corpora={z:v['corpus'] for z,v in zones.items()}))
    def existing_zone(out,zone):
        if zone not in numeric.engine.authority(out)['prepared_corpora']:raise ValueError('New commune extraction forbidden')
        mini.verified(zones[zone]['corpus'])
        if zones[zone]['site_binding']:mini.verified(zones[zone]['site_binding'])
        return prepare_zone(out,zone)
    def existing_dense(out,zone,variant):
        if variant!='v3_official512':raise ValueError('Only original SITE bank')
        dest=numeric.engine.zone_path(out,zone)/variant
        os.symlink(zones[zone]['site_bank'],dest,target_is_directory=True)
    numeric.corpora=corpora;numeric.engine.prepare_zone=existing_zone;numeric.engine.dense_zone=existing_dense
    numeric.IdentityBank=lambda *a,**kw:ExistingIdentityBank(*a,**kw,zones=zones)
    numeric.reconstruct=lambda panel,out:reconstruct(panel,out,encoder_panel=companions)
    try:pipeline.prepare(directory,args['ids'])
    finally:
        numeric.corpora=original;numeric.reconstruct=reconstruct;numeric.IdentityBank=bank
        numeric.engine.prepare_zone=prepare_zone;numeric.engine.dense_zone=dense
    result=mini.read(directory/'result.json')
    if args.get('reference_replay'):
        root=Path(selection['inventory']['reference_root']);old=selection['inventory']['reference_bindings']
        for qid,ref in result['bindings'].items():
            fresh=mini.read(mini.verified(ref));expected=mini.read(mini.verified(old[qid]))
            with np.load(mini.verified(fresh['features']),allow_pickle=False) as x,np.load(mini.verified(expected['features']),allow_pickle=False) as y:
                if any(x[k].dtype!=y[k].dtype or not np.array_equal(x[k],y[k]) for k in ('keys','features')):raise ValueError('STOP_CACHE_ONLY_NUMERIC_NONFIDELITY')
            for name in ('context','initials','supplier','address','rna','banatic','osm_context','shapes','bano_context'):
                if mini.read(Path(ref['path']).parent/(name+'.json'))!=mini.read(Path(old[qid]['path']).parent/(name+'.json')):
                    raise ValueError('STOP_CACHE_ONLY_CONTEXT_NONFIDELITY:'+name)
            f1=mini.read(directory/'features_generation.json')['activity_shards'][qid]
            f2=mini.read(root/'features_generation.json')['activity_shards'][qid]
            with np.load(mini.verified(f1),allow_pickle=False) as x,np.load(mini.verified(f2),allow_pickle=False) as y:
                if any(x[k].dtype!=y[k].dtype or not np.array_equal(x[k],y[k]) for k in ('refs','confidence')):raise ValueError('STOP_CACHE_ONLY_ACTIVITY_NONFIDELITY')
        mini.save(directory/'cache_only_fidelity.json',dict(queries=len(result['bindings']),numeric_bitwise=True,all_contexts_identical=True,max_error=0))
    mini.save(directory/'no_extraction.json',dict(new_commune_extractions=0,new_SITE_encodings=0,new_PARENT_encodings=0,
        encoder_companions=args['companions'],only_query_conditioned_evidence_recomputed=True))


def evaluate(directory,seal_ref):
    # This subset-specific authorization does NOT relax the3000 full-run gate.
    import duckdb
    import pyarrow as pa
    import pyarrow.dataset as ds
    from scripts import retrieval_site_operational_metric as metric
    seal=mini.read(mini.verified(seal_ref));selection=mini.read(mini.verified(seal['selection']))
    ids=set(selection['selected_query_ids'])
    if seal['kind']!='COVERED_COMPLETE_NO_LABELS' or set(seal['outputs'])!=ids:raise ValueError('Complete retained-lot seal required')
    outputs={}
    for qid,item in seal['outputs'].items():
        ex=mini.read(mini.verified(item['receipt']))
        if ex['exit_code'] or not ex['owned_child_exit_confirmed']:raise ValueError('Owned exit0 before labels')
        outputs[qid]=mini.read(mini.verified(item['result']))
    binding=mini.read(pipeline.control.OUT/'cohort_binding.json')
    side=mini.verified(binding['refs']['label_side_sample.json'])
    labels={r['query_id']:r for r in mini.bano.cohort.projected(side,
        fields=('query_id','fold','human_ground_truth_siret','human_ground_truth_siren'),selected=ids,root='rows.item')}
    ledger_ref=mini.read(mini.ROOT/'docs/step2_baseline_v142.json')['source_refs']['ledger']
    official=ds.dataset(mini.verified(ledger_ref)).to_table(columns=['query_id','fold','human_ground_truth_siret','human_ground_truth_siren'],
        filter=ds.field('fold').isin([2,3,4]) & ds.field('query_id').isin(sorted(ids))).to_pylist()
    if len(official)!=len(ids) or any(labels.get(r['query_id'],{}).get(k)!=r[k] for r in official for k in r):raise ValueError('No label repair/filtering')
    requested={c['siret'] for part in outputs.values() for c in part['candidates']}
    launch=mini.read(mini.bano.cohort.preflight.RUN/'launch.json');study=metric.read(metric.STUDY_RECEIPT)['file']
    con=duckdb.connect();con.execute('SET threads=1');con.execute("SET memory_limit='512MB'")
    con.execute('SET temp_directory=?',[str(directory/'tmp')]);con.register('requested',pa.table({'siret':sorted(requested)}))
    def load(path,columns):
        cursor=con.execute('SELECT '+','.join('s.'+k for k in columns)+' FROM read_parquet(?) s SEMI JOIN requested r USING(siret)',[str(path)])
        result={}
        while batch:=cursor.fetchmany(4096):
            mini.guard()
            for row in batch:
                value=dict(zip(columns,row));result[value['siret']]=value
        return result
    sites=load(mini.verified(launch['stock_source']),['siret','numeroVoieEtablissement','indiceRepetitionEtablissement','dernierNumeroVoieEtablissement',
        'indiceRepetitionDernierNumeroVoieEtablissement','typeVoieEtablissement','libelleVoieEtablissement','codeCommuneEtablissement'])
    points=load(study,['siret','x','y','qualite_xy','epsg','distance_precision']);con.close();rows=[]
    for qid,part in outputs.items():
        q=part['query'];label=labels[qid];gt=label['human_ground_truth_siret'];company=label['human_ground_truth_siren']
        geometry=mini.read(mini.verified(part['binding']))['geometry'];cp=metric.crm_point(geometry)
        exact_rank=siren_rank=site_rank=None
        for rank,c in enumerate(part['candidates'],1):
            if c['siret']==gt and exact_rank is None:exact_rank=rank
            if c['siren']==company:
                if siren_rank is None:siren_rank=rank
                p=metric.candidate_point(points.get(c['siret']));d,t=metric.distance(cp,p),metric.threshold(cp,p)
                site=metric.address_match(q,sites.get(c['siret'])) or (d is not None and t is not None and d<=t)
                if site and site_rank is None:site_rank=rank
        ranks=[v for v in (exact_rank,site_rank) if v is not None];union_rank=min(ranks) if ranks else None
        rows.append(dict(query_id=qid,fold=q['fold'],exact=exact_rank is not None,siren=siren_rank is not None,site=site_rank is not None,
            site_union=union_rank is not None,exact_rank=exact_rank,siren_rank=siren_rank,site_rank=site_rank,site_union_rank=union_rank,
            candidates=len(part['candidates'])))
    mini.save(directory/'result.json',dict(status='COMPLETE_COVERED_POST_SEAL_EVALUATION',queries=len(ids),
        counts={k:sum(r[k] for r in rows) for k in ('exact','siren','site','site_union')},per_query=rows,
        denominator_filtering=False,descriptive_coverage_biased=True,coefficient_oof_only=True,encoder_oof=False,
        source_refs=dict(ledger=ledger_ref,sidecar=binding['refs']['label_side_sample.json']),generation_seal=seal_ref))


def publish(report):
    # Path-limited commits leave the parallel agent's work/index entries alone.
    while subprocess.run(['git','diff','--quiet','--','handover.md'],cwd=mini.ROOT).returncode:
        print('WAIT_FOREIGN_HANDOVER_60S',flush=True);time.sleep(60)
    stem=mini.ROOT/'docs/train3000_covered_result';jp=stem.with_suffix('.json');mp=stem.with_suffix('.md')
    with jp.open('w') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    c=report.get('selection_counts',{});metrics=report.get('metrics',{});counts=report.get('evaluation',{}).get('counts',{})
    text='# Lot A — '+report['status']+'\n\n'+json.dumps(c,ensure_ascii=False)+'\n\n'
    text+='Générés : '+str(report.get('generated',0))+' ; métriques observées : '+json.dumps(metrics,ensure_ascii=False)+'.\n'
    text+='Présences descriptives exact/SIREN/site/union : '+json.dumps(counts,ensure_ascii=False)+'.\n'
    text+='Rangs par dossier et références : `train3000_covered_result.json`.\n'
    text+='Aucune extraction communale/SITE/PARENT ; scores et quatre contextes de requête recalculés depuis les sources officielles existantes.\n'
    text+='Sélection géographiquement couverte puis budgetée, NON représentative des3000 ; exclus sans prédiction, pas des succès.\n'
    text+='Coefficient-OOF seulement, pas encoder-OOF, aucun recall national/SLO promis ; full3000 non lancé.\n'
    if 'error' in report:text+='STOP : '+report['error']+'\n'
    with mp.open('w') as f:f.write(text)
    head=subprocess.check_output(['git','show','HEAD:handover.md'],cwd=mini.ROOT)
    hand=mini.ROOT/'handover.md'
    if hand.read_bytes()!=head:raise RuntimeError('Concurrent handover edit preserved; report not committed')
    line=('\n- '+datetime.date.today().isoformat()+' — Lot A '+report['status']+' : `docs/train3000_covered_result.{json,md}` ; '
        'commit isolé auto-référencé par `git log -1 --format=%H -- docs/train3000_covered_result.json`. '
        'Retenus/exclus '+json.dumps(c)+' ; générés '+str(report.get('generated',0))+' ; métriques '+json.dumps(metrics)+' ; '
        'présences descriptives '+json.dumps(counts)+'. Aucun full3000, nouvelle extraction, fit/web/action VM/label modifié ; travaux étrangers préservés.\n')
    with hand.open('ab') as f:f.write(line.encode())
    if hand.read_bytes()!=head+line.encode():raise RuntimeError('Concurrent handover edit; no unrelated commit')
    paths=[str(p.relative_to(mini.ROOT)) for p in (jp,mp,hand)]
    subprocess.run(['git','add','--',*paths],cwd=mini.ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','docs: record cache-covered Lot A '+report['status'],'--',*paths],cwd=mini.ROOT,check=True)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=mini.ROOT,text=True).strip()


def run(supervisor):
    import fcntl
    import signal
    OUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,supervisor.control.interrupted);signal.signal(signal.SIGINT,supervisor.control.interrupted)
    with (supervisor.OUT/'owned.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        complete=OUT/'complete.json'
        if complete.exists():
            value=mini.read(complete);mini.verified(value['report'])
            for ref in value['sources']:mini.record(ref['path'],ref['sha256'])
            print('ALREADY_COMPLETE_COVERED',flush=True);return
        outputs={};selection=None;preps=[];started=time.monotonic()
        try:
            selection_path=OUT/'selection_v2.json'
            if not selection_path.exists():mini.save(selection_path,plan())
            selection=mini.read(selection_path);selection_ref=mini.record(selection_path)
            print(json.dumps(dict(phase='SEALED_COVERED_SELECTION',counts=selection['counts'],estimated_seconds=selection['estimated_active_seconds'],
                estimated_all_covered_seconds=selection['estimated_all_covered_active_seconds'])),flush=True)
            audit=adopt(supervisor,'audit',dict(frozen_manifest=mini.record(mini.bano.cohort.preflight.FROZEN),mode='covered'))
            smoke=selection['reference_query_ids']
            replay=adopt(supervisor,'covered-prepare',dict(selection=selection_ref,ids=smoke,companions=smoke,reference_replay=True));preps.append(replay)
            proof=Path(replay['result']['path']).parent/'cache_only_fidelity.json'
            mini.verified(mini.record(proof))
            for qid in smoke:outputs[qid]=selection['inventory']['existing_outputs'][qid]
            for batch in selection['batches']:
                prepared=adopt(supervisor,'covered-prepare',dict(selection=selection_ref,ids=batch['ids'],companions=batch['companions']))
                preps.append(prepared);result=mini.read(mini.verified(prepared['result']));root=Path(prepared['result']['path']).parent
                for qid in batch['ids']:outputs[qid]=adopt(supervisor,'generate',dict(binding=result['bindings'][qid],prepared=str(root)))
            if set(outputs)!=set(selection['selected_query_ids']):raise ValueError('No partial retained-lot seal')
            seal=OUT/'generation_seal.json'
            if not seal.exists():mini.save(seal,dict(kind='COVERED_COMPLETE_NO_LABELS',selection=selection_ref,outputs=outputs,raw_labels_read=False))
            evaluation=supervisor.task('covered-evaluate',dict(seal=mini.record(seal)))
            er=mini.read(mini.verified(evaluation['result']))
            receipts=[mini.read(mini.verified(v['receipt'])) for v in outputs.values()]
            results=[mini.read(mini.verified(v['result'])) for v in outputs.values()]
            prep_receipts=[mini.read(mini.verified(v['receipt'])) for v in preps]
            gen_seconds=sum(r['elapsed_seconds'] for r in receipts);prep_seconds=sum(r['elapsed_seconds'] for r in prep_receipts)
            new_seconds=sum(mini.read(mini.verified(v['receipt']))['elapsed_seconds'] for qid,v in outputs.items() if qid not in set(smoke))
            metrics=dict(generation_seconds=gen_seconds,generation_seconds_per_query=gen_seconds/len(outputs),preparation_seconds=prep_seconds,
                preparation_seconds_per_query=prep_seconds/len(outputs),new_generation_seconds=new_seconds,
                new_generation_seconds_per_query=new_seconds/(len(outputs)-len(smoke)) if len(outputs)>len(smoke) else None,
                generation_peak_rss_bytes=max([r['peak_sampled_child_rss_bytes'] for r in receipts]+[r['peak_rss_bytes'] for r in results]),
                preparation_peak_rss_bytes=max(r['peak_sampled_child_rss_bytes'] for r in prep_receipts),
                elapsed_controller_seconds=time.monotonic()-started,process_times_exclude_resource_waits=True)
            report=dict(status='COMPLETE',selection_counts=selection['counts'],selection=selection_ref,generated=len(outputs),
                new_generated=len(outputs)-len(smoke),reused_generated=len(smoke),
                metrics=metrics,evaluation=er,evaluation_receipt=evaluation,outputs=outputs,preparations=preps,cache_only_fidelity=mini.record(proof),
                full_hash_audit=audit,source_refs=supervisor.sources(),full3000_started=False,new_commune_extractions=0,
                new_SITE_encodings=0,new_PARENT_encodings=0,labels_only_after_complete_lot_seal=True,
                estimate_seconds=selection['estimated_active_seconds'],budget_reduction=selection['counts']['excluded_budget']>0)
            mini.save(OUT/'result.json',report);commit=publish(report)
            mini.save(complete,dict(commit=commit,report=mini.record(OUT/'result.json'),sources=supervisor.sources()))
        except Exception as error:
            report=dict(status='STOP',error=repr(error),generated=len(outputs),selection_counts=selection['counts'] if selection else {},
                source_refs=supervisor.sources(),full3000_started=False,labels_not_used_to_select=True)
            mini.save(OUT/('stop_'+str(time.time_ns())+'.json'),report);publish(report);raise
