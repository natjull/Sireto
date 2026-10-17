"""Label-free preparation/native100 workers for the sealed v174 cohort.

Invoked only by the serial, admitted supervisor. Evaluation is a DIFFERENT
worker after the generation seal. No fitting or model-metadata editing.
"""
import hashlib
import os
from pathlib import Path
import resource
import sqlite3
import time

import numpy as np
from scripts import train3000_frozen_v142_brief10 as control
from scripts import train3000_frozen_v142_numeric as numeric
mini=control.mini


def selected(ids,verify_source=False):
    binding=mini.read(control.OUT/'cohort_binding.json')
    for name in ('sample.json','queries.json','assignment.json'):mini.verified(binding['refs'][name])
    panel=mini.read(mini.bano.cohort.OUT/'queries.json')
    sample=mini.read(mini.bano.cohort.OUT/'sample.json')
    if len(panel)!=3000 or mini.bano.cohort.digest_ids(q['query_id'] for q in panel)!=sample['selected_id_set_sha256']:
        raise ValueError('Sealed draw drift')
    if len(ids)!=len(set(ids)) or not set(ids)<={q['query_id'] for q in panel}:
        raise ValueError('Only the sealed cohort')
    out=[q for q in panel if q['query_id'] in set(ids)]
    if any(set(q)!=set(mini.bano.cohort.old.CRM) or q['fold'] not in (2,3,4) for q in out):
        raise ValueError('Query-only CRM8/folds234 required')
    if verify_source:
        import pyarrow.parquet as pq
        actual={}
        for ref in sample['query_sources']:
            mini.guard()
            rows=pq.read_table(mini.verified(ref),columns=list(mini.bano.cohort.old.CRM),
                filters=[('fold','in',[2,3,4]),('query_id','in',sorted(ids))],use_threads=False).to_pylist()
            for q in rows:
                if q['query_id'] in actual:raise ValueError('Duplicate original query projection')
                actual[q['query_id']]=q
        if actual!={q['query_id']:q for q in out}:raise ValueError('CRM cache differs from original query-only source')
    return out


def prepare(directory,ids):
    from scripts import train3000_frozen_v142_activity as activity
    from scripts import train3000_frozen_v142_osm as osm
    from scripts import train3000_frozen_v142_contexts as contexts
    from scripts import embeddinggemma2_v142_bano_native_ablation as v142
    started=time.monotonic();panel=selected(ids,verify_source=True);times={}
    before=time.monotonic()
    values,body=activity.context(panel,directory/'activity')
    activity_shards={};target=directory/'activity_shards';target.mkdir()
    for q in panel:
        qid=q['query_id'];(refs,confidence),_=activity.project(values.pop(qid))
        path=target/(hashlib.sha256(qid.encode()).hexdigest()+'.npz')
        np.savez_compressed(path,refs=refs,confidence=confidence);activity_shards[qid]=mini.record(path)
    del values
    times['activity']=time.monotonic()-before
    before=time.monotonic();numeric_root=directory/'numeric'
    work=dict(prepare=[],dense=[],frozen_prepared_zones=[],historical_site_bank_zones=[])
    original_prepare=numeric.engine.prepare_zone;original_dense=numeric.engine.dense_zone
    def prepare_zone(out,zone):
        launch=numeric.engine.authority(out)
        work['frozen_prepared_zones']=sorted(launch['prepared_corpora'])
        spec=next(z for z in mini.read(mini.verified(launch['census']))['zones'] if z['zone']==zone)
        start=time.monotonic();result=original_prepare(out,zone)
        work['prepare'].append(dict(zone=zone,indexed_sites=spec['indexed_sites'],
            frozen_prepared_reuse=zone in launch['prepared_corpora'],seconds=time.monotonic()-start))
        return result
    def dense_zone(out,zone,variant):
        start=time.monotonic();result=original_dense(out,zone,variant)
        work['dense'].append(dict(zone=zone,variant=variant,seconds=time.monotonic()-start))
        return result
    for path in (numeric.own.SOURCE/'zones').iterdir():
        if (path/'v3_official512').exists() or (path/'v3_official512.seal.json').exists():
            work['historical_site_bank_zones'].append('' if path.name=='UNKNOWN' else path.name)
    numeric.engine.prepare_zone=prepare_zone;numeric.engine.dense_zone=dense_zone
    try:refs=numeric.reconstruct(panel,numeric_root)
    finally:numeric.engine.prepare_zone=original_prepare;numeric.engine.dense_zone=original_dense
    times['numeric51']=time.monotonic()-before
    construction=mini.read(numeric_root/'geometry/geometries.json')
    before=time.monotonic();osm_context,shapes,_=osm.context(panel,construction,directory/'osm')
    times['osm']=time.monotonic()-before
    before=time.monotonic()
    launch=mini.read(mini.bano.cohort.preflight.RUN/'launch.json')
    bg=mini.read(mini.verified(launch['bano_generation']))
    build,_,_,_=mini.bano.context_builder()
    with sqlite3.connect(f"file:{mini.verified(bg['index'])}?mode=ro&immutable=1",uri=True) as con:
        con.execute('PRAGMA query_only=ON');bano_context=build(panel,con)
    mini.save(directory/'bano_context.json',bano_context)
    times['bano']=time.monotonic()-before
    before=time.monotonic();other=contexts.prepare(panel,numeric_root,directory/'collector_contexts')
    times['other_original_collector_contexts']=time.monotonic()-before
    fg=mini.read(numeric_root/'v75/features_generation.json')
    fg.update(activity_shards=activity_shards,context_expansion=other['context'],
        initial_alignment=other['initials'],indirect_names=other['indirect'])
    mini.save(directory/'features_generation.json',fg)
    mini.save(directory/'launch.json',dict(launch,panel=mini.record(numeric_root/'panel.json')))
    mappings={k:mini.read(mini.verified(v)) for k,v in other.items() if k!='indirect'}
    assignment=mini.read(mini.bano.cohort.OUT/'assignment.json')
    bindings={};root=directory/'queries';root.mkdir()
    for q in panel:
        qid=q['query_id'];d=root/hashlib.sha256(qid.encode()).hexdigest();d.mkdir()
        for name,value in dict(context=mappings['context'][qid],initials=mappings['initials'][qid],
            supplier=mappings['supplier'][qid],address=mappings['address'][qid],rna=mappings['rna'][qid],
            banatic=mappings['banatic'][qid],osm_context=osm_context[qid],shapes=shapes[qid],bano_context=bano_context[qid]).items():
            mini.save(d/(name+'.json'),value)
        mini.save(d/'binding.json',dict(query=q,features=refs[qid],geometry=construction[qid],
            model_id=assignment[qid],raw_labels_read=False))
        bindings[qid]=mini.record(d/'binding.json')
    mini.save(directory/'result.json',dict(status='COMPLETE_LABEL_FREE_PREPARATION',queries=len(panel),
        bindings=bindings,features_generation=mini.record(directory/'features_generation.json'),
        launch=mini.record(directory/'launch.json'),stage_seconds=times,commune_work=work,
        commune_counts_ref=mini.record(mini.bano.cohort.OUT/'commune_counts.json'),elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,raw_labels_read=False,new_fitting=False))


def generate(directory,binding_ref,prepared_root):
    from scripts import embeddinggemma2_v142_bano_native_ablation as v142
    from scripts.retrieval_miss_score_explain import preparer_kernel
    started=time.monotonic();binding=mini.read(mini.verified(binding_ref))
    q=binding['query'];qid=q['query_id']
    if selected([qid])!=[q]:raise ValueError('Exact query binding to the sealed cohort')
    assignment=mini.read(mini.bano.cohort.OUT/'assignment.json')
    if assignment[qid]!=binding['model_id']:raise ValueError('Historical coefficient group drift')
    launch=mini.read(prepared_root/'launch.json');fg=mini.read(prepared_root/'features_generation.json')
    cal=mini.read(mini.bano.cohort.preflight.RUN/'calibration/generation.json')
    model_ref=cal['models'][str(assignment[qid])];model=mini.read(mini.verified(model_ref))
    if qid in model['train_queries']:raise ValueError('No training query may be newly served')
    # Company exclusion/assignment was proven at the committed cohort seal.
    # Do not edit heldout_queries to pretend the new IDs occurred in old models.
    v115=v142.baseline;v99=v115.baseline;v81=v99.source;base=v81.pipeline.base
    v115.bind_runtime(prepared_root);base.authority=lambda out:launch
    source_dir=Path(binding_ref['path']).parent
    read=lambda name:mini.read(source_dir/(name+'.json'))
    v99.CONTEXT={qid:read('context')};v99.INITIAL_DATA={qid:read('initials')}
    v81.INDIRECT=mini.read(mini.verified(fg['indirect_names']))
    for name in ('corporate_names','published_parent_names'):mini.verified(fg[name])
    mini.verified(fg['activity_shards'][qid])
    w=np.asarray(model['coefficients'],dtype=float)
    backend=v142.TantivyBackend(v142.io.INDEX)
    if backend.searcher.num_docs!=82772691:raise ValueError('Unrestricted original national index')
    with np.load(mini.verified(binding['features']),allow_pickle=False) as f:keys=f['keys'];features=f['features']
    all_keys=np.load(mini.verified(launch['keys']),mmap_mode='r',allow_pickle=False)
    xy=np.load(mini.verified(launch['xy']),mmap_mode='r',allow_pickle=False)
    public=np.load(mini.verified(launch['public_mask']),mmap_mode='r',allow_pickle=False)
    bano,bsha=preparer_kernel(v142,'bano');osm,osha=preparer_kernel(v142,'osm')
    points,_=v142.compatible_points(read('bano_context'))
    delta=bano(keys,features,all_keys,xy,public,points,model)
    geom,named=osm(keys,features,all_keys,xy,public,w,{qid:binding['geometry']},
        {qid:read('shapes')},qid,q,read('osm_context'))
    supplier=read('supplier')
    own=v142.upstream.projected(keys,supplier['own']);temporal=v142.upstream.projected(keys,supplier['temporal'])
    continuity=v142.upstream.projected(keys,read('address'))
    scores=np.maximum(delta,np.maximum(np.maximum(geom,named),w[6]*np.maximum(np.maximum(own,temporal),continuity)))
    merged={r['siren_ref']:r['confidence'] for r in read('rna')}
    for r in read('banatic'):merged[r['siren_ref']]=max(merged.get(r['siren_ref'],0.),r['confidence'])
    entries=[dict(siren_ref=k,confidence=v) for k,v in sorted(merged.items())]
    proxy=v99.BackendProxy(backend,v142.parent.EvidenceCollector(backend,q,keys,scores,entries,float(w[6])))
    candidates,cost=v115.native_search(proxy,q,keys,features,w)
    if len(candidates)>100 or len({r['siret'] for r in candidates})!=len(candidates):raise ValueError('Unique absolute native100 cap')
    mini.save(directory/'result.json',dict(status='COMPLETE_NATIVE100',query=q,candidates=candidates,cost=cost,
        binding=binding_ref,model=model_ref,model_id=assignment[qid],raw_labels_read=False,new_fitting=False,
        source_kernels=dict(bano=bsha,osm=osha),elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def audit(directory):
    start=time.monotonic();frozen=mini.read(mini.bano.cohort.preflight.FROZEN);count=0;size=0
    signatures={}
    for ref in frozen['artifacts']:
        mini.guard();checked=mini.record(ref['path'],ref['sha256']);count+=1;size+=ref['size_bytes']
        signatures[ref['path']]=[checked['size_bytes'],checked['mtime_ns'],checked['inode']]
    mini.save(directory/'result.json',dict(status='PASS_FULL_FROZEN_SHA256',files=count,bytes=size,
        elapsed_seconds=time.monotonic()-start,labels_decoded=False,full3000_started=False,signatures=signatures))


def evaluate(directory,seal_ref):
    # Only this separate post-seal worker can open the human sidecar.
    import duckdb
    import pyarrow as pa
    from scripts import retrieval_site_operational_metric as metric
    seal=mini.read(mini.verified(seal_ref));sample=mini.read(mini.bano.cohort.OUT/'sample.json')
    ids=set(sample['selected_query_ids'])
    if set(seal['outputs'])!=ids or len(ids)!=3000:raise ValueError('Complete generation seal required before labels')
    outputs={}
    for qid,item in seal['outputs'].items():
        ex=mini.read(mini.verified(item['receipt']))
        if ex['exit_code'] or not ex['owned_child_exit_confirmed']:raise ValueError('Only exit0 owned outputs')
        outputs[qid]=mini.read(mini.verified(item['result']))
    label_ref=mini.read(control.OUT/'cohort_binding.json')['refs']['label_side_sample.json']
    labels={r['query_id']:r for r in mini.read(mini.verified(label_ref))['rows']}
    import pyarrow.dataset as ds
    ledger_ref=mini.read(mini.ROOT/'docs/step2_baseline_v142.json')['source_refs']['ledger']
    ledger=ds.dataset(mini.verified(ledger_ref))
    official=ledger.to_table(columns=['query_id','fold','human_ground_truth_siret','human_ground_truth_siren'],
        filter=ds.field('fold').isin([2,3,4]) & ds.field('query_id').isin(sorted(ids))).to_pylist()
    if len(official)!=3000:raise ValueError('Closed folds cannot supply missing evaluation labels')
    if any(labels.get(r['query_id'],{}).get(k)!=r[k] for r in official for k in r):
        raise ValueError('Sealed label sidecar differs from official truth; do not repair')
    requested={c['siret'] for v in outputs.values() for c in v['candidates']}
    launch=mini.read(mini.bano.cohort.preflight.RUN/'launch.json')
    study=metric.read(metric.STUDY_RECEIPT)['file']
    con=duckdb.connect();con.execute('SET threads=1');con.execute("SET memory_limit='512MB'")
    con.register('requested',pa.table({'siret':sorted(requested)}))
    columns=['siret','numeroVoieEtablissement','indiceRepetitionEtablissement','dernierNumeroVoieEtablissement',
        'indiceRepetitionDernierNumeroVoieEtablissement','typeVoieEtablissement','libelleVoieEtablissement','codeCommuneEtablissement']
    def load(path,columns):
        cursor=con.execute('SELECT '+','.join('s.'+k for k in columns)+' FROM read_parquet(?) s SEMI JOIN requested r USING(siret)',[str(path)])
        result={}
        while batch:=cursor.fetchmany(4096):
            mini.guard()
            for row in batch:
                d=dict(zip(columns,row));result[d['siret']]=d
        return result
    sites=load(mini.verified(launch['stock_source']),columns)
    points=load(study,['siret','x','y','qualite_xy','epsg','distance_precision']);con.close()
    rows=[]
    for qid,part in outputs.items():
        q=part['query'];label=labels.get(qid);cs=part['candidates']
        valid=bool(label and len(label['human_ground_truth_siret'])==14)
        gt=label['human_ground_truth_siret'] if valid else None
        company=label['human_ground_truth_siren'] if valid else None
        keys=[c['siret'] for c in cs];exact=gt in keys if valid else False
        binding=mini.read(mini.verified(part['binding']));cp=metric.crm_point(binding['geometry'])
        site=False
        for c in cs:
            if not valid or c['siren']!=company:continue
            p=metric.candidate_point(points.get(c['siret']));d,t=metric.distance(cp,p),metric.threshold(cp,p)
            site=site or metric.address_match(q,sites.get(c['siret'])) or (d is not None and t is not None and d<=t)
        rows.append(dict(query_id=qid,fold=q['fold'],exact=exact,siren=any(c['siren']==company for c in cs) if valid else False,
            site=bool(site),site_union=bool(exact or site),exact_rank=keys.index(gt)+1 if exact else None,label_absent_error=not valid))
    mini.save(directory/'result.json',dict(status='COMPLETE_POST_SEAL_EVALUATION',queries=3000,full_encoder_oof=False,
        coefficient_oof_only=True,denominator_filtering=False,missing_labels=sum(r['label_absent_error'] for r in rows),
        counts={k:sum(r[k] for r in rows) for k in ('exact','siren','site','site_union')},
        by_fold={str(f):dict(n=sum(r['fold']==f for r in rows),**{k:sum(r[k] for r in rows if r['fold']==f) for k in ('exact','siren','site_union')}) for f in (2,3,4)},
        per_query=rows,generation_seal=seal_ref))
