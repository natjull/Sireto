"""Post-seal audit of uncapped, correctly bound official names/addresses.

Truth is diagnostic only. This code emits no candidates/models/recall and
cannot be used as a production source or label-dependent index overlay.
"""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from src.xgb_matcher.retrieval_proof_v56 import signal
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json,sql_path
CONFIG=ROOT/'config/retrieval_v58_raw_proofs.json'

def eligible(proof,target,phase):
    if proof['siren']!=target[:9]: return False
    if proof['subject_kind']=='SIRET': return proof['siret']==target
    # Legal-unit names establish identity, NOT a site. An unbound parent
    # address cannot establish target-site address correspondence.
    if phase=='addresses' and proof['subject_kind']=='SIREN': return proof.get('resolved_siret')==target
    return phase=='names' and proof['subject_kind']=='SIREN' and not proof['siret']

def compare(text,proofs,target,phase):
    text=signal(text); terms=set(v38.tokens(text)); scored=[]
    for p in proofs:
        if not eligible(p,target,phase): continue
        value=signal(p['normalized_value']); overlap=len(terms&set(v38.tokens(value)))/len(terms) if terms else 0.
        scored.append((p,bool(text and value==text),overlap))
    return dict(bound_observations=len(scored),distinct_values=len({p['normalized_value'] for p,_,_ in scored}),
        exact_any_source=any(e for _,e,_ in scored),maximum_single_alias_word_coverage=max((v for _,_,v in scored),default=0.),
        exact_sources=sorted({p['source'] for p,e,_ in scored if e}),
        exact_name_kinds=sorted({p.get('name_kind','') for p,e,_ in scored if e}),
        exact_subject_kinds=sorted({p['subject_kind'] for p,e,_ in scored if e}),
        date_reference_unknown=True,not_a_native_score_or_recall=True)

def coherent_pairs(names,addresses,target):
    pairs=[]
    for n in names:
        if not eligible(n,target,'names') or n['subject_kind']!='SIRET' or not n['evidence_id']: continue
        for a in addresses:
            if (eligible(a,target,'addresses') and a['source']==n['source'] and a['evidence_id']==n['evidence_id']
                    and all(a.get(k)==n.get(k) for k in ('source_record_id','valid_from','valid_to','observed_at'))):
                pairs.append(dict(name=n['normalized_value'],address=a['normalized_value'],source=n['source'],
                    evidence_id=n['evidence_id'],source_record_id=n['source_record_id'],
                    valid_from=n['valid_from'],valid_to=n['valid_to'],observed_at=n['observed_at'],
                    temporal_admissibility_unknown=True))
    return pairs

def context_worker(output):
    import numpy as np
    from scripts.retrieval_v56_exact_fields import compatible
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch); cfg=launch['config']; data=launch['rows']
    manifest=json.loads(verify(cfg['dossier_manifest']).read_text()); root=Path(cfg['dossier_manifest']['path']).parent; sources={}
    with Budget(cfg['resources']) as budget:
        connection=duckdb.connect(); connection.execute("SET threads=2"); connection.execute("SET memory_limit='1GB'")
        connection.register('wanted',pa.Table.from_pylist([dict(siren=r['truth'][:9],siret=r['truth']) for r in data]))
        context={}
        for phase,key in (('legal_units','units'),('establishments','sites'),('relations','relations')):
            name=cfg['tables'][key]; ref=dict(path=str(root/name),sha256=manifest['files'][name]['sha256'])
            p=verify(ref); sources[key]=ref
            if key=='units': where="siren IN (SELECT DISTINCT siren FROM wanted)"
            elif key=='sites': where="siret IN (SELECT DISTINCT siret FROM wanted)"
            else: where="""(from_kind='SIREN' AND from_identifier IN (SELECT DISTINCT siren FROM wanted))
                OR (to_kind='SIREN' AND to_identifier IN (SELECT DISTINCT siren FROM wanted))
                OR (from_kind='SIRET' AND from_identifier IN (SELECT DISTINCT siret FROM wanted))
                OR (to_kind='SIRET' AND to_identifier IN (SELECT DISTINCT siret FROM wanted))"""
            table=connection.execute(f"SELECT * FROM read_parquet({sql_path(p)}) WHERE {where}").fetch_arrow_table()
            rows=table.to_pylist()
            context[phase]=[{k:(v.isoformat() if hasattr(v,'isoformat') else v) for k,v in r.items()} for r in rows]
        connection.close()
        gm=json.loads(verify(cfg['query_geometry']).read_text()); gl=json.loads(verify(gm['launch']).read_text())
        if gm['status']!='SEALED_INITIAL_TOP100' or gl['queries']!=[{k:r[k] for k in v38.QUERY_FIELDS} for r in data]:
            raise ValueError('Query-only geometry population drift')
        geos={g['query_id']:g for g in json.loads(verify(gl['geo_queries']).read_text())}
        sm=json.loads(verify(cfg['spatial_index']).read_text())
        if sm['status']!='SEALED_OFFICIAL_SPATIAL_INDEX' or sm['truth_opened'] or sm['crm_opened']: raise ValueError('Official spatial authority drift')
        xy=np.load(verify(sm['files']['xy.npy']),mmap_mode='r',allow_pickle=False)
        keys=np.load(verify(sm['files']['keys.npy']),mmap_mode='r',allow_pickle=False)
        if len(keys)!=sm['compatible_coordinate_pairs'] or xy.shape!=(len(keys),2): raise ValueError('Spatial alignment drift')
        contexts=[]
        for r in data:
            g=compatible(geos[r['query_id']],r); pos=int(np.searchsorted(keys,np.uint64(int(r['truth']))))
            present=bool(pos<len(keys) and keys[pos]==int(r['truth']))
            distance=float(np.linalg.norm(xy[pos]-[g['ban_x_l93'],g['ban_y_l93']])) if present and g['spatial_crs_compatible'] else None
            relations=[p for p in context['relations'] if (p['from_kind']=='SIREN' and p['from_identifier']==r['truth'][:9])
                or (p['to_kind']=='SIREN' and p['to_identifier']==r['truth'][:9])
                or (p['from_kind']=='SIRET' and p['from_identifier']==r['truth'])
                or (p['to_kind']=='SIRET' and p['to_identifier']==r['truth'])]
            contexts.append(dict(query_id=r['query_id'],fold=r['fold'],exact_hit=r['exact_hit'],
                legal_unit=[p for p in context['legal_units'] if p['siren']==r['truth'][:9]],
                site=[p for p in context['establishments'] if p['siret']==r['truth']],
                relations=relations,ban=g,official_coordinate_present=present,distance_to_ban_m=distance))
    v38.verify_launch(launch)
    write_json(output/'context.json',dict(rows=contexts,sources=sources,spatial_sources={k:sm['files'][k] for k in ('xy.npy','keys.npy')},
        geometry_source=gl['geo_queries'],resources=budget.report(),recall_measured=False,
        accounts_officers_free_text_excluded=True,scope='POST_SEAL_OFFICIAL_CONTEXT_ONLY_NOT_MODEL_FEATURES'))

def worker(output,phase):
    if phase=='context': return context_worker(output)
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    cfg=launch['config']; data=launch['rows']; name=cfg['tables'][phase]
    manifest=json.loads(verify(cfg['dossier_manifest']).read_text()); root=Path(cfg['dossier_manifest']['path']).parent
    reference=dict(path=str(root/name),sha256=manifest['files'][name]['sha256'])
    with Budget(cfg['resources']) as budget:
        path=verify(reference); before=(path.stat().st_size,path.stat().st_mtime_ns)
        connection=duckdb.connect(); connection.execute("SET threads=2")
        connection.execute("SET memory_limit='1GB'"); connection.execute("SET preserve_insertion_order=false")
        connection.register('wanted',pa.Table.from_pylist([dict(siren=r['truth'][:9],siret=r['truth']) for r in data]))
        cols=['siren','siret','subject_kind','source','normalized_value','evidence_id','source_record_id','is_current','source_priority']
        if phase=='names': cols.append('name_kind')
        else: cols.extend(['insee','postcode','street_number','street_number_suffix'])
        datecols=['valid_from','valid_to','observed_at']
        selected=','.join(['p.'+c for c in cols]+['cast(p.'+c+' as varchar) as '+c for c in datecols])
        sql=f"""SELECT {selected} FROM read_parquet({sql_path(path)}) p
            WHERE p.siren IN (SELECT DISTINCT siren FROM wanted)
            AND (p.subject_kind='SIREN' OR p.siret IN (SELECT DISTINCT siret FROM wanted))
            ORDER BY p.siren,p.siret,p.source,p.evidence_id,p.normalized_value"""
        proofs=connection.execute(sql).fetch_arrow_table().to_pylist(); resolution_ref=None
        if phase=='addresses':
            rn='address_site_resolution.parquet'; resolution_ref=dict(path=str(root/rn),sha256=manifest['files'][rn]['sha256'])
            rp=verify(resolution_ref)
            resolutions=connection.execute(f"""SELECT siren,source,normalized_value,postcode,insee,
                cast(valid_from as varchar) valid_from,cast(valid_to as varchar) valid_to,
                candidate_siret_count,resolved_siret,resolution_status
                FROM read_parquet({sql_path(rp)}) WHERE siren IN (SELECT DISTINCT siren FROM wanted)""").fetch_arrow_table().to_pylist()
            group=defaultdict(list); fields=('siren','source','normalized_value','postcode','insee','valid_from','valid_to')
            for r in resolutions: group[tuple(r[k] for k in fields)].append(r)
            for p in proofs:
                p['resolved_siret']=''
                if p['subject_kind']=='SIREN':
                    matching=group.get(tuple(p[k] for k in fields),[])
                    unique={r['resolved_siret'] for r in matching if r['candidate_siret_count']==1 and r['resolved_siret']}
                    if len(unique)==1 and all(r['candidate_siret_count']==1 for r in matching):
                        p['resolved_siret']=next(iter(unique))
        connection.close()
        grouped=defaultdict(list)
        for p in proofs: grouped[p['siren']].append(p)
        summaries=[]
        for r in data:
            q=v42.as_query(r); text=q.name if phase=='names' else q.address
            summaries.append(dict(query_id=r['query_id'],fold=r['fold'],exact_hit=r['exact_hit'],
                **compare(text,grouped[r['truth'][:9]],r['truth'],phase)))
        if before!=(path.stat().st_size,path.stat().st_mtime_ns): raise ValueError('Raw source changed during scan')
    v38.verify_launch(launch)
    write_json(output/(phase+'.json'),dict(phase=phase,source=reference,source_sha256_verified=True,
        source_size_bytes=before[0],summaries=summaries,proofs=proofs,resources=budget.report(),
        resolution_source=resolution_ref,excluded_unbound_parent_addresses=phase=='addresses',
        query_truth_scope='POST_SEAL_DIAGNOSTIC_ONLY_NEVER_GENERATION',recall_measured=False))

def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(CONFIG.read_text()); began=time.monotonic()
    try:
        source=json.loads(verify(cfg['generation']).read_text())
        evaluation=json.loads(verify(cfg['evaluation']).read_text()); audit=json.loads(verify(cfg['native_audit']).read_text())
        if source['status']!='SEALED_INITIAL_TOP100' or source['queries']!=128 or source['truth_opened']:
            raise ValueError('Complete blind generation required')
        if evaluation['generation']['sha256']!=cfg['generation']['sha256'] or audit['evaluation']['sha256']!=cfg['evaluation']['sha256']:
            raise ValueError('Post-seal audit authority drift')
        launch=json.loads(verify(source['launch']).read_text()); v38.verify_launch(launch)
        for ref in source['batches']: verify(ref)
        ids={r['query_id'] for r in launch['queries']}; outcomes={r['query_id']:r for r in evaluation['outcomes']}
        labels={r['query_id']:r for r in pq.read_table(verify(evaluation['ledger']),filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
        if len(ids)!=128 or set(labels)!=ids or set(outcomes)!=ids: raise ValueError('All-human audit denominator drift')
        rows=[]
        for q in launch['queries']:
            label=labels[q['query_id']]
            if label['fold']!=q['fold'] or q['fold'] not in (2,3,4): raise ValueError('Protected fold drift')
            rows.append(dict(q,truth=label['human_ground_truth_siret'],exact_hit=outcomes[q['query_id']]['primary_exact']))
        implementation=dict(launch['implementation'])
        for p in (Path(__file__),CONFIG): implementation[str(p)]=record(p)
        outlaunch=dict(implementation=implementation,index_manifest=launch['index_manifest'],index_meta=launch['index_meta'],
            inventory=launch['inventory'],rows=rows,config=cfg,labels_scope='POST_EVALUATION_DIAGNOSTIC_ONLY')
        write_json(output/'launch.json',outlaunch)
        for phase in ('names','addresses','context'):
            with (output/(phase+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.audit_retrieval_v58_raw_proofs','worker','--phase',phase,'--output',str(output)],
                    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=min(90,max(1,cfg['controller_seconds']-(time.monotonic()-began))),
                    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})
            part=json.loads((output/(phase+'.json')).read_text())
            if len(part['rows'] if phase=='context' else part['summaries'])!=128 or not part['resources']['pass_resource_policy']:
                raise ValueError('Raw audit scope/resource drift')
            print(json.dumps(dict(phase=phase,observations=len(part['rows'] if phase=='context' else part['proofs']),resources=part['resources'])),flush=True)
        names=json.loads((output/'names.json').read_text()); addresses=json.loads((output/'addresses.json').read_text())
        native={r['query_id']:r for r in audit['rows']}; missing=[]; paired=[]
        name_groups=defaultdict(list); address_groups=defaultdict(list)
        for p in names['proofs']: name_groups[p['siren']].append(p)
        for p in addresses['proofs']: address_groups[p['siren']].append(p)
        for r in rows:
            pairs=coherent_pairs(name_groups[r['truth'][:9]],address_groups[r['truth'][:9]],r['truth'])
            paired.append(dict(query_id=r['query_id'],fold=r['fold'],pairs=pairs))
        amap={r['query_id']:r for r in addresses['summaries']}
        for n in names['summaries']:
            if n['exact_hit']: continue
            nr=native[n['query_id']]; a=amap[n['query_id']]
            missing.append(dict(query_id=n['query_id'],fold=n['fold'],
                native_no_name_proof=nr['no_name_proof'],native_name_coverage=nr['name_coverage'],
                raw_name=n,raw_site_address=a,
                raw_exact_name_not_in_current_exact=bool(n['exact_any_source'] and not nr['current_exact_name']),
                raw_word_name_evidence_when_native_none=bool(n['maximum_single_alias_word_coverage']>0 and nr['no_name_proof'])))
        v38.verify_launch(outlaunch)
        write_json(output/'manifest.json',dict(status='SEALED_POST_EVALUATION_RAW_OFFICIAL_PROOF_AUDIT',queries=128,misses=len(missing),
            missing=missing,coherent_observations=paired,phases={phase:record(output/(phase+'.json')) for phase in ('names','addresses','context')},
            launch=record(output/'launch.json'),seconds=time.monotonic()-began,
            raw_exact_name_missing_current_exact=sum(r['raw_exact_name_not_in_current_exact'] for r in missing),
            raw_word_name_when_native_none=sum(r['raw_word_name_evidence_when_native_none'] for r in missing),
            fold0_opened=False,fold1_opened=False,ready_for_fold1=False,recall_measured=False,model_changed=False,
            no_production_overlay_allowed_from_this_label_selected_audit=True,
            limitations=['SUBJECT_BOUND_RAW_ROWS_ONLY_NOT_CROSS_SIREN_RELATION_PATHS','UNKNOWN_QUERY_REFERENCE_DATE_NO_TEMPORAL_ADMISSIBILITY_CLAIM',
                'RAW_WORD_COVERAGE_NOT_NATIVE_CHARGRAM_SCORE','EXACT_RAW_NAME_NOT_AUTOMATIC_ADMISSIBLE_SITE_PROOF']))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,recall_measured=False)); raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--phase',choices=['names','addresses','context']); a=p.parse_args()
    if a.mode=='run': run(a.output)
    else:
        if a.phase is None: p.error('--phase required')
        worker(a.output,a.phase)
