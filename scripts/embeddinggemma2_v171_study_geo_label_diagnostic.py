"""Authorized postseal source rank intervals. No retrieval or label changes."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'

import ijson
import numpy as np
import psutil
from scripts import embeddinggemma2_night as io
from scripts.embeddinggemma2_v171_study_geo_source_components import DTYPE, supported

HOME=Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
RECIPE=dict(scope='POSTSEAL_ALL526_AND_FROZEN_CURRENT19_SOURCE_COMPONENT_NOT_RETRIEVAL',
    authorization='ROOT_EXPLICIT_PHASEB_20261014_AFTER_SOURCE_COMPLETION',
    protected_skip='BEFORE_LABEL_SIRET_FIELD_ACCESS_OR_TARGET_ID_LOOKUP',
    ranks='BEST=1+STRICTLY_GREATER_WORST=GREATER_OR_EQUAL_ON_FINITE_ORIGINAL_UNIVERSE_COMPONENT',
    caps=[50,100],all_denominators_retained=True,no_rank_outside_original_numeric_universe=True,
    no_qc_transfer=True,no_coordinate_label_rule_or_model_change=True,no_gain_recall_or_causality_claim=True,
    threads=1,max_rss_bytes=int(2.5*1024**3),max_seconds=600,deadline_unix=1791954368)


def read(p):return json.loads(Path(p).read_text())


def write(p,obj):
    p=Path(p);temporary=p.with_suffix('.pending.json')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,p)


def summarize(rows):
    eligible=[r['interval'] for r in rows if r.get('interval',{}).get('available')]
    result=dict(denominator=len(rows),scope_counts=dict(Counter(r['scope'] for r in rows)),available=len(eligible),absent=len(rows)-len(eligible))
    for cap in (50,100):
        result[f'definitely_le{cap}']=sum(r['worst_rank']<=cap for r in eligible)
        result[f'possibly_le{cap}']=sum(r['best_rank']<=cap for r in eligible)
    for key in ('best_rank','worst_rank','equal','eligible_candidates'):
        result[key+'_median']=float(np.median([r[key] for r in eligible])) if eligible else None
    result['current_xy_missing_with_study_component']=sum(r.get('interval',{}).get('available',False) and not r.get('current_xy_supported',False) for r in rows)
    result['current_xy_supported_with_study_component']=sum(r.get('interval',{}).get('available',False) and r.get('current_xy_supported',False) for r in rows)
    for current_supported in (False,True):
        selected=[r for r in rows if r.get('interval',{}).get('available') and r.get('current_xy_supported') is current_supported]
        name='missing_current_xy' if not current_supported else 'supported_current_xy'
        result[name+'_component']=dict(available=len(selected),definitely_le50=sum(r['interval']['worst_rank']<=50 for r in selected),
            definitely_le100=sum(r['interval']['worst_rank']<=100 for r in selected),possibly_le100=sum(r['interval']['best_rank']<=100 for r in selected))
    return result


def main(out):
    started=time.monotonic();peak=0;process=psutil.Process()
    def guard():
        nonlocal peak
        rss=process.memory_info().rss;peak=max(peak,rss);free,pressure=io.pressure()
        if time.time()>=1791954368 or time.monotonic()-started>600 or rss>RECIPE['max_rss_bytes'] or free<25 or pressure not in (0,1,2) or psutil.disk_usage(HOME).free<40*1024**3:
            raise RuntimeError('STOP label-side resource guard; preserve incomplete diagnostic')
        return dict(peak_guarded_rss_bytes=peak,elapsed_seconds=time.monotonic()-started,free_percent=free,pressure_level=pressure)
    def record(p,expected=None):
        p=Path(p);before=p.stat();h=hashlib.sha256();blocks=0
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):
                h.update(block);blocks+=1
                if blocks%16==0:guard()
        after=p.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Input changed while hashing')
        if expected and h.hexdigest()!=expected:raise ValueError('Sealed binding SHA mismatch')
        return dict(path=str(p),sha256=h.hexdigest(),size_bytes=after.st_size,mtime_ns=after.st_mtime_ns,inode=after.st_ino)
    def verify(ref):return record(ref['path'],ref['sha256'])
    root=out/'label_side'
    if root.exists():raise ValueError('Fresh immutable label-side namespace required')
    root.mkdir()
    try:
        guard();completion=read(out/'source_completion.json')
        sm=read(verify(completion['source_manifest'])['path']);ex=read(verify(completion['owned_exit'])['path'])
        if ex['exit_code']!=0 or not ex['own_tool_exit_confirmed']:raise ValueError('Source real owned0 mandatory')
        sr=read(verify(sm['source_report'])['path']);sl=read(verify(sr['launch'])['path']);verify(sr['recipe'])
        lm=read(verify(sm['lookup_manifest'])['path'])
        if sr['status']!='COMPLETE_ALL526_SEPARATE_STUDY_SOURCE_COMPONENTS_NO_LABELS' or len(sr['queries'])!=526 or sr['rows']!=128313036:
            raise ValueError('Complete original526 source seal mandatory')
        for ref in lm['artifacts'].values():verify(ref)
        numeric_verified={}
        for n,q in enumerate(sr['queries']):
            guard();verify(q['component']);numeric_verified[q['query_id']]=verify(q['original_numeric_ref'])
            if (n+1)%100==0:print('PRELABEL_BINDINGS',n+1,flush=True)
        for name in ('keys','xy','public_mask'):verify(sl['inputs'][name])
        # All component/original-numeric bindings and observed source exit are verified above.
        nr=read(HOME/'v101/postseal_name_roles_20261013_a/report.json')
        st=read(HOME/'v103/raw_alias_coverage_20261013_a/report.json')
        previous=read(HOME/'v157/geography_component_feasibility_20261014_night/label_side/report.json')
        bindings=dict(labels=nr['cases'],statuses=st['statuses'],previous_all526=previous['cases'])
        for ref in bindings.values():verify(ref)
        statuses={r['qid']:r for r in read(bindings['statuses']['path'])}
        old={r['query_id']:r for r in read(bindings['previous_all526']['path'])}
        scope={q['query_id'] for q in sr['queries']}
        if set(statuses)!=scope or set(old)!=scope or sum(not r['current_exact'] for r in old.values())!=19:
            raise ValueError('Exact prior ALL526/current19 sidecar scope')
        labels={};seen=set()
        with Path(bindings['labels']['path']).open('rb') as stream:
            for r in ijson.items(stream,'item'):
                qid=r['query_id']
                if qid in seen or qid not in scope:raise ValueError('Exact unique all526 label-side authority')
                seen.add(qid);s=statuses[qid]
                if s['site_diffusion']!='O' or s['owner_diffusion']!='O':continue
                value=r['label_siret']
                labels[qid]=np.uint64(int(value)) if isinstance(value,str) and len(value)==14 and value.isdigit() else None
        if seen!=scope:raise ValueError('Every human query remains in label-side denominator')
        write(root/'recipe.json',RECIPE)
        write(root/'launch.json',dict(source_completion=record(out/'source_completion.json'),source_manifest=record(out/'source_manifest.json'),
            labels_only_after_complete_bindings=True,bindings=bindings,recipe=record(root/'recipe.json'),implementation=record(Path(__file__))))
        allkeys=np.load(sl['inputs']['keys']['path'],mmap_mode='r');current=np.load(sl['inputs']['xy']['path'],mmap_mode='r')
        public=np.load(sl['inputs']['public_mask']['path'],mmap_mode='r');qflags=np.load(lm['artifacts']['qflags.npy']['path'],mmap_mode='r')
        precision=np.load(lm['artifacts']['distance_precision.npy']['path'],mmap_mode='r')
        rows=[]
        for q in sr['queries']:
            guard();qid=q['query_id'];s=statuses[qid]
            row=dict(query_id=qid,fold=q['fold'],current_exact=bool(old[qid]['current_exact']),
                source_semantic_date='2026-07-01',source_publication_date='2026-07-21',source_point_last_measured_date='UNKNOWN',
                query_geometry_available=q['query_geometry_available'],query_ban_declared_weak=q['query_ban_declared_weak'],query_ban_uncertainty_m=q['query_ban_uncertainty_m'])
            rows.append(row)
            if s['site_diffusion']!='O' or s['owner_diffusion']!='O':row['scope']='PROTECTED_SKIP_BEFORE_TARGET_ID_LOOKUP';continue
            target=labels[qid]
            if target is None:row['scope']='HUMAN_TARGET_UNAVAILABLE';continue
            at=int(np.searchsorted(allkeys,target))
            if at>=len(allkeys) or allkeys[at]!=target:row['scope']='CANONICAL_TARGET_ABSENT';continue
            if not public[at]:raise ValueError('Public status versus frozen eO/uOmask mismatch')
            flag=int(qflags[at]);row['source_present']=bool(flag&128);row['source_epsg2154']=bool(flag&64);row['source_raw_qc']=flag&63
            row['current_xy_supported']=bool(supported(np.asarray(current[at],dtype=float).reshape(1,2))[0])
            dp=float(precision[at]);row['source_distance_precision_m']=dp if np.isfinite(dp) else None
            nrref=numeric_verified[qid];stat=Path(nrref['path']).stat()
            if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=(nrref['size_bytes'],nrref['mtime_ns'],nrref['inode']):raise ValueError('Original numeric source changed after prelabel hash')
            with np.load(nrref['path'],allow_pickle=False) as z:keys=z['keys']
            if len(keys)!=q['rows'] or hashlib.sha256(keys.tobytes()).hexdigest()!=q['ordered_keys_sha256']:raise ValueError('Original ordered universe binding')
            index=int(np.searchsorted(keys,target));inside=index<len(keys) and keys[index]==target;del keys
            row['target_in_original_numeric_universe']=inside
            if not inside:row['scope']='NOT_IN_ORIGINAL_QUERY_NUMERIC_UNIVERSE';continue
            packed=np.load(q['component']['path'],mmap_mode='r')
            if packed.dtype!=DTYPE or len(packed)!=q['rows']:raise ValueError('Exact source component representation')
            flags=int(packed['flags'][index]);value=float(packed['study_raw'][index])
            if not flags&4:row['scope']='STUDY_OBSERVATION_ABSENT'
            elif not flags&8:row['scope']='STUDY_OTHER_OR_MISSING_CRS'
            elif not flags&16:row['scope']='STUDY_XY_MISSING_OR_UNSUPPORTED'
            elif not flags&32:row['scope']='RAW_QC_NOT_NUMBER_IDENTIFIED'
            elif not flags&64:row['scope']='QUERY_GEOMETRY_UNAVAILABLE'
            else:row['scope']='AVAILABLE_SEPARATE_STUDY_COMPONENT'
            values=packed['study_raw'];finite=np.isfinite(values);eligible=int(np.count_nonzero(finite))
            if np.isfinite(value):
                greater=int(np.count_nonzero(values>value));equal=int(np.count_nonzero(values==value))
                row['interval']=dict(available=True,value=value,eligible_candidates=eligible,strictly_greater=greater,equal=equal,best_rank=greater+1,worst_rank=greater+equal)
            else:row['interval']=dict(available=False,eligible_candidates=eligible)
            packed._mmap.close();del packed,values,finite;gc.collect()
        for a in (allkeys,current,public,qflags,precision):a._mmap.close()
        current19=[r for r in rows if not r['current_exact']]
        if len(rows)!=526 or len(current19)!=19:raise ValueError('Unchanged ALL526/current19 denominators')
        summary=dict(all526=summarize(rows),current19=summarize(current19))
        write(root/'all526.json',rows)
        report=dict(status='COMPLETE_SEPARATE_STUDY_COMPONENT_RANK_INTERVALS_NOT_RETRIEVAL',summary=summary,cases=record(root/'all526.json'),
            source_completion=record(out/'source_completion.json'),launch=record(root/'launch.json'),
            source_age_not_point_measurement_date=True,QC11_21_not_certified_precision=True,
            exact_original_query_universe_only=True,no_quality_transfer_coordinate_label_or_model_change=True,
            no_gain_recall_causal_national_SLO_or_ready_claim=True,**guard())
        write(root/'report.json',report)
        write(root/'manifest.json',dict(status=report['status'],report=record(root/'report.json'),cases=report['cases'],labels_only_after_source_owned0=True))
        print(json.dumps(summary),flush=True)
    except BaseException as exc:
        write(root/'stop.json',dict(status='STOP_INCOMPLETE_LABEL_DIAGNOSTIC',reason=str(exc),elapsed_seconds=time.monotonic()-started,peak_guarded_rss_bytes=peak));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    main(p.parse_args().source)
