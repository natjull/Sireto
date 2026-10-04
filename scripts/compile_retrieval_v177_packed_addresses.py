"""Exact complete address CSR from sealed V175, compact source lifetime."""
import argparse
import json
import os
from pathlib import Path
import sys

import duckdb
import numpy as np

import scripts.seal_retrieval_v175_address_parts as prior
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget,quiet,LIMITS,POLICY,validate_report
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_packed_names_v134 import pack
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v177_packed_addresses.json'
STATUS='SEALED_V177_ALL_OFFICIAL_ADDRESS_CHANNEL_ACCESS'
FLAGS={k:False for k in PROTECTED}


def config():
    cfg=json.loads(CONFIG.read_text())
    if (cfg['resources']!=LIMITS or cfg['duckdb_version']!=duckdb.__version__ or cfg['numpy_version']!=np.__version__
        or any(type(cfg[k]) is not int or cfg[k]<=0 for k in ('aliases','postings','term_buckets','max_phase_bytes','max_total_bytes'))
        or cfg['term_buckets']!=64):raise ValueError('Exact whole-source typed resource policy required')
    return cfg


def fixed_epochs(items,refs):
    by_path={}
    for e in [*items,*prior.epochs(refs)]:
        key=e['reference']['path']
        if key in by_path and (by_path[key]['reference']['sha256']!=e['reference']['sha256'] or by_path[key]['fingerprint']!=e['fingerprint']):
            raise ValueError('Conflicting original immutable source epochs')
        by_path[key]=e
    result=[by_path[k] for k in sorted(by_path)];prior.check_epochs(result);return result


def complete_source(cfg):
    path=verify(cfg['staging']);root=path.parent;launch,ledger=prior.context(root)
    parent_exit(root/'seal');m=json.loads(path.read_text());validate_report(m['seal_resources'])
    if (path.resolve()!=root/'staging_manifest.json' or m['status']!=prior.STATUS
        or m['launch']!=record(root/'launch.json') or m['aliases']!=cfg['aliases'] or m['postings']!=cfg['postings']
        or not m['every_original_whole_alias_and_posting_retained'] or m['complete_retrieval_index'] or m['ready_for_fold1']
        or any(m.get(k) is not False for k in PROTECTED)):
        raise ValueError('Protected entire V175 stage and supervised seal required')
    sr,sl,p=prior.source(launch['config']);features=[];buckets={str(b):[] for b in range(cfg['term_buckets'])}
    refs=[cfg['staging'],m['launch'],launch['ledger'],launch['preflight'],record(root/'seal/supervision.json'),
        launch['config']['source_launch'],launch['config']['source_plan']]
    parts=[];audits=[];cursor=posts=empty=size=0;scopes=[root,root/'seal']
    count=(launch['config']['parts']+launch['config']['parts_per_batch']-1)//launch['config']['parts_per_batch']
    for i in range(count):
        a=prior.batch_record(root,i);dest=prior.directory(root,i);ref=record(dest/'manifest.json');audits.append(ref)
        refs.extend([ref,parent_exit(dest),a['preflight']]);scopes.append(dest)
        for row in a['rows']:
            spec=p['parts'][row['index']]
            if (row['feature']['part']!=spec or spec['alias_start']+spec['row_start']!=cursor
                or row['aliases']!=spec['row_stop']-spec['row_start']):raise ValueError('Every whole original alias once required')
            features.append(row['feature']);parts.append(row['manifest']);scopes.append(Path(row['manifest']['path']).parent)
            for item in row['posts']:buckets[item['bucket']].append({k:v for k,v in item.items() if k!='bucket'})
            cursor+=row['aliases'];posts+=row['postings'];empty+=row['feature_empty'];size+=row['output_bytes']
    if (features!=m['features'] or buckets!=m['buckets'] or parts!=m['phase_authorities'] or audits!=m['audit_authorities']
        or (cursor,posts,empty,size)!=(m['aliases'],m['postings'],m['feature_empty'],m['output_bytes'])
        or m['source']!=p['source_store']):raise ValueError('Every original staged feature/artifact/count exactly required')
    for kind in ('prefix','reverse'):
        ref,authority=prior.backend.audit_record(sr,'seal',kind)
        if ref!=m['sealed_source_audits'][kind] or parent_exit(Path(ref['path']).parent)!=m['sealed_source_audit_supervisions'][kind]:
            raise ValueError('Complete full-source rehash and actual exit binding required')
        refs.extend([ref,m['sealed_source_audit_supervisions'][kind]]);scopes.append(Path(ref['path']).parent)
        if kind=='reverse' and any(authority[k]!=p[k] for k in ('dictionary','catalog',*prior.COUNT_FIELDS,'parts','source_epochs')):
            raise ValueError('Same full raw source and inverse facts required')
    refs.extend(launch['implementation'].values());refs.extend(sl['implementation'].values())
    for ref in refs:verify(ref)
    prior.backend.revalidate(sr,sl,p)
    all_epochs=fixed_epochs([*p['source_epochs'],*(e for item in ledger['parts'] for e in item['epochs'])],refs)
    return dict(source_store=p['source_store'],dictionary=p['dictionary'],catalog=p['catalog'],
        **{k:p[k] for k in prior.COUNT_FIELDS},postings=posts,buckets=buckets),all_epochs,refs,scopes


def implementation():
    refs=prior.implementation()
    files=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v177_packed_addresses_contract.md',
        ROOT/'src/xgb_matcher/retrieval_packed_names_v134.py',ROOT/'src/xgb_matcher/retrieval_name_access_v131.py']
    refs.update({str(f):record(f) for f in files});return refs


def init(output):
    cfg=config()
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):raise ValueError('CATNAT-only fresh index root required')
    if output.exists():raise FileExistsError('Immutable address access root exists')
    gate=quiet(cfg['quiet_seconds'],50,55);budget=None
    try:
        with Budget(LIMITS) as budget:data,epochs,refs,scopes=complete_source(cfg)
        output.mkdir(parents=True,exist_ok=False);write_json(output/'preflight.json',gate)
        write_json(output/'source_epochs.json',dict(epochs=epochs,refs=refs,stop_scopes=[str(s) for s in scopes],**FLAGS))
        write_json(output/'plan.json',dict(status='SEALED_V177_COMPACT_COMPLETE_ADDRESS_PLAN',staging=cfg['staging'],
            **data,resources=budget.report(),complete_retrieval_index=False,**FLAGS))
        write_json(output/'launch.json',dict(config=cfg,plan=record(output/'plan.json'),epochs=record(output/'source_epochs.json'),
            preflight=record(output/'preflight.json'),implementation=implementation(),**FLAGS))
    except BaseException as exc:
        if not output.exists():output.mkdir(parents=True,exist_ok=False)
        write_json(output/'init_stop.json',dict(status='STOP',reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,**FLAGS));raise


def check_live(l,p,e):
    for scope in e['stop_scopes']:
        if any((Path(scope)/n).exists() for n in ('init_stop.json','seal_stop.json','worker_stop.json','status.json','supervisor_stop.json')):
            raise ValueError('STOP_SOURCE_AUTHORITY_OR_PHASE_DO_NOT_USE')
    prior.check_epochs(e['epochs'])
    for ref in [*e['refs'],*l['implementation'].values(),l['plan'],l['epochs']]:verify(ref)
    if config()!=l['config']:raise ValueError('Pinned access config changed')


def load(output):
    if (output/'init_stop.json').exists() or (output/'seal_stop.json').exists():raise ValueError('STOP_ACCESS_DO_NOT_RESTART')
    l=json.loads((output/'launch.json').read_text());cfg=l['config']
    if cfg!=config() or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()) or any(l.get(k) is not False for k in PROTECTED):
        raise ValueError('Immutable complete-source compact access launch required')
    p=json.loads(verify(l['plan']).read_text());e=json.loads(verify(l['epochs']).read_text());validate_report(p['resources'])
    gate=json.loads(verify(l['preflight']).read_text())
    if (p['status']!='SEALED_V177_COMPACT_COMPLETE_ADDRESS_PLAN' or p['staging']!=cfg['staging']
        or p['aliases']!=cfg['aliases'] or p['postings']!=cfg['postings'] or p['complete_retrieval_index']
        or set(p['buckets'])!={str(b) for b in range(cfg['term_buckets'])}
        or any(any(x.get(k) is not False for k in PROTECTED) for x in (p,e))
        or gate['policy_id']!=POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']):
        raise ValueError('Protected full original source/epoch/preflight authority required')
    check_live(l,p,e);return l,p,e


def request(cfg,channel,bucket):
    if type(channel) is not int or not 0<=channel<4 or type(bucket) is not int or not 0<=bucket<cfg['term_buckets']:
        raise ValueError('Actual whole physical channel/bucket required')


def directory(output,channel,bucket):return output/'access'/f'channel{channel}'/f'bucket{bucket:03d}'


def worker(output,channel,bucket):
    dest=directory(output,channel,bucket);budget=c=None
    if any(f.name not in ('preflight.json','worker.log') for f in dest.iterdir()):raise ValueError('Immutable pack already started, no retry')
    try:
        with Budget(LIMITS) as budget:
            l,p,e=load(output);cfg=l['config'];request(cfg,channel,bucket)
            gate=json.loads((dest/'preflight.json').read_text())
            if gate['policy_id']!=POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
                raise ValueError('Actual authorized pack pressure required')
            items=p['buckets'][str(bucket)];files=[verify(x['artifact']) for x in items]
            expected=sum(x['channel_counts'][str(channel)] for x in items)
            c=duckdb.connect();c.execute('SET threads=1');c.execute('SET memory_limit=?',[cfg['duckdb_memory']])
            c.execute('SET temp_directory=?',[str(dest/'spill')]);c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            stats=pack(c,files,dest,channel,bucket,cfg['term_buckets'],expected,p['aliases'])
            for item in items:verify(item['artifact'])
            c.close();c=None
            artifacts={f.name:record(f) for f in dest.glob('*.npy')};check_live(l,p,e)
            size=sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())
            if size>cfg['max_phase_bytes']:raise ValueError('STOP_PACK_PHASE_DISK_BUDGET')
        write_json(dest/'manifest.json',dict(status='SEALED_V177_EXACT_ADDRESS_PACK',launch=record(output/'launch.json'),
            channel=channel,bucket=bucket,stats=stats,artifacts=artifacts,preflight=record(dest/'preflight.json'),
            output_bytes=size,resources=budget.report(),complete_retrieval_index=False,**FLAGS))
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,**FLAGS));raise
    finally:
        if c is not None:c.close()


def phase_record(output,channel,bucket,context=None):
    l,p,e=load(output) if context is None else context;request(l['config'],channel,bucket);dest=directory(output,channel,bucket)
    if any((dest/n).exists() for n in ('worker_stop.json','status.json','supervisor_stop.json')):raise ValueError('STOP_PACK_DO_NOT_RESTART')
    parent_exit(dest);m=json.loads((dest/'manifest.json').read_text());validate_report(m['resources']);s=m['stats']
    if (m['status']!='SEALED_V177_EXACT_ADDRESS_PACK' or m['launch']!=record(output/'launch.json')
        or (m['channel'],m['bucket'])!=(channel,bucket) or (s['channel'],s['bucket'])!=(channel,bucket)
        or s['postings']!=sum(x['channel_counts'][str(channel)] for x in p['buckets'][str(bucket)])
        or not all(s[k] for k in ('every_original_posting_and_token_reconstructed_exactly','all_postings_sorted_unique',
            'native_unsigned_batched_csr_audit','source_semantics_unchanged','no_posting_or_owner_cap'))
        or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index']
        or m['preflight']!=record(dest/'preflight.json') or set(m['artifacts'])!={'post_ids.npy','post_offsets.npy','token_data.npy','token_offsets.npy'}):
        raise ValueError('Protected exact all-source pack certificate required')
    gate=json.loads(verify(m['preflight']).read_text())
    if gate['policy_id']!=POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
        raise ValueError('Actual authorized pack preflight required')
    for name,ref in m['artifacts'].items():
        if verify(ref).resolve()!=dest/name:raise ValueError('Exact immutable own pack artifact required')
    return m


def execute(output,channel,bucket):
    l,_,_=load(output);request(l['config'],channel,bucket);dest=directory(output,channel,bucket)
    dest.mkdir(parents=True,exist_ok=False)
    try:
        write_json(dest/'preflight.json',quiet(l['config']['phase_quiet_seconds'],50,55))
        with (dest/'worker.log').open('x') as log:
            s=supervise([sys.executable,'-u','-m','scripts.compile_retrieval_v177_packed_addresses','worker','--output',str(output),
                '--channel',str(channel),'--bucket',str(bucket)],log,
                {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},dest/'supervisor_stop.json')
        write_json(dest/'supervision.json',s);phase_record(output,channel,bucket)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**FLAGS));raise


def remaining(output):
    l,p,e=load(output)
    for ch in range(4):
        for b in range(l['config']['term_buckets']):
            if directory(output,ch,b).exists():phase_record(output,ch,b,context=(l,p,e))
            else:execute(output,ch,b)


def seal_worker(output):
    budget=None
    if (output/'manifest.json').exists():raise FileExistsError('Immutable complete access seal exists')
    try:
        with Budget(LIMITS) as budget:
            l,p,e=load(output);access=[];refs=[];posts=terms=size=0
            for ch in range(4):
                for b in range(l['config']['term_buckets']):
                    m=phase_record(output,ch,b,context=(l,p,e));s=m['stats']
                    specs=dict(post_ids=(np.uint32,s['postings']),post_offsets=(np.uint64,s['terms']+1),
                        token_data=(np.uint8,s['token_bytes']),token_offsets=(np.uint64,s['terms']+1))
                    for name,(dtype,n) in specs.items():
                        a=np.load(verify(m['artifacts'][name+'.npy']),mmap_mode='r',allow_pickle=False)
                        try:
                            if a.dtype!=dtype or a.shape!=(n,):raise ValueError('Every exact unsigned packed source array required')
                            if name.endswith('offsets') and (a[0]!=0 or a[-1]!=(s['postings'] if name=='post_offsets' else s['token_bytes']) or np.any(a[1:]<=a[:-1])):
                                raise ValueError('Whole strictly increasing original CSR offsets required')
                        finally:a._mmap.close()
                    access.append(dict(channel=ch,bucket=b,postings=s['postings'],terms=s['terms'],artifacts=m['artifacts']))
                    refs.append(record(directory(output,ch,b)/'manifest.json'));posts+=s['postings'];terms+=s['terms'];size+=m['output_bytes']
            if posts!=p['postings'] or size>l['config']['max_total_bytes']:raise ValueError('Whole address posting/storage denominator required')
            check_live(l,p,e)
        write_json(output/'manifest.json',dict(status=STATUS,launch=record(output/'launch.json'),staging=p['staging'],
            source_store=p['source_store'],dictionary=p['dictionary'],catalog=p['catalog'],aliases=p['aliases'],sites=p['sites'],facts=p['facts'],
            access=access,phase_authorities=refs,postings=posts,terms=terms,output_bytes=size,seal_resources=budget.report(),
            complete_retrieval_index=False,ready_for_fold1=False,**FLAGS))
    except BaseException as exc:
        write_json(output/'seal_stop.json',dict(status='STOP',reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,**FLAGS));raise


def seal(output):
    l,_,_=load(output);dest=output/'seal'
    if dest.exists() or (output/'manifest.json').exists():raise FileExistsError('Immutable access seal started')
    dest.mkdir(parents=True,exist_ok=False)
    try:
        write_json(dest/'preflight.json',quiet(l['config']['phase_quiet_seconds'],50,55))
        with (dest/'worker.log').open('x') as log:
            s=supervise([sys.executable,'-u','-m','scripts.compile_retrieval_v177_packed_addresses','seal-worker','--output',str(output)],
                log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},dest/'supervisor_stop.json')
        write_json(dest/'supervision.json',s);parent_exit(dest)
        m=json.loads((output/'manifest.json').read_text());validate_report(m['seal_resources'])
        if m['status']!=STATUS or m['launch']!=record(output/'launch.json'):raise ValueError('Exact supervised complete address access required')
    except BaseException as exc:write_json(output/'seal_stop.json',dict(status='STOP',reason=str(exc),**FLAGS));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','phase','worker','remaining','seal','seal-worker'])
    p.add_argument('--output',required=True,type=Path);p.add_argument('--channel',type=int);p.add_argument('--bucket',type=int);a=p.parse_args()
    if a.mode in ('phase','worker'):
        if a.channel is None or a.bucket is None:p.error('Actual channel and bucket required')
        (execute if a.mode=='phase' else worker)(a.output,a.channel,a.bucket)
    else:{'init':init,'remaining':remaining,'seal':seal,'seal-worker':seal_worker}[a.mode](a.output)
