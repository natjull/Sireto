"""Explicit compatibility-gated physical resume; never reads quality labels."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import retrieval_v33_full_replay as v33
from scripts.retrieval_v34_runtime import FastRuntime
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json

OLD = v33.DATA/'audits/retrieval_v33_full_replay/fast'
OLD_PREPARED_SHA = '765635a3b751f02e1e63b4d7df0af1253abc25239b5b4a36de4621f3856136f8'
COMMON = ('inputs.parquet','queries.parquet','raw.parquet','ban.parquet','address.parquet',
          'parent_signals.parquet','geo_sites.parquet','geo_parents.parquet','parent_candidates.parquet',
          'parent_presence.parquet','edges.parquet','proof.parquet','structured.parquet','direct.parquet',
          'joint.parquet','v23_stable/master.parquet','v30_stable/master.parquet',
          'v23_stable/pool.parquet','v30_stable/pool.parquet','v23_stable/parents.parquet','v30_stable/parents.parquet')


def accelerator_code():
    return {n:record(v33.ROOT/n) for n in ('scripts/retrieval_v34_runtime.py',
      'scripts/retrieval_v34_parallel_legacy.py',
      'scripts/retrieval_v34_resume.py','docs/retrieval_v34_physical_resume_contract.md')}


def old_batches():
    result=[]
    for start in range(0,704,32):
        path=OLD/f'full_job/generation/batch{start:05d}/manifest.json'
        m=json.loads(path.read_text())
        if m['status']!='SEALED_PAIRED_BATCH' or m['start']!=start or m['count']!=32 or m['parity']:
            raise ValueError('Old batch scope')
        shards=[json.loads(verify(r).read_text()) for r in m['shards']]
        for s in shards:
            for r in s['files'].values():verify(r)
        result.append((path,m,shards))
    return result


def compatibility(prepared):
    new=v33.authority(prepared)
    old=json.loads(verify(dict(path=OLD/'prepared/manifest.json',sha256=OLD_PREPARED_SHA)).read_text())
    for key in ('sources','stats','indexes','edges','population','stability','queries'):
        if new[key]!=old[key]:raise ValueError('Changed source authority: '+key)
    if set(old['code'])!=set(new['code']):raise ValueError('Code inventory drift')
    changed=[k for k in old['code'] if old['code'][k]!=new['code'][k]]
    if changed!=['src/xgb_matcher/retrieval_geo_v16.py']:raise ValueError('Unexpected business-code changes')
    c=duckdb.connect()
    for name in ('inputs','scope'):v33.same(c,str(verify(old[name])),str(verify(new[name])))
    c.close();return old,new


def geo_surface(rows):
    out=[]
    for r in rows:
        q=v33.normalize_input(r)
        out.append(dict(query_id=r['query_id'],fold=r['oof_fold'],crm_number=q.number,
          crm_street_norm=v33.normalize_street(q.address,q.number),crm_insee=q.insee,crm_postcode=q.postcode))
    return pa.Table.from_pylist(out)


def ban_audit(prepared,output):
    old,m=compatibility(prepared);rows=pq.read_table(verify(m['inputs'])).to_pylist()
    output.mkdir(parents=True,exist_ok=False)
    with Budget(v33.RESOURCE) as b:
        c=v33.open_workbench_db(output);c.execute('SET threads=2')
        for start in range(0,len(rows),256):
            result=v33._resolve_ban(c,geo_surface(rows[start:start+256]),Path(m['sources']['ban']['path']))
            pq.write_table(result,output/f'ban{start:05d}.parquet',compression='zstd')
            print('BAN',min(start+256,len(rows)),len(rows),flush=True)
        parts=[str(p) for p in sorted(output.glob('ban*.parquet'))]
        old_files=[]
        for _,bm,ss in old_batches():
            if bm['code']!=old['code']:raise ValueError('Old batch code drift')
            old_files+=v33.files_for(ss,'ban.parquet')
        c.execute('CREATE TABLE old_ban AS SELECT * FROM read_parquet(?)',[old_files])
        c.execute('CREATE TABLE new_ban AS SELECT * FROM read_parquet(?)',[parts])
        if c.execute('SELECT count(*) FROM old_ban').fetchone()[0]!=704:raise ValueError('Adoption BAN scope')
        if c.execute('SELECT count(*) FROM new_ban').fetchone()[0]!=23571:raise ValueError('BAN sweep scope')
        mismatch=c.execute('''SELECT count(*) FROM ((SELECT * FROM old_ban EXCEPT ALL
          SELECT n.* FROM new_ban n SEMI JOIN old_ban o USING(query_id,fold)) UNION ALL
          (SELECT n.* FROM new_ban n SEMI JOIN old_ban o USING(query_id,fold) EXCEPT ALL SELECT * FROM old_ban))''').fetchone()[0]
        if mismatch:raise ValueError('BAN adoption mismatch')
        c.close();v33.authority(prepared)
    write_json(output/'manifest.json',dict(status='PASS_BAN_FULL_SWEEP_AND_704_PARITY',queries=23571,
      preserved_queries=704,prepared=record(prepared/'manifest.json'),old_prepared=record(OLD/'prepared/manifest.json'),
      accelerator_code=accelerator_code(),resources=b.report(),truth_columns_opened=False,
      files=[record(Path(p)) for p in parts]))


def batch(prepared,output,start,count,subshard,reverse):
    frozen=accelerator_code();runtime=FastRuntime();runtime.install()
    try:v33.batch(prepared,output,start,count,subshard,reverse)
    finally:runtime.close()
    if accelerator_code()!=frozen:raise ValueError('Accelerator drift')
    p=output/'manifest.json';m=json.loads(p.read_text());m['accelerator_code']=frozen
    physical=sorted(output.glob('shard*/parallel_legacy.json'))
    m['physical_artifacts']=[record(p) for p in physical]
    m['resources']['peak_aggregate_rss_bytes']=max([m['resources']['peak_rss_bytes']]+
      [json.loads(p.read_text())['peak_aggregate_rss_bytes'] for p in physical])
    if m['resources']['peak_aggregate_rss_bytes']>v33.RESOURCE['max_rss_bytes']:raise ValueError('Aggregate RSS guard')
    write_json(p,m)


def checked_batch(path,prepared,start,reverse=False):
    m,s=v33.manifests(path)
    if m['start']!=start or m['count']!=32 or m['reverse']!=reverse or m['parity'] or m['prepared']!=record(prepared/'manifest.json') or m.get('accelerator_code')!=accelerator_code():
        raise ValueError('Validation scope/code drift')
    if not m['resources']['pass_resource_policy']:raise ValueError('Resource failure')
    if len(m.get('physical_artifacts',[]))!=len(s):raise ValueError('Missing aggregate resource proof')
    for ref in m.get('physical_artifacts',[]):
        physical=json.loads(verify(ref).read_text())
        if physical['workers']!=2 or physical['swapout_pages_delta'] or physical['peak_aggregate_rss_bytes']>v33.RESOURCE['max_rss_bytes']:raise ValueError('Parallel budget')
        for r in physical['workers_manifests']:
            worker=json.loads(verify(r).read_text());verify(worker['inputs']);verify(worker['raw'])
    return m,s


def gate(prepared,output,pilot,repaired,repeated,ban):
    old,m=compatibility(prepared);pm,ps=checked_batch(pilot,prepared,32)
    fm,fs=checked_batch(repaired,prepared,704);rm,rs=checked_batch(repeated,prepared,704,True)
    if rm['subshard']!=8:raise ValueError('Independent chunking required')
    bm=json.loads((ban/'manifest.json').read_text())
    if bm['status']!='PASS_BAN_FULL_SWEEP_AND_704_PARITY' or bm['prepared']!=record(prepared/'manifest.json') or bm['accelerator_code']!=accelerator_code():raise ValueError('BAN audit drift')
    for r in bm['files']:verify(r)
    previous=old_batches();old_ps=previous[1][2];c=duckdb.connect();c.execute("SET memory_limit='2GB'")
    all_rows=pq.read_table(verify(m['inputs'])).to_pylist()
    for path,ob,ss in previous:
        if ob['code']!=old['code'] or ob['prepared']!=record(OLD/'prepared/manifest.json'):raise ValueError('Adoption authority')
        got=pq.read_table(v33.files_for(ss,'inputs.parquet')).to_pylist()
        expected=all_rows[ob['start']:ob['start']+32]
        if sorted(got,key=lambda r:r['query_id'])!=sorted(expected,key=lambda r:r['query_id']):raise ValueError('Adoption input drift')
    for name in COMMON:
        v33.same(c,v33.files_for(old_ps,name),v33.files_for(ps,name))
        v33.same(c,v33.files_for(fs,name),v33.files_for(rs,name))
    c.close()
    seconds=max(pm['resources']['wall_seconds'],fm['resources']['wall_seconds'])
    size=max(sum(Path(r['path']).stat().st_size for s in ss for r in s['files'].values()) for ss in (ps,fs))
    hours=seconds/32*(23571-704)/3600;gib=size/32*23571/1024**3
    if hours>36 or gib>100 or shutil.disk_usage(v33.DATA).free<max(50*1024**3,2*gib*1024**3):raise ValueError('Cost gate')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'manifest.json',dict(status='PASS_PHYSICAL_RESUME',prepared=record(prepared/'manifest.json'),
      code=v33.code(),accelerator_code=accelerator_code(),ban_audit=record(ban/'manifest.json'),
      pilot=record(pilot/'manifest.json'),repaired=record(repaired/'manifest.json'),repeated=record(repeated/'manifest.json'),
      adopted=[record(p) for p,_,_ in previous],old_prepared=record(OLD/'prepared/manifest.json'),
      exhaustive_equal_files=list(COMMON),quality_measured=False,truth_columns_opened=False,
      cost=dict(conservative_pilot_seconds=seconds,remaining_projection_hours=hours,output_projection_gib=gib,
        uncertainty='TWO_LOTS_NOT_A_DURATION_GUARANTEE')))


def gate_authority(prepared,gate_path):
    v33.authority(prepared);g=json.loads(verify(gate_path).read_text())
    if g['status']!='PASS_PHYSICAL_RESUME' or g['prepared']!=record(prepared/'manifest.json') or g['code']!=v33.code() or g['accelerator_code']!=accelerator_code():raise ValueError('Resume gate drift')
    for key in ('ban_audit','pilot','repaired','repeated','old_prepared'):verify(g[key])
    for r in g['adopted']:verify(r)
    return g


def full(prepared,gate_dir,output):
    gate_ref=record(gate_dir/'manifest.json');g=gate_authority(prepared,gate_ref)
    output.mkdir(parents=True,exist_ok=False);generation=output/'generation';generation.mkdir()
    evaluator=record(v33.ROOT/'scripts/evaluate_retrieval_v33_full_replay.py')
    policy=record(v33.ROOT/'docs/siret_operational_equivalence_policy.md')
    write_json(output/'launch.json',dict(pid=os.getpid(),gate=gate_ref,evaluator=evaluator,policy=policy,accelerator_code=accelerator_code()))
    started=time.monotonic();done=[];completed=0
    try:
        for r in g['adopted']:
            p=verify(r);old=json.loads(p.read_text());new=dict(old)
            for sr in old['shards']:
                for f in json.loads(verify(sr).read_text())['files'].values():verify(f)
            new.update(prepared=record(prepared/'manifest.json'),code=v33.code(),
              adoption=dict(original=r,compatibility=gate_ref,reason='BAN704_EXACT_AND_PHYSICAL_OUTPUT_PARITY'))
            stage=generation/f"batch{old['start']:05d}";stage.mkdir();write_json(stage/'manifest.json',new)
            done.append(record(stage/'manifest.json'));completed+=old['count']
        # This newly recomputed lot already passed reverse/chunking parity.
        done.append(g['repaired']);completed+=32
        write_json(generation/'progress.json',dict(completed_queries=completed,total_queries=23571,adopted_queries=704,seconds=time.monotonic()-started))
        for start in range(736,23571,32):
            gate_authority(prepared,gate_ref)
            if time.monotonic()-started>36*3600 or shutil.disk_usage(v33.DATA).free<30*1024**3:raise RuntimeError('STOP_FULL_BUDGET')
            stage=generation/f'batch{start:05d}'
            with (generation/f'batch{start:05d}.log').open('x') as log:
                subprocess.run([sys.executable,'-m','scripts.retrieval_v34_resume','batch','--prepared',str(prepared),
                  '--output',str(stage),'--start',str(start),'--count',str(min(32,23571-start))],cwd=v33.ROOT,
                  stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1900)
            bm,_=v33.manifests(stage)
            if bm.get('accelerator_code')!=g['accelerator_code']:raise ValueError('Batch accelerator drift')
            for r in bm['physical_artifacts']:verify(r)
            done.append(record(stage/'manifest.json'));completed=min(start+32,23571)
            write_json(generation/'progress.json',dict(completed_queries=completed,total_queries=23571,adopted_queries=704,seconds=time.monotonic()-started))
        gate_authority(prepared,gate_ref)
        write_json(generation/'manifest.json',dict(status='SEALED_FULL_PAIRED_GENERATION',queries=23571,
          prepared=record(prepared/'manifest.json'),gate=gate_ref,batches=done,code=v33.code(),
          accelerator_code=accelerator_code(),quality_measured=False,truth_columns_opened=False,
          seconds=time.monotonic()-started,evaluation_pending=True))
        verify(evaluator);verify(policy)
        with (output/'evaluation.log').open('x') as log:
            subprocess.run([sys.executable,str(verify(evaluator)),'--generation',str(generation),
              '--output',str(output/'evaluation')],cwd=v33.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1900)
        write_json(output/'status.json',dict(status='COMPLETED_FULL_CRM_MASTER_COMPARISON',evaluation=record(output/'evaluation/manifest.json')))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',error=repr(exc),completed_queries=completed));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['batch','ban-audit','gate','full'])
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=32)
    p.add_argument('--subshard',type=int,default=32);p.add_argument('--reverse',action='store_true')
    for n in ('pilot','repaired','repeated','ban','gate'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.phase=='batch':batch(a.prepared,a.output,a.start,a.count,a.subshard,a.reverse)
    elif a.phase=='ban-audit':ban_audit(a.prepared,a.output)
    elif a.phase=='gate':gate(a.prepared,a.output,a.pilot,a.repaired,a.repeated,a.ban)
    else:full(a.prepared,a.gate,a.output)


if __name__=='__main__':main()
