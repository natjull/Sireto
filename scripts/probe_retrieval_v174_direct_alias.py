"""One independently certified physical part; exact equivalence/cost, no recall."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import scripts.compile_retrieval_v168_address_access as backend
from scripts.probe_retrieval_v138_national_names import query_rows
from src.xgb_matcher.retrieval_direct_alias_v174 import evaluate
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget,quiet,LIMITS,POLICY,validate_report
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v174_direct_alias.json'
STATUS='SEALED_V174_ONE_PHYSICAL_PART_VECTOR_PRESENCE_COST_NOT_NATIONAL'
FLAGS={k:k=='crm_opened' for k in PROTECTED}


def inputs(cfg):
    if (cfg['resources']!=LIMITS or cfg['duckdb_version']!=duckdb.__version__
        or cfg['numpy_version']!=np.__version__ or type(cfg['part']) is not int or cfg['repetitions']!=3):
        raise ValueError('Pinned source and exact authorized pressure policy required')
    root=verify(cfg['source_launch']).parent;l,p=backend.load(root)
    m=backend.phase_record(root,'part',cfg['part'],context=(l,p))
    if (verify(cfg['part_manifest']).resolve()!=backend.directory(root,'part',cfg['part'])/'manifest.json'
        or cfg['part_manifest']['sha256']!=record(backend.directory(root,'part',cfg['part'])/'manifest.json')['sha256']
        or m['inputs']!=dict(part=p['parts'][cfg['part']])
        or not all(m['stats'][k] for k in ('all_scalar_feature_texts_exact','all_scalar_channel_sets_exact',
            'all_exported_postings_exact','all_raw_addresses_retained'))
        or m['stats']['address_fact_attributes_or_scope_changed'] or m['stats']['alias_or_proof_cap'] is not None):
        raise ValueError('Exactly certified successful whole-address part required')
    rows=query_rows(json.loads(verify(cfg['query_only_launch']).read_text()),1)
    return root,l,p,m,rows[0]


def oracle(c,files,ids,text,reader_group=4):
    """Every guarded posting, bounded readers; sum before fraction division."""
    q=channels(text);terms=[(ch,t) for ch,ts in enumerate(q) for t in sorted(ts)]
    if normalized(text):terms.append((3,normalized(text)))
    qt=pa.table(dict(channel=pa.array([x[0] for x in terms],type=pa.uint8()),
                     token=pa.array([x[1] for x in terms],type=pa.string())))
    c.register('query172',qt);counts=np.zeros((len(ids),4),dtype=np.uint32)
    if reader_group!=4:raise ValueError('Exact bounded four physical readers required')
    try:
        for begin in range(0,len(files),reader_group):
            rows=c.execute('SELECT p.alias_id,p.channel,count(*) n FROM read_parquet($files,hive_partitioning=false) p '
                'JOIN query172 q USING(channel,token) GROUP BY p.alias_id,p.channel',
                dict(files=[str(f) for f in files[begin:begin+reader_group]])).fetch_arrow_table()
            aid=rows['alias_id'].to_numpy();ch=rows['channel'].to_numpy();n=rows['n'].to_numpy()
            pos=aid.astype(np.int64)-int(ids[0])
            if np.any((pos<0)|(pos>=len(ids))) or np.any(ch>3) or np.any(n<=0):
                raise ValueError('Every original posting belongs to this complete alias interval')
            np.add.at(counts,(pos,ch),n.astype(np.uint32))
    finally:c.unregister('query172')
    cov=np.zeros(len(ids),dtype=np.float64)
    for ch,tokens in enumerate(q):
        if np.any(counts[:,ch]>len(tokens)):raise ValueError('Distinct unchanged query posting sets required')
        if tokens:np.maximum(cov,counts[:,ch]/len(tokens),out=cov)
    if np.any(counts[:,3]>1):raise ValueError('Every whole-alias exact posting once required')
    return cov,counts[:,3].astype(np.bool_)


def implementation():
    files=[*backend.implementation_paths(),Path(__file__),CONFIG,
        ROOT/'src/xgb_matcher/retrieval_direct_alias_v172.py',ROOT/'src/xgb_matcher/retrieval_direct_alias_v174.py',
        ROOT/'scripts/probe_retrieval_v138_national_names.py',ROOT/'docs/retrieval_v174_direct_alias_contract.md']
    return {str(f):record(f) for f in files}


def load(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config']
    if (cfg!=json.loads(CONFIG.read_text()) or any(launch.get(k) is not v for k,v in FLAGS.items())
        or launch['old_candidates_opened'] or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())):
        raise ValueError('Immutable source-bound closed-label cost launch required')
    for ref in launch['implementation'].values():verify(ref)
    gate=json.loads(verify(launch['preflight']).read_text())
    if (verify(launch['preflight']).resolve()!=output/'preflight.json' or gate['policy_id']!=POLICY
        or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']):
        raise ValueError('Actual authorized pressure preflight required')
    return launch,inputs(cfg)


def worker(output):
    if any(f.name not in ('launch.json','preflight.json','worker.log') for f in output.iterdir()):
        raise ValueError('Immutable cost worker, no retry')
    budget=c=None
    try:
        with Budget(LIMITS) as budget:
            launch,(root,l,p,m,row)=load(output);cfg=launch['config'];spec=p['parts'][cfg['part']]
            table=pq.ParquetFile(verify(m['artifacts']['features.parquet'])).read(use_threads=False)
            start=spec['alias_start']+spec['row_start'];n=spec['row_stop']-spec['row_start']
            expected=np.arange(start,start+n,dtype=np.uint32)
            if len(table)!=n or not np.array_equal(table['alias_id'].to_numpy(),expected):
                raise ValueError('Every original alias exactly once in this physical part required')
            c=duckdb.connect();c.execute('SET threads=1');c.execute('SET memory_limit=?',[cfg['duckdb_memory']])
            c.execute('SET temp_directory=?',[str(output/'spill')]);c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            text=row['crm_street'];trials=[]
            for _ in range(cfg['repetitions']):
                aid,ac,ae,trial=evaluate(c,table,text)
                if trials and (not np.array_equal(ids,aid) or not np.array_equal(cov,ac) or not np.array_equal(exact,ae)):
                    raise ValueError('STOP_WARM_REPETITION_NOT_BIT_EXACT')
                ids,cov,exact=aid,ac,ae;trials.append(trial)
            stats=dict(trials=trials,repetitions=cfg['repetitions'],all_repetitions_bit_exact=True,
                same_query_and_all_aliases_each_time=True,cold_runtime_startup_not_isolated=True,
                disk_cache_state='UNCONTROLLED_NOT_COLD_DISK',no_alias_or_candidate_cap=True)
            files=[verify(ref) for name,ref in m['artifacts'].items() if name!='features.parquet']
            began=time.monotonic();oc,oe=oracle(c,files,ids,text);oracle_seconds=time.monotonic()-began
            if not np.array_equal(cov,oc) or not np.array_equal(exact,oe):
                raise ValueError('STOP_DIRECT_FEATURE_NOT_BIT_EXACT_TO_EVERY_ORIGINAL_POSTING')
            digest=hashlib.sha256(cov.tobytes()+exact.tobytes()).hexdigest()
            c.close();c=None
            for ref in m['artifacts'].values():verify(ref)
            backend.revalidate(root,l,p)
            for ref in launch['implementation'].values():verify(ref)
            verify(cfg['query_only_launch']);verify(cfg['part_manifest']);verify(cfg['source_launch'])
        write_json(output/'manifest.json',dict(status=STATUS,launch=record(output/'launch.json'),resources=budget.report(),
            part=cfg['part'],aliases=n,postings=m['stats']['postings'],query_id=row['query_id'],fold=row['fold'],
            direct=stats,oracle_seconds=oracle_seconds,every_score_bit_exact=True,sha256=digest,
            positive_aliases=int(np.count_nonzero(cov)),exact_aliases=int(np.count_nonzero(exact)),
            maximum_oracle_readers=4,every_part_artifact_full_hash_before_and_after=True,
            national_corpus_measured=False,initial100_emitted=False,
            complete_retrieval_index=False,ready_for_fold1=False,old_candidates_opened=False,**FLAGS))
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,
            initial100_emitted=False,**FLAGS));raise
    finally:
        if c is not None:c.close()


def run(output):
    cfg=json.loads(CONFIG.read_text())
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):raise ValueError('CATNAT-only cost root required')
    if output.exists():raise FileExistsError('Immutable cost root exists')
    gate=quiet(cfg['quiet_seconds'],50,55)
    output.mkdir(parents=True,exist_ok=False);write_json(output/'preflight.json',gate)
    write_json(output/'launch.json',dict(config=cfg,preflight=record(output/'preflight.json'),
        implementation=implementation(),old_candidates_opened=False,**FLAGS))
    try:
        with (output/'worker.log').open('x') as log:
            s=supervise([sys.executable,'-u','-m','scripts.probe_retrieval_v174_direct_alias','worker','--output',str(output)],
                log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},output/'supervisor_stop.json')
        write_json(output/'supervision.json',s);parent_exit(output)
        m=json.loads((output/'manifest.json').read_text());validate_report(m['resources'])
        if m['status']!=STATUS or m['launch']!=record(output/'launch.json'):raise ValueError('Exact supervised part cost required')
        print(json.dumps(m),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),**FLAGS));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
