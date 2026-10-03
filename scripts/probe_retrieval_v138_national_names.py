"""Bounded national name/one-hop evidence cost, not initial100 or recall."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.xgb_matcher.retrieval_national_names_v138 import NationalNames,scatter_positive
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED,one_hop_names
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v138_national_names.json'


def query_rows(old,count):
    fields={'query_id','fold','crm_name','crm_number','crm_suffix','crm_street','crm_postcode','crm_insee'}
    if (old['truth_opened'] or old['old_candidates_opened'] or len(old['queries'])!=128
        or not isinstance(count,int) or isinstance(count,bool) or not 1<=count<=128
        or any(set(q)!=fields or q['fold'] not in (2,3,4) or not isinstance(q['crm_name'],str) for q in old['queries'])
        or len({q['query_id'] for q in old['queries']})!=128):
        raise ValueError('Frozen query-only folds234 without labels or old hits required')
    return old['queries'][:count]


def load(output):
    launch=json.loads((output/'launch.json').read_text())
    for ref in launch['implementation'].values():verify(ref)
    cfg=launch['config'];bridge=json.loads(verify(cfg['bridge']).read_text())
    if (bridge['status']!='SEALED_V137_COMPLETE_OFFICIAL_PROOF_POSITION_BRIDGE'
        or any(bridge.get(k) is not False for k in PROTECTED) or bridge['complete_retrieval_index'] or bridge['ready_for_fold1']):
        raise ValueError('Protected complete official name/graph bridge required')
    source=json.loads(verify(bridge['sources']['owners']).read_text());graph=json.loads(verify(bridge['sources']['graph']).read_text())
    if any(any(m.get(k) is not False for k in PROTECTED) for m in (source,graph)):
        raise ValueError('Protected original official observations required')
    if [x['bucket'] for x in bridge['buckets']]!=source['buckets']:raise ValueError('Whole original raw alias/owner buckets required')
    rows=query_rows(json.loads(verify(cfg['query_only_launch']).read_text()),cfg['queries'])
    return launch,bridge,source,graph,rows


def mapped(ref,refs):
    refs.append(ref);return np.load(verify(ref),mmap_mode='r',allow_pickle=False)


def close_arrays(arrays):
    for a in arrays.values():a._mmap.close()


def worker(output):
    if any((output/k).exists() for k in ('manifest.json','worker_stop.json','cost_outputs.json')):
        raise ValueError('Immutable national proof cost already started')
    budget=None;index=None;graph_arrays={}
    try:
        with Budget(json.loads((output/'launch.json').read_text())['config']['resources']) as budget:
            launch,bridge,source,graph,rows=load(output);cfg=launch['config'];began=time.monotonic();refs=[]
            index=NationalNames(cfg['names'])
            if index.source!=source['source'] or index.names!=bridge['stats']['names']:
                raise ValueError('Identical sealed global raw alias positions required')
            for b in bridge['buckets']:
                for ref in b['artifacts'].values():verify(ref)
                for key in ('subject_code','current_exact','name_fact_offsets'):verify(b['bucket']['artifacts'][key+'.npy'])
            for key in ('graph_site_positions','graph_parent_positions','predecessor_graph_positions','successor_graph_positions'):
                graph_arrays[key]=mapped(bridge['graph'][key+'.npy'],refs)
            graph_arrays['continuity_codes']=mapped(graph['artifacts']['continuity_codes.npy'],refs)
            load_seconds=time.monotonic()-began;outputs=[]
            for raw in rows:
                began=time.monotonic();coverage,exact,lexical=index.coverage(raw['crm_name'],max_seconds=cfg['max_query_seconds'])
                lexical_done=time.monotonic();ns=bridge['stats']['sites'];np_=bridge['stats']['parents']
                pc=np.zeros(np_);pe=np.zeros(np_,dtype=bool);sc=np.zeros(ns);se=np.zeros(ns,dtype=bool)
                references_processed=0
                for item in bridge['buckets']:
                    # Source hashes were verified at opening. Readonly maps
                    # opened/closed per physical owner bucket, not per owner.
                    b=item['bucket'];old={key:np.load(b['artifacts'][key+'.npy']['path'],mmap_mode='r',allow_pickle=False)
                        for key in ('subject_code','current_exact','name_fact_offsets')}
                    new={key.removesuffix('.npy'):np.load(ref['path'],mmap_mode='r',allow_pickle=False) for key,ref in item['artifacts'].items()}
                    start=b['name_start'];sl=slice(start,start+b['names'])
                    try:
                        stats=scatter_positive(coverage[sl],exact[sl],old,new,pc,pe,sc,se)
                        references_processed+=stats['references_processed']
                    finally:close_arrays(old);close_arrays(new)
                    if time.monotonic()-began>cfg['max_query_seconds']:raise TimeoutError('STOP_NATIONAL_NAME_OWNER_COST')
                scatter_done=time.monotonic()
                relation=one_hop_names(pc,sc,graph_arrays['graph_site_positions'],graph_arrays['graph_parent_positions'],
                    graph_arrays['predecessor_graph_positions'],graph_arrays['successor_graph_positions'],graph_arrays['continuity_codes'])
                elapsed=time.monotonic()-began
                if elapsed>cfg['max_query_seconds']:raise TimeoutError('STOP_NATIONAL_NAME_RELATION_COST')
                trace=dict(seconds=elapsed,lexical_seconds=lexical_done-began,owner_scatter_seconds=scatter_done-lexical_done,
                    relation_seconds=time.monotonic()-scatter_done,lexical=lexical,references_processed=references_processed,
                    direct_parent_proofs=int(np.count_nonzero(pc)),direct_site_proofs=int(np.count_nonzero(sc)),
                    related_endpoint_proofs=int(np.count_nonzero(relation)),parent_exact=int(np.count_nonzero(pe)),site_exact=int(np.count_nonzero(se)))
                outputs.append(dict(query_id=raw['query_id'],fold=raw['fold'],trace=trace))
                print(json.dumps(outputs[-1]),flush=True)
                del coverage,exact,pc,pe,sc,se,relation;gc.collect()
            # Full inputs verified again AFTER all outputs, not every query.
            for b in index.metadata['access']:
                for ref in b['artifacts'].values():verify(ref)
            for b in bridge['buckets']:
                for ref in b['artifacts'].values():verify(ref)
                for key in ('subject_code','current_exact','name_fact_offsets'):verify(b['bucket']['artifacts'][key+'.npy'])
            for ref in refs:verify(ref)
            load(output)
            write_json(output/'cost_outputs.json',dict(rows=outputs,truth_opened=False,recall_measured=False,initial100_emitted=False))
        write_json(output/'manifest.json',dict(status='SEALED_V138_NATIONAL_NAME_AND_ONEHOP_EVIDENCE_COST_ONLY',
            launch=record(output/'launch.json'),outputs=record(output/'cost_outputs.json'),queries=len(outputs),load_seconds=load_seconds,
            resources=budget.report(),initial100_emitted=False,complete_retrieval_index=False,national_name_and_graph_corpus=True,
            truth_opened=False,crm_opened=True,fold0_opened=False,fold1_opened=False,old_candidates_opened=False,
            model_fit=False,recall_measured=False,downstream_models_changed=False,ready_for_fold1=False,
            disk_cache_state='UNCONTROLLED_NOT_COLD_DISK',score_stage_only_not_retrieval_latency=True))
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise
    finally:
        if index is not None:index.close()
        close_arrays(graph_arrays)


def run(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text())
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v138_national_names.md',
        *[ROOT/('src/xgb_matcher/'+k+'.py') for k in ('retrieval_national_names_v138','retrieval_proof_bridge_v137',
            'retrieval_extended_catalog_v136','retrieval_coupled_blocks_v121','retrieval_activity_joint_v114',
            'retrieval_variants_v118','retrieval_v20_workbench','retrieval_parent_discovery_v18')]]
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v138_national_names','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-8000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
