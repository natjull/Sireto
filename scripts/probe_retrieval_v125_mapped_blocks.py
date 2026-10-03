"""Bounded persistent-index probe; fresh query process, scalar oracle sealed separately."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pyarrow.parquet as pq

from scripts.probe_retrieval_v118_official_variants import fingerprint
from scripts.probe_retrieval_v124_vector_blocks import inputs,extra_source
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
from src.xgb_matcher.retrieval_coupled_source_v122 import build_sites
from src.xgb_matcher.retrieval_vector_blocks_v124 import VectorCoupledBlocks
from src.xgb_matcher.retrieval_mapped_blocks_v125 import export_fixture,MappedBlocks
from src.xgb_matcher.retrieval_variants_v118 import observations
from src.xgb_matcher.retrieval_joint_v46 import geo_parameters
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v125_mapped_blocks.json'


def query(raw,geography):
    p,_,reliable=geo_parameters(geography[raw['query_id']])
    return Query(raw['crm_name'],' '.join(raw[k] for k in ('crm_number','crm_suffix','crm_street')).strip(),
        raw['crm_postcode'],raw['crm_insee'],raw['crm_number'],None if p is None else tuple(p),reliable)


def authorities(output):
    launch=json.loads((output/'launch.json').read_text())
    for ref in launch['implementation'].values():verify(ref)
    return launch


def output_size(output,cfg):
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if size>cfg['max_output_bytes']:raise ValueError('STOP_OUTPUT_SIZE')
    return size


def compile_worker(output):
    launch=authorities(output);cfg=launch['config']
    with Budget(cfg['resources']) as budget:
        refs,old,models,geography=inputs(cfg);began=time.monotonic()
        extra_names,name_source=extra_source(cfg,output,'name');extra_addresses,address_source=extra_source(cfg,output,'address')
        expanded=[p for row in extra_names for p in observations(row,'official_names',int(row['source_row_ordinal']))]
        def table(key,columns=None):return pq.read_table(verify(refs[key]),columns=columns,use_threads=False).to_pylist()
        sites,source_stats=build_sites(table('sample_current_site.parquet'),table('names.parquet'),table('facts.parquet'),
            table('observations.parquet',['fact_id','is_current']),expanded,extra_addresses,table('sample_successions.parquet'))
        source_seconds=time.monotonic()-began;began=time.monotonic();index=VectorCoupledBlocks(sites,cfg['block_size'])
        build_seconds=time.monotonic()-began;source_refs=[record(output/'launch.json'),name_source['record'],address_source['record']]
        for key in ('variant_manifest','sample_manifest'):
            source_refs.append(cfg[key]);source_refs.extend(json.loads(verify(cfg[key]).read_text())['artifacts'].values())
        source_refs=list({r['path']:r for r in source_refs}.values())
        began=time.monotonic();state=export_fixture(index,output/'index',source_refs);export_seconds=time.monotonic()-began
        # Expected results use the scalar functions on ALL32768 sites. This
        # file is never opened by the fresh mapped query worker below.
        rows=[];began=time.monotonic()
        for raw in old['queries'][:cfg['queries']]:
            result=index.exhaustive(query(raw,geography),models[str(raw['fold'])]['weights'],limit=cfg['collector_cap'])
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],candidates=result))
        oracle_seconds=time.monotonic()-began
        write_json(output/'scalar_oracles.json',dict(rows=rows,truth_opened=False,recall_measured=False,national_corpus=False))
        for source in (name_source,address_source):
            if fingerprint(source['path'])!=source['fingerprint']:raise ValueError('Official source changed')
        inputs(cfg);size=output_size(output,cfg)
    authorities(output)
    result=dict(status='SEALED_BOUNDED_MAPPED_INDEX_AND_SEPARATE_SCALAR_ORACLES',index=record(output/'index/index.json'),
        oracles=record(output/'scalar_oracles.json'),source_stats=source_stats,extra_sources=[name_source,address_source],
        source_seconds=source_seconds,build_seconds=build_seconds,export_seconds=export_seconds,oracle_seconds=oracle_seconds,
        array_bytes=state['array_bytes'],array_files=len(state['files']),aliases=state['aliases'],resources=budget.report(),
        output_bytes=size,pid=os.getpid(),sources_selected_by_query_or_truth=False,truth_opened=False,national_corpus=False,
        recall_measured=False,model_fit=False,activity_grades_enabled=False)
    write_json(output/'compile_manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_stats','extra_sources')}),flush=True)


def query_worker(output):
    launch=authorities(output);cfg=launch['config']
    with Budget(cfg['resources']) as budget:
        _,old,models,geography=inputs(cfg)
        compiler=json.loads((output/'compile_manifest.json').read_text());verify(compiler['index'])
        if compiler['pid']==os.getpid():raise ValueError('Fresh mapped process required')
        began=time.monotonic();index=MappedBlocks(output/'index');load_seconds=time.monotonic()-began;rows=[]
        for raw in old['queries'][:cfg['queries']]:
            found,trace=index.search(query(raw,geography),models[str(raw['fold'])]['weights'],
                limit=cfg['collector_cap'],max_seconds=cfg['max_query_seconds'])
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],candidates=found,trace=trace,recall_measured=False))
            print(json.dumps(dict(query_id=raw['query_id'],trace=trace)),flush=True)
        inputs(cfg);authorities(output)
        write_json(output/'cost_outputs.json',dict(rows=rows,truth_opened=False,recall_measured=False,national_corpus=False,
            source_python_index_loaded=False,oracle_file_opened=False,old_candidates_opened=False))
        size=output_size(output,cfg)
    result=dict(status='SEALED_MAPPED_QUERYONLY_COST_OUTPUTS_NOT_YET_ORACLE_COMPARED',outputs=record(output/'cost_outputs.json'),
        resources=budget.report(),output_bytes=size,load_seconds=load_seconds,pid=os.getpid(),fresh_python_process=True,
        disk_cache_state='UNCONTROLLED_NOT_COLD_DISK',source_python_index_loaded=False,oracle_file_opened=False,
        old_candidates_opened=False,truth_opened=False,recall_measured=False,national_corpus=False)
    write_json(output/'query_manifest.json',result);print(json.dumps(result),flush=True)


def run(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text())
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v125_mapped_blocks.md',
        *[ROOT/('src/xgb_matcher/'+name+'.py') for name in ('retrieval_coupled_blocks_v121','retrieval_coupled_source_v121',
          'retrieval_coupled_source_v122','retrieval_tight_blocks_v123','retrieval_vector_blocks_v124','retrieval_mapped_blocks_v125',
          'retrieval_variants_v118','retrieval_activity_joint_v114','retrieval_joint_v46','retrieval_v20_workbench','retrieval_parent_discovery_v18')],
        ROOT/'scripts/probe_retrieval_v124_vector_blocks.py',ROOT/'scripts/probe_retrieval_v118_official_variants.py']
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        for mode in ('compile','query'):
            with (output/(mode+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v125_mapped_blocks',mode,'--output',str(output)],
                    check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            print((output/(mode+'.log')).read_text()[-10000:],flush=True)
        # Only after ALL mapped outputs are sealed: exact independent oracle
        # comparison in this separate supervisor, never input to generation.
        c=json.loads((output/'compile_manifest.json').read_text());q=json.loads((output/'query_manifest.json').read_text())
        expected=json.loads(verify(c['oracles']).read_text())['rows'];actual=json.loads(verify(q['outputs']).read_text())['rows']
        if len(expected)!=cfg['queries'] or len(actual)!=len(expected):raise ValueError('Oracle query denominator drift')
        for a,b in zip(expected,actual):
            if any(a[k]!=b[k] for k in ('query_id','fold','candidates')):raise ValueError('Mapped/scalar complete oracle drift')
        if c['pid']==q['pid'] or q['oracle_file_opened'] or q['old_candidates_opened']:raise ValueError('Generation isolation drift')
        authorities(output);size=output_size(output,cfg)
        write_json(output/'manifest.json',dict(status='SEALED_MAPPED_SAMPLE_EXACT_SCALAR_ORACLE_COST_ONLY',
            launch=record(output/'launch.json'),compile=record(output/'compile_manifest.json'),query=record(output/'query_manifest.json'),
            outputs=q['outputs'],index=c['index'],oracle_comparison_exact=True,queries=len(actual),sites=c['source_stats']['sites'],
            array_bytes=c['array_bytes'],array_files=c['array_files'],aliases=c['aliases'],output_bytes=size,
            fresh_python_process=True,oracle_opened_only_after_query_output_seal=True,truth_opened=False,crm_opened=True,
            fold0_opened=False,fold1_opened=False,model_fit=False,model_used=True,activity_grades_enabled=False,
            national_corpus=False,recall_measured=False,identifiers_as_features=False,downstream_models_changed=False,ready_for_fold1=False))
        print('EXACT_SCALAR_ORACLE_PASS_ALL_FOUR_FRESH_MAPPED_OUTPUTS',flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','compile','query']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();{'run':run,'compile':compile_worker,'query':query_worker}[a.mode](a.output)
