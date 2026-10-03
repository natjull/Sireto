"""All128 label-side name catalog AFTER complete blind generation and point audit.

This artifact is label-selected diagnostics, NEVER a serving/training source.
"""
import argparse
import json
from pathlib import Path
import time

import duckdb
import pyarrow as pa

from scripts.audit_retrieval_v102_national_evidence import population
from src.xgb_matcher.retrieval_proof_v56 import signal
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json


def attach(queries,points,facts):
    if (len(queries)!=128 or len(points)!=128 or
            [(q['query_id'],q['fold']) for q in queries]!=[(r['query_id'],r['fold']) for r in points] or
            any(q['fold'] not in (2,3,4) for q in queries)):
        raise ValueError('Complete234 audit denominator required')
    by_owner={}
    for f in facts:
        if f['subject_code'] not in (1,2): raise ValueError('Unknown source owner grain')
        if ((f['subject_code']==1 and f['owner_key']!=f['official_parent']) or
                (f['subject_code']==2 and f['owner_key']//100000!=f['official_parent'])):
            raise ValueError('Official owner membership drift')
        by_owner.setdefault((f['subject_code'],f['owner_key']),[]).append(f)
    output=[]
    for q,r in zip(queries,points):
        if not r['human_siret'].startswith(r['human_siren']): raise ValueError('Audit label membership drift')
        own=by_owner.get((2,int(r['human_siret'])),[])
        parent=by_owner.get((1,int(r['human_siren'])),[])
        words=set(signal(q['crm_name']).split())
        evidence=[dict(**f,shared_literal_words=sorted(words&set(signal(f['name_text']).split())),
                       lexical_eligible=f['scope_code']!=0) for f in own+parent]
        output.append(dict(query_id=q['query_id'],fold=q['fold'],crm_name=q['crm_name'],
            exact_hit=r['exact_hit'],human_siret=r['human_siret'],human_siren=r['human_siren'],
            point_branch_reason=r['branch_zero_or_positive_reason'],facts=evidence))
    return output


def run(generation,point_report,output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    evaluation,launch,_=population(generation)
    report=json.loads(point_report.read_text())
    if (report['status']!='SEALED_EVALUATION_ONLY_POINT_PROOF_AUDIT' or report['queries']!=128 or
            report['generation']!=record(generation/'manifest.json') or
            report['evaluation']!=record(generation/'evaluation.json') or report['generation_modified']):
        raise ValueError('Complete consistent point audit required')
    words=json.loads(verify(launch['word_manifest']).read_text())
    source=json.loads(verify(words['facts_source']).read_text()); parts=source['stats']['parts']
    if source['stats']['observations']!=162537492 or sum(p['facts'] for p in parts)!=57804962:
        raise ValueError('Complete official name catalog required')
    refs=[p['artifact'] for p in parts]; paths=[str(verify(r)) for r in refs]
    codepaths=(Path(__file__),ROOT/'docs/retrieval_v103_name_catalog_audit.md')
    authority=dict(generation=record(generation/'manifest.json'),point_report=record(point_report),
        facts_manifest=words['facts_source'],parts=refs,implementation={str(p):record(p) for p in codepaths},
        truth_scope='LABEL_SELECTED_DIAGNOSTIC_ONLY_AFTER_SEALED128',not_a_serving_source=True)
    write_json(output/'launch.json',authority); budget=None
    try:
        with Budget(dict(max_seconds=80,max_rss_bytes=4294967296,max_swapout_pages_delta=0)) as budget:
            keys=sorted({(1,int(r['human_siren'])) for r in report['rows']}|
                        {(2,int(r['human_siret'])) for r in report['rows']})
            owners=pa.table({'subject_code':pa.array([s for s,k in keys],type=pa.uint8()),
                             'owner_key':pa.array([k for s,k in keys],type=pa.uint64())})
            c=duckdb.connect(); c.execute("SET threads=2"); c.execute("SET memory_limit='1GB'")
            c.register('audit_owners',owners)
            facts=c.execute('SELECT n.* FROM read_parquet(?) n JOIN audit_owners o USING(subject_code,owner_key) '
                            'ORDER BY subject_code,owner_key,scope_code,name_text',[paths]).fetch_arrow_table().to_pylist()
            c.close(); rows=attach(launch['queries'],report['rows'],facts)
        for ref in refs+list(authority['implementation'].values())+[authority['generation'],authority['point_report']]: verify(ref)
        write_json(output/'report.json',dict(status='SEALED_ALL128_LABEL_SIDE_OFFICIAL_NAME_CATALOG',rows=rows,
            queries=128,official_facts=len(facts),seconds=time.monotonic()-began,resources=budget.report(),
            launch=record(output/'launch.json'),truth_opened=True,fold0_opened=False,fold1_opened=False,
            truth_scope=authority['truth_scope'],not_a_serving_source=True,outputs_promoted_to_features=False,
            generation_modified=False,model_fit=False,ready_for_fold1=False))
        print(json.dumps(dict(queries=128,official_facts=len(facts),resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),generation_modified=False,
            not_a_serving_source=True,resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--generation',required=True,type=Path)
    p.add_argument('--point-report',required=True,type=Path); p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); run(a.generation,a.point_report,a.output)
