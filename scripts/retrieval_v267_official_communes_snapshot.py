"""One fixed public official source GET, no CRM/labels/model/candidate search.

All returned current municipalities AND municipal arrondissements, ALL zones.
Server listing completeness is not independently recertified against COG.
No credentials, paid APIs, external compute or query-specific request.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.request import Request,urlopen

from src.xgb_matcher.retrieval_memory_policy_v214 import Budget,LIMITS,POLICY,quiet,validate_report,parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

URL='https://geo.api.gouv.fr/communes?fields=nom,code,codesPostaux,siren,codeEpci&type=commune-actuelle,arrondissement-municipal&zone=metro,drom,com&format=json'
OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v267_official_communes_snapshot')
MAX_BYTES=16*1024*1024 # Physical payload guard, not truncation/filtering.
FALSE={k:False for k in ('truth_opened','crm_opened','fold0_opened','fold1_opened','recall_measured',
    'ready_for_fold1','downstream_models_changed','identifiers_as_features','candidates_emitted')}
STATUS='SEALED_V267_ALL_RETURNED_OFFICIAL_COMMUNES_SOURCE_ONLY'


def parse(payload):
    rows=json.loads(payload)
    if not isinstance(rows,list) or not 30000<=len(rows)<=50000:
        raise ValueError('Whole unfiltered official listing sanity required, not a query result')
    seen=set();stats=dict(rows=len(rows),missing_siren=0,missing_epci=0,postal_memberships=0)
    for r in rows:
        if (not isinstance(r,dict) or set(r)-{'nom','code','codesPostaux','siren','codeEpci'}
                or not {'nom','code','codesPostaux'}<=set(r)
                or not isinstance(r['nom'],str) or not r['nom'].strip()
                or not isinstance(r['code'],str) or not re.fullmatch('[0-9A-Z]{5}',r['code'])
                or r['code'] in seen or not isinstance(r['codesPostaux'],list)
                or len(set(r['codesPostaux']))!=len(r['codesPostaux'])
                or any(not isinstance(p,str) or not re.fullmatch('[0-9]{5}',p) for p in r['codesPostaux'])):
            raise ValueError('Every row typed/unique, no silent dropping or source reconstruction')
        seen.add(r['code']);stats['postal_memberships']+=len(r['codesPostaux'])
        for field,counter in (('siren','missing_siren'),('codeEpci','missing_epci')):
            value=r.get(field)
            if value is None or value=='':stats[counter]+=1
            elif not isinstance(value,str) or not re.fullmatch('[0-9]{9}',value):
                raise ValueError('Explicit official parent identity or UNKNOWN required')
    return rows,stats


def implementation():
    return {str(p):record(p) for p in (Path(__file__),ROOT/'docs/retrieval_v267_official_communes_contract.md',
        ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py',ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py')}


def load(root):
    if (not root.resolve().is_relative_to(OUTPUT.resolve()) or (root/'status.json').exists()
            or (root/'phase/worker_stop.json').exists() or (root/'phase/supervisor_stop.json').exists()):
        raise ValueError('Own fresh source snapshot, no STOP retry')
    l=json.loads((root/'launch.json').read_text())
    if l['url']!=URL or l['limits']!=LIMITS or l['max_bytes']!=MAX_BYTES or any(l.get(k) is not False for k in FALSE):
        raise ValueError('Frozen source-only GET and authorized limits required')
    for ref in l['implementation'].values():verify(ref)
    return l


def worker(root):
    load(root);dest=root/'phase';budget=None
    try:
        with Budget(LIMITS) as budget:
            def guard():
                if budget.failure:raise RuntimeError(budget.failure)
                if time.monotonic()-budget.start>80:raise TimeoutError('STOP_OFFICIAL_SOURCE_FETCH_COST')
            request=Request(URL,headers={'User-Agent':'SIRETO-OfficialSourceSnapshot/267','Accept':'application/json'})
            with urlopen(request,timeout=30) as response:
                if response.status!=200 or response.geturl()!=URL:
                    raise ValueError('Successful exact official HTTPS endpoint, no redirect source drift')
                headers=dict(content_type=response.headers.get('Content-Type'),date=response.headers.get('Date'),
                    last_modified=response.headers.get('Last-Modified'),etag=response.headers.get('ETag'))
                chunks=[];length=0
                while True:
                    guard();part=response.read(65536)
                    if not part:break
                    length+=len(part)
                    if length>MAX_BYTES:raise ValueError('STOP_WHOLE_SOURCE_PAYLOAD_BUDGET_NO_TRUNCATION')
                    chunks.append(part)
            payload=b''.join(chunks);rows,stats=parse(payload);guard()
            (dest/'response.json').write_bytes(payload)
            if parse((dest/'response.json').read_bytes())[1]!=stats:raise ValueError('Persisted source parse drift')
            write_json(dest/'source.json',dict(url=URL,headers=headers,response=record(dest/'response.json'),stats=stats,
                all_returned_rows_preserved=True,query_filters_sent=False,source_fields_filtered=True,
                current_and_arrondissement_requested=True,all_zones_requested=True,independent_cog_completeness_audited=False,
                source_api_listing_not_full_siret_corpus=True,**FALSE))
            guard();load(root)
        validate_report(budget.report())
        write_json(dest/'manifest.json',dict(status='QUARANTINED_V267_SOURCE',launch=record(root/'launch.json'),
            preflight=record(dest/'preflight.json'),source=record(dest/'source.json'),resources=budget.report(),**FALSE))
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**FALSE));raise


def admission(root):
    load(root);dest=root/'phase';parent_exit(dest)
    m=json.loads((dest/'manifest.json').read_text());validate_report(m['resources'])
    gate=json.loads(verify(m['preflight']).read_text())
    if (m['status']!='QUARANTINED_V267_SOURCE' or m['launch']!=record(root/'launch.json')
            or m['preflight']!=record(dest/'preflight.json') or gate['policy_id']!=POLICY
            or gate['pass_system_quiet_policy'] is not True or gate['quiet_seconds']<30
            or any(m.get(k) is not False for k in FALSE)):
        raise ValueError('Actual own source producer successful/quiet/hash required')
    s=json.loads(verify(m['source']).read_text());rows,stats=parse(verify(s['response']).read_bytes())
    if (s['url']!=URL or s['stats']!=stats or any(s.get(k) is not False for k in FALSE)
            or s['all_returned_rows_preserved'] is not True or s['query_filters_sent'] is not False
            or s['independent_cog_completeness_audited'] is not False):
        raise ValueError('Whole original official source and declared coverage limitation required')
    return m,s


def sealed(root):
    root=Path(root);m,s=admission(root);out=json.loads((root/'manifest.json').read_text())
    if out['status']!=STATUS or out['producer']!=record(root/'phase/manifest.json') or out['source']!=m['source']:
        raise ValueError('Same actual source producer and immutable snapshot required')
    return out,s


def run(root):
    if root.exists() or not root.resolve().is_relative_to(OUTPUT.resolve()):raise ValueError('Fresh own output root only')
    root.mkdir(parents=True,exist_ok=False);dest=root/'phase';dest.mkdir()
    try:
        write_json(root/'launch.json',dict(url=URL,limits=LIMITS,max_bytes=MAX_BYTES,implementation=implementation(),**FALSE))
        write_json(dest/'preflight.json',quiet(30,50))
        with (dest/'worker.log').open('x') as log:
            s=supervise([sys.executable,'-u','-m','scripts.retrieval_v267_official_communes_snapshot','worker','--output',str(root)],
                log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','RAYON_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},dest/'supervisor_stop.json')
        write_json(dest/'supervision.json',s);m,source=admission(root)
        write_json(root/'manifest.json',dict(status=STATUS,producer=record(dest/'manifest.json'),source=m['source'],stats=source['stats'],**FALSE))
        sealed(root);print(json.dumps(dict(status=STATUS,stats=source['stats'])),flush=True)
    except BaseException as exc:
        write_json(root/'status.json',dict(status='STOP',reason=str(exc),**FALSE));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','worker'))
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    (run if args.mode=='run' else worker)(args.output)
