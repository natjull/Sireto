"""Whole current group catalog and query-triggered public membership audit.

Current API is explicitly POSTCUTOFF_DIAGNOSTIC. No model input, alias or GT
selection. Member != group identity, membership isn't physical site ownership.
"""
from pathlib import Path
import json
import time
import urllib.parse
import urllib.request

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as words
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v119/banatic_public_context_20261013_a'
API='https://api.collectivites.beta.gouv.fr/referentiel/v1'
DOC='https://api.collectivites.beta.gouv.fr/api/referentiel/swagger-ui-init.js'
STOP=words.STOP|set('SYNDICAT SYNDICATS MIXTE MIXTES POUR SM SMO SMF CC CA EPCI EPT CU'.split())


def get(url):
    with urllib.request.urlopen(url,timeout=30) as response:
        return json.load(response),dict(url=url,http_date=response.headers.get('Date'),
            http_last_modified=response.headers.get('Last-Modified'),collected_unix=time.time())


def main():
    if OUTPUT.exists():raise ValueError('Fresh current diagnostic namespace')
    OUTPUT.mkdir(parents=True);(OUTPUT/'members').mkdir()
    with urllib.request.urlopen(DOC,timeout=30) as response:doc=response.read().decode()
    options=doc[doc.index('let options = ')+len('let options = '):].lstrip()
    spec,_=json.JSONDecoder().raw_decode(options);spec=spec['swaggerDoc']
    for endpoint in ('/referentiel/v1/groupements','/referentiel/v1/groupements/{siren}/membres'):
        if endpoint not in spec['paths']:raise ValueError('Only observed documented API routes')
    io.write(OUTPUT/'openapi.json',spec)
    groups=[];calls=[];offset=0;seen=set()
    while True:
        url=API+'/groupements?'+urllib.parse.urlencode(dict(limit=20,offset=offset))
        values,meta=get(url)
        if not isinstance(values,list):raise ValueError('Observed list pagination shape changed')
        calls.append(dict(meta,received=len(values),offset=offset))
        if not values:break
        for row in values:
            if row['siren'] in seen:raise ValueError('Duplicate current API pages, no fake completeness')
            seen.add(row['siren']);groups.append(row)
        offset+=len(values)
    io.write(OUTPUT/'groups.json',groups)
    launch=io.read(io.HOME/'v99/initial_alignment_native_20261013_a/launch.json')
    queries=io.read(verify(launch['panel']));matching={}
    for group in groups:
        source_tokens=words.tokens(group['nom'])-STOP
        for q in queries:
            tokens=words.tokens(q['crm_name'])-STOP
            # Name-only exploration, not claim of resolved identity. All group
            # source names were inventoried before reading the query surface.
            if len(tokens)>=2 and tokens<=source_tokens:
                matching.setdefault(group['siren'],[]).append(q['query_id'])
    sources={}
    for parent,qids in matching.items():
        payload,meta=get(API+'/groupements/'+parent+'/membres')
        path=OUTPUT/'members'/(parent+'.json');io.write(path,payload)
        sources[parent]=dict(artifact=record(path),request=meta,query_triggers=qids)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_CURRENT_BANATIC_QUERY_CONTEXT_POSTCUTOFF_DIAGNOSTIC',
        implementation=record(Path(__file__)),api_spec=record(OUTPUT/'openapi.json'),
        groups=record(OUTPUT/'groups.json'),groups_count=len(groups),requests=calls,
        query_panel=launch['panel'],name_triggered_groups=len(matching),memberships=sources,
        all_catalog_pages_to_empty=True,current_source_not_asof_august=True,
        source_eligibility=False,relationship_not_identity=True,no_gt_labels_read=True,
        no_feature_alias_or_prediction_created=True,observed_unix=time.time()))
    print('COMPLETE',len(groups),matching,flush=True)


if __name__=='__main__':main()
