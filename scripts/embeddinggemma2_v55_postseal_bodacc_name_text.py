"""National literal full-text discovery audit, never a retrieval candidate pool.

Complements subject-ID searches with CRM-name tokens. Partial large searches
are explicitly marked; token/official-name co-occurrence is not a legal edge.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import re
import time
import urllib.parse
import urllib.request

from scripts.embeddinggemma2_v52_postseal_bodacc_fulltext import API,CUTOFF,strings
from scripts.embeddinggemma2_v49_postseal_bodacc_roles import BASE,read,write,reference,decoded,unwrap,siren,names,norm

OUT=BASE/'v55/postseal_bodacc_name_text_20261012_a'
WORDS=['NAOS','INFODESCA','CEGEDIM','SMARTRX','CAMPHES','GEDIMAT','ATOL','SCHMIDT','EBSN','CONVERGENCE MEDIAS','TRESCAL','SB ALLIANCE']
LIMIT=1000 # Diagnostic materialization only, not any product candidate budget.


def collect(word):
    clauses=[f'search({f},"{word}")' for f in ('acte','modificationsgenerales','divers')]
    where='('+' OR '.join(clauses)+f') AND dateparution <= date\'{CUTOFF}\''
    output=[];offset=0;total=None;requests=[]
    while total is None or offset<min(total,LIMIT):
        url=API+'?'+urllib.parse.urlencode(dict(where=where,limit=100,offset=offset,order_by='dateparution desc, id asc'))
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=45) as response:body=json.load(response)
                break
            except Exception:
                if attempt==2:raise
                time.sleep(attempt+1)
        count=int(body['total_count'])
        if total is not None and total!=count:raise ValueError('API pagination drift')
        total=count;items=body['results']
        if not items and offset<min(total,LIMIT):raise ValueError('Incomplete page')
        requests.append(dict(url=url,total=total,offset=offset,received=len(items)))
        for r in items:
            parties=[]
            for field,role in (('listepersonnes','SUBJECT'),('listeprecedentexploitant','PREVIOUS_OPERATOR'),('listeprecedentproprietaire','PREVIOUS_OWNER')):
                for raw in decoded(r.get(field)):
                    p=unwrap(raw);parties.append(dict(role=role,siren=siren(p),names=names(p)))
            texts=[dict(field=key,text=value) for f in ('acte','modificationsgenerales','divers') for key,value in strings(r.get(f),f) if len(value)>5]
            verified=[t for t in texts if re.search(r'(?<![A-Z0-9])'+re.escape(norm(word))+r'(?![A-Z0-9])',norm(t['text']))]
            if verified:output.append(dict(id=r['id'],date=r['dateparution'][:10],parties=parties,
                url=r.get('url_complete'),matching_passages=verified))
        offset+=len(items)
    return dict(word=word,api_total=total,api_records_read=offset,complete_api_search=offset>=total,
        verified_literal_records=len(output),requests=requests,records=output,
        not_semantic_search=True,no_universal_absence_claim=True)


def main():
    if OUT.exists():raise ValueError('Fresh immutable diagnosis namespace')
    prior=BASE/'v53/postseal_bodacc_reverse_20261012_a'
    if read(prior/'summary.json')['status']!='COMPLETE_POSTSEAL_REVERSE_BODACC_TEXT_NOT_RETRIEVAL':raise ValueError('Complete source audit')
    cases=read(prior/'cases.json');allowed={r['label']['siren'] for r in cases if not r['protected_not_backfilled']}
    OUT.mkdir(parents=True);(OUT/'words').mkdir()
    write(OUT/'launch.json',dict(implementation=reference(Path(__file__)),source_cases=reference(prior/'cases.json'),
        keyword_scope=WORDS,cutoff=CUTOFF,diagnostic_read_limit=LIMIT,no_retrieval_input=True))
    results={};start=time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(collect,w):w for w in WORDS}
        for f in as_completed(futures):
            word=futures[f];r=f.result()
            for a in r['records']:
                a['pilot_miss_parent_structured_references']=sorted({p['siren'] for p in a['parties']} & allowed)
                # Free-text ID mention is a diagnostic flag, NEVER inferred relation.
                a['pilot_miss_parent_text_mentions']=sorted(p for p in allowed if any(re.search(r'(?<![0-9])'+r'\s*'.join(p)+r'(?![0-9])',t['text']) for t in a['matching_passages']))
            path=OUT/'words'/(word.replace(' ','_')+'.json');write(path,r);results[word]=r
            print(json.dumps({k:r[k] for k in ('word','api_total','api_records_read','complete_api_search','verified_literal_records')}),flush=True)
    summary=dict(status='COMPLETE_DECLARED_SCOPE_LITERAL_TEXT_AUDIT_NOT_RETRIEVAL',
        keywords=[dict(word=w,api_total=r['api_total'],read=r['api_records_read'],complete=r['complete_api_search'],
            literal_records=r['verified_literal_records'],
            records_with_pilot_miss_parent_reference=sum(bool(a['pilot_miss_parent_structured_references'] or a['pilot_miss_parent_text_mentions']) for a in r['records'])) for w,r in sorted(results.items())],
        sources=[reference(OUT/'words'/(w.replace(' ','_')+'.json')) for w in sorted(results)],
        implementation=reference(Path(__file__)),elapsed=time.monotonic()-start,
        no_training=True,no_predictions_changed=True,no_recall=True,no_positive_candidate_injected=True,
        no_closed_fold_opened=True,literal_absence_not_semantic_absence=True,
        structured_or_textual_reference_not_a_relation_without_role_proof=True)
    write(OUT/'summary.json',summary);print(json.dumps(dict(status=summary['status'],keywords=summary['keywords'],elapsed=summary['elapsed'])),flush=True)


if __name__=='__main__':main()
