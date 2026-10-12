"""Read full official text where miss parents are subjects OR mentioned parties.

Adds separately identified corporate anchors for inspecting provenance chains.
All work is label-side diagnosis; no retrieval candidate or new metric output.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import re
import time
import urllib.parse
import urllib.request

from scripts.embeddinggemma2_v52_postseal_bodacc_fulltext import API,CUTOFF,strings
from scripts.embeddinggemma2_v49_postseal_bodacc_roles import BASE,read,write,reference,decoded,unwrap,siren,names,tokens,norm

OUT=BASE/'v53/postseal_bodacc_reverse_20261012_a'
# Official SIRENE anchors; an audit lookup, never a CRM alias or model input.
ANCHORS=['780876405','409080538','343222311','961500501','351014352']


def collect(parent):
    spaced=' '.join(parent[i:i+3] for i in range(0,9,3))
    clauses=[f'search(registre,"{parent}")']
    for field in ('acte','modificationsgenerales','divers'):
        for spelling in (parent,spaced):clauses.append(f'search({field},"{spelling}")')
    where='('+' OR '.join(clauses)+f') AND dateparution <= date\'{CUTOFF}\''
    output=[];requests=[];offset=0;total=None
    while total is None or offset<total:
        url=API+'?'+urllib.parse.urlencode(dict(where=where,limit=100,offset=offset,order_by='dateparution desc, id asc'))
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=45) as response:body=json.load(response)
                break
            except Exception:
                if attempt==2:raise
                time.sleep(attempt+1)
        count=int(body['total_count'])
        if total is not None and total!=count:raise ValueError('Public source pagination drift')
        total=count;items=body['results']
        if not items and offset<total:raise ValueError('Incomplete source')
        requests.append(dict(url=url,total=total,offset=offset,received=len(items)))
        for r in items:
            if r['dateparution'][:10]>CUTOFF:raise ValueError('Future source')
            parties=[];text=[]
            for field,kind in (('listepersonnes','SUBJECT'),('listeprecedentexploitant','PREVIOUS_OPERATOR'),('listeprecedentproprietaire','PREVIOUS_OWNER')):
                for raw in decoded(r.get(field)):
                    p=unwrap(raw);party=dict(role=kind,siren=siren(p),names=names(p))
                    parties.append(party)
                    for n in party['names']:text.append(dict(field=kind+'.name',text=n))
                    if kind=='SUBJECT' and p.get('administration'):
                        text.append(dict(field='administration',text=str(p['administration'])))
            for field in ('acte','modificationsgenerales','divers','radiationaurcs','jugement'):
                text.extend(dict(field=key,text=value) for key,value in strings(r.get(field),field) if len(value)>5)
            for raw in decoded(r.get('listeetablissements')):
                e=unwrap(raw)
                for k in ('enseigne','nomCommercial','denomination','activite'):
                    if e.get(k):text.append(dict(field='establishment.'+k,text=str(e[k])))
                # Published business-place evidence, NOT domicile/representative addresses.
                for key,value in strings(e.get('adresse'),'establishment.address'):
                    text.append(dict(field=key,text=value))
            digits=re.sub('[^0-9]','',parent)
            textual=any(re.search(r'(?<![0-9])'+r'\s*'.join(digits)+r'(?![0-9])',x['text']) for x in text)
            structured=any(p['siren']==parent for p in parties)
            if not structured and not textual:continue # search token hits are not identity evidence
            output.append(dict(id=r['id'],date=r['dateparution'][:10],family=r.get('familleavis_lib'),
                audited_parent=parent,parties=parties,target_structured=structured,target_free_text_mention=textual,
                url=r.get('url_complete') or 'https://www.bodacc.fr/pages/annonces-commerciales-detail/?q.id='+r['id'],texts=text))
        offset+=len(items)
    if len({r['id'] for r in output})!=len(output):raise ValueError('Duplicate source ID')
    return dict(audited_parent=parent,api_total=total,verified_records=len(output),requests=requests,records=output)


def main():
    if OUT.exists():raise ValueError('Fresh immutable diagnosis namespace')
    prev=BASE/'v52/postseal_bodacc_fulltext_20261012_a'
    summary=read(prev/'summary.json')
    if summary['status']!='COMPLETE_DEEP_POSTSEAL_BODACC_TEXT_DOSSIERS_NOT_RETRIEVAL':raise ValueError('Completed source audit')
    cases=read(prev/'cases.json')
    wanted=sorted({r['label']['siren'] for r in cases if not r['protected_not_backfilled']}|set(ANCHORS))
    OUT.mkdir(parents=True);(OUT/'subjects').mkdir()
    write(OUT/'launch.json',dict(implementation=reference(Path(__file__)),source_cases=reference(prev/'cases.json'),
        anchors=ANCHORS,cutoff=CUTOFF,purpose='POSTSEAL_SOURCE_DIAGNOSIS_NEVER_RETRIEVAL',
        no_directors_birth_or_home_address_field_extracted=True,no_private_target_reconstructed=True))
    by_parent={};start=time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(collect,p):p for p in wanted}
        for f in as_completed(futures):
            p=futures[f];value=f.result();write(OUT/'subjects'/(p+'.json'),value);by_parent[p]=value
            print(json.dumps(dict(subject=p,verified=value['verified_records'],api_total=value['api_total'],complete=len(by_parent))),flush=True)
    result=[]
    for r in cases:
        p=r['label']['siren'];qt=tokens(r['crm']['crm_name']);records=by_parent.get(p,{}).get('records',[])
        matches=[]
        for a in records:
            for x in a['texts']:
                overlap=sorted(qt & tokens(x['text']))
                if overlap:matches.append(dict(id=a['id'],date=a['date'],field=x['field'],text=x['text'],shared_tokens=overlap,
                    all_informative_query=bool(qt and qt<=tokens(x['text'])),url=a['url']))
        result.append(dict(query_id=r['query_id'],crm=r['crm'],label=r['label'],protected_not_backfilled=r['protected_not_backfilled'],
            records=len(records),structured_subject_records=sum(any(party['role']=='SUBJECT' and party['siren']==p for party in a['parties']) for a in records),
            reverse_or_other_subject_records=sum(not any(party['role']=='SUBJECT' and party['siren']==p for party in a['parties']) for a in records),
            literal_matches=matches))
    write(OUT/'cases.json',result)
    summary=dict(status='COMPLETE_POSTSEAL_REVERSE_BODACC_TEXT_NOT_RETRIEVAL',misses=31,
        with_source=sum(bool(r['records']) for r in result),
        with_reverse_or_other_subject=sum(bool(r['reverse_or_other_subject_records']) for r in result),
        with_literal_overlap=sum(bool(r['literal_matches']) for r in result),
        cases=reference(OUT/'cases.json'),sources=[reference(OUT/'subjects'/(p+'.json')) for p in wanted],
        implementation=reference(Path(__file__)),elapsed=time.monotonic()-start,
        no_new_recall=True,no_prediction_changed=True,no_training=True,no_closed_fold_opened=True,
        no_global_semantic_absence_claim=True,source_publication_is_not_operation_realization=True)
    write(OUT/'summary.json',summary);print(json.dumps({k:summary[k] for k in ('status','with_source','with_reverse_or_other_subject','with_literal_overlap','elapsed')}),flush=True)


if __name__=='__main__':main()
