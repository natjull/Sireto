"""Full public BODACC text dossiers for sealed pilot misses, not retrieval.

Public read-only API backfill restores omitted fields for diagnosis only.
No candidates, predictions, aliases, weights or labels are produced/changed.
Masked target sites/owners are not used to recover withheld identity/address.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import re
import time
import urllib.parse
import urllib.request

from scripts.embeddinggemma2_v49_postseal_bodacc_roles import (
    BASE, read, write, reference, norm, tokens, decoded, unwrap, siren, names)

OUT=BASE/'v52/postseal_bodacc_fulltext_20261012_a'
API='https://bodacc-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/annonces-commerciales/records'
CUTOFF='2026-08-01'


def strings(value, prefix=''):
    if not value:return []
    if isinstance(value,str):
        try:return strings(json.loads(value),prefix)
        except ValueError:return [(prefix,value)]
    if isinstance(value,dict):
        return [p for k,v in value.items() for p in strings(v,(prefix+'.'+k).strip('.'))]
    if isinstance(value,list):
        return [p for i,v in enumerate(value) for p in strings(v,prefix+'.'+str(i))]
    return []


def collect(parent):
    where=f'search(registre, "{parent}") AND dateparution <= date\'{CUTOFF}\''
    output=[];requests=[];offset=0;total=None
    while total is None or offset<total:
        args=dict(where=where,limit=100,offset=offset,order_by='dateparution desc, id asc')
        url=API+'?'+urllib.parse.urlencode(args)
        error=None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=45) as response:body=json.load(response)
                break
            except Exception as e:
                error=e
                if attempt==2:raise
                time.sleep(1+attempt)
        count=int(body['total_count'])
        if total is not None and total!=count:raise ValueError('API total drift during paging')
        total=count;items=body['results']
        if not items and offset<total:raise ValueError('Incomplete API pagination')
        requests.append(dict(url=url,offset=offset,total=total,received=len(items)))
        for r in items:
            if r['dateparution'][:10]>CUTOFF:raise ValueError('Future source')
            people=[unwrap(p) for p in decoded(r.get('listepersonnes'))]
            own=[p for p in people if siren(p)==parent]
            if not own:continue # register match alone cannot attach a claim
            text=[]
            for p in own:
                for name in names(p):text.append(dict(field='subject_name',text=name))
                if p.get('administration'):text.append(dict(field='administration',text=str(p['administration'])))
            for field in ('acte','modificationsgenerales','divers','radiationaurcs','jugement'):
                text.extend(dict(field=key,text=value) for key,value in strings(r.get(field),field) if len(value)>5)
            for establishment in decoded(r.get('listeetablissements')):
                e=unwrap(establishment)
                for key in ('enseigne','nomCommercial','denomination','activite'):
                    if e.get(key):text.append(dict(field='establishment.'+key,text=str(e[key])))
            output.append(dict(id=r['id'],date=r['dateparution'][:10],family=r.get('familleavis_lib'),
                subject_siren=parent,url=r.get('url_complete') or
                'https://www.bodacc.fr/pages/annonces-commerciales-detail/?q.id='+r['id'],texts=text))
        offset+=len(items)
    ids=[r['id'] for r in output]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate ID within subject source')
    return dict(subject_siren=parent,api_total=total,structured_subject_records=len(output),
        requests=requests,records=output)


def main():
    if OUT.exists():raise ValueError('Fresh immutable diagnosis namespace')
    source=BASE/'v32/global_public_parent_names_20261012_a'
    gen=read(source/'generation.json');exit_record=read(source/'generate_exit.json')
    if gen['status']!='COMPLETE_SEALED_ALL526' or exit_record['exit_code']!=0 or not exit_record['own_child_exit_confirmed']:
        raise ValueError('Complete sealed owned source before labels')
    if reference(Path(gen['outputs']['path']))['sha256']!=gen['outputs']['sha256']:raise ValueError('Source seal changed')
    outcomes={r['query_id']:r for r in read(source/'outcomes.json')['learned_native_ooc']}
    rows=[r for r in read(BASE/'v20/deep_audit_20261011_c/rows.json') if not outcomes[r['query_id']]['exact']]
    if len(rows)!=31 or any(r['label']['fold'] not in (2,3,4) for r in rows):raise ValueError('Same all31 safe misses')
    status={r['query_id']:r for r in read(BASE/'v47/postseal_stock_status_20261012_a/cases.json')}
    protected={r['label']['siren'] for r in rows if status[r['query_id']]['site_diffusion']=='P' or status[r['query_id']]['owner_diffusion']=='P'}
    wanted=sorted({r['label']['siren'] for r in rows}-protected)
    OUT.mkdir(parents=True);(OUT/'subjects').mkdir()
    write(OUT/'launch.json',dict(implementation=reference(Path(__file__)),api=API,cutoff=CUTOFF,
        selected_after_seal=True,all_misses=31,protected_subjects_not_backfilled=sorted(protected),
        source_generation=reference(source/'generation.json'),no_retrieval_input=True,
        dates_publication_not_asserted_effective=True,not_historical_snapshot_reproduction=True))
    by_parent={};started=time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures={executor.submit(collect,p):p for p in wanted}
        for future in as_completed(futures):
            p=futures[future];value=future.result();path=OUT/'subjects'/(p+'.json');write(path,value)
            by_parent[p]=value
            print(json.dumps(dict(subject=p,records=value['structured_subject_records'],api_total=value['api_total'],completed=len(by_parent),elapsed=round(time.monotonic()-started,1))),flush=True)
            write(OUT/'state.json',dict(status='PUBLIC_SOURCE_READ_ONLY_AUDIT',subjects=len(by_parent)))
    local={r['query_id']:r for r in read(BASE/'v49/postseal_bodacc_roles_20261012_a/cases.json')}
    result=[]
    for r in rows:
        p=r['label']['siren'];q=r['crm'];qt=tokens(q['crm_name']);records=by_parent.get(p,{}).get('records',[])
        matches=[];event=[];count=0;chars=0
        for ann in records:
            for text in ann['texts']:
                count+=1;chars+=len(text['text']);t=tokens(text['text']);overlap=sorted(qt & t)
                if overlap:
                    matches.append(dict(id=ann['id'],date=ann['date'],url=ann['url'],field=text['field'],
                        shared_tokens=overlap,complete_informative_query=bool(qt and qt<=t),text=text['text']))
                if text['field'].startswith(('acte','modificationsgenerales','divers')) and re.search(r'\b(FUSION|ABSORPTION|TRANSMISSION UNIVERSELLE|SOCIETE MERE|DENOMINATION|ENSEIGNE|EXPLOITANT|LOCATION GERANCE)\b',norm(text['text'])):
                    event.append(dict(id=ann['id'],date=ann['date'],field=text['field'],url=ann['url'],text=text['text']))
        result.append(dict(query_id=r['query_id'],crm=q,label=r['label'],stratum=r['stratum'],
            protected_not_backfilled=p in protected,announcements=len(records),local_announcements=local[r['query_id']]['announcements'],
            text_fragments=count,text_characters=chars,literal_matches=matches,event_passages=event,
            target_site_vs_headquarters_not_inferred=True))
    write(OUT/'cases.json',result)
    summary=dict(status='COMPLETE_DEEP_POSTSEAL_BODACC_TEXT_DOSSIERS_NOT_RETRIEVAL',
        misses=31,distinct_subjects=len(wanted)+len(protected),backfilled_subjects=len(wanted),
        protected_subjects_skipped=len(protected),with_announcements=sum(bool(r['announcements']) for r in result),
        with_literal_match=sum(bool(r['literal_matches']) for r in result),
        with_nonadmin_nonsubject_literal_match=sum(any(m['field'] not in ('administration','subject_name') for m in r['literal_matches']) for r in result),
        with_event_passage=sum(bool(r['event_passages']) for r in result),
        cases=reference(OUT/'cases.json'),sources=[reference(OUT/'subjects'/(p+'.json')) for p in wanted],
        implementation=reference(Path(__file__)),elapsed_seconds=time.monotonic()-started,
        requests=sum(len(v['requests']) for v in by_parent.values()),
        no_new_recall=True,no_positive_candidate_injected=True,no_feature_created=True,no_denominator_filtered=True,
        no_training=True,no_encoder_inference_yet=True,no_closed_fold_opened=True,
        absence_of_literal_overlap_not_semantic_impossibility=True,
        external_data_read_only_public_free=True)
    write(OUT/'summary.json',summary);write(OUT/'state.json',dict(status=summary['status'],summary=reference(OUT/'summary.json')))
    print(json.dumps({k:summary[k] for k in ('status','misses','with_announcements','with_literal_match','with_nonadmin_nonsubject_literal_match','with_event_passage','requests','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
