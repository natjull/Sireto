"""Selective raw RNE role audit of already-consumed, sealed pilot misses.

ZIP member ordering is only a lookup heuristic, not a source-absence proof.
All matches require exact structured subject SIREN; five neighboring members
are inspected around each binary lookup. No protected identities, beneficial
owners, birth information or home addresses are extracted into this audit.
"""
from bisect import bisect_left
from collections import Counter
from pathlib import Path
import json
import re
import zipfile

from scripts.embeddinggemma2_v49_postseal_bodacc_roles import BASE, read, write, reference, tokens

OUT=BASE/'v50/postseal_rne_directors_20261012_a'
ARCHIVE=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/official_sync/rne-ftp-bulk/6882485c77a98d8f/stock_RNE_formalites_NIVEAU1_20260304_1400.zip')


def main():
    if OUT.exists(): raise ValueError('Fresh audit namespace')
    source=BASE/'v49/postseal_bodacc_roles_20261012_a'
    summary=read(source/'summary.json')
    if summary['status']!='COMPLETE_POSTSEAL_SOURCE_AUDIT_NOT_RETRIEVAL': raise ValueError('Completed post-seal audit required')
    cases=[r for r in read(source/'cases.json') if not r['exact_hit']]
    if len(cases)!=31: raise ValueError('All31 misses retained')
    wanted={r['siren'] for r in cases}; protected={r['siren'] for r in cases if r['owner_diffusion']=='P'}
    found={}; probes={}; members_read=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        infos=sorted(i.filename for i in z.infolist() if i.filename.endswith('.json'))
        decoder=json.JSONDecoder()
        def first(index):
            if index in probes:return probes[index]
            with z.open(infos[index]) as f:
                buf=b''
                while len(buf)<1024*1024:
                    buf+=f.read(16384)
                    text=buf.decode('utf-8',errors='ignore').lstrip()[1:].lstrip()
                    try:r,end=decoder.raw_decode(text);break
                    except ValueError:continue
                else:raise ValueError('First source record not decoded')
            probes[index]=str(r.get('siren',r.get('formality',{}).get('siren','')))
            return probes[index]
        inspect=set()
        for target in sorted(wanted):
            lo,hi=0,len(infos)
            while lo<hi:
                mid=(lo+hi)//2
                if first(mid)<target:lo=mid+1
                else:hi=mid
            inspect.update(range(max(0,lo-3),min(len(infos),lo+3)))
        for index in sorted(inspect):
            payload=z.read(infos[index]);rows=json.loads(payload)
            ids=[str(r.get('siren',r.get('formality',{}).get('siren',''))) for r in rows]
            members_read.append(dict(member=infos[index],records=len(rows),min_siren=min(ids),max_siren=max(ids)))
            for r,s in zip(rows,ids):
                if s not in wanted:continue
                if s in found:raise ValueError('Duplicate RNE subject: do not silently choose')
                roles=[];root=r.get('formality',r)
                if s not in protected:
                    for kind in ('personneMorale','personnePhysique','exploitation'):
                        for p in root.get('content',{}).get(kind,{}).get('composition',{}).get('pouvoirs',[]):
                            if not isinstance(p,dict):continue
                            company=p.get('entreprise') or {}
                            individual=p.get('individu') or {}
                            description=individual.get('descriptionPersonne') or {}
                            entity_type=str(p.get('typeDePersonne') or '')
                            if company:
                                name=str(company.get('denomination') or company.get('nomEntreprise') or '')
                            elif individual:
                                name=' '.join(str(description.get(k) or '') for k in ('nom','nomUsage','prenoms','prenom')).strip()
                            else:name=''
                            if not name:continue
                            roles.append(dict(entity_type=entity_type,name=name,
                                role_code=str(p.get('roleEntreprise') or ''),
                                other_role=str(p.get('autreRoleEntreprise') or ''),
                                active=p.get('actif'),effective_date=str(p.get('dateEffetRoleDeclarant') or '')[:10]))
                found[s]=dict(member=infos[index],record_id=str(r.get('id') or root.get('id') or ''),
                    observed_at=str(r.get('updatedAt') or '')[:10],roles=roles,
                    protected_identity_skipped=s in protected)
            print(json.dumps(dict(member=infos[index],found_subjects=len(found))),flush=True)
    OUT.mkdir(parents=True)
    result=[]
    for c in cases:
        item=dict(query_id=c['query_id'],crm_name=c['crm_name'],siren=c['siren'],
            source_subject_found=c['siren'] in found,source=found.get(c['siren']),matches=[])
        q=tokens(c['crm_name'])
        for role in (item['source'] or {}).get('roles',[]):
            overlap=sorted(q & tokens(role['name']))
            if overlap:item['matches'].append(dict(role,shared_tokens=overlap,complete_informative_query=bool(q and q<=tokens(role['name']))))
        result.append(item)
    write(OUT/'cases.json',result)
    report=dict(status='COMPLETE_SELECTIVE_LABEL_SIDE_RNE_ROLE_INSPECTION_NOT_RETRIEVAL',
        misses=31,distinct_subjects=len(wanted),found_queries=sum(r['source_subject_found'] for r in result),
        found_subjects=len(found),protected_subjects_skipped=len(protected),
        with_roles=sum(bool((r['source'] or {}).get('roles')) for r in result),
        with_any_overlap=sum(bool(r['matches']) for r in result),
        with_full_query_overlap=sum(any(m['complete_informative_query'] for m in r['matches']) for r in result),
        cases=reference(OUT/'cases.json'),implementation=reference(Path(__file__)),
        archive=dict(path=str(ARCHIVE),size_bytes=ARCHIVE.stat().st_size),
        source_manifest=reference(ARCHIVE.parent/'manifest.json'),members_read=members_read,
        method='Binary heuristic on first record SIREN, six neighboring ZIP members; exact structured subject check',
        absence_is_not_claimed=True,no_retrieval_change=True,no_label_edit=True,no_birth_or_home_address_extracted=True,
        no_beneficial_owner_read=True,no_personal_names_as_company_aliases=True)
    write(OUT/'summary.json',report)
    print(json.dumps({k:report[k] for k in ('status','misses','distinct_subjects','found_queries','found_subjects','with_roles','with_any_overlap','with_full_query_overlap')}),flush=True)


if __name__=='__main__':main()
