"""Versioned projection of official RNE powers, preserving site/date fields.

INPI JSON v4.0 documents etablissementConcerne[].siret, dateEffet and
entreprise.libelleRoleEntreprise. The old projection omitted these keys.
These roles remain contextual: their presence does not establish identity.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

from src.xgb_matcher.rne_legal_entity_roles import _branches, extract_legal_entity_roles, normalize_text
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json

SOURCE = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/official_sync/rne-ftp-bulk/6882485c77a98d8f')
ZIP = SOURCE/'stock_RNE_formalites_NIVEAU1_20260304_1400.zip'
ZIP_SHA = '87daf4d13eef5c8610d77cf9d245deb0922c84db5ae95aa62af1e03906c76218'
DOC = 'https://www.inpi.fr/sites/default/files/2025-06/documentation%20technique%20API%20formalit%C3%A9s_v4.0.pdf'


def identifier(value, size):
    value=str(value or '').strip()
    return value if re.fullmatch('[0-9]{'+str(size)+'}',value) else ''


def project(record_value):
    root=record_value.get('formality') or record_value.get('formalite') or record_value
    subject=identifier(record_value.get('siren') or root.get('siren'),9)
    source_id=str(record_value.get('id') or root.get('id') or '')
    if not subject or not source_id: return ()
    found={}
    for branch in _branches(record_value):
        for ordinal,power in enumerate((branch.get('composition') or {}).get('pouvoirs') or []):
            if not isinstance(power,dict): continue
            enterprise=power.get('entreprise') or {}
            if not isinstance(enterprise,dict) or not enterprise: continue
            denomination=str(enterprise.get('denomination') or enterprise.get('nomEntreprise') or power.get('denomination') or '').strip()
            if not denomination: continue
            description=enterprise.get('descriptionPersonne') or {}
            related=identifier(enterprise.get('siren') or description.get('siren') or power.get('siren'),9)
            if related==subject: continue
            sites=[]
            for key in ('etablissementConcerne','etablissementsConcernes','etablissements'):
                values=power.get(key) or []
                if isinstance(values,dict): values=[values]
                if not isinstance(values,list): continue
                for item in values:
                    data=item if isinstance(item,dict) else dict(siret=item)
                    siret=identifier(data.get('siret'),14)
                    if siret:
                        sites.append(dict(siret=siret,date=data.get('dateEffetEtablissementConcerne'),source_field=key))
            for key in ('siretEtablissement','siretEtablissementConcerne','siretEtablissementRattachement'):
                siret=identifier(power.get(key),14)
                if siret: sites.append(dict(siret=siret,date=None,source_field=key))
            canonical=dict(source='RNE',source_record_id=f'{source_id}:power:{ordinal}',subject_siren=subject,
                related_siren=related,related_denomination=denomination,normalized_denomination=normalize_text(denomination),
                role_code=str(power.get('roleEntreprise') or enterprise.get('roleEntreprise') or ''),
                official_role_label=str(enterprise.get('libelleRoleEntreprise') or power.get('libelleRoleEntreprise') or ''),
                free_role_label=str(power.get('autreRoleEntreprise') or enterprise.get('autreRoleEntreprise') or ''),
                role_effective_date=power.get('dateEffet'),
                legacy_role_date=power.get('dateEffetRoleDeclarant') or enterprise.get('dateEffetRoleDeclarant'),
                active=power.get('actif') if isinstance(power.get('actif'),bool) else None,
                site_evidence=sorted(sites,key=lambda x:(x['siret'],x['source_field'],str(x['date']))),
                observed_at=record_value.get('updatedAt') or root.get('updatedAt'),
                navigation='CONTEXT_ONLY_ROLE_MEANING_AND_OFFICIAL_MEMBERSHIP_UNVERIFIED')
            proof_id=hashlib.sha256(json.dumps(canonical,sort_keys=True,default=str).encode()).hexdigest()
            found[proof_id]=dict(proof_id=proof_id,**canonical)
    return tuple(found[k] for k in sorted(found))


def audit(output):
    import ijson
    output.mkdir(parents=True,exist_ok=False)
    budget=Budget(dict(max_seconds=90,max_rss_bytes=1024**3,max_swapout_pages_delta=0))
    try:
        with budget:
            verify(dict(path=ZIP,sha256=ZIP_SHA))
            proofs=[]; legacy=[]; members=[]; records=0
            with zipfile.ZipFile(ZIP) as archive:
                infos=sorted([i for i in archive.infolist() if not i.is_dir() and i.filename.endswith('.json')],
                    key=lambda i:hashlib.sha256(i.filename.encode()).hexdigest())[:2]
                for info in infos:
                    # Fully read selected members: ZIP CRC and per-member hash are checked.
                    data=archive.read(info)
                    members.append(dict(name=info.filename,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                    for value in ijson.items(data,'item'):
                        if not isinstance(value,dict): continue
                        records+=1; proofs.extend(project(value))
                        legacy.extend(r.to_dict() for r in extract_legal_entity_roles(value))
            write_json(output/'context_proofs.json',proofs)
        stats=dict(source_records=records,projected_roles=len(proofs),legacy_roles=len(legacy),
            projected_site_anchored_roles=sum(bool(r['site_evidence']) for r in proofs),
            legacy_site_anchored_roles=sum(bool(r['establishment_sirets']) for r in legacy),
            projected_official_label_roles=sum(bool(r['official_role_label']) for r in proofs),
            projected_effective_date_roles=sum(bool(r['role_effective_date']) for r in proofs),
            legacy_effective_date_roles=sum(bool(r['effective_date']) for r in legacy),
            role_codes=dict(Counter(r['role_code'] for r in proofs)))
        write_json(output/'manifest.json',dict(status='SEALED_RNE_FIELD_PROJECTION_DIAGNOSTIC',stats=stats,
            resources=budget.report(),source_zip=dict(path=str(ZIP),sha256=ZIP_SHA,size_bytes=ZIP.stat().st_size),selected_members=members,
            member_selection='FIRST_TWO_SHA256_MEMBER_FILENAMES_WITHOUT_CRM_OR_LABELS',
            implementation=record(Path(__file__)),official_documentation=DOC,
            artifact=record(output/'context_proofs.json'),truth_opened=False,crm_opened=False,
            search_corpus=False,national_projection=False,retrieval_performed=False,ready_for_fold1=False))
        print(json.dumps(dict(stats=stats,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),ready_for_fold1=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output)
