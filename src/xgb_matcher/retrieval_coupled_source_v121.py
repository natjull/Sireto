"""Explicit official source adapter for the V121 COST sample, not national.

No truth/CRM selection. Parent names join only their official members. Site
names remain local. One-hop succession text is a graded hypothesis, never
an exact-current alias; no arbitrary alias/member/edge cap.
"""
from collections import defaultdict
import re

import numpy as np
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_coupled_blocks_v121 import Proof, Site
from src.xgb_matcher.retrieval_variants_v118 import clean


def family_batches(path, rows_per_family, families=('SIRENE_CURRENT','BODACC','RNE')):
    """Select by official source-family footer, never a query or truth lookup."""
    if not isinstance(rows_per_family,int) or isinstance(rows_per_family,bool) or rows_per_family<=0:
        raise ValueError('Positive physical family sample limit required')
    file=pq.ParquetFile(path);column=file.schema_arrow.names.index('source');groups=defaultdict(list)
    offsets=[0]
    for g in range(file.metadata.num_row_groups):
        offsets.append(offsets[-1]+file.metadata.row_group(g).num_rows)
        stats=file.metadata.row_group(g).column(column).statistics
        if stats is not None and stats.has_min_max and stats.min==stats.max:
            groups[str(stats.min)].append(g)
    for family in families:
        if not groups[family]:
            raise ValueError('No pure physical source-family group: '+family)
        group=groups[family][len(groups[family])//2]
        batch=next(file.iter_batches(batch_size=rows_per_family,row_groups=[group],use_threads=False))
        yield family,group,offsets[group],batch


def source_coordinate(row):
    # Same validated metropolitan CRS scope as V43; all other sites remain
    # indexed, without a coordinate feature. No old name gets a fake old XY.
    insee=clean(row.get('codeCommuneEtablissement'))
    if not insee or insee.startswith(('97','98')):
        return None
    try:
        x=float(row.get('coordonneeLambertAbscisseEtablissement'))
        y=float(row.get('coordonneeLambertOrdonneeEtablissement'))
    except (TypeError,ValueError):
        return None
    return (x,y) if np.isfinite([x,y]).all() and 0<=x<=1300000 and 6000000<=y<=7300000 else None


def public_code(value):
    value=clean(value).upper()
    return value if re.fullmatch('[0-9A-Z]{5}',value) else ''


def build_sites(current_rows, names, facts, observation_rows, extra_name_rows=(), address_rows=(), edges=()):
    pool={int(r['name_id']):r['name_text'] for r in names}
    current={int(r['fact_id']) for r in observation_rows if r['fact_id'] is not None and r['is_current'] is True}
    parents=defaultdict(list);endpoints=defaultdict(list)
    for fact in facts:
        ref=int(fact['fact_id']);proof=Proof(pool[int(fact['name_id'])],current_exact=ref in current,
                                          reference=('V120_FACT_ALL_OBSERVATIONS',ref))
        (parents if fact['subject_kind']=='SIREN' else endpoints)[fact['siren'] if fact['subject_kind']=='SIREN' else fact['siret']].append(proof)
    for row in extra_name_rows:
        if not row['lexical_eligible']:
            continue
        proof=Proof(row['name_text'],current_exact=row['is_current'] is True,
                    reference=('FULL_SOURCE_FAMILY_NAME',row['source'],row['source_row_ordinal'],tuple(row['source_fields'])))
        (parents if row['subject_kind']=='SIREN' else endpoints)[row['siren'] if row['subject_kind']=='SIREN' else row['siret']].append(proof)
    addresses=defaultdict(list);context_addresses=0
    for row in address_rows:
        # Never spread a SIREN address across all of its establishments.
        parent=clean(row.get('siren'));sid=clean(row.get('siret'))
        if row.get('subject_kind')!='SIRET' or len(sid)!=14 or not sid.isascii() or not sid.isdecimal() or not sid.startswith(parent):
            context_addresses+=1;continue
        text=row.get('raw_value') or row.get('normalized_value')
        addresses[sid].append(Proof(clean(text),current_exact=row.get('is_current') is True,
            reference=('FULL_SOURCE_FAMILY_ADDRESS',row.get('source'),row['source_row_ordinal'])))
    links=defaultdict(list)
    for edge in edges:
        a=clean(edge.get('siretEtablissementPredecesseur'));b=clean(edge.get('siretEtablissementSuccesseur'))
        if any(len(s)!=14 or not s.isascii() or not s.isdecimal() for s in (a,b)) or a==b:
            raise ValueError('Invalid explicit official succession endpoints')
        grade=.5 if edge.get('continuiteEconomique') is True else .25
        ref=('SIRENE_SUCCESSION',a,b,clean(edge.get('dateLienSuccession')))
        links[a].append((b,grade,ref));links[b].append((a,grade,ref))
    # Classes are an explicit local (nomenclature, code) dictionary. No activity
    # prediction is applied by this cost probe; the full raw pair remains in
    # original snapshots and this metadata, UNKNOWN has no classifier evidence.
    classes={};sites=[];related_proofs=0
    for row in current_rows:
        sid=clean(row['siret']);parent=clean(row['siren']);native_names=[*parents[parent],*endpoints[sid]]
        for neighbour,grade,ref in links[sid]:
            for proof in (*parents[neighbour[:9]],*endpoints[neighbour]):
                native_names.append(Proof(proof.text,grade=proof.grade*grade,current_exact=False,
                    reference=(*ref,proof.reference),relation_hypothesis=True));related_proofs+=1
        site_addresses=list(addresses[sid])
        for suffix in ('','2'):
            fields=[f'numeroVoie{suffix}Etablissement',f'indiceRepetition{suffix}Etablissement',
                    f'typeVoie{suffix}Etablissement',f'libelleVoie{suffix}Etablissement']
            text=' '.join(clean(row.get(field)) for field in fields).strip()
            if text:
                site_addresses.append(Proof(text,current_exact=not suffix,
                    reference=('SIRENE_CURRENT_ADDRESS_FIELDS',sid,tuple(fields))))
        pair=(clean(row.get('nomenclatureActivitePrincipaleEtablissement')),clean(row.get('activitePrincipaleEtablissement')))
        code=-1
        if all(pair) and all('[ND]' not in field.upper() for field in pair):
            if pair not in classes:
                classes[pair]=len(classes)
            code=classes[pair]
        hq=row.get('etablissementSiege')
        if hq is not None and not isinstance(hq,bool):
            raise ValueError('Uninterpreted source headquarters flag')
        sites.append(Site(sid,parent,tuple(native_names),tuple(site_addresses),
            public_code(row.get('codePostalEtablissement')),public_code(row.get('codeCommuneEtablissement')),
            clean(row.get('numeroVoieEtablissement')),hq,source_coordinate(row),code,
            clean(row.get('etatAdministratifEtablissement')) or None))
    return sites,dict(sites=len(sites),with_geometry=sum(s.xy is not None for s in sites),
        state_counts={state if state is not None else 'UNKNOWN':sum(s.state==state for s in sites) for state in set(s.state for s in sites)},
        official_parent_name_facts=sum(map(len,parents.values())),site_name_facts=sum(map(len,endpoints.values())),
        name_proof_attachments=sum(len(s.names) for s in sites),address_proof_attachments=sum(len(s.addresses) for s in sites),
        context_only_addresses_not_spread_to_sites=context_addresses,relation_hypothesis_attachments=related_proofs,
        activity_pairs=[dict(nomenclature=nom,activity=code,class_id=i) for (nom,code),i in classes.items()],
        relation_hops=1,member_or_alias_cap=None,national_membership_coverage=False)
