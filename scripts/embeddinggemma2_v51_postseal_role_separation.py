"""Evidence discrimination audit, not a hypothetical retrieval evaluation.

Find all corporate role subjects nationally from three query strings, then
inspect ALL their SIRENE sites and the already sealed competitor outputs.
The label is annotated only afterwards; no threshold/weights are fitted.
"""
from collections import defaultdict
from pathlib import Path
import json
import re

import duckdb
import pyarrow as pa

from scripts.embeddinggemma2_v49_postseal_bodacc_roles import BASE, read, write, reference, tokens, norm

OUT=BASE/'v51/postseal_role_separation_20261012_a'
ROLE=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/rne_legal_entity_roles_v1/legal_entity_roles.parquet')
STOCK=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockEtablissement.parquet')
QUERIES=['NEWCRM:40b61ff2e887dae7a9bb0b1ed3b88f19',
         'NEWCRM:77a4735fa9892ba314d70973a2801a00',
         'NEWCRM:83c6f78db686a131bc3e8e00de7b73a0']


def street(s):
    substitutions={'AV':'AVENUE','BD':'BOULEVARD','R':'RUE','RTE':'ROUTE'}
    return ' '.join(substitutions.get(t,t) for t in norm(s).split())


def main():
    if OUT.exists(): raise ValueError('Fresh postseal audit namespace')
    source=BASE/'v32/global_public_parent_names_20261012_a'
    gen=read(source/'generation.json');ex=read(source/'generate_exit.json')
    if gen['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code']!=0 or not ex['own_child_exit_confirmed']:
        raise ValueError('Complete owned output')
    if reference(Path(gen['outputs']['path']))['sha256']!=gen['outputs']['sha256']:
        raise ValueError('Output seal changed')
    cases={r['query_id']:r for r in read(BASE/'v20/deep_audit_20261011_c/rows.json')}
    outcomes=read(source/'outcomes.json')['learned_native_ooc']
    if len(cases)!=526 or any(r['label']['fold'] not in (2,3,4) for r in cases.values()):raise ValueError('Same consumed pilot only')
    emitted=read(source/'outputs.json')['learned_native_ooc']
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    role_rows=c.execute('SELECT subject_siren,related_siren,related_denomination,role_code,active,effective_date,observed_at FROM read_parquet(?)',[str(ROLE)]).fetch_arrow_table().to_pylist()
    relations={q:[] for q in QUERIES}
    for row in role_rows:
        if not re.fullmatch('[0-9]{9}',row['related_siren'] or '') or row['subject_siren']==row['related_siren']:continue
        if str(row['effective_date'] or '')[:10]>'2026-08-01' or str(row['observed_at'] or '')[:10]>'2026-08-01':continue
        name=re.split(r'\b(?:REPRESENTE|REPRESENTEE|REPRESENTES|REPRESENTEES|REPR)\s+(?:PAR|P[.]?)\b',norm(row['related_denomination']),maxsplit=1)[0]
        for q in QUERIES:
            qt=tokens(cases[q]['crm']['crm_name'])
            if qt and qt<=tokens(name):relations[q].append(row)
    parents=sorted({r['subject_siren'] for rows in relations.values() for r in rows})
    c.register('selected_parents',pa.Table.from_pylist([dict(siren=p) for p in parents]))
    sites=c.execute('''SELECT e.siret,e.siren,e.statutDiffusionEtablissement diffusion,
        e.etatAdministratifEtablissement state,e.codeCommuneEtablissement insee,
        e.numeroVoieEtablissement number,e.indiceRepetitionEtablissement suffix,
        e.typeVoieEtablissement street_type,e.libelleVoieEtablissement street_name
        FROM read_parquet(?) e JOIN selected_parents p USING(siren)''',[str(STOCK)]).fetch_arrow_table().to_pylist()
    c.close();del role_rows
    OUT.mkdir(parents=True)
    result=[]
    for q in QUERIES:
        query=cases[q]['crm'];label=cases[q]['label'];owners={r['subject_siren'] for r in relations[q]}
        leaves=[s for s in sites if s['siren'] in owners]
        # No active-only filter: exact truth can be closed.
        same_commune=[s for s in leaves if s['insee']==query['crm_insee']]
        same_street=[s for s in same_commune if s['diffusion']=='O' and
            street(' '.join(str(s[k] or '') for k in ('street_type','street_name')))==street(query['crm_street'])]
        same_number=[s for s in same_street if norm(s['number'])==norm(query['crm_number']) and norm(s['suffix'])==norm(query['crm_suffix']) and query['crm_number']]
        competing=[s for s in emitted[q] if s['siren'] in owners]
        overlap_street=[s for s in competing if s['siret'] in {x['siret'] for x in same_street}]
        item=dict(query_id=q,crm=query,label=label,structured_role_parents=len(owners),
            national_sites=len(leaves),active_sites=sum(s['state']=='A' for s in leaves),closed_sites=sum(s['state']=='F' for s in leaves),
            actor_sirens=sorted({r['related_siren'] for r in relations[q]}),
            target_parent_has_query_role=label['siren'] in owners,
            sites_same_commune=len(same_commune),sites_same_public_current_street=len(same_street),
            sites_same_public_current_number_suffix_street=len(same_number),
            target_same_commune=label['siret'] in {s['siret'] for s in same_commune},
            target_same_public_current_street=label['siret'] in {s['siret'] for s in same_street},
            target_same_public_current_number_suffix_street=label['siret'] in {s['siret'] for s in same_number},
            baseline100_same_role_parents=len(competing),baseline100_same_role_and_public_current_street=len(overlap_street),
            baseline100_cutoff=min(s['score'] for s in emitted[q]),
            local_role_sites=[dict(siret=s['siret'],siren=s['siren'],state=s['state'],
                public_current_street_matches=s['siret'] in {x['siret'] for x in same_street},
                public_current_number_suffix_street_matches=s['siret'] in {x['siret'] for x in same_number}) for s in same_commune],
            role_records=relations[q])
        result.append(item)
    write(OUT/'cases.json',result)
    hypothetical={r['query_id'] for r in result if r['target_parent_has_query_role']}
    # Explicitly label-side arithmetic illustration, never a candidate output.
    total=sum(float(r['calibrated_query_weight']) for r in outcomes)
    count=sum(r['exact'] or r['query_id'] in hypothetical for r in outcomes)
    weighted=sum(float(r['calibrated_query_weight']) for r in outcomes if r['exact'] or r['query_id'] in hypothetical)/total
    summary=dict(status='COMPLETE_POSTSEAL_EVIDENCE_SEPARATION_NOT_RETRIEVAL',queries=len(result),
        cases=reference(OUT/'cases.json'),implementation=reference(Path(__file__)),
        baseline_generation=reference(source/'generation.json'),source_roles=reference(ROLE),
        stock_metadata=dict(path=str(STOCK),size_bytes=STOCK.stat().st_size),
        optimistic_three_cases_only=dict(exact_count=count,denominator=len(outcomes),raw=count/len(outcomes),weighted=weighted,
            not_recall_measurement=True,assumes_no_new_losses=True,not_a_universal_ceiling=True),
        no_feature_or_weight_learned=True,no_candidate_output_created=True,no_prediction_changed=True,
        historical_addresses_not_scanned=True,no_target_specific_source_repair=True,
        source_family_defined_from_query_and_whole_source_not_labels=True)
    write(OUT/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],optimistic_three_cases_only=summary['optimistic_three_cases_only'],
        cases=[{k:r[k] for k in ('crm','structured_role_parents','national_sites','sites_same_commune','sites_same_public_current_street','sites_same_public_current_number_suffix_street','target_same_public_current_street','target_same_public_current_number_suffix_street','baseline100_same_role_parents','baseline100_same_role_and_public_current_street')} for r in result]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
