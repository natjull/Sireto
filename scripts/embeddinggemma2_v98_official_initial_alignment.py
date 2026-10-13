"""Whole-source letter/initial alignment, not a generated company alias.

Initials are a derived similarity feature over published legal-name tokens;
optional city tokens are supported by the candidate's OWN public current site.
No identifier as text, CRM/GT alias learning, guessed acronym or name rewrite.
"""
from collections import defaultdict
import csv
from pathlib import Path
import re
import sqlite3
import time
import urllib.request
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v23_native_features as original
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v98/official_initial_alignment_20261013_b'
SOURCE=io.HOME/'v81/indirect_lexical_query_20261013_a'
COG='https://www.insee.fr/fr/statistiques/fichier/8740222/v_commune_2026.csv'
STOP=set('ASS ASSOCIATION ASSOC SOC SOCIETE ETS ETABLISSEMENTS DE DU DES D L LA LE LES ET EN AU AUX A'.split())|original.structural.LEGAL


def words(value):
    return [w for w in original.structural.normalize(original.data.safe(value)).split() if w not in STOP and w[0].isalpha()]


def main():
    import duckdb
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh source-derived feature namespace')
    OUTPUT.mkdir(parents=True);launch=io.read(SOURCE/'launch.json');panel=io.read(verify(launch['panel']))
    by_token=defaultdict(set);cities_needed={q['crm_insee'] for q in panel if q['crm_insee']}
    for q in panel:
        for w in words(q['crm_name']):
            if re.fullmatch('[A-Z]{2,8}',w):by_token[w].add(q['query_id'])
    path=OUTPUT/'v_commune_2026.csv'
    with urllib.request.urlopen(COG,timeout=30) as response:path.write_bytes(response.read())
    city={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'COM','TYPECOM','LIBELLE'}<=set(reader.fieldnames):raise ValueError('Actual official COG schema')
        for row in reader:
            if row['COM'] in cities_needed and row['TYPECOM'] in ('COM','ARM','COMA','COMD'):
                city[row['COM']]=''.join(w[0] for w in words(row['LIBELLE']))
    engine.seal(path)
    # Missing COG city disables only this extra cue, never original retrieval.
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    eligible={int(r[0]) for r in c.execute("SELECT siren FROM read_parquet(?) WHERE statutDiffusionUniteLegale='O' AND categorieJuridiqueUniteLegale<>'1000'",[str(mandate.audit_utils.STOCK)]).fetchall()}
    direct=defaultdict(dict);with_city=defaultdict(dict);alias_rows=0;read_rows=0
    db=sqlite3.connect(f"file:{verify(launch['names'])}?mode=ro&immutable=1",uri=True)
    cursor=db.execute('SELECT owner_key,name FROM aliases WHERE kind=1')
    qcity={q['query_id']:q['crm_insee'] for q in panel}
    city_matches=defaultdict(list)
    for cm,suffix in city.items():
        for token in by_token:
            if suffix and len(token)>len(suffix) and token.endswith(suffix):city_matches[token[:-len(suffix)]].append((cm,token))
    while True:
        rows=cursor.fetchmany(8192)
        if not rows:break
        mandate.guard(mandate.DEADLINE)
        for parent,name in rows:
            read_rows+=1
            if parent not in eligible:continue
            alias_rows+=1;initial=''.join(w[0] for w in words(name))
            if not initial:continue
            # All matching windows; no topk parent/alias pruning.
            found={initial:1.} if initial in by_token else {}
            for start in range(len(initial)):
                for end in range(start+3,min(len(initial),start+8)+1):
                    token=initial[start:end]
                    if token in by_token:found[token]=max(found.get(token,0.),.5)
            for token,strength in found.items():
                for qid in by_token[token]:direct[qid][parent]=max(direct[qid].get(parent,0.),strength)
            for cm,token in city_matches.get(initial,()):
                for qid in by_token[token]:
                    if qcity[qid]==cm:with_city[qid][parent]=.5
        if read_rows%(8192*32)==0:io.write(OUTPUT/'state.json',dict(status='ALL_SOURCE_LEGAL_INITIAL_ALIGNMENT',read_rows=read_rows,public_corporate_aliases=alias_rows,updated_unix=time.time()))
    db.close()
    # Site lists are source matches to each query's named city, not GT lookups.
    parent_city=defaultdict(set)
    for qid,parents in with_city.items():parent_city[qcity[qid]].update(parents)
    own_city=defaultdict(list)
    if parent_city:
        rows=c.execute("SELECT siret,siren,codeCommuneEtablissement FROM read_parquet(?) WHERE statutDiffusionEtablissement='O' AND codeCommuneEtablissement IN (SELECT unnest(?))",[str(mandate.STOCK),sorted(parent_city)]).fetch_record_batch(rows_per_batch=8192)
        for batch in rows:
            mandate.guard(mandate.DEADLINE)
            for r in batch.to_pylist():
                p=int(r['siren']);cm=r['codeCommuneEtablissement']
                if p in parent_city[cm]:own_city[(p,cm)].append(r['siret'])
    c.close();result={};N=max(1,len(eligible))
    for q in panel:
        qid=q['query_id'];parents=direct.get(qid,{})
        matched_city={p:v for p,v in with_city.get(qid,{}).items() if own_city.get((p,q['crm_insee']))}
        df=len(set(parents)|set(matched_city));idf=np.log1p(N/max(1,df))/np.log1p(N)
        result[qid]=dict(parent_initials=[dict(siren_ref=str(p).zfill(9),confidence=float(v*idf)) for p,v in sorted(parents.items())],
            own_city_initials=[dict(siret_refs=sorted(own_city[(p,q['crm_insee'])]),confidence=float(v*idf),source_commune=q['crm_insee']) for p,v in sorted(matched_city.items())],
            derived_token_alignment_not_declared_alias=True,whole_source_no_topk=True)
    io.write(OUTPUT/'initial_alignment.json',result);engine.seal(OUTPUT/'initial_alignment.json')
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_WHOLE_SOURCE_INITIAL_ALIGNMENT',implementation=record(Path(__file__)),
        query_panel=launch['panel'],published_legal_names=launch['names'],public_owner_stock=record(mandate.audit_utils.STOCK),
        public_site_stock=record(mandate.STOCK),cog=record(path),cog_url=COG,cog_reference_year=2026,
        alignment=record(OUTPUT/'initial_alignment.json'),scanned_legal_alias_rows=read_rows,eligible_public_corporate_alias_rows=alias_rows,
        eligible_public_corporate_owners=N,queries_with_alignment=sum(bool(v['parent_initials'] or v['own_city_initials']) for v in result.values()),
        unknown_cog_query_communes=sorted(cities_needed-set(city)),no_alias_text_created=True,
        no_contact_or_physical_person_identity_used=True,no_ids_as_similarity_features=True,labels_read=False,
        source_policy='DIAGNOSTIC_ONLY_DERIVED_NORMALIZATION_FEATURE_NO_PROMOTION'))
    print('complete',read_rows,alias_rows,sum(bool(v['parent_initials'] or v['own_city_initials']) for v in result.values()),flush=True)


if __name__=='__main__':main()

