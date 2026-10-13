"""Post-seal documentary rival audit of ALL26 V81 misses, no GT correction.

Current own public addresses + published historical/current legal/site aliases.
No web-name-to-target backfill, no protected identity/address reconstruction.
Strong rival proof is a review reason, NOT proof the human label is false.
"""
from collections import defaultdict
from pathlib import Path
import sqlite3
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v23_native_features as official
from src.xgb_matcher.retrieval_v20_workbench import record,verify

SOURCE=io.HOME/'v81/indirect_lexical_query_20261013_a'
OUTPUT=io.HOME/'v92/official_rival_audit_20261013_a'


def main():
    import duckdb
    import pyarrow as pa
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh diagnostic only')
    generation=io.read(SOURCE/'generation.json');ex=io.read(SOURCE/'generate_exit.json')
    if generation['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned full source before labels')
    launch=io.read(SOURCE/'launch.json');outputs=io.read(verify(generation['outputs']))['learned_native_ooc']
    cases=io.read(io.HOME/'v85/postseal_proof_gap_20261013_a/cases.json');misses=[r for r in cases if not r['exact']]
    status={r['query_id']:r for r in io.read(io.HOME/'v47/postseal_stock_status_20261012_a/cases.json')}
    selected=[]
    for row in misses:
        s=status[row['query_id']]
        if s['site_diffusion']=='P' or s['owner_diffusion']=='P':continue
        for candidate in outputs[row['query_id']]:selected.append(int(candidate['siret']))
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    c.register('wanted',pa.table({'ref':np.array(sorted(set(selected)),dtype=np.uint64)}))
    data=c.execute("""SELECT e.siret,e.siren,e.codeCommuneEtablissement,e.numeroVoieEtablissement,
        e.indiceRepetitionEtablissement,e.typeVoieEtablissement,e.libelleVoieEtablissement,
        e.etatAdministratifEtablissement,u.denominationUniteLegale
        FROM read_parquet(?) e JOIN wanted w ON CAST(e.siret AS UBIGINT)=w.ref
        JOIN read_parquet(?) u ON e.siren=u.siren
        WHERE e.statutDiffusionEtablissement='O' AND u.statutDiffusionUniteLegale='O'
        AND u.categorieJuridiqueUniteLegale<>'1000'""",[str(mandate.STOCK),str(mandate.audit_utils.STOCK)]).fetch_arrow_table().to_pylist();c.close()
    sites={r['siret']:r for r in data};parents=sorted({int(r['siren']) for r in data})
    db=sqlite3.connect(f"file:{verify(launch['names'])}?mode=ro&immutable=1",uri=True)
    portfolios,legal,site_names=official.load_name_roles(db,parents);db.close()
    normalize=official.structural.normalize;result=[]
    for row in misses:
        q=row['query'];s=status[row['query_id']];protected=s['site_diffusion']=='P' or s['owner_diffusion']=='P'
        review=[];name=normalize(q['crm_name']);street=normalize(q['crm_street'],street=True)
        number=normalize(q['crm_number']);suffix=normalize(q['crm_suffix'])
        if not protected:
            for rank,candidate in enumerate(outputs[row['query_id']],1):
                r=sites.get(candidate['siret'])
                if not r:continue
                names=legal.get(int(r['siren']),())+site_names.get(int(r['siret']),())
                exact_name=bool(name and name in names)
                st=normalize(' '.join(str(r[k] or '') for k in ('typeVoieEtablissement','libelleVoieEtablissement')),street=True)
                exact_street=bool(street and street==st and q['crm_insee'] and q['crm_insee']==r['codeCommuneEtablissement'])
                n=normalize(r['numeroVoieEtablissement'] or '');su=normalize(r['indiceRepetitionEtablissement'] or '')
                exact_house=bool(exact_street and number and number==n and suffix==su)
                if exact_name or exact_house:
                    review.append(dict(rank=rank,candidate=candidate,published_current_legal_name=r['denominationUniteLegale'],
                        published_own_current_address=dict(insee=r['codeCommuneEtablissement'],number=n,suffix=su,street=st),
                        exact_published_own_or_legal_name=exact_name,exact_current_own_street_commune=exact_street,
                        exact_current_own_number_suffix_street_commune=exact_house,
                        current_site_state=r['etatAdministratifEtablissement'],
                        same_as_human_siren=candidate['siren']==row['label_siret'][:9],
                        aliases_may_be_historical_not_asserted_current=True))
        result.append(dict(query_id=row['query_id'],query=q,label_siret=row['label_siret'],
            protected_no_backfill=protected,documentary_rivals=review,
            contradictory_exact_name_address_other_owner=sum(r['exact_published_own_or_legal_name'] and r['exact_current_own_number_suffix_street_commune'] and not r['same_as_human_siren'] for r in review),
            label_error_proven=False,no_label_or_denominator_modified=True))
    OUTPUT.mkdir(parents=True);io.write(OUTPUT/'cases.json',result)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_OFFICIAL_RIVAL_DIAGNOSIS_NOT_LABEL_CORRECTION',
        implementation=record(Path(__file__)),source_generation=record(SOURCE/'generation.json'),
        owned_source_exit=record(SOURCE/'generate_exit.json'),source_outcomes=record(SOURCE/'outcomes.json'),
        names=launch['names'],stock_sites=record(mandate.STOCK),stock_owners=record(mandate.audit_utils.STOCK),
        cases=record(OUTPUT/'cases.json'),all26_misses_retained=True,primary526_denominator_unchanged=True,
        protected_no_backfill=sum(r['protected_no_backfill'] for r in result),
        queries_with_documentary_exact_other_owner=sum(r['contradictory_exact_name_address_other_owner']>0 for r in result),
        label_errors_proven=0,no_new_recall_or_prediction=True,observed_unix=time.time()))
    print([(r['query']['crm_name'],r['contradictory_exact_name_address_other_owner']) for r in result],flush=True)


if __name__=='__main__':main()
