"""Query-only sector facilities with corroborated own physical address.

Two distinct future native contexts: source SIRET own-facility evidence, and
its explicit registered owner's current public HQ as an administrative option.
Never copy facility name/address to HQ or assert the HQ is the same site.
"""
from collections import defaultdict
from pathlib import Path
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v2_data as formatting
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as names
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v118/addressed_facility_context_20261013_a'
BASE=io.HOME/'v99/initial_alignment_native_20261013_a'
STOCK=names.STOCK.with_name('StockEtablissement.parquet')


def main():
    import duckdb
    import pyarrow as pa
    engine.guard()
    if OUTPUT.exists():raise ValueError('Fresh source context namespace')
    launch=io.read(BASE/'launch.json');source=verify(launch['sector_source']);panel=io.read(verify(launch['panel']))
    if len(panel)!=526:raise ValueError('All same query-only panel')
    OUTPUT.mkdir(parents=True);(OUTPUT/'spill').mkdir()
    recipe=dict(hypothesis='EXPLICIT_FACILITY_NAME_PLUS_OWN_CURRENT_HOUSE_STREET_COMMUNE_CORROBORATION',
        informative_query_words_required='MIN_TWO_OR_SINGLE_WHOLE_SOURCE_NAME',
        source_role='SOURCE_DECLARED_SIRET_FACILITY_NOT_MANAGER_IDENTITY',
        address='SOURCE_OWN_AND_CURRENT_STOCK_OWN_BOTH_AGREE_WITH_QUERY_HOUSE_STREET_COMMUNE',
        source_date_cutoff='2026-08-01_UNKNOWN_RETAINED',facility_confidence=1.,administrative_hq_confidence=.5,
        HQ_current_explicit_stock_flag_required=True,no_forced_hq_slot_or_source_address_inheritance=True,
        IDs_are_references_only=True,no_query_labels_read=True,no_new_recall=True)
    io.write(OUTPUT/'recipe.json',recipe)
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='1GB'")
    con.execute('SET temp_directory=?',[str(OUTPUT/'spill')])
    con.read_parquet(str(source)).create_view('sector')
    # Ownership always follows explicit site prefix; reported manager is not
    # substituted for the real owner and isn't used as an own-name source.
    sites=con.execute("""SELECT s.source,s.source_record_id,s.siret,s.name,s.address,s.insee,s.observed_at,
        e.codeCommuneEtablissement current_insee,e.numeroVoieEtablissement number,
        e.indiceRepetitionEtablissement suffix,e.typeVoieEtablissement street_type,e.libelleVoieEtablissement street,
        e.siren FROM sector s JOIN read_parquet(?) e ON s.siret=e.siret
        JOIN read_parquet(?) u ON e.siren=u.siren
        WHERE s.subject_kind='SIRET' AND e.statutDiffusionEtablissement='O'
        AND u.statutDiffusionUniteLegale='O' AND u.categorieJuridiqueUniteLegale<>'1000'
        AND (coalesce(s.observed_at,'')='' OR s.observed_at<='2026-08-01')""",
        [str(STOCK),str(names.STOCK)]).fetch_arrow_table().to_pylist()
    query_index=defaultdict(list)
    for q in panel:
        keys=formatting.query_site_keys(q)
        for key in keys:
            if q['crm_insee']:query_index[(q['crm_insee'],key)].append(q)
    matched={q['query_id']:{} for q in panel};parents=set()
    for site in sites:
        source_key=formatting.street_key(site['address'])
        current_key=formatting.street_key(' '.join(str(site[k] or '') for k in ('number','suffix','street_type','street')))
        if not source_key or source_key!=current_key or site['insee']!=site['current_insee']:continue
        for q in query_index.get((site['current_insee'],current_key),[]):
            qt=names.tokens(q['crm_name']);st=names.tokens(site['name'])
            if not qt or not qt<=st or (len(qt)<2 and qt!=st):continue
            item=matched[q['query_id']].setdefault(site['siret'],dict(siret_ref=site['siret'],owner_siren_ref=site['siren'],
                confidence=1.,source_evidence=[],no_alias_or_address_inherited=True))
            item['source_evidence'].append(dict(source=site['source'],record_id=site['source_record_id'],
                published_facility_name=site['name'],date=site['observed_at'],name_and_own_address_corroborated=True))
            parents.add(site['siren'])
    if parents:
        con.register('owners',pa.table({'siren':sorted(parents)}))
        hqs=con.execute("""SELECT e.siret,e.siren FROM read_parquet(?) e JOIN owners w USING(siren)
            WHERE e.etablissementSiege AND e.statutDiffusionEtablissement='O'""",[str(STOCK)]).fetchall()
    else:hqs=[]
    con.close();by_owner=defaultdict(list)
    for ref,owner in hqs:by_owner[owner].append(ref)
    output={}
    for qid,records in matched.items():
        own=list(records.values());administrative={}
        for r in own:
            for hq in by_owner[r['owner_siren_ref']]:
                if hq in records:continue
                entry=administrative.setdefault(hq,dict(siret_ref=hq,confidence=.5,kind='ADMINISTRATIVE_OWNER_HQ_ALTERNATIVE',
                    facility_source_refs=[],not_same_physical_site_claim=True,not_facility_own_alias=True))
                entry['facility_source_refs'].append(r['siret_ref'])
        output[qid]=dict(own_facilities=own,administrative_hq_alternatives=list(administrative.values()))
    io.write(OUTPUT/'context.json',output);engine.seal(OUTPUT/'context.json')
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_WHOLE_SOURCE_ADDRESSED_FACILITY_CONTEXT_QUERY_ONLY',
        implementation=record(Path(__file__)),recipe=record(OUTPUT/'recipe.json'),query_panel=launch['panel'],
        sector_source=launch['sector_source'],stock_sites=record(STOCK),stock_owners=record(names.STOCK),
        context=record(OUTPUT/'context.json'),public_source_site_records=len(sites),all_queries=526,
        queries_with_corroborated_facility=sum(bool(v['own_facilities']) for v in output.values()),
        facility_references=sum(len(v['own_facilities']) for v in output.values()),
        headquarters_alternatives=sum(len(v['administrative_hq_alternatives']) for v in output.values()),
        no_query_or_denominator_filtered=True,no_gt_or_retrieval_change=True,source_roles_separate=True,
        no_production_or_cutoff_admission_granted=True,observed_unix=time.time()))
    print('COMPLETE',sum(bool(v['own_facilities']) for v in output.values()),sum(len(v['administrative_hq_alternatives']) for v in output.values()),flush=True)


if __name__=='__main__':main()
