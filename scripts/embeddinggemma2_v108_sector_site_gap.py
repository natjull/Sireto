"""Post-seal audit of the full existing sector source and unused SITE names.

No aliases copied across sites, no geographic inheritance or new predictions.
Structured SIRET prefix owns a site; a different reported manager is context.
"""
from collections import Counter
from pathlib import Path
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as words
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v108/sector_site_gap_20261013_a'
PIPELINE = io.HOME / 'v99/initial_alignment_native_20261013_a'
CASES = io.HOME / 'v101/postseal_name_roles_20261013_a/report.json'
STATUS = io.HOME / 'v103/raw_alias_coverage_20261013_a/report.json'


def main():
    import duckdb
    import pyarrow as pa
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():
        raise ValueError('Fresh audit namespace')
    generation = io.read(PIPELINE / 'generation.json')
    owned = io.read(PIPELINE / 'generate_exit.json')
    if generation['status'] != 'COMPLETE_SEALED_ALL526' or owned['exit_code'] or not owned['own_child_exit_confirmed']:
        raise ValueError('Whole sealed owned outputs required')
    verify(generation['outputs'])
    launch = io.read(PIPELINE / 'launch.json')
    source = verify(launch['sector_source'])
    verify(launch['sector_manifest'])
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET memory_limit='1GB'")
    schema_counts = con.execute('SELECT source,subject_kind,count(*) FROM read_parquet(?) GROUP BY 1,2',
                               [str(source)]).fetchall()
    # Whole source, not a GT-selected source compiler. Check current public
    # site AND its true owner before reading any protected identity names.
    con.read_parquet(str(source)).create_view('sector')
    con.read_parquet(str(mandate.STOCK)).project(
        'siret,siren,statutDiffusionEtablissement').create_view('sites')
    con.read_parquet(str(mandate.audit_utils.STOCK)).project(
        'siren,statutDiffusionUniteLegale').create_view('owners')
    whole = con.execute("""SELECT source,subject_kind,count(*) n,
        count(*) FILTER(WHERE subject_kind='SIRET' AND siren<>left(siret,9)) mismatched_manager_prefix
        FROM sector GROUP BY 1,2""").fetch_arrow_table().to_pylist()
    records = con.execute("""SELECT s.*,CASE WHEN s.subject_kind='SIRET' THEN left(s.siret,9)
        ELSE s.siren END true_owner FROM sector s LEFT JOIN sites e ON s.siret=e.siret
        JOIN owners u ON u.siren=CASE WHEN s.subject_kind='SIRET' THEN left(s.siret,9) ELSE s.siren END
        WHERE u.statutDiffusionUniteLegale='O'
        AND (s.subject_kind='SIREN' OR (s.subject_kind='SIRET' AND e.statutDiffusionEtablissement='O'))
        AND (coalesce(s.observed_at,'')='' OR s.observed_at<='2026-08-01')""").fetch_arrow_table().to_pylist()
    con.close()
    mandate.guard(mandate.DEADLINE)
    audit = io.read(CASES)
    rows = io.read(verify(audit['cases']))
    status = io.read(STATUS)
    statuses = {r['qid']: r for r in io.read(verify(status['statuses']))}
    by_owner = {}
    for r in records:
        by_owner.setdefault(r['true_owner'], []).append(r)
    result = []
    for row in rows:
        if row['exact']:
            continue
        s = statuses[row['query_id']]
        protected = s['site_diffusion'] != 'O' or s['owner_diffusion'] != 'O'
        matches = []
        if not protected:
            qt = words.tokens(row['query']['crm_name'])
            for r in by_owner.get(row['label_siret'][:9], []):
                shared = qt & words.tokens(r['name'])
                if not shared:
                    continue
                role = ('OWN_TARGET_SITE' if r['siret'] == row['label_siret'] else
                        'OTHER_SITE_SAME_OWNER' if r['subject_kind'] == 'SIRET' else 'OWN_PARENT_NAME')
                matches.append(dict(role=role,source=r['source'],source_record_id=r['source_record_id'],
                    subject_kind=r['subject_kind'],siret=r['siret'],published_name=r['name'],
                    shared_informative_query_tokens=sorted(shared),full_informative_query=bool(qt and qt<=words.tokens(r['name'])),
                    observed_at=r['observed_at'],date_not_source_snapshot_cutoff_proof=True,
                    no_name_or_address_inherited_to_target=True))
        result.append(dict(query_id=row['query_id'],protected_not_backfilled=protected,matches=matches))
    if len(result) != 23:
        raise ValueError('All23 misses retained')
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'cases.json', result)
    io.write(OUTPUT/'report.json', dict(status='COMPLETE_FULL_EXISTING_SECTOR_SITE_GAP_AUDIT_NOT_RETRIEVAL',
        implementation=record(Path(__file__)),pipeline_generation=record(PIPELINE/'generation.json'),
        pipeline_exit=record(PIPELINE/'generate_exit.json'),sector_source=launch['sector_source'],
        sector_manifest=launch['sector_manifest'],name_audit=record(CASES),diffusion_audit=record(STATUS),
        whole_source_rows=sum(r[2] for r in schema_counts),whole_source_scope=whole,
        public_before_record_update_cutoff_rows=len(records),
        cases=record(OUTPUT/'cases.json'),misses=len(result),
        misses_with_any_informative_sector_word=sum(bool(r['matches']) for r in result),
        matching_roles=dict(Counter(m['role'] for r in result for m in r['matches'])),
        sirets_not_borrowed_as_parent_aliases=True,manager_not_owner_identity=True,
        existing_v32_global_branch_reads_siren_rows_only=True,
        no_new_recall_or_prediction=True,no_feature_or_alias_changed=True,
        source_capture_cutoff_admission_not_newly_established=True,observed_unix=time.time()))
    print(whole,[(r['query_id'],r['matches']) for r in result if r['matches']],flush=True)


if __name__ == '__main__':
    main()
