"""Complete postseal miss ledger for renewed five-step mandate.

All23 errors retained. Evidence flags aren't label corrections or predictions.
No closed folds or private target reconstruction, no historical run resumed.
"""
from collections import Counter
from pathlib import Path
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v56_corporate_site as old
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v111/five_steps_ledger_20261013_a'
PIPELINE=io.HOME/'v99/initial_alignment_native_20261013_a'


def main():
    import duckdb
    import pyarrow as pa
    engine.guard()
    if OUTPUT.exists():raise ValueError('Fresh mandate namespace')
    g=io.read(PIPELINE/'generation.json');ex=io.read(PIPELINE/'generate_exit.json')
    if g['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:
        raise ValueError('Whole sealed owned source before labels')
    verify(g['outputs'])
    names_report=io.HOME/'v101/postseal_name_roles_20261013_a/report.json'
    names=io.read(verify(io.read(names_report)['cases']))
    coverage_report=io.HOME/'v103/raw_alias_coverage_20261013_a/report.json'
    coverage=io.read(coverage_report)
    statuses={r['qid']:r for r in io.read(verify(coverage['statuses']))}
    raw={r['query_id']:r for r in io.read(verify(coverage['cases']))}
    misses=[r for r in names if not r['exact']]
    public=[r for r in misses if statuses[r['query_id']]['site_diffusion']=='O' and statuses[r['query_id']]['owner_diffusion']=='O']
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='1GB'")
    con.register('wanted',pa.table({'siret':[r['label_siret'] for r in public]}))
    records=con.execute("""SELECT e.siret,e.siren,e.etatAdministratifEtablissement state,
        e.etablissementSiege headquarters,e.numeroVoieEtablissement number,
        e.indiceRepetitionEtablissement suffix,e.typeVoieEtablissement street_type,
        e.libelleVoieEtablissement street,e.codeCommuneEtablissement insee,
        u.denominationUniteLegale published_owner_name,u.sigleUniteLegale published_owner_acronym
        FROM read_parquet(?) e JOIN wanted w USING(siret)
        JOIN read_parquet(?) u ON e.siren=u.siren
        WHERE e.statutDiffusionEtablissement='O' AND u.statutDiffusionUniteLegale='O'""",
        [str(old.STOCK),str(old.audit_utils.STOCK)]).fetch_arrow_table().to_pylist();con.close()
    sites={r['siret']:r for r in records}
    if len(sites)!=len({r['label_siret'] for r in public}):raise ValueError('All public targets present')
    past={}
    paths={
        'bodacc_full':io.HOME/'v52/postseal_bodacc_fulltext_20261012_a/cases.json',
        'bodacc_reverse':io.HOME/'v53/postseal_bodacc_reverse_20261012_a/cases.json',
        'rivals':io.HOME/'v92/official_rival_audit_20261013_a/cases.json',
        'sector':io.HOME/'v108/sector_site_gap_20261013_a/cases.json',
        'site_conflicts':io.HOME/'v109/site_parent_conflicts_20261013_a/cases.json'}
    for name,path in paths.items():past[name]={r['query_id']:r for r in io.read(path)}
    result=[]
    for r in misses:
        engine.guard();qid=r['query_id'];s=statuses[qid]
        protected=s['site_diffusion']!='O' or s['owner_diffusion']!='O'
        bucket='PUBLIC_INFORMATION_LIMIT' if protected else 'SITE_SELECTION' if r['siren'] else 'OWNER_BRIDGE_UNPROVEN'
        b=past['bodacc_full'].get(qid,{});rival=past['rivals'].get(qid,{})
        result.append(dict(query_id=qid,query=r['query'],human_target_siret=r['label_siret'],
            diagnostic_bucket=bucket,exact=False,siren_found=r['siren'],operational=r['operational'],
            protected_not_backfilled=protected,public_target=sites.get(r['label_siret']) if not protected else None,
            raw_own_name_query_word_overlap=raw[qid]['overlapping_query_words'] if not protected else [],
            indexed_word_name_score=r['all_fields_maximum'] if not protected else None,
            bodacc_announcements=b.get('announcements',0) if not protected else None,
            bodacc_event_passages=len(b.get('event_passages',[])) if not protected else None,
            bodacc_nonadmin_nonname_literal_matches=[dict(field=m['field'],shared_tokens=m['shared_tokens'],
                complete=m['complete_informative_query'],announcement_id=m['id'])
                for m in b.get('literal_matches',[]) if m['field'] not in ('administration','subject_name')] if not protected else [],
            documentary_other_owner_exact_name_and_address_rivals=rival.get('contradictory_exact_name_address_other_owner',0) if not protected else None,
            sector_evidence=past['sector'].get(qid,{}).get('matches',[]) if not protected else [],
            same_parent_site_audit=past['site_conflicts'].get(qid) if not protected else None,
            label_error_proven=False,label_not_modified=True,diagnostic_not_search_input=True))
    if len(result)!=23:raise ValueError('All23 misses retained')
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'cases.json',result)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_ALL23_POSTSEAL_FIVE_STEP_LEDGER',
        implementation=record(Path(__file__)),source_generation=record(PIPELINE/'generation.json'),
        source_exit=record(PIPELINE/'generate_exit.json'),names_audit=record(names_report),raw_audit=record(coverage_report),
        official_stock_sites=record(old.STOCK),official_stock_owners=record(old.audit_utils.STOCK),
        evidence_sources={k:record(p) for k,p in paths.items()},cases=record(OUTPUT/'cases.json'),
        errors=23,buckets=dict(Counter(r['diagnostic_bucket'] for r in result)),
        renewed_human_mandate=True,old_automation_not_resumed=True,no_new_recall=True,
        no_label_or_denominator_changed=True,no_closed_test_opened=True,observed_unix=time.time()))
    print(Counter(r['diagnostic_bucket'] for r in result),flush=True)


if __name__=='__main__':main()
