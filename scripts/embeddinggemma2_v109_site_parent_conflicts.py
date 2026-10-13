"""Post-seal current-site audit of all exact misses with a retrieved parent.

Current address discrepancy is not a certified label error; preserve all labels.
No score, source alias, model, candidate list or operational policy changed.
"""
from pathlib import Path
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v109/site_parent_conflicts_20261013_a'
PIPELINE = io.HOME / 'v99/initial_alignment_native_20261013_a'
AUDIT = io.HOME / 'v101/postseal_name_roles_20261013_a/report.json'


def main():
    import duckdb
    import pyarrow as pa
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():
        raise ValueError('Fresh postseal namespace')
    launch = io.read(PIPELINE/'launch.json')
    generation = io.read(PIPELINE/'generation.json')
    owned = io.read(PIPELINE/'generate_exit.json')
    if generation['status'] != 'COMPLETE_SEALED_ALL526' or owned['exit_code'] or not owned['own_child_exit_confirmed']:
        raise ValueError('Complete sealed owned output')
    outputs = io.read(verify(generation['outputs']))['learned_native_ooc']
    outcomes = io.read(PIPELINE/'outcomes.json')['learned_native_ooc']
    rows = {r['query_id']: r for r in io.read(verify(io.read(AUDIT)['cases']))}
    selected = [r for r in outcomes if not r['exact'] and r['siren']]
    if len(selected) != 4 or len(outcomes) != 526:
        raise ValueError('Whole fixed pilot, all four parent hits')
    wanted = set()
    for r in selected:
        gt = rows[r['query_id']]['label_siret']
        wanted.add(gt)
        wanted.update(c['siret'] for c in outputs[r['query_id']] if c['siren'] == gt[:9])
    con = duckdb.connect()
    con.execute('SET threads=1')
    con.execute("SET memory_limit='512MB'")
    con.register('wanted', pa.table({'siret': sorted(wanted)}))
    stock = verify(launch['stock_source'])
    source = con.execute("""SELECT e.siret,e.etatAdministratifEtablissement state,
        e.etablissementSiege headquarters,e.numeroVoieEtablissement number,
        e.typeVoieEtablissement street_type,e.libelleVoieEtablissement street,
        e.codeCommuneEtablissement insee FROM read_parquet(?) e JOIN wanted w USING(siret)
        JOIN read_parquet(?) u ON e.siren=u.siren
        WHERE e.statutDiffusionEtablissement='O' AND u.statutDiffusionUniteLegale='O'""",
        [str(stock),str(verify(launch['public_legal_units']) if 'public_legal_units' in launch else mandate.audit_utils.STOCK)]).fetch_arrow_table().to_pylist()
    con.close()
    sites = {r['siret']: r for r in source}
    if set(sites) != wanted:
        raise ValueError('Public complete selected site authority, no backfill')
    cases = []
    for r in selected:
        row = rows[r['query_id']]
        gt = row['label_siret']
        alternatives = [dict(rank=i+1,site=sites[c['siret']])
                        for i,c in enumerate(outputs[r['query_id']]) if c['siren']==gt[:9]]
        cases.append(dict(query_id=r['query_id'],query=row['query'],human_target=sites[gt],
            same_parent_retrieved_sites=alternatives,operational_match=r['operational'],
            operational_policy_not_recomputed=True,current_stock_not_full_address_history=True,
            label_error_proven=False,label_and_exact_denominator_unchanged=True))
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'cases.json', cases)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_POSTSEAL_SITE_PARENT_CONFLICT_AUDIT_NOT_LABEL_REPAIR',
        implementation=record(Path(__file__)),pipeline_generation=record(PIPELINE/'generation.json'),
        pipeline_exit=record(PIPELINE/'generate_exit.json'),name_audit=record(AUDIT),
        stock_source=launch['stock_source'],public_legal_units=record(mandate.audit_utils.STOCK),
        cases=record(OUTPUT/'cases.json'),queries=4,human_target_current_headquarters=sum(r['human_target']['headquarters'] for r in cases),
        no_label_model_source_or_prediction_changed=True,no_new_recall_measured=True,observed_unix=time.time()))
    print([(r['query']['crm_name'],r['human_target']['headquarters'],r['operational_match']) for r in cases],flush=True)


if __name__=='__main__':
    main()
