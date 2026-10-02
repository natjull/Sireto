"""Post-seal geography audit of already returned references; no search or fit."""
import json
from pathlib import Path

import duckdb
import pyarrow as pa

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json


def main():
    root = v38.DATA/'audits/retrieval_v38_initial_top100/pilot_20261002_a'
    generation = v38.load(root/'manifest.json', 'e37d9d874a79bf0c05dea1003b120a12fe7451c773134877b64aef847235b7fa')
    evaluation = v38.load(root/'evaluation.json', '98b4816d04b6d1b3c7506a466851b825ade0a9257a37e7257c0ca5b61804bec5')
    launch = json.loads(verify(generation['launch']).read_text())
    queries = {r['query_id']: r for r in launch['queries']}
    outcomes = {r['query_id']: r for r in evaluation['outcomes']}
    rows = []
    for ref in generation['batches']:
        for r in json.loads(verify(ref).read_text())['rows']:
            q, out = queries[r['query_id']], outcomes[r['query_id']]
            group = ('gain' if out['primary_exact'] and not out['comparator_exact'] else
                     'regression' if out['comparator_exact'] and not out['primary_exact'] else
                     'both_hit' if out['primary_exact'] else 'both_miss')
            for mode in v38.MODES:
                for candidate in r['results'][mode]:
                    rows.append(dict(query_id=r['query_id'], fold=r['fold'], mode=mode, category=group,
                        siret=candidate['siret'], crm_insee=q['crm_insee'], crm_postcode=q['crm_postcode']))
    source = dict(path=v38.DATA/'indexes/retrieval_sidecars_v11/de762d80b7d61d35/siren_local_sites.parquet',
                  sha256='5d967a0b2bb3fc38138601b5bb80ab997bc7d10afc85c7544e65558ab146ad60')
    source_path = verify(source)
    dest = root/'geography_audit.json'
    if dest.exists():
        raise ValueError('Refuse to overwrite audit')
    with Budget(dict(max_seconds=60, max_rss_bytes=2*1024**3, max_swapout_pages_delta=0)) as budget:
        db = duckdb.connect(); db.execute("SET memory_limit='1GB'"); db.execute('SET threads=2')
        db.register('returned', pa.Table.from_pylist(rows))
        db.execute('''CREATE TEMP TABLE sites AS SELECT siret,insee,postcode FROM read_parquet(?)
                      WHERE siret IN (SELECT DISTINCT siret FROM returned)''', [str(source_path)])
        if db.execute('SELECT count(*)-count(DISTINCT siret) FROM sites').fetchone()[0]:
            raise ValueError('Duplicate official site')
        perquery = db.execute('''SELECT r.query_id,r.fold,r.mode,r.category,count(*) returned,
           count(s.siret) official_known,
           count(*) FILTER(WHERE s.siret IS NOT NULL AND
             ((r.crm_insee<>'' AND r.crm_insee=s.insee) OR
              (r.crm_postcode<>'' AND r.crm_postcode=s.postcode))) same_insee_or_postcode
           FROM returned r LEFT JOIN sites s USING(siret)
           GROUP BY r.query_id,r.fold,r.mode,r.category ORDER BY r.query_id,r.mode''').fetch_arrow_table().to_pylist()
        db.close()
    report = {}
    for mode in v38.MODES:
        report[mode] = {}
        for category in ('all', 'gain', 'regression', 'both_hit', 'both_miss'):
            group = [r for r in perquery if r['mode'] == mode and (category == 'all' or r['category'] == category)]
            known = sum(r['official_known'] for r in group)
            local = sum(r['same_insee_or_postcode'] for r in group)
            report[mode][category] = dict(queries_with_nonempty_output=len(group), returned=sum(r['returned'] for r in group),
                official_known=known, same_insee_or_postcode=local,
                outside_insee_and_postcode=known-local, outside_fraction_among_known=(known-local)/known if known else None)
    write_json(dest, dict(generation=record(root/'manifest.json'), evaluation=record(root/'evaluation.json'),
        official_sites=record(source_path), summary=report, perquery=perquery, resources=budget.report(),
        new_searches=0, new_model_fits=0, scientific_role='POSTHOC_DIAGNOSTIC_NOT_A_SELECTION_FEATURE',
        caveat='Different locality is not proof of a wrong match. Missing official rows remain unknown.',
        implementation=record(Path(__file__))))
    print(json.dumps(dict(summary=report, resources=budget.report(), artifact=record(dest))))


if __name__ == '__main__':
    main()
