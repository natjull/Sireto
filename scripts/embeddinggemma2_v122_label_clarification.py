"""Read-only postseal label audit. No predictions, aliases or label corrections.

Public corporate source fields only; protected subjects are not reconstructed.
Natural-person names are excluded. Same-owner sites and succession facts are
diagnostic and cannot establish an unseen CRM invoice's contracting party.
"""
from pathlib import Path
from datetime import date
import time

from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v122/label_clarification_20261013_b'
BASE = io.HOME / 'v115/streamed_activity_ablation_20261013_a'
LEDGER = io.HOME / 'v111/five_steps_ledger_20261013_a'
# Independently identified documentary rivals, not a proposed replacement set.
RIVALS = ['481904753', '402787717', '562047050', '842876773', '790173066', '342280609']


def main():
    import duckdb
    import pyarrow as pa
    if OUTPUT.exists():
        raise ValueError('Fresh diagnostic namespace required')
    gen, ex = io.read(BASE/'generation.json'), io.read(BASE/'generate_exit.json')
    if gen['status'] != 'COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:
        raise ValueError('Complete owned generation before labels')
    verify(gen['outputs'])
    ledger = io.read(LEDGER/'report.json')
    rows = io.read(verify(ledger['cases']))
    outcomes = io.read(BASE/'outcomes.json')['learned_native_ooc']
    remaining = {r['query_id'] for r in outcomes if not r['exact']}
    cases = [r for r in rows if r['query_id'] in remaining]
    if len(cases) != 22 or {r['query_id'] for r in cases} != remaining:
        raise ValueError('All22 actual remaining misses, no filtering')
    sites, owners = ledger['official_stock_sites'], ledger['official_stock_owners']
    launch = io.read(BASE/'launch.json')
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'spill').mkdir()
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET memory_limit='512MB'")
    con.execute('SET temp_directory=?', [str(OUTPUT/'spill')])
    public = [r for r in cases if not r['protected_not_backfilled']]
    wanted = sorted({r['human_target_siret'][:9] for r in public} | set(RIVALS))
    con.register('wanted', pa.table({'siren': wanted}))
    legal = con.execute("""SELECT u.siren,u.statutDiffusionUniteLegale diffusion,
      u.categorieJuridiqueUniteLegale legal_category,u.activitePrincipaleUniteLegale activity,
      CASE WHEN u.statutDiffusionUniteLegale='O' AND u.categorieJuridiqueUniteLegale<>'1000'
           THEN u.denominationUniteLegale ELSE NULL END corporate_name
      FROM read_parquet(?) u JOIN wanted USING(siren)""", [owners['path']]).fetch_arrow_table().to_pylist()
    legal_by_id = {r['siren']: r for r in legal}
    # Guard is in SQL BEFORE address/enseigne fields are returned.
    records = con.execute("""SELECT e.siret,e.siren,e.etatAdministratifEtablissement state,
      e.etablissementSiege headquarters,e.dateCreationEtablissement created,
      e.numeroVoieEtablissement number,e.typeVoieEtablissement street_type,
      e.libelleVoieEtablissement street,e.complementAdresseEtablissement complement,
      e.codeCommuneEtablissement insee,e.codePostalEtablissement postcode,
      e.activitePrincipaleEtablissement activity,e.enseigne1Etablissement own_enseigne
      FROM read_parquet(?) e JOIN wanted USING(siren)
      JOIN read_parquet(?) u USING(siren)
      WHERE e.statutDiffusionEtablissement='O' AND u.statutDiffusionUniteLegale='O'
        AND u.categorieJuridiqueUniteLegale<>'1000'""",
      [sites['path'], owners['path']]).fetch_arrow_table().to_pylist()
    primary_refs = sorted({r['human_target_siret'] for r in public} |
                          {r['siret'] for r in records if r['siren'] in RIVALS})
    con.register('refs', pa.table({'siret': primary_refs}))
    schema = con.execute('DESCRIBE SELECT * FROM read_parquet(?)',
                         [launch['raw_successions']['path']]).fetchall()
    columns = {r[0] for r in schema}
    predecessor, successor = 'siretEtablissementPredecesseur', 'siretEtablissementSuccesseur'
    if not {predecessor, successor} <= columns:
        raise ValueError('Observed official succession schema required')
    edges = con.execute(f"""SELECT * FROM read_parquet(?)
      WHERE {predecessor} IN (SELECT siret FROM refs)
         OR {successor} IN (SELECT siret FROM refs)""",
      [launch['raw_successions']['path']]).fetch_arrow_table().to_pylist()
    for row in records + edges:
        for key, value in row.items():
            if isinstance(value, date):
                row[key] = value.isoformat()
    con.close()
    result = []
    for r in cases:
        owner = r['human_target_siret'][:9]
        protected = r['protected_not_backfilled']
        same_zone = [s for s in records if s['siren'] == owner and
                     (s['insee'] == r['query']['crm_insee'] or s['siret'] == r['human_target_siret'])]
        result.append(dict(query_id=r['query_id'], crm=r['query'],
            label_siret=r['human_target_siret'], protected_no_reconstruction=protected,
            public_legal_category_only=None if protected else legal_by_id.get(owner),
            same_owner_public_sites_in_query_zone=[] if protected else same_zone,
            direct_succession_edges=[] if protected else [e for e in edges if
                r['human_target_siret'] in (e[predecessor],e[successor])],
            previous_ledger=r, no_label_change=True, no_retrieval_input=True))
    io.write(OUTPUT/'cases.json', result)
    io.write(OUTPUT/'documentary_rivals.json', dict(
        legal_entities=[r for r in legal if r['siren'] in RIVALS],
        public_sites=[r for r in records if r['siren'] in RIVALS],
        succession_edges=edges, rivals_not_automatic_replacement_labels=True))
    io.write(OUTPUT/'report.json', dict(status='COMPLETE_ALL22_POSTSEAL_PUBLIC_FACT_CLARIFICATION',
        implementation=record(Path(__file__)), baseline_generation=record(BASE/'generation.json'),
        baseline_exit=record(BASE/'generate_exit.json'), ledger=record(LEDGER/'report.json'),
        source_sites=sites, source_owners=owners, source_successions=launch['raw_successions'],
        cases=record(OUTPUT/'cases.json'), documentary_rivals=record(OUTPUT/'documentary_rivals.json'),
        remaining_queries=22, protected_not_reconstructed=2, no_physical_person_names=True,
        no_closed_dataset_opened=True, no_label_or_denominator_change=True,
        no_training_or_retrieval_change=True, observed_unix=time.time()))
    print('COMPLETE_ALL22_POSTSEAL_PUBLIC_FACT_CLARIFICATION', flush=True)


if __name__ == '__main__':
    main()
