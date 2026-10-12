"""Official-stock status diagnosis only, never a retrieval source or label edit."""
from collections import Counter
from pathlib import Path
import duckdb
import pyarrow as pa
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record,verify

SOURCE=io.HOME/'v32/global_public_parent_names_20261012_a'
OUT=io.HOME/'v47/postseal_stock_status_20261012_a'
STOCK=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01')

def main():
    if OUT.exists():raise ValueError('Fresh label-side namespace')
    gen=io.read(SOURCE/'generation.json');ex=io.read(SOURCE/'generate_exit.json')
    if gen['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code']!=0 or not ex['own_child_exit_confirmed']:
        raise ValueError('Complete owned source before labels')
    verify(gen['outputs']);rows=io.read(SOURCE/'outcomes.json')['learned_native_ooc']
    labels={r['query_id']:r['label'] for r in io.read(io.HOME/'v20/deep_audit_20261011_c/rows.json')}
    selected=[dict(query_id=r['query_id'],siret=labels[r['query_id']]['siret'],siren=labels[r['query_id']]['siren']) for r in rows if not r['exact']]
    if len(rows)!=526 or any(labels[r['query_id']]['fold'] not in (2,3,4) for r in rows):raise ValueError('Same all526')
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='1GB'")
    con.register('wanted',pa.Table.from_pylist(selected))
    cases=con.execute("""SELECT w.query_id,w.siret,w.siren,
      e.siret IS NOT NULL site_in_current_stock,e.statutDiffusionEtablissement site_diffusion,
      e.etatAdministratifEtablissement site_state,e.dateCreationEtablissement site_created,
      u.siren IS NOT NULL owner_in_current_stock,u.statutDiffusionUniteLegale owner_diffusion,
      u.etatAdministratifUniteLegale owner_state,u.unitePurgeeUniteLegale owner_purged,
      coalesce(u.denominationUniteLegale,'') NOT IN ('','[ND]') public_legal_denomination_present,
      coalesce(u.nomUniteLegale,'') NOT IN ('','[ND]') public_individual_name_present,
      coalesce(e.libelleVoieEtablissement,'') NOT IN ('','[ND]') public_street_present,
      coalesce(e.numeroVoieEtablissement,'') NOT IN ('','[ND]') public_number_present
      FROM wanted w LEFT JOIN read_parquet(?) e ON w.siret=e.siret
      LEFT JOIN read_parquet(?) u ON w.siren=u.siren ORDER BY w.query_id""",
      [str(STOCK/'StockEtablissement.parquet'),str(STOCK/'StockUniteLegale.parquet')]).fetch_arrow_table().to_pylist()
    con.close()
    for c in cases:c['site_created']=str(c['site_created']) if c['site_created'] is not None else None
    if len(cases)!=len(selected):raise ValueError('No duplicate/missing audit query rows')
    OUT.mkdir(parents=True);io.write(OUT/'cases.json',cases)
    summary=dict(status='COMPLETE_LABEL_SIDE_STOCK_STATUS_NOT_LABEL_CORRECTION',misses=len(cases),
      site_diffusion=dict(Counter(c['site_diffusion'] for c in cases)),owner_diffusion=dict(Counter(c['owner_diffusion'] for c in cases)),
      site_state=dict(Counter(c['site_state'] for c in cases)),purged_owners=sum(c['owner_purged'] is True for c in cases),
      no_public_legal_or_personal_name=sum(not c['public_legal_denomination_present'] and not c['public_individual_name_present'] for c in cases),
      missing_sites=sum(not c['site_in_current_stock'] for c in cases),missing_owners=sum(not c['owner_in_current_stock'] for c in cases),
      cases=record(OUT/'cases.json'),stocks=[record(STOCK/'StockEtablissement.parquet'),record(STOCK/'StockUniteLegale.parquet')],
      source_outcomes=record(SOURCE/'outcomes.json'),implementation=record(Path(__file__)),
      no_retrieval_feature_created=True,no_denominator_changed=True,no_label_declared_false=True,
      absence_in_current_stock_not_absence_everywhere=True)
    io.write(OUT/'summary.json',summary);print(summary,flush=True)

if __name__=='__main__':main()
