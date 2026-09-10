"""Independent output reconciliation and historical-source novelty check."""
import json
from pathlib import Path

import duckdb

from src.xgb_matcher.retrieval_v20_workbench import verify
from src.xgb_matcher.retrieval_v21_membership import manifest_files

ROOT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100')
AUDITS=ROOT/'audits'
PINS={
    'geo':dict(path=str(AUDITS/'retrieval_v27_geo/f6f0ed019069defc-memory8/manifest.json'),sha256='201c6a05c31cdd8d1a747545eea7f03a285edde46bed5d25f4a155a2d8a18504'),
    'names':dict(path=str(AUDITS/'retrieval_v27_names/f6f0ed019069defc/manifest.json'),sha256='130b4e119870aa18b196e40f2c0db8494d18cb890141e994c6c10a150dc2d234'),
}


def main():
    c=duckdb.connect();c.execute("SET threads=2;SET memory_limit='2GB'")
    manifests={};files={}
    for kind,pin in PINS.items():
        m,fs=manifest_files(pin);manifests[kind]=m;files[kind]=fs
        assert not m['new_candidates'] and not m['fold0_opened'] and not m['fold1_opened']
        c.execute(f'CREATE TEMP TABLE {kind} AS SELECT * FROM read_parquet(?)',[str(fs['result.parquet'])])
        assert c.execute(f'SELECT count(*),count(DISTINCT query_id),sum(exact_hit::INT),sum(operational_hit::INT) FROM {kind}').fetchone()==(1034,1034,823,825)
        assert c.execute(f'SELECT DISTINCT fold FROM {kind} ORDER BY 1').fetchall()==[(2,),(3,),(4,)]
        pre=Path(pin['path']).parent.parent/'preflight16-final/result.parquet'
        c.execute(f'CREATE TEMP TABLE {kind}_pre AS SELECT * FROM read_parquet(?)',[str(pre)])
        assert c.execute(f'''SELECT count(*) FROM ((SELECT * FROM {kind}_pre EXCEPT SELECT * FROM {kind})
          UNION ALL (SELECT g.* FROM {kind} g SEMI JOIN {kind}_pre USING(query_id) EXCEPT SELECT * FROM {kind}_pre))''').fetchone()[0]==0
    ledger=verify(dict(path=str(ROOT/'datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet'),sha256='24f5928cbf2669b0a79ac00a284d14e5fe2288abb5a4ddcbd77a46677c315720'))
    c.execute('''CREATE TEMP TABLE labels AS SELECT query_id,fold,human_ground_truth_siret,human_ground_truth_siren
      FROM read_parquet(?) SEMI JOIN geo USING(query_id) WHERE fold IN (2,3,4)''',[str(ledger)])
    assert c.execute('SELECT count(*) FROM labels').fetchone()[0]==1034
    for kind in PINS:
        assert c.execute(f'''SELECT count(*) FROM {kind} g JOIN labels l USING(query_id) WHERE g.fold IS DISTINCT FROM l.fold
          OR g.gt_siret IS DISTINCT FROM l.human_ground_truth_siret OR g.gt_siren IS DISTINCT FROM l.human_ground_truth_siren''').fetchone()[0]==0
    cfg=json.loads(Path(manifests['geo']['config']['path']).read_text())
    oldmanifest=verify(cfg['v16_manifest']);old=json.loads(oldmanifest.read_text())
    for key,replay,exclude in [('geo_neighbor_candidates.parquet','replay_sites.parquet',''),
                              ('geo_neighbor_parents.parquet','replay_parents.parquet','EXCLUDE(nearest_siret)')]:
        p=verify(dict(path=str(oldmanifest.parent/key),sha256=old['files'][key]['sha256']))
        # Exact set equality, not a floating-point tolerance check.
        c.execute(f'CREATE OR REPLACE TEMP TABLE old AS SELECT * {exclude} FROM read_parquet(?)',[str(p)])
        c.execute(f'CREATE OR REPLACE TEMP TABLE replay AS SELECT * {exclude} FROM read_parquet(?)',[str(files['geo'][replay])])
        assert c.execute('SELECT count(*) FROM ((SELECT * FROM old EXCEPT SELECT * FROM replay) UNION ALL (SELECT * FROM replay EXCEPT SELECT * FROM old))').fetchone()[0]==0
    c.execute('CREATE TEMP TABLE parent_replay AS SELECT * FROM read_parquet(?)',[str(files['geo']['replay_parents.parquet'])])
    assert c.execute('''SELECT count(*) FROM geo g WHERE (geo_reason='PRESENT') IS DISTINCT FROM
      EXISTS(SELECT 1 FROM parent_replay p WHERE p.query_id=g.query_id AND p.siren=g.gt_siren)''').fetchone()[0]==0
    assert c.execute('''SELECT count(*) FROM geo g WHERE focus_missing_near500 AND nearest_excluded_by_commune''').fetchone()[0]==0
    # Every focused omission is farther than (or tied with) all 512 retained parents.
    assert c.execute('''SELECT count(*) FROM geo g WHERE focus_missing_near500 AND
      (SELECT count(*) FROM parent_replay p WHERE p.query_id=g.query_id AND p.geo_parent_distance_m<=g.local_nearest_m)<>512''').fetchone()[0]==0
    c.execute('CREATE TEMP TABLE details AS SELECT * FROM read_parquet(?)',[str(files['names']['name_details.parquet'])])
    assert c.execute("SELECT count(*) FROM details WHERE source_group LIKE '%current%' AND (NOT in_evidence OR NOT in_portfolio_global)").fetchone()[0]==0
    assert c.execute("SELECT count(*) FROM details WHERE field IN ('nomUniteLegale','nomUsageUniteLegale')").fetchone()[0]==0
    c.execute('''CREATE TABLE targets AS SELECT DISTINCT r.query_id,r.gt_siren siren,d.name FROM names r JOIN details d USING(query_id)
      WHERE r.history_missing_exact AND NOT r.portfolio_exact_name AND d.exact_name AND NOT d.in_portfolio_global''')
    index=verify(dict(path=str(ROOT/'indices/sirene_temporal_v1/506bb98cfdcccfbe/manifest.json'),sha256='e460f1dd33c6b8984d8cdf0164c0add02c26cbadef44edd5adfcd75e09390e80'))
    temporal=json.loads(index.read_text());paths=[]
    for item in temporal['sources']:
        for grain in ('siren','siret'):
            p=ROOT/'sources/sirene_temporal_v1'/item['projection_build_id']/('historical_'+grain+'_identity.parquet')
            paths.append(str(verify(dict(path=str(p),sha256=item['historical_'+grain+'_sha256']))))
    c.execute('''CREATE TABLE h AS SELECT DISTINCT siren,
      trim(regexp_replace(upper(strip_accents(normalized_name)),'[^A-Z0-9]+',' ','g')) AS name
      FROM read_parquet(?,union_by_name=true) SEMI JOIN (SELECT DISTINCT siren FROM targets) USING(siren)''',[paths])
    counts=c.execute('''SELECT count(DISTINCT query_id),count(DISTINCT query_id) FILTER(WHERE EXISTS
      (SELECT 1 FROM h WHERE h.siren=t.siren AND h.name=t.name)) FROM targets t''').fetchone()
    assert counts==(8,8)
    print(json.dumps(dict(decision='PASS_V27_OUTPUT_RECONCILIATION',queries=1034,baseline_exact=823,baseline_operational=825,
      exact_legacy_replay=True,preflight_full_equal=True,focused_omissions=12,historical_exact_additions=8,
      already_in_temporal_sources=8,new_recall_measured=False,raw_rne_population_audited=False)))
    c.close()


if __name__=='__main__':main()
