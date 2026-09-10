"""Read-only reconciliation of sealed v26 outputs; no raw coordinates exported."""
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits')
RUN = '5c845b51f346e54c'
PINS = {
    'geo': '0e50b9b49267bf2712368b0d6ccdf0439e47969e7bbe82643ca63abce0b4e3d8',
    'names': '1a6e7ea9522e0e09ca9d1a9012e24abf2ea842a2f84ae3896768e33430b3883e',
}


def main():
    c = duckdb.connect()
    c.execute("SET threads=2; SET memory_limit='2GB'")
    for kind, pin in PINS.items():
        root = ROOT / ('retrieval_v26_' + kind) / RUN
        manifest = root / 'manifest.json'
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == pin
        m = json.loads(manifest.read_text())
        for f in m['files'].values():
            p = Path(f['path'])
            assert p.stat().st_size == f['size_bytes']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == f['sha256']
        assert not m['fold0_opened'] and not m['fold1_opened']
        assert not m['new_candidates'] and m['truth_side_only']
        c.execute(f"CREATE VIEW {kind} AS SELECT * FROM read_parquet('{root}/result.parquet')")
        assert c.execute(f'SELECT count(*),count(DISTINCT query_id),sum(exact_hit::INT),sum(operational_hit::INT) FROM {kind}').fetchone() == (1034, 1034, 823, 825)
        assert c.execute(f'SELECT DISTINCT fold FROM {kind} ORDER BY 1').fetchall() == [(2,), (3,), (4,)]
        pre = 'preflight16' if kind == 'geo' else 'preflight16-sqlfix'
        small = c.execute(f"SELECT * FROM read_parquet('{root.parent}/{pre}/result.parquet') ORDER BY query_id").df()
        full = c.execute(f'SELECT * FROM {kind} ORDER BY query_id LIMIT 16').df()
        # A nearest witness may tie; its minimum distance is checked separately.
        cols = [x for x in full if x != 'nearest_siret']
        pd.testing.assert_frame_equal(small[cols], full[cols], check_dtype=False)
    ledger = '/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet'
    c.execute(f"CREATE TEMP TABLE labels AS SELECT query_id,fold,human_ground_truth_siret,human_ground_truth_siren FROM read_parquet('{ledger}') SEMI JOIN geo USING(query_id) WHERE fold IN (2,3,4)")
    assert c.execute('SELECT count(*) FROM labels').fetchone()[0] == 1034
    for kind in PINS:
        assert c.execute(f'''SELECT count(*) FROM {kind} g JOIN labels l USING(query_id)
          WHERE g.fold IS DISTINCT FROM l.fold OR g.gt_siret IS DISTINCT FROM l.human_ground_truth_siret
          OR g.gt_siren IS DISTINCT FROM l.human_ground_truth_siren''').fetchone()[0] == 0
    c.execute(f"CREATE VIEW sites AS SELECT * FROM read_parquet('{ROOT}/retrieval_v26_geo/{RUN}/site_distances.parquet')")
    assert set(x[0] for x in c.execute('DESCRIBE sites').fetchall()) == {'query_id','siret','state','is_exact','different_street','valid_coordinate','distance_m'}
    assert c.execute('SELECT count(*) FROM sites').fetchone()[0] == 60507
    assert c.execute('SELECT count(*) FROM (SELECT query_id,siret FROM sites GROUP BY ALL HAVING count(*)<>1)').fetchone()[0] == 0
    c.execute('''CREATE TEMP TABLE agg AS SELECT query_id,count(*) n,sum(valid_coordinate::INT) nc,
      min(distance_m) near,min(distance_m) FILTER(WHERE is_exact) exact,
      min(distance_m) FILTER(WHERE state='A') active,
      min(distance_m) FILTER(WHERE different_street) other FROM sites GROUP BY query_id''')
    assert c.execute('''SELECT count(*) FROM geo g LEFT JOIN agg a USING(query_id)
      WHERE g.parent_site_count IS DISTINCT FROM coalesce(a.n,0)
      OR g.valid_coordinate_sites IS DISTINCT FROM coalesce(a.nc,0)
      OR g.nearest_parent_m IS DISTINCT FROM a.near OR g.exact_distance_m IS DISTINCT FROM a.exact
      OR g.nearest_active_m IS DISTINCT FROM a.active OR g.nearest_different_street_m IS DISTINCT FROM a.other''').fetchone()[0] == 0
    assert c.execute('''SELECT count(*) FROM geo g WHERE nearest_parent_m IS NOT NULL AND NOT EXISTS
      (SELECT 1 FROM sites s WHERE s.query_id=g.query_id AND s.siret=g.nearest_siret AND s.distance_m=g.nearest_parent_m)''').fetchone()[0] == 0
    assert c.execute('''SELECT count(*) FROM sites s JOIN geo g USING(query_id) WHERE
      (s.distance_m IS NOT NULL AND (NOT s.valid_coordinate OR NOT g.query_coordinate_valid OR NOT g.ban_resolved))
      OR s.is_exact IS DISTINCT FROM (s.siret=g.gt_siret)''').fetchone()[0] == 0
    # Independently join the official full stock, not the audit's site inventory.
    stock = '/Users/nathanjullia/Documents/Projets/SIRETO/data/StockEtablissement_utf8.parquet'
    c.execute(f'''CREATE TEMP TABLE expected AS SELECT g.query_id,s.siret FROM read_parquet('{stock}') s
      JOIN (SELECT query_id,gt_siren FROM geo) g ON s.siren=g.gt_siren''')
    assert c.execute('SELECT count(*) FROM ((SELECT * FROM expected EXCEPT SELECT query_id,siret FROM sites) UNION ALL (SELECT query_id,siret FROM sites EXCEPT SELECT * FROM expected))').fetchone()[0] == 0
    assert c.execute('SELECT sum(unrepresented_names),sum((portfolio_exact_name AND NOT global_docs_exact_name)::INT) FROM names').fetchone() == (0, 0)
    print(json.dumps({'decision':'PASS_V26_OUTPUT_RECONCILIATION','queries':1034,'site_pairs':60507,
      'preflight_16_equal':True,'all_stock_parent_sites_equal':True,'ledger_equal':True,
      'baseline_exact':823,'baseline_operational':825,'new_recall_measured':False}))
    c.close()


if __name__ == '__main__':
    main()
