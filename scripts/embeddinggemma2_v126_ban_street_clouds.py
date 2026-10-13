"""Uniform offline street-position geometry for every query-only weak street.

Use actual BAN address positions rather than one averaged point. They are a
point cloud, NOT an invented continuous road or certified property footprint.
All526 are represented; strong numbered addresses stay unchanged.
"""
from pathlib import Path
import time
from collections import defaultdict

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from src.xgb_matcher.retrieval_query_ban_v327 import BAN
from src.xgb_matcher.retrieval_geo_v16 import lonlat_to_l93
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v126/ban_street_clouds_20261013_a'
BASE=io.HOME/'v115/streamed_activity_ablation_20261013_a'


def main():
    import duckdb
    import pyarrow as pa
    engine.guard()
    if OUTPUT.exists():raise ValueError('Fresh offline source namespace')
    launch=io.read(BASE/'launch.json');panel=io.read(verify(launch['panel']))
    gs={r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    wanted=[dict(query_id=q['query_id'],commune=gs[q['query_id']]['matched_commune'],
                 street=normalize(gs[q['query_id']]['matched_street'],street=True))
            for q in panel if gs[q['query_id']]['spatial_crs_compatible'] and
            gs[q['query_id']].get('precision')=='STREET_FOOTPRINT_NOT_HOUSE']
    OUTPUT.mkdir(parents=True);(OUTPUT/'spill').mkdir()
    source=Path(BAN['path']);verify(BAN)
    io.write(OUTPUT/'launch.json',dict(implementation=record(Path(__file__)),panel=launch['panel'],
        original_geometries=launch['geometry'],ban=BAN,queries=526,weak_street_queries=len(wanted),
        eligibility='DIAGNOSTIC_ONLY_POSTCUTOFF_BAN_SCORE_INELIGIBLE',no_labels_read=True,
        source_points_not_road_lines_or_parcels=True,created_unix=time.time()))
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='512MB'")
    con.execute('SET temp_directory=?',[str(OUTPUT/'spill')])
    con.register('wanted',pa.Table.from_pylist(wanted))
    con.execute("CREATE MACRO textkey(v) AS trim(regexp_replace(upper(strip_accents(coalesce(v,''))), '[^A-Z0-9]+',' ','g'))")
    # Same source-global street abbreviation rules as the established resolver.
    from scripts.embeddinggemma2_v17_structural_retrieval import STREET
    expression='textkey(v)'
    for short,long in STREET.items():
        expression=f"regexp_replace({expression}, '(^| ){short}( |$)', '\\1{long}\\2', 'g')"
    con.execute('CREATE MACRO streetkey(v) AS '+expression)
    rows=con.execute("""SELECT w.query_id,b.longitude,b.latitude FROM wanted w JOIN read_parquet(?) b
      ON w.commune=b.code_commune AND w.street=streetkey(b.nom_voie)
      WHERE isfinite(b.longitude) AND isfinite(b.latitude)
      AND b.longitude BETWEEN -6 AND 10 AND b.latitude BETWEEN 41 AND 52""",[str(source)]).fetchall()
    con.close();points=defaultdict(set)
    for qid,lon,lat in rows:points[qid].add(lonlat_to_l93(float(lon),float(lat)))
    shapes={}
    for q in panel:
        p=sorted(points[q['query_id']])
        # Degenerate segments preserve the existing exact Euclidean primitive;
        # no nearest-neighbour connections between address points are invented.
        shapes[q['query_id']]=dict(status='ACTUAL_BAN_STREET_POSITION_CLOUD' if p else 'UNCHANGED_QUERY',
            points_l93=[list(x) for x in p],lines_l93=[[list(x),list(x)] for x in p],
            not_osm_road_axis=True,not_house_precision=True,no_candidate_or_label_in_source=True)
    io.write(OUTPUT/'shapes.json',shapes);engine.seal(OUTPUT/'shapes.json');verify(BAN)
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_ALL526_QUERY_ONLY_STREET_SOURCE',
        launch=record(OUTPUT/'launch.json'),shapes=record(OUTPUT/'shapes.json'),all_queries=526,
        weak_street_queries=len(wanted),matched_queries=sum(bool(s['lines_l93']) for s in shapes.values()),
        actual_unique_positions=sum(len(s['points_l93']) for s in shapes.values()),
        no_labels_read=True,not_a_recall_measurement=True,community_osm_not_used_in_this_source=True,
        eligibility='DIAGNOSTIC_ONLY_POSTCUTOFF_BAN_SCORE_INELIGIBLE',completed_unix=time.time()))
    print('COMPLETE',len(wanted),sum(bool(s['lines_l93']) for s in shapes.values()),flush=True)


if __name__=='__main__':main()
