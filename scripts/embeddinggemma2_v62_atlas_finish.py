"""Finish V60 from sealed projected points; no national BAN re-encoding.

The source scan is complete. Only final SQL reserved identifier is corrected.
All source semantics remain diagnostic-only; source/model hashes unchanged.
"""
import argparse
import os
from pathlib import Path
import signal
import sys
import time

from scripts import embeddinggemma2_v60_public_site_ban_diagnostic as atlas
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

SOURCE=atlas.OUTPUT
OUTPUT=io.HOME/'v62/public_site_ban_finish_20261013_a'


def main(out):
    import duckdb
    import pyarrow.parquet as pq
    if out.exists():raise ValueError('Fresh recovery namespace')
    launch=atlas.authority(SOURCE);ex=io.read(SOURCE/'prepare_exit.json')
    if ex['exit_code']!=1 or not ex['own_child_exit_confirmed']:raise ValueError('Stopped source process only')
    points=engine.sealed(SOURCE/'ban_points.parquet')
    if points is None or pq.ParquetFile(points).metadata.num_rows!=12149538:
        raise ValueError('Exactly complete sealed national matching point stream')
    if {'longitude','latitude'}&set(pq.read_schema(points).names):raise ValueError('No raw BAN lonlat persisted')
    for ref in (launch['stock'],launch['ban'],launch['ban_catalog']):verify(ref)
    out.mkdir(parents=True);(out/'spill').mkdir()
    io.write(out/'launch.json',dict(implementation=record(Path(__file__)),source_launch=record(SOURCE/'launch.json'),
        source_exit=record(SOURCE/'prepare_exit.json'),source_points=record(points),
        source_rows=12149538,deadline_unix=launch['deadline_unix'],
        eligibility='DIAGNOSTIC_ONLY_BAN_SCORE_INELIGIBLE',labels_used=False,production_promotion_allowed=False))
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    c.execute('SET temp_directory=?',[str(out/'spill')]);started=time.monotonic()
    def stop(*_):engine.INTERRUPTED=True;c.interrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    c.execute("CREATE MACRO textkey(v) AS trim(regexp_replace(upper(strip_accents(coalesce(v,''))), '[^A-Z0-9]+',' ','g'))")
    expression='textkey(v)'
    for short,long in atlas.mandate.original.structural.STREET.items():
        expression=f"regexp_replace({expression}, '(^| ){short}( |$)', '\\1{long}\\2', 'g')"
    c.execute('CREATE MACRO streetkey(v) AS '+expression)
    try:
        c.execute("""CREATE TABLE missing AS SELECT CAST(siret AS UBIGINT) AS site_key,
            codeCommuneEtablissement commune,
            streetkey(concat_ws(' ',typeVoieEtablissement,libelleVoieEtablissement)) street,
            textkey(numeroVoieEtablissement) number,textkey(indiceRepetitionEtablissement) suffix
            FROM read_parquet(?) WHERE statutDiffusionEtablissement='O'
            AND codeCommuneEtablissement IS NOT NULL AND codeCommuneEtablissement NOT LIKE '97%'
            AND codeCommuneEtablissement NOT LIKE '98%' AND coalesce(libelleVoieEtablissement,'')<>''
            AND NOT coalesce(try_cast(coordonneeLambertAbscisseEtablissement AS DOUBLE) BETWEEN 0 AND 1300000
                AND try_cast(coordonneeLambertOrdonneeEtablissement AS DOUBLE) BETWEEN 6000000 AND 7300000,FALSE)""",[str(verify(launch['stock']))])
        count=c.execute('SELECT count(*) FROM missing').fetchone()[0]
        if count!=9051971:raise ValueError('Same all-national public missing source denominator')
        atlas.mandate.guard(launch['deadline_unix'])
        io.write(out/'state.json',dict(status='AGGREGATING_COMPLETE_SEALED_POINTS',pid=os.getpid(),updated_unix=time.time()))
        c.execute("""CREATE TABLE houses AS SELECT commune,street,number,suffix,avg(x) x,avg(y) y,
            count(*) points,min(point_quality) quality,
            max(uncertainty_m)+sqrt(pow(max(x)-min(x),2)+pow(max(y)-min(y),2)) uncertainty
            FROM read_parquet(?) GROUP BY commune,street,number,suffix""",[str(points)])
        c.execute("""CREATE TABLE streets AS SELECT commune,street,avg(x) x,avg(y) y,count(*) points,
            max(uncertainty_m)+sqrt(pow(max(x)-min(x),2)+pow(max(y)-min(y),2)) uncertainty
            FROM read_parquet(?) GROUP BY commune,street""",[str(points)])
        table=c.execute("""SELECT m.site_key,coalesce(h.x,s.x) x,coalesce(h.y,s.y) y,
            coalesce(h.uncertainty,s.uncertainty) uncertainty_m,
            CASE WHEN h.x IS NOT NULL THEN h.quality ELSE .25 END reliability,
            CASE WHEN h.x IS NOT NULL THEN 'EXACT_HOUSE_PUBLIC_OWN_ADDRESS'
                ELSE 'FULL_STREET_FOOTPRINT_NOT_HOUSE' END geo_method,
            coalesce(h.points,s.points) source_points
            FROM missing m JOIN streets s USING(commune,street)
            LEFT JOIN houses h ON m.commune=h.commune AND m.street=h.street
                AND m.number=h.number AND m.suffix=h.suffix AND m.number<>''
            WHERE coalesce(h.x,s.x) BETWEEN 0 AND 1300000 AND coalesce(h.y,s.y) BETWEEN 6000000 AND 7300000
            ORDER BY m.site_key""").fetch_arrow_table()
        pq.write_table(table,out/'site_geometry.pending.parquet',compression='zstd')
        os.replace(out/'site_geometry.pending.parquet',out/'site_geometry.parquet');engine.seal(out/'site_geometry.parquet')
        methods=dict(c.execute('SELECT geo_method,count(*) FROM read_parquet(?) GROUP BY geo_method',[str(out/'site_geometry.parquet')]).fetchall())
        atlas.mandate.guard(launch['deadline_unix']);verify(launch['stock']);verify(launch['ban']);verify(launch['ban_catalog'])
        io.write(out/'generation.json',dict(status='COMPLETE_NATIONAL_PUBLIC_OWN_ADDRESS_GEOMETRY_ATLAS',
            eligibility='DIAGNOSTIC_ONLY_BAN_SCORE_INELIGIBLE',production_promotion_allowed=False,
            raw_ban_lonlat_persisted=False,ban_catalog=launch['ban_catalog'],sites=len(table),
            eligible_missing_public_sites=count,unresolved=count-len(table),methods=methods,
            geometry=record(out/'site_geometry.parquet'),ban_point_provenance=record(points),
            labels_used=False,no_gt_directed_source_repair=True,not_recall_measurement=True,elapsed_s=time.monotonic()-started))
        io.write(out/'state.json',dict(status='COMPLETE_SOURCE_ATLAS_NO_RECALL',sites=len(table),methods=methods,updated_unix=time.time()))
    finally:c.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    try:main(a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/'execution_error.json',dict(error=repr(e),updated_unix=time.time()))
        raise
