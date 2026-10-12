"""National public OWN-address coordinate coverage, never label-directed.

Fill only missing/incompatible SIRENE L93 coordinates. Exact house evidence
and street-footprint approximations remain distinguishable and uncertain.
No names/private addresses are recovered; closed sites stay eligible.
This source atlas does not itself claim a retrieval improvement.
"""
import argparse
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher import retrieval_geo_v16 as geometry
from src.xgb_matcher.retrieval_query_ban_v327 import BAN
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v60/public_site_ban_diagnostic_20261013_a'
RECIPE=dict(scope='ENTIRE_NATIONAL_PUBLIC_STOCK_NOT_LABEL_OR_TOP100_SELECTED',
    missing_coordinates_only=True, site_diffusion_required='O',
    own_current_public_address_only=True, never_copy_sibling_or_hq_point=True,
    active_closed_both=True, commune_street_join='NORMALIZED_EXACT_NOT_FUZZY',
    house_join='EXACT_NUMBER_AND_REPETITION_SUFFIX', street_fallback='FULL_STREET_FOOTPRINT_NOT_HOUSE',
    existing_coordinates_not_overwritten=True, uncertainty='BAN_PROXY_PLUS_FULL_POINT_DISPERSION',
    street_quality=.25, house_weak_quality=.5, house_strong_quality=1.,
    compatible_crs='METROPOLITAN_LAMBERT93_ONLY', ban_semantics='ADDRESS_MATCH_ONLY_SCORE_INELIGIBLE_DIAGNOSTIC',
    status='DIAGNOSTIC_ONLY_BAN_SCORE_INELIGIBLE', raw_ban_lonlat_persisted=False,
    source_dates='SIRENE_20260801_BAN_20260814_NOT_HISTORICAL_GEOCODING_CLAIM',
    names_not_read=True, labels_not_read=True, not_a_recall_measurement=True,
    deadline_unix=mandate.DEADLINE)


def initialize(out):
    mandate.guard(mandate.DEADLINE)
    if out.exists():raise ValueError('Fresh national source namespace')
    out.mkdir(parents=True);(out/'spill').mkdir()
    catalog=Path('/Volumes/CATNAT_DATA/oubusiness/snapshots/ban_national_build_2026-08-14_semantics-v1.json')
    semantics=io.read(catalog)
    if semantics['address_match_claim_allowed'] is not True or semantics['score_eligible'] is not False or semantics['address_absence_claim_allowed'] is not False:
        raise ValueError('Declared non-decisional BAN semantics required')
    io.write(out/'recipe.json',RECIPE)
    if io.digest(Path(BAN['path']))!=BAN['sha256']:raise ValueError('Pinned BAN semantics hash')
    io.write(out/'launch.json',dict(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        stock=record(mandate.STOCK),ban=record(Path(BAN['path'])),ban_catalog=record(catalog),
        diagnostic_only=True,production_promotion_allowed=False,
        helpers=[record(Path(m.__file__)) for m in (mandate,io,engine,geometry)],
        deadline_unix=mandate.DEADLINE,labels_used=False,all_national_source=True))


def authority(out):
    launch=io.read(out/'launch.json')
    for ref in (launch['implementation'],launch['recipe'],*launch['helpers']):verify(ref)
    mandate.guard(launch['deadline_unix']);return launch


def projected(lon,lat):
    lon=np.deg2rad(lon);lat=np.deg2rad(lat)
    s=np.sin(lat);e=geometry._GRS80_E
    t=np.tan(np.pi/4-lat/2)/(((1-e*s)/(1+e*s))**(e/2))
    rho=geometry._GRS80_A*geometry._L93_F*t**geometry._L93_N
    angle=geometry._L93_N*(lon-geometry._L93_LON0)
    return 700000.+rho*np.sin(angle),6600000.+geometry._L93_RHO0-rho*np.cos(angle)


def prepare(out):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    launch=authority(out);started=time.monotonic();c=duckdb.connect()
    c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    c.execute('SET temp_directory=?',[str(out/'spill')])
    def stop(*_):
        engine.INTERRUPTED=True;c.interrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def state(stage,**kw):io.write(out/'state.json',dict(stage=stage,pid=os.getpid(),elapsed_s=time.monotonic()-started,updated_unix=time.time(),**kw))
    c.execute("CREATE MACRO textkey(v) AS trim(regexp_replace(upper(strip_accents(coalesce(v,''))), '[^A-Z0-9]+',' ','g'))")
    # Source-wide fixed spelling normalization, not learned CRM aliases.
    replacements=mandate.original.structural.STREET
    expression='textkey(v)'
    for short,long in replacements.items():
        expression=f"regexp_replace({expression}, '(^| ){short}( |$)', '\\1{long}\\2', 'g')"
    c.execute('CREATE MACRO streetkey(v) AS '+expression)
    try:
        state('COMPILING_ALL_PUBLIC_MISSING_SITE_ADDRESSES')
        c.execute("""CREATE TABLE missing AS SELECT CAST(siret AS UBIGINT) AS site_key,
            codeCommuneEtablissement commune,
            streetkey(concat_ws(' ',typeVoieEtablissement,libelleVoieEtablissement)) street,
            textkey(numeroVoieEtablissement) number,textkey(indiceRepetitionEtablissement) suffix
            FROM read_parquet(?) WHERE statutDiffusionEtablissement='O'
            AND codeCommuneEtablissement IS NOT NULL AND codeCommuneEtablissement NOT LIKE '97%'
            AND codeCommuneEtablissement NOT LIKE '98%'
            AND coalesce(libelleVoieEtablissement,'')<>''
            AND NOT coalesce(try_cast(coordonneeLambertAbscisseEtablissement AS DOUBLE) BETWEEN 0 AND 1300000
                AND try_cast(coordonneeLambertOrdonneeEtablissement AS DOUBLE) BETWEEN 6000000 AND 7300000,FALSE)""",[str(verify(launch['stock']))])
        count=c.execute('SELECT count(*) FROM missing').fetchone()[0]
        if not count:raise ValueError('No eligible missing public address')
        mandate.guard(launch['deadline_unix']);state('COMPILING_BAN_FOR_ALL_SOURCE_STREETS',eligible_missing_public_sites=count)
        # Semi-join source-wide address keys, independent of all CRM queries.
        raw=out/'ban_points.pending.parquet'
        result=c.execute("""SELECT code_commune commune,streetkey(nom_voie) street,
            textkey(numero) number,textkey(rep) suffix,longitude,latitude,type_position,source_position,
            certification_commune,address_id_hash,address_record_hash
            FROM read_parquet(?) b WHERE score_eligible IS FALSE
            AND latitude BETWEEN 40 AND 52 AND longitude BETWEEN -6 AND 10
            AND isfinite(latitude) AND isfinite(longitude)
            AND (code_commune,streetkey(nom_voie)) IN (SELECT commune,street FROM missing)""",[str(verify(launch['ban']))]).fetch_record_batch(rows_per_batch=65536)
        writer=None;points=0
        try:
            for batch in result:
                mandate.guard(launch['deadline_unix']);t=pa.Table.from_batches([batch])
                x,y=projected(t['longitude'].to_numpy(),t['latitude'].to_numpy())
                uncertainty=[];quality=[]
                for p,s,cert in zip(t['type_position'].to_pylist(),t['source_position'].to_pylist(),t['certification_commune'].to_pylist()):
                    args=dict(type_position=p,source_position=s,certified=bool(cert),street_score=1.)
                    uncertainty.append(geometry.ban_uncertainty_m(**args));quality.append(.5 if geometry.ban_declared_weak(**args) else 1.)
                t=t.drop(['longitude','latitude'])
                t=t.append_column('x',pa.array(x)).append_column('y',pa.array(y))
                t=t.append_column('uncertainty_m',pa.array(uncertainty)).append_column('point_quality',pa.array(quality))
                if writer is None:writer=pq.ParquetWriter(raw,t.schema,compression='zstd')
                writer.write_table(t);points+=len(t);state('PROJECTING_ALL_MATCHING_PUBLIC_BAN_POINTS',ban_points=points,eligible_missing_public_sites=count)
        finally:
            if writer is not None:writer.close()
        if not points:raise ValueError('Empty national matching BAN points')
        os.replace(raw,out/'ban_points.parquet');engine.seal(out/'ban_points.parquet')
        mandate.guard(launch['deadline_unix']);state('AGGREGATING_HOUSES_AND_FULL_STREET_FOOTPRINTS',ban_points=points)
        c.execute("""CREATE TABLE houses AS SELECT commune,street,number,suffix,
            avg(x) x,avg(y) y,count(*) points,min(point_quality) quality,
            max(uncertainty_m)+sqrt(pow(max(x)-min(x),2)+pow(max(y)-min(y),2)) uncertainty
            FROM read_parquet(?) GROUP BY commune,street,number,suffix""",[str(out/'ban_points.parquet')])
        c.execute("""CREATE TABLE streets AS SELECT commune,street,avg(x) x,avg(y) y,count(*) points,
            max(uncertainty_m)+sqrt(pow(max(x)-min(x),2)+pow(max(y)-min(y),2)) uncertainty
            FROM read_parquet(?) GROUP BY commune,street""",[str(out/'ban_points.parquet')])
        raw=out/'site_geometry.pending.parquet'
        relation=c.execute("""SELECT m.site_key,
            coalesce(h.x,s.x) x,coalesce(h.y,s.y) y,
            coalesce(h.uncertainty,s.uncertainty) uncertainty_m,
            CASE WHEN h.x IS NOT NULL THEN h.quality ELSE .25 END reliability,
            CASE WHEN h.x IS NOT NULL THEN 'EXACT_HOUSE_PUBLIC_OWN_ADDRESS'
                 ELSE 'FULL_STREET_FOOTPRINT_NOT_HOUSE' END method,
            coalesce(h.points,s.points) source_points
            FROM missing m JOIN streets s USING(commune,street)
            LEFT JOIN houses h ON m.commune=h.commune AND m.street=h.street
                AND m.number=h.number AND m.suffix=h.suffix AND m.number<>''
            WHERE coalesce(h.x,s.x) BETWEEN 0 AND 1300000 AND coalesce(h.y,s.y) BETWEEN 6000000 AND 7300000
            ORDER BY m.site_key""")
        table=relation.fetch_arrow_table();pq.write_table(table,raw,compression='zstd')
        os.replace(raw,out/'site_geometry.parquet');engine.seal(out/'site_geometry.parquet')
        methods={r[0]:r[1] for r in c.execute('SELECT method,count(*) FROM read_parquet(?) GROUP BY method',[str(out/'site_geometry.parquet')]).fetchall()}
        authority(out);verify(launch['stock']);verify(launch['ban']);verify(launch['ban_catalog'])
        io.write(out/'generation.json',dict(status='COMPLETE_NATIONAL_PUBLIC_OWN_ADDRESS_GEOMETRY_ATLAS',
            eligibility='DIAGNOSTIC_ONLY_BAN_SCORE_INELIGIBLE',production_promotion_allowed=False,
            raw_ban_lonlat_persisted=False,ban_catalog=launch['ban_catalog'],
            sites=len(table),eligible_missing_public_sites=count,unresolved=count-len(table),methods=methods,
            all_unresolved_retained_in_future_retrieval_denominator=True,ban_points=points,
            geometry=record(out/'site_geometry.parquet'),ban_point_provenance=record(out/'ban_points.parquet'),
            labels_used=False,no_gt_directed_source_repair=True,not_recall_measurement=True,elapsed_s=time.monotonic()-started))
        state('COMPLETE_SOURCE_ATLAS_NO_RECALL',sites=len(table),methods=methods)
    finally:c.close()


def run(out):
    import fcntl
    import subprocess
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v60_public_site_ban_diagnostic','prepare','--output',str(out)]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out))
    with (out/'prepare.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        io.write(out/'prepare_process.json',dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
        while child.poll() is None:
            if engine.INTERRUPTED or time.time()>=mandate.DEADLINE-30:
                child.terminate()
                try:child.wait(timeout=25)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            time.sleep(5)
        rc=child.wait()
    io.write(out/'prepare_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
    if rc:raise RuntimeError(f'Atlas owned exit {rc}; no incomplete source admitted')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare','run']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:globals()[a.action](a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise


