"""Whole-source query-only direct corporate activity, separate from V112.

Reads ONLY strict personne OBJECT's own activite/registration/type. Arrays
are counted/skipped, no establishment/actor cross-product, no CRM labels.
Disk-backed aggregates keep all evidence/owners without in-memory pruning.
"""
import argparse
from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','ARROW_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import psutil
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as words

OUT = io.HOME/'v154/direct_corporate_activity_20261014_night'
PRIOR = io.HOME/'v112/bodacc_activity_context_20261013_a'
DEADLINE = 1791954368
CUTOFF = '2026-08-01'
RECIPE = dict(scope='WHOLE_CANONICAL_19YEAR_DIRECT_SINGLE_CORPORATE_SUBJECT_ACTIVITY',
    source_field='listepersonnes.personne.activite', subject_binding='SAME_PERSONNE_OBJECT_OWN_REGISTRATION',
    parser='PERSONNE_OBJECT_ONLY_TYPE_PM_9DIGIT_CURRENT_PUBLIC_NON_EI_LEGAL_UNIT',
    arrays='COUNTED_AND_SKIPPED', no_establishment_join=True, no_actor_cross_product=True,
    source_cutoff=CUTOFF, publication_not_effective_date=True,
    direct_activity_is_historical_owner_context_not_name_or_site_identity=True,
    confidence='SAME_V112_0.5_QUERY_TOKEN_OWNER_IDF_COVERAGE', confidence_cap=.5,
    owner_idf='LOG((ALL_DIRECT_ACTIVITY_OWNERS+1)/(DISTINCT_TOKEN_OWNER_DF+1))/LOG(ALL_DIRECT_ACTIVITY_OWNERS+1)',
    token_union_can_span_notices=True, deduplicate='DISTINCT_OWNER_TOKEN_MAX_NOT_ANNOUNCEMENT_SUM',
    notice_type='PRESERVE_WHEN_AVAILABLE_ELSE_NULL_NOT_INFERRED', publication_edition_preserved=True,
    all526_no_topk=True, labels_read=False, no_recall_or_model_training=True,
    identifiers='TECHNICAL_SUBJECT_AND_PUBLICATION_JOIN_ONLY_NOT_FEATURES',
    no_names_administration_people_addresses_or_activity_text_exported=True,
    threads=1, duckdb_memory_limit='128MB', rss_max_bytes=512*1024**2,
    disk_reserve_bytes=40*1024**3, free_admit_percent=25, pressure_levels=[0,1,2], deadline_unix=DEADLINE)


def guard():
    if time.time() >= DEADLINE:
        raise TimeoutError('Absolute night deadline; no sealed partial source')
    rss = psutil.Process().memory_info().rss
    free, level = io.pressure()
    disk = psutil.disk_usage(str(io.HOME)).free
    if rss > RECIPE['rss_max_bytes'] or free < 25 or level not in (0,1,2) or disk < RECIPE['disk_reserve_bytes']:
        raise RuntimeError(f'Resource yield rss={rss} free={free} pressure={level} disk={disk}')


def record(path):
    path = Path(path); digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4*1024**2):
            digest.update(chunk)
    return dict(path=str(path.resolve()),sha256=digest.hexdigest(),size_bytes=path.stat().st_size)


def verify(ref):
    if record(ref['path']) != ref:
        raise ValueError('Pinned source/helper drift: '+ref['path'])
    return Path(ref['path'])


def initialize():
    guard()
    if OUT.exists():
        raise ValueError('Fresh immutable whole-source namespace')
    prior = io.read(PRIOR/'generation.json'); ex = io.read(PRIOR/'owned_tool_exit.json')
    if prior['status'] != 'COMPLETE_WHOLE_SOURCE_QUERY_ONLY_BODACC_ACTIVITY_CONTEXT' or ex['exit_code'] or not ex['own_tool_exit_confirmed']:
        raise ValueError('Existing complete canonical activity source authority')
    original = io.read(verify(prior['launch']))
    panel = io.read(verify(original['query_panel']))
    if len(panel) != 526 or any(q['fold'] not in (2,3,4) for q in panel):
        raise ValueError('Exact safe input-only all526')
    if len(original['sources']) != 19:
        raise ValueError('All19 canonical years, no source-specific selection')
    OUT.mkdir(parents=True); (OUT/'spill').mkdir(); (OUT/'activity_shards').mkdir()
    io.write(OUT/'recipe.json',RECIPE)
    io.write(OUT/'launch.json',dict(implementation=record(Path(__file__)),recipe=record(OUT/'recipe.json'),
        tokenizer=record(Path(words.__file__)),source_authority=record(PRIOR/'generation.json'),
        source_actual_exit=record(PRIOR/'owned_tool_exit.json'),query_panel=original['query_panel'],
        sources=original['sources'],public_units=original['public_units'],deadline_unix=DEADLINE,
        labels_read=False,no_query_specific_source_selection=True,created_unix=time.time()))


def worker():
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    guard(); launch = io.read(OUT/'launch.json')
    for field in ('implementation','recipe','tokenizer','query_panel','source_authority','source_actual_exit'):
        verify(launch[field])
    stock = verify(launch['public_units'])
    panel = io.read(verify(launch['query_panel']))
    qs = {q['query_id']:words.tokens(q['crm_name']) for q in panel}
    wanted = set().union(*qs.values())
    pattern = '(^|[^A-Z0-9])('+'|'.join(re.escape(t) for t in sorted(wanted))+')([^A-Z0-9]|$)'
    con = duckdb.connect(str(OUT/'aggregate.duckdb'))
    con.execute('SET threads=1'); con.execute("SET memory_limit='128MB'")
    con.execute('SET preserve_insertion_order=false')
    con.execute('SET temp_directory=?',[str(OUT/'spill')])
    con.execute('CREATE TABLE IF NOT EXISTS all_owners(subject VARCHAR)')
    con.execute('CREATE TABLE IF NOT EXISTS evidence(token VARCHAR,subject VARCHAR,publication_id VARCHAR,publication_date DATE,publication_edition VARCHAR,notice_type VARCHAR,notice_family VARCHAR)')
    con.execute("CREATE TEMP TABLE public_units AS SELECT siren FROM read_parquet(?) WHERE statutDiffusionUniteLegale='O' AND categorieJuridiqueUniteLegale<>'1000'",[str(stock)])
    progress = io.read(OUT/'progress.json') if (OUT/'progress.json').exists() else dict(years=[],coverage=[])
    started = time.monotonic()
    for ref in launch['sources']:
        guard(); verify(ref['manifest']); source = verify(ref['source'])
        if str(source) in progress['years']:
            continue
        fields = pq.read_schema(source).names
        nt = 'typeavis' if 'typeavis' in fields else 'CAST(NULL AS VARCHAR)'
        nf = 'familleavis' if 'familleavis' in fields else 'CAST(NULL AS VARCHAR)'
        con.read_parquet(str(source)).create_view('ann',replace=True)
        counts = con.execute("""SELECT count(*) total,
          count(*) FILTER(WHERE dateparution<=CAST(? AS DATE)) before_cutoff,
          count(*) FILTER(WHERE dateparution<=CAST(? AS DATE) AND json_type(listepersonnes,'$.personne')='ARRAY') person_arrays_skipped,
          count(*) FILTER(WHERE dateparution<=CAST(? AS DATE) AND json_type(listepersonnes,'$.personne')='OBJECT'
            AND json_extract_string(listepersonnes,'$.personne.typePersonne')='pm') strict_corporate_object
          FROM ann""",[CUTOFF]*3).fetchone()
        con.execute(f"""CREATE OR REPLACE TEMP TABLE eligible AS SELECT CAST(a.id AS VARCHAR) publication_id,
            a.dateparution publication_date,CAST(a.publicationavis AS VARCHAR) publication_edition,
            {nt} notice_type,{nf} notice_family,
            regexp_replace(json_extract_string(a.listepersonnes,'$.personne.numeroImmatriculation.numeroIdentification'),'[^0-9]','','g') subject,
            json_extract_string(a.listepersonnes,'$.personne.activite') activity
            FROM ann a JOIN public_units u ON u.siren=regexp_replace(json_extract_string(a.listepersonnes,
                '$.personne.numeroImmatriculation.numeroIdentification'),'[^0-9]','','g')
            WHERE a.dateparution<=CAST(? AS DATE) AND json_type(a.listepersonnes,'$.personne')='OBJECT'
            AND json_extract_string(a.listepersonnes,'$.personne.typePersonne')='pm'
            AND length(regexp_replace(json_extract_string(a.listepersonnes,'$.personne.numeroImmatriculation.numeroIdentification'),'[^0-9]','','g'))=9
            AND coalesce(json_extract_string(a.listepersonnes,'$.personne.activite'),'')<>''""",[CUTOFF])
        n = con.execute('SELECT count(*) FROM eligible').fetchone()[0]
        con.execute('INSERT INTO all_owners SELECT DISTINCT subject FROM eligible')
        # Separate cursor permits disk-backed writes while bounded batches are
        # consumed. No text survives in exported evidence/owner tables.
        cursor = con.cursor()
        reader = con.execute('SELECT publication_id,publication_date,publication_edition,notice_type,notice_family,subject,activity FROM eligible WHERE regexp_matches(upper(strip_accents(activity)),?)',[pattern]).fetch_record_batch(2048)
        matched = 0; inserted = 0
        for batch in reader:
            guard(); pairs = []
            for row in batch.to_pylist():
                matched += 1
                for token in words.tokens(row['activity']) & wanted:
                    pairs.append(dict(token=token,subject=row['subject'],publication_id=row['publication_id'],
                        publication_date=row['publication_date'],publication_edition=row['publication_edition'],
                        notice_type=row['notice_type'],notice_family=row['notice_family']))
            if pairs:
                cursor.register('token_batch',pa.Table.from_pylist(pairs))
                cursor.execute('INSERT INTO evidence SELECT * FROM token_batch'); cursor.unregister('token_batch')
                inserted += len(pairs)
        cursor.close(); con.execute('DROP TABLE eligible'); con.execute('CHECKPOINT')
        verify(ref['source'])
        progress['years'].append(str(source))
        progress['coverage'].append(dict(source=ref['source'],total_rows=counts[0],before_cutoff=counts[1],
            person_arrays_skipped=counts[2],strict_corporate_objects=counts[3],eligible_direct_public_activity_rows=n,
            query_token_matching_rows=matched,observed_token_publication_rows=inserted,
            notice_type_available='typeavis' in fields,notice_family_available='familleavis' in fields))
        io.write(OUT/'progress.json',progress)
        io.write(OUT/'state.json',dict(status='COMPILING_WHOLE_SOURCE_DIRECT_CORPORATE_ACTIVITY',
            completed_years=len(progress['years']),elapsed_seconds=time.monotonic()-started,updated_unix=time.time()))
        print('YEAR',len(progress['years']),source.name,n,matched,flush=True)
    guard()
    owners = con.execute('SELECT count(DISTINCT subject) FROM all_owners').fetchone()[0]
    con.execute('CREATE OR REPLACE TABLE token_owner AS SELECT DISTINCT token,subject FROM evidence')
    dfs = dict(con.execute('SELECT token,count(*) FROM token_owner GROUP BY token').fetchall())
    idf = {t:math.log((owners+1)/(dfs.get(t,0)+1))/math.log(owners+1) if owners else 0. for t in wanted}
    weights = [dict(query_id=qid,token=t,idf=idf[t]) for qid,ts in qs.items() for t in sorted(ts)]
    norms = [dict(query_id=qid,denominator=math.fsum(idf[t] for t in sorted(ts))) for qid,ts in qs.items()]
    con.register('query_weights',pa.Table.from_pylist(weights)); con.register('query_norms',pa.Table.from_pylist(norms))
    con.execute("""CREATE OR REPLACE TABLE query_context AS SELECT w.query_id,t.subject,
        .5*sum(w.idf)/max(n.denominator) confidence
        FROM token_owner t JOIN query_weights w USING(token) JOIN query_norms n USING(query_id)
        WHERE n.denominator>0 GROUP BY w.query_id,t.subject HAVING confidence>0""")
    con.execute('CHECKPOINT')
    shards = {}; cumulative = 0; maximum = 0; positive_queries = 0
    for qid in qs:
        guard()
        n = con.execute('SELECT count(*) FROM query_context WHERE query_id=?',[qid]).fetchone()[0]
        refs = np.empty(n,dtype='S9'); confidence = np.empty(n,dtype=np.float64)
        cursor = con.execute('SELECT subject,confidence FROM query_context WHERE query_id=? ORDER BY subject',[qid])
        first = 0
        while rows := cursor.fetchmany(8192):
            end = first+len(rows)
            refs[first:end] = [r[0] for r in rows]; confidence[first:end] = [r[1] for r in rows]
            first = end
        if first != n:
            raise ValueError('Whole owner shard cursor count')
        if len(refs) != len(np.unique(refs)) or not np.isfinite(confidence).all() or (confidence<0).any() or (confidence>.500001).any():
            raise ValueError('Full source bounded unique owner confidence')
        target = OUT/'activity_shards'/(hashlib.sha256(qid.encode()).hexdigest()+'.npz')
        np.savez_compressed(target,refs=refs,confidence=confidence)
        shards[qid] = record(target); cumulative += len(refs); maximum = max(maximum,len(refs)); positive_queries += bool(n)
    provenance = OUT/'token_publication_provenance.parquet'
    con.execute("COPY (SELECT DISTINCT * FROM evidence ORDER BY token,subject,publication_date,publication_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",[str(provenance)])
    io.write(OUT/'token_owner_idf.json',dict(all_direct_activity_owners=owners,
        tokens={t:dict(df=dfs.get(t,0),idf=idf[t]) for t in sorted(wanted)}))
    con.execute('DROP TABLE public_units'); con.close()
    # Sources are immutable and have been hashed before/after each year. The
    # stock/helper authority is rechecked before this complete source seal.
    verify(launch['public_units']); verify(launch['implementation']); verify(launch['tokenizer']); guard()
    io.write(OUT/'generation.json',dict(status='COMPLETE_SEALED_WHOLE_SOURCE_DIRECT_CORPORATE_ACTIVITY_ALL526',
        implementation=launch['implementation'],launch=record(OUT/'launch.json'),recipe=record(OUT/'recipe.json'),
        query_panel=launch['query_panel'],source_owners=owners,coverage=progress['coverage'],
        query_count=len(shards),queries_with_context=positive_queries,
        cumulative_owner_references=cumulative,maximum_owner_references=maximum,activity_shards=shards,
        provenance=record(provenance),idf=record(OUT/'token_owner_idf.json'),
        owner_token_pairs=sum(dfs.values()),all_source_subjects_no_topk=True,
        no_labels_or_outcomes_read=True,no_alias_or_name_or_site_identity_created=True,
        source_only_no_recall_measured=True,current_public_subject_check=True,
        publication_dates_not_effective_dates=True,arrays_skipped_not_cross_product=True,
        complete_source_requires_actual_owned_exit0=True,elapsed_seconds=time.monotonic()-started))
    print('COMPLETE',owners,cumulative,positive_queries,flush=True)


def run():
    command=[sys.executable,'-u','-m','scripts.embeddinggemma2_v154_direct_corporate_activity_source','worker']
    with (OUT/'worker.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(OUT/'spill')))
        io.write(OUT/'owned_process.json',dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
        while child.poll() is None:
            if time.time() >= DEADLINE-20:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            time.sleep(3)
        rc=child.wait()
    io.write(OUT/'owned_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
    if rc:
        raise RuntimeError(f'Whole-source worker exit{rc}; no recall/partial source admitted')
    print('OWNED_SOURCE_EXIT0',child.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','worker','run']);a=p.parse_args()
    try:
        globals()[a.action]()
    except BaseException as error:
        if OUT.exists():io.write(OUT/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
