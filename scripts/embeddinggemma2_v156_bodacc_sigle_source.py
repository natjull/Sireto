"""Whole19-year public corporate SIGLE source; no index/native/labels.

Only actual personne.sigle on its same strict pm object/registration.
All normalized official aliases retained, then same-owner exact dedup against
current SIRENE SIGLE and the complete legacy canonical alias dictionary.
Coverage is source-only; confidence uses max ONE alias, never token synthesis.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','ARROW_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[name]='1'

import numpy as np
import psutil
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as words
from src.xgb_matcher.retrieval_variants_v118 import normalized

OUT=io.HOME/'v156/bodacc_sigle_20261014_night'
CANONICAL=io.HOME/'v112/bodacc_activity_context_20261013_a'
ALIASES=io.HOME/'v14/full_validation_20261008_a/names/generation.json'
CUTOFF='2026-08-01'
DEADLINE=1791954368
RECIPE=dict(scope='ALL19CANONICAL_YEARS_DIRECT_CORPORATE_ACTUAL_SIGLE_NO_QUERY_OR_LABEL_SOURCE_SELECTION',
    field='listepersonnes.personne.sigle',subject='SAME_PERSONNE_OBJECT_OWN9DIGIT_REGISTRATION_PM_CURRENT_O_NON_EI',
    arrays='COUNTED_SKIPPED_NOT_CROSS_PRODUCTS',source_cutoff=CUTOFF,publication_not_effective_date=True,
    normalization='EXACT_UNCHANGED_RETRIEVAL_VARIANTS_V118_NORMALIZED',no_synthetic_initialism_or_crm_alias=True,
    dedup='EXACT_SAME_OWNER_NORMALIZED_NAME_VS_CURRENT_SIRENE_SIGLE_AND_COMPLETE_LEGACY_CANONICAL_ALIASES',
    existing_dictionary='CURRENT_HISTORY_SIRENE_AND_CANONICAL_RNE_BODACC_TYPED_OWNER_NAMES_NOT_ASOF_CERTIFIED',
    no_separate_raw_rne_rescan=True,dictionary_sources_not_individually_identifiable_after_sqlite_projection=True,
    confidence='.5_MAX_ONE_SIGLE_QUERY_TOKEN_OWNER_IDF_COVERAGE_NO_TOKEN_UNION_BETWEEN_ALIASES',
    idf='LOG((ALL_RAW_VALID_PUBLIC_SIGLE_OWNERS+1)/(ALL_RAW_TOKEN_DISTINCT_OWNER_DF+1))/LOG(N+1)',
    confidence_cap=.5,all526=True,no_topk=True,no_labels_or_outcomes=True,
    owner_names_not_site_or_current_identity=True,typed_owner_TRADE_source_not_serving_alias=True,
    source_identifiers='TECHNICAL_BINDING_ONLY_NOT_FEATURES',no_person_name_address_administration_read_or_export=True,
    exact_whole_normalized_query_name_counts='SEPARATE_SOURCE_DIAGNOSTIC_NOT_SCORE_OR_CONFIDENCE_OVERRIDE',
    threads=1,duckdb_memory='1GB',max_rss_bytes=2*1024**3,free25=True,pressure_levels=[0,1,2],disk_reserve_bytes=40*1024**3,
    deadline_unix=DEADLINE,no_native_no_fit_no_index_or_corpus_or_serving_change=True)


def guard():
    free,level=io.pressure();rss=psutil.Process().memory_info().rss;disk=psutil.disk_usage(str(io.HOME)).free
    if time.time()>=DEADLINE or rss>RECIPE['max_rss_bytes'] or free<25 or level not in (0,1,2) or disk<RECIPE['disk_reserve_bytes']:
        raise RuntimeError(f'Source yields rss={rss} free={free} level={level} disk={disk}; no partial source admission')


def record(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(4*1024**2):h.update(chunk)
    return dict(path=str(path.resolve()),sha256=h.hexdigest(),size_bytes=path.stat().st_size)


def verify(ref):
    if record(ref['path'])!=ref:raise ValueError('Frozen source authority drift: '+ref['path'])
    return Path(ref['path'])


def initialize():
    guard()
    if OUT.exists():raise ValueError('Fresh immutable missing-field source namespace')
    g=io.read(CANONICAL/'generation.json');ex=io.read(CANONICAL/'owned_tool_exit.json')
    if ex['exit_code'] or not ex['own_tool_exit_confirmed'] or g['query_count']!=526:
        raise ValueError('Original canonical19-year source authority')
    old=io.read(verify(g['launch']));panel=io.read(verify(old['query_panel']))
    alias=io.read(ALIASES)
    if len(old['sources'])!=19 or len(panel)!=526 or any(q['fold'] not in (2,3,4) for q in panel):
        raise ValueError('Whole source and safe ALL526 query input')
    if alias['status']!='SEALED_OFFICIAL_NAMES_ALL_OWNERS' or Path(alias['database']['path']).stat().st_mtime_ns!=alias['mtime_ns']:
        raise ValueError('Original complete owner dictionary')
    OUT.mkdir(parents=True);(OUT/'spill').mkdir();(OUT/'raw_shards').mkdir();(OUT/'novel_shards').mkdir()
    io.write(OUT/'recipe.json',RECIPE)
    io.write(OUT/'launch.json',dict(implementation=record(Path(__file__)),recipe=record(OUT/'recipe.json'),
        informative_tokenizer=record(Path(words.__file__)),normalizer=record(io.ROOT/'src/xgb_matcher/retrieval_variants_v118.py'),
        query_panel=old['query_panel'],sources=old['sources'],public_units=old['public_units'],
        aliases_generation=record(ALIASES),legacy_alias_database=alias['database'],legacy_alias_mtime_ns=alias['mtime_ns'],
        deadline_unix=DEADLINE,labels_read=False,no_protected_target_or_source_selection=True,created_unix=time.time()))


def write_batch(con,name,rows,table):
    import pyarrow as pa
    if not rows:return
    con.register(name,pa.Table.from_pylist(rows));con.execute(f'INSERT INTO {table} SELECT * FROM {name}');con.unregister(name)


def export_shards(con,table,directory,queries):
    result={};counts={}
    for qid in queries:
        guard();n=con.execute(f'SELECT count(*) FROM {table} WHERE query_id=?',[qid]).fetchone()[0]
        refs=np.empty(n,dtype='S9');confidence=np.empty(n,dtype=np.float64)
        cur=con.execute(f'SELECT subject,confidence FROM {table} WHERE query_id=? ORDER BY subject',[qid]);at=0
        while rows:=cur.fetchmany(8192):
            end=at+len(rows);refs[at:end]=[r[0] for r in rows];confidence[at:end]=[r[1] for r in rows];at=end
        if at!=n or len(np.unique(refs))!=n or not np.isfinite(confidence).all() or (confidence<0).any() or (confidence>.500001).any():
            raise ValueError('Whole bounded owner source confidence')
        path=OUT/directory/(hashlib.sha256(qid.encode()).hexdigest()+'.npz');np.savez_compressed(path,refs=refs,confidence=confidence)
        result[qid]=record(path);counts[qid]=n
    return result,counts


def worker():
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1);pa.set_io_thread_count(1);guard()
    l=io.read(OUT/'launch.json')
    for k in ('implementation','recipe','informative_tokenizer','normalizer','query_panel','aliases_generation'):verify(l[k])
    stock=verify(l['public_units']);legacy=verify(l['legacy_alias_database'])
    if legacy.stat().st_mtime_ns!=l['legacy_alias_mtime_ns']:raise ValueError('Alias epoch drift')
    qs={q['query_id']:words.tokens(q['crm_name']) for q in io.read(verify(l['query_panel']))}
    wanted=set().union(*qs.values());started=time.monotonic()
    con=duckdb.connect(str(OUT/'source.duckdb'));con.execute('SET threads=1');con.execute("SET memory_limit='1GB'")
    con.execute('SET preserve_insertion_order=false');con.execute('SET temp_directory=?',[str(OUT/'spill')])
    con.execute('CREATE TABLE IF NOT EXISTS sigle_publications(subject VARCHAR,name_text VARCHAR,publication_id VARCHAR,publication_date DATE,publication_edition VARCHAR,notice_type VARCHAR,notice_family VARCHAR)')
    con.execute("CREATE TEMP TABLE public_units AS SELECT siren FROM read_parquet(?) WHERE statutDiffusionUniteLegale='O' AND categorieJuridiqueUniteLegale<>'1000'",[str(stock)])
    progress=io.read(OUT/'progress.json') if (OUT/'progress.json').exists() else dict(years=[],coverage=[])
    for src in l['sources']:
        guard();verify(src['manifest']);path=verify(src['source'])
        if str(path) in progress['years']:continue
        cols=pq.read_schema(path).names;nt='typeavis' if 'typeavis' in cols else 'CAST(NULL AS VARCHAR)';nf='familleavis' if 'familleavis' in cols else 'CAST(NULL AS VARCHAR)'
        con.read_parquet(str(path)).create_view('ann',replace=True)
        summary=con.execute("""SELECT count(*) total,count(*) FILTER(WHERE dateparution<=CAST(? AS DATE)) before_cutoff,
          count(*) FILTER(WHERE dateparution<=CAST(? AS DATE) AND json_type(listepersonnes,'$.personne')='ARRAY') arrays_skipped FROM ann""",[CUTOFF,CUTOFF]).fetchone()
        con.execute(f"""CREATE OR REPLACE TEMP TABLE eligible AS SELECT
            regexp_replace(json_extract_string(a.listepersonnes,'$.personne.numeroImmatriculation.numeroIdentification'),'[^0-9]','','g') subject,
            json_extract_string(a.listepersonnes,'$.personne.sigle') raw_sigle,CAST(a.id AS VARCHAR) publication_id,
            a.dateparution publication_date,CAST(a.publicationavis AS VARCHAR) publication_edition,{nt} notice_type,{nf} notice_family
            FROM ann a JOIN public_units u ON u.siren=regexp_replace(json_extract_string(a.listepersonnes,
                '$.personne.numeroImmatriculation.numeroIdentification'),'[^0-9]','','g')
            WHERE a.dateparution<=CAST(? AS DATE) AND json_type(a.listepersonnes,'$.personne')='OBJECT'
            AND json_extract_string(a.listepersonnes,'$.personne.typePersonne')='pm'
            AND length(regexp_replace(json_extract_string(a.listepersonnes,'$.personne.numeroImmatriculation.numeroIdentification'),'[^0-9]','','g'))=9
            AND coalesce(json_extract_string(a.listepersonnes,'$.personne.sigle'),'')<>''""",[CUTOFF])
        total=con.execute('SELECT count(*) FROM eligible').fetchone()[0];rejected=0;written=0
        writer=con.cursor();reader=con.execute('SELECT * FROM eligible').fetch_record_batch(4096)
        for batch in reader:
            guard();items=[]
            for r in batch.to_pylist():
                name=normalized(r['raw_sigle'])
                if not name:rejected+=1;continue
                items.append(dict(subject=r['subject'],name_text=name,publication_id=r['publication_id'],
                    publication_date=r['publication_date'],publication_edition=r['publication_edition'],notice_type=r['notice_type'],notice_family=r['notice_family']))
            write_batch(writer,'sigle_batch',items,'sigle_publications');written+=len(items)
        writer.close();con.execute('DROP TABLE eligible');con.execute('CHECKPOINT');verify(src['source'])
        progress['years'].append(str(path));progress['coverage'].append(dict(source=src['source'],total_rows=summary[0],
            before_cutoff=summary[1],person_arrays_skipped=summary[2],public_corporate_raw_sigle_rows=total,
            normalization_empty_mask_or_identifier_rejected=rejected,normalized_sigle_observations=written,
            notice_type_available='typeavis' in cols,notice_family_available='familleavis' in cols))
        io.write(OUT/'progress.json',progress);io.write(OUT/'state.json',dict(status='WHOLE_SOURCE_SIGLE_SCANNING',years=len(progress['years']),updated_unix=time.time()))
        print('YEAR',len(progress['years']),total,written,flush=True)
    con.execute('CREATE OR REPLACE TABLE owner_names AS SELECT DISTINCT subject,name_text FROM sigle_publications')
    raw_names=con.execute('SELECT count(*) FROM owner_names').fetchone()[0];owners=con.execute('SELECT count(DISTINCT subject) FROM owner_names').fetchone()[0]
    # Disk-backed exact same-owner membership, not a query-target-specific
    # lookup. Read-only attachment leaves the3GiB official dictionary intact.
    scratch=sqlite3.connect(str(OUT/'dedup.sqlite'),uri=True);scratch.execute('PRAGMA cache_size=-32768');scratch.execute('PRAGMA temp_store=FILE')
    scratch.execute('CREATE TABLE IF NOT EXISTS pairs(subject TEXT,name_text TEXT,known INTEGER DEFAULT 0,current_sigle INTEGER DEFAULT 0,PRIMARY KEY(subject,name_text)) WITHOUT ROWID')
    scratch.execute('ATTACH DATABASE ? AS official',(f'file:{legacy}?mode=ro',))
    cur=con.execute('SELECT subject,name_text FROM owner_names ORDER BY subject,name_text')
    while batch:=cur.fetchmany(8192):
        guard();scratch.executemany('INSERT OR IGNORE INTO pairs(subject,name_text) VALUES(?,?)',batch);scratch.commit()
    scratch.execute('UPDATE pairs SET known=EXISTS(SELECT 1 FROM official.aliases a WHERE a.kind=1 AND a.owner_key=CAST(pairs.subject AS INTEGER) AND a.name=pairs.name_text)');scratch.commit()
    # Additional explicit current SIRENE SIGLE check. All source owner refs
    # are public corporate already; no physical-person name field is read.
    con.execute('CREATE OR REPLACE TEMP TABLE sigle_owners AS SELECT DISTINCT subject FROM owner_names')
    current=con.execute('SELECT u.siren,u.sigleUniteLegale FROM read_parquet(?) u JOIN sigle_owners s ON s.subject=u.siren WHERE u.statutDiffusionUniteLegale=\'O\' AND u.categorieJuridiqueUniteLegale<>\'1000\' AND coalesce(u.sigleUniteLegale,\'\')<>\'\'',[str(stock)])
    while batch:=current.fetchmany(8192):
        guard();pairs=[(str(s),normalized(n)) for s,n in batch if normalized(n)]
        scratch.executemany('UPDATE pairs SET current_sigle=1 WHERE subject=? AND name_text=?',pairs);scratch.commit()
    con.execute('CREATE TABLE IF NOT EXISTS novelty(subject VARCHAR,name_text VARCHAR,already_admitted BOOLEAN,current_sigle BOOLEAN)')
    cur=scratch.execute('SELECT subject,name_text,known,current_sigle FROM pairs ORDER BY subject,name_text')
    while batch:=cur.fetchmany(8192):
        guard();write_batch(con,'novelty_batch',[dict(subject=s,name_text=n,already_admitted=bool(k),current_sigle=bool(c)) for s,n,k,c in batch],'novelty')
    scratch.close()
    # Informative tokens are mined only after every raw alias is fixed and
    # deduplicated. IDF denominator/DF cover ALL valid raw SIGLE owners.
    con.execute('CREATE TABLE IF NOT EXISTS name_tokens(subject VARCHAR,name_text VARCHAR,token VARCHAR)')
    reader=con.execute('SELECT subject,name_text FROM owner_names').fetch_record_batch(4096);writer=con.cursor()
    for batch in reader:
        guard();items=[dict(subject=r['subject'],name_text=r['name_text'],token=t) for r in batch.to_pylist() for t in words.tokens(r['name_text'])&wanted]
        write_batch(writer,'name_token_batch',items,'name_tokens')
    writer.close();dfs=dict(con.execute('SELECT token,count(DISTINCT subject) FROM name_tokens GROUP BY token').fetchall())
    idf={t:math.log((owners+1)/(dfs.get(t,0)+1))/math.log(owners+1) if owners else 0. for t in wanted}
    weights=[dict(query_id=qid,token=t,idf=idf[t]) for qid,ts in qs.items() for t in sorted(ts)]
    norms=[dict(query_id=qid,denominator=math.fsum(idf[t] for t in sorted(ts))) for qid,ts in qs.items()]
    con.register('query_weights',pa.Table.from_pylist(weights));con.register('query_norms',pa.Table.from_pylist(norms))
    con.execute("""CREATE OR REPLACE TABLE raw_alias_scores AS SELECT w.query_id,t.subject,t.name_text,.5*sum(w.idf)/max(n.denominator) confidence
        FROM name_tokens t JOIN query_weights w USING(token) JOIN query_norms n USING(query_id)
        WHERE n.denominator>0 GROUP BY w.query_id,t.subject,t.name_text HAVING confidence>0""")
    con.execute('CREATE OR REPLACE TABLE raw_query_context AS SELECT query_id,subject,max(confidence) confidence FROM raw_alias_scores GROUP BY query_id,subject ORDER BY query_id,subject')
    con.execute('CREATE OR REPLACE TABLE novel_query_context AS SELECT s.query_id,s.subject,max(s.confidence) confidence FROM raw_alias_scores s JOIN novelty n USING(subject,name_text) WHERE NOT n.already_admitted AND NOT n.current_sigle GROUP BY s.query_id,s.subject ORDER BY s.query_id,s.subject')
    raw_shards,raw_counts=export_shards(con,'raw_query_context','raw_shards',qs)
    novel_shards,novel_counts=export_shards(con,'novel_query_context','novel_shards',qs)
    counts=con.execute('SELECT count(*),count(*) FILTER(WHERE already_admitted),count(*) FILTER(WHERE current_sigle),count(*) FILTER(WHERE NOT already_admitted AND NOT current_sigle),count(DISTINCT subject) FILTER(WHERE NOT already_admitted AND NOT current_sigle) FROM novelty').fetchone()
    con.execute('COPY (SELECT * FROM novelty ORDER BY subject,name_text) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(OUT/'owner_sigles_novelty.parquet')])
    con.execute('COPY (SELECT DISTINCT * FROM sigle_publications ORDER BY subject,name_text,publication_date,publication_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(OUT/'sigle_publication_provenance.parquet')])
    query_names={q['query_id']:normalized(q['crm_name']) for q in io.read(verify(l['query_panel']))}
    exact_raw={};exact_novel={}
    for qid,name in query_names.items():
        guard()
        exact_raw[qid]=con.execute('SELECT count(DISTINCT subject) FROM novelty WHERE name_text=?',[name]).fetchone()[0] if name else 0
        exact_novel[qid]=con.execute('SELECT count(DISTINCT subject) FROM novelty WHERE name_text=? AND NOT already_admitted AND NOT current_sigle',[name]).fetchone()[0] if name else 0
    io.write(OUT/'query_coverage.json',dict(raw=raw_counts,novel=novel_counts,
        raw_exact_whole_normalized_name=exact_raw,novel_exact_whole_normalized_name=exact_novel,
        exact_counts_not_used_as_confidence_or_scoring=True))
    io.write(OUT/'idf.json',dict(all_raw_valid_sigle_owners=owners,tokens={t:dict(df=dfs.get(t,0),idf=idf[t]) for t in sorted(wanted)}))
    con.execute('CHECKPOINT');con.close();verify(l['public_units']);verify(l['implementation']);verify(l['informative_tokenizer']);verify(l['normalizer']);guard()
    if legacy.stat().st_mtime_ns!=l['legacy_alias_mtime_ns'] or legacy.stat().st_size!=l['legacy_alias_database']['size_bytes']:
        raise ValueError('Read-only alias dictionary changed')
    io.write(OUT/'generation.json',dict(status='COMPLETE_SEALED_WHOLE19YEAR_PUBLIC_CORPORATE_SIGLE_SOURCE_ALL526',
        launch=record(OUT/'launch.json'),implementation=l['implementation'],recipe=record(OUT/'recipe.json'),query_panel=l['query_panel'],
        source_owner_denominator=owners,raw_unique_owner_sigle_pairs=raw_names,novelty_counts=dict(owner_sigle_pairs=counts[0],
            already_admitted_dictionary=counts[1],same_current_sirene_sigle=counts[2],novel_owner_sigle_pairs=counts[3],novel_source_owners=counts[4]),
        coverage=progress['coverage'],query_count=len(qs),raw_queries_with_context=sum(bool(n) for n in raw_counts.values()),
        novel_queries_with_context=sum(bool(n) for n in novel_counts.values()),raw_owner_refs=sum(raw_counts.values()),novel_owner_refs=sum(novel_counts.values()),
        raw_shards=raw_shards,novel_shards=novel_shards,provenance=record(OUT/'sigle_publication_provenance.parquet'),
        owner_sigles=record(OUT/'owner_sigles_novelty.parquet'),idf=record(OUT/'idf.json'),query_coverage=record(OUT/'query_coverage.json'),
        no_labels_no_recall_no_native_no_fit_no_serving_or_index_change=True,source_aliases_are_actual_published_sigle_only=True,
        no_physical_person_name_address_or_administration_extracted=True,legacy_dedup_not_asof_certification=True,
        rne_and_sirene_other_field_dedup_via_full_mixed_dictionary_not_individually_attributed=True,
        actual_owned_exit0_required_before_any_label_side=True,elapsed_seconds=time.monotonic()-started))
    print('COMPLETE',owners,raw_names,counts,sum(bool(n) for n in novel_counts.values()),flush=True)


def run():
    command=[sys.executable,'-u','-m','scripts.embeddinggemma2_v156_bodacc_sigle_source','worker'];peak=0;stopped=None
    with (OUT/'worker.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(OUT/'spill')))
        io.write(OUT/'owned_process.json',dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
        while child.poll() is None:
            try:rss=psutil.Process(child.pid).memory_info().rss;peak=max(peak,rss)
            except psutil.NoSuchProcess:break
            free,level=io.pressure();disk=psutil.disk_usage(str(io.HOME)).free
            if time.time()>=DEADLINE-20 or rss>2*1024**3 or free<25 or level not in (0,1,2) or disk<40*1024**3:
                stopped=dict(rss=rss,free=free,level=level,disk=disk,deadline=time.time()>=DEADLINE-20)
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            time.sleep(3)
        rc=child.wait()
    io.write(OUT/'owned_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,
        observed_worker_peak_rss_bytes=peak,actual_guard_stop=stopped,finished_unix=time.time()))
    if rc:raise RuntimeError(f'Whole source exit{rc}; no partial source admission')
    print('OWNED_SOURCE_EXIT0',child.pid,peak,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','worker','run']);a=p.parse_args()
    try:globals()[a.action]()
    except BaseException as error:
        if OUT.exists():io.write(OUT/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
