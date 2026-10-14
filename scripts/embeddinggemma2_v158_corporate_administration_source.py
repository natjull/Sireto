"""Strict historical corporate administration source, never personal directors.

Own public pm + explicit role colon + explicit legal form + exact unique
public corporate-name resolution. Ambiguous clauses are skipped, not repaired.
No administration text, natural names, addresses or labels are exported.
"""
import argparse
from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','ARROW_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_k]='1'
import numpy as np
import psutil
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v156_bodacc_sigle_source as prior
from src.xgb_matcher.retrieval_variants_v118 import normalized

OUT=io.HOME/'v158/corporate_administration_20261014_night'
RNE=io.DATA/'sources/rne_legal_entity_roles_v1/legal_entity_roles.parquet'
CUTOFF='2026-08-01'
DEADLINE=1791954368
ROLE=r'(?:COMMISSAIRE AUX COMPTES(?: TITULAIRE| SUPPLEANT)?|PRESIDENT(?: DU CONSEIL D ADMINISTRATION)?|VICE PRESIDENT|DIRECTEUR GENERAL(?: DELEGUE)?|GERANT|CO GERANT|ADMINISTRATEUR|ASSOCIE(?: UNIQUE)?)'
HEADER=re.compile(r'(?<![A-Z0-9])('+ROLE+r')\s*:\s*')
FORM=r'(?:SOCIETE PAR ACTIONS SIMPLIFIEE(?: UNIPERSONNELLE)?|SOCIETE A RESPONSABILITE LIMITEE(?: UNIPERSONNELLE)?|SOCIETE ANONYME|SOCIETE CIVILE IMMOBILIERE|SOCIETE EN NOM COLLECTIF|SASU|SAS|SARL|EURL|SCI|SNC|SELARL|SELAFA|SA)'
LEAD=re.compile(r'^(?:LA )?(?:SOCIETE )?('+FORM+r')\s+(.+)$')
TAIL=re.compile(r'^(.+?)\s+(?:EN FORME DE |FORME JURIDIQUE )?('+FORM+r')$')
FORBIDDEN=re.compile(r'\b(?:REPRESENTE|REPRESENTEE|REPRESENTES|REPRESENTEES|REPR|MONSIEUR|MADAME|MME|MR|M|NEE|NE|DOMICILIE|DOMICILIEE|ADRESSE|SIEGE|CAPITAL|RCS|IMMATRICULE|IMMATRICULEE|SIREN|SIRET)\b|[0-9]{9}|@')
RECIPE=dict(hypothesis='HISTORICAL_PUBLISHED_TYPED_CORPORATE_ADMINISTRATOR_CONTEXT_NOT_IDENTITY',
    own_subject='STRICT_PERSONNE_OBJECT_PM_OWN_REGISTRATION_CURRENT_PUBLIC_O_NON_EI',
    years=19,cutoff=CUTOFF,publication_not_effective_date=True,arrays='COUNTED_SKIPPED',
    parser='ROLE_COLON_CLAUSE_EXPLICIT_LEGAL_FORM_PREFIX_OR_SUFFIX_NO_AMBIGUOUS_LIST',
    role_expression=ROLE,legal_forms=FORM,natural_or_address_suffixes='REJECT_FULL_CLAUSE_NOT_TRIM_TO_REPAIR',
    exact_resolution='UNIQUE_NORMALIZED_NAME_CURRENT_PUBLIC_CORPORATE_SIRENE_PLUS_ADMITTED_PARENT_ALIASES',
    no_fuzzy_person_or_actor_resolution=True,no_crossproduct=True,no_self_role=True,
    no_ownership_current_affiliation_operator_identity_or_site_alias_claim=True,
    rne_novelty='CONSERVATIVE_PAIR_SUBJECT_ACTOR_ANY_PUBLISHED_RNE_ROLE_BEFORE_CUTOFF',
    rne_role_codes_not_equated_to_free_bodacc_role=True,legacy_alias_pool_not_asof_certified=True,
    confidence='.5_DIV_SQRT_ALL_LINKED_PUBLIC_SUBJECT_DEGREE_TIMES_QUERY_TOKEN_OWNER_IDF_COVERAGE_MAX_ACTOR_NAME',
    idf='LOG((ALL_RESOLVED_ACTORS+1)/(TOKEN_DISTINCT_ACTOR_DF+1))/LOG(N+1)',
    all526=True,no_topk=True,no_labels=True,no_native_fit_index_or_serving_change=True,
    no_raw_administration_person_name_contact_address_export=True,
    threads=1,duckdb_memory='512MB',rss_max_bytes=2*1024**3,free25=True,pressure_levels=[0,1,2],
    disk_reserve_bytes=40*1024**3,deadline_unix=DEADLINE)


def guard():
    free,level=io.pressure();rss=psutil.Process().memory_info().rss;disk=psutil.disk_usage(str(io.HOME)).free
    if time.time()>=DEADLINE or rss>2*1024**3 or free<25 or level not in (0,1,2) or disk<40*1024**3:
        raise RuntimeError(f'Source guard rss={rss} free={free} level={level} disk={disk}')


def clauses(text):
    # Only role-marked segments are inspected. No rejected text survives.
    import unicodedata
    value=''.join(c for c in unicodedata.normalize('NFKD',str(text or '').upper()) if not unicodedata.combining(c))
    value=value.replace('’',"'").replace("'",' ')
    headers=list(HEADER.finditer(value));counts=Counter();accepted=[]
    if not headers:counts['no_explicit_role_colon']+=1
    for i,h in enumerate(headers):
        end=headers[i+1].start() if i+1<len(headers) else len(value)
        piece=value[h.end():end].strip(' ;\r\n. ')
        if any(c in piece for c in ',;\n\r()'):
            counts['ambiguous_list_or_parenthesis']+=1;continue
        if FORBIDDEN.search(piece):counts['person_or_nonname_suffix_rejected']+=1;continue
        matched=LEAD.fullmatch(piece)
        name=matched.group(2) if matched else None
        if name is None:
            matched=TAIL.fullmatch(piece);name=matched.group(1) if matched else None
        if name is None:counts['no_explicit_corporate_form']+=1;continue
        name=normalized(name.strip(' "«»'))
        if not name or len(name)<2:counts['empty_or_invalid_name']+=1;continue
        accepted.append((normalized(h.group(1)),name));counts['explicit_corporate_clause']+=1
    return accepted,counts


def initialize():
    guard()
    if OUT.exists():raise ValueError('Fresh immutable corporate-administration namespace')
    source=io.read(prior.OUT/'generation.json');exit=io.read(prior.OUT/'owned_exit.json')
    if exit['exit_code'] or not exit['own_child_exit_confirmed']:raise ValueError('Owned whole-source authority')
    old=io.read(prior.verify(source['launch']));panel=io.read(prior.verify(old['query_panel']))
    if len(panel)!=526 or len(old['sources'])!=19 or any(q['fold'] not in (2,3,4) for q in panel):raise ValueError('All526 safe whole-source binding')
    OUT.mkdir(parents=True);(OUT/'spill').mkdir();(OUT/'raw_shards').mkdir();(OUT/'novel_shards').mkdir()
    io.write(OUT/'recipe.json',RECIPE)
    io.write(OUT/'launch.json',dict(implementation=prior.record(Path(__file__)),recipe=prior.record(OUT/'recipe.json'),
        source_authority=prior.record(prior.OUT/'generation.json'),source_exit=prior.record(prior.OUT/'owned_exit.json'),
        tokenizer=prior.record(Path(prior.words.__file__)),normalizer=old['normalizer'],query_panel=old['query_panel'],
        public_units=old['public_units'],sources=old['sources'],legacy_alias_database=old['legacy_alias_database'],
        legacy_alias_mtime_ns=old['legacy_alias_mtime_ns'],rne=prior.record(RNE),labels_read=False))


def batch_insert(con,table,rows):
    import pyarrow as pa
    if rows:
        con.register('batch_insert',pa.Table.from_pylist(rows));con.execute(f'INSERT INTO {table} SELECT * FROM batch_insert');con.unregister('batch_insert')


def worker():
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1);pa.set_io_thread_count(1);guard();start=time.monotonic();l=io.read(OUT/'launch.json')
    for k in ('implementation','recipe','source_authority','source_exit','tokenizer','normalizer','query_panel','rne'):prior.verify(l[k])
    stock=prior.verify(l['public_units']);legacy=prior.verify(l['legacy_alias_database'])
    panel=io.read(prior.verify(l['query_panel']));qs={q['query_id']:prior.words.tokens(q['crm_name']) for q in panel};wanted=set().union(*qs.values())
    c=duckdb.connect(str(OUT/'source.duckdb'));c.execute('SET threads=1');c.execute("SET memory_limit='512MB'");c.execute('SET preserve_insertion_order=false');c.execute('SET temp_directory=?',[str(OUT/'spill')])
    c.execute("CREATE TEMP TABLE public_units AS SELECT siren,denominationUniteLegale,denominationUsuelle1UniteLegale,denominationUsuelle2UniteLegale,denominationUsuelle3UniteLegale,sigleUniteLegale FROM read_parquet(?) WHERE statutDiffusionUniteLegale='O' AND categorieJuridiqueUniteLegale<>'1000'",[str(stock)])
    c.execute('CREATE TABLE clauses(subject VARCHAR,role VARCHAR,actor_name VARCHAR,publication_id VARCHAR,publication_date DATE,edition VARCHAR,notice_type VARCHAR,notice_family VARCHAR)')
    progress=[]
    for src in l['sources']:
        guard();prior.verify(src['manifest']);path=prior.verify(src['source']);fields=pq.read_schema(path).names
        nt='typeavis' if 'typeavis' in fields else 'CAST(NULL AS VARCHAR)';nf='familleavis' if 'familleavis' in fields else 'CAST(NULL AS VARCHAR)'
        c.read_parquet(str(path)).create_view('ann',replace=True)
        summary=c.execute("SELECT count(*),count(*) FILTER(WHERE dateparution<=CAST(? AS DATE)),count(*) FILTER(WHERE dateparution<=CAST(? AS DATE) AND json_type(listepersonnes,'$.personne')='ARRAY') FROM ann",[CUTOFF,CUTOFF]).fetchone()
        # Temporary text is never exported or retained as an evidence table.
        c.execute(f"""CREATE OR REPLACE TEMP TABLE eligible AS SELECT u.siren subject,
            json_extract_string(a.listepersonnes,'$.personne.administration') admin,CAST(a.id AS VARCHAR) publication_id,
            a.dateparution publication_date,CAST(a.publicationavis AS VARCHAR) edition,{nt} notice_type,{nf} notice_family
            FROM ann a JOIN public_units u ON u.siren=regexp_replace(json_extract_string(a.listepersonnes,'$.personne.numeroImmatriculation.numeroIdentification'),'[^0-9]','','g')
            WHERE a.dateparution<=CAST(? AS DATE) AND json_type(a.listepersonnes,'$.personne')='OBJECT'
            AND json_extract_string(a.listepersonnes,'$.personne.typePersonne')='pm'
            AND length(u.siren)=9 AND coalesce(json_extract_string(a.listepersonnes,'$.personne.administration'),'')<>''""",[CUTOFF])
        n=c.execute('SELECT count(*) FROM eligible').fetchone()[0];counts=Counter();writer=c.cursor()
        for batch in c.execute('SELECT * FROM eligible').fetch_record_batch(4096):
            guard();items=[]
            for r in batch.to_pylist():
                accepted,cc=clauses(r['admin']);counts.update(cc)
                for role,name in accepted:items.append(dict(subject=r['subject'],role=role,actor_name=name,publication_id=r['publication_id'],publication_date=r['publication_date'],edition=r['edition'],notice_type=r['notice_type'],notice_family=r['notice_family']))
            batch_insert(writer,'clauses',items)
        writer.close();c.execute('DROP TABLE eligible');c.execute('CHECKPOINT');prior.verify(src['source'])
        progress.append(dict(source=src['source'],total_rows=summary[0],before_cutoff=summary[1],arrays_skipped=summary[2],eligible_own_public_pm_admin_rows=n,parser_counts=dict(counts)))
        io.write(OUT/'progress.json',progress);print('YEAR',len(progress),n,dict(counts),flush=True)
    # Disk-backed exact matching. Natural-name clauses were rejected above;
    # unresolved or ambiguous corporate strings are never exported.
    names={r[0] for r in c.execute('SELECT DISTINCT actor_name FROM clauses').fetchall()};s=sqlite3.connect(str(OUT/'resolution.sqlite'))
    s.execute('PRAGMA cache_size=-32768');s.execute('PRAGMA temp_store=FILE')
    s.execute('CREATE TABLE names(name TEXT PRIMARY KEY) WITHOUT ROWID');s.executemany('INSERT INTO names VALUES(?)',((n,) for n in sorted(names)))
    s.execute('CREATE TABLE public_units(owner INTEGER PRIMARY KEY) WITHOUT ROWID');s.execute('CREATE TABLE matches(name TEXT,owner INTEGER,PRIMARY KEY(name,owner)) WITHOUT ROWID')
    for batch in c.execute('SELECT * FROM public_units').fetch_record_batch(4096):
        guard();rows=batch.to_pylist();s.executemany('INSERT INTO public_units VALUES(?)',((int(r['siren']),) for r in rows))
        pairs={(normalized(v),int(r['siren'])) for r in rows for k,v in r.items() if k!='siren' and v and normalized(v) in names}
        s.executemany('INSERT OR IGNORE INTO matches VALUES(?,?)',pairs)
    s.commit();s.execute('ATTACH DATABASE ? AS official',(f'file:{legacy}?mode=ro',))
    s.execute('INSERT OR IGNORE INTO matches SELECT a.name,a.owner_key FROM official.aliases a CROSS JOIN names n CROSS JOIN public_units u WHERE a.kind=1 AND a.name=n.name AND a.owner_key=u.owner');s.commit();guard()
    mapping=dict(s.execute('SELECT name,min(owner) FROM matches GROUP BY name HAVING count(*)=1'));ambiguous=s.execute('SELECT count(*) FROM (SELECT name FROM matches GROUP BY name HAVING count(*)>1)').fetchone()[0]
    s.close();c.register('resolved',pa.table({'actor_name':list(mapping),'actor':[str(v).zfill(9) for v in mapping.values()]}))
    c.execute('CREATE TABLE evidence AS SELECT x.*,r.actor FROM clauses x JOIN resolved r USING(actor_name) WHERE x.subject<>r.actor')
    c.execute('DROP TABLE clauses');c.execute('CHECKPOINT');del names,mapping
    c.execute("CREATE TEMP TABLE rne_pairs AS SELECT DISTINCT subject_siren subject,related_siren actor FROM read_parquet(?) WHERE (coalesce(effective_date,'')='' OR substr(CAST(effective_date AS VARCHAR),1,10)<=?) AND (coalesce(observed_at,'')='' OR substr(CAST(observed_at AS VARCHAR),1,10)<=?)",[str(prior.verify(l['rne'])),CUTOFF,CUTOFF])
    c.execute('CREATE TABLE relations AS SELECT DISTINCT e.subject,e.actor,e.actor_name,e.role,(r.subject IS NOT NULL) already_rne_pair FROM evidence e LEFT JOIN rne_pairs r USING(subject,actor)')
    c.execute('CREATE TABLE degrees AS SELECT actor,count(DISTINCT subject) degree FROM relations GROUP BY actor')
    owners=c.execute('SELECT count(DISTINCT actor) FROM relations').fetchone()[0];c.execute('CREATE TABLE actor_tokens(actor VARCHAR,actor_name VARCHAR,token VARCHAR)');writer=c.cursor()
    for batch in c.execute('SELECT DISTINCT actor,actor_name FROM relations').fetch_record_batch(4096):
        guard();batch_insert(writer,'actor_tokens',[dict(actor=r['actor'],actor_name=r['actor_name'],token=t) for r in batch.to_pylist() for t in prior.words.tokens(r['actor_name'])&wanted])
    writer.close();dfs=dict(c.execute('SELECT token,count(DISTINCT actor) FROM actor_tokens GROUP BY token').fetchall())
    idf={t:math.log((owners+1)/(dfs.get(t,0)+1))/math.log(owners+1) if owners else 0. for t in wanted}
    c.register('weights',pa.Table.from_pylist([dict(query_id=qid,token=t,idf=idf[t]) for qid,ts in qs.items() for t in sorted(ts)]));c.register('norms',pa.Table.from_pylist([dict(query_id=qid,norm=math.fsum(idf[t] for t in sorted(ts))) for qid,ts in qs.items()]))
    c.execute('CREATE TABLE actor_scores AS SELECT w.query_id,t.actor,t.actor_name,sum(w.idf)/max(n.norm) coverage FROM actor_tokens t JOIN weights w USING(token) JOIN norms n USING(query_id) WHERE n.norm>0 GROUP BY w.query_id,t.actor,t.actor_name HAVING coverage>0')
    for tag,where in [('raw',''),('novel','WHERE NOT r.already_rne_pair')]:
        c.execute(f'CREATE TABLE {tag}_query_context AS SELECT s.query_id,r.subject,max(.5*s.coverage/sqrt(d.degree)) confidence FROM actor_scores s JOIN relations r USING(actor,actor_name) JOIN degrees d USING(actor) {where} GROUP BY s.query_id,r.subject')
    raw_refs={};novel_refs={};coverage={}
    for qid in qs:
        guard();coverage[qid]={}
        for tag,refs in [('raw',raw_refs),('novel',novel_refs)]:
            rows=c.execute(f'SELECT subject,confidence FROM {tag}_query_context WHERE query_id=? ORDER BY subject',[qid]).fetchall();path=OUT/(tag+'_shards')/(hashlib.sha256(qid.encode()).hexdigest()+'.npz')
            vals=np.asarray([r[1] for r in rows],dtype=np.float64)
            if not np.isfinite(vals).all() or (vals<0).any() or (vals>.500001).any():raise ValueError('Bounded corporate context')
            np.savez_compressed(path,refs=np.asarray([r[0] for r in rows],dtype='S9'),confidence=vals);refs[qid]=prior.record(path);coverage[qid][tag]=len(rows)
    stats=dict(zip(('relations','subjects','actors','already_rne_pairs','novel_relations'),c.execute('SELECT count(*),count(DISTINCT subject),count(DISTINCT actor),count(*) FILTER(WHERE already_rne_pair),count(*) FILTER(WHERE NOT already_rne_pair) FROM relations').fetchone()))
    for table,name in [('relations','corporate_relations.parquet'),('evidence','publication_provenance.parquet')]:c.execute(f'COPY (SELECT DISTINCT * FROM {table} ORDER BY subject,actor) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(OUT/name)])
    c.execute('CHECKPOINT');c.close();guard()
    for k in ('implementation','public_units','rne'):prior.verify(l[k])
    if legacy.stat().st_mtime_ns!=l['legacy_alias_mtime_ns']:raise ValueError('Read-only dictionary epoch drift')
    io.write(OUT/'query_coverage.json',coverage)
    io.write(OUT/'generation.json',dict(status='COMPLETE_WHOLE19YEAR_CORPORATE_ADMINISTRATION_SOURCE_NOT_RECALL',launch=prior.record(OUT/'launch.json'),recipe=prior.record(OUT/'recipe.json'),coverage=progress,
        stats=stats,exact_resolved_ambiguous_names_skipped=ambiguous,raw_shards=raw_refs,novel_shards=novel_refs,query_panel=l['query_panel'],query_count=526,
        raw_queries_with_context=sum(bool(v['raw']) for v in coverage.values()),novel_queries_with_context=sum(bool(v['novel']) for v in coverage.values()),
        provenance=prior.record(OUT/'publication_provenance.parquet'),relations=prior.record(OUT/'corporate_relations.parquet'),query_coverage=prior.record(OUT/'query_coverage.json'),
        all_resolved_actors_denominator=owners,no_personal_or_administration_raw_export=True,no_native_or_labels=True,legacy_dictionary_not_asof_certified=True,elapsed_seconds=time.monotonic()-start))
    print('COMPLETE',stats,flush=True)


def run():
    command=[sys.executable,'-u','-m','scripts.embeddinggemma2_v158_corporate_administration_source','worker'];peak=0;stop=None
    with (OUT/'worker.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(OUT/'spill')))
        io.write(OUT/'owned_process.json',dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
        while child.poll() is None:
            try:rss=psutil.Process(child.pid).memory_info().rss;peak=max(peak,rss)
            except psutil.NoSuchProcess:break
            free,level=io.pressure();disk=psutil.disk_usage(str(io.HOME)).free
            if time.time()>=DEADLINE-20 or rss>2*1024**3 or free<25 or level not in (0,1,2) or disk<40*1024**3:
                stop=dict(rss=rss,free=free,level=level,disk=disk);child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            time.sleep(3)
        rc=child.wait()
    io.write(OUT/'owned_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,observed_peak_rss_bytes=peak,guard_stop=stop,finished_unix=time.time()))
    if rc:raise RuntimeError(f'Whole-source child exit {rc}; no partial source admission')
    print('OWNED_EXIT0',child.pid,peak,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','worker','run']);a=p.parse_args()
    try:globals()[a.action]()
    except BaseException as error:
        if OUT.exists():io.write(OUT/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
