"""Full local BODACC parent-name clause compiler; no CRM or labels.

Output is an unapproved evidence namespace pending blind precision audit.
Names in explicit ownership clauses are contextual facts, not own aliases.
Publication/project statements never imply confirmed current ownership.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as source
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v61/bodacc_parent_clauses_20261013_a'
PATTERNS=[
    ('NAMED_COMPANY_HOLDS_ALL_SHARES',re.compile(
        r'\b(?:la|une)\s+soci[ée]t[ée]\s+(?P<name>[^.;]{2,300}?)\s+d[ée]ten(?:ant|t)\s+la\s+totalit[ée]\s+des\s+(?:actions|parts)',re.I)),
    ('EXPLICIT_NAMED_COMMON_PARENT',re.compile(
        r'\bsoci[ée]t[ée]\s+m[èe]re\s*(?:,\s*(?:la\s+soci[ée]t[ée]\s+)?|:\s*)(?P<name>[^.;:]{2,250})',re.I))]
LEGAL_BOUNDARY=re.compile(r',|\b(?:SASU?|SARL|EURL|SA|soci[ée]t[ée]\s+(?:par|anonyme|civile|[àa])|au\s+capital|dont\s+le\s+si[èe]ge|immatricul[ée]e?)\b',re.I)
RECIPE=dict(scope='ALL_2022_2025_LOCAL_FULL_BODACC_FIELDS_NO_GT_SELECTION',
    candidate_patterns=[p for p,_ in PATTERNS], own_announced_company_public_required=True,
    annotation='PUBLISHED_PARENT_CLAUSE_CONTEXT_NOT_CONFIRMED_CURRENT_OWNERSHIP',
    relations_inferred_between_all_numbers=False, parent_id_resolution=False,
    new_own_alias=False, no_physical_people_or_private_addresses_extracted=True,
    blind_sample='32_LOWEST_SHA256_ANNOUNCEMENT_ID_RULE_NAME',
    admission='REQUIRES_BLIND_PRECISION_REVIEW_BEFORE_RETRIEVAL',
    model_frozen=True,deadline_unix=mandate.DEADLINE)


def initialize(out):
    mandate.guard(mandate.DEADLINE)
    if out.exists():raise ValueError('Fresh independent source namespace')
    out.mkdir(parents=True);io.write(out/'recipe.json',RECIPE)
    manifests=[source.SOURCES/str(y)/'manifest.json' for y in range(2022,2026)]
    sources=[io.read(p)['sources'][0] for p in manifests]
    io.write(out/'launch.json',dict(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        sources=sources,manifests=[record(p) for p in manifests],public_legal_units=record(source.STOCK),
        helpers=[record(Path(m.__file__)) for m in (mandate,source,engine,io)],
        deadline_unix=mandate.DEADLINE,labels_read=False,crm_read=False))


def compile(out):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    launch=io.read(out/'launch.json')
    for ref in (launch['implementation'],launch['recipe'],*launch['helpers'],*launch['manifests']):verify(ref)
    c=duckdb.connect();c.execute('SET threads=1');c.execute("SET memory_limit='512MB'")
    started=time.monotonic();facts=[];coverage=[];seen=set();rejections=Counter()
    try:
        for ref in launch['sources']:
            mandate.guard(launch['deadline_unix']);path=verify(ref)
            rows=c.execute("""SELECT id,dateparution,listepersonnes,
                json_extract_string(acte,'$.descriptif') text
                FROM read_parquet(?) WHERE dateparution <= DATE '2026-08-01'
                AND regexp_matches(lower(coalesce(acte,'')),
                    'société mère|societe mere|détenant la totalité|detenant la totalite')""",[str(path)]).fetch_arrow_table().to_pylist()
            coverage.append(dict(source=ref,source_rows=pq.ParquetFile(path).metadata.num_rows,examined=len(rows)))
            for row in rows:
                text=row['text'] or ''
                parties=[]
                for item in source.decoded(row['listepersonnes']):
                    person=source.unwrap(item)
                    if str(person.get('typePersonne') or '').lower()=='pp':continue
                    if not person.get('denomination'):continue
                    key=source.siren(person)
                    if key:parties.append(key)
                if not parties:rejections['NO_STRUCTURED_LEGAL_PARTY']+=1;continue
                for rule,pattern in PATTERNS:
                    for match in pattern.finditer(text):
                        name=LEGAL_BOUNDARY.split(match.group('name'),maxsplit=1)[0].strip(' \"«»:-')
                        normalized=source.norm(name)
                        if not source.tokens(name) or len(normalized)>120 or normalized.startswith(('ABSORBANTE','ABSORBEE','MEME SOCIETE','C EST A DIRE')):
                            rejections['UNRESOLVED_OR_GENERIC_CAPTURE']+=1;continue
                        # Parent identity is NOT deduced from nearby numbers.
                        clause=text[match.start():match.end()]
                        identity=(str(row['id']),rule,normalized)
                        if identity in seen:continue
                        seen.add(identity)
                        for child in sorted(set(parties)):
                            facts.append(dict(announcement_id=str(row['id']),publication_date=str(row['dateparution'])[:10],
                                subject_siren=child,context_parent_name=name,normalized_parent_name=normalized,
                                rule=rule,exact_clause=clause,clause_start=match.start(),clause_end=match.end(),
                                source_field='acte.descriptif',source_file=str(path),parent_siren=None,
                                effective_date=None,statement_status='PUBLISHED_CLAUSE_PROJECT_NOT_REALIZATION',
                                url='https://www.bodacc.fr/pages/annonces-commerciales-detail/?q.id='+str(row['id'])))
            io.write(out/'state.json',dict(status='COMPILING_SOURCE_ONLY_PARENT_CLAUSES',years=len(coverage),
                candidate_facts=len(facts),elapsed_s=time.monotonic()-started,pid=os.getpid(),updated_unix=time.time()))
        if not facts:raise ValueError('No source candidate facts')
        c.register('facts',pa.Table.from_pylist(facts))
        # Public subject guard. No protected identities reconstructed from text.
        public=c.execute("""SELECT f.* FROM facts f JOIN read_parquet(?) u
            ON u.siren=f.subject_siren WHERE u.statutDiffusionUniteLegale='O'
            ORDER BY f.announcement_id,f.subject_siren,f.rule,f.normalized_parent_name""",[str(verify(launch['public_legal_units']))]).fetch_arrow_table()
        pq.write_table(public,out/'candidate_facts.parquet',compression='zstd');engine.seal(out/'candidate_facts.parquet')
        unique={}
        for r in public.to_pylist():unique[(r['announcement_id'],r['rule'],r['normalized_parent_name'])]=r
        sample=sorted(unique.values(),key=lambda r:hashlib.sha256((r['announcement_id']+'|'+r['rule']+'|'+r['normalized_parent_name']).encode()).digest())[:32]
        io.write(out/'blind_precision_sample.json',sample);engine.seal(out/'blind_precision_sample.json')
        io.write(out/'generation.json',dict(status='COMPLETE_SOURCE_CANDIDATES_PENDING_BLIND_PRECISION_REVIEW',
            candidate_facts=record(out/'candidate_facts.parquet'),blind_sample=record(out/'blind_precision_sample.json'),
            facts=len(public),before_public_guard=len(facts),distinct_statements=len(unique),
            coverage=coverage,rejections=dict(rejections),elapsed_s=time.monotonic()-started,
            labels_read=False,crm_read=False,admitted_to_retrieval=False,parent_ids_not_inferred=True))
    finally:c.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','compile']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:globals()[a.action](a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise
