"""Whole July RNA public-owner names and uniform query-only normalisations.

No human labels, target lists, private persons/contact fields or predictions.
An initialism is a derived normalisation, NOT an asserted published alias.
Degree attenuation and administrative corroboration are fixed before outcomes.
"""
import argparse
from collections import Counter, defaultdict
import csv
import io as streamio
from pathlib import Path
import re
import shutil
import time
import zipfile

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as naming
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v130/structural_20261013_night'
ARCHIVE_ROOT = Path('/Volumes/CATNAT_DATA/oubusiness/raw/rna_association_stock/2026-07-01_rna-v1')
ARCHIVE = ARCHIVE_ROOT / 'rna_waldec_20260701.zip'
BASE = io.HOME / 'v115/streamed_activity_ablation_20261013_a'
STOCK = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockUniteLegale.parquet')
RECIPE = dict(hypothesis='EXPLICIT_PUBLIC_OWNER_RNA_TITLE_AND_GENERIC_INITIALISM_NORMALISATION',
    scope='WHOLE_JULY_NATIONAL_WALDEC_NOT_GT_SEEDED', source_snapshot='2026-07-01',
    join='EXACT_PUBLIC_SIRENE_identifiantAssociationUniteLegale_TO_RNA_ID',
    names='COMPLETE_TITLE_AND_UNTRUNCATED_SHORT_TITLE_ONLY_NO_CONTACT_OR_PERSON_FIELDS',
    initialism='INITIALS_OF_3_TO_8_MEANINGFUL_ALPHABETIC_TOKENS_AFTER_EXISTING_STOP',
    initialism_query='EXACT_SINGLE_INFORMATIVE_TOKEN_3_TO_8_ASCII_LETTERS',
    initialism_not_asserted_published_sigle=True,
    full_title_query='ALL_QUERY_INFORMATIVE_TOKENS_PRESENT_MIN_TWO_OR_EQUAL_SINGLE',
    confidence='0.5/SQRT_DISTINCT_PUBLIC_OWNER_DEGREE_OF_MATCHING_SOURCE_KEY',
    no_learned_aliases=True, no_query_specific_rules=True,
    no_address_or_city_copied_to_other_establishments=True,
    retrieval_requirement='QUERY_ADMIN_CORROBORATION_BEFORE_ONE_NATIVE_COLLECTOR100',
    sources_only_no_native_retrieval_or_labels=True,
    source_eligibility='BEFORE_CUTOFF_SOURCE_NOT_IMPLICIT_SERVING_OR_BENCHMARK_ADMISSION')


def words(value):
    return tuple(w for w in re.findall('[A-Z0-9]+', naming.norm(value or '')) if w not in naming.STOP)


def name_keys(title, short):
    result = []
    for field, text in [('title', title), ('short_title', short)]:
        if not text or '...' in text or '…' in text:
            continue
        tokens = words(text)
        if not tokens:
            continue
        result.append(('FULL_TITLE', frozenset(tokens), text, field))
        if 3 <= len(tokens) <= 8 and all(re.fullmatch('[A-Z]+', t) for t in tokens):
            result.append(('DERIVED_INITIALISM', ''.join(t[0] for t in tokens), text, field))
    return result


def prepare(out):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    started = time.monotonic()
    if out.exists():
        raise ValueError('Fresh immutable whole-source namespace')
    if shutil.disk_usage(io.HOME).free < 80 * 1024**3:
        raise ValueError('CATNAT reserve 80 GiB')
    metadata = io.read(ARCHIVE_ROOT / 'dataset_metadata_sanitized.json')
    resource = next(r for r in metadata['selected_resources'] if r['title'] == ARCHIVE.name)
    if resource['last_modified'][:10] > '2026-08-01' or ARCHIVE.stat().st_size != resource['filesize']:
        raise ValueError('Complete archived official pre-cutoff RNA source required')
    launch = io.read(BASE / 'launch.json')
    panel = io.read(verify(launch['panel']))
    if len(panel) != 526 or any(q['fold'] not in (2, 3, 4) for q in panel):
        raise ValueError('Same complete development query-only panel')
    out.mkdir(parents=True)
    (out / 'spill').mkdir()
    io.write(out / 'recipe.json', RECIPE)
    io.write(out / 'launch.json', dict(implementation=record(Path(__file__)), recipe=record(out/'recipe.json'),
        archive=record(ARCHIVE), metadata=record(ARCHIVE_ROOT/'dataset_metadata_sanitized.json'),
        resource=resource, owners=record(STOCK), query_panel=launch['panel'], labels_read=False))
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET memory_limit='768MB'")
    con.execute('SET temp_directory=?', [str(out/'spill')])
    records = con.execute("""SELECT siren,identifiantAssociationUniteLegale FROM read_parquet(?)
        WHERE statutDiffusionUniteLegale='O' AND categorieJuridiqueUniteLegale<>'1000'
        AND regexp_matches(identifiantAssociationUniteLegale,'^W[A-Z0-9]{9}$')""", [str(STOCK)]).fetchall()
    con.close()
    owners = defaultdict(set)
    for siren, rna in records:
        owners[rna].add(siren)
    qwords = {q['query_id']: frozenset(words(q['crm_name'])) for q in panel}
    single_acronyms = defaultdict(set)
    inverse = defaultdict(set)
    for qid, tokens in qwords.items():
        for token in tokens:
            inverse[token].add(qid)
        if len(tokens) == 1:
            token = next(iter(tokens))
            if re.fullmatch('[A-Z]{3,8}', token):
                single_acronyms[token].add(qid)
    schema = pa.schema([('siren', pa.string()), ('rna', pa.string()), ('title', pa.string()),
        ('short_title', pa.string()), ('source_member', pa.string()), ('source_line', pa.int64()),
        ('snapshot', pa.string())])
    path = out/'public_owner_names.parquet'
    writer = pq.ParquetWriter(path, schema, compression='zstd')
    total = joined = 0
    key_degree = defaultdict(set)
    matching = defaultdict(list)
    headers = {}
    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            members = sorted(n for n in archive.namelist() if n.lower().endswith('.csv'))
            for member in members:
                if shutil.disk_usage(io.HOME).free < 80 * 1024**3:
                    raise ValueError('CATNAT reserve while streaming')
                with archive.open(member) as raw:
                    first = raw.readline()
                    try:
                        first.decode('utf-8-sig'); encoding = 'utf-8-sig'
                    except UnicodeDecodeError:
                        encoding = 'iso-8859-15'
                with archive.open(member) as raw, streamio.TextIOWrapper(raw, encoding=encoding, newline='') as text:
                    reader = csv.DictReader(text, delimiter=';')
                    if not {'id', 'titre', 'titre_court'} <= set(reader.fieldnames):
                        raise ValueError('Actual official RNA title schema required')
                    headers[member] = dict(encoding=encoding, selected_columns=['id', 'titre', 'titre_court'])
                    batch = []
                    for line, row in enumerate(reader, 2):
                        total += 1
                        if None in row or any(v is None for v in row.values()):
                            raise ValueError(('Malformed complete source row', member, line))
                        rna = row['id'].strip()
                        if rna not in owners:
                            continue
                        title, short = row['titre'].strip(), row['titre_court'].strip()
                        if not title and not short:
                            continue
                        for siren in sorted(owners[rna]):
                            batch.append(dict(siren=siren, rna=rna, title=title, short_title=short,
                                source_member=member, source_line=line, snapshot='2026-07-01'))
                            joined += 1
                        for kind, key, name, field in name_keys(title, short):
                            possible = single_acronyms.get(key, set()) if kind == 'DERIVED_INITIALISM' else set().union(*(inverse.get(t,set()) for t in key))
                            qualified = possible if kind == 'DERIVED_INITIALISM' else {qid for qid in possible
                                if qwords[qid] and qwords[qid] <= key and (len(qwords[qid]) >= 2 or qwords[qid] == key)}
                            if not qualified:
                                continue
                            source_key = kind + ':' + (key if isinstance(key,str) else ' '.join(sorted(key)))
                            key_degree[source_key].update(owners[rna])
                            for qid in qualified:
                                for siren in sorted(owners[rna]):
                                    matching[qid].append(dict(siren=siren, rna_ref=rna, source_key=source_key,
                                        source_name=name, source_field=field, source_member=member, source_line=line,
                                        kind=kind, own_association_title=True, not_site_alias_or_address=True))
                        if len(batch) >= 8192:
                            writer.write_table(pa.Table.from_pylist(batch, schema=schema)); batch=[]
                    if batch:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                io.write(out/'state.json', dict(status='WHOLE_PUBLIC_JULY_RNA_TITLE_JOIN', member=member,
                    scanned_rows=total, public_joined_rows=joined, queries_with_context=len(matching), updated_unix=time.time()))
    finally:
        writer.close()
    context = {}
    for q in panel:
        by_owner = {}
        for fact in matching[q['query_id']]:
            degree = len(key_degree[fact['source_key']])
            confidence = .5 / degree**.5
            item = by_owner.setdefault(fact['siren'], dict(siren_ref=fact['siren'], confidence=0.,
                evidence=[], administrative_corroboration_required=True, no_source_address_inheritance=True))
            item['confidence'] = max(item['confidence'], confidence)
            item['evidence'].append(dict(fact, source_key_public_owner_degree=degree, confidence=confidence))
        context[q['query_id']] = sorted(by_owner.values(), key=lambda r:r['siren_ref'])
    io.write(out/'context.json', context)
    engine.seal(path); engine.seal(out/'context.json')
    io.write(out/'generation.json', dict(status='COMPLETE_SEALED_WHOLE_JULY_RNA_QUERY_ONLY_CONTEXT_NOT_RECALL',
        implementation=record(Path(__file__)), launch=record(out/'launch.json'), recipe=record(out/'recipe.json'),
        names=record(path), context=record(out/'context.json'), headers=headers,
        national_rows_scanned=total, public_joined_rows=joined, all_queries=526,
        queries_with_context=sum(bool(v) for v in context.values()), owner_query_references=sum(map(len,context.values())),
        largest_query_owner_degree=max(map(len,context.values())), labels_read=False,
        no_predictions_or_recall_or_source_promotion=True, no_person_names_contacts_or_addresses_exported=True,
        elapsed_seconds=time.monotonic()-started, observed_unix=time.time()))
    print({k:v for k,v in io.read(out/'generation.json').items() if not isinstance(v,dict)}, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=OUTPUT)
    prepare(parser.parse_args().output)
