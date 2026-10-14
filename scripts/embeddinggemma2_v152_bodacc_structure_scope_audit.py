"""Bounded source-only BODACC structure audit, NOT retrieval or extraction.

Three fixed rowgroups/year, 512 evenly spaced rows/group. Counts describe
this deterministic sample ONLY, not a probability sample or national estimate.
Never exports names, people, activity text, addresses or subject identifiers.
"""
from collections import Counter
from pathlib import Path
import gc
import hashlib
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = io.HOME/'v112/bodacc_activity_context_20261013_a'
OUT = io.HOME/'v152/bodacc_structure_scope_20261014_night_b'
XSD = 'https://echanges.dila.gouv.fr/schemabook/bodacc/sor/Bodacc_rediff_all.xsd'
XSD_SHA = 'bb29490b5096908d563c772623bfee6b6cf51d26aeb510310d0b08c8dd63b95b'
LIMIT = 512
DEADLINE = 1791954368


def guard():
    import psutil
    free, level = io.pressure()
    if time.time() >= DEADLINE or psutil.Process().memory_info().rss > 512*1024**2 or free < 25 or level not in (0,1,2):
        raise RuntimeError('Light source audit yields on real resources/deadline')


def decoded(value):
    if not value:
        return None
    try:
        return json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return None


def items(value, key):
    if isinstance(value, dict) and key in value:
        child = value[key]
        return [child] if isinstance(child, dict) else child if isinstance(child, list) else []
    return []  # Unknown wrapper is not silently reinterpreted as a person.


def subject(person):
    number = person.get('numeroImmatriculation', {})
    if not isinstance(number, dict):
        return ''
    value = re.sub('[^0-9]', '', str(number.get('numeroIdentification', '')))
    return value if len(value) == 9 else ''


def has_activity(item):
    return isinstance(item, dict) and isinstance(item.get('activite'), str) and bool(item['activite'].strip())


def classify(row):
    people_json, sites_json = decoded(row['listepersonnes']), decoded(row['listeetablissements'])
    people = [p for p in items(people_json, 'personne') if isinstance(p, dict)]
    sites = [s for s in items(sites_json, 'etablissement') if isinstance(s, dict)]
    pm = [p for p in people if p.get('typePersonne') == 'pm' and subject(p)]
    owners = {subject(p) for p in pm}
    sole_corporate_subject = bool(people and len(owners) == 1 and len(pm) == len(people))
    strict_person = (isinstance(people_json, dict) and isinstance(people_json.get('personne'), dict)
                     and people_json['personne'].get('typePersonne') == 'pm')
    strict_site = isinstance(sites_json, dict) and isinstance(sites_json.get('etablissement'), dict)
    strict_structure = strict_person and strict_site
    source_activity = bool(strict_structure and has_activity(sites_json['etablissement']))
    site_activity = sum(has_activity(s) for s in sites)
    pm_activity_owners = {subject(p) for p in pm if has_activity(p)}
    category = ('CURRENT_STRICT_PM_OBJECT_SITE_OBJECT_ACTIVITY' if source_activity else
                'SOLE_PM_SUBJECT_MULTISITE_ACTIVITY' if sole_corporate_subject and len(sites) > 1 and site_activity else
                'SOLE_PM_SUBJECT_SINGLETON_ARRAY_ACTIVITY' if sole_corporate_subject and len(sites) == 1 and site_activity else
                'NONCORPORATE_SITE_ACTIVITY' if site_activity and people and all(p.get('typePersonne') == 'pp' for p in people) else
                'MULTIPLE_OR_UNKNOWN_SUBJECTS_SITE_ACTIVITY_AMBIGUOUS' if site_activity and not sole_corporate_subject else 'NO_SITE_ACTIVITY')
    return dict(year='', before_cutoff=str(row['dateparution'])[:10] <= '2026-08-01', category=category,
        publication_family=row.get('familleavis','UNAVAILABLE_IN_PINNED_ALLOWLIST'), strict_structure=strict_structure,
        current_activity=source_activity, site_activity_nodes=site_activity,
        person_activity_nodes=sum(has_activity(p) for p in people),
        directly_bound_pm_person_activity_owners=pm_activity_owners,
        sole_subject_owner=next(iter(owners)) if sole_corporate_subject else None,
        corporate_owners=owners, people_count=len(people), sites_count=len(sites),
        person_wrapper_type=type(people_json.get('personne')).__name__ if isinstance(people_json, dict) else type(people_json).__name__,
        site_wrapper_type=type(sites_json.get('etablissement')).__name__ if isinstance(sites_json, dict) else type(sites_json).__name__,
        site_nodes_explicit_person_ref=sum(any(k in s for k in ('siren','siret','numeroImmatriculation','numeroIdentification','personne')) for s in sites),
        person_fields=tuple(sorted(set().union(*(p.keys() for p in people)))) if people else (),
        site_fields=tuple(sorted(set().union(*(s.keys() for s in sites)))) if sites else ())


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    import duckdb
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    started = time.monotonic(); guard()
    if OUT.exists():
        raise ValueError('Fresh immutable source audit namespace')
    gen, ex = io.read(SOURCE/'generation.json'), io.read(SOURCE/'owned_tool_exit.json')
    if gen['status'] != 'COMPLETE_WHOLE_SOURCE_QUERY_ONLY_BODACC_ACTIVITY_CONTEXT' or ex['exit_code'] or not ex['own_tool_exit_confirmed']:
        raise ValueError('Existing whole-source activity authority')
    launch = io.read(verify(gen['launch']))
    verify(gen['implementation']); verify(gen['recipe'])
    payload = urllib.request.urlopen(XSD, timeout=15).read()
    if hashlib.sha256(payload).hexdigest() != XSD_SHA:
        raise ValueError('Official schema drift')
    tree = ET.fromstring(payload)
    schema = dict(url=XSD, sha256=XSD_SHA, bytes=len(payload), declarations={})
    for node in tree:
        if node.attrib.get('name') in ('Personne_Type', 'Etablissement_Type'):
            schema['declarations'][node.attrib['name']] = [dict(e.attrib) for e in node.iter() if e.tag.endswith('element')]
    schema['person_and_establishment_lists_unbounded'] = [dict(n.attrib) for n in tree.iter()
        if n.tag.endswith('element') and n.attrib.get('name') in ('personne','etablissement') and n.attrib.get('maxOccurs') == 'unbounded']
    OUT.mkdir(parents=True); (OUT/'spill').mkdir()
    io.write(OUT/'recipe.json',dict(scope='SOURCE_ONLY_NO_CRM_OR_LABELS', sample='FIRST_MIDDLE_LAST_ROWGROUP_EVENLY_SPACED_512_ROWS_EACH_YEAR',
        nationally_exhaustive_counts_inherited_from_v112_only=True, sample_not_probability_or_national_estimate=True,
        no_actor_cross_product=True, no_text_or_identifier_export=True, maximum_rss_mib=512, threads=1,
        official_schema_url=XSD, official_schema_sha=XSD_SHA, deadline_unix=DEADLINE))
    observed = []; selections = []; field_counts = Counter(); rows_digest = hashlib.sha256()
    for ref in launch['sources']:
        guard(); path = Path(ref['source']['path'])
        if path.stat().st_size != ref['source']['size_bytes']:
            raise ValueError('Pinned existing source size drift')
        pf = pq.ParquetFile(path)
        groups = sorted({0, pf.num_row_groups//2, pf.num_row_groups-1})
        year = re.search(r'(20\d\d)', path.name).group(1)
        selections.append(dict(year=year, inherited_pinned_source=ref['source'], manifest=ref['manifest'],
            whole_source_hash_not_recomputed_to_avoid_heavy_IO=True, source_rows=pf.metadata.num_rows,
            selected_rowgroups=groups, selected_rows_per_group=LIMIT))
        for group in groups:
            n = pf.metadata.row_group(group).num_rows
            positions = {int(i*n/min(n,LIMIT)) for i in range(min(n,LIMIT))}
            offset = 0
            columns = [c for c in ['dateparution','familleavis','listepersonnes','listeetablissements'] if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=1024, row_groups=[group], columns=columns, use_threads=False):
                for j, row in enumerate(batch.to_pylist()):
                    if offset+j not in positions:
                        continue
                    # Digest observed input bytes without exporting source values.
                    rows_digest.update(json.dumps(row,sort_keys=True,default=str).encode())
                    value = classify(row); value['year'] = year; observed.append(value)
                    field_counts.update('person.'+k for k in value['person_fields'])
                    field_counts.update('establishment.'+k for k in value['site_fields'])
                offset += len(batch)
            guard()
        del pf; gc.collect()
    # Current public corporate eligibility: ONLY codes/IDs, never name/address.
    wanted = sorted(set().union(*(r['corporate_owners'] for r in observed)))
    con = duckdb.connect(); con.execute('SET threads=1'); con.execute("SET memory_limit='128MB'")
    con.execute('SET temp_directory=?',[str(OUT/'spill')])
    con.register('sample_subjects', pa.table({'siren':wanted}))
    stock = launch['public_units']; path = Path(stock['path'])
    if path.stat().st_size != stock['size_bytes']:
        raise ValueError('Existing public-unit source size drift')
    eligible = {s for s, in con.execute("""SELECT u.siren FROM read_parquet(?) u
        JOIN sample_subjects s USING(siren)
        WHERE u.statutDiffusionUniteLegale='O' AND u.categorieJuridiqueUniteLegale<>'1000'""",[str(path)]).fetchall()}
    con.close(); gc.collect(); guard()
    counts = Counter(); by_year = {}; by_family = {}; rejected = Counter()
    for r in observed:
        if not r['before_cutoff']:
            counts['after_cutoff'] += 1; continue
        counts['before_cutoff'] += 1; counts[r['category']] += 1
        counts['strict_structure'] += r['strict_structure']
        counts['site_activity_rows'] += r['site_activity_nodes'] > 0
        counts['person_activity_rows_all_types'] += r['person_activity_nodes'] > 0
        counts['person_activity_nodes_all_types'] += r['person_activity_nodes']
        counts['site_nodes_explicit_person_ref'] += r['site_nodes_explicit_person_ref']
        public_sole = r['sole_subject_owner'] in eligible
        current_public = bool(r['current_activity'] and public_sole)
        safe_site_extension = bool(public_sole and r['site_activity_nodes'] and not r['current_activity'])
        direct_public_owners = r['directly_bound_pm_person_activity_owners'] & eligible
        counts['current_strict_public_corporate_activity_rows'] += current_public
        counts['additional_single_owner_public_site_activity_rows'] += safe_site_extension
        counts['directly_bound_public_pm_person_activity_rows'] += bool(direct_public_owners)
        counts['additional_direct_public_pm_person_activity_rows_currently_rejected'] += bool(direct_public_owners and not current_public)
        counts['direct_person_activity_public_owner_refs'] += len(direct_public_owners)
        if not current_public and r['site_activity_nodes']:
            reason = ('SOLE_OWNER_ARRAY_SHAPE_ONLY_PUBLIC' if safe_site_extension else
                      'MULTI_OR_UNKNOWN_SUBJECT_AMBIGUOUS' if r['category'] == 'MULTIPLE_OR_UNKNOWN_SUBJECTS_SITE_ACTIVITY_AMBIGUOUS' else
                      'CURRENT_PUBLIC_CORPORATE_ELIGIBILITY_OR_NONCORPORATE')
            rejected[reason] += 1
        for mapping, key in ((by_year,r['year']), (by_family,r['publication_family'])):
            c = mapping.setdefault(str(key),Counter()); c['before_cutoff'] += 1
            c['current_strict_public_activity'] += current_public
            c['single_owner_public_array_extension'] += safe_site_extension
            c['direct_pm_person_activity_currently_rejected'] += bool(direct_public_owners and not current_public)
            c['ambiguous_multiple_subject_site_activity'] += r['category'] == 'MULTIPLE_OR_UNKNOWN_SUBJECTS_SITE_ACTIVITY_AMBIGUOUS'
    whole_counts = {k:sum(r[k] for r in gen['coverage']) for k in ('total_rows','before_cutoff','strict_single_objects','eligible_public_activity_rows')}
    io.write(OUT/'official_schema_declarations.json',schema)
    io.write(OUT/'source_sample_manifest.json',dict(selections=selections, observed_fields_sha256=rows_digest.hexdigest(),
        sampled_rows=len(observed), stock_inherited_pinned_reference=stock, whole_stock_hash_not_recomputed=True))
    io.write(OUT/'counts.json',dict(sample=dict(counts), rejected_site_activity=dict(rejected), by_year=by_year,
        by_family=by_family, field_presence=dict(field_counts), sampled_rows=len(observed),
        whole_source_counts_inherited_exact=whole_counts))
    io.write(OUT/'report.json',dict(status='COMPLETE_SOURCE_ONLY_BOUNDED_BODACC_STRUCTURE_AUDIT',
        implementation=record(Path(__file__)), recipe=record(OUT/'recipe.json'), source_generation=record(SOURCE/'generation.json'),
        source_actual_exit=record(SOURCE/'owned_tool_exit.json'), schema=record(OUT/'official_schema_declarations.json'),
        sample_manifest=record(OUT/'source_sample_manifest.json'), counts=record(OUT/'counts.json'),
        no_crm_query_or_label_read=True, no_retrieval_or_scoring_or_alias=True, no_names_people_addresses_text_identifiers_exported=True,
        sample_counts_not_extrapolated_to_national_population=True, current_public_corporate_gate_checked=True,
        hypothetical_binding_not_new_evidence_pipeline=True,
        observed='FIELD_PATHS_AND_SUBJECT_CARDINALITY_IN_FIXED_SYSTEMATIC_SOURCE_SAMPLE',
        inference='SINGLE_KNOWN_CORPORATE_SUBJECT_ALLOWS_PARENT_CONTEXT_NOT_SIRET_BINDING; MULTIPLE_SUBJECT_ESTABLISHMENTS_REQUIRE_EXPLICIT_LINK_NOT_CROSS_PRODUCT',
        no_subject_id_in_official_establishment_type=True, directly_nested_person_activity_role_safe_at_owner_level=True,
        absence_site_identifier_prevents_exact_siret_binding=True, closed_sets_opened=False,
        elapsed_seconds=time.monotonic()-started, observed_unix=time.time()))
    print(dict(sampled_rows=len(observed),counts=dict(counts),rejected=dict(rejected),elapsed_s=time.monotonic()-started),flush=True)


if __name__ == '__main__':
    main()
