"""Compile official BODACC name -> announcement -> current-site proof paths.

This source-only compilation never reads CRM queries or human labels. It
does not yet build a national search index or measure Recall@100.
"""
import argparse
import json
from pathlib import Path
import time

import duckdb
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v39_bodacc_compilation.json'


def rows(conn, sql, params=()):
    return conn.execute(sql, params).fetch_arrow_table().to_pylist()


def compile_paths(conn, sources, output, progress=lambda _: None):
    """Join exact announcement keys, official memberships and scoped names.

    A person/siege address cannot bind a sold business. Announcements with
    multiple distinct establishment addresses are kept in the diagnostics,
    awaiting a reliable predecessor-to-establishment mapping. All matching
    active AND closed official sites are retained at a grounded address.
    """
    phases = {}

    def execute(name, sql, params):
        start = time.monotonic()
        conn.execute(sql, params)
        phases[name] = time.monotonic()-start
        progress(dict(phase=name, seconds=phases[name]))

    execute('relations', '''CREATE TEMP TABLE relations AS
        SELECT DISTINCT *,regexp_replace(source_record_id, ':predecessor:[0-9]+$', '') event_id
        FROM read_parquet(?) WHERE source='BODACC' AND relation_type='ASSET_TRANSFER'
        AND from_kind='SIREN' AND to_kind='SIREN'
        AND regexp_full_match(from_identifier,'[0-9]{9}')
        AND regexp_full_match(to_identifier,'[0-9]{9}') AND from_identifier<>to_identifier
        AND regexp_matches(source_record_id, ':predecessor:[0-9]+$')''', [str(sources['relations'])])
    conflict = conn.execute('''SELECT count(*) FROM (
        SELECT relation_id FROM relations GROUP BY relation_id HAVING count(*)>1)''').fetchone()[0]
    if conflict:
        raise ValueError('Conflicting BODACC relation identifiers')
    # Retain event evidence provenance. No join by SIREN alone or prefix LIKE.
    execute('event_addresses', '''CREATE TEMP TABLE event_addresses AS
        SELECT DISTINCT a.siren,a.siret,a.subject_kind,a.normalized_value,a.postcode,a.insee,
            a.evidence_id,a.source_record_id,
            regexp_replace(a.source_record_id, ':establishment:[0-9]+$', '') event_id
        FROM read_parquet(?) a
        SEMI JOIN (SELECT DISTINCT to_identifier,event_id FROM relations) r
            ON r.to_identifier=a.siren
            AND r.event_id=regexp_replace(a.source_record_id, ':establishment:[0-9]+$', '')
        WHERE a.source='BODACC' AND a.normalized_value<>''
        AND regexp_matches(a.source_record_id, ':establishment:[0-9]+$')''',
        [str(sources['addresses'])])
    execute('event_scope', '''CREATE TEMP TABLE event_scope AS
        SELECT siren,event_id,count(DISTINCT struct_pack(address := normalized_value,
            postcode := coalesce(postcode,''),insee := coalesce(insee,''))) address_count
        FROM event_addresses GROUP BY siren,event_id''', [])
    # Materialize narrow address keys before touching the national site file.
    # Otherwise a multiway plan can expand all siblings before checking place.
    execute('grounded_sites', '''CREATE TEMP TABLE grounded_sites AS
        SELECT DISTINCT s.siren,s.siret,s.administrative_state,s.current_address_normalized,s.postcode,s.insee
        FROM read_parquet(?) s JOIN (
            SELECT DISTINCT a.siren,a.normalized_value,a.postcode,a.insee
            FROM event_addresses a JOIN event_scope e ON e.siren=a.siren AND e.event_id=a.event_id
            WHERE e.address_count=1 AND (coalesce(a.postcode,'')<>'' OR coalesce(a.insee,'')<>'')
        ) a ON s.siren=a.siren AND s.current_address_normalized=a.normalized_value
            AND (coalesce(a.postcode,'')='' OR a.postcode=s.postcode)
            AND (coalesce(a.insee,'')='' OR a.insee=s.insee)
        WHERE regexp_full_match(s.siret,'[0-9]{14}') AND regexp_full_match(s.siren,'[0-9]{9}')''',
        [str(sources['sites'])])
    # Sites are selected by official SIREN field, never inferred from a SIRET.
    execute('site_bindings', '''CREATE TEMP TABLE bindings AS
        SELECT DISTINCT r.relation_id,r.source_record_id relation_record_id,r.event_id,
            r.from_identifier anchor_siren,r.to_identifier target_siren,
            r.effective_date reported_date,r.observed_at relation_observed_at,
            a.evidence_id address_evidence_id,a.source_record_id address_record_id,
            a.normalized_value event_address,a.postcode event_postcode,a.insee event_insee,
            s.siret,s.administrative_state,s.current_address_normalized target_address,
            s.postcode target_postcode,s.insee target_insee
        FROM grounded_sites s
        JOIN event_addresses a ON s.siren=a.siren AND s.current_address_normalized=a.normalized_value
            AND (coalesce(a.postcode,'')='' OR a.postcode=s.postcode)
            AND (coalesce(a.insee,'')='' OR a.insee=s.insee)
            AND (coalesce(a.postcode,'')<>'' OR coalesce(a.insee,'')<>'')
            AND (a.subject_kind='SIREN' OR (a.subject_kind='SIRET' AND a.siret=s.siret))
        JOIN event_scope e ON e.siren=a.siren AND e.event_id=a.event_id AND e.address_count=1
        JOIN relations r ON r.to_identifier=a.siren AND r.event_id=a.event_id
        WHERE regexp_full_match(s.siret,'[0-9]{14}') AND regexp_full_match(s.siren,'[0-9]{9}')''', [])
    # Publications repeat identical lexical facts. Preserve one coherent
    # provenance row per (parent,text,source,kind), not millions of occurrences.
    execute('name_facts', '''CREATE TEMP TABLE name_facts AS
        SELECT n.siren,min(struct_pack(text := n.normalized_value,
            source := n.source,name_kind := n.name_kind,evidence_id := n.evidence_id,
            source_record_id := n.source_record_id,valid_from := n.valid_from,
            valid_to := n.valid_to)) fact
        FROM read_parquet(?) n
        SEMI JOIN (SELECT DISTINCT anchor_siren FROM bindings) b ON b.anchor_siren=n.siren
        WHERE n.subject_kind='SIREN' AND n.normalized_value<>''
        AND n.source IN ('SIRENE_CURRENT','SIRENE_HISTORY','SIRENE','RNE','BODACC')
        GROUP BY n.siren,n.normalized_value,n.source,n.name_kind''', [str(sources['names'])])
    execute('anchor_names', '''CREATE TEMP TABLE anchor_names AS
        SELECT siren,list_sort(list(fact)) anchor_values FROM name_facts GROUP BY siren''', [])
    execute('proof_paths', '''COPY (
        SELECT b.*,n.anchor_values anchor_names,'SIREN' anchor_name_scope,'BODACC' relation_source,
            'ASSET_TRANSFER' relation_type,'FORWARD' direction,
            'REPORTED_DATE_EFFECTIVITY_UNVERIFIED' date_semantics,
            'SINGLE_ANNOUNCEMENT_ESTABLISHMENT_ADDRESS_AND_CURRENT_OFFICIAL_SITE' site_binding,
            'PARENT_NAME_VIA_TRANSFER_NOT_TARGET_ALIAS' lexical_semantics
        FROM bindings b JOIN anchor_names n ON n.siren=b.anchor_siren
        ORDER BY b.siret,b.relation_id,b.address_evidence_id
        ) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''', [str(output/'proof_paths.parquet')])
    # Keep every valid relation in a diagnostic denominator, including failures.
    execute('coverage', '''COPY (
        SELECT r.relation_id,r.event_id,r.from_identifier,r.to_identifier,
            coalesce(e.address_count,0) event_site_address_count,
            coalesce(b.site_count,0) compatible_site_count,
            coalesce(b.proof_count,0) searchable_proof_count,
            CASE WHEN coalesce(e.address_count,0)=0 THEN 'NO_EVENT_ESTABLISHMENT_ADDRESS'
                 WHEN e.address_count>1 THEN 'MULTIPLE_EVENT_ADDRESSES_MAPPING_UNVERIFIED'
                 WHEN coalesce(b.site_count,0)=0 THEN 'NO_COMPATIBLE_CURRENT_OFFICIAL_SITE_OR_GEO'
                 WHEN coalesce(b.proof_count,0)=0 THEN 'NO_OFFICIAL_PARENT_ANCHOR_NAME'
                 ELSE 'COMPILED' END status
        FROM relations r LEFT JOIN event_scope e ON e.siren=r.to_identifier AND e.event_id=r.event_id
        LEFT JOIN (SELECT b.relation_id,count(DISTINCT b.siret) site_count,
            count(*) FILTER(WHERE n.siren IS NOT NULL) proof_count FROM bindings b
            LEFT JOIN anchor_names n ON n.siren=b.anchor_siren GROUP BY b.relation_id) b
            ON b.relation_id=r.relation_id ORDER BY r.relation_id
        ) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''', [str(output/'coverage.parquet')])
    return dict(phases=phases,
        source_rows=conn.execute('SELECT count(*) FROM read_parquet(?)', [str(sources['relations'])]).fetchone()[0],
        valid_unique_relations=conn.execute('SELECT count(*) FROM relations').fetchone()[0],
        coverage=rows(conn,'SELECT status,count(*) row_count FROM read_parquet(?) GROUP BY status ORDER BY status',
                      [str(output/'coverage.parquet')]),
        proofs=rows(conn,'''SELECT count(*) proof_paths,count(DISTINCT relation_id) relations,
            count(DISTINCT siret) target_sites,count(DISTINCT anchor_siren) anchor_parents
            FROM read_parquet(?)''', [str(output/'proof_paths.parquet')]))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(CONFIG.read_text()); budget = Budget(cfg['resources']); conn = None
    try:
        with budget:
            source_manifest = verify(cfg['source_manifest'])
            sm = json.loads(source_manifest.read_text())
            if sm['policy']['crm_labels_present'] or sm['policy']['model_scores_present']:
                raise ValueError('Forbidden dossier source')
            sources = {}; source_refs={}; fingerprints={}
            for key, name in cfg['files'].items():
                path = source_manifest.parent/name
                sources[key] = verify(dict(path=path,sha256=sm['files'][name]['sha256']))
                source_refs[key]=dict(path=str(path),sha256=sm['files'][name]['sha256'],size_bytes=path.stat().st_size)
                fingerprints[key]=(path.stat().st_size,path.stat().st_mtime_ns)
            # Logical projected cost, independent of macOS disk caches.
            cost = {key:dict(rows=pq.ParquetFile(p).metadata.num_rows,
                             row_groups=pq.ParquetFile(p).metadata.num_row_groups,
                             bytes=p.stat().st_size) for key,p in sources.items()}
            write_json(output/'launch.json',dict(config=record(CONFIG), source_manifest=record(source_manifest),
                sources=source_refs, cost=cost,
                implementation=record(Path(__file__)), crm_opened=False, truth_opened=False))
            conn=duckdb.connect()
            conn.execute("SET memory_limit='512MB'"); conn.execute('SET threads=2')
            conn.execute('SET preserve_insertion_order=false')
            # Spill only inside the task output, with an explicit disk ceiling.
            conn.execute('SET temp_directory=?',[str(output/'spill')])
            conn.execute("SET max_temp_directory_size='1GB'")
            stats=compile_paths(conn,sources,output,lambda x:print(json.dumps(x),flush=True))
            conn.close(); conn=None
            if any((p.stat().st_size,p.stat().st_mtime_ns)!=fingerprints[k] for k,p in sources.items()):
                raise ValueError('Source file changed during compilation')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>cfg['max_output_bytes']:
                raise ValueError('STOP_OUTPUT_SIZE')
        manifest=dict(status='SEALED_BODACC_SOURCE_PROOF_COMPILATION', stats=stats,
            resources=budget.report(),launch=record(output/'launch.json'),
            artifacts={p.name:record(p) for p in output.glob('*.parquet')},
            truth_opened=False,crm_opened=False,retrieval_performed=False,model_fit=False,
            initial_recall_measured=False,ready_for_fold1=False,
            limitations=cfg['limitations'],decision='STOP_INITIAL_SEARCH_INDEX_AND_RECALL_NOT_MEASURED')
        write_json(output/'manifest.json',manifest)
        print(json.dumps(dict(stats=stats,resources=manifest['resources'])),flush=True)
    except BaseException as exc:
        if conn is not None: conn.close()
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,
            resources=budget.report() if hasattr(budget,'elapsed') else None,ready_for_fold1=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
