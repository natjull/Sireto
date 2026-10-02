"""Bounded, label-free audit and adapter validation of official relation sources."""
import argparse
from collections import Counter
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from scripts import retrieval_v39_relational_proofs as proof
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v39_relational_proofs.json'
SOURCE_CONFIG = ROOT/'config/retrieval_all_human_v18_parent_discovery_generation.json'
SOURCE_SHA = 'e270a60d520d7c7be7928988382ce91fc8f326c7bc1541a7c53eff0ce1aa80e6'


def records(connection, sql, params):
    return connection.execute(sql, params).fetch_arrow_table().to_pylist()


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(CONFIG.read_text()); limits = cfg['source_audit']
    source = json.loads(verify(dict(path=SOURCE_CONFIG, sha256=SOURCE_SHA)).read_text())['generation_authorities']
    resources = {k: limits[k] for k in ('max_seconds', 'max_rss_bytes', 'max_swapout_pages_delta')}
    budget = Budget(resources)
    try:
        with budget:
            paths = {k: verify(source[k]) for k in ('bodacc_relations', 'rne_roles')}
            temporal = source['temporal_sources']; tp = Path(temporal['index_path'])/'manifest.json'
            tm = json.loads(verify(dict(path=tp, sha256=temporal['manifest_sha256'])).read_text())
            if tm['contains_crm_labels'] or tm['contains_synthetic_rows']:
                raise ValueError('Forbidden temporal inputs')
            temporal_refs = []
            for s in tm['sources']:
                p = Path(temporal['source_root'])/s['projection_build_id']/'succession_edges.parquet'
                verify(dict(path=p, sha256=s['successions_sha256'])); temporal_refs.append(record(p))
            if len(temporal_refs) != temporal['expected_file_count']:
                raise ValueError('Incomplete succession authority')
            conn = duckdb.connect(); conn.execute("SET memory_limit='512MB'"); conn.execute('SET threads=2')
            conn.execute("SET preserve_insertion_order=false")
            b, r = str(paths['bodacc_relations']), str(paths['rne_roles']); ts = [p['path'] for p in temporal_refs]
            stats = {
                'bodacc': records(conn, '''SELECT source,relation_type,from_kind,to_kind,count(*) rows,
                    count(effective_date) reported_dates,count(observed_at) observation_dates
                    FROM read_parquet(?) GROUP BY ALL''', [b]),
                'rne': records(conn, '''SELECT count(*) rows,
                    count(*) FILTER(WHERE coalesce(role_code,'')='') missing_role_code,
                    count(*) FILTER(WHERE coalesce(role_label,'')='') empty_role_label,
                    count(*) FILTER(WHERE coalesce(array_length(establishment_sirets),0)>0) site_anchored_rows,
                    count(*) FILTER(WHERE subject_siren=related_siren) self_relations
                    FROM read_parquet(?)''', [r]),
                'rne_codes': records(conn, 'SELECT role_code,count(*) rows FROM read_parquet(?) GROUP BY role_code ORDER BY rows DESC', [r]),
                'succession': records(conn, '''SELECT count(*) raw_rows,
                    count(*) FILTER(WHERE economic_continuity) continuity_rows,
                    count(*) FILTER(WHERE economic_continuity IS NULL) unknown_continuity_rows,
                    count(*) FILTER(WHERE predecessor_siren<>successor_siren) cross_siren_rows,
                    count(effective_date) effective_dates FROM read_parquet(?)''', [ts])}
            n = limits['sample_per_source']
            samples = {
                'bodacc': records(conn, 'SELECT * FROM read_parquet(?) ORDER BY sha256(relation_id),relation_id LIMIT ?', [b, n]),
                'rne': records(conn, '''SELECT role_id,source_record_id,subject_siren,related_siren,role_code,role_label,
                    effective_date,observed_at,source FROM read_parquet(?) ORDER BY sha256(role_id),role_id LIMIT ?''', [r, n]),
                'succession': records(conn, '''SELECT * FROM read_parquet(?) WHERE economic_continuity
                    ORDER BY sha256(relation_id),relation_id LIMIT ?''', [ts, n])}
            adapters = dict(bodacc=proof.bodacc_transfer, rne=proof.rne_role, succession=proof.succession)
            adapted = {}; failures = []
            for key, rows in samples.items():
                edges = []
                for row in rows:
                    try:
                        edge = adapters[key](row)
                        if edge is not None: edges.append(edge)
                    except ValueError as exc:
                        failures.append(dict(source=key, error=str(exc)))
                adapted[key] = proof.deduplicate(edges)
            conn.close()
            # This diagnostic never becomes a retrieval corpus or a label-derived lookup.
            write_json(output/'sample_proofs.json', {key: [e.to_dict() for e in edges] for key, edges in adapted.items()})
            write_json(output/'sample_logical_bindings.json', {key: [p for e in edges for p in e.bindings()] for key, edges in adapted.items()})
        manifest = dict(status='AUDITED_RELATIONAL_SOURCES_AND_TYPED_ADAPTERS', decision='STOP',
            decision_reason='RELATIONAL_INITIAL_SEARCH_INDEX_NOT_BUILT_NO_RECALL_GATE_MEASURED',
            stats=stats, sample_input_rows={k:len(v) for k,v in samples.items()},
            sample_unique_proofs={k:len(v) for k,v in adapted.items()}, sample_adapter_failures=failures,
            resources=budget.report(), config=record(CONFIG), source_config=record(SOURCE_CONFIG),
            sources={k:record(p) for k,p in paths.items()}, temporal_manifest=record(tp), temporal_files=temporal_refs,
            implementation={str(p):record(p) for p in [Path(__file__),Path(proof.__file__)]},
            artifacts={p.name:record(p) for p in output.glob('sample_*.json')},
            missing_links=['BODACC_EVENT_ADDRESS_BINDING', 'VERIFIED_RNE_ROLE_DICTIONARY_AND_SITE_BINDING',
                           'LEXICAL_SPATIAL_RELATIONAL_INDEX_AND_UNIQUE_TOP100_COLLECTOR'],
            truth_opened=False, crm_queries_opened=False, model_fit=False, retrieval_performed=False,
            full_relation_deduplication_performed=False, sample_is_not_a_search_corpus=True, ready_for_fold1=False)
        write_json(output/'manifest.json', manifest)
        print(json.dumps({k:manifest[k] for k in ('stats','sample_input_rows','sample_unique_proofs','sample_adapter_failures','resources','decision')}))
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), ready_for_fold1=False))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
