#!/usr/bin/env python3
"""Label-side source/index reachability audit, never a candidate generator."""
import argparse
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files, open_workbench_db
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery, TantivyBackend
from scripts.run_retrieval_parent_discovery_v18 import _load_retrieval_queries


def classify(row):
    if row['official_sites'] == 0:
        return 'NO_OFFICIAL_PARENT_SITE'
    if row['indexed_parent_sites'] == 0:
        return 'NO_INDEXED_PARENT_SITE'
    if row['indexed_geo_sites'] == 0:
        return 'OUTSIDE_EFFECTIVE_INDEX_GEO'
    if not row['matching_channels']:
        return 'NO_V18_TEXT_MATCH_IN_GEO'
    if not row['parent_in_evidence']:
        return 'TEXT_REACHABLE_BUT_NOT_RETRIEVED'
    return 'ALREADY_RETRIEVED'


def index_inventory(root):
    return {str(p.relative_to(root)): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(root.rglob('*')) if p.is_file() and not p.name.endswith('.lock')}


def text_queries(backend, query):
    clauses = [('fielded_name_bm25', backend._fielded_name_query(query.name))]
    for channel, field, value in [
        ('site_name_word', 'site_names', query.name), ('trade_name_word', 'trade_names', query.name),
        ('historical_name_word', 'historical_names', query.name), ('supporting_name_word', 'supporting_names', query.name),
        ('address_word', 'addresses', query.address), ('historical_address_word', 'historical_addresses', query.address),
        ('supporting_address_word', 'supporting_addresses', query.address)]:
        parsed, _ = backend.index.parse_query_lenient(value, default_field_names=[field], conjunction_by_default=False)
        clauses.append((channel, parsed))
    clauses.extend([('address_char', backend._character_query('address_ngrams', query.address)),
                    ('current_name_char', backend._character_query('current_name_ngrams', query.name))])
    return clauses


def main(output):
    v18path = verify(dict(path='config/retrieval_all_human_v18_parent_discovery_generation.json',
                         sha256='e270a60d520d7c7be7928988382ce91fc8f326c7bc1541a7c53eff0ce1aa80e6'))
    v23path = verify(dict(path='config/retrieval_v23_joint_parent.json',
                         sha256='bb12058773f80d6f5d7ff3ba7b0fc03d1b2a4eb5e95911b5deb7b33580746d69'))
    a = json.loads(v18path.read_text())['generation_authorities']; v23 = json.loads(v23path.read_text())
    bm, bf = manifest_files(dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v24_evaluation/78d37b7cdebea46b/manifest.json',
                                sha256='cac499c5b326cff564c2d9a30305308272a1d5f34de5d8ff86e89754dd191bdb'))
    sources = [verify(a['queries']), verify(a['raw_union']), verify(v23['sites']), verify(v23['names'])]
    indexroot = Path(a['hierarchical_index']['path']); sceneroot = Path(a['scene_ir']['path'])
    for key, root in [('hierarchical_index', indexroot), ('scene_ir', sceneroot)]:
        sources.append(verify(dict(path=str(root/'manifest.json'), sha256=a[key]['manifest_sha256'])))
    sources.extend([verify(bm['official_loss_sidecar']), verify(bm['acceptable_operational_sidecar'])])
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    initial_inventory = index_inventory(indexroot/'tantivy')
    meta = record(indexroot/'tantivy/meta.json')
    with Budget(dict(max_seconds=1800, max_rss_bytes=15032385536, max_swapout_pages_delta=0)) as budget:
        qsel = pq.read_table(sources[0]).to_pylist()
        if len(qsel) != 1034 or {r['fold'] for r in qsel} != {2, 3, 4}:
            raise ValueError('Population drift')
        queries = _load_retrieval_queries(tuple((r['query_id'], r['fold']) for r in qsel), sceneroot)
        c = open_workbench_db(output)
        c.execute('CREATE TEMP TABLE labels AS SELECT * FROM read_parquet(?)', [str(bf['outcomes.parquet'])])
        c.execute('CREATE TEMP TABLE official_loss AS SELECT * FROM read_parquet(?)', [str(sources[-2])])
        if c.execute('SELECT count(*),count(DISTINCT query_id) FROM labels WHERE fold IN (2,3,4)').fetchone() != (1034, 1034):
            raise ValueError('Label population drift')
        c.execute('''CREATE TEMP TABLE source_sites AS SELECT s.* FROM read_parquet(?) s
          SEMI JOIN (SELECT DISTINCT gt_siren siren FROM labels) t USING(siren)''', [str(sources[2])])
        c.execute('''CREATE TEMP TABLE source_names AS SELECT n.* FROM read_parquet(?) n
          SEMI JOIN (SELECT DISTINCT gt_siren siren FROM labels) t USING(siren)''', [str(sources[3])])
        table = c.execute('''SELECT l.*,o.parent_in_evidence,o.official_catalog_present,
          (SELECT count(*) FROM source_sites s WHERE s.siren=l.gt_siren) official_sites,
          (SELECT count(*) FROM source_names n WHERE n.siren=l.gt_siren AND n.normalized_value<>'') official_names,
          (SELECT count(*) FROM source_names n WHERE n.siren=l.gt_siren AND n.normalized_value<>'' AND n.selected_for_subject) selected_names
          FROM labels l JOIN official_loss o USING(query_id,fold,gt_siren,gt_siret)''').to_arrow_table()
        c.close()
        byid = {r['query_id']: r for r in table.to_pylist()}
        backend = TantivyBackend(indexroot); tv = backend._tantivy
        out = []
        for i, (qid, fold, query) in enumerate(queries):
            row = byid[qid]
            if row['fold'] != fold:
                raise ValueError('Fold mismatch')
            term = tv.Query.term_query(backend.index.schema, 'siren', row['gt_siren'], 'basic')
            typ = tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic')
            all_sites = tv.Query.boolean_query([(tv.Occur.Must, term), (tv.Occur.Must, typ)])
            row['indexed_parent_sites'] = backend.searcher.search(all_sites, limit=1).count
            local = backend._combined_query(query, term, document_type='siret')
            row['indexed_geo_sites'] = backend.searcher.search(local, limit=1).count
            row['indexed_exact_siret'] = backend.by_siret(row['gt_siret']) is not None
            use_insee = bool(query.insee and backend._has_insee(query.insee))
            row.update(query_name=query.name, query_address=query.address, query_insee=query.insee,
                       query_postcode=query.postcode, effective_geo_field='insee' if use_insee else 'postcode')
            row['matching_channels'] = []
            if row['indexed_geo_sites']:
                for channel, text in text_queries(backend, query):
                    scoped = backend._combined_query(query, text, document_type='siret', siren=row['gt_siren'])
                    if backend.searcher.search(scoped, limit=1, count=False).hits:
                        row['matching_channels'].append(channel)
            row['discovery_diagnosis'] = classify(row)
            out.append(row)
            if (i+1) % 100 == 0:
                print(json.dumps(dict(processed=i+1, total=len(queries))), flush=True)
        if index_inventory(indexroot/'tantivy') != initial_inventory or record(indexroot/'tantivy/meta.json') != meta:
            raise ValueError('Index mutated during audit')
        df = pa.Table.from_pylist(out).to_pandas()
        pq.write_table(pa.Table.from_pylist(out), output/'label_side_diagnoses.parquet', compression='zstd')
    segments = {dim: {str(k): dict(queries=len(v), diagnoses={str(x): int(n) for x,n in v.discovery_diagnosis.value_counts().items()},
                                  exact=int(v.exact_hit.sum()), operational=int(v.operational_hit.sum()))
                      for k,v in df.groupby(dim, dropna=False)}
                for dim in ('parent_in_evidence', 'loss_stage', 'cohort_kind', 'fold', 'data_origin', 'label_kind', 'qualification_v2', 'qualification_v3')}
    result = dict(schema='sireto-v25-discovery-audit-1', decision='PIVOT_GENERATION', queries=len(df),
                  baseline_exact=int(df.exact_hit.sum()), baseline_operational=int(df.operational_hit.sum()),
                  segments=segments, diagnoses={str(k): int(v) for k,v in df.discovery_diagnosis.value_counts().items()},
                  sources=[record(p) for p in sources], configs=[record(v18path),record(v23path)],
                  index_meta=meta, index_files=initial_inventory, index_full_bytes_rehashed=False,
                  implementation=record(Path(__file__)), truth_side_only=True, candidate_generation_performed=False,
                  fold0_opened=False, fold1_opened=False, national_projection=None, resources=budget.report(),
                  files={p.name:record(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'manifest.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('index_files','sources','files')}), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
