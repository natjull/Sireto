"""Static national temporal lineage: metadata/footers only, no names or labels.

Known V117 gap revalidated, not a new recall result. No full corpus hashes,
anti-joins, rebuild, native queries or parsed data values are performed.
"""
from pathlib import Path
import hashlib
import time
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_source_coverage_v117 import temporal_coverage

OUT=io.HOME/'v159/temporal_lineage_20261014_night'
DATA=io.DATA


def small_record(path):
    path=Path(path)
    if path.stat().st_size>10*1024**2:raise ValueError('Metadata-only hash scope')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size)


def footer(path,source_stats=False):
    import pyarrow.parquet as pq
    path=Path(path);p=pq.ParquetFile(path);m=p.metadata
    row=dict(path=str(path),size_bytes=path.stat().st_size,rows=m.num_rows,row_groups=m.num_row_groups,
        schema_columns=p.schema_arrow.names,data_values_read=False,full_content_sha256_recomputed=False)
    if source_stats:
        positions=[i for i in range(m.num_columns) if m.schema.column(i).path=='source']
        if len(positions)!=1:raise ValueError('Structured source footer column')
        ranges={};missing=0;nulls=0
        for i in range(m.num_row_groups):
            s=m.row_group(i).column(positions[0]).statistics
            if s is None or not s.has_min_max:missing+=1;continue
            key=(str(s.min),str(s.max));ranges[key]=ranges.get(key,0)+1;nulls+=s.null_count or 0
        row.update(source_ranges=[dict(minimum=a,maximum=b,row_groups=n) for (a,b),n in sorted(ranges.items())],
            source_stats_missing_row_groups=missing,source_nulls_declared=nulls)
    return row


def main():
    start=time.monotonic()
    if OUT.exists():raise ValueError('Fresh metadata-only lineage namespace')
    index=DATA/'indexes/hierarchical_retrieval_v3/4b332899d8698354/manifest.json';im=io.read(index)
    doc=Path(im['sources']['dossier_manifest']['path']);dm=io.read(doc)
    if small_record(doc)!=im['sources']['dossier_manifest']:raise ValueError('Actual index→projection manifest binding')
    dossier=DATA/'sources/siren_dossier_v3/01eb624cca97c6f3/manifest.json';ds=io.read(dossier)
    if small_record(dossier)['sha256']!=dm['dossier_manifest_sha256']:raise ValueError('Actual projection→dossier binding')
    builder=io.ROOT/'scripts/build_hierarchical_retrieval_index.py';policy=io.ROOT/'config/siren_name_portfolio_v1.json'
    if small_record(builder)['sha256']!=im['builder_sha256'] or small_record(policy)['sha256']!=dm['name_portfolio_policy_sha256']:raise ValueError('Actual index builder/policy code hashes')
    merge=DATA/'sources/official_evidence/official_history_2008_2026_v1/manifest.json';merged=io.read(merge)
    emitted=merged['artifacts']['official_evidence.parquet']
    if emitted['sha256'] not in {r['sha256'] for r in ds['inputs']['official_evidence']}:raise ValueError('Actual merged input hash binding')
    bulk=DATA/'sources/official_evidence/rne_bulk_formalities_20260304_v3'
    input_refs=ds['inputs']['official_evidence']
    core_footers={name:footer(path,True) for name,path in [('merged_rne_bodacc',merge.parent/'official_evidence.parquet'),('bulk_rne_candidate',bulk/'official_evidence.parquet')]}
    expected_other=[r for r in input_refs if r['sha256']!=emitted['sha256']]
    if len(expected_other)!=1 or expected_other[0]['size_bytes']!=core_footers['bulk_rne_candidate']['size_bytes']:
        raise ValueError('Candidate bulk input size binding')
    old=DATA/'audits/retrieval_v129_national_variants/projection_20261003_a/plan.json';plan=io.read(old)
    dictionary=DATA/'audits/retrieval_v130_variant_dictionary/compile_20261003_a/manifest.json';compiled=io.read(dictionary)
    hist={key:footer(value['path']) for key,value in plan['sources'].items() if key.startswith('history_')}
    raw=DATA/'sources/official_evidence/sirene_history_20260801_v1/manifest.json';raw_meta=io.read(raw)
    OUT.mkdir(parents=True)
    io.write(OUT/'report.json',dict(status='COMPLETE_METADATA_ONLY_TEMPORAL_LINEAGE_KNOWN_GAP_NOT_SOURCE_DIFFERENCE_COUNT_OR_RECALL',
        implementation=small_record(Path(__file__)),known_prior_audit_commit='6e80f66a',prior_report=small_record(io.ROOT/'docs/retrieval_v117_structural_coverage_audit.md'),
        actual_index_manifest=small_record(index),actual_builder=small_record(builder),actual_portfolio_policy=small_record(policy),
        actual_projection_manifest=small_record(doc),actual_dossier_manifest=small_record(dossier),actual_merge_manifest=small_record(merge),
        direct_index_projection_and_dossier_metadata_hashes_verified=True,merged_input_declared_hash_links_actual_dossier=True,
        source_input_content_sha256s_are_declared_not_rehashed=True,bulk_candidate_size_footer_and_manifest_consistent_not_cryptographic_identity_replay=True,
        dossier_declared_input_hashes=input_refs,core_input_footers=core_footers,
        source_family_status=temporal_coverage(dm),separate_insee_history_manifest=small_record(raw),separate_insee_canonical_counts=raw_meta['counts'],
        full_variant_plan=small_record(old),history_source_footers=hist,history_site_no_address_or_xy_columns=not any(c in hist['history_site']['schema_columns'] for c in ('numeroVoieEtablissement','libelleVoieEtablissement','codePostalEtablissement','coordonneeLambertAbscisseEtablissement')),
        full_variant_dictionary_manifest=small_record(dictionary),dictionary_summary={k:compiled[k] for k in ('status','original_observations','lexical_observations','facts','names','complete_retrieval_index')},
        static_semantic_losses=dict(canonical_insee_history_name_kind_replaced_by_generic_HISTORICAL=True,
            bodacc_publication_date_reuses_valid_from_no_separate_date_kind=True,
            portfolio_max_dates_and_min_ids_are_summary_not_all_intervals=True,
            index_payload_has_flat_names_and_source_id_sets_no_per_name_validity_or_subject_scope_pairing=True,
            source_specific_name_fields_preserved_but_not_name_source_date_mapping=True,
            only_current_name_ngrams_and_current_exact_fields_in_native_main=True,
            dictionary_observations_keep_period_and_name_scope_but_runtime_sqlite_owner_alias_lookup_loses_it=True,
            native_successions_V23_read_raw_economic_continuity_and_dated_endpoints_independently=True),
        no_global_missing_historical_name_pairs_count_measured=True,no_exact_absence_in_other_sources_claim=True,
        no_person_names_addresses_labels_data_values_or_closed_set_read=True,no_native_fit_or_index_change=True,elapsed_seconds=time.monotonic()-start))
    print(dict(source_family_status=temporal_coverage(dm),history_rows={k:v['rows'] for k,v in hist.items()},
        input_source_footer_ranges={k:v['source_ranges'] for k,v in core_footers.items()},elapsed_seconds=time.monotonic()-start),flush=True)


if __name__=='__main__':main()
