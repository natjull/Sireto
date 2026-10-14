"""Bounded official documentation/metadata capture, never dataset records.

No business queries, contacts records, labels, normalizer changes or retrieval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

OUT = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v149/topad_official_metadata_20261014_night')
DEADLINE = 1791954368
CAP_EACH = 2 * 1024**2
CAP_TOTAL = 8 * 1024**2
DATASETS = dict(topo='topo-fichier-des-entites-topographiques',
    structures='fichier-des-structures-de-la-dgfip',
    missions='fichiers-des-structures-du-referentiel-topad-territorialise',
    competences='fichier-des-competences-territoriales-des-services-dgfip')
TOPO = Path('/Users/nathanjullia/Documents/Projets/SIRETO/tmp/pdfs/night_topo_reference_20261014_TyIEDZ/description_fichier_topo_202602.pdf')
TOPO_SHA = '0ebf1891d3747b614c7a3b2da8a5885f6b7456c59d548c607174fa85c0530412'
RECIPE = dict(role='OFFICIAL_SCHEMAS_DOCUMENTATION_RESOURCE_METADATA_ONLY',
    no_dataset_records=True, no_contact_records=True, no_labels_or_outcomes=True,
    no_business_queries_or_names=True, no_normalizer_or_score_change=True,
    no_retrieval_or_training=True, no_geocoder_claim=True, no_recall_claim=True,
    domains=['data.economie.gouv.fr', 'www.data.gouv.fr', 'www.insee.fr'],
    max_bytes_per_response=CAP_EACH, max_total_network_bytes=CAP_TOTAL, deadline_unix=DEADLINE,
    cutoff='LIVE_PUBLICATION_METADATA_NOT_CERTIFIED_HISTORICAL_DATA_ASOF',
    source_ambiguity='TOPO_CODES_TYPES_CURRENT_LABELS_NOT_CRM_ALIASES_NOT_HISTORY_ENDPOINTS')


def guard():
    if time.time() >= DEADLINE:
        raise TimeoutError('Absolute night deadline')


def record(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def capture():
    guard()
    if OUT.exists():
        raise ValueError('Fresh immutable metadata namespace required')
    OUT.mkdir(parents=True)
    write('recipe.json', RECIPE)
    receipts = []
    network = 0

    def fetch(name, url):
        nonlocal network
        guard()
        request = urllib.request.Request(url, headers={'User-Agent': 'SIRETO-readonly-official-schema-audit/1.0'})
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(CAP_EACH + 1)
            headers = dict(response.headers)
            actual_url = response.url
        if len(body) > CAP_EACH or network + len(body) > CAP_TOTAL:
            raise ValueError('Bounded documentation budget exceeded')
        network += len(body)
        path = OUT / name
        path.write_bytes(body)
        receipt = dict(request_url=url, response_url=actual_url, headers=headers,
                       captured_unix=time.time(), artifact=record(path), bytes=len(body))
        receipts.append(receipt)
        return body

    summaries = {}
    for role, dataset in DATASETS.items():
        endpoint = 'https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/' + dataset
        d = json.loads(fetch(role + '_metadata.json', endpoint))
        meta = d['metas']['default']
        fields = {f['name']: dict(type=f['type'], label=f.get('label'), description=f.get('description')) for f in d['fields']}
        summaries[role] = dict(dataset=dataset, metadata_url=endpoint,
            producer=meta.get('publisher'), record_count=meta.get('records_count'),
            modified=meta.get('modified'), data_processed=meta.get('data_processed'),
            frequency=meta.get('update_frequency') or d['metas'].get('dcat', {}).get('accrualperiodicity'),
            licence=meta.get('license'), licence_url=meta.get('license_url'),
            temporal_coverage=d['metas'].get('dcat', {}).get('temporal'),
            fields=fields, field_count=len(fields), geometry_types=meta.get('geometry_types'),
            bbox=meta.get('bbox'), attachments=d.get('attachments'),
            advertised_alternative_exports=d.get('alternative_exports'),
            live_export_urls=dict(csv=endpoint + '/exports/csv', json=endpoint + '/exports/json'),
            data_records_downloaded=False)
        if role != 'topo':
            attachment = d['attachments'][0]
            fetch(role + '_producer_description.pdf', attachment['url'])
    # Resource listings only, not live exports or dataset records.
    slugs = dict(topo='fichier-des-entites-topographiques-topo-dgfip-1',
        structures='fichier-des-structures-de-la-dgfip-1',
        missions='fichier-des-missions-du-referentiel-topad',
        competences='fichier-des-competences-territoriales-des-services-dgfip-1')
    for role, slug in slugs.items():
        url = 'https://www.data.gouv.fr/api/1/datasets/' + slug + '/'
        try:
            d = json.loads(fetch(role + '_datagouv_metadata.json', url))
            summaries[role]['datagouv_resources'] = [dict(title=r.get('title'), format=r.get('format'),
                url=r.get('url'), filesize=r.get('filesize'), checksum=r.get('checksum'),
                last_modified=r.get('last_modified'), resource_id=r.get('id')) for r in d.get('resources', [])]
        except urllib.error.HTTPError as error:
            summaries[role]['datagouv_metadata_lookup_unavailable'] = dict(url=url, status=error.code)
    optional_documents = []
    for name, url in [('insee_sirene_variable_description.pdf', 'https://www.insee.fr/fr/statistiques/fichier/3711695/Description-liste-sirene-fr.pdf'),
                      ('insee_ril_2024_metropole.pdf', 'https://www.insee.fr/fr/statistiques/fichier/1303418/presentation_diffusion_ril_2024_metropole.pdf')]:
        try:
            fetch(name, url)
            optional_documents.append(dict(url=url, captured=True, artifact=record(OUT / name)))
        except urllib.error.HTTPError as error:
            optional_documents.append(dict(url=url, captured=False, http_status=error.code,
                no_full_pdf_schema_validation_claim=True))
    topo = record(TOPO)
    if topo['sha256'] != TOPO_SHA:
        raise ValueError('Provided root-inspected TOPO PDF binding drift')
    write('http_receipts.json', dict(receipts=receipts, network_bytes=network,
                                   no_dataset_or_contact_records_downloaded=True))
    write('schema_audit.json', dict(status='CAPTURED_OFFICIAL_DOCUMENTATION_AND_SCHEMAS_NOT_RETRIEVAL',
        implementation=record(Path(__file__)), recipe=record(OUT / 'recipe.json'),
        datasets=summaries, existing_topo_documentation=topo, optional_documents=optional_documents,
        http_receipts=record(OUT / 'http_receipts.json'), network_bytes=network,
        no_dataset_or_contact_records_downloaded=True, no_labels_or_outcomes_read=True,
        no_normalizer_or_score_change=True, no_retrieval_or_training=True,
        schema_findings=dict(topo_has_xy=False,
            topo_coordinates_absence_basis='NO_COORDINATE_FIELDS_GEOMETRY_TYPES_NULL_BBOX_NULL',
            topo_has_explicit_old_to_new_street_endpoint=False,
            structures_has_siret='code_siret' in summaries['structures']['fields'],
            structures_has_typed_functional_links='codes_type_liens_fonctionnels' in summaries['structures']['fields'],
            structures_has_technical_hierarchy=all(k in summaries['structures']['fields'] for k in ['numero_ua_de_la_dir', 'numero_ua_de_la_ci', 'numero_ua_de_la_cb']),
            scope='DGFIP_ADMINISTRATIVE_UNITS_AND_THEIR_MISSIONS_TERRITORIAL_COMPETENCES_NOT_GENERAL_ORGANISMS_REGISTRY',
            archives='NO_DATED_DATA_EXPORT_ARCHIVE_ADVERTISED_IN_METADATA_INSPECTED_NO_GLOBAL_ABSENCE_CLAIM',
            historical_cutoff_admissibility='LIVE_DATA_PROCESSED_20261013_NOT_CERTIFIED_20260801_OR_EARLIER_ASOF'),
        completed_unix=time.time()))
    print('CAPTURED_OFFICIAL_METADATA_BYTES', network, flush=True)


def run():
    child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.embeddinggemma2_v149_topad_metadata_audit', 'capture'])
    while child.poll() is None:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                guard()
            except BaseException:
                child.terminate()
                child.wait(timeout=10)
                raise
    if not OUT.exists():
        raise RuntimeError('Metadata capture did not create its namespace')
    write('owned_tool_exit.json', dict(exit_code=child.returncode, own_child_exit_confirmed=True,
        owned_pid=child.pid, completed_unix=time.time(), no_dataset_or_contact_records_downloaded=True,
        no_retrieval_or_training=True))
    if child.returncode:
        raise RuntimeError('Owned metadata capture failed')
    write('manifest.json', dict(status='SEALED_BOUNDED_OFFICIAL_METADATA_DOCUMENTATION_AUDIT',
        schema_audit=record(OUT / 'schema_audit.json'), http_receipts=record(OUT / 'http_receipts.json'),
        owned_exit=record(OUT / 'owned_tool_exit.json'), recipe=record(OUT / 'recipe.json'),
        source_artifacts={p.name: record(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.pdf', '.json') and p.name not in ('manifest.json',)},
        no_labels_or_outcomes_read=True, no_normalizer_or_score_change=True, no_retrieval_or_training=True))
    print(json.dumps(record(OUT / 'manifest.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'capture'))
    args = parser.parse_args()
    globals()[args.action]()
