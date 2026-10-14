"""Seven numeric-only STRUCTURES aggregates, no individual data records.

Addendum resolves the 2024 notice versus 2026 live-schema presence ambiguity.
Two preceding diagnostic aggregate requests plus these seven stay below ten.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from scripts import embeddinggemma2_v149_topad_metadata_audit as base

OUT = base.OUT.parent / 'topad_aggregate_addendum_20261014_night'
FIELDS = ['liens_fonctionnels', 'codes_type_liens_fonctionnels', 'code_siret',
          'numero_ua_de_la_dir', 'numero_ua_de_la_ci', 'numero_ua_de_la_cb']
RECIPE = dict(role='OFFICIAL_DGFIP_NUMERIC_ONLY_STRUCTURES_AGGREGATE_DOCUMENTARY_ADDENDUM',
    requests=7, preceding_exploratory_requests=2, total_aggregate_requests=9,
    timeout_seconds=10, response_cap_bytes=8192, no_individual_records=True,
    no_names_or_contacts=True, no_labels_or_outcomes=True, no_normalizer_or_score_change=True,
    nonempty='NONNULL_AND_NOT_EMPTY_STRING_OR_SINGLE_SPACE',
    whitespace_only_other_lengths='NOT_CERTIFIED_WITHOUT_SERVER_TRIM_TRIM_WHERE_REJECTED_ODSQL400',
    technical_nonzero='UA_PARENT_NOT_TEN_ZEROS_SIRET_NOT_FOURTEEN_ZEROS',
    meaning='CURRENT_FIELD_PRESENCE_NOT_FUNCTIONAL_LINK_CODE_SEMANTICS_OR_PHYSICAL_SITE_IDENTITY',
    deadline_unix=1791954368)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def record(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)


def capture():
    base.guard()
    if OUT.exists():
        raise ValueError('Fresh immutable documentary addendum')
    OUT.mkdir(parents=True)
    write('recipe.json', RECIPE)
    endpoint = 'https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/fichier-des-structures-de-la-dgfip/records'
    selects = ['count(*) as total'] + ['count(' + f + ') as ' + f + '_nonnull' for f in FIELDS]
    queries = [('all_nonnull', dict(select=','.join(selects), limit=1))]
    for field in FIELDS:
        where = field + " is not null and " + field + " != '' and " + field + " != ' '"
        if field.startswith('numero_ua_'):
            where += ' and ' + field + " != '0000000000'"
        elif field == 'code_siret':
            where += ' and ' + field + " != '00000000000000'"
        queries.append((field + '_nonempty', dict(select='count(*) as n', where=where, limit=1)))
    results = {}
    network = 0
    for key, params in queries:
        base.guard()
        url = endpoint + '?' + urllib.parse.urlencode(params)
        receipt = dict(url=url, params=params, captured_unix=time.time())
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'SIRETO-bounded-numeric-only-aggregate-audit/1.0'})
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read(8193)
                receipt['response_headers'] = dict(response.headers)
            if len(raw) > 8192:
                raise ValueError('Numeric aggregate response cap')
            network += len(raw)
            data = json.loads(raw)
            rows = data.get('results')
            allowed = {'total'} | {f + '_nonnull' for f in FIELDS} if key == 'all_nonnull' else {'n'}
            if not isinstance(rows, list) or len(rows) != 1 or set(rows[0]) != allowed or any(not isinstance(v, (int, float)) for v in rows[0].values()):
                raise ValueError('Not exclusively the requested numeric aggregates; do not persist response')
            receipt.update(status='NUMERIC_ONLY_AGGREGATE_PASS', result=rows[0], response_bytes=len(raw))
        except urllib.error.HTTPError as error:
            receipt.update(status='UNAVAILABLE_NOT_ZERO', http_status=error.code)
        results[key] = receipt
        write(key + '.json', receipt)
    write('interpretation.json', dict(status='CURRENT_AGGREGATE_PRESENCE_CHECK_NOT_RECORD_INSPECTION',
        results=results, network_bytes=network, source_metadata_manifest=record(base.OUT / 'manifest.json'),
        structures_notice=record(base.OUT / 'structures_producer_description.pdf'),
        missions_notice=record(base.OUT / 'missions_producer_description.pdf'),
        competences_notice=record(base.OUT / 'competences_producer_description.pdf'),
        documentary_limits=dict(
            functional_links='2024_NOTICE_PAGE9_SAYS_FIELDS39_AND44_NOT_PUBLISHED_BUT_2026_API_NONNULL_AGGREGATE318_REQUIRES_CURRENT_DOCUMENTATION',
            hierarchy='NOTICE_FIELDS14_19_DOCUMENT_UA_DIR_CI_CB_TECHNICAL_HIERARCHY_NOT_SIRET_IDENTITY',
            recodification='NOTICE_FIELDS12_13_UA_FUNCTIONAL_CODE_RECODIFICATION_NOT_STREET_TRANSFER_OR_SIRET_SUCCESSION',
            scope='ONLY_DGFIP_SERVICE_UNITS_NOT_GENERAL_PUBLIC_ORGANISM_OPERATOR_REGISTRY',
            cutoff='LIVE_DATA_PROCESSED13OCT2026_NO_CERTIFIED_PRE20260801_ARCHIVE',
            archives='NONE_ADVERTISED_IN_CAPTURED_RESOURCE_LISTINGS_NO_GLOBAL_ABSENCE_CLAIM',
            insee_type_table='SIRENE_DESCRIPTION_URL404_RIL2024_URL500_FULL_CURRENT_TABLE_NOT_FROZEN',
            contacts='PHONE_EMAIL_PERSON_CONTACTS_EXCLUDED_FROM_FEATURES_NO_RECORDS_DOWNLOADED'),
        generic_typed_representation_direction=dict(implemented=False,
            provenance='SOURCE_AUTHORITY_TABLE_VERSION_CONTENT_HASH_RAW_TYPE_CODE_AND_OFFICIAL_LABEL',
            own_street='KEEP_RAW_NATURE_AND_OWN_LABEL_SEPARATE_TYPE_EXPANSION_ONLY_IN_TYPE_FIELD_NOT_GLOBAL_WORD_REPLACEMENT',
            topo_class='KEEP_TYPE_VOIE1_ROAD_2_IMMOBILIER_3_LIEUDIT_4_PSEUDOVOIE_SEPARATE_FROM_NATURE_ABBREVIATION',
            ambiguity='ALL_SAME_COMMUNE_TYPED_LABEL_ENTITIES_RETAINED_CANCELED_STATE_DATES_PRESERVED_NO_SINGLE_ALIAS_CHOSEN',
            history='CANCELED_TOPO_AND_LAST_TRANSITION_DATE_NOT_EXPLICIT_OLD_TO_NEW_ENDPOINT_OR_OLD_NAME_CHAIN',
            coordinates='NO_TOPO_XY_NO_GEOCODER_TOPO_NO_STREET_OR_SITE_COORDINATE_INVENTED',
            dgfip_relations='UA_HAS_MISSION_AND_UA_HAS_TERRITORIAL_COMPETENCE_VALIDITY_PARITY_BOUNDS_NOT_IMPLANTATION_OR_IDENTITY',
            identifiers='TECHNICAL_JOIN_ONLY_NOT_MATCHING_FEATURES',
            source_folding='OFFICIAL_SOURCE_TABLE_NOT_CRM_ALIAS_REQUIRES_SEPARATE_VERSION_FREEZE_AND_ADMISSIBILITY_BEFORE_ANY_DIAGNOSTIC'),
        no_labels_or_outcomes=True, no_retrieval_or_training=True, no_normalizer_or_score_change=True,
        no_recall_claim=True, no_future_activation_inferred=True, completed_unix=time.time()))
    print('NUMERIC_AGGREGATE_ADDENDUM_NETWORK_BYTES', network, flush=True)


def run():
    child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.embeddinggemma2_v150_topad_aggregate_addendum', 'capture'])
    while child.poll() is None:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                base.guard()
            except BaseException:
                child.terminate()
                child.wait(timeout=10)
                raise
    if not OUT.exists():
        raise RuntimeError('No documentary addendum namespace')
    write('owned_tool_exit.json', dict(exit_code=child.returncode, own_child_exit_confirmed=True,
                                      owned_pid=child.pid, completed_unix=time.time()))
    if child.returncode:
        raise RuntimeError('Owned documentary addendum failed')
    write('manifest.json', dict(status='SEALED_BOUNDED_NUMERIC_AGGREGATE_DOCUMENTARY_ADDENDUM',
        implementation=record(Path(__file__)), interpretation=record(OUT / 'interpretation.json'),
        artifacts={p.name: record(p) for p in OUT.iterdir() if p.suffix == '.json'},
        no_individual_records_names_contacts_labels_or_outcomes=True,
        no_normalizer_score_retrieval_or_training_change=True))
    print(json.dumps(record(OUT / 'manifest.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'capture'))
    args = parser.parse_args()
    globals()[args.action]()
