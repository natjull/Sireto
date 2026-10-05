"""Whole public Culture export, source-only; no CRM request, GT or retrieval."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata
from urllib.parse import quote
from urllib.request import Request, urlopen
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget, LIMITS, POLICY, quiet, validate_report, parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_postscore_audit_v187 import canonical_epochs, check_epochs
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

OUTPUT = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v310_culture_source')
URL = 'https://ministere-culture.s3.sbg.io.cloud.ovh.net/BASE_DES_LIEUX/base_des_lieux_culturels_ouverts.csv'
SOURCE_PAGE = 'https://www.data.gouv.fr/datasets/base-des-lieux-culturels-ouverts'
EXPECTED_BYTES = 42496931
MAX_DOWNLOAD_BYTES = 67108864
MAX_OUTPUT_BYTES = 268435456
STATUS = 'SEALED_V310_WHOLE_PUBLIC_CULTURE_SOURCE'
FIELDS = tuple(('id;nom_du_lieu;nom_usuel_du_lieu;entite_juridique_rattachement;siret;adresse_complete;'
    'adresse_entree_public;numero_et_libelle_voie;complement_adresse;code_postal;commune;autres_communes;'
    'code_commune;code_arrondissement;departement;code_departement;region;code_region;pays;latitude;longitude;'
    'site_internet_et_autres_liens;telephones;courriels;domaines;domaines_noms_publics;sous_domaines;'
    'sous_domaines_noms_publics;types;types_noms_publics;labels;labels_noms_publics;annee_obtention;description;'
    'images;mentions_legales;auteur_nom_illustre;identifiant_deps;identifiant_origine;source_origine;'
    'accessible_au_public;conditions_ouverture;livraison;retraits_sur_place;accueil_public_sur_rendezvous;'
    'conditions_acces;date_creation;date_modification;coordonnees_geographiques_lieu;'
    'coordonnees_geographiques_commune_lieu').split(';'))
FALSE = {k: False for k in ('truth_opened', 'crm_opened', 'model_fit', 'recall_measured', 'fold0_opened',
    'fold1_opened', 'identifiers_as_features', 'candidates_emitted', 'ready_for_fold1', 'downstream_models_changed',
    'crm_request_sent', 'physical_equivalence_certified', 'juridical_name_inferred_identifier')}


def runtime():
    return dict(python=sys.version, sqlite=sqlite3.sqlite_version, unicode=unicodedata.unidata_version)


def source_rows(path):
    csv.field_size_limit(1048576)
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';', strict=True)
        if reader.fieldnames != list(FIELDS):
            raise ValueError('Exact complete public Culture schema required')
        for ordinal, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()) or set(row) != set(FIELDS):
                raise ValueError('Every original cell retained without silent row drops')
            yield ordinal, tuple(row[field] for field in FIELDS)


def canonical(ordinal, values):
    if type(ordinal) is not int or ordinal < 0 or len(values) != len(FIELDS) or any(type(v) is not str for v in values):
        raise ValueError('Exact source ordinal and raw strings required')
    raw = values[4].strip()
    sid = raw if re.fullmatch('[0-9]{14}', raw) and raw[:9] != '000000000' else None
    return (ordinal, *values, sid)


def alias_rows(ordinal, values):
    return [(ordinal, role, normalized(values[index])) for role, index in (('PLACE_NAME', 1), ('USUAL_NAME', 2))]


def readonly(path):
    c = sqlite3.connect('file:' + quote(str(Path(path).resolve()), safe='/') + '?mode=ro&immutable=1', uri=True)
    c.execute('PRAGMA query_only=ON')
    c.execute('PRAGMA cache_size=-2048')
    return c


def download(path, guard=lambda: None):
    request = Request(URL, headers={'User-Agent': 'SIRETO-WholeOfficialSource/310', 'Accept-Encoding': 'identity'})
    count = 0
    with urlopen(request, timeout=20) as response:
        if (response.status != 200 or response.geturl() != URL
            or response.headers.get('Content-Length') != str(EXPECTED_BYTES)
            or response.headers.get('Content-Type', '').split(';')[0].strip() != 'text/csv'
            or response.headers.get('Content-Encoding', 'identity') != 'identity'):
            raise ValueError('Exact complete public CSV endpoint, bytes and encoding required')
        headers = {key: response.headers.get(key) for key in ('Content-Type', 'Content-Length', 'ETag', 'Last-Modified', 'Date')}
        with Path(path).open('xb') as stream:
            while True:
                guard()
                chunk = response.read(1048576)
                if not chunk:
                    break
                count += len(chunk)
                if count > MAX_DOWNLOAD_BYTES or count > EXPECTED_BYTES:
                    raise ValueError('STOP_WHOLE_DOWNLOAD_BYTES_NO_TRUNCATION')
                stream.write(chunk)
    if count != EXPECTED_BYTES:
        raise ValueError('Complete original response required, no partial snapshot')
    guard()
    return dict(url=URL, source_page=SOURCE_PAGE, headers=headers, response=record(path), full_response_preserved=True)


def compile_db(path, source, guard=lambda: None):
    if Path(path).exists():
        raise ValueError('Fresh own whole Culture projection required')
    c = sqlite3.connect(path)
    count, block, aliases = 0, [], []
    try:
        c.execute('PRAGMA cache_size=-4096')
        c.execute('CREATE TABLE records(ordinal INTEGER PRIMARY KEY,' + ','.join(name + ' TEXT NOT NULL' for name in FIELDS) + ',explicit_siret TEXT)')
        c.execute('CREATE TABLE aliases(ordinal INTEGER NOT NULL,role TEXT NOT NULL,name TEXT NOT NULL,PRIMARY KEY(ordinal,role)) WITHOUT ROWID')
        statement = 'INSERT INTO records VALUES(' + ','.join('?' for _ in range(len(FIELDS) + 2)) + ')'
        for ordinal, values in source_rows(source):
            if ordinal != count:
                raise ValueError('All original rows in physical order required')
            block.append(canonical(ordinal, values))
            aliases.extend(alias_rows(ordinal, values))
            count += 1
            if len(block) == 2048:
                guard()
                c.executemany(statement, block)
                c.executemany('INSERT INTO aliases VALUES(?,?,?)', aliases)
                block, aliases = [], []
        guard()
        if not count:
            raise ValueError('Nonempty complete official source required')
        c.executemany(statement, block)
        c.executemany('INSERT INTO aliases VALUES(?,?,?)', aliases)
        c.execute('CREATE INDEX alias_lookup ON aliases(name,ordinal)')
        c.commit()
    finally:
        c.close()
    if Path(path).stat().st_size > MAX_OUTPUT_BYTES:
        raise ValueError('STOP_SOURCE_PROJECTION_OUTPUT_BYTES')
    return validate_db(path, source, guard)


def validate_db(path, source=None, guard=lambda: None):
    c = readonly(path)
    try:
        guard()
        if c.execute('PRAGMA integrity_check').fetchone() != ('ok',):
            raise ValueError('Complete source integrity required')
        count, invalid, distinct = c.execute('SELECT count(*),sum(explicit_siret IS NULL),count(DISTINCT explicit_siret) FROM records').fetchone()
        aliases, nonempty = c.execute("SELECT count(*),sum(name!='') FROM aliases").fetchone()
        if not count or aliases != 2 * count:
            raise ValueError('All original rows and both name observations required')
        if source is not None:
            rows = c.execute('SELECT * FROM records ORDER BY ordinal')
            names = c.execute('SELECT ordinal,role,name FROM aliases ORDER BY ordinal,role')
            observed = 0
            for ordinal, values in source_rows(source):
                if ordinal % 2048 == 0:
                    guard()
                if rows.fetchone() != canonical(ordinal, values) or [names.fetchone(), names.fetchone()] != sorted(alias_rows(ordinal, values)):
                    raise ValueError('Every original raw field, ordinal and normalized name must roundtrip')
                observed += 1
            if observed != count or rows.fetchone() is not None or names.fetchone() is not None:
                raise ValueError('All source rows roundtrip, no source selection')
        return dict(records=count, name_observations=aliases, nonempty_name_observations=nonempty,
            missing_or_invalid_siret_records=invalid, distinct_explicit_sirets=distinct)
    finally:
        c.close()


def implementation():
    paths = (Path(__file__), ROOT/'docs/retrieval_v310_culture_source_contract.md',
        ROOT/'src/xgb_matcher/retrieval_variants_v118.py', ROOT/'src/xgb_matcher/retrieval_structural_channels_v210.py',
        ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py', ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py',
        ROOT/'src/xgb_matcher/retrieval_postscore_audit_v187.py')
    return {str(path): record(path) for path in paths}


def load(root):
    root = Path(root)
    if not root.resolve().is_relative_to(OUTPUT.resolve()) or (root/'status.json').exists() or any(root.glob('phase/*stop.json')):
        raise ValueError('Own fresh source root, no STOP reuse')
    l = json.loads((root/'launch.json').read_text())
    if (l['url'] != URL or l['source_page'] != SOURCE_PAGE or l['expected_bytes'] != EXPECTED_BYTES
        or l['max_download_bytes'] != MAX_DOWNLOAD_BYTES or l['max_output_bytes'] != MAX_OUTPUT_BYTES
        or l['limits'] != LIMITS or l['fields'] != list(FIELDS) or l['runtime'] != runtime()
        or any(l.get(k) is not False for k in FALSE)):
        raise ValueError('Same frozen complete public source launch required')
    for ref in l['implementation'].values():
        verify(ref)
    return l


def worker(root):
    load(root)
    d = root/'phase'
    try:
        with Budget(LIMITS) as budget:
            def guard():
                if budget.failure:
                    raise RuntimeError(budget.failure)
            snapshot = download(d/'original.csv', guard)
            files = SourceFiles(guard)
            original = files.file(snapshot['response'])
            stats = compile_db(d/'culture.sqlite', original, guard)
            files.verify_after()
            load(root)
            write_json(d/'payload.json', dict(snapshot=snapshot, database=record(d/'culture.sqlite'), stats=stats,
                source_epochs=list(files.entries.values()), all_fields_roundtrip=True, source_only=True, **FALSE))
        validate_report(budget.report())
        write_json(d/'manifest.json', dict(status='QUARANTINED_V310_SOURCE', launch=record(root/'launch.json'),
            preflight=record(d/'preflight.json'), payload=record(d/'payload.json'), resources=budget.report(), **FALSE))
    except BaseException as exc:
        write_json(d/'worker_stop.json', dict(status='STOP', reason=str(exc), **FALSE))
        raise


def input_authority(entries, reference):
    if len(entries) != 1 or entries[0]['reference'] != reference or reference.get('size_bytes') != EXPECTED_BYTES:
        raise ValueError('Exactly one complete original public CSV reference required')
    result = canonical_epochs(entries)
    if result[0]['reference'] != {key: reference[key] for key in ('path', 'sha256')} or result[0]['fingerprint'][2] != EXPECTED_BYTES:
        raise ValueError('Exact original path, checksum and byte size required')
    return result


def admission(root):
    root = Path(root)
    load(root)
    d = root/'phase'
    parent_exit(d)
    m = json.loads((d/'manifest.json').read_text())
    validate_report(m['resources'])
    gate = json.loads(verify(m['preflight']).read_text())
    p = json.loads(verify(m['payload']).read_text())
    snapshot = p['snapshot']
    if (m['status'] != 'QUARANTINED_V310_SOURCE' or m['launch'] != record(root/'launch.json')
        or m['preflight'] != record(d/'preflight.json') or m['payload'] != record(d/'payload.json')
        or p['database'] != record(d/'culture.sqlite') or snapshot['response'] != record(d/'original.csv')
        or snapshot['url'] != URL or snapshot['source_page'] != SOURCE_PAGE or snapshot['full_response_preserved'] is not True
        or snapshot['headers']['Content-Length'] != str(EXPECTED_BYTES) or gate['policy_id'] != POLICY
        or gate['quiet_seconds'] < 30 or gate['pass_system_quiet_policy'] is not True
        or p['all_fields_roundtrip'] is not True or p['source_only'] is not True
        or any(x.get(k) is not False for x in (m, p) for k in FALSE)):
        raise ValueError('Actual own successful complete Culture source producer required')
    entries = input_authority(p['source_epochs'], snapshot['response'])
    check_epochs(entries)
    if verify(p['database']).stat().st_size > MAX_OUTPUT_BYTES or validate_db(verify(p['database'])) != p['stats']:
        raise ValueError('Same full original database required')
    return m, p


def sealed(root):
    root = Path(root)
    m, p = admission(root)
    s = json.loads((root/'manifest.json').read_text())
    if (s['status'] != STATUS or s['producer'] != record(root/'phase/manifest.json')
        or s['database'] != p['database'] or s['original'] != p['snapshot']['response'] or s['stats'] != p['stats']
        or s['source_only'] is not True or any(s.get(k) is not False for k in FALSE)):
        raise ValueError('Same actual full public source producer required')
    return s, p


def run(root):
    if root.exists() or not root.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError('Fresh own source output required')
    root.mkdir(parents=True, exist_ok=False)
    d = root/'phase'
    d.mkdir()
    try:
        write_json(root/'launch.json', dict(url=URL, source_page=SOURCE_PAGE, expected_bytes=EXPECTED_BYTES,
            max_download_bytes=MAX_DOWNLOAD_BYTES, max_output_bytes=MAX_OUTPUT_BYTES,
            limits=LIMITS, fields=FIELDS, runtime=runtime(), implementation=implementation(), **FALSE))
        write_json(d/'preflight.json', quiet(30, 50))
        with (d/'worker.log').open('x') as log:
            supervision = supervise([sys.executable, '-u', '-m', 'scripts.retrieval_v310_culture_source', 'worker',
                '--output', str(root)], log, {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, d/'supervisor_stop.json')
        write_json(d/'supervision.json', supervision)
        m, p = admission(root)
        write_json(root/'manifest.json', dict(status=STATUS, producer=record(d/'manifest.json'),
            original=p['snapshot']['response'], database=p['database'], stats=p['stats'], source_only=True, **FALSE))
        sealed(root)
        print(json.dumps(dict(status=STATUS, stats=p['stats'])), flush=True)
    except BaseException as exc:
        write_json(root/'status.json', dict(status='STOP', reason=str(exc), **FALSE))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('run', 'worker'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    (run if a.mode == 'run' else worker)(a.output)
