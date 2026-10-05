"""All-row source-only Data ES projection; no CRM, labels or retrieval."""
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
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget, LIMITS, POLICY, quiet, validate_report, parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_postscore_audit_v187 import canonical_epochs, check_epochs
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

OUTPUT = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v303_data_es_source')
INPUT = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/open_data_v5/data_es_complet_20260819.csv',
    sha256='b7c49db1667af423e3ee381cdcd7371eb0a5cc86cd895f4b1f88abbbc2ca6a28', size_bytes=331746052)
MAX_OUTPUT_BYTES = 268435456
STATUS = 'SEALED_V303_ALL_ROWS_DATA_ES_SOURCE_PROJECTION'
FALSE = {k: False for k in ('truth_opened', 'crm_opened', 'model_fit', 'recall_measured', 'fold0_opened',
    'fold1_opened', 'identifiers_as_features', 'candidates_emitted', 'ready_for_fold1', 'downstream_models_changed')}
FIELDS = (
    ('equipment_id', "Numéro de l'équipement sportif"),
    ('installation_id', "Numéro de l'installation sportive"),
    ('installation_name', "Nom de l'installation sportive"),
    ('equipment_name', "Nom de l'équipement sportif"),
    ('raw_siret', 'SIRET Installation'), ('raw_insee', 'Commune INSEE'), ('raw_postcode', 'Code Postal'),
    ('address', 'Adresse'), ('owner_name', 'Nom du propriétaire'), ('owner_type', 'Type de propriétaire'),
    ('operator_type', 'Type de gestionnaire'), ('out_of_service', 'Installation hors-service'),
    ('updated', 'Mise à jour de la fiche équipement'), ('raw_latitude', 'Latitude'), ('raw_longitude', 'Longitude'))


def runtime():
    return dict(python=sys.version, sqlite=sqlite3.sqlite_version, unicode=unicodedata.unidata_version)


def source_rows(path):
    csv.field_size_limit(1048576)
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';', strict=True)
        header = reader.fieldnames
        if not header or len(set(header)) != len(header) or not {label for _, label in FIELDS} <= set(header):
            raise ValueError('Original unique complete Data ES field schema required')
        for ordinal, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()) or set(row) != set(header):
                raise ValueError('Every source row must parse without dropped or extra cells')
            yield ordinal, tuple(row[label] for _, label in FIELDS)


def canonical(ordinal, values):
    if type(ordinal) is not int or ordinal < 0 or len(values) != len(FIELDS) or any(type(v) is not str for v in values):
        raise ValueError('Exact source ordinal and unchanged raw string fields required')
    raw = values[4].strip()
    # No stripping scientific notation/letters into an invented ID.
    siret = raw if re.fullmatch('[0-9]{14}', raw) and raw[:9] != '000000000' else None
    return (ordinal, *values, siret)


def alias_rows(ordinal, values):
    return [(ordinal, role, normalized(values[index])) for role, index in (('INSTALLATION', 2), ('EQUIPMENT', 3))]


def readonly(path):
    c = sqlite3.connect('file:' + quote(str(Path(path).resolve()), safe='/') + '?mode=ro&immutable=1', uri=True)
    c.execute('PRAGMA query_only=ON')
    c.execute('PRAGMA cache_size=-2048')
    return c


def compile_db(path, source, guard=lambda: None):
    if Path(path).exists():
        raise ValueError('Fresh own Data ES projection required')
    c = sqlite3.connect(path)
    count, block, aliases = 0, [], []
    try:
        c.execute('PRAGMA cache_size=-4096')
        columns = ','.join(name + ' TEXT NOT NULL' for name, _ in FIELDS)
        c.execute('CREATE TABLE records(ordinal INTEGER PRIMARY KEY,' + columns + ',siret TEXT)')
        c.execute('CREATE TABLE aliases(ordinal INTEGER NOT NULL,role TEXT NOT NULL,name TEXT NOT NULL,PRIMARY KEY(ordinal,role)) WITHOUT ROWID')
        statement = 'INSERT INTO records VALUES(' + ','.join('?' for _ in range(len(FIELDS) + 2)) + ')'
        for ordinal, values in source_rows(source):
            if ordinal != count:
                raise ValueError('All physical source rows in original order required')
            block.append(canonical(ordinal, values))
            aliases.extend(alias_rows(ordinal, values))
            count += 1
            if len(block) == 4096:
                guard()
                c.executemany(statement, block)
                c.executemany('INSERT INTO aliases VALUES(?,?,?)', aliases)
                block, aliases = [], []
        guard()
        if not count:
            raise ValueError('Complete nonempty source required')
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
            raise ValueError('Complete source projection integrity required')
        count, invalid, distinct = c.execute('SELECT count(*),sum(siret IS NULL),count(DISTINCT siret) FROM records').fetchone()
        alias_count, nonempty = c.execute("SELECT count(*),sum(name!='') FROM aliases").fetchone()
        if not count or alias_count != 2 * count:
            raise ValueError('All records and both source name observations required')
        if source is not None:
            rows = c.execute('SELECT * FROM records ORDER BY ordinal')
            names = c.execute('SELECT ordinal,role,name FROM aliases ORDER BY ordinal,role')
            observed = 0
            for ordinal, values in source_rows(source):
                if ordinal % 4096 == 0:
                    guard()
                if rows.fetchone() != canonical(ordinal, values) or [names.fetchone(), names.fetchone()] != sorted(alias_rows(ordinal, values)):
                    raise ValueError('Every original projected field, ordinal and alias must roundtrip')
                observed += 1
            if observed != count or rows.fetchone() is not None or names.fetchone() is not None:
                raise ValueError('All source rows roundtrip, no source selection')
        return dict(records=count, name_observations=alias_count, nonempty_name_observations=nonempty,
            missing_or_invalid_siret_records=invalid, distinct_explicit_sirets=distinct)
    finally:
        c.close()


def implementation():
    paths = (Path(__file__), ROOT/'docs/retrieval_v303_data_es_source_contract.md',
        ROOT/'src/xgb_matcher/retrieval_variants_v118.py', ROOT/'src/xgb_matcher/retrieval_structural_channels_v210.py',
        ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py', ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py',
        ROOT/'src/xgb_matcher/retrieval_postscore_audit_v187.py')
    return {str(path): record(path) for path in paths}


def load(root):
    root = Path(root)
    if not root.resolve().is_relative_to(OUTPUT.resolve()) or (root/'status.json').exists() or any(root.glob('phase/*stop.json')):
        raise ValueError('Own fresh source projection, no STOP reuse')
    launch = json.loads((root/'launch.json').read_text())
    if (launch['input'] != INPUT or launch['limits'] != LIMITS or launch['max_output_bytes'] != MAX_OUTPUT_BYTES
        or launch['fields'] != [list(pair) for pair in FIELDS] or launch['runtime'] != runtime()
        or any(launch.get(k) is not False for k in FALSE)):
        raise ValueError('Original source-only immutable launch required')
    for reference in launch['implementation'].values():
        verify(reference)
    return launch


def worker(root):
    load(root)
    d = root/'phase'
    try:
        with Budget(LIMITS) as budget:
            def guard():
                if budget.failure:
                    raise RuntimeError(budget.failure)
            files = SourceFiles(guard)
            source = files.file(INPUT)
            stats = compile_db(d/'data_es.sqlite', source, guard)
            files.verify_after()
            load(root)
            write_json(d/'payload.json', dict(database=record(d/'data_es.sqlite'), stats=stats,
                source_epochs=list(files.entries.values()), all_projected_records_roundtrip=True,
                all_unprojected_cells_preserved_in_original_source=True, source_only=True, **FALSE))
        validate_report(budget.report())
        write_json(d/'manifest.json', dict(status='QUARANTINED_V303_SOURCE', launch=record(root/'launch.json'),
            preflight=record(d/'preflight.json'), payload=record(d/'payload.json'), resources=budget.report(), **FALSE))
    except BaseException as exc:
        write_json(d/'worker_stop.json', dict(status='STOP', reason=str(exc), **FALSE))
        raise


def admission(root):
    root = Path(root)
    load(root)
    d = root/'phase'
    parent_exit(d)
    m = json.loads((d/'manifest.json').read_text())
    validate_report(m['resources'])
    gate = json.loads(verify(m['preflight']).read_text())
    p = json.loads(verify(m['payload']).read_text())
    if (m['status'] != 'QUARANTINED_V303_SOURCE' or m['launch'] != record(root/'launch.json')
        or m['preflight'] != record(d/'preflight.json') or m['payload'] != record(d/'payload.json')
        or p['database'] != record(d/'data_es.sqlite') or gate['policy_id'] != POLICY or gate['quiet_seconds'] < 30
        or gate['pass_system_quiet_policy'] is not True or p['all_projected_records_roundtrip'] is not True
        or p['all_unprojected_cells_preserved_in_original_source'] is not True or p['source_only'] is not True
        or any(x.get(k) is not False for x in (m, p) for k in FALSE)):
        raise ValueError('Actual own complete successful source producer required')
    entries = canonical_epochs(p['source_epochs'])
    if len(entries) != 1 or entries[0]['reference'] != INPUT:
        raise ValueError('Original complete CSV input authority required')
    check_epochs(entries)
    if verify(p['database']).stat().st_size > MAX_OUTPUT_BYTES or validate_db(verify(p['database'])) != p['stats']:
        raise ValueError('Same complete source database required')
    return m, p


def sealed(root):
    root = Path(root)
    m, p = admission(root)
    s = json.loads((root/'manifest.json').read_text())
    if (s['status'] != STATUS or s['input'] != INPUT or s['producer'] != record(root/'phase/manifest.json')
        or s['database'] != p['database'] or s['stats'] != p['stats'] or any(s.get(k) is not False for k in FALSE)):
        raise ValueError('Same actual all-row source producer required')
    return s, p


def run(root):
    if root.exists() or not root.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError('Fresh own source output required')
    root.mkdir(parents=True, exist_ok=False)
    d = root/'phase'
    d.mkdir()
    try:
        write_json(root/'launch.json', dict(input=INPUT, limits=LIMITS, max_output_bytes=MAX_OUTPUT_BYTES,
            fields=FIELDS, runtime=runtime(), implementation=implementation(), **FALSE))
        write_json(d/'preflight.json', quiet(30, 50))
        with (d/'worker.log').open('x') as log:
            supervision = supervise([sys.executable, '-u', '-m', 'scripts.retrieval_v303_data_es_source', 'worker',
                '--output', str(root)], log, {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, d/'supervisor_stop.json')
        write_json(d/'supervision.json', supervision)
        m, p = admission(root)
        write_json(root/'manifest.json', dict(status=STATUS, input=INPUT, producer=record(d/'manifest.json'),
            database=p['database'], stats=p['stats'], source_only=True, **FALSE))
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
