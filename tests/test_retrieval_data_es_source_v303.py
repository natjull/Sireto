"""Tiny source fixtures, not national metrics or real process evidence."""
import csv
import sqlite3
import pytest
from scripts import retrieval_v303_data_es_source as source


def write_source(path, rows, extra=True):
    fields = [label for _, label in source.FIELDS] + (['Unprojected official field'] if extra else [])
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields, delimiter=';')
        w.writeheader()
        for row in rows:
            base = {label: '' for _, label in source.FIELDS}
            base.update(row)
            if extra:
                base['Unprojected official field'] = 'retained in full original input'
            w.writerow(base)


def rows():
    return [
        {'SIRET Installation': '10000000100001', "Nom de l'installation sportive": 'ALPHA BETA',
         "Nom de l'équipement sportif": 'GYM ALPHA', 'Nom du propriétaire': 'PUBLIC OWNER',
         'Type de propriétaire': 'Commune', 'Installation hors-service': 'Oui', 'Adresse': 'RUE\nBLEUE'},
        {'SIRET Installation': '', "Nom de l'installation sportive": '[ND]', 'Nom du propriétaire': 'CITY'},
        {'SIRET Installation': '1.00000001E13', "Nom de l'équipement sportif": 'EQUIPMENT'},
        {'SIRET Installation': '00000000000000'},
    ]


def test_all_rows_unknowns_both_names_raw_fields_and_readonly(tmp_path):
    csvpath, db = tmp_path/'source.csv', tmp_path/'projection.sqlite'
    write_source(csvpath, rows())
    stats = source.compile_db(db, csvpath)
    assert stats == dict(records=4, name_observations=8, nonempty_name_observations=3,
        missing_or_invalid_siret_records=3, distinct_explicit_sirets=1)
    assert source.validate_db(db, csvpath) == stats
    c = source.readonly(db)
    assert c.execute('SELECT out_of_service,address,siret FROM records WHERE ordinal=0').fetchone() == ('Oui', 'RUE\nBLEUE', '10000000100001')
    assert c.execute('SELECT raw_siret,siret FROM records WHERE ordinal=2').fetchone() == ('1.00000001E13', None)
    with pytest.raises(sqlite3.OperationalError):
        c.execute('DELETE FROM records')
    c.close()
    with pytest.raises(ValueError):
        source.compile_db(db, csvpath)


def test_changed_or_dropped_source_row_fails_full_roundtrip(tmp_path):
    csvpath, db = tmp_path/'source.csv', tmp_path/'projection.sqlite'
    write_source(csvpath, rows())
    source.compile_db(db, csvpath)
    changed = rows()
    changed[0]['Nom du propriétaire'] = 'INVENTED OWNER'
    write_source(csvpath, changed)
    with pytest.raises(ValueError, match='roundtrip'):
        source.validate_db(db, csvpath)
    write_source(csvpath, rows()[:-1])
    with pytest.raises(ValueError, match='roundtrip'):
        source.validate_db(db, csvpath)


def test_guard_failure_never_admits_partial_source(tmp_path):
    csvpath, db = tmp_path/'source.csv', tmp_path/'projection.sqlite'
    write_source(csvpath, rows())
    def fail():
        raise TimeoutError('Source-only guard')
    with pytest.raises(TimeoutError):
        source.compile_db(db, csvpath, fail)


@pytest.mark.parametrize('bad', [True, -1, 1.5])
def test_ordinals_not_coerced(bad):
    with pytest.raises(ValueError):
        source.canonical(bad, tuple('' for _ in source.FIELDS))


def test_missing_schema_and_non_ascii_or_letter_identifiers_not_repaired(tmp_path):
    path = tmp_path/'bad.csv'
    path.write_text('SIRET Installation\n10000000100001\n')
    with pytest.raises(ValueError):
        list(source.source_rows(path))
    for identifier in ('１０００００００１００００１', '100000001-00001', 'S10000000100001'):
        values = [''] * len(source.FIELDS)
        values[4] = identifier
        assert source.canonical(0, tuple(values))[-1] is None


def test_published_owner_name_never_infers_parent(tmp_path):
    path, db = tmp_path/'source.csv', tmp_path/'source.sqlite'
    write_source(path, [{'Nom du propriétaire': 'COMMUNE ALPHA', 'Type de propriétaire': 'Commune'}])
    source.compile_db(db, path)
    c = source.readonly(db)
    assert c.execute('SELECT siret FROM records').fetchone() == (None,)
    c.close()
