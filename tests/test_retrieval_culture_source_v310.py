"""Small source fixtures, never national producer proof or recall."""
import csv
import sqlite3
import pytest
from scripts import retrieval_v310_culture_source as source
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_v20_workbench import record


def write_source(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=source.FIELDS, delimiter=';')
        w.writeheader()
        for row in rows:
            base = dict.fromkeys(source.FIELDS, '')
            base.update(row)
            w.writerow(base)


def test_all_fifty_fields_unknowns_aliases_and_readonly(tmp_path):
    path, db = tmp_path/'source.csv', tmp_path/'projection.sqlite'
    rows = [dict(nom_du_lieu='LIEU ALPHA', nom_usuel_du_lieu='ALPHA BLEU', siret='10000000100001',
        entite_juridique_rattachement='PUBLIC NAME', description='TEXT\nMULTILINE',
        coordonnees_geographiques_commune_lieu='UNPARSED ORIGINAL'), dict(siret='1.00000001E13'),
        dict(siret='', nom_du_lieu='LIEU FERME', accessible_au_public='fermé'), dict(siret='00000000000000')]
    write_source(path, rows)
    stats = source.compile_db(db, path)
    assert len(source.FIELDS) == 50
    assert stats == dict(records=4, name_observations=8, nonempty_name_observations=3,
        missing_or_invalid_siret_records=3, distinct_explicit_sirets=1)
    assert source.validate_db(db, path) == stats
    c = source.readonly(db)
    assert c.execute('SELECT description,coordonnees_geographiques_commune_lieu,explicit_siret FROM records WHERE ordinal=0').fetchone() == ('TEXT\nMULTILINE', 'UNPARSED ORIGINAL', '10000000100001')
    assert c.execute('SELECT siret,explicit_siret FROM records WHERE ordinal=1').fetchone() == ('1.00000001E13', None)
    with pytest.raises(sqlite3.OperationalError):
        c.execute('DELETE FROM records')
    c.close()
    with pytest.raises(ValueError):
        source.compile_db(db, path)
    rows[0]['description'] = 'CHANGED LAST RAW COLUMN'
    write_source(path, rows)
    with pytest.raises(ValueError, match='roundtrip'):
        source.validate_db(db, path)


@pytest.mark.parametrize('identifier', ['', 'S10000000100001', '１０００００００１００００１', '100000001-00001', '1.0E13', '00000000000000'])
def test_no_invented_identifier_or_juridical_name_join(identifier):
    row = [''] * len(source.FIELDS)
    row[3], row[4] = 'COMMUNE ALPHA', identifier
    assert source.canonical(0, tuple(row))[-1] is None


def test_schema_and_guard_no_partial_admission(tmp_path):
    path = tmp_path/'bad.csv'
    path.write_text('nom_du_lieu;siret\nLIEU;10000000100001\n')
    with pytest.raises(ValueError):
        list(source.source_rows(path))
    write_source(path, [dict(nom_du_lieu='LIEU ALPHA')])
    def fail():
        raise TimeoutError('guard')
    with pytest.raises(TimeoutError):
        source.compile_db(tmp_path/'bad.sqlite', path, fail)


def test_size_aware_canonical_epoch_binding(tmp_path, monkeypatch):
    path = tmp_path/'full.csv'
    path.write_text('Complete original fixture')
    files = SourceFiles(lambda: None)
    ref = record(path)
    files.file(ref)
    monkeypatch.setattr(source, 'EXPECTED_BYTES', ref['size_bytes'])
    entries = list(files.entries.values())
    normalized = source.input_authority(entries, ref)
    assert normalized[0]['reference'] == {k: ref[k] for k in ('path', 'sha256')}
    assert normalized[0]['fingerprint'][2] == ref['size_bytes']
    for bad in ({**ref, 'size_bytes': ref['size_bytes'] + 1}, {**ref, 'sha256': '0'*64}, {**ref, 'path': str(tmp_path/'wrong.csv')}):
        with pytest.raises(ValueError):
            source.input_authority(entries, bad)
    with pytest.raises(ValueError):
        source.input_authority(entries * 2, ref)


def test_full_http_original_no_redirect_no_truncation(tmp_path, monkeypatch):
    import io
    class Response(io.BytesIO):
        status = 200
        headers = {'Content-Length': '7', 'Content-Type': 'text/csv; charset=utf-8'}
        def geturl(self):
            return source.URL
    observed = []
    def open_response(request, timeout):
        observed.append((request.full_url, request.data, timeout, dict(request.header_items())))
        return Response(b'ALL CSV')
    monkeypatch.setattr(source, 'urlopen', open_response)
    monkeypatch.setattr(source, 'EXPECTED_BYTES', 7)
    out = tmp_path/'original.csv'
    snapshot = source.download(out)
    assert out.read_bytes() == b'ALL CSV'
    assert snapshot['full_response_preserved'] is True
    assert observed[0][:3] == (source.URL, None, 20)
    assert not any(k.lower() in ('authorization', 'cookie') for k in observed[0][3])
    Response.status = 206
    with pytest.raises(ValueError):
        source.download(tmp_path/'partial.csv')
    Response.status = 200
    monkeypatch.setattr(source, 'urlopen', lambda *a, **k: Response(b'SHORT'))
    with pytest.raises(ValueError, match='Complete original response'):
        source.download(tmp_path/'short.csv')
