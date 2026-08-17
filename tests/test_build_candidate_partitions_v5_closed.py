import pandas as pd

from scripts.build_candidate_partitions_v5 import _query_etab_ul


class _Result:
    def __init__(self, owner):
        self.owner = owner

    def df(self):
        return pd.DataFrame()


class _Connection:
    def __init__(self):
        self.query = ""

    def unregister(self, _name):
        raise RuntimeError("not registered")

    def register(self, _name, _frame):
        pass

    def execute(self, query):
        self.query = query
        return _Result(self)


def test_include_closed_has_no_age_cutoff(tmp_path):
    con = _Connection()
    _query_etab_ul(
        con,
        tmp_path / "etab.parquet",
        tmp_path / "ul.parquet",
        ["75056"],
        "codeCommuneEtablissement",
        True,
    )
    assert "dateDebut" not in con.query
    assert "etatAdministratifEtablissement != 'F'" not in con.query


def test_exclude_closed_filters_every_closed_establishment(tmp_path):
    con = _Connection()
    _query_etab_ul(
        con,
        tmp_path / "etab.parquet",
        tmp_path / "ul.parquet",
        ["75056"],
        "codeCommuneEtablissement",
        False,
    )
    assert "etatAdministratifEtablissement != 'F'" in con.query
