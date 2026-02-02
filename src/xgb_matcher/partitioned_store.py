"""
Partitioned candidate store for multi-blocking retrieval.

Loads candidates from partitioned parquet folders:
  - <partitions_dir>/insee (partitioned by insee)
  - <partitions_dir>/cp (partitioned by postcode)
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import pyarrow.dataset as ds

from .blocking import normalize_code, department_from_code

# Maximum number of entries per store cache (INSEE / CP / dept)
_MAX_STORE_CACHE_SIZE = 100

# Columns actually used by features, naming, filtering, and output.
# Excludes unused columns (nom_usage_ul, pseudonyme_ul) to reduce I/O and RAM.
REQUIRED_COLUMNS = [
    "siret", "siren",
    "denomination", "enseigne1", "enseigne2", "enseigne3",
    "etablissementSiege", "is_siege",
    "numeroVoie", "typeVoie", "libelleVoie", "complementAdresse",
    "postcode", "city", "insee",
    "cj_ul", "etat_admin", "last_treatment_date",
    "sigle_ul", "denomination_ul", "denomination_usuelle_ul",
    "nom_ul", "prenom_usuel_ul",
    "pm_dirigeant_names",
]


class PartitionedCandidateStore:
    """Lazy loader for candidates by INSEE / CP / department."""

    def __init__(self, partitions_dir: Path):
        self.partitions_dir = Path(partitions_dir)
        self._dataset_insee = ds.dataset(self.partitions_dir / "insee", format="parquet", partitioning="hive")
        self._dataset_cp = ds.dataset(self.partitions_dir / "cp", format="parquet", partitioning="hive")
        self._cache_insee: OrderedDict[str, List[dict]] = OrderedDict()
        self._cache_cp: OrderedDict[str, List[dict]] = OrderedDict()
        self._cache_dept: OrderedDict[str, List[dict]] = OrderedDict()

        # Determine which columns are actually present in the datasets
        # (some columns like pm_dirigeant_names may be absent in older partitions)
        self._columns_insee = self._available_columns(self._dataset_insee)
        self._columns_cp = self._available_columns(self._dataset_cp)

    @staticmethod
    def _available_columns(dataset: ds.Dataset) -> List[str] | None:
        """Return REQUIRED_COLUMNS filtered to those present in the dataset schema."""
        try:
            schema_names = set(dataset.schema.names)
            cols = [c for c in REQUIRED_COLUMNS if c in schema_names]
            return cols if cols else None
        except Exception:
            return None

    @staticmethod
    def _coerce_candidate_types(rows: List[dict]) -> List[dict]:
        for r in rows:
            if "siret" in r:
                r["siret"] = str(r.get("siret") or "")
            if "siren" in r:
                r["siren"] = str(r.get("siren") or "")
        return rows

    @staticmethod
    def _evict_lru(cache: OrderedDict, max_size: int) -> None:
        """Evict oldest entries until cache is within max_size."""
        while len(cache) > max_size:
            cache.popitem(last=False)

    def load_by_insee(self, insee: Optional[str]) -> List[dict]:
        code = normalize_code(insee)
        if not code:
            return []
        if code in self._cache_insee:
            self._cache_insee.move_to_end(code)
            return self._cache_insee[code]
        try:
            code_int = int(code)
            table = self._dataset_insee.to_table(
                filter=ds.field("insee") == code_int,
                columns=self._columns_insee,
            )
        except Exception:
            return []
        rows = self._coerce_candidate_types(table.to_pylist())
        self._cache_insee[code] = rows
        self._evict_lru(self._cache_insee, _MAX_STORE_CACHE_SIZE)
        return rows

    def load_by_postcode(self, postcode: Optional[str]) -> List[dict]:
        code = normalize_code(postcode)
        if not code:
            return []
        if code in self._cache_cp:
            self._cache_cp.move_to_end(code)
            return self._cache_cp[code]
        try:
            code_int = int(code)
            table = self._dataset_cp.to_table(
                filter=ds.field("postcode") == code_int,
                columns=self._columns_cp,
            )
        except Exception:
            return []
        rows = self._coerce_candidate_types(table.to_pylist())
        self._cache_cp[code] = rows
        self._evict_lru(self._cache_cp, _MAX_STORE_CACHE_SIZE)
        return rows

    def load_by_department(self, insee: Optional[str], postcode: Optional[str]) -> List[dict]:
        dept = department_from_code(insee, postcode)
        if not dept:
            return []
        if dept in self._cache_dept:
            self._cache_dept.move_to_end(dept)
            return self._cache_dept[dept]
        try:
            dept_int = int(dept)
        except Exception:
            return []
        if len(dept) >= 3:
            start = dept_int * 100
            end = start + 99
        else:
            start = dept_int * 1000
            end = start + 999
        try:
            filt = (ds.field("postcode") >= start) & (ds.field("postcode") <= end)
            table = self._dataset_cp.to_table(
                filter=filt,
                columns=self._columns_cp,
            )
        except Exception:
            return []
        rows = self._coerce_candidate_types(table.to_pylist())
        self._cache_dept[dept] = rows
        self._evict_lru(self._cache_dept, _MAX_STORE_CACHE_SIZE)
        return rows
