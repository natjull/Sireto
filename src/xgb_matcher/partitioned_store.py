"""
Partitioned candidate store for multi-blocking retrieval.

Loads candidates from partitioned parquet folders:
  - <partitions_dir>/insee (partitioned by insee)
  - <partitions_dir>/cp (partitioned by postcode)
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .blocking import normalize_code, department_from_code

# Maximum number of entries per store cache (INSEE / CP / dept)
_MAX_STORE_CACHE_SIZE = 5

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
        insee_part = ds.partitioning(pa.schema([("insee", pa.string())]), flavor="hive")
        cp_part = ds.partitioning(pa.schema([("postcode", pa.string())]), flavor="hive")
        self._dataset_insee = ds.dataset(self.partitions_dir / "insee", format="parquet", partitioning=insee_part)
        self._dataset_cp = ds.dataset(self.partitions_dir / "cp", format="parquet", partitioning=cp_part)
        self._cache_insee: OrderedDict[str, List[dict]] = OrderedDict()
        self._cache_cp: OrderedDict[str, List[dict]] = OrderedDict()
        self._cache_cp_insee: OrderedDict[str, List[dict]] = OrderedDict()
        self._cache_dept: OrderedDict[str, List[dict]] = OrderedDict()
        self._cache_insee_count: OrderedDict[str, int] = OrderedDict()
        self._manifest_insee_counts = self._load_insee_manifest_counts()

        # Determine which columns are actually present in the datasets
        # (some columns like pm_dirigeant_names may be absent in older partitions)
        self._columns_insee = self._available_columns(self._dataset_insee)
        self._columns_cp = self._available_columns(self._dataset_cp)

    def _load_insee_manifest_counts(self) -> Dict[str, int]:
        """Load optional V7 manifest counts for O(1) INSEE row lookup."""
        manifest_path = self.partitions_dir / "manifest" / "insee_counts.parquet"
        if not manifest_path.exists():
            return {}
        try:
            table = pq.read_table(manifest_path, columns=["insee", "row_count"])
        except Exception:
            return {}
        counts: Dict[str, int] = {}
        for row in table.to_pylist():
            code = normalize_code(row.get("insee"))
            if not code:
                continue
            try:
                counts[code] = int(row.get("row_count") or 0)
            except Exception:
                continue
        return counts

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

    def _count_insee_rows(self, insee: Optional[str]) -> int:
        code = normalize_code(insee)
        if not code:
            return 0
        if code in self._cache_insee_count:
            self._cache_insee_count.move_to_end(code)
            return self._cache_insee_count[code]
        if code in self._manifest_insee_counts:
            count = int(self._manifest_insee_counts[code])
            self._cache_insee_count[code] = count
            self._evict_lru(self._cache_insee_count, _MAX_STORE_CACHE_SIZE)
            return count
        try:
            count = self._dataset_insee.count_rows(filter=ds.field("insee") == code)
        except Exception:
            return 0
        self._cache_insee_count[code] = int(count)
        self._evict_lru(self._cache_insee_count, _MAX_STORE_CACHE_SIZE)
        return int(count)

    def load_by_insee(self, insee: Optional[str]) -> List[dict]:
        code = normalize_code(insee)
        if not code:
            return []
        if code in self._cache_insee:
            self._cache_insee.move_to_end(code)
            return self._cache_insee[code]
        try:
            table = self._dataset_insee.to_table(
                filter=ds.field("insee") == code,
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
            table = self._dataset_cp.to_table(
                filter=ds.field("postcode") == code,
                columns=self._columns_cp,
            )
        except Exception:
            return []
        rows = self._coerce_candidate_types(table.to_pylist())
        self._cache_cp[code] = rows
        self._evict_lru(self._cache_cp, _MAX_STORE_CACHE_SIZE)
        return rows

    def load_by_postcode_filtered_insee(self, postcode: Optional[str], insee: Optional[str]) -> List[dict]:
        code_cp = normalize_code(postcode)
        code_insee = normalize_code(insee)
        if not code_cp or not code_insee:
            return []
        cache_key = f"{code_cp}|{code_insee}"
        if cache_key in self._cache_cp_insee:
            self._cache_cp_insee.move_to_end(cache_key)
            return self._cache_cp_insee[cache_key]
        try:
            filt = (ds.field("postcode") == code_cp) & (ds.field("insee") == code_insee)
            table = self._dataset_cp.to_table(
                filter=filt,
                columns=self._columns_cp,
            )
        except Exception:
            return []
        rows = self._coerce_candidate_types(table.to_pylist())
        self._cache_cp_insee[cache_key] = rows
        self._evict_lru(self._cache_cp_insee, _MAX_STORE_CACHE_SIZE)
        return rows

    def load_by_insee_then_postcode(
        self,
        insee: Optional[str],
        postcode: Optional[str],
        *,
        mega_insee_max_rows: int = 100_000,
        mega_insee_policy: str = "cp_filter_insee",
    ) -> List[dict]:
        code_insee = normalize_code(insee)
        code_cp = normalize_code(postcode)
        if code_insee:
            if mega_insee_max_rows:
                count = self._count_insee_rows(code_insee)
                if count > mega_insee_max_rows:
                    if mega_insee_policy == "cp_filter_insee" and code_cp:
                        return self.load_by_postcode_filtered_insee(code_cp, code_insee)
                    if mega_insee_policy == "full_insee":
                        return self.load_by_insee(code_insee)
            rows = self.load_by_insee(code_insee)
            if rows:
                return rows
        if code_cp:
            return self.load_by_postcode(code_cp)
        return []

    def load_by_department(self, insee: Optional[str], postcode: Optional[str]) -> List[dict]:
        raise RuntimeError("Department fallback is disabled under SSOT.")
