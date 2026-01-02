"""
Partitioned candidate store for multi-blocking retrieval.

Loads candidates from partitioned parquet folders:
  - <partitions_dir>/insee (partitioned by insee)
  - <partitions_dir>/cp (partitioned by postcode)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.dataset as ds

from .blocking import normalize_code, department_from_code


class PartitionedCandidateStore:
    """Lazy loader for candidates by INSEE / CP / department."""

    def __init__(self, partitions_dir: Path):
        self.partitions_dir = Path(partitions_dir)
        self._dataset_insee = ds.dataset(self.partitions_dir / "insee", format="parquet", partitioning="hive")
        self._dataset_cp = ds.dataset(self.partitions_dir / "cp", format="parquet", partitioning="hive")
        self._cache_insee: Dict[str, List[dict]] = {}
        self._cache_cp: Dict[str, List[dict]] = {}
        self._cache_dept: Dict[str, List[dict]] = {}

    @staticmethod
    def _coerce_candidate_types(rows: List[dict]) -> List[dict]:
        for r in rows:
            if "siret" in r:
                r["siret"] = str(r.get("siret") or "")
            if "siren" in r:
                r["siren"] = str(r.get("siren") or "")
        return rows

    def load_by_insee(self, insee: Optional[str]) -> List[dict]:
        code = normalize_code(insee)
        if not code:
            return []
        if code in self._cache_insee:
            return self._cache_insee[code]
        try:
            code_int = int(code)
            table = self._dataset_insee.to_table(filter=ds.field("insee") == code_int)
        except Exception:
            return []
        rows = self._coerce_candidate_types(table.to_pylist())
        self._cache_insee[code] = rows
        return rows

    def load_by_postcode(self, postcode: Optional[str]) -> List[dict]:
        code = normalize_code(postcode)
        if not code:
            return []
        if code in self._cache_cp:
            return self._cache_cp[code]
        try:
            code_int = int(code)
            table = self._dataset_cp.to_table(filter=ds.field("postcode") == code_int)
        except Exception:
            return []
        rows = self._coerce_candidate_types(table.to_pylist())
        self._cache_cp[code] = rows
        return rows

    def load_by_department(self, insee: Optional[str], postcode: Optional[str]) -> List[dict]:
        dept = department_from_code(insee, postcode)
        if not dept:
            return []
        if dept in self._cache_dept:
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
            table = self._dataset_cp.to_table(filter=filt)
        except Exception:
            return []
        rows = self._coerce_candidate_types(table.to_pylist())
        self._cache_dept[dept] = rows
        return rows

