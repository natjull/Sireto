"""Leak-resistant V4.1 splits based on connected SIREN components."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import pandas as pd

from .v41_features import normalize_siren, normalize_siret


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _siren(row: pd.Series, siren_column: str, siret_column: str) -> str | None:
    siren = normalize_siren(row.get(siren_column))
    siret = normalize_siret(row.get(siret_column))
    return siren or (siret[:9] if siret else None)


def _hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def assign_connected_siren_splits(
    rows: pd.DataFrame,
    *,
    query_id_column: str = "query_id",
    input_siren_column: str = "input_siren",
    input_siret_column: str = "input_siret",
    target_siren_column: str = "target_siren",
    target_siret_column: str = "target_siret",
    dev_fraction: float = 0.2,
    oof_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign whole input/target-SIREN components to fit/dev and OOF folds.

    Components are shared by all rows connected through either the suspect
    input SIREN or the labelled target SIREN.  Rows without either identifier
    receive a query-specific component and therefore cannot bridge splits.
    """
    if query_id_column not in rows:
        raise ValueError(f"Missing query id column: {query_id_column}")
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be strictly between 0 and 1")
    if oof_folds < 2:
        raise ValueError("oof_folds must be at least 2")
    if rows[query_id_column].astype(str).duplicated().any():
        raise ValueError("Connected-component splitting requires unique query IDs")

    frame = rows.copy()
    union = _UnionFind()
    row_nodes: dict[Any, list[str]] = {}
    for index, row in frame.iterrows():
        input_siren = _siren(row, input_siren_column, input_siret_column)
        target_siren = _siren(row, target_siren_column, target_siret_column)
        nodes = [f"siren:{value}" for value in (input_siren, target_siren) if value]
        if not nodes:
            nodes = [f"query:{row[query_id_column]}"]
        for node in nodes:
            union.find(node)
        for node in nodes[1:]:
            union.union(nodes[0], node)
        row_nodes[index] = nodes

    members: dict[str, set[str]] = defaultdict(set)
    for node in union.parent:
        members[union.find(node)].add(node)
    component_ids = {
        root: hashlib.sha256("|".join(sorted(nodes)).encode("utf-8")).hexdigest()[:16]
        for root, nodes in members.items()
    }
    row_components = {
        index: component_ids[union.find(nodes[0])] for index, nodes in row_nodes.items()
    }

    component_sizes = pd.Series(row_components).value_counts().to_dict()
    dev_components = {
        component
        for component in component_sizes
        if int(_hash(seed, component)[:16], 16) / float(16**16) < dev_fraction
    }
    # Keep both partitions non-empty on small fixtures without splitting a component.
    ordered = sorted(component_sizes, key=lambda value: _hash(seed, value))
    if len(ordered) > 1 and not dev_components:
        dev_components.add(ordered[0])
    if len(ordered) > 1 and len(dev_components) == len(ordered):
        dev_components.remove(ordered[-1])

    fold_loads = [0] * oof_folds
    component_folds: dict[str, int] = {}
    for component in sorted(
        component_sizes,
        key=lambda value: (-component_sizes[value], _hash(seed + 1, value)),
    ):
        minimum = min(fold_loads)
        choices = [fold for fold, load in enumerate(fold_loads) if load == minimum]
        selected = choices[
            int(_hash(seed + 2, component)[:8], 16) % len(choices)
        ]
        component_folds[component] = selected
        fold_loads[selected] += int(component_sizes[component])

    frame["siren_component_id"] = frame.index.map(row_components)
    frame["split"] = frame["siren_component_id"].map(
        lambda component: "dev" if component in dev_components else "fit"
    )
    frame["oof_fold"] = frame["siren_component_id"].map(component_folds).astype(int)
    return frame


def validate_connected_siren_split(rows: pd.DataFrame) -> None:
    """Assert that no connected component crosses split or OOF boundaries."""
    required = {"siren_component_id", "split", "oof_fold"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    split_counts = rows.groupby("siren_component_id")["split"].nunique()
    fold_counts = rows.groupby("siren_component_id")["oof_fold"].nunique()
    if int(split_counts.max()) > 1 or int(fold_counts.max()) > 1:
        raise ValueError("A connected SIREN component crosses a data boundary")


__all__ = [
    "assign_connected_siren_splits",
    "validate_connected_siren_split",
]
