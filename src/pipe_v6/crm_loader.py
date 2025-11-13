"""CRM ingestion utilities (task 1.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import re

import pandas as pd

DEFAULT_COLUMN_MAP: Mapping[str, str] = {
    "crm_id": "N° Service",
    "name": "Client final",
    "street": "Adresse",
    "city": "Commune",
    "postcode": "Code Postal",
    "insee": "Code INSEE",
}

OUTPUT_COLUMNS = [
    "crm_id",
    "crm_name",
    "street_number",
    "street_name",
    "postcode",
    "city",
    "insee_code",
]

_ADDRESS_PATTERN = re.compile(
    r"^\s*(?P<number>\d+(?:\s*(?:BIS|TER|QUATER|B|T))?)\s+(?P<street>.+)$",
    re.IGNORECASE,
)


def load_crm(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
    *,
    encoding: str = "utf-8",
    sep: str = ";",
    split_address: bool = True,
) -> pd.DataFrame:
    """Return a normalized CRM dataframe ready for downstream tasks."""

    active_map = dict(DEFAULT_COLUMN_MAP)
    if column_map:
        active_map.update(column_map)

    _validate_column_map(active_map)

    source_columns = {src_name for src_name in active_map.values()}
    csv_path = Path(path)

    df = pd.read_csv(
        csv_path,
        sep=sep,
        encoding=encoding,
        dtype=str,
        usecols=sorted(source_columns),
        keep_default_na=True,
    )

    rename_map = {
        active_map["crm_id"]: "crm_id",
        active_map["name"]: "crm_name",
        active_map["street"]: "street_raw",
        active_map["city"]: "city",
        active_map["postcode"]: "postcode",
        active_map["insee"]: "insee_code",
    }
    missing_sources = [src for src in rename_map if src not in df.columns]
    if missing_sources:
        raise KeyError(
            f"Missing expected CRM columns in {csv_path}: {', '.join(missing_sources)}"
        )

    df = df.rename(columns=rename_map)

    df["crm_id"] = df["crm_id"].map(_normalize_text)
    if df["crm_id"].isna().any():  # crm_id is mandatory
        raise ValueError("CRM data contains empty 'crm_id' values after normalization")

    df["crm_name"] = df["crm_name"].map(_normalize_text)
    df["street_raw"] = df["street_raw"].map(_normalize_text)
    df["city"] = df["city"].map(lambda x: _normalize_text(x, upper=True))
    df["postcode"] = df["postcode"].map(_normalize_postcode)
    df["insee_code"] = df["insee_code"].map(_normalize_insee)

    if split_address:
        street_numbers: list[str | pd.NA] = []
        street_names: list[str | pd.NA] = []
        for value in df["street_raw"].tolist():
            number, name = _split_address(value)
            street_numbers.append(number)
            street_names.append(name)
        df["street_number"] = street_numbers
        df["street_name"] = street_names
    else:
        df["street_number"] = pd.NA
        df["street_name"] = df["street_raw"]

    df = df.drop(columns=["street_raw"])

    return df[OUTPUT_COLUMNS].copy()


def _validate_column_map(column_map: Mapping[str, str]) -> None:
    required = {"crm_id", "name", "street", "city", "postcode", "insee"}
    missing = required - set(column_map)
    if missing:
        raise KeyError(f"column_map missing required keys: {', '.join(sorted(missing))}")


def _normalize_text(value: object, *, upper: bool = False) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    text = " ".join(text.split())
    if upper:
        text = text.upper()
    return text


def _normalize_postcode(value: object) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value)
    digits = re.sub(r"\D", "", text)
    if not digits:
        return pd.NA
    return digits.zfill(5)


def _normalize_insee(value: object) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if not text:
        return pd.NA
    if text.isdigit():
        return text.zfill(5)
    return text


def _split_address(value: object) -> tuple[str | pd.NA, str | pd.NA]:
    if value is None or pd.isna(value):
        return pd.NA, pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA, pd.NA

    match = _ADDRESS_PATTERN.match(text)
    if match:
        number = match.group("number").replace(" ", "").upper()
        street = match.group("street").strip()
        street = " ".join(street.split()) if street else street
        return number, (street or pd.NA)
    return pd.NA, text


__all__ = ["DEFAULT_COLUMN_MAP", "load_crm"]
