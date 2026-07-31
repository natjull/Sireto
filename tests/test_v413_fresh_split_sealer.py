from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/seal_v413_fresh_splits.py"
SPEC = importlib.util.spec_from_file_location("seal_v413_fresh_splits", SCRIPT)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def row(
    query_id: str,
    *,
    group: str | None = None,
    sirens: list[str] | None = None,
) -> dict:
    return {
        "schema_version": subject.SCHEMA,
        "query_id": query_id,
        "source_group_id": group,
        "authoritative_sirens": sirens or [],
    }


def test_union_find_closes_transitive_group_and_all_siren_edges() -> None:
    rows = [
        row("a" * 64, group="g1", sirens=["111111111"]),
        row("b" * 64, group="g1", sirens=["222222222"]),
        row("c" * 64, group="g2", sirens=["222222222", "333333333"]),
        row("d" * 64, group="g2"),
        row("e" * 64),
    ]
    result = subject.assign_components(rows)
    linked = {result[key]["component_sha256"] for key in ("a" * 64, "b" * 64, "c" * 64, "d" * 64)}
    assert len(linked) == 1
    assert result["e" * 64]["component_sha256"] not in linked
    assert len({result[key]["split"] for key in ("a" * 64, "b" * 64, "c" * 64, "d" * 64)}) == 1


def test_frozen_vector_and_thresholds_match_contract() -> None:
    result = subject.assign_components(
        [row("a" * 64, group="same"), row("b" * 64, group="same")]
    )
    assert result["a" * 64]["split_uint64"] == 3000513557974646524
    assert result["a" * 64]["split"] == "fit"
    assert subject.FIT_UPPER == 12912720851596686131
    assert subject.DEV_UPPER == 15679732462653118873


def test_order_is_irrelevant_and_manifests_cover_each_query_once() -> None:
    rows = [row("c" * 64), row("a" * 64), row("b" * 64)]
    forward = subject.build_manifests(rows)
    reverse = subject.build_manifests(reversed(rows))
    assert forward == reverse
    flattened = [
        item["query_id"]
        for manifest in forward.values()
        for item in manifest["assignments"]
    ]
    assert sorted(flattened) == sorted(item["query_id"] for item in rows)
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [row("z" * 64)],
        [row("a" * 64), row("a" * 64)],
        [row("a" * 64, group="")],
        [row("a" * 64, sirens=["123"])],
        [row("a" * 64, sirens=["222222222", "111111111"])],
    ],
)
def test_invalid_or_ambiguous_inputs_fail_closed(bad: list[dict]) -> None:
    with pytest.raises(subject.SplitStopped):
        subject.assign_components(bad)


def test_sealing_is_exclusive_and_physically_separate(tmp_path: Path) -> None:
    rows = [row("a" * 64), row("b" * 64), row("c" * 64)]
    hashes = subject.seal_manifests(rows, tmp_path)
    assert set(hashes) == {"fit", "dev", "test"}
    for split in hashes:
        path = tmp_path / split / "split_manifest.json"
        assert path.is_file()
        assert json.loads(path.read_text())["split"] == split
    with pytest.raises(FileExistsError):
        subject.seal_manifests(rows, tmp_path)
