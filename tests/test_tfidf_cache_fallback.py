from __future__ import annotations

import pickle

from src.xgb_matcher.tfidf_cache import TfidfPersistentCache


def test_persistent_cache_reads_legacy_fallback(tmp_path) -> None:
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    expected = ("name", "matrix", [], None, None, None, None)
    with (legacy_dir / "75056_.pkl").open("wb") as handle:
        pickle.dump(expected, handle)

    cache = TfidfPersistentCache(
        "artifact",
        cache_dir=tmp_path,
        fallback_config_hashes=["legacy"],
    )

    assert cache.get("75056_") == expected
    assert cache.stats() == {"hits": 1, "misses": 0}


def test_persistent_cache_writes_only_primary_namespace(tmp_path) -> None:
    cache = TfidfPersistentCache(
        "artifact",
        cache_dir=tmp_path,
        fallback_config_hashes=["legacy"],
    )
    value = ("name", "matrix", [], None, None, None, None)

    cache.put("75056_", value)

    assert (tmp_path / "artifact" / "75056_.pkl").exists()
    assert not (tmp_path / "legacy" / "75056_.pkl").exists()
