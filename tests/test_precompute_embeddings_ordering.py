from scripts.precompute_embeddings import _order_partition_codes


def test_order_partition_codes_sort_by_size() -> None:
    codes = ["02001", "01001", "03001"]
    counts = {"01001": 10, "02001": 100, "03001": 50}

    ordered = _order_partition_codes(
        codes,
        "insee",
        counts,
        sort_by_size=True,
        mega_first=False,
        mega_threshold=100_000,
        min_rows=0,
    )

    assert ordered == ["02001", "03001", "01001"]


def test_order_partition_codes_mega_first_with_filter() -> None:
    codes = ["02001", "01001", "03001", "04001"]
    counts = {"01001": 10, "02001": 120_000, "03001": 80_000, "04001": 200_000}

    ordered = _order_partition_codes(
        codes,
        "insee",
        counts,
        sort_by_size=False,
        mega_first=True,
        mega_threshold=100_000,
        min_rows=50_000,
    )

    assert ordered == ["02001", "04001", "03001"]


def test_order_partition_codes_no_manifest_fallback() -> None:
    codes = ["02001", "01001", "03001"]

    ordered = _order_partition_codes(
        codes,
        "insee",
        {},
        sort_by_size=True,
        mega_first=True,
        mega_threshold=100_000,
        min_rows=10,
    )

    assert ordered == ["01001", "02001", "03001"]


def test_order_partition_codes_cp_always_lexical() -> None:
    codes = ["75015", "01000", "13001"]
    counts = {"75015": 1000}

    ordered = _order_partition_codes(
        codes,
        "cp",
        counts,
        sort_by_size=True,
        mega_first=True,
        mega_threshold=100_000,
        min_rows=500,
    )

    assert ordered == ["01000", "13001", "75015"]
