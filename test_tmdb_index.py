import datetime
import os

import polars as pl
import pytest

from tmdb_index import align_id_col, tmdb_changes, tmdb_export, update_or_append


def test_align_id_col_fills_missing_ids() -> None:
    df = pl.DataFrame(
        {"id": [0, 2], "value": [10, 30]}, schema={"id": pl.UInt32, "value": pl.Int64}
    )
    result = align_id_col(df)
    assert result["id"].to_list() == [0, 1, 2]
    assert result["value"].to_list() == [10, None, 30]


def test_align_id_col_empty_df_returns_empty() -> None:
    df = pl.DataFrame(schema=[("id", pl.UInt32)])
    result = align_id_col(df)
    assert result.is_empty()


def test_update_or_append_merges_and_updates() -> None:
    df1 = pl.DataFrame(
        {"id": [0, 1], "value": [10, 20]}, schema={"id": pl.UInt32, "value": pl.Int64}
    )
    df2 = pl.DataFrame(
        {"id": [1, 2], "value": [200, 30]}, schema={"id": pl.UInt32, "value": pl.Int64}
    )
    result = update_or_append(df1, df2)
    assert result.sort("id")["value"].to_list() == [10, 200, 30]
    assert result.sort("id")["id"].to_list() == [0, 1, 2]


@pytest.mark.skipif(
    not os.environ.get("TMDB_API_KEY"),
    reason="TMDB_API_KEY not set",
)
def test_tmdb_changes() -> None:
    tmdb_api_key = os.environ["TMDB_API_KEY"]
    df = tmdb_changes(
        tmdb_type="movie",
        date=datetime.date(2025, 1, 1),
        tmdb_api_key=tmdb_api_key,
    )
    assert df.columns == ["id", "date", "adult"]
    assert df.shape == (100, 3)


@pytest.mark.skipif(
    not os.environ.get("TMDB_API_KEY"),
    reason="TMDB_API_KEY not set",
)
def test_tmdb_changes_future_date() -> None:
    tmdb_api_key = os.environ["TMDB_API_KEY"]
    df = tmdb_changes(
        tmdb_type="movie",
        date=datetime.date.today() + datetime.timedelta(days=2),
        tmdb_api_key=tmdb_api_key,
    )
    assert df.columns == ["id", "date", "adult"]
    assert df.shape == (0, 3)


def test_tmdb_export_movie() -> None:
    df = tmdb_export(tmdb_type="movie")
    assert df.columns == ["id", "in_export"]
    assert df.shape[0] > 1_000_000


def test_tmdb_export_tv() -> None:
    df = tmdb_export(tmdb_type="tv")
    assert df.columns == ["id", "in_export"]
    assert df.shape[0] > 100_000


def test_tmdb_export_person() -> None:
    df = tmdb_export(tmdb_type="person")
    assert df.columns == ["id", "in_export"]
    assert df.shape[0] > 1_000_000
