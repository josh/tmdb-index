import os
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from tmdb_index import (
    align_id_col,
    insert_tmdb_latest_changes,
    tmdb_changes,
    tmdb_changes_backfill_date_range,
    tmdb_export,
    tmdb_external_ids,
    update_or_append,
)


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
        date=date(2025, 1, 1),
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
        date=date.today() + timedelta(days=2),
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


@pytest.mark.skipif(
    not os.environ.get("TMDB_API_KEY"),
    reason="TMDB_API_KEY not set",
)
def test_insert_tmdb_latest_changes() -> None:
    tmdb_api_key = os.environ["TMDB_API_KEY"]

    initial_date = date.today() + timedelta(days=-2)
    df = tmdb_changes(
        tmdb_type="movie",
        date=initial_date,
        tmdb_api_key=tmdb_api_key,
    )
    df2 = insert_tmdb_latest_changes(
        df,
        tmdb_type="movie",
        tmdb_api_key=tmdb_api_key,
    )
    assert df2.columns == ["id", "date", "adult"]
    assert len(df2) > len(df)


def test_tmdb_changes_backfill_date_range() -> None:
    d = date.today()
    df = pl.DataFrame({"date": [d]})
    dates = tmdb_changes_backfill_date_range(df)
    assert dates == [
        d - timedelta(days=1),
        d,
    ], dates

    d = date.today() + timedelta(days=-1)
    df = pl.DataFrame({"date": [d]})
    dates = tmdb_changes_backfill_date_range(df)
    assert dates == [
        d - timedelta(days=1),
        d,
        d + timedelta(days=1),
    ], dates

    d = date.today() + timedelta(days=-2)
    df = pl.DataFrame({"date": [d]})
    dates = tmdb_changes_backfill_date_range(df)
    assert dates == [
        d - timedelta(days=1),
        d,
        d + timedelta(days=1),
        d + timedelta(days=2),
    ], dates

    d = date.today() + timedelta(days=-3)
    df = pl.DataFrame({"date": [d]})
    dates = tmdb_changes_backfill_date_range(df)
    assert dates == [
        d - timedelta(days=1),
        d,
        d + timedelta(days=1),
        d + timedelta(days=2),
        d + timedelta(days=3),
    ], dates


_FEW_MINUTES_AGO: datetime = datetime.now(UTC) - timedelta(minutes=5)


@pytest.mark.skipif(
    not os.environ.get("TMDB_API_KEY"),
    reason="TMDB_API_KEY not set",
)
def test_tmdb_external_ids() -> None:
    tmdb_api_key = os.environ["TMDB_API_KEY"]
    result = tmdb_external_ids(
        tmdb_type="movie",
        tmdb_id=603,
        tmdb_api_key=tmdb_api_key,
    )
    assert result["id"] == 603
    assert result["success"] is True
    assert result["retrieved_at"] >= _FEW_MINUTES_AGO
    assert result["imdb_numeric_id"] == 133093
    assert result["tvdb_id"] is None
    assert result["wikidata_numeric_id"] == 83495
