import os
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from tmdb_index import (
    TMDB_CHANGES_EPOCH,
    align_id_col,
    change_summary,
    export_available,
    export_date,
    fetch_jsonl_gz,
    insert_tmdb_external_ids,
    insert_tmdb_latest_changes,
    process,
    tmdb_changes,
    tmdb_changes_backfill_date_range,
    tmdb_export,
    tmdb_external_ids,
    update_or_append,
    update_tmdb_export_flag,
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


def test_align_id_col_missing_column() -> None:
    df = pl.DataFrame({"id2": [1, 2, 3]}, schema={"id2": pl.UInt32})
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        align_id_col(df)


def test_update_or_append_merges_and_updates() -> None:
    df1 = pl.DataFrame(
        {"id": [0, 1], "value": [10, 20]},
        schema={"id": pl.UInt32, "value": pl.Int64},
    )
    df2 = pl.DataFrame(
        {"id": [1, 2], "value": [200, 30]},
        schema={"id": pl.UInt32, "value": pl.Int64},
    )
    result = update_or_append(df1, df2)
    assert result.columns == ["id", "value"]
    assert result.sort("id")["value"].to_list() == [10, 200, 30]
    assert result.sort("id")["id"].to_list() == [0, 1, 2]


def test_update_or_append_mismatched_columns() -> None:
    df1 = pl.DataFrame(
        {"id": [1, 2], "a": [10, 20], "b": [100, 200]},
        schema={"id": pl.UInt32, "a": pl.Int64, "b": pl.Int64},
    )
    df2 = pl.DataFrame(
        {"id": [2, 3], "b": [222, 333], "c": [42, 43]},
        schema={"id": pl.UInt32, "b": pl.Int64, "c": pl.Int64},
    )
    result = update_or_append(df1, df2)
    assert result.columns == ["id", "a", "b", "c"]
    result = result.sort("id")
    assert result["id"].to_list() == [1, 2, 3]
    assert result.row(0) == (1, 10, 100, None)
    assert result.row(1) == (2, 20, 222, 42)
    assert result.row(2) == (3, None, 333, 43)


def test_change_summary_reports_diffs() -> None:
    df1 = pl.DataFrame(
        {"id": [0, 1], "value": [10, 20]},
        schema={"id": pl.UInt32, "value": pl.Int64},
    )
    df2 = pl.DataFrame(
        {"id": [1, 2], "value": [200, 30]},
        schema={"id": pl.UInt32, "value": pl.Int64},
    )
    summary = change_summary(df1, df2)
    assert summary == "+1 -1 ~1"


def test_change_summary_identical_rows() -> None:
    df1 = pl.DataFrame(
        {"id": [0], "value": [10]},
        schema={"id": pl.UInt32, "value": pl.Int64},
    )
    df2 = pl.DataFrame(
        {"id": [0], "value": [10]},
        schema={"id": pl.UInt32, "value": pl.Int64},
    )
    summary = change_summary(df1, df2)
    assert summary == "+0 -0 ~0"


def test_fetch_jsonl_gz_gzip_response() -> None:
    d = date.today() - timedelta(days=3)
    url = (
        f"http://files.tmdb.org/p/exports/keyword_ids_{d.strftime('%m_%d_%Y')}.json.gz"
    )
    gen = fetch_jsonl_gz(url)
    result = [next(gen) for _ in range(100)]
    assert hasattr(gen, "close")
    gen.close()

    assert len(result) == 100
    assert set(result[0]) == {"id", "name"}


def test_export_date_before_8am_returns_previous_day() -> None:
    now = datetime(2024, 1, 2, 7, 59, tzinfo=UTC)
    assert export_date(now) == date(2024, 1, 1)


def test_export_date_after_8am_returns_current_day() -> None:
    now = datetime(2024, 1, 2, 8, 0, tzinfo=UTC)
    assert export_date(now) == date(2024, 1, 2)


def test_export_available_recent_date_true() -> None:
    recent = date.today() - timedelta(days=3)
    assert export_available("movie", recent) is True


def test_export_available_old_date_false() -> None:
    old = date.today() - timedelta(days=365)
    assert export_available("movie", old) is False


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
        days_limit=7,
    )
    assert df2.columns == ["id", "date", "adult"]
    assert len(df2) > len(df)


def test_tmdb_changes_backfill_date_range() -> None:
    d = date.today()
    df = pl.DataFrame({"date": [d]})
    dates = tmdb_changes_backfill_date_range(df, tmdb_type="movie")
    assert dates == [
        d - timedelta(days=1),
        d,
    ], dates

    d = date.today() + timedelta(days=-1)
    df = pl.DataFrame({"date": [d]})
    dates = tmdb_changes_backfill_date_range(df, tmdb_type="movie")
    assert dates == [
        d - timedelta(days=1),
        d,
        d + timedelta(days=1),
    ], dates

    d = date.today() + timedelta(days=-2)
    df = pl.DataFrame({"date": [d]})
    dates = tmdb_changes_backfill_date_range(df, tmdb_type="movie")
    assert dates == [
        d - timedelta(days=1),
        d,
        d + timedelta(days=1),
        d + timedelta(days=2),
    ], dates

    d = date.today() + timedelta(days=-3)
    df = pl.DataFrame({"date": [d]})
    dates = tmdb_changes_backfill_date_range(df, tmdb_type="movie")
    assert dates == [
        d - timedelta(days=1),
        d,
        d + timedelta(days=1),
        d + timedelta(days=2),
        d + timedelta(days=3),
    ], dates


def test_tmdb_changes_backfill_date_range_empty_df() -> None:
    df = pl.DataFrame({"date": []}, schema={"date": pl.Date})
    dates = tmdb_changes_backfill_date_range(df, tmdb_type="movie")
    expected_start = date(2012, 10, 5)
    expected_end = date.today()
    expected_days = (expected_end - expected_start).days + 1
    assert len(dates) == expected_days
    assert dates[0] == expected_start
    assert dates[-1] == expected_end


def test_tmdb_changes_backfill_date_range_missing_date_column() -> None:
    df = pl.DataFrame({"missing_date": [1, 2, 3]}, schema={"missing_date": pl.Int32})
    dates = tmdb_changes_backfill_date_range(df, tmdb_type="movie")
    expected_start = TMDB_CHANGES_EPOCH["movie"]
    expected_end = date.today()
    expected_days = (expected_end - expected_start).days + 1
    assert len(dates) == expected_days
    assert dates[0] == expected_start
    assert dates[-1] == expected_end


_FEW_MINUTES_AGO: datetime = datetime.now(UTC) - timedelta(minutes=5)


@pytest.mark.skipif(
    not os.environ.get("TMDB_API_KEY"),
    reason="TMDB_API_KEY not set",
)
def test_insert_tmdb_external_ids() -> None:
    tmdb_api_key = os.environ["TMDB_API_KEY"]
    df = pl.DataFrame(
        [{"id": 603, "date": date.today(), "retrieved_at": None}],
        schema={
            "id": pl.UInt32,
            "date": pl.Date,
            "retrieved_at": pl.Datetime(time_unit="ns"),
        },
    )
    result = insert_tmdb_external_ids(
        df,
        tmdb_type="movie",
        tmdb_api_key=tmdb_api_key,
        backfill_limit=1,
        refresh_limit=0,
    )
    row = result.filter(pl.col("id") == 603).row(0, named=True)
    assert row["id"] == 603
    assert row["success"] is True
    assert row["retrieved_at"] is not None
    assert row["imdb_numeric_id"] == 133093
    assert row["tvdb_id"] is None
    assert row["wikidata_numeric_id"] == 83495


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


@pytest.mark.skipif(
    not os.environ.get("TMDB_API_KEY"),
    reason="TMDB_API_KEY not set",
)
def test_process() -> None:
    tmdb_api_key = os.environ["TMDB_API_KEY"]
    df = process(
        df=None,
        tmdb_type="movie",
        tmdb_api_key=tmdb_api_key,
        backfill_limit=0,
        refresh_limit=0,
        changes_days_limit=3,
    )

    assert df.columns == [
        "id",
        "date",
        "adult",
        "in_export",
        "success",
        "retrieved_at",
        "imdb_numeric_id",
        "tvdb_id",
        "wikidata_numeric_id",
    ]
    assert df.height > 0

    start_date = TMDB_CHANGES_EPOCH["movie"]
    df_dates: list[date] = df["date"].unique().sort().to_list()
    assert df_dates == [
        None,
        start_date,
        start_date + timedelta(days=1),
        start_date + timedelta(days=2),
    ]

    row = df.row(index=522, named=True)
    assert row["id"] == 522
    assert row["date"] == date(2012, 10, 7)
    assert row["adult"] is False
    assert row["in_export"] is True


@pytest.mark.skipif(
    not os.environ.get("TMDB_API_KEY"),
    reason="TMDB_API_KEY not set",
)
def test_process_with_backfill() -> None:
    tmdb_api_key = os.environ["TMDB_API_KEY"]
    df = process(
        df=None,
        tmdb_type="movie",
        tmdb_api_key=tmdb_api_key,
        backfill_limit=12,
        refresh_limit=0,
        changes_days_limit=3,
    )
    assert df.columns == [
        "id",
        "date",
        "adult",
        "in_export",
        "success",
        "retrieved_at",
        "imdb_numeric_id",
        "tvdb_id",
        "wikidata_numeric_id",
    ]
    df2 = df.filter(pl.col("retrieved_at").is_not_null())
    assert df2.height == 12


def test_update_tmdb_export_flag_append() -> None:
    df = pl.DataFrame(
        {"id": [1, 2, 9999999], "value": [10, 20, 30]},
        schema={"id": pl.UInt32, "value": pl.Int64},
    )
    assert df.columns == ["id", "value"]
    df2 = update_tmdb_export_flag(df, tmdb_type="movie")
    assert df2.columns == ["id", "value", "in_export"]
    assert df2.shape == (3, 3)
    assert df2.row(index=2) == (9999999, 30, False)


def test_update_tmdb_export_flag_replace() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 9999999],
            "in_export": [True, True, True],
            "value": [10, 20, 30],
        },
        schema={"id": pl.UInt32, "in_export": pl.Boolean, "value": pl.Int64},
    )
    assert df.columns == ["id", "in_export", "value"]
    df2 = update_tmdb_export_flag(df, tmdb_type="movie")
    assert df2.columns == ["id", "in_export", "value"]
    assert df2.shape == (3, 3)
    assert df2.row(index=2) == (9999999, False, 30)


def test_update_tmdb_export_flag_empty() -> None:
    df = pl.DataFrame(schema={"id": pl.UInt32, "value": pl.Int64})
    assert df.columns == ["id", "value"]
    df2 = update_tmdb_export_flag(df, tmdb_type="movie")
    assert df2.columns == ["id", "value", "in_export"]
    assert df2.shape == (0, 3)
