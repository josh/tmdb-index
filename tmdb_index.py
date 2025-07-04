import gzip
import json
import logging
import re
import urllib.request
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from itertools import islice
from typing import Any, Literal

import click
import polars as pl

logger = logging.getLogger("tmdb-index")

TMDB_TYPE = Literal["movie", "tv", "person"]
_TMDB_TYPES: set[TMDB_TYPE] = {"movie", "tv", "person"}

_IMDB_ID_PATTERN: dict[TMDB_TYPE, str] = {
    "movie": r"tt(\d+)",
    "tv": r"tt(\d+)",
    "person": r"nm(\d+)",
}

_EXTERNAL_IDS_RESPONSE_SCHEMA = pl.Schema(
    [
        ("success", pl.Boolean),
        ("id", pl.UInt32),
        ("retrieved_at", pl.Datetime(time_unit="ns")),
        ("imdb_numeric_id", pl.UInt32),
        ("tvdb_id", pl.UInt32),
        ("wikidata_numeric_id", pl.UInt32),
    ]
)


def align_id_col(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    max_id: int = df.select(pl.col("id").max()).item()
    id_df = (
        pl.int_range(end=max_id + 1, dtype=pl.UInt32, eager=True)
        .cast(pl.UInt32)
        .to_frame("id")
    )
    return id_df.join(df, on="id", how="left", coalesce=True).select(df.columns)


def update_or_append(df: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
    other_cols = list(other.columns)
    other_cols.remove("id")
    other = other.join(df.drop(other_cols), on="id", how="left", coalesce=True).select(
        df.columns
    )
    return pl.concat([df, other]).unique(subset="id", keep="last", maintain_order=True)


def change_summary(df_old: pl.DataFrame, df_new: pl.DataFrame) -> str:
    added = df_new.join(df_old.select("id"), on="id", how="anti").height
    removed = df_old.join(df_new.select("id"), on="id", how="anti").height
    common = df_old.join(df_new.select("id"), on="id", how="semi").height
    unchanged = df_old.join(df_new, on=df_old.columns, how="inner").height
    updated = common - unchanged
    return f"+{added} -{removed} ~{updated}"


_TMDB_CHANGES_SCHEMA = pl.Schema(
    [
        ("id", pl.UInt32),
        ("adult", pl.Boolean),
    ]
)

_TMDB_CHANGES_EPOCH: dict[TMDB_TYPE, date] = {
    "movie": date(2012, 10, 5),
    "tv": date(2012, 12, 31),
    "person": date(2012, 10, 5),
}


def tmdb_changes(
    tmdb_type: TMDB_TYPE,
    date: date,
    tmdb_api_key: str,
) -> pl.DataFrame:
    assert date >= _TMDB_CHANGES_EPOCH[tmdb_type], (
        f"Date must be after {_TMDB_CHANGES_EPOCH[tmdb_type]}"
    )

    start_date = date.strftime("%Y-%m-%d")
    end_date = (date + timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"https://api.themoviedb.org/3/{tmdb_type}/changes?start_date={start_date}&end_date={end_date}&api_key={tmdb_api_key}"

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.load(response)

    df = (
        pl.from_dicts(data["results"], schema=_TMDB_CHANGES_SCHEMA)
        .with_columns(pl.lit(date).alias("date"))
        .select("id", "date", "adult")
        .drop_nulls(subset=["id"])
        .unique(subset=["id"], keep="last", maintain_order=True)
    )
    logger.debug("_tmdb_changes(tmdb_type=%s, date=%s): %s", tmdb_type, date, df)
    return df


def tmdb_changes_backfill_date_range(
    df: pl.DataFrame,
    tmdb_type: TMDB_TYPE,
) -> list[date]:
    start_date = _TMDB_CHANGES_EPOCH[tmdb_type]
    if not df.is_empty():
        max_date = df["date"].max()
        assert max_date
        assert isinstance(max_date, date)
        start_date = max_date - timedelta(days=1)
    end_date = date.today()
    days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=i) for i in range(days)]


def insert_tmdb_latest_changes(
    df: pl.DataFrame,
    tmdb_type: TMDB_TYPE,
    tmdb_api_key: str,
    days_limit: int,
) -> pl.DataFrame:
    date_range = tmdb_changes_backfill_date_range(df, tmdb_type=tmdb_type)
    for d in islice(date_range, days_limit):
        changes = tmdb_changes(
            tmdb_type=tmdb_type,
            date=d,
            tmdb_api_key=tmdb_api_key,
        )
        df = df.pipe(update_or_append, changes)

    return df.pipe(align_id_col)


def fetch_jsonl_gz(url: str) -> Iterator[Any]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as response:
        logger.debug(
            "fetch_jsonl_gz(%s): %s %s",
            url,
            response.status,
            response.reason,
        )
        with gzip.open(response, mode="rt", encoding="utf-8") as gz:
            for line in gz:
                yield json.loads(line)


def export_date(now: datetime | None = None) -> date:
    if now is None:
        now = datetime.now(UTC)

    if 0 <= now.hour < 8:
        return (now - timedelta(days=1)).date()

    return now.date()


_TMDB_EXPORT_TYPE = Literal["movie", "tv_series", "person", "collection"]


def export_available(tmdb_type: _TMDB_EXPORT_TYPE, d: date) -> bool:
    url = (
        f"http://files.tmdb.org/p/exports/"
        f"{tmdb_type}_ids_{d.strftime('%m_%d_%Y')}.json.gz"
    )

    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status: int = getattr(response, "status", 0)
            return status == 200
    except Exception as exc:
        logger.warning("export_available(%s, %s): %s", tmdb_type, d, exc)
        return False


def _tmdb_raw_export(tmdb_type: _TMDB_EXPORT_TYPE) -> pl.DataFrame:
    date = export_date()
    logger.debug("export_date: %s", date)

    if not export_available(tmdb_type, date):
        logger.warning("export unavailable for %s on %s", tmdb_type, date)
        date2 = date - timedelta(days=1)
        if export_available(tmdb_type, date2):
            date = date2
        else:
            logger.warning("export unavailable for %s on %s", tmdb_type, date2)

    url = (
        f"http://files.tmdb.org/p/exports/"
        f"{tmdb_type}_ids_{date.strftime('%m_%d_%Y')}.json.gz"
    )
    data = fetch_jsonl_gz(url)
    df = (
        pl.from_dicts(data, schema=[("id", pl.UInt32)])
        .sort("id")
        .select(
            pl.col("id"),
            pl.lit(True).alias("in_export"),
        )
    )
    logger.debug("_tmdb_raw_export(tmdb_type=%s): %s", tmdb_type, df)
    return df


def tmdb_export(tmdb_type: TMDB_TYPE) -> pl.DataFrame:
    if tmdb_type == "movie":
        return pl.concat(
            [_tmdb_raw_export("movie"), _tmdb_raw_export("collection")]
        ).sort("id")
    elif tmdb_type == "tv":
        return _tmdb_raw_export("tv_series")
    elif tmdb_type == "person":
        return _tmdb_raw_export("person")


def _insert_tmdb_export_flag(df: pl.DataFrame, tmdb_type: TMDB_TYPE) -> pl.DataFrame:
    return (
        df.drop("in_export")
        .join(tmdb_export(tmdb_type), on="id", how="left", coalesce=True)
        .with_columns(pl.col("in_export").fill_null(False))
        .select(df.columns)
    )


def tmdb_external_ids(
    tmdb_type: TMDB_TYPE,
    tmdb_id: int,
    tmdb_api_key: str,
) -> dict[str, Any]:
    url = f"https://api.themoviedb.org/3/{tmdb_type}/{tmdb_id}/external_ids?api_key={tmdb_api_key}"

    success: bool = False
    retrieved_at: datetime = datetime.now(UTC)
    imdb_numeric_id: int | None = None
    tvdb_id: int | None = None
    wikidata_numeric_id: int | None = None

    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.load(response)
            success = True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            data = {}
            success = False
        else:
            logger.error(f"Error fetching external IDs for {tmdb_type} {tmdb_id}: {e}")
            raise

    if data.get("imdb_id"):
        if m := re.search(_IMDB_ID_PATTERN[tmdb_type], data["imdb_id"]):
            imdb_numeric_id = int(m.group(1))
        else:
            logger.warning(f"IMDb ID parse error: {data['imdb_id']}")

    if data.get("wikidata_id"):
        if m := re.search(r"Q(\d+)", data["wikidata_id"]):
            wikidata_numeric_id = int(m.group(1))
        else:
            logger.warning(f"Wikidata ID parse error: {data['wikidata_id']}")

    if data.get("tvdb_id"):
        tvdb_id = data["tvdb_id"]

    return {
        "id": tmdb_id,
        "success": success,
        "retrieved_at": retrieved_at,
        "imdb_numeric_id": imdb_numeric_id,
        "tvdb_id": tvdb_id,
        "wikidata_numeric_id": wikidata_numeric_id,
    }


def _tmdb_external_ids_iter(
    tmdb_type: TMDB_TYPE,
    tmdb_ids: Iterable[int],
    tmdb_api_key: str,
) -> Iterator[dict[str, Any]]:
    for tmdb_id in tmdb_ids:
        yield tmdb_external_ids(
            tmdb_type=tmdb_type,
            tmdb_id=tmdb_id,
            tmdb_api_key=tmdb_api_key,
        )


def _insert_tmdb_external_ids(
    df: pl.DataFrame,
    tmdb_type: TMDB_TYPE,
    tmdb_api_key: str,
    backfill_limit: int = 10000,
    refresh_limit: int = 1000,
) -> pl.DataFrame:
    df_to_update = df.filter(
        # If the tmdb change date is newer than our last retrieved at date
        (pl.col("date") >= pl.col("retrieved_at").dt.round("1d"))
        # Backfill any items we never retrieved
        | (
            pl.col("retrieved_at").is_null()
            & (pl.col("retrieved_at").is_not_null().rank("ordinal") <= backfill_limit)
        )
        # Refresh some of the oldest items
        | (pl.col("retrieved_at").rank("ordinal") <= refresh_limit)
    ).select("id")

    data = _tmdb_external_ids_iter(
        tmdb_type=tmdb_type,
        tmdb_ids=df_to_update["id"],
        tmdb_api_key=tmdb_api_key,
    )
    df_changes = pl.from_dicts(data, schema=_EXTERNAL_IDS_RESPONSE_SCHEMA)
    logger.debug("external id changes: %s", df_changes)

    return df.pipe(update_or_append, df_changes).pipe(align_id_col)


def process(
    df: pl.DataFrame,
    tmdb_type: TMDB_TYPE,
    tmdb_api_key: str,
    backfill_limit: int,
    refresh_limit: int,
    changes_days_limit: int,
) -> pl.DataFrame:
    df = insert_tmdb_latest_changes(
        df,
        tmdb_type=tmdb_type,
        tmdb_api_key=tmdb_api_key,
        days_limit=changes_days_limit,
    )
    df = _insert_tmdb_export_flag(df, tmdb_type=tmdb_type)
    df = _insert_tmdb_external_ids(
        df,
        tmdb_type=tmdb_type,
        tmdb_api_key=tmdb_api_key,
        backfill_limit=backfill_limit,
        refresh_limit=refresh_limit,
    )
    return df


@click.command()
@click.argument(
    "filename",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--tmdb-type",
    type=click.Choice(["movie", "tv", "person"]),
    required=True,
    envvar="TMDB_TYPE",
    help="TMDB type to update",
)
@click.option(
    "--tmdb-api-key",
    type=str,
    required=True,
    envvar="TMDB_API_KEY",
    help="TMDB API key",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Dry run",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Verbose output",
)
@click.option(
    "--backfill-limit",
    type=int,
    default=10000,
    envvar="TMDB_BACKFILL_LIMIT",
    help="Number of never-fetched rows to backfill",
)
@click.option(
    "--refresh-limit",
    type=int,
    default=1000,
    envvar="TMDB_REFRESH_LIMIT",
    help="Number of old rows to refresh",
)
@click.option(
    "--days-limit",
    type=int,
    default=30,
    envvar="TMDB_DAYS_LIMIT",
    help="Limit of changes days to backfill",
)
def main(
    filename: str,
    tmdb_type: TMDB_TYPE,
    tmdb_api_key: str,
    dry_run: bool,
    verbose: bool,
    backfill_limit: int,
    refresh_limit: int,
    days_limit: int,
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    pl.enable_string_cache()

    df = pl.read_parquet(filename)
    logger.debug("original df: %s", df)

    df2 = process(
        df=df,
        tmdb_type=tmdb_type,
        tmdb_api_key=tmdb_api_key,
        backfill_limit=backfill_limit,
        refresh_limit=refresh_limit,
        changes_days_limit=days_limit,
    )

    if df2.height < df.height:
        logger.warning(
            "df2 height %s is smaller than df height %s", df2.height, df.height
        )
    logger.info(df2)
    logger.info(change_summary(df, df2))

    assert df.schema == df2.schema, f"{df.schema} != {df2.schema}"

    if not dry_run:
        df2.write_parquet(
            filename,
            compression="zstd",
            statistics=True,
        )
    else:
        logger.debug("dry run, skipping write")


if __name__ == "__main__":
    main()
