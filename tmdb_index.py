import datetime
import gzip
import json
import logging
import re
import urllib.request
from collections.abc import Iterable, Iterator
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


_TMDB_CHANGES_SCHEMA = pl.Schema(
    [
        ("id", pl.UInt32),
        ("adult", pl.Boolean),
    ]
)


def tmdb_changes(
    tmdb_type: TMDB_TYPE,
    date: datetime.date,
    tmdb_api_key: str,
) -> pl.DataFrame:
    start_date = date.strftime("%Y-%m-%d")
    end_date = (date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
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


def _insert_tmdb_latest_changes(
    df: pl.DataFrame,
    tmdb_type: TMDB_TYPE,
    tmdb_api_key: str,
) -> pl.DataFrame:
    dates_df = df.select(
        pl.date_range(
            pl.col("date").max().dt.offset_by("-1d").alias("start_date"),
            datetime.date.today(),
            interval="1d",
            eager=False,
        ).alias("date")
    )
    dates_lst: list[datetime.date] = (
        dates_df.select(pl.col("date")).to_series().to_list()
    )

    for d in dates_lst:
        changes = tmdb_changes(
            tmdb_type=tmdb_type,
            date=d,
            tmdb_api_key=tmdb_api_key,
        )
        df = df.pipe(update_or_append, changes)

    return df.pipe(align_id_col)


def _fetch_jsonl_gz(url: str) -> Iterator[Any]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as response:
        with gzip.open(response, mode="rb") as gz:
            for line in gz:
                yield json.loads(line.decode("utf-8"))


def _export_date() -> datetime.date:
    now = datetime.datetime.now(datetime.UTC)
    if now.hour >= 8:
        return now.date()
    else:
        return (now - datetime.timedelta(days=1)).date()


_TMDB_EXPORT_TYPE = Literal["movie", "tv_series", "person", "collection"]


def _tmdb_raw_export(tmdb_type: _TMDB_EXPORT_TYPE) -> pl.DataFrame:
    date = _export_date()
    logger.debug("_export_date: %s", date)
    url = f"http://files.tmdb.org/p/exports/{tmdb_type}_ids_{date.strftime('%m_%d_%Y')}.json.gz"
    data = _fetch_jsonl_gz(url)
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


def _tmdb_export(tmdb_type: TMDB_TYPE) -> pl.DataFrame:
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
        .join(_tmdb_export(tmdb_type), on="id", how="left", coalesce=True)
        .with_columns(pl.col("in_export").fill_null(False))
        .select(df.columns)
    )


def _tmdb_external_ids(
    tmdb_type: TMDB_TYPE,
    tmdb_id: int,
    tmdb_api_key: str,
) -> dict[str, Any]:
    url = f"https://api.themoviedb.org/3/{tmdb_type}/{tmdb_id}/external_ids?api_key={tmdb_api_key}"

    success: bool = False
    retrieved_at: datetime.datetime = datetime.datetime.now(datetime.UTC)
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
        yield _tmdb_external_ids(
            tmdb_type=tmdb_type,
            tmdb_id=tmdb_id,
            tmdb_api_key=tmdb_api_key,
        )


_CHANGED = pl.col("date") >= pl.col("retrieved_at").dt.round("1d")
_NEVER_FETCHED = pl.col("retrieved_at").is_null()
_OLDEST_METADATA = pl.col("retrieved_at").rank("ordinal") <= 1_000


def _insert_tmdb_external_ids(
    df: pl.DataFrame,
    tmdb_type: TMDB_TYPE,
    tmdb_api_key: str,
) -> pl.DataFrame:
    df_to_update = df.filter(_CHANGED | _NEVER_FETCHED | _OLDEST_METADATA).select("id")

    data = _tmdb_external_ids_iter(
        tmdb_type=tmdb_type,
        tmdb_ids=df_to_update["id"],
        tmdb_api_key=tmdb_api_key,
    )
    df_changes = pl.from_dicts(data, schema=_EXTERNAL_IDS_RESPONSE_SCHEMA)
    logger.debug("external id changes: %s", df_changes)

    return df.pipe(update_or_append, df_changes).pipe(align_id_col)


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
def main(
    filename: str,
    tmdb_type: TMDB_TYPE,
    tmdb_api_key: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    pl.enable_string_cache()

    df = pl.read_parquet(filename)
    logger.debug("original df: %s", df)

    df2 = (
        df.pipe(_insert_tmdb_latest_changes, tmdb_type, tmdb_api_key)
        .pipe(_insert_tmdb_export_flag, tmdb_type)
        .pipe(_insert_tmdb_external_ids, tmdb_type, tmdb_api_key)
    )
    logger.info(df2)

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
