# tmdb-index

An enhanced version of the [TMDB Daily ID Export](https://developer.themoviedb.org/docs/daily-id-exports) as a [parquet file](https://parquet.apache.org/).

```
>>> import polars as pl
>>> pl.read_parquet("https://josh.github.io/tmdb-index/tmdb-movie.parquet").filter(pl.col("imdb_numeric_id") == 111161)
┌──────────┬─────────────┬──────────────┬──────────────────┬────────────────┬─────────────────────────────┬───────────────────────┬───────────────────────────┐
│ id (u32) ┆ date (date) ┆ adult (bool) ┆ in_export (bool) ┆ success (bool) ┆ retrieved_at (datetime[ns]) ┆ imdb_numeric_id (u32) ┆ wikidata_numeric_id (u32) │
╞══════════╪═════════════╪══════════════╪══════════════════╪════════════════╪═════════════════════════════╪═══════════════════════╪═══════════════════════════╡
│ 278      ┆ 2023-03-05  ┆ false        ┆ true             ┆ true           ┆ 2023-03-05 16:04:53         ┆ 111161                ┆ 172241                    │
└──────────┴─────────────┴──────────────┴──────────────────┴────────────────┴─────────────────────────────┴───────────────────────┴───────────────────────────┘
```
