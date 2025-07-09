# tmdb-index

An enhanced version of the [TMDB Daily ID Export](https://developer.themoviedb.org/docs/daily-id-exports) as a parquet file.

```
>>> import polars as pl
>>> pl.read_parquet("https://josh.github.io/tmdb-index/tmdb-movie.parquet").filter(pl.col("imdb_numeric_id") == 111161)
shape: (1, 9)
┌─────┬────────────┬───────┬───────────┬─────────┬─────────────────────┬─────────────────┬─────────┬─────────────────────┐
│ id  ┆ date       ┆ adult ┆ in_export ┆ success ┆ retrieved_at        ┆ imdb_numeric_id ┆ tvdb_id ┆ wikidata_numeric_id │
│ --- ┆ ---        ┆ ---   ┆ ---       ┆ ---     ┆ ---                 ┆ ---             ┆ ---     ┆ ---                 │
│ u32 ┆ date       ┆ bool  ┆ bool      ┆ bool    ┆ datetime[ns]        ┆ u32             ┆ u32     ┆ u32                 │
╞═════╪════════════╪═══════╪═══════════╪═════════╪═════════════════════╪═════════════════╪═════════╪═════════════════════╡
│ 278 ┆ 2023-03-05 ┆ false ┆ true      ┆ true    ┆ 2023-03-05 16:04:53 ┆ 111161          ┆ null    ┆ 172241              │
└─────┴────────────┴───────┴───────────┴─────────┴─────────────────────┴─────────────────┴─────────┴─────────────────────┘
```
