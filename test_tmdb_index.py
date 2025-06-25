import polars as pl

from tmdb_index import align_id_col


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
