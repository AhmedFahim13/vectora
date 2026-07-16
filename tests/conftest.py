import pytest

from vectora import db as vdb


@pytest.fixture()
def test_db(tmp_path):
    """Fresh DuckDB with full schema, isolated per test.

    Pins backfill_parquet to a path that never exists so tests stay isolated
    from the real committed data/reference/backfill_2012_2026.parquet -
    without this, init_schema()'s default would union in >1M real backfill
    rows once that file exists on disk.
    """
    path = tmp_path / "test.duckdb"
    con = vdb.connect(path)
    vdb.init_schema(con, backfill_parquet=tmp_path / "no_backfill.parquet")
    yield con
    con.close()


@pytest.fixture()
def fixtures_dir():
    from pathlib import Path
    return Path(__file__).parent / "fixtures"
