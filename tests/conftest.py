import pytest

from vectora import db as vdb


@pytest.fixture()
def test_db(tmp_path):
    """Fresh DuckDB with full schema, isolated per test."""
    path = tmp_path / "test.duckdb"
    con = vdb.connect(path)
    vdb.init_schema(con)
    yield con
    con.close()


@pytest.fixture()
def fixtures_dir():
    from pathlib import Path
    return Path(__file__).parent / "fixtures"
