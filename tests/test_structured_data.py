
import pandas as pd
import duckdb
import pytest

from modules.structured_data import (
    index_structured_file,
    list_structured_sources,
    get_database_schemas,
    delete_structured_source,
    execute_persistent_query,
    ensure_registry,
)


def create_csv(tmp_path, filename, data):
    path = tmp_path / filename
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_index_structured_csv(tmp_path):
    csv_path = create_csv(
        tmp_path,
        "employees.csv",
        {
            "Name": ["John", "Sarah"],
            "Department": ["IT", "HR"],
        },
    )

    db_path = tmp_path / "test.duckdb"

    result = index_structured_file(
        csv_path, "employees.csv", db_path, file_id="emp001"
    )

    assert result["status"] == "indexed"
    assert result["source"] == "employees.csv"
    assert result["table_count"] == 1
    assert result["row_count"] == 2
    assert len(result["tables"]) == 1


def test_list_structured_sources(tmp_path):
    csv_path = create_csv(
        tmp_path,
        "employees.csv",
        {"Name": ["John"], "Department": ["IT"]},
    )
    db_path = tmp_path / "test.duckdb"

    index_structured_file(
        csv_path, "employees.csv", db_path, file_id="emp001"
    )

    sources = list_structured_sources(db_path)

    assert len(sources) == 1
    assert sources[0]["source"] == "employees.csv"
    assert sources[0]["row_count"] == 1


def test_get_database_schemas(tmp_path):
    csv_path = create_csv(
        tmp_path,
        "employees.csv",
        {"Name": ["John"], "Department": ["IT"]},
    )
    db_path = tmp_path / "test.duckdb"

    result = index_structured_file(
        csv_path, "employees.csv", db_path, file_id="emp001"
    )

    schemas = get_database_schemas(db_path)

    table_name = result["tables"][0]

    assert table_name in schemas
    assert schemas[table_name]["source"] == "employees.csv"
    assert schemas[table_name]["row_count"] == 1

    column_names = [
        column["name"] for column in schemas[table_name]["columns"]
    ]

    assert column_names == ["name", "department"]


def test_reindex_replaces_existing_source(tmp_path):
    csv_path = create_csv(
        tmp_path,
        "employees.csv",
        {"Name": ["John"]},
    )
    db_path = tmp_path / "test.duckdb"

    first = index_structured_file(
        csv_path, "employees.csv", db_path, file_id="emp001"
    )

    create_csv(
        tmp_path,
        "employees.csv",
        {"Name": ["Sarah", "David"]},
    )

    second = index_structured_file(
        csv_path, "employees.csv", db_path, file_id="emp002"
    )

    sources = list_structured_sources(db_path)

    assert first["status"] == "indexed"
    assert second["row_count"] == 2
    assert len(sources) == 1
    assert sources[0]["row_count"] == 2
    assert sources[0]["file_id"] == "emp002"


def test_delete_structured_source(tmp_path):
    csv_path = create_csv(
        tmp_path,
        "employees.csv",
        {"Name": ["John"]},
    )
    db_path = tmp_path / "test.duckdb"

    index_structured_file(
        csv_path, "employees.csv", db_path, file_id="emp001"
    )

    deleted = delete_structured_source(db_path, "employees.csv")

    assert deleted == 1
    assert list_structured_sources(db_path) == []


def test_delete_one_source_preserves_other_sources(tmp_path):
    first_csv = create_csv(
        tmp_path, "employees.csv", {"Name": ["John"]}
    )
    second_csv = create_csv(
        tmp_path, "departments.csv", {"Department": ["IT"]}
    )

    db_path = tmp_path / "test.duckdb"

    index_structured_file(
        first_csv, "employees.csv", db_path, file_id="emp001"
    )
    index_structured_file(
        second_csv, "departments.csv", db_path, file_id="dep001"
    )

    deleted = delete_structured_source(db_path, "employees.csv")
    remaining = list_structured_sources(db_path)

    assert deleted == 1
    assert len(remaining) == 1
    assert remaining[0]["source"] == "departments.csv"


def test_execute_persistent_query_returns_results(tmp_path):
    csv_path = create_csv(
        tmp_path,
        "employees.csv",
        {
            "Name": ["John", "Sarah"],
            "Department": ["IT", "HR"],
        },
    )
    db_path = tmp_path / "test.duckdb"

    result = index_structured_file(
        csv_path, "employees.csv", db_path, file_id="emp001"
    )

    table_name = result["tables"][0]

    query_result = execute_persistent_query(
        db_path,
        f"SELECT * FROM {table_name}",
    )

    assert query_result["returned_rows"] == 2
    assert query_result["truncated"] is False
    assert "name" in query_result["columns"]
    assert len(query_result["dataframe"]) == 2


def test_execute_persistent_query_enforces_row_limit(tmp_path):
    csv_path = create_csv(
        tmp_path,
        "employees.csv",
        {"Name": ["John", "Sarah", "David", "Alex"]},
    )
    db_path = tmp_path / "test.duckdb"

    result = index_structured_file(
        csv_path, "employees.csv", db_path, file_id="emp001"
    )

    table_name = result["tables"][0]

    query_result = execute_persistent_query(
        db_path,
        f"SELECT * FROM {table_name}",
        max_rows=2,
    )

    assert query_result["returned_rows"] == 2
    assert query_result["truncated"] is True
    assert len(query_result["dataframe"]) == 2


def test_execute_query_rejects_empty_database(tmp_path):
    db_path = tmp_path / "empty.duckdb"

    connection = duckdb.connect(str(db_path))
    try:
        ensure_registry(connection)
    finally:
        connection.close()

    with pytest.raises(ValueError, match="No structured-data tables"):
        execute_persistent_query(
            db_path,
            "SELECT * FROM employees",
        )
