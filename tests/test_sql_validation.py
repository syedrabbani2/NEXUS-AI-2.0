
import pytest

from modules.structured_data import validate_readonly_sql


ALLOWED_TABLES = {"employees"}


def test_valid_select_query():
    result = validate_readonly_sql(
        "SELECT * FROM employees",
        allowed_tables=ALLOWED_TABLES,
    )
    assert "employees" in result.lower()


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        validate_readonly_sql("", allowed_tables=ALLOWED_TABLES)


def test_delete_query_is_rejected():
    with pytest.raises(ValueError):
        validate_readonly_sql(
            "DELETE FROM employees",
            allowed_tables=ALLOWED_TABLES,
        )


def test_drop_query_is_rejected():
    with pytest.raises(ValueError):
        validate_readonly_sql(
            "DROP TABLE employees",
            allowed_tables=ALLOWED_TABLES,
        )


def test_multiple_statements_are_rejected():
    with pytest.raises(ValueError):
        validate_readonly_sql(
            "SELECT * FROM employees; DROP TABLE employees",
            allowed_tables=ALLOWED_TABLES,
        )


def test_unregistered_table_is_rejected():
    with pytest.raises(ValueError, match="not registered"):
        validate_readonly_sql(
            "SELECT * FROM secret_data",
            allowed_tables=ALLOWED_TABLES,
        )


def test_external_csv_access_is_rejected():
    with pytest.raises(ValueError, match="not permitted"):
        validate_readonly_sql(
            "SELECT * FROM read_csv('test.csv')",
            allowed_tables=ALLOWED_TABLES,
        )


def test_cross_database_reference_is_rejected():
    with pytest.raises(ValueError, match="Database-qualified"):
        validate_readonly_sql(
            "SELECT * FROM other_db.main.employees",
            allowed_tables=ALLOWED_TABLES,
        )


def test_authorized_cte_is_accepted():
    result = validate_readonly_sql(
        "WITH staff AS (SELECT * FROM employees) SELECT * FROM staff",
        allowed_tables=ALLOWED_TABLES,
    )
    assert "staff" in result.lower()
