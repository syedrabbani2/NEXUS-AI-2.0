
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp


MAX_QUERY_ROWS = 500
MAX_FILE_SIZE = 25 * 1024 * 1024

BLOCKED_SQL_FUNCTIONS = {
    "READ_CSV", "READ_CSV_AUTO", "READ_PARQUET",
    "READ_JSON", "READ_JSON_AUTO", "READ_TEXT",
    "READ_BLOB", "GLOB", "QUERY", "QUERY_TABLE",
    "PARQUET_SCAN", "CSV_SCAN", "HTTP_GET",
    "HTTP_POST", "READ_NDJSON",
}

BLOCKED_SQL_KEYWORDS = {
    "ATTACH", "DETACH", "INSTALL", "LOAD", "COPY",
    "EXPORT", "IMPORT", "PRAGMA", "SET", "RESET",
    "CREATE", "DROP", "ALTER", "INSERT", "UPDATE",
    "DELETE", "MERGE", "CALL", "EXECUTE",
}


def safe_table_name(value: str) -> str:
    """Convert a name into a safe SQL identifier."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", str(value).strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")

    if not name:
        name = "data_table"

    if name[0].isdigit():
        name = f"table_{name}"

    return name[:60]


def quote_identifier(value: str) -> str:
    """Quote an internally generated DuckDB identifier."""
    return '"' + value.replace('"', '""') + '"'


def calculate_file_hash(file_path: str | Path) -> str:
    digest = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
    return digest[:16]


def ensure_registry(connection):
    """Create the metadata registry if it does not already exist."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS nexus_source_registry (
            file_id VARCHAR NOT NULL,
            source VARCHAR NOT NULL,
            table_name VARCHAR NOT NULL,
            sheet_name VARCHAR,
            row_count BIGINT,
            columns_json VARCHAR,
            indexed_at VARCHAR
        )
    """)


def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and remove entirely empty rows."""
    dataframe = dataframe.copy()

    cleaned_columns = []
    used_names = set()

    for index, column in enumerate(dataframe.columns, start=1):
        base_name = safe_table_name(column)

        if base_name == "data_table":
            base_name = f"column_{index}"

        name = base_name
        suffix = 2

        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1

        used_names.add(name)
        cleaned_columns.append(name)

    dataframe.columns = cleaned_columns
    dataframe = dataframe.dropna(how="all")
    dataframe = dataframe.reset_index(drop=True)

    return dataframe


def load_structured_file(
    file_path: str | Path,
    original_name: str | None = None,
) -> dict:
    """Load CSV or Excel sheets into clean DataFrames."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.name}")

    if path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError("Structured file exceeds the 25 MB size limit.")

    extension = path.suffix.lower()
    source_name = original_name or path.name
    raw_tables = {}

    if extension == ".csv":
        dataframe = pd.read_csv(path)
        raw_tables["CSV"] = dataframe

    elif extension == ".xlsx":
        raw_tables = pd.read_excel(
            path,
            sheet_name=None,
            engine="openpyxl",
        )

    else:
        raise ValueError("Only CSV and XLSX files are supported.")

    cleaned_tables = {}

    for index, (sheet_name, dataframe) in enumerate(
        raw_tables.items(), start=1
    ):
        dataframe = clean_dataframe(dataframe)

        if dataframe.empty:
            continue

        cleaned_tables[str(sheet_name)] = dataframe

    if not cleaned_tables:
        raise ValueError("The file contains no usable data rows.")

    return {
        "filename": source_name,
        "tables": cleaned_tables,
        "table_count": len(cleaned_tables),
        "row_count": sum(len(df) for df in cleaned_tables.values()),
    }


def index_structured_file(
    file_path: str | Path,
    original_name: str,
    database_path: str | Path,
    file_id: str | None = None,
) -> dict:
    """Persist each CSV/XLSX sheet as a DuckDB table."""
    path = Path(file_path)

    file_id = file_id or calculate_file_hash(path)
    loaded = load_structured_file(path, original_name)

    connection = duckdb.connect(str(database_path))

    try:
        ensure_registry(connection)

        # Build deterministic, file-specific table names.
        prepared = []

        for index, (sheet_name, dataframe) in enumerate(
            loaded["tables"].items(), start=1
        ):
            sheet_slug = safe_table_name(sheet_name)[:20]
            table_name = safe_table_name(
                f"nexus_{file_id}_{index}_{sheet_slug}"
            )

            prepared.append(
                (table_name, sheet_name, dataframe)
            )

        connection.execute("BEGIN TRANSACTION")

        try:
            # Remove old tables belonging to this source only.
            old_tables = connection.execute(
                """
                SELECT DISTINCT table_name
                FROM nexus_source_registry
                WHERE source = ?
                """,
                [original_name],
            ).fetchall()

            for (old_table,) in old_tables:
                connection.execute(
                    f"DROP TABLE IF EXISTS {quote_identifier(old_table)}"
                )

            connection.execute(
                "DELETE FROM nexus_source_registry WHERE source = ?",
                [original_name],
            )

            indexed_at = datetime.now(timezone.utc).isoformat()

            for table_name, sheet_name, dataframe in prepared:
                connection.register("_nexus_dataframe", dataframe)

                try:
                    connection.execute(
                        f"""
                        CREATE TABLE {quote_identifier(table_name)}
                        AS SELECT * FROM _nexus_dataframe
                        """
                    )
                finally:
                    connection.unregister("_nexus_dataframe")

                columns = [
                    {
                        "name": str(column),
                        "dtype": str(dataframe[column].dtype),
                    }
                    for column in dataframe.columns
                ]

                connection.execute(
                    """
                    INSERT INTO nexus_source_registry
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        file_id,
                        original_name,
                        table_name,
                        sheet_name,
                        len(dataframe),
                        json.dumps(columns),
                        indexed_at,
                    ],
                )

            connection.execute("COMMIT")

        except Exception:
            connection.execute("ROLLBACK")
            raise

        return {
            "status": "indexed",
            "source": original_name,
            "file_id": file_id,
            "table_count": len(prepared),
            "row_count": loaded["row_count"],
            "tables": [item[0] for item in prepared],
        }

    finally:
        connection.close()


def list_structured_sources(database_path: str | Path) -> list:
    """List all indexed structured-data files and their tables."""
    connection = duckdb.connect(str(database_path))

    try:
        ensure_registry(connection)

        rows = connection.execute("""
            SELECT
                file_id,
                source,
                table_name,
                sheet_name,
                row_count,
                indexed_at
            FROM nexus_source_registry
            ORDER BY source, table_name
        """).fetchall()

        return [
            {
                "file_id": row[0],
                "source": row[1],
                "table_name": row[2],
                "sheet_name": row[3],
                "row_count": row[4],
                "indexed_at": row[5],
            }
            for row in rows
        ]

    finally:
        connection.close()


def get_database_schemas(database_path: str | Path) -> dict:
    """Return registered tables, columns and row counts."""
    connection = duckdb.connect(str(database_path))

    try:
        ensure_registry(connection)

        rows = connection.execute("""
            SELECT table_name, source, sheet_name, row_count, columns_json
            FROM nexus_source_registry
            ORDER BY source, table_name
        """).fetchall()

        return {
            row[0]: {
                "source": row[1],
                "sheet_name": row[2],
                "row_count": row[3],
                "columns": json.loads(row[4] or "[]"),
            }
            for row in rows
        }

    finally:
        connection.close()


def delete_structured_source(
    database_path: str | Path,
    source: str,
) -> int:
    """Delete only the DuckDB tables belonging to one source."""
    connection = duckdb.connect(str(database_path))

    try:
        ensure_registry(connection)

        rows = connection.execute(
            """
            SELECT table_name
            FROM nexus_source_registry
            WHERE source = ?
            """,
            [source],
        ).fetchall()

        connection.execute("BEGIN TRANSACTION")

        try:
            for (table_name,) in rows:
                connection.execute(
                    f"DROP TABLE IF EXISTS {quote_identifier(table_name)}"
                )

            connection.execute(
                "DELETE FROM nexus_source_registry WHERE source = ?",
                [source],
            )

            connection.execute("COMMIT")

        except Exception:
            connection.execute("ROLLBACK")
            raise

        return len(rows)

    finally:
        connection.close()


def validate_readonly_sql(sql: str, allowed_tables=None) -> str:
    """Validate one read-only SELECT and optionally restrict physical tables."""

    if not sql or not sql.strip():
        raise ValueError("SQL query is empty.")

    if len(sql) > 10000:
        raise ValueError("SQL query is too long.")

    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except Exception as exc:
        raise ValueError("The SQL query could not be parsed.") from exc

    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Only one SQL statement is permitted.")

    tree = statements[0]

    if not isinstance(
        tree,
        (exp.Select, exp.Union, exp.Intersect, exp.Except),
    ):
        raise ValueError("Only read-only SELECT queries are permitted.")

    for node in tree.walk():
        if isinstance(node, exp.Func):
            function_name = node.sql_name().upper()

            if function_name in BLOCKED_SQL_FUNCTIONS:
                raise ValueError(
                    f"The SQL function {function_name} is not permitted."
                )

        if isinstance(node, exp.Command):
            raise ValueError("SQL commands are not permitted.")

    if allowed_tables is not None:
        allowed = {name.lower() for name in allowed_tables}

        # Permit CTE aliases, but only registered physical tables.
        cte_names = {
            cte.alias_or_name.lower()
            for cte in tree.find_all(exp.CTE)
            if cte.alias_or_name
        }

        for table in tree.find_all(exp.Table):
            table_name = table.name.lower()

            # Do not permit cross-database or catalog-qualified references.
            if table.db or table.catalog:
                raise ValueError(
                    "Database-qualified table references are not permitted."
                )

            if table_name in cte_names:
                continue

            if table_name not in allowed:
                raise ValueError(
                    f"Table '{table.name}' is not registered in NEXUS."
                )

    return tree.sql(dialect="duckdb")


def execute_persistent_query(
    database_path: str | Path,
    sql: str,
    max_rows: int = MAX_QUERY_ROWS,
) -> dict:
    """Execute validated read-only SQL against registered NEXUS tables."""

    max_rows = max(1, min(int(max_rows), MAX_QUERY_ROWS))

    connection = duckdb.connect(
        str(database_path),
        read_only=True,
        config={"enable_external_access": "false"},
    )

    try:
        # Retrieve the physical table allowlist from the existing registry.
        registry_rows = connection.execute(
            "SELECT table_name FROM nexus_source_registry"
        ).fetchall()

        allowed_tables = {row[0] for row in registry_rows}

        if not allowed_tables:
            raise ValueError("No structured-data tables are indexed yet.")

        safe_sql = validate_readonly_sql(
            sql,
            allowed_tables=allowed_tables,
        )

        # Enforce the result limit before execution.
        tree = sqlglot.parse_one(safe_sql, read="duckdb")
        tree = tree.limit(max_rows + 1)
        limited_sql = tree.sql(dialect="duckdb")

        result = connection.execute(limited_sql)

        rows = result.fetchmany(max_rows + 1)
        columns = [item[0] for item in result.description]

        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

        dataframe = pd.DataFrame(rows, columns=columns)

        return {
            "dataframe": dataframe,
            "returned_rows": len(dataframe),
            "truncated": truncated,
            "columns": columns,
            "sql": limited_sql,
        }

    finally:
        connection.close()
