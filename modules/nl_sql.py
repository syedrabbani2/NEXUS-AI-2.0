
import re

from langchain_openai import ChatOpenAI

from modules.structured_data import (
    get_database_schemas,
    execute_persistent_query,
)


STRUCTURED_KEYWORDS = {
    "count",
    "total",
    "average",
    "avg",
    "sum",
    "maximum",
    "minimum",
    "highest",
    "lowest",
    "workload",
    "class-wise",
    "spreadsheet",
    "excel",
    "csv",
    "worksheet",
    "records",
    "rows",
    "columns",
    "group by",
    "distribution",
    "breakdown",
    "how many",
    "how much",
    "number of",
    "percentage",
    "per department",
    "by department",
}

def is_structured_question(question: str) -> bool:
    """Identify questions requiring structured spreadsheet analytics."""

    text = question.lower().strip()

    return any(
        keyword in text
        for keyword in STRUCTURED_KEYWORDS
    )


def generate_sql_from_question(
    question,
    database_path,
    api_key,
    model_name,
):
    """Generate, validate and execute read-only SQL from a user question."""

    schemas = get_database_schemas(database_path)

    if not schemas:
        raise ValueError("No structured-data tables are indexed.")

    schema_description = []

    for table_name, details in schemas.items():
        columns = ", ".join(
            f"{column['name']} ({column['dtype']})"
            for column in details["columns"]
        )

        schema_description.append(
            f"""
Table: {table_name}
Source: {details['source']}
Worksheet: {details['sheet_name']}
Columns: {columns}
"""
        )

    schema_text = "\n".join(schema_description)

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_retries=2,
    )

    prompt = f"""
You are the SQL generation engine for NEXUS AI.

Generate exactly one DuckDB SQL SELECT query for the user's question.

AVAILABLE DATABASE SCHEMA:
{schema_text}

STRICT RULES:
1. Use only the exact registered table names and columns above.
2. Generate only one SELECT query.
3. Never generate INSERT, UPDATE, DELETE, DROP, ALTER or CREATE.
4. Never use external files, URLs, extensions or filesystem functions.
5. Do not invent table names or column names.
6. Prefer aggregate results when the question asks for totals,
   counts, averages or distributions.
7. Avoid exposing personal contact details or addresses.
8. Return SQL only, without Markdown fences or explanations.
9. If the question cannot be answered from this schema,
   return exactly: CANNOT_ANSWER

USER QUESTION:
{question}

SQL:
"""

    response = llm.invoke(prompt)
    sql = str(response.content).strip()

    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql).strip()

    if sql.upper() == "CANNOT_ANSWER":
        raise ValueError(
            "The available structured data cannot answer this question."
        )

    result = execute_persistent_query(
        database_path=database_path,
        sql=sql,
        max_rows=500,
    )

    return {
        "sql": result["sql"],
        "dataframe": result["dataframe"],
        "returned_rows": result["returned_rows"],
        "truncated": result["truncated"],
    }
