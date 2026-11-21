
from unittest.mock import MagicMock

import pandas as pd
import pytest

from modules import nl_sql


# --------------------------------------------------
# STRUCTURED QUESTION DETECTION
# --------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "What is the total fee collection?",
        "How many students are enrolled?",
        "Show average marks",
        "Give me the highest salary",
        "Show records from the spreadsheet",
        "Give me a department-wise breakdown",
        "Show the CSV distribution",
        "What is the workload per department?",
    ],
)
def test_is_structured_question_detects_keywords(question):
    assert nl_sql.is_structured_question(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Explain photosynthesis",
        "Write a Python function",
        "What is artificial intelligence?",
        "Tell me about cloud computing",
    ],
)
def test_is_structured_question_rejects_general_questions(question):
    assert nl_sql.is_structured_question(question) is False


def test_is_structured_question_is_case_insensitive():
    assert nl_sql.is_structured_question("TOTAL FEE COLLECTION")


def test_is_structured_question_handles_whitespace():
    assert nl_sql.is_structured_question("   How many students?   ")


def test_empty_question_is_not_structured():
    assert nl_sql.is_structured_question("") is False


# --------------------------------------------------
# SQL GENERATION
# --------------------------------------------------

def sample_schema():
    return {
        "students": {
            "source": "students.csv",
            "sheet_name": "Sheet1",
            "columns": [
                {"name": "student_id", "dtype": "INTEGER"},
                {"name": "student_name", "dtype": "VARCHAR"},
                {"name": "fee_amount", "dtype": "DOUBLE"},
            ],
        }
    }


def setup_mock_llm(monkeypatch, response_content):
    mock_instance = MagicMock()
    mock_instance.invoke.return_value.content = response_content

    mock_class = MagicMock(return_value=mock_instance)

    monkeypatch.setattr(nl_sql, "ChatOpenAI", mock_class)

    return mock_class, mock_instance


def test_generate_sql_rejects_empty_database(monkeypatch):
    monkeypatch.setattr(
        nl_sql,
        "get_database_schemas",
        lambda path: {},
    )

    with pytest.raises(ValueError, match="No structured-data tables"):
        nl_sql.generate_sql_from_question(
            "How many students?",
            "test.duckdb",
            "test-key",
            "test-model",
        )


def test_generate_sql_executes_generated_query(monkeypatch):
    monkeypatch.setattr(
        nl_sql,
        "get_database_schemas",
        lambda path: sample_schema(),
    )

    mock_class, mock_instance = setup_mock_llm(
        monkeypatch,
        "SELECT COUNT(*) AS total FROM students",
    )

    expected_df = pd.DataFrame({"total": [100]})

    expected_result = {
        "sql": "SELECT COUNT(*) AS total FROM students",
        "dataframe": expected_df,
        "returned_rows": 1,
        "truncated": False,
    }

    mock_execute = MagicMock(return_value=expected_result)

    monkeypatch.setattr(
        nl_sql,
        "execute_persistent_query",
        mock_execute,
    )

    result = nl_sql.generate_sql_from_question(
        "How many students?",
        "test.duckdb",
        "test-key",
        "test-model",
    )

    assert result == expected_result

    mock_class.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_retries=2,
    )

    mock_execute.assert_called_once_with(
        database_path="test.duckdb",
        sql="SELECT COUNT(*) AS total FROM students",
        max_rows=500,
    )

    assert "students.csv" in mock_instance.invoke.call_args[0][0]
    assert "student_id (INTEGER)" in mock_instance.invoke.call_args[0][0]


def test_generate_sql_removes_sql_markdown_fences(monkeypatch):
    monkeypatch.setattr(
        nl_sql,
        "get_database_schemas",
        lambda path: sample_schema(),
    )

    setup_mock_llm(
        monkeypatch,
        "```sql\nSELECT * FROM students\n```",
    )

    mock_execute = MagicMock(
        return_value={
            "sql": "SELECT * FROM students",
            "dataframe": pd.DataFrame(),
            "returned_rows": 0,
            "truncated": False,
        }
    )

    monkeypatch.setattr(
        nl_sql,
        "execute_persistent_query",
        mock_execute,
    )

    nl_sql.generate_sql_from_question(
        "Show students",
        "test.duckdb",
        "test-key",
        "test-model",
    )

    mock_execute.assert_called_once_with(
        database_path="test.duckdb",
        sql="SELECT * FROM students",
        max_rows=500,
    )


def test_generate_sql_rejects_cannot_answer(monkeypatch):
    monkeypatch.setattr(
        nl_sql,
        "get_database_schemas",
        lambda path: sample_schema(),
    )

    setup_mock_llm(monkeypatch, "CANNOT_ANSWER")

    mock_execute = MagicMock()

    monkeypatch.setattr(
        nl_sql,
        "execute_persistent_query",
        mock_execute,
    )

    with pytest.raises(ValueError, match="cannot answer"):
        nl_sql.generate_sql_from_question(
            "What is the weather?",
            "test.duckdb",
            "test-key",
            "test-model",
        )

    mock_execute.assert_not_called()


def test_generate_sql_propagates_execution_errors(monkeypatch):
    monkeypatch.setattr(
        nl_sql,
        "get_database_schemas",
        lambda path: sample_schema(),
    )

    setup_mock_llm(
        monkeypatch,
        "SELECT * FROM students",
    )

    monkeypatch.setattr(
        nl_sql,
        "execute_persistent_query",
        MagicMock(side_effect=ValueError("Unsafe SQL query")),
    )

    with pytest.raises(ValueError, match="Unsafe SQL"):
        nl_sql.generate_sql_from_question(
            "Show students",
            "test.duckdb",
            "test-key",
            "test-model",
        )
