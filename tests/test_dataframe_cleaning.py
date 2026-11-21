
import pandas as pd
import pytest

from modules.structured_data import clean_dataframe


def test_empty_rows_are_removed():
    dataframe = pd.DataFrame({
        "Name": ["Rabbani", None, "John"],
        "Age": [25, None, 30],
    })

    result = clean_dataframe(dataframe)

    assert len(result) == 2


def test_column_names_are_normalized():
    dataframe = pd.DataFrame({
        "Employee Name": ["Rabbani"],
        "Department Name": ["IT"],
    })

    result = clean_dataframe(dataframe)

    assert "employee_name" in result.columns
    assert "department_name" in result.columns


def test_duplicate_column_names_are_unique():
    dataframe = pd.DataFrame(
        [["Rabbani", "IT"]],
        columns=["Department", "Department"],
    )

    result = clean_dataframe(dataframe)

    assert len(set(result.columns)) == len(result.columns)


def test_completely_empty_dataset_returns_empty_dataframe():
    dataframe = pd.DataFrame({
        "Name": [None, None],
        "Age": [None, None],
    })

    result = clean_dataframe(dataframe)

    assert result.empty


def test_whitespace_only_rows_are_removed():
    dataframe = pd.DataFrame({
        "Name": ["Rabbani", "   ", "John"],
        "Age": ["25", "   ", "30"],
    })

    result = clean_dataframe(dataframe)

    assert len(result) == 2
