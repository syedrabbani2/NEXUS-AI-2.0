
import pytest
from pathlib import Path
from docx import Document as DocxDocument
from openpyxl import Workbook

from modules.document_processor import (
    get_file_hash,
    load_txt,
    load_csv,
    load_docx,
    load_xlsx,
    process_document,
)


def test_file_hash_is_stable(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("NEXUS AI testing", encoding="utf-8")

    first_hash = get_file_hash(file_path)
    second_hash = get_file_hash(file_path)

    assert first_hash == second_hash
    assert len(first_hash) == 16


def test_txt_loading(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("NEXUS AI document testing", encoding="utf-8")

    documents, count, label = load_txt(file_path, "sample.txt")

    assert count == 1
    assert label == "text file"
    assert "NEXUS AI" in documents[0].page_content
    assert documents[0].metadata["file_type"] == "txt"


def test_empty_txt_is_rejected(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_txt(file_path, "empty.txt")


def test_csv_loading(tmp_path):
    file_path = tmp_path / "employees.csv"
    file_path.write_text(
        "Name,Department\nRabbani,IT\nJohn,HR\n",
        encoding="utf-8",
    )

    documents, count, label = load_csv(file_path, "employees.csv")

    assert count == 2
    assert label == "rows"
    assert "Rabbani" in documents[0].page_content
    assert documents[0].metadata["row_number"] == 2


def test_invalid_csv_is_rejected(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("Name,Department\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_csv(file_path, "empty.csv")


def test_docx_loading(tmp_path):
    file_path = tmp_path / "sample.docx"

    document = DocxDocument()
    document.add_paragraph("NEXUS AI document testing")

    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Rabbani"

    document.save(file_path)

    documents, count, label = load_docx(file_path, "sample.docx")

    assert count >= 2
    assert label == "content sections"
    assert "NEXUS AI" in documents[0].page_content
    assert "Rabbani" in documents[0].page_content


def test_empty_docx_is_rejected(tmp_path):
    file_path = tmp_path / "empty.docx"

    document = DocxDocument()
    document.save(file_path)

    with pytest.raises(ValueError, match="No readable text"):
        load_docx(file_path, "empty.docx")


def test_xlsx_loading(tmp_path):
    file_path = tmp_path / "employees.xlsx"

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Name", "Department"])
    sheet.append(["Rabbani", "IT"])
    sheet.append(["John", "HR"])
    workbook.save(file_path)

    documents, count, label = load_xlsx(file_path, "employees.xlsx")

    assert count == 2
    assert label == "spreadsheet rows"
    assert "Rabbani" in documents[0].page_content
    assert documents[0].metadata["sheet_name"] == "Sheet"


def test_unsupported_file_format_is_rejected(tmp_path):
    file_path = tmp_path / "sample.exe"
    file_path.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        process_document(file_path, "sample.exe")


def test_document_processing_generates_chunks_and_ids(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "NEXUS AI knowledge assistant document processing test. " * 30,
        encoding="utf-8",
    )

    chunks, ids, count, label = process_document(
        file_path,
        "sample.txt",
    )

    assert len(chunks) > 0
    assert len(chunks) == len(ids)
    assert count == 1
    assert label == "text file"

    assert chunks[0].metadata["file_type"] == "txt"
    assert chunks[0].metadata["source"] == "sample.txt"
    assert chunks[0].metadata["file_id"]
    assert chunks[0].metadata["chunk_index"] == 0

    assert len(set(ids)) == len(ids)
