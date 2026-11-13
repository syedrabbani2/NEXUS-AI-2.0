
from pathlib import Path
import hashlib
import csv

from docx import Document as DocxDocument
from openpyxl import load_workbook

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_file_hash(file_path):
    """Generate a stable hash for the uploaded file."""
    return hashlib.sha256(
        Path(file_path).read_bytes()
    ).hexdigest()[:16]

def load_pdf(file_path, original_name):
    pages = PyPDFLoader(str(file_path)).load()

    documents = []

    for page in pages:
        text = (page.page_content or "").strip()

        if not text:
            continue

        page_number = int(page.metadata.get("page", 0)) + 1

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": original_name,
                    "file_type": "pdf",
                    "page_number": page_number
                }
            )
        )

    return documents, len(documents), "pages"

def load_docx(file_path, original_name):
    word_document = DocxDocument(str(file_path))

    content = []

    for paragraph in word_document.paragraphs:
        text = paragraph.text.strip()

        if text:
            content.append(text)

    # Extract Word tables as readable text.
    for table_index, table in enumerate(word_document.tables, start=1):
        for row_index, row in enumerate(table.rows, start=1):
            values = [
                cell.text.strip().replace("\n", " ")
                for cell in row.cells
            ]

            if any(values):
                content.append(
                    f"Table {table_index}, Row {row_index}: "
                    + " | ".join(values)
                )

    if not content:
        raise ValueError("No readable text found in this DOCX file.")

    text = "\n\n".join(content)

    documents = [
        Document(
            page_content=text,
            metadata={
                "source": original_name,
                "file_type": "docx"
            }
        )
    ]

    return documents, len(content), "content sections"

def load_txt(file_path, original_name):
    path = Path(file_path)

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252")

    text = text.strip()

    if not text:
        raise ValueError("The TXT file is empty.")

    documents = [
        Document(
            page_content=text,
            metadata={
                "source": original_name,
                "file_type": "txt"
            }
        )
    ]

    return documents, 1, "text file"

def load_csv(file_path, original_name):
    path = Path(file_path)

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file))
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252", newline="") as file:
            rows = list(csv.reader(file))

    if len(rows) < 2:
        raise ValueError("CSV must contain a header and at least one data row.")

    headers = [
        str(value).strip() if value else f"Column {index + 1}"
        for index, value in enumerate(rows[0])
    ]

    documents = []

    for row_number, row in enumerate(rows[1:], start=2):
        values = []

        for index, value in enumerate(row):
            if index < len(headers):
                header = headers[index]
            else:
                header = f"Column {index + 1}"

            if str(value).strip():
                values.append(f"{header}: {value.strip()}")

        if not values:
            continue

        documents.append(
            Document(
                page_content="\n".join(values),
                metadata={
                    "source": original_name,
                    "file_type": "csv",
                    "row_number": row_number
                }
            )
        )

    if not documents:
        raise ValueError("No readable data rows found in CSV.")

    return documents, len(documents), "rows"

def load_xlsx(file_path, original_name):
    workbook = load_workbook(
        filename=str(file_path),
        read_only=True,
        data_only=True
    )

    documents = []
    total_rows = 0

    try:
        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(values_only=True)

            headers = next(rows, None)

            if not headers:
                continue

            headers = [
                str(value).strip() if value is not None
                else f"Column {index + 1}"
                for index, value in enumerate(headers)
            ]

            for row_number, row in enumerate(rows, start=2):
                values = []

                for index, value in enumerate(row):
                    if value is None or str(value).strip() == "":
                        continue

                    header = (
                        headers[index]
                        if index < len(headers)
                        else f"Column {index + 1}"
                    )

                    values.append(f"{header}: {value}")

                if not values:
                    continue

                documents.append(
                    Document(
                        page_content="\n".join(values),
                        metadata={
                            "source": original_name,
                            "file_type": "xlsx",
                            "sheet_name": sheet.title,
                            "row_number": row_number
                        }
                    )
                )

                total_rows += 1

    finally:
        workbook.close()

    if not documents:
        raise ValueError("No readable data rows found in XLSX.")

    return documents, total_rows, "spreadsheet rows"

def process_document(file_path, original_name):
    """Process supported documents and generate searchable chunks."""

    path = Path(file_path)
    extension = path.suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
        ".csv": load_csv,
        ".xlsx": load_xlsx
    }

    if extension not in loaders:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    documents, unit_count, unit_label = loaders[extension](
        path,
        original_name
    )

    if not documents:
        raise ValueError("No readable content found in this document.")

    file_hash = get_file_hash(path)

    for document in documents:
        document.metadata["file_id"] = file_hash

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    ids = []

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        ids.append(f"{file_hash}-{index}")

    if not chunks:
        raise ValueError("Document processing generated no text chunks.")

    return chunks, ids, unit_count, unit_label

def process_pdf(file_path, original_name):
    """Backward-compatible PDF processing function."""

    chunks, ids, page_count, _ = process_document(
        file_path,
        original_name
    )

    return chunks, ids, page_count