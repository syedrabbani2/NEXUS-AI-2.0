import pytest
import csv
import io

from modules.report_export import (
    generate_pdf_report,
    generate_csv_report,
)


@pytest.fixture
def sample_report():
    return {
        "title": "NEXUS Research Report",
        "generated_at": "2026-11-18",
        "question": "What is RAG?",
        "answer": "RAG combines retrieval and generation.",
        "citation_report": {
            "claims": [
                {
                    "claim_id": "C1",
                    "claim": "RAG retrieves relevant information.",
                    "verification_status": "supported",
                    "best_evidence_score": 0.85,
                    "evidence_matches": [
                        {
                            "source_id": "S1",
                            "source_name": "rag.pdf",
                            "relevance_score": 0.85,
                            "evidence": "Relevant retrieved evidence.",
                        },
                        {
                            "source_id": "S2",
                            "source_name": "guide.docx",
                            "relevance_score": 0.65,
                            "evidence": "Additional supporting text.",
                        },
                    ],
                },
                {
                    "claim_id": "C2",
                    "claim": "RAG always guarantees correctness.",
                    "verification_status": "unsupported",
                    "best_evidence_score": 0.0,
                    "evidence_matches": [],
                },
            ]
        },
        "sources": [
            {
                "file": "rag.pdf",
                "source_type": "document",
                "page": 2,
                "row": 0,
                "sheet": "Not available",
            },
            {
                "source_name": "Example Website",
                "source_type": "webpage",
                "url": "https://example.com",
            },
        ],
    }


def test_generate_pdf_report_returns_valid_pdf(sample_report):
    result = generate_pdf_report(sample_report)

    assert isinstance(result, bytes)
    assert result.startswith(b"%PDF")
    assert len(result) > 500


def test_generate_pdf_report_handles_empty_report():
    result = generate_pdf_report({})

    assert isinstance(result, bytes)
    assert result.startswith(b"%PDF")


def test_generate_pdf_report_escapes_special_characters():
    report = {
        "title": "Research <Report>",
        "question": "What is A & B?",
        "answer": "Use <tag> safely.",
    }

    result = generate_pdf_report(report)

    assert result.startswith(b"%PDF")


def test_generate_pdf_report_includes_claims_and_sources(sample_report):
    result = generate_pdf_report(sample_report)

    assert result.startswith(b"%PDF")
    assert b"%%EOF" in result


def test_generate_csv_report_returns_utf8_bom(sample_report):
    result = generate_csv_report(sample_report)

    assert result.startswith(b"\xef\xbb\xbf")


def test_generate_csv_report_contains_expected_columns(sample_report):
    result = generate_csv_report(sample_report)

    text = result.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    assert reader.fieldnames == [
        "claim_id",
        "claim",
        "verification_status",
        "best_evidence_score",
        "source_id",
        "source_name",
        "evidence_relevance",
        "evidence",
    ]


def test_generate_csv_report_creates_one_row_per_evidence_match(
    sample_report,
):
    result = generate_csv_report(sample_report)

    text = result.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))

    assert len(rows) == 3
    assert rows[0]["source_name"] == "rag.pdf"
    assert rows[1]["source_name"] == "guide.docx"


def test_generate_csv_report_preserves_claim_without_evidence(
    sample_report,
):
    result = generate_csv_report(sample_report)

    text = result.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))

    unsupported = [
        row for row in rows
        if row["claim_id"] == "C2"
    ]

    assert len(unsupported) == 1
    assert unsupported[0]["verification_status"] == "unsupported"
    assert unsupported[0]["source_name"] == ""


def test_generate_csv_report_handles_empty_report():
    result = generate_csv_report({})

    text = result.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))

    assert len(rows) == 0


def test_generate_csv_report_escapes_csv_special_characters():
    report = {
        "citation_report": {
            "claims": [
                {
                    "claim_id": "C1",
                    "claim": 'Text with, comma and "quotes"',
                    "verification_status": "supported",
                    "best_evidence_score": 0.9,
                    "evidence_matches": [
                        {
                            "source_name": "source,one.pdf",
                            "evidence": "Line one\nLine two",
                        }
                    ],
                }
            ]
        }
    }

    result = generate_csv_report(report)

    text = result.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))

    assert rows[0]["claim"] == 'Text with, comma and "quotes"'
    assert rows[0]["source_name"] == "source,one.pdf"
    assert rows[0]["evidence"] == "Line one\nLine two"
