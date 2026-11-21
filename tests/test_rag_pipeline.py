
import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

import modules.rag_pipeline as rag


def make_document(content, metadata=None):
    return Document(
        page_content=content,
        metadata=metadata or {}
    )


def test_normalize_removes_according_to_prefix():
    result = rag.normalize_retrieval_question(
        "According to the uploaded HR policy, what is leave?"
    )
    assert result == "what is leave?"


def test_normalize_removes_based_on_prefix():
    result = rag.normalize_retrieval_question(
        "Based on the sample document: explain the process"
    )
    assert result == "explain the process"


def test_normalize_preserves_regular_question():
    question = "What is the leave policy?"
    assert rag.normalize_retrieval_question(question) == question


def test_normalize_handles_empty_input():
    assert rag.normalize_retrieval_question("   ") == ""


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="Please enter a question"):
        rag.answer_question_with_diagnostics(
            "", MagicMock(), "key", "model"
        )


def test_no_evidence_returns_fallback(monkeypatch):
    monkeypatch.setattr(
        rag,
        "multi_hop_search",
        lambda **kwargs: ([], {"initial_queries": []})
    )

    answer, sources, diagnostics = (
        rag.answer_question_with_diagnostics(
            "What is the policy?",
            MagicMock(),
            "key",
            "model"
        )
    )

    assert "couldn't find sufficient information" in answer
    assert sources == []
    assert diagnostics["evidence_sufficiency"] == "No evidence retrieved"


def test_retrieved_evidence_generates_sources_and_diagnostics(monkeypatch):
    doc = make_document(
        "Employees receive 20 days of annual leave.",
        {
            "source": "hr_policy.pdf",
            "source_type": "uploaded_document",
            "page_number": 3,
        }
    )

    monkeypatch.setattr(
        rag,
        "multi_hop_search",
        lambda **kwargs: ([doc], {"initial_queries": ["leave policy"]})
    )

    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = (
        "Employees receive 20 days of annual leave."
    )

    monkeypatch.setattr(rag, "ChatOpenAI", mock_llm)

    monkeypatch.setattr(
        rag,
        "build_citation_report",
        lambda **kwargs: {"claims": []}
    )

    answer, sources, diagnostics = (
        rag.answer_question_with_diagnostics(
            "How many leave days?",
            MagicMock(),
            "key",
            "model"
        )
    )

    assert "20 days" in answer
    assert len(sources) == 1
    assert sources[0]["file"] == "hr_policy.pdf"
    assert sources[0]["page"] == 3
    assert len(diagnostics["evidence"]) == 1
    assert diagnostics["unique_sources"] == 1
    assert "citation_report" in diagnostics


def test_duplicate_sources_are_removed(monkeypatch):
    first = make_document(
        "First leave reference.",
        {"source": "policy.pdf", "page_number": 2}
    )
    second = make_document(
        "Second leave reference.",
        {"source": "policy.pdf", "page_number": 2}
    )

    monkeypatch.setattr(
        rag,
        "multi_hop_search",
        lambda **kwargs: ([first, second], {})
    )

    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = "Leave information."

    monkeypatch.setattr(rag, "ChatOpenAI", mock_llm)
    monkeypatch.setattr(
        rag,
        "build_citation_report",
        lambda **kwargs: {}
    )

    _, sources, diagnostics = rag.answer_question_with_diagnostics(
        "Explain leave.",
        MagicMock(),
        "key",
        "model"
    )

    assert len(sources) == 1
    assert len(diagnostics["evidence"]) == 2


def test_webpage_source_metadata_is_preserved(monkeypatch):
    doc = make_document(
        "The website describes the service.",
        {
            "source": "https://example.com",
            "source_type": "web_page",
            "title": "Example Website",
            "url": "https://example.com",
        }
    )

    monkeypatch.setattr(
        rag,
        "multi_hop_search",
        lambda **kwargs: ([doc], {})
    )

    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = "Service information."

    monkeypatch.setattr(rag, "ChatOpenAI", mock_llm)
    monkeypatch.setattr(
        rag,
        "build_citation_report",
        lambda **kwargs: {}
    )

    _, sources, _ = rag.answer_question_with_diagnostics(
        "What does the website say?",
        MagicMock(),
        "key",
        "model"
    )

    assert sources[0]["source_type"] == "web_page"
    assert sources[0]["title"] == "Example Website"
    assert sources[0]["url"] == "https://example.com"


def test_backward_compatible_answer_returns_two_values(monkeypatch):
    monkeypatch.setattr(
        rag,
        "answer_question_with_diagnostics",
        lambda **kwargs: ("Answer", ["source"], {"diagnostics": True})
    )

    result = rag.answer_question(
        "Question",
        MagicMock(),
        "key",
        "model"
    )

    assert result == ("Answer", ["source"])
