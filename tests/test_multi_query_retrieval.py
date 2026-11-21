
import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

import modules.multi_query_retrieval as mq


def make_document(content, index=0):
    return Document(
        page_content=content,
        metadata={"source": "sample.txt", "chunk_index": index}
    )


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="Please enter a question"):
        mq.generate_query_variations("", "key", "model")


def test_original_question_is_preserved():
    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = (
        "employee leave policy"
    )
    # Alternative is the same as the original.
    monkey_question = "What is the leave policy?"

    from unittest.mock import patch
    with patch.object(mq, "ChatOpenAI", mock_llm):
        queries = mq.generate_query_variations(
            monkey_question, "key", "model"
        )

    assert queries[0] == monkey_question
    assert len(queries) == 2


def test_query_variations_remove_duplicates():
    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = (
        "leave policy\nLeave Policy\nannual leave rules"
    )

    from unittest.mock import patch
    with patch.object(mq, "ChatOpenAI", mock_llm):
        queries = mq.generate_query_variations(
            "What is the leave policy?", "key", "model"
        )

    assert len(queries) == 3
    assert queries[1] == "leave policy"
    assert queries[2] == "annual leave rules"


def test_query_variations_remove_numbering_and_bullets():
    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = (
        "1. annual leave rules\n- employee eligibility\n* leave balance"
    )

    from unittest.mock import patch
    with patch.object(mq, "ChatOpenAI", mock_llm):
        queries = mq.generate_query_variations(
            "What is the leave policy?", "key", "model",
            max_queries=3
        )

    assert queries[1:] == [
        "annual leave rules",
        "employee eligibility",
        "leave balance"
    ]


def test_query_variations_respect_max_queries():
    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = (
        "query one\nquery two\nquery three\nquery four"
    )

    from unittest.mock import patch
    with patch.object(mq, "ChatOpenAI", mock_llm):
        queries = mq.generate_query_variations(
            "Original question", "key", "model", max_queries=2
        )

    assert len(queries) == 3


def test_query_variations_support_list_response():
    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = [
        {"text": "annual leave rules"},
        {"text": "employee eligibility"}
    ]

    from unittest.mock import patch
    with patch.object(mq, "ChatOpenAI", mock_llm):
        queries = mq.generate_query_variations(
            "What is leave?", "key", "model"
        )

    assert "annual leave rules" in queries
    assert "employee eligibility" in queries


def test_query_generation_failure_preserves_original():
    mock_llm = MagicMock()
    mock_llm.return_value.invoke.side_effect = RuntimeError("API error")

    from unittest.mock import patch
    with patch.object(mq, "ChatOpenAI", mock_llm):
        queries = mq.generate_query_variations(
            "What is leave?", "key", "model"
        )

    assert queries == ["What is leave?"]


def test_multi_query_fuses_results(monkeypatch):
    first = make_document("Leave policy", 1)
    second = make_document("Annual leave eligibility", 2)

    monkeypatch.setattr(
        mq,
        "generate_query_variations",
        lambda **kwargs: ["leave policy", "annual leave"]
    )

    search_calls = []

    def fake_hybrid_search(**kwargs):
        search_calls.append(kwargs["query"])
        return [first] if kwargs["query"] == "leave policy" else [second]

    monkeypatch.setattr(mq, "hybrid_search", fake_hybrid_search)

    monkeypatch.setattr(
        mq,
        "reciprocal_rank_fusion",
        lambda **kwargs: [first, second]
    )

    results, diagnostics = mq.multi_query_search(
        "What is leave?",
        MagicMock(),
        "key",
        "model",
        return_diagnostics=True
    )

    assert results == [first, second]
    assert search_calls == ["leave policy", "annual leave"]
    assert diagnostics["query_count"] == 2
    assert diagnostics["retrieved_chunks"] == 2


def test_multi_query_returns_empty_when_no_results(monkeypatch):
    monkeypatch.setattr(
        mq,
        "generate_query_variations",
        lambda **kwargs: ["leave policy"]
    )
    monkeypatch.setattr(
        mq,
        "hybrid_search",
        lambda **kwargs: []
    )

    results, diagnostics = mq.multi_query_search(
        "What is leave?",
        MagicMock(),
        "key",
        "model",
        return_diagnostics=True
    )

    assert results == []
    assert diagnostics["retrieved_chunks"] == 0
    assert diagnostics["queries"][0]["results_found"] == 0


def test_multi_query_without_diagnostics_returns_documents(monkeypatch):
    doc = make_document("Leave policy")

    monkeypatch.setattr(
        mq,
        "generate_query_variations",
        lambda **kwargs: ["leave policy"]
    )
    monkeypatch.setattr(
        mq,
        "hybrid_search",
        lambda **kwargs: [doc]
    )
    monkeypatch.setattr(
        mq,
        "reciprocal_rank_fusion",
        lambda **kwargs: [doc]
    )

    result = mq.multi_query_search(
        "What is leave?",
        MagicMock(),
        "key",
        "model"
    )

    assert result == [doc]
