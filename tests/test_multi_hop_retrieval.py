
import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

import modules.multi_hop_retrieval as mh


def make_document(content, index=0):
    return Document(
        page_content=content,
        metadata={"source": "sample.txt", "chunk_index": index}
    )


def test_response_text_converts_string():
    response = MagicMock()
    response.content = "NEXUS AI"
    assert mh._response_text(response) == "NEXUS AI"


def test_response_text_converts_list():
    response = MagicMock()
    response.content = [
        {"text": "NEXUS"},
        {"text": "AI"}
    ]
    assert mh._response_text(response) == "NEXUS\nAI"


def test_followup_returns_none_for_empty_question():
    assert mh.generate_followup_query("", [make_document("Evidence")], "key", "model") is None


def test_followup_returns_none_without_documents():
    assert mh.generate_followup_query("What is the policy?", [], "key", "model") is None


def test_followup_returns_none_for_empty_document_content():
    assert mh.generate_followup_query(
        "What is the policy?",
        [make_document("   ")],
        "key",
        "model"
    ) is None


def test_followup_generates_query(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = "employee eligibility"

    monkeypatch.setattr(mh, "ChatOpenAI", mock_llm)

    result = mh.generate_followup_query(
        "What is the leave policy?",
        [make_document("Employees receive annual leave.")],
        "key",
        "model"
    )

    assert result == "employee eligibility"


@pytest.mark.parametrize("response", ["NONE", "N/A", "NO QUERY"])
def test_followup_rejects_no_query_responses(monkeypatch, response):
    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = response

    monkeypatch.setattr(mh, "ChatOpenAI", mock_llm)

    result = mh.generate_followup_query(
        "What is the policy?",
        [make_document("Policy evidence.")],
        "key",
        "model"
    )

    assert result is None


def test_followup_rejects_original_question(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.return_value.invoke.return_value.content = "What is the policy?"

    monkeypatch.setattr(mh, "ChatOpenAI", mock_llm)

    result = mh.generate_followup_query(
        "What is the policy?",
        [make_document("Policy evidence.")],
        "key",
        "model"
    )

    assert result is None


def test_followup_handles_llm_failure(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.return_value.invoke.side_effect = RuntimeError("API failure")

    monkeypatch.setattr(mh, "ChatOpenAI", mock_llm)

    result = mh.generate_followup_query(
        "What is the policy?",
        [make_document("Policy evidence.")],
        "key",
        "model"
    )

    assert result is None


def test_multi_hop_empty_question_with_diagnostics():
    results, diagnostics = mh.multi_hop_search(
        "",
        MagicMock(),
        "key",
        "model",
        return_diagnostics=True
    )

    assert results == []
    assert diagnostics["hops_completed"] == 0
    assert diagnostics["followup"]["attempted"] is False


def test_multi_hop_rejects_invalid_limits():
    with pytest.raises(ValueError, match="top_k and fetch_k"):
        mh.multi_hop_search(
            "Question", MagicMock(), "key", "model", top_k=0
        )

    with pytest.raises(ValueError, match="max_hops"):
        mh.multi_hop_search(
            "Question", MagicMock(), "key", "model", max_hops=-1
        )


def test_multi_hop_returns_initial_results_when_no_followup(monkeypatch):
    initial = [make_document("Initial evidence.")]

    monkeypatch.setattr(
        mh,
        "multi_query_search",
        lambda **kwargs: (initial, {"query_count": 2})
    )

    monkeypatch.setattr(
        mh,
        "generate_followup_query",
        lambda **kwargs: None
    )

    results, diagnostics = mh.multi_hop_search(
        "Question",
        MagicMock(),
        "key",
        "model",
        return_diagnostics=True
    )

    assert results == initial
    assert diagnostics["hops_completed"] == 0
    assert diagnostics["followup"]["attempted"] is False


def test_multi_hop_skips_followup_when_max_hops_zero(monkeypatch):
    initial = [make_document("Initial evidence.")]

    monkeypatch.setattr(
        mh,
        "multi_query_search",
        lambda **kwargs: (initial, {})
    )

    def should_not_run(**kwargs):
        pytest.fail("Follow-up generation should not run")

    monkeypatch.setattr(mh, "generate_followup_query", should_not_run)

    results = mh.multi_hop_search(
        "Question",
        MagicMock(),
        "key",
        "model",
        max_hops=0
    )

    assert results == initial


def test_multi_hop_fuses_followup_results(monkeypatch):
    initial = [make_document("Initial evidence.", 1)]
    followup = [make_document("Follow-up evidence.", 2)]
    fused = [initial[0], followup[0]]

    monkeypatch.setattr(
        mh,
        "multi_query_search",
        lambda **kwargs: (initial, {"query_count": 2})
    )

    monkeypatch.setattr(
        mh,
        "generate_followup_query",
        lambda **kwargs: "related evidence"
    )

    monkeypatch.setattr(
        mh,
        "hybrid_search",
        lambda **kwargs: followup
    )

    monkeypatch.setattr(
        mh,
        "reciprocal_rank_fusion",
        lambda **kwargs: fused
    )

    results, diagnostics = mh.multi_hop_search(
        "Question",
        MagicMock(),
        "key",
        "model",
        top_k=4,
        return_diagnostics=True
    )

    assert results == fused
    assert diagnostics["followup"]["attempted"] is True
    assert diagnostics["followup"]["query"] == "related evidence"
    assert diagnostics["followup"]["results_found"] == 1
    assert diagnostics["hops_completed"] == 1


def test_multi_hop_preserves_initial_results_when_followup_empty(monkeypatch):
    initial = [make_document("Initial evidence.")]

    monkeypatch.setattr(
        mh,
        "multi_query_search",
        lambda **kwargs: (initial, {})
    )
    monkeypatch.setattr(
        mh,
        "generate_followup_query",
        lambda **kwargs: "follow-up query"
    )
    monkeypatch.setattr(
        mh,
        "hybrid_search",
        lambda **kwargs: []
    )

    results = mh.multi_hop_search(
        "Question", MagicMock(), "key", "model"
    )

    assert results == initial


def test_multi_hop_supports_plain_initial_results(monkeypatch):
    initial = [make_document("Initial evidence.")]

    monkeypatch.setattr(
        mh,
        "multi_query_search",
        lambda **kwargs: initial
    )

    monkeypatch.setattr(
        mh,
        "generate_followup_query",
        lambda **kwargs: None
    )

    results, diagnostics = mh.multi_hop_search(
        "Question",
        MagicMock(),
        "key",
        "model",
        return_diagnostics=True
    )

    assert results == initial
    assert diagnostics["initial"]["retrieved_chunks"] == 1
