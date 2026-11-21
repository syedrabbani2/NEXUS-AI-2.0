
from modules.citation_engine import (
    tokenize,
    extract_claims,
    calculate_evidence_score,
    verify_claims,
    build_citation_report,
)


def test_tokenize_removes_stop_words():
    result = tokenize("The employee is eligible for annual leave")
    assert result == {"employee", "eligible", "annual", "leave"}


def test_tokenize_handles_punctuation():
    assert tokenize("NEXUS-AI, Version 2.0!") == {
        "nexus", "ai", "version", "2", "0"
    }


def test_extract_claims_splits_sentences():
    answer = (
        "Employees receive annual leave benefits. "
        "Managers approve leave requests."
    )
    claims = extract_claims(answer)
    assert len(claims) == 2


def test_extract_claims_ignores_short_sentences():
    assert extract_claims("Yes. Employees receive annual leave benefits.") == [
        "Employees receive annual leave benefits."
    ]


def test_extract_claims_handles_empty_input():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_evidence_score_full_match():
    score = calculate_evidence_score(
        "Employees receive annual leave",
        "Employees receive annual leave benefits"
    )
    assert score == 1.0


def test_evidence_score_partial_match():
    score = calculate_evidence_score(
        "Employees receive annual leave benefits",
        "Employees receive annual leave"
    )
    assert 0 < score < 1


def test_evidence_score_no_match():
    assert calculate_evidence_score(
        "Annual leave policy",
        "School transport management"
    ) == 0.0


def test_evidence_score_empty_content():
    assert calculate_evidence_score("", "Annual leave") == 0.0
    assert calculate_evidence_score("Annual leave", "") == 0.0


def test_verify_claims_identifies_candidate_support():
    claims = ["Employees receive annual leave benefits."]
    sources = [{
        "source_id": "S1",
        "source_name": "HR Policy",
        "content": "Employees receive annual leave benefits."
    }]

    result = verify_claims(claims, sources)

    assert result[0]["verification_status"] == "candidate_support"
    assert result[0]["best_evidence_score"] == 1.0
    assert result[0]["evidence_matches"][0]["source_id"] == "S1"


def test_verify_claims_identifies_insufficient_evidence():
    result = verify_claims(
        ["Employees receive annual leave benefits."],
        [{"content": "School transport management"}]
    )

    assert result[0]["verification_status"] == "insufficient_evidence"
    assert result[0]["best_evidence_score"] == 0.0


def test_verify_claims_uses_default_source_metadata():
    result = verify_claims(
        ["Employees receive annual leave benefits."],
        [{"content": "Employees receive annual leave benefits."}]
    )

    match = result[0]["evidence_matches"][0]

    assert match["source_id"] == "S1"
    assert match["source_name"] == "Unknown source"


def test_verify_claims_respects_custom_threshold():
    result = verify_claims(
        ["Employees receive annual leave benefits."],
        [{"content": "Employees receive annual leave"}],
        threshold=0.9
    )

    assert result[0]["verification_status"] == "insufficient_evidence"


def test_verify_claims_sorts_evidence_by_score():
    sources = [
        {"source_id": "S1", "content": "Employees receive annual leave"},
        {"source_id": "S2", "content": "Employees receive annual leave benefits"}
    ]

    result = verify_claims(
        ["Employees receive annual leave benefits."],
        sources
    )

    matches = result[0]["evidence_matches"]

    assert matches[0]["relevance_score"] >= matches[1]["relevance_score"]


def test_build_citation_report_returns_summary():
    answer = "Employees receive annual leave benefits."
    sources = [{
        "source_id": "S1",
        "source_name": "HR Policy",
        "content": "Employees receive annual leave benefits."
    }]

    report = build_citation_report(answer, sources)

    assert report["total_claims"] == 1
    assert report["total_sources"] == 1
    assert len(report["claims"]) == 1


def test_build_citation_report_handles_empty_answer():
    report = build_citation_report("", [])

    assert report["total_claims"] == 0
    assert report["total_sources"] == 0
    assert report["claims"] == []
