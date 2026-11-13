def simple_grounding_check(answer: str, expected_terms: list[str]):
    """A transparent keyword smoke test, NOT a formal RAGAS metric."""
    if not expected_terms:
        return {"score": None, "matched": [], "missing": []}

    normalized = answer.lower()
    matched = [term for term in expected_terms if term.lower() in normalized]
    missing = [term for term in expected_terms if term.lower() not in normalized]
    score = len(matched) / len(expected_terms)
    return {"score": score, "matched": matched, "missing": missing}
