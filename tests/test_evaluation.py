
import pytest

from modules.evaluation import simple_grounding_check


def test_empty_expected_terms():
    result = simple_grounding_check("Some answer", [])

    assert result == {
        "score": None,
        "matched": [],
        "missing": [],
    }


def test_all_terms_match():
    result = simple_grounding_check(
        "Python supports machine learning.",
        ["Python", "machine learning"],
    )

    assert result["score"] == 1.0
    assert result["matched"] == ["Python", "machine learning"]
    assert result["missing"] == []


def test_partial_term_match():
    result = simple_grounding_check(
        "Python is useful.",
        ["Python", "Java", "SQL"],
    )

    assert result["score"] == pytest.approx(1 / 3)
    assert result["matched"] == ["Python"]
    assert result["missing"] == ["Java", "SQL"]


def test_no_terms_match():
    result = simple_grounding_check(
        "This is a general answer.",
        ["Python", "Java"],
    )

    assert result["score"] == 0.0
    assert result["matched"] == []
    assert result["missing"] == ["Python", "Java"]


def test_matching_is_case_insensitive():
    result = simple_grounding_check(
        "PYTHON and machine learning",
        ["python", "MACHINE LEARNING"],
    )

    assert result["score"] == 1.0
    assert result["matched"] == ["python", "MACHINE LEARNING"]


def test_original_term_format_is_preserved():
    result = simple_grounding_check(
        "Python is powerful.",
        ["PYTHON", "Java"],
    )

    assert result["matched"] == ["PYTHON"]
    assert result["missing"] == ["Java"]
