from modules.evaluation import simple_grounding_check

def test_keyword_check_full_match():
    result = simple_grounding_check("Employees receive 12 casual leave days.", ["casual leave", "12"])
    assert result["score"] == 1.0
    assert result["missing"] == []

def test_keyword_check_partial_match():
    result = simple_grounding_check("Employees receive leave.", ["leave", "18 days"])
    assert result["score"] == 0.5
    assert result["missing"] == ["18 days"]
