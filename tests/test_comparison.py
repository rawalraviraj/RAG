import pytest
from tests.test_cases import TEST_CASES
from tests.evaluation_runner import run_single_test

def test_comparison_queries():
    """Asserts comparison questions generate markdown comparison tables."""
    comparison_cases = [tc for tc in TEST_CASES if tc["category"] == "comparison"]
    assert len(comparison_cases) > 0
    
    tc = comparison_cases[0]
    record, passed, reason = run_single_test(tc)
    assert passed, f"Comparison query failed: {reason}"
    assert "|" in record["answer"], "Answer was not formatted as a markdown table (missing '|' separator)"
