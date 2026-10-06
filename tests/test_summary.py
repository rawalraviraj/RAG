import pytest
from tests.test_cases import TEST_CASES
from tests.evaluation_runner import run_single_test

def test_summary_queries():
    """Asserts summary questions execute under correct intent constraints."""
    summary_cases = [tc for tc in TEST_CASES if tc["category"] == "summary"]
    assert len(summary_cases) > 0
    
    tc = summary_cases[0]
    record, passed, reason = run_single_test(tc)
    assert passed, f"Summary query failed: {reason}"
    assert record["predicted_intent"] == "SUMMARY", f"Expected intent SUMMARY, got {record['predicted_intent']}"
