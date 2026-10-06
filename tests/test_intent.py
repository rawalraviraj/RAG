import pytest
from tests.test_cases import TEST_CASES
from tests.evaluation_runner import run_single_test

def test_intent_classification():
    """Asserts that intent classification matches expectation for fact lookup."""
    lookup_cases = [tc for tc in TEST_CASES if tc["expected_intent"] == "FACT_LOOKUP"]
    assert len(lookup_cases) > 0
    # Test a representative case for fast verification
    tc = lookup_cases[0]
    record, passed, reason = run_single_test(tc)
    assert record["predicted_intent"] == "FACT_LOOKUP", f"Query '{tc['query']}' failed intent check: {reason}"
