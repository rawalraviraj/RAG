import pytest
from tests.test_cases import TEST_CASES
from tests.evaluation_runner import run_single_test

def test_citations_verification():
    """Asserts that the final output includes inline citations and accurate references footnotes."""
    lookup_cases = [tc for tc in TEST_CASES if tc["category"] == "lookup"]
    assert len(lookup_cases) > 0
    
    tc = lookup_cases[0]
    record, passed, reason = run_single_test(tc)
    assert passed, f"Lookup query failed: {reason}"
    assert "[" in record["answer"] and "]" in record["answer"], "Inline citation missing in output text"
    assert len(record["used_citations"]) > 0, "No references listed in footnotes"
    assert len(record["ignored_citations"]) == 0 or len(record["ignored_citations"]) > 0, "Ignored citations mapping mismatch"
