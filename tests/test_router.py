import pytest
from tests.test_cases import TEST_CASES
from tests.evaluation_runner import run_single_test

def test_document_routing():
    """Asserts that metadata-aware document routing targets correct PDFs."""
    # Find simple routed lookups
    routed_cases = [tc for tc in TEST_CASES if tc["expected_routed_docs"]]
    assert len(routed_cases) > 0
    
    # Test routing on first lookup case
    tc = routed_cases[0]
    record, passed, reason = run_single_test(tc)
    for expected_doc in tc["expected_routed_docs"]:
        assert expected_doc in record["actual_routed"], f"Document routing failed: expected {expected_doc} in actual routed {record['actual_routed']}"
