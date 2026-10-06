import pytest
from tests.test_cases import TEST_CASES
from tests.evaluation_runner import run_single_test

def test_confidence_thresholds():
    """Asserts that successful factual lookups have retrieval confidence scores above threshold."""
    lookup_cases = [tc for tc in TEST_CASES if tc["category"] == "lookup"]
    assert len(lookup_cases) > 0
    
    tc = lookup_cases[0]
    record, passed, reason = run_single_test(tc)
    assert passed, f"Lookup query failed: {reason}"
    assert record["confidence"] >= tc.get("min_confidence", 0.40), f"Confidence level ({record['confidence']:.2f}) below threshold"
