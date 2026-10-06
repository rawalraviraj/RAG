import pytest
from app.retriever.memory import memory
from tests.test_cases import CONVERSATION_MEMORY_TESTS
from tests.evaluation_runner import run_single_test

def test_conversational_memory():
    """Asserts that conversational query resolution decodes pronoun context."""
    memory.clear()
    assert len(CONVERSATION_MEMORY_TESTS) > 0
    seq = CONVERSATION_MEMORY_TESTS[0]
    
    # Run Turn 1 (Initial topic establishment)
    rec1, p1, r1 = run_single_test(seq[0])
    assert p1, f"Turn 1 failed: {r1}"
    memory.add_conversation(seq[0]["query"], rec1["answer"], rec1["references"], "")
    
    # Run Turn 2 (Pronoun reference query)
    rec2, p2, r2 = run_single_test(seq[1])
    assert p2, f"Turn 2 failed: {r2}"
    assert "insurance" in rec2["resolved_query"].lower(), f"Memory query resolution failed. Query resolved to: '{rec2['resolved_query']}'"
