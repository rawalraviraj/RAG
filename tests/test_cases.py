from typing import List, Dict, Any

# A detailed catalog of RAG pipeline evaluation test cases
# Latency thresholds are set to 25s / 35s to account for API rate-limit sleep intervals.
TEST_CASES: List[Dict[str, Any]] = [
    # Category 1: Simple Fact Lookup
    {
        "id": 1,
        "category": "lookup",
        "query": "What is the medical insurance amount?",
        "expected_intent": "FACT_LOOKUP",
        "expected_complexity": "SIMPLE",
        "expected_routed_docs": ["Insurance.pdf"],
        "expected_keywords": ["5,00,000"],
        "max_latency_ms": 25000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    {
        "id": 2,
        "category": "lookup",
        "query": "How many paid leaves are there?",
        "expected_intent": "FACT_LOOKUP",
        "expected_complexity": "SIMPLE",
        "expected_routed_docs": ["HR_Policy.pdf"],
        "expected_keywords": ["18"],
        "max_latency_ms": 25000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    {
        "id": 3,
        "category": "lookup",
        "query": "What is the travel reimbursement limit?",
        "expected_intent": "FACT_LOOKUP",
        "expected_complexity": "SIMPLE",
        "expected_routed_docs": ["Finance_Policy.pdf"],
        "expected_keywords": ["25,000"],
        "max_latency_ms": 25000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    {
        "id": 4,
        "category": "lookup",
        "query": "What is the internet reimbursement?",
        "expected_intent": "FACT_LOOKUP",
        "expected_complexity": "SIMPLE",
        "expected_routed_docs": ["Finance_Policy.pdf"],
        "expected_keywords": ["1,500", "internet"],  # 1500 matches document content
        "max_latency_ms": 25000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    
    # Category 2: Comparison Questions
    {
        "id": 5,
        "category": "comparison",
        "query": "Compare leave policy and insurance policy.",
        "expected_intent": "COMPARISON",
        "expected_complexity": "COMPLEX",
        "expected_routed_docs": ["HR_Policy.pdf", "Insurance.pdf"],
        "expected_keywords": ["leave", "insurance", "18", "5,00,000"],
        "max_latency_ms": 35000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    {
        "id": 6,
        "category": "comparison",
        "query": "Compare HR and Finance policies.",
        "expected_intent": "COMPARISON",
        "expected_complexity": "COMPLEX",
        "expected_routed_docs": ["HR_Policy.pdf", "Finance_Policy.pdf"],
        "expected_keywords": ["leave", "reimbursement", "salary"],
        "max_latency_ms": 35000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    
    # Category 3: Summary Questions
    {
        "id": 7,
        "category": "summary",
        "query": "Summarize HR Policy.",
        "expected_intent": "SUMMARY",
        "expected_complexity": "COMPLEX",
        "expected_routed_docs": ["HR_Policy.pdf"],
        "expected_keywords": ["leave", "leaves", "timings"],
        "max_latency_ms": 35000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    {
        "id": 8,
        "category": "summary",
        "query": "Summarize Finance Policy.",
        "expected_intent": "SUMMARY",
        "expected_complexity": "COMPLEX",
        "expected_routed_docs": ["Finance_Policy.pdf"],
        "expected_keywords": ["salary", "bonus", "travel"],
        "max_latency_ms": 35000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    
    # Category 4: List Questions
    {
        "id": 9,
        "category": "list",
        "query": "List all employee benefits.",
        "expected_intent": "LIST",
        "expected_complexity": "COMPLEX",
        "expected_routed_docs": ["HR_Policy.pdf", "Insurance.pdf"],  # Matches primary routed documents
        "expected_keywords": ["leaves", "insurance"],
        "max_latency_ms": 35000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    {
        "id": 10,
        "category": "list",
        "query": "List all leave types.",
        "expected_intent": "LIST",
        "expected_complexity": "COMPLEX",
        "expected_routed_docs": ["HR_Policy.pdf"],
        "expected_keywords": ["paid", "leave"],  # matches actual HR_Policy PDF leaf definitions
        "max_latency_ms": 35000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    
    # Category 5: Yes / No
    {
        "id": 11,
        "category": "yes_no",
        "query": "Does insurance include dental?",
        "expected_intent": "YES_NO",
        "expected_complexity": "SIMPLE",
        "expected_routed_docs": ["Insurance.pdf"],
        "expected_keywords": ["yes", "dental"],
        "max_latency_ms": 25000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    {
        "id": 12,
        "category": "yes_no",
        "query": "Can unused leave be carried forward?",
        "expected_intent": "YES_NO",
        "expected_complexity": "SIMPLE",
        "expected_routed_docs": ["HR_Policy.pdf"],
        "expected_keywords": ["yes", "carried", "forward", "10"],
        "max_latency_ms": 25000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    },
    {
        "id": 13,
        "category": "yes_no",
        "query": "Can spouse be covered?",
        "expected_intent": "YES_NO",
        "expected_complexity": "SIMPLE",
        "expected_routed_docs": ["Insurance.pdf"],
        "expected_keywords": ["yes", "spouse", "covered"],
        "max_latency_ms": 25000,
        "min_confidence": 0.40,
        "is_conversation_sequence": False
    }
]

# Category 6: Conversation Memory Sequences
CONVERSATION_MEMORY_TESTS: List[List[Dict[str, Any]]] = [
    [
        {
            "query": "What is the medical insurance amount?",
            "expected_intent": "FACT_LOOKUP",
            "expected_complexity": "SIMPLE",
            "expected_routed_docs": ["Insurance.pdf"],
            "expected_keywords": ["5,00,000"]
        },
        {
            "query": "Does it include dental?",
            "expected_intent": "YES_NO",
            "expected_complexity": "SIMPLE",
            "expected_routed_docs": ["Insurance.pdf"],
            "expected_keywords": ["yes", "dental"],
            "resolved_query_keywords": ["insurance", "dental"]
        },
        {
            "query": "Can my family be covered?",
            "expected_intent": "YES_NO",
            "expected_complexity": "SIMPLE",
            "expected_routed_docs": ["Insurance.pdf"],
            "expected_keywords": ["yes", "family"],
            "resolved_query_keywords": ["insurance", "family"]
        }
    ]
]

# Category 7: Context Switching Sequences
CONTEXT_SWITCHING_TESTS: List[List[Dict[str, Any]]] = [
    [
        {
            "query": "What is the medical insurance amount?",
            "expected_intent": "FACT_LOOKUP",
            "expected_complexity": "SIMPLE",
            "expected_routed_docs": ["Insurance.pdf"],
            "expected_keywords": ["5,00,000"]
        },
        {
            "query": "What is the leave policy?",
            "expected_intent": "FACT_LOOKUP",
            "expected_complexity": "SIMPLE",  # or MEDIUM depending on class routing
            "expected_routed_docs": ["HR_Policy.pdf"],
            "expected_keywords": ["18"],
            "resolved_query_keywords": ["leave", "policy"],
            "should_be_standalone": True
        },
        {
            "query": "What is the travel reimbursement limit?",
            "expected_intent": "FACT_LOOKUP",
            "expected_complexity": "SIMPLE",
            "expected_routed_docs": ["Finance_Policy.pdf"],
            "expected_keywords": ["25,000"],
            "resolved_query_keywords": ["travel", "reimbursement"],
            "should_be_standalone": True
        }
    ]
]
