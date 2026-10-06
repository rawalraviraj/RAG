import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from app.chatbot.intent_classifier import classify_intent, local_classify_intent
from app.retriever.complexity_classifier import classify_complexity, local_classify_complexity

def test_simple_factual_question_classification():
    q = "What are the office timings?"
    intent = classify_intent(q)
    complexity = classify_complexity(q)
    
    assert intent == "FACT_LOOKUP"
    assert complexity == "SIMPLE"

def test_complex_question_classification():
    q = "Compare HR policy with finance policy."
    intent = classify_intent(q)
    complexity = classify_complexity(q)
    
    assert intent == "COMPARISON"
    assert complexity == "COMPLEX"

def test_yes_no_question_classification():
    q = "Does insurance include dental?"
    intent = classify_intent(q)
    complexity = classify_complexity(q)
    
    assert intent == "YES_NO"
    assert complexity == "SIMPLE"

def test_summary_question_classification():
    q = "Summarize HR Policy."
    intent = classify_intent(q)
    complexity = classify_complexity(q)
    
    assert intent == "SUMMARY"
    assert complexity == "COMPLEX"

def test_ambiguous_question_fallback():
    # Ambiguous query triggers local_classify -> None
    q = "X97z Q111 Alpha Beta Gamma Delta"
    loc_intent = local_classify_intent(q)
    loc_complexity = local_classify_complexity(q)
    
    assert loc_intent is None, "Ambiguous query should return None for local intent"
    assert loc_complexity is None, "Ambiguous query should return None for local complexity"

def test_valid_enum_outputs():
    valid_intents = {"FACT_LOOKUP", "SUMMARY", "COMPARISON", "LIST", "YES_NO", "EXPLANATION"}
    valid_complexities = {"SIMPLE", "MEDIUM", "COMPLEX"}
    
    queries = [
        "What is the medical insurance coverage amount?",
        "List all employee benefits.",
        "How to request work from home?",
        "Is flexible arrival allowed?"
    ]
    
    for q in queries:
        i = classify_intent(q)
        c = classify_complexity(q)
        assert i in valid_intents, f"Invalid intent output: {i}"
        assert c in valid_complexities, f"Invalid complexity output: {c}"
