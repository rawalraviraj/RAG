import pytest
from app.semantic_cache import SemanticCache, normalize_query, get_kb_identifier

@pytest.fixture(scope="module")
def cache():
    sc = SemanticCache(threshold=0.82, ttl=300)
    sc.clear()
    yield sc
    sc.clear()

def test_normalization():
    assert normalize_query("  What ARE the office timings?!  ") == "what are the office timings"
    assert normalize_query("What   is   the dress code???") == "what is the dress code"
    assert normalize_query("Password policy.") == "password policy"

def test_kb_identifier():
    assert get_kb_identifier(["HR_Policy.pdf"]) == "HR_Policy.pdf"
    assert get_kb_identifier(["IT_Security.pdf", "HR_Policy.pdf"]) == "HR_Policy.pdf,IT_Security.pdf"
    assert get_kb_identifier([]) == "default"
    assert get_kb_identifier(None) == "default"

def test_exact_same_query(cache):
    cache.clear()
    query = "What are the office timings?"
    answer = "The standard office timings are 9:00 AM to 6:00 PM."
    cache.set(query, answer, target_documents=["HR_Policy.pdf"])

    hit = cache.get(query, target_documents=["HR_Policy.pdf"])
    assert hit is not None
    assert hit["answer"] == answer
    assert hit["similarity"] >= 0.99

def test_punctuation_differences(cache):
    cache.clear()
    query1 = "Is there any dress code policy?"
    query2 = "Is there any dress code policy!!!"
    answer = "Employees are expected to dress in smart casual attire."
    cache.set(query1, answer, target_documents=["HR_Policy.pdf"])

    hit = cache.get(query2, target_documents=["HR_Policy.pdf"])
    assert hit is not None
    assert hit["answer"] == answer

def test_capitalization_differences(cache):
    cache.clear()
    query1 = "what is the annual performance bonus"
    query2 = "WHAT IS THE ANNUAL PERFORMANCE BONUS?"
    answer = "Annual bonus is based on individual and company metrics."
    cache.set(query1, answer, target_documents=["HR_Policy.pdf"])

    hit = cache.get(query2, target_documents=["HR_Policy.pdf"])
    assert hit is not None
    assert hit["answer"] == answer

def test_whitespace_differences(cache):
    cache.clear()
    query1 = "What are the office timings?"
    query2 = "What   are   the    office    timings  ? "
    answer = "Office hours are from 9 AM to 6 PM."
    cache.set(query1, answer, target_documents=["HR_Policy.pdf"])

    hit = cache.get(query2, target_documents=["HR_Policy.pdf"])
    assert hit is not None
    assert hit["answer"] == answer

def test_minor_typos(cache):
    cache.clear()
    query1 = "What is the policy on password security?"
    query2 = "What is the policy on password security"
    answer = "Passwords must be at least 12 characters long."
    cache.set(query1, answer, target_documents=["IT_Security.pdf"])

    hit = cache.get(query2, target_documents=["IT_Security.pdf"], threshold=0.75)
    assert hit is not None
    assert hit["answer"] == answer

def test_semantically_equivalent_questions(cache):
    cache.clear()
    query1 = "What is the dress code policy?"
    query2 = "Is there any dress code policy?"
    answer = "Smart casual attire."
    cache.set(query1, answer, target_documents=["HR_Policy.pdf"])

    hit = cache.get(query2, target_documents=["HR_Policy.pdf"], threshold=0.75)
    assert hit is not None
    assert hit["answer"] == answer

def test_genuinely_different_questions(cache):
    cache.clear()
    query1 = "What are the office timings?"
    query2 = "How do I request a reimbursement for medical expenses?"
    answer = "Office hours are 9 AM to 6 PM."
    cache.set(query1, answer, target_documents=["HR_Policy.pdf"])

    hit = cache.get(query2, target_documents=["HR_Policy.pdf"])
    assert hit is None or "answer" not in hit

def test_same_question_different_knowledge_bases(cache):
    cache.clear()
    query = "What is the security policy?"
    answer_hr = "HR physical security policy requires wearing badges."
    answer_it = "IT security policy requires 2FA authentication."

    cache.set(query, answer_hr, target_documents=["HR_Policy.pdf"])
    cache.set(query, answer_it, target_documents=["IT_Security.pdf"])

    hit_hr = cache.get(query, target_documents=["HR_Policy.pdf"])
    hit_it = cache.get(query, target_documents=["IT_Security.pdf"])

    assert hit_hr is not None and hit_hr["answer"] == answer_hr
    assert hit_it is not None and hit_it["answer"] == answer_it
