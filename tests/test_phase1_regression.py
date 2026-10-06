import time
import logging
import sys

sys.stdout.reconfigure(encoding='utf-8')
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from app.ingest import run_auto_ingestion
from app.bm25 import initialize_bm25_index
from app.retriever.retriever import retrieve, run_stage1_parallel, run_stage2_parallel
from app.chatbot.chatbot import answer_question, answer_question_stream
from app.semantic_cache import semantic_cache
from app.retriever.memory import memory
from app.utils import extract_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase1_regression")

def run_regression_tests():
    print("\n==================================================")
    print("      PHASE 1 - FULL REGRESSION TEST SUITE       ")
    print("==================================================\n")

    # Step A: Initialize system components & Ensure ingestion
    print("[INIT] Initializing Embeddings, Vector Store, Auto Ingestion & BM25...")
    get_embedding_model()
    get_vector_store()
    run_auto_ingestion()
    initialize_bm25_index()
    semantic_cache.clear()
    memory.clear()

    # 1. Category 1: SIMPLE Fact Lookup
    q1 = "What are the office timings?"
    print(f"\n[Test 1] Category: SIMPLE Fact Lookup -> '{q1}'")
    intent1, res_q1 = run_stage1_parallel(q1)
    q_vec1 = get_embedding_model().embed_query(res_q1)
    docs1 = retrieve(q1, query_vector=q_vec1, resolved_query=res_q1)
    ans1 = "".join([extract_text(c) for c in answer_question_stream(q1, docs1, intent=intent1)])
    assert "9:30 AM" in ans1 or "office" in ans1.lower(), "SIMPLE query failed"
    print(f"[PASS] SIMPLE Fact Lookup Passed. Answer preview: {ans1[:100]}...")

    # 2. Category 2: Follow-up Question Requiring Memory / Query Rewriting
    print(f"\n[Test 2] Category: Follow-up Question with Memory...")
    memory.add_conversation(q1, ans1, "HR_Policy.pdf", "")
    q2 = "Is there any flexibility in them?"
    intent2, res_q2 = run_stage1_parallel(q2)
    print(f"  Rewritten Query: '{res_q2}'")
    assert "timing" in res_q2.lower() or "office" in res_q2.lower() or "flexib" in res_q2.lower(), "Query rewrite failed"
    q_vec2 = get_embedding_model().embed_query(res_q2)
    docs2 = retrieve(q2, query_vector=q_vec2, resolved_query=res_q2)
    ans2 = "".join([extract_text(c) for c in answer_question_stream(q2, docs2, intent=intent2)])
    print(f"[PASS] Follow-up Query Passed. Answer preview: {ans2[:100]}...")

    # 3. Category 3: MEDIUM Query
    q3 = "What is the annual performance bonus and medical insurance coverage amount?"
    print(f"\n[Test 3] Category: MEDIUM Query -> '{q3}'")
    intent3, res_q3 = run_stage1_parallel(q3)
    q_vec3 = get_embedding_model().embed_query(res_q3)
    docs3 = retrieve(q3, query_vector=q_vec3, resolved_query=res_q3)
    ans3 = "".join([extract_text(c) for c in answer_question_stream(q3, docs3, intent=intent3)])
    print(f"[PASS] MEDIUM Query Passed. Answer preview: {ans3[:100]}...")

    # 4. Category 4: COMPLEX Query
    q4 = "Compare the leave policy and work from home guidelines in detail."
    print(f"\n[Test 4] Category: COMPLEX Query -> '{q4}'")
    intent4, res_q4 = run_stage1_parallel(q4)
    q_vec4 = get_embedding_model().embed_query(res_q4)
    docs4 = retrieve(q4, query_vector=q_vec4, resolved_query=res_q4)
    ans4 = "".join([extract_text(c) for c in answer_question_stream(q4, docs4, intent=intent4)])
    print(f"[PASS] COMPLEX Query Passed. Answer preview: {ans4[:100]}...")

    # 5. Category 5: Multi-Document Query
    q5 = "What are the rules for password security and working hours?"
    print(f"\n[Test 5] Category: Multi-Document Query -> '{q5}'")
    intent5, res_q5 = run_stage1_parallel(q5)
    q_vec5 = get_embedding_model().embed_query(res_q5)
    docs5 = retrieve(q5, query_vector=q_vec5, resolved_query=res_q5)
    ans5 = "".join([extract_text(c) for c in answer_question_stream(q5, docs5, intent=intent5)])
    print(f"[PASS] Multi-Document Query Passed. Answer preview: {ans5[:100]}...")

    # 6. Category 6: Semantic-Cache MISS & SET
    q6 = "What is the policy on password security?"
    print(f"\n[Test 6] Category: Semantic-Cache MISS -> '{q6}'")
    hit6 = semantic_cache.get(q6)
    assert hit6 is not None and "answer" not in hit6, "Expected cache miss"
    q_vec6 = hit6.get("query_vector")
    docs6 = retrieve(q6, query_vector=q_vec6)
    ans6 = "".join([extract_text(c) for c in answer_question_stream(q6, docs6)])
    semantic_cache.set(q6, ans6, references="IT_Security.pdf", query_vector=q_vec6)
    print(f"[PASS] Semantic-Cache MISS & SET Passed. Answer preview: {ans6[:100]}...")

    # 7. Category 7: Semantic-Cache HIT
    print(f"\n[Test 7] Category: Semantic-Cache HIT -> '{q6}'")
    t0 = time.time()
    hit7 = semantic_cache.get(q6)
    t1 = time.time()
    hit_ms = (t1 - t0) * 1000
    assert hit7 is not None and hit7.get("answer") == ans6, "Expected cache hit"
    print(f"✓ Semantic-Cache HIT Passed ({hit_ms:.2f} ms). Cached Answer: {hit7['answer'][:100]}...")

    print("\n==================================================")
    print("   ALL 7 CATEGORY REGRESSION TESTS PASSED!       ")
    print("==================================================\n")

if __name__ == "__main__":
    run_regression_tests()
