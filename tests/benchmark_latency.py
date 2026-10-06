import time
import logging
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from app.ingest import run_auto_ingestion
from app.bm25 import initialize_bm25_index
from app.retriever.retriever import retrieve
from app.chatbot.chatbot import answer_question
from app.semantic_cache import semantic_cache

logging.basicConfig(level=logging.INFO)

import time
import logging
import torch
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from app.ingest import run_auto_ingestion
from app.bm25 import initialize_bm25_index
from app.retriever.retriever import retrieve, run_stage1_parallel, run_stage2_parallel
from app.chatbot.chatbot import answer_question, answer_question_stream
from app.chatbot.intent_classifier import classify_intent
from app.retriever.complexity_classifier import classify_complexity
from app.retriever.router import route_documents
from app.retriever.memory import memory, resolve_query
from app.retriever.reranker import get_reranker
from app.semantic_cache import semantic_cache
from app.utils import extract_text

logging.basicConfig(level=logging.INFO)

def benchmark():
    print("=== LATENCY BENCHMARK SUITE ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    print("Initializing system components...")
    get_embedding_model()
    get_vector_store()
    run_auto_ingestion()
    initialize_bm25_index()
    semantic_cache.clear()

    query = "What are the office timings?"
    target_docs = ["HR_Policy.pdf"]

    # Cold Start Run (Model weight initialization & warmup)
    print(f"\n[0] Cold Start Initialization Run for: '{query}'")
    t_cold_0 = time.time()
    docs_cold = retrieve(query, target_documents=target_docs)
    ans_cold = answer_question(query, docs_cold)
    t_cold_1 = time.time()
    cold_start_ms = (t_cold_1 - t_cold_0) * 1000
    print(f"Cold Start Initial Run Latency: {cold_start_ms:.2f} ms")
    semantic_cache.clear()

    # Stage-by-Stage Detailed Warm Pipeline Benchmark
    print(f"\n[1] Executing Detailed Warm RAG Pipeline Benchmark for: '{query}'")
    
    # Measure Stage 1 Parallel: Intent + Query Rewrite
    t0 = time.time()
    intent, resolved_q = run_stage1_parallel(query)
    t1 = time.time()
    stage1_ms = (t1 - t0) * 1000

    # Measure Stage 2 Parallel: Complexity + Router
    t0 = time.time()
    complexity, routed_docs = run_stage2_parallel(resolved_q, target_documents=target_docs)
    t1 = time.time()
    stage2_ms = (t1 - t0) * 1000

    # Measure Standalone Embedding
    emb_model = get_embedding_model()
    t0 = time.time()
    q_vec = emb_model.embed_query(query)
    t1 = time.time()
    embedding_ms = (t1 - t0) * 1000

    # Measure Reranker specifically
    reranker_model = get_reranker()
    from langchain_core.documents import Document
    sample_docs = [Document(page_content="Office timings are 9:30 AM to 6:30 PM.", metadata={"filename": "HR_Policy.pdf"})]
    t0 = time.time()
    pairs = [[query, doc.page_content] for doc in sample_docs]
    _ = reranker_model.predict(pairs)
    t1 = time.time()
    reranker_ms = (t1 - t0) * 1000

    # Measure Retrieval Phase reusing query vector and resolved query
    t0 = time.time()
    retrieved_chunks = retrieve(query, target_documents=target_docs, query_vector=q_vec, resolved_query=resolved_q)
    t1 = time.time()
    retrieval_ms = (t1 - t0) * 1000

    # Measure LLM Stream (TTFT and Total Generation Time) passing precomputed intent
    t0 = time.time()
    stream = answer_question_stream(query, retrieved_chunks, intent=intent)
    ttft_ms = 0.0
    tokens = []
    for chunk in stream:
        if ttft_ms == 0.0:
            ttft_ms = (time.time() - t0) * 1000
        text = extract_text(chunk)
        if text:
            tokens.append(text)
    t1 = time.time()
    llm_total_ms = (t1 - t0) * 1000
    final_answer = "".join(tokens)

    # Store in Redis Cache
    semantic_cache.set(query, final_answer, "HR_Policy.pdf", target_documents=target_docs, query_vector=q_vec)

    # Measure Cache Hit Latency
    t0 = time.time()
    cached = semantic_cache.get(query, target_documents=target_docs)
    t1 = time.time()
    cache_hit_ms = (t1 - t0) * 1000

    total_warm_ms = stage1_ms + stage2_ms + retrieval_ms + llm_total_ms

    print("\n================ LATENCY OPTIMIZED REPORT ================")
    print(f"• Cold-Start Run Latency : {cold_start_ms:.2f} ms ({cold_start_ms/1000:.2f} s)")
    print(f"• Cache HIT Latency      : {cache_hit_ms:.2f} ms")
    print(f"• Stage 1 (Parallel)     : {stage1_ms:.2f} ms")
    print(f"• Stage 2 (Parallel)     : {stage2_ms:.2f} ms")
    print(f"• Standalone Embedding    : {embedding_ms:.2f} ms")
    print(f"• Retrieval Pipeline     : {retrieval_ms:.2f} ms")
    print(f"• Standalone Reranker     : {reranker_ms:.2f} ms")
    print(f"• LLM TTFT               : {ttft_ms:.2f} ms")
    print(f"• LLM Generation Total    : {llm_total_ms:.2f} ms")
    print(f"• Total Warm Request      : {total_warm_ms:.2f} ms ({total_warm_ms/1000:.2f} s)")
    print("=========================================================")

if __name__ == "__main__":
    benchmark()
