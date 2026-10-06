import logging
import json
import time
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

# pyrefly: ignore [missing-import]
from qdrant_client import models
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

from app.config import settings
from app.vector_store import get_vector_store
from app.bm25 import retrieve_bm25
from app.retriever.multi_query import generate_alternative_queries
from app.retriever.rrf import reciprocal_rank_fusion
from app.retriever.mmr import maximum_marginal_relevance
from app.retriever.confidence import compute_retrieval_confidence
from app.retriever.router import route_documents
from app.retriever.reranker import rerank_documents

# New modules integration
from app.retriever.memory import memory, resolve_query
from app.retriever.compressor import compress_chunks, count_tokens
from app.retriever.complexity_classifier import classify_complexity
from app.chatbot.intent_classifier import classify_intent

# Set up logging for the retriever module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_stage1_parallel(question: str, history=None) -> Tuple[str, str]:
    """
    Stage 1: Executes intent classification and query rewriting concurrently using ThreadPoolExecutor.
    Both operations depend solely on the raw user question.
    """
    if history is None:
        history = memory.get_history()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_intent = executor.submit(classify_intent, question)
        f_rewrite = executor.submit(resolve_query, question, history)
        intent = f_intent.result()
        resolved_query = f_rewrite.result()
    return intent, resolved_query

def run_stage2_parallel(resolved_query: str, target_documents: List[str] = None) -> Tuple[str, List[str]]:
    """
    Stage 2: Executes complexity classification and document routing concurrently using ThreadPoolExecutor.
    Both operations depend on the resolved query.
    """
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_complexity = executor.submit(classify_complexity, resolved_query)
        f_route = executor.submit(route_documents, resolved_query, candidate_filenames=target_documents)
        complexity = f_complexity.result()
        relevant_documents = f_route.result()
    t_stage2 = time.perf_counter() - t0
    logger.info(f"⏱️ Stage 2 (Complexity + Router) completed in {t_stage2*1000:.2f} ms")
    return complexity, relevant_documents

def retrieve(question: str, target_documents: List[str] = None, query_vector: List[float] = None, resolved_query: str = None, device: str = None) -> List[Document]:
    """
    Performs metadata-aware hybrid retrieval with query resolution (memory history),
    complexity classification, query expansion, RRF, MMR, and context compression.
    """
    # 1. Resolve Pronouns / Rewrite query based on Conversation Memory history if not pre-computed
    history = memory.get_history()
    if resolved_query is None:
        resolved_query = resolve_query(question, history)
    
    # 2. Parallel Stage 2: Complexity Classification + Document Routing
    complexity, relevant_documents = run_stage2_parallel(resolved_query, target_documents)
    
    # Configure dynamic Top-K and query expansion settings based on complexity
    if complexity == "SIMPLE":
        top_k = 3
        expansion_limit = 1
    elif complexity == "MEDIUM":
        top_k = 6
        expansion_limit = 3
    else:  # COMPLEX
        top_k = 10
        expansion_limit = 5
        
    logger.info(f"Complexity: {complexity} | Dynamic Top-K: {top_k} | Expansion Limit: {expansion_limit}")
    
    # Print "Selected Documents" before retrieval (Requirement 7)
    print("\nSelected Documents:")
    if relevant_documents:
        for doc in relevant_documents:
            print(doc)
    else:
        print("All Documents")
    print()
    
    # 4. Create Qdrant Filter representation
    qdrant_filter = None
    if relevant_documents:
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.filename",
                    match=models.MatchAny(any=relevant_documents)
                )
            ]
        )
        
    # 5. Multi-Query generation based on resolved query and dynamic limit
    if complexity == "SIMPLE":
        logger.info("Fast path active: skipping multi-query expansion LLM call for SIMPLE query.")
        queries = [resolved_query]
    else:
        queries = generate_alternative_queries(resolved_query, limit=expansion_limit)
        
    logger.info(f"Search queries: {queries}")
    
    vector_store = get_vector_store(device=device)
    all_runs: List[List[Document]] = []
    
    # Tracking counts for debugging
    chunks_per_document: Dict[str, int] = {}
    
    # 6. Retrieve documents using dense & sparse search on search queries
    t0_retrieval = time.perf_counter()
    for q in queries:
        # Dense search with Qdrant metadata filter
        t0_dense = time.perf_counter()
        if query_vector is not None and q == resolved_query:
            dense_results = vector_store.similarity_search_with_score_by_vector(query_vector, k=top_k, filter=qdrant_filter)
        else:
            dense_results = vector_store.similarity_search_with_score(q, k=top_k, filter=qdrant_filter)
        t_dense = time.perf_counter() - t0_dense
        dense_docs = [doc for doc, _ in dense_results]
        all_runs.append(dense_docs)
        logger.info(f"⏱️ Dense search (Qdrant) completed in {t_dense*1000:.2f} ms ({len(dense_docs)} chunks)")
        
        # Track dense candidate counts per document
        for doc in dense_docs:
            fn = doc.metadata.get("filename", "Unknown")
            chunks_per_document[fn] = chunks_per_document.get(fn, 0) + 1
        
        # BM25 sparse search with relevant documents subset filtering
        t0_bm25 = time.perf_counter()
        bm25_results = retrieve_bm25(q, top_k=top_k, relevant_documents=relevant_documents)
        t_bm25 = time.perf_counter() - t0_bm25
        bm25_docs = [doc for doc, _ in bm25_results]
        all_runs.append(bm25_docs)
        logger.info(f"⏱️ Sparse search (BM25) completed in {t_bm25*1000:.2f} ms ({len(bm25_docs)} chunks)")
        
        # Track BM25 candidate counts per document
        for doc in bm25_docs:
            fn = doc.metadata.get("filename", "Unknown")
            chunks_per_document[fn] = chunks_per_document.get(fn, 0) + 1
        
    # 7. Reciprocal Rank Fusion (RRF)
    t0_fusion = time.perf_counter()
    rrf_scored_docs = reciprocal_rank_fusion(all_runs)
    fused_candidates = [doc for doc, _ in rrf_scored_docs]
    
    # 8. Maximum Marginal Relevance (MMR) for diversification (Fetch more for reranker)
    final_selected = maximum_marginal_relevance(resolved_query, fused_candidates, top_k=top_k * 2)
    t_fusion = time.perf_counter() - t0_fusion
    logger.info(f"⏱️ RRF + MMR fusion completed in {t_fusion*1000:.2f} ms ({len(final_selected)} candidates)")
    
    # 8.5 Re-ranking with Cross-Encoder
    t0_rerank = time.perf_counter()
    logger.info(f"Reranking {len(final_selected)} chunks on device: {device or 'default'}...")
    reranked_docs = rerank_documents(resolved_query, final_selected, top_k=top_k, device=device)
    t_rerank = time.perf_counter() - t0_rerank
    logger.info(f"⏱️ CrossEncoder reranking completed in {t_rerank*1000:.2f} ms ({len(reranked_docs)} reranked)")
    
    # 9. Context Compression (extracting only the relevant facts)
    if complexity == "SIMPLE":
        logger.info("Fast path active: skipping context compression LLM call for SIMPLE query.")
        compressed_chunks = reranked_docs
    else:
        t0_comp = time.perf_counter()
        compressed_chunks = compress_chunks(reranked_docs, resolved_query, relevant_documents)
        t_comp = time.perf_counter() - t0_comp
        logger.info(f"⏱️ Context Compression completed in {t_comp*1000:.2f} ms")
    
    # 10. Compute retrieval confidence score based on the compressed chunks
    confidence_score = compute_retrieval_confidence(resolved_query, compressed_chunks)
    
    # Calculate token statistics for compression ratio
    orig_text = " ".join([d.page_content for d in final_selected])
    comp_text = " ".join([d.page_content for d in compressed_chunks])
    orig_tokens = count_tokens(orig_text)
    comp_tokens = count_tokens(comp_text)
    ratio = (orig_tokens - comp_tokens) / orig_tokens if orig_tokens > 0 else 0.0
    
    # 11. Print detailed retrieval logs to console (DEBUG MODE)
    print("\n=========================")
    print("Hybrid Retrieval (RRF + MMR + Compression)")
    print("=========================")
    print(f"Original Query: {question}")
    
    print("\nConversation Memory:")
    if history:
        for idx, turn in enumerate(history):
            print(f"  Turn {idx + 1}:")
            print(f"    User: {turn['question']}")
            print(f"    Bot: {turn['answer'][:60].strip()}...")
    else:
        print("  Empty")
        
    print(f"\nResolved Query: {resolved_query}")
    print(f"Complexity: {complexity}")
    print(f"Dynamic Top-K: {top_k}")
    print(f"Expansion Queries: {expansion_limit}")
    
    print("\nSelected Documents:")
    print(relevant_documents if relevant_documents else "All")
    
    print("\nMetadata Filter:")
    if qdrant_filter:
        print(json.dumps(qdrant_filter.model_dump(), indent=2))
    else:
        print("None")
        
    print("\nChunks Retrieved Per Document:")
    if chunks_per_document:
        for doc_name, count in chunks_per_document.items():
            print(f"  {doc_name}: {count} chunks")
    else:
        print("  None")
        
    print("\nCompressed Context:")
    if compressed_chunks:
        for idx, doc in enumerate(compressed_chunks):
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", 0) + 1
            print(f"  [{idx + 1}] Source: {source} (Page {page})")
            text_str = doc.page_content.strip()
            try:
                print(f"      Text: {text_str}")
            except UnicodeEncodeError:
                safe_text = text_str.encode('ascii', errors='replace').decode('ascii')
                print(f"      Text: {safe_text}")
    else:
        print("  None (All chunks discarded as irrelevant)")
        
    print(f"\nOriginal Tokens  : {orig_tokens}")
    print(f"Compressed Tokens: {comp_tokens}")
    print(f"Compression Ratio: {ratio * 100:.1f}%")
    
    # Print inline citation mappings
    print("\nInline Citation Mapping:")
    for idx, doc in enumerate(compressed_chunks):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", 0) + 1
        print(f"  [{idx + 1}] -> {source} (Page {page})")
        
    from app.chatbot.answer_formatter import get_answer_confidence_label
    print(f"\nAnswer Confidence: {get_answer_confidence_label(confidence_score)} ({confidence_score * 100:.1f}%)")
    print("=========================\n")
    
    # Store parameters on the return list object for main orchestrator
    class RetrievalResults(list):
        confidence: float = 0.0
        resolved_query: str = ""
        orig_tokens: int = 0
        comp_tokens: int = 0
        ratio: float = 0.0
        original_chunks: List[Document] = []
        complexity: str = ""
        top_k: int = 0
        expansion_limit: int = 0
        relevant_documents: List[str] = []
        
    results = RetrievalResults(compressed_chunks)
    results.confidence = confidence_score
    results.resolved_query = resolved_query
    results.orig_tokens = orig_tokens
    results.comp_tokens = comp_tokens
    results.ratio = ratio
    results.original_chunks = final_selected
    results.complexity = complexity
    results.top_k = top_k
    results.expansion_limit = expansion_limit
    results.relevant_documents = relevant_documents
    return results

if __name__ == "__main__":
    # Test question
    default_question = "What are the office timings?"
    try:
        user_input = input(f"Enter search question [default: {default_question}]: ").strip()
        question = user_input if user_input else default_question
    except (KeyboardInterrupt, EOFError):
        question = default_question
        print(f"\nUsing default question: {question}")
        
    print(f"Searching for: {question}")
    try:
        retrieved_docs = retrieve(question)
        print(f"Total retrieved docs: {len(retrieved_docs)}")
        print(f"Retrieved confidence: {retrieved_docs.confidence * 100:.1f}%")
    except Exception as e:
        logger.error(f"Retrieval test run failed: {e}")
