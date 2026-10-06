import time
import sys
import logging
from typing import List, Dict, Any, Tuple
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

from app.retriever.retriever import retrieve
from app.chatbot.chatbot import answer_question
from app.chatbot.citation_tracker import track_and_filter_citations
from app.retriever.memory import memory
from tests.test_cases import TEST_CASES, CONVERSATION_MEMORY_TESTS, CONTEXT_SWITCHING_TESTS
from tests.metrics import evaluate_test_case
from tests.report_generator import generate_report

# Set up logging
logging.basicConfig(level=logging.ERROR)  # Suppress debug/info logs during runner printouts
logger = logging.getLogger(__name__)

def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 40) -> None:
    """Prints a custom visual character progress bar to the stdout console using safe ASCII."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '#' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

def run_single_test(test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
    """Runs a single test case query against the active RAG pipeline and records metrics."""
    query = test_case["query"]
    
    start_time = time.time()
    max_retries = 4
    
    for attempt in range(max_retries):
        try:
            # 1. Retrieve
            retrieved_chunks = retrieve(query)
            
            # Extract metadata properties from results
            predicted_complexity = getattr(retrieved_chunks, "complexity", "MEDIUM")
            actual_routed_docs = getattr(retrieved_chunks, "relevant_documents", [])
            confidence = getattr(retrieved_chunks, "confidence", 0.0)
            ratio = getattr(retrieved_chunks, "ratio", 0.0)
            resolved_query = getattr(retrieved_chunks, "resolved_query", query)
            
            # 2. Generate
            answer = answer_question(query, retrieved_chunks)
            predicted_intent = getattr(answer, "intent", "FACT_LOOKUP")
            verified = getattr(answer, "verified", True)
            unsupported = getattr(answer, "unsupported_sentences", [])
            
            # 3. Citation verify and references map
            references_str, used_citations, ignored_citations = track_and_filter_citations(answer, retrieved_chunks)
            
            # Successful run, break out of retry loop
            break
            
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str or "ratelimit" in err_str:
                wait_time = 45 * (attempt + 1)
                print(f"\n[Rate Limit 429 Hit] Sleeping for {wait_time}s to let tokens quota reset... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    else:
        raise Exception("Failed to execute test case query due to persistent rate limiting.")
        
    latency_ms = (time.time() - start_time) * 1000
    
    # 4. Evaluate assertions
    passed, reason = evaluate_test_case(
        test_case=test_case,
        predicted_intent=predicted_intent,
        predicted_complexity=predicted_complexity,
        actual_routed_docs=actual_routed_docs,
        retrieved_chunks=retrieved_chunks,
        answer=answer,
        references=references_str,
        used_citations=used_citations,
        ignored_citations=ignored_citations,
        verified=verified,
        confidence=confidence,
        latency_ms=latency_ms
    )
    
    record = {
        "query": query,
        "resolved_query": resolved_query,
        "category": test_case.get("category", "custom"),
        "expected_intent": test_case.get("expected_intent"),
        "predicted_intent": predicted_intent,
        "expected_complexity": test_case.get("expected_complexity"),
        "predicted_complexity": predicted_complexity,
        "expected_routed": test_case.get("expected_routed_docs", []),
        "actual_routed": actual_routed_docs,
        "retrieved_chunks_count": len(getattr(retrieved_chunks, "original_chunks", retrieved_chunks)),
        "compressed_chunks_count": len(retrieved_chunks),
        "compression_ratio": ratio,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "answer": str(answer),
        "references": references_str,
        "used_citations": used_citations,
        "ignored_citations": ignored_citations,
        "verified": verified,
        "unsupported_sentences": unsupported,
        "passed": passed,
        "reason": reason
    }
    
    return record, passed, reason

def run_evaluation(category_filter: str = None, single_query: str = None) -> Dict[str, Any]:
    """Orchestrates test pipeline iterations, collects statistics, and outputs reports."""
    # 1. Collect tests matching filter parameters
    run_cases = []
    
    if single_query:
        # Create ad-hoc test case query
        run_cases.append({
            "query": single_query,
            "category": "single_query",
            "expected_intent": None,
            "expected_complexity": None,
            "expected_routed_docs": [],
            "expected_keywords": [],
            "max_latency_ms": 10000,
            "min_confidence": 0.0,
            "is_conversation_sequence": False
        })
    elif category_filter:
        cat_lower = category_filter.lower()
        if cat_lower == "memory":
            # Map memory tests to sequences
            for seq in CONVERSATION_MEMORY_TESTS:
                run_cases.extend(seq)
        elif cat_lower == "switching":
            for seq in CONTEXT_SWITCHING_TESTS:
                run_cases.extend(seq)
        else:
            # Filter regular catalog test cases
            for tc in TEST_CASES:
                if tc["category"].lower() == cat_lower:
                    run_cases.append(tc)
    else:
        # Run entire suite: catalog + sequences
        run_cases.extend(TEST_CASES)
        for seq in CONVERSATION_MEMORY_TESTS:
            run_cases.extend(seq)
        for seq in CONTEXT_SWITCHING_TESTS:
            run_cases.extend(seq)
            
    if not run_cases:
        print(f"No test cases matched filter: category='{category_filter}', query='{single_query}'")
        return {}
        
    print(f"\nInitializing RAG Evaluation Suite...")
    print(f"Total Test Cases to run: {len(run_cases)}\n")
    
    results = []
    failed_cases = []
    
    passed_count = 0
    total_latency_ms = 0.0
    total_confidence = 0.0
    total_compression = 0.0
    
    intent_correct = 0
    intent_total = 0
    complexity_correct = 0
    complexity_total = 0
    routing_correct = 0
    routing_total = 0
    citation_correct = 0
    citation_total = 0
    answer_correct = 0
    answer_total = 0
    memory_correct = 0
    memory_total = 0
    
    # Latencies by complexity level
    latencies_by_complexity = {"SIMPLE": [], "MEDIUM": [], "COMPLEX": []}
    
    # Track sequence-related memory states
    is_in_sequence = False
    
    for idx, tc in enumerate(run_cases):
        print_progress_bar(idx, len(run_cases), prefix='Running Tests', suffix=f'Case {idx+1}/{len(run_cases)}', length=30)
        
        # Clear memory registry if transitioning to a new topic or standalone case
        if tc.get("should_be_standalone", False) or not is_in_sequence:
            memory.clear()
            
        if tc.get("is_conversation_sequence") or "resolved_query_keywords" in tc:
            is_in_sequence = True
        else:
            is_in_sequence = False
            
        record, passed, reason = run_single_test(tc)
        results.append(record)
        
        # Accumulate metrics
        total_latency_ms += record["latency_ms"]
        total_confidence += record["confidence"]
        total_compression += record["compression_ratio"]
        latencies_by_complexity.setdefault(record["predicted_complexity"], []).append(record["latency_ms"])
        
        # Increments accuracy counters
        if tc.get("expected_intent"):
            intent_total += 1
            if record["predicted_intent"] == tc["expected_intent"]:
                intent_correct += 1
                
        if tc.get("expected_complexity"):
            complexity_total += 1
            if record["predicted_complexity"] == tc["expected_complexity"]:
                complexity_correct += 1
                
        if tc.get("expected_routed_docs"):
            routing_total += 1
            actual_routed_set = set(record["actual_routed"])
            expected_routed_set = set(tc["expected_routed_docs"])
            # Routing is correct if all expected documents are in the actual documents routed
            if expected_routed_set.issubset(actual_routed_set) and (actual_routed_set == expected_routed_set or not expected_routed_set):
                routing_correct += 1
                
        # Citations correctness: all mapped footnote reference count matches inline numbers
        citation_total += 1
        if record["verified"] and "citation mismatch" not in record["reason"].lower() and "citations missing" not in record["reason"].lower():
            citation_correct += 1
            
        # Answer accuracy: contains keywords, no hallucinations
        answer_total += 1
        if record["verified"] and "keyword missing" not in record["reason"].lower():
            answer_correct += 1
            
        # Follow-up and memory correctness (if resolving query)
        if "resolved_query_keywords" in tc:
            memory_total += 1
            resolved_query_lower = record["resolved_query"].lower()
            if any(kw.lower() in resolved_query_lower for kw in tc["resolved_query_keywords"]):
                memory_correct += 1
                
        if passed:
            passed_count += 1
        else:
            # Map suggested fixes
            suggested_fix = "Check Qdrant payload fields or query mappings."
            if "intent" in reason.lower():
                suggested_fix = "Review IntentClassifier training examples or adjust intent prompt tags."
            elif "complexity" in reason.lower():
                suggested_fix = "Review ComplexityClassifier prompt configurations."
            elif "routing" in reason.lower():
                suggested_fix = "Check document loader path matches or adjust Router mappings."
            elif "keyword" in reason.lower():
                suggested_fix = "Verify retrieved document chunks contain the target answers."
            elif "citation" in reason.lower():
                suggested_fix = "Check inline citations parsing inside citation_tracker.py or verify LLM formats inline tags correctly."
            elif "confidence" in reason.lower():
                suggested_fix = "Retrieval scores are low. Check BM25 alignment or adjust threshold constraints."
            elif "latency" in reason.lower():
                suggested_fix = "Retrieve or generation calls took too long. Check Groq rate limits or cache models locally."
            elif "verification" in reason.lower():
                suggested_fix = "Hallucination guard rejected the answer. Verify context is rich enough to support the question."
                
            record["suggested_fix"] = suggested_fix
            failed_cases.append(record)
            
    print_progress_bar(len(run_cases), len(run_cases), prefix='Running Tests', suffix='Completed', length=30)
    
    # Calculate final aggregated scores
    total_count = len(run_cases)
    failed_count = total_count - passed_count
    
    intent_acc = (intent_correct / intent_total) if intent_total > 0 else 1.0
    complexity_acc = (complexity_correct / complexity_total) if complexity_total > 0 else 1.0
    routing_acc = (routing_correct / routing_total) if routing_total > 0 else 1.0
    citation_acc = (citation_correct / citation_total) if citation_total > 0 else 1.0
    answer_acc = (answer_correct / answer_total) if answer_total > 0 else 1.0
    memory_acc = (memory_correct / memory_total) if memory_total > 0 else 1.0
    
    avg_latency = total_latency_ms / total_count
    avg_confidence = total_confidence / total_count
    avg_compression = total_compression / total_count
    
    # Overall score combines layered accuracies
    overall_score = (intent_acc + complexity_acc + routing_acc + citation_acc + answer_acc + memory_acc) / 6.0
    if single_query:
        # For single query, overall is just passed status
        overall_score = 1.0 if passed_count == total_count else 0.0
        
    metrics_summary = {
        "total_count": total_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "overall_score": overall_score,
        "intent_accuracy": intent_acc,
        "complexity_accuracy": complexity_acc,
        "routing_accuracy": routing_acc,
        "citation_accuracy": citation_acc,
        "answer_accuracy": answer_acc,
        "memory_accuracy": memory_acc,
        "avg_latency_ms": avg_latency,
        "avg_confidence": avg_confidence,
        "avg_compression_ratio": avg_compression,
        # Averages by complexity level
        "latency_simple": sum(latencies_by_complexity["SIMPLE"]) / len(latencies_by_complexity["SIMPLE"]) if latencies_by_complexity["SIMPLE"] else 0.0,
        "latency_medium": sum(latencies_by_complexity["MEDIUM"]) / len(latencies_by_complexity["MEDIUM"]) if latencies_by_complexity["MEDIUM"] else 0.0,
        "latency_complex": sum(latencies_by_complexity["COMPLEX"]) / len(latencies_by_complexity["COMPLEX"]) if latencies_by_complexity["COMPLEX"] else 0.0
    }
    
    # 5. Generate output reports and charts
    generate_report(results, failed_cases, metrics_summary)
    
    # Print clean summary table to stdout console
    print("\n" + "=" * 50)
    print("EVALUATION RUN COMPLETE SUMMARY")
    print("=" * 50)
    print(f"Overall Score            : {overall_score * 100:.1f}%")
    print(f"Total / Passed / Failed  : {total_count} / {passed_count} / {failed_count}")
    print(f"Average Latency          : {avg_latency:.0f} ms")
    print(f"Average Confidence       : {avg_confidence * 100:.1f}%")
    print(f"Context Compression Ratio: {avg_compression * 100:.1f}%")
    print("-" * 50)
    print(f"Intent Classification    : {intent_acc * 100:.1f}%")
    print(f"Complexity Routing       : {complexity_acc * 100:.1f}%")
    print(f"Document Routing         : {routing_acc * 100:.1f}%")
    print(f"Citation Correctness     : {citation_acc * 100:.1f}%")
    print(f"Fact Grounding & Answers : {answer_acc * 100:.1f}%")
    if memory_total > 0:
        print(f"Conversational Memory    : {memory_acc * 100:.1f}%")
    print("=" * 50 + "\n")
    
    return metrics_summary
