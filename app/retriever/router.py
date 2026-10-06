import json
import re
import logging
from typing import List, Any
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from app.config import settings
from app.utils import extract_text
from app.ingest import scan_data_folder

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """You are a highly precise document routing assistant.
Your job is to identify which documents from the list of available documents are relevant to answer the user's question.

Available documents:
{available_docs}

User Question:
{question}

Instructions:
- Return ONLY a JSON object containing a list of relevant filenames under the key "documents".
- If the question is general, comparative, or you are uncertain which document is relevant, return all available documents.
- Provide NO other text, explanations, markdown wrappers, or formatting. Return raw JSON only.

Example Output format:
{{
  "documents": ["Insurance.pdf"]
}}"""

def keyword_routing_fallback(query: str, available_filenames: List[str]) -> List[str]:
    """
    Fallback keyword-based mapping logic if LLM routing fails or returns empty results.
    """
    query_lower = query.lower()
    
    # Map keywords to target search sub-strings
    mapping = {
        "insurance": ["insurance", "medical", "hospital", "health"],
        "hr_policy": ["leave", "vacation", "holiday", "policy"],
        "finance_policy": ["salary", "bonus", "tax", "travel", "reimbursement", "finance"],
        "python_guide": ["python", "pip", "class", "function", "programming"],
        "employee_handbook_sample": ["employee", "office", "password", "conduct", "helpdesk"]
    }
    
    selected = []
    for filename in available_filenames:
        fn_lower = filename.lower()
        for target_key, keywords in mapping.items():
            # If the filename contains the primary mapping key
            if target_key in fn_lower:
                if any(kw in query_lower for kw in keywords):
                    selected.append(filename)
            # If any keyword matches both query and filename
            elif any(kw in fn_lower for kw in keywords) and any(kw in query_lower for kw in keywords):
                selected.append(filename)
                
    return list(set(selected))

import os
from pathlib import Path

def get_all_available_filenames() -> List[str]:
    """Scans both data/ and uploads/ directories for PDF files."""
    filenames = []
    if os.path.exists("data"):
        filenames.extend([p.name for p in Path("data").iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if os.path.exists("uploads"):
        filenames.extend([p.name for p in Path("uploads").iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    return list(set(filenames))

def route_documents(question: str, candidate_filenames: List[str] = None) -> List[str]:
    """
    Routes the query to relevant document filenames.
    For small collections (ROUTER_STRATEGY == 'DIRECT' or candidate count <= threshold),
    bypasses the expensive LLM router call to perform direct hybrid retrieval across candidate documents.
    For large collections (if strategy is set to 'LLM' and candidates > threshold),
    uses LLM-based routing with deterministic keyword fallback.
    """
    if candidate_filenames is None or len(candidate_filenames) == 0:
        available_filenames = get_all_available_filenames()
    else:
        available_filenames = candidate_filenames
        
    if not available_filenames:
        return []

    # Check routing strategy and collection threshold from centralized settings
    strategy = getattr(settings, "ROUTER_STRATEGY", "DIRECT").upper()
    threshold = getattr(settings, "ROUTER_LARGE_COLLECTION_THRESHOLD", 20)

    # 1. DIRECT Strategy / Small Collection Fast Path:
    # Bypasses LLM router call completely when strategy is DIRECT or total available documents <= threshold
    if strategy == "DIRECT" or len(available_filenames) <= threshold:
        logger.info(
            f"⚡ Direct Retrieval active for candidate documents ({len(available_filenames)} file(s) <= threshold {threshold}). Bypassing LLM router."
        )
        return available_filenames

    # 2. LLM Strategy (for large document collections when explicitly configured)
    try:
        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_HELPER_MODEL,
            temperature=0.0
        )
        
        # Build prompt inputs
        available_docs_str = "\n".join(f"- {f}" for f in available_filenames)
        prompt_content = ROUTER_PROMPT.format(
            available_docs=available_docs_str,
            question=question
        )
        
        logger.info("Routing query using LLM (Large Collection Strategy)...")
        response = llm.invoke(prompt_content)
        
        # Parse JSON block
        data = parse_json_from_response(response)
        selected_docs = data.get("documents", [])
        
        # Filter selected docs to match available filenames
        valid_selected = [doc for doc in selected_docs if doc in available_filenames]
        
        if valid_selected:
            logger.info(f"LLM routed query to: {valid_selected}")
            return valid_selected
            
    except Exception as e:
        logger.warning(f"LLM routing failed: {e}. Falling back to keyword routing.")
        
    # 3. Fallback to keyword mapping
    fallback_docs = keyword_routing_fallback(question, available_filenames)
    if fallback_docs:
        logger.info(f"Keyword routing mapped query to: {fallback_docs}")
        return fallback_docs
        
    # 4. Fall back to all available candidate documents
    logger.info("No documents mapped. Falling back to all candidate documents.")
    return available_filenames

def parse_json_from_response(content: Any) -> dict:
    """Helper to locate and parse JSON within the response string."""
    content_str = extract_text(content)
    match = re.search(r'\{.*\}', content_str, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("Response does not contain a valid JSON block.")
