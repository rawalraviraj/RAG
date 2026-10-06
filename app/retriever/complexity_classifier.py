import json
import re
import logging
from typing import Type, Any

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from app.config import settings
from app.utils import extract_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplexityResponse(BaseModel):
    complexity: str = Field(description="Query complexity level: 'SIMPLE', 'MEDIUM', or 'COMPLEX'")

COMPLEXITY_PROMPT = """You are a highly precise query complexity classifier.
Classify the user query into one of three complexity levels:
- SIMPLE: Short, single-fact keyword lookups (e.g. "What is the insurance amount?")
- MEDIUM: Multi-fact lookup or explanation queries (e.g. "What benefits are provided?")
- COMPLEX: Comparison, listing, or full document summarization queries (e.g. "Compare HR policy with finance policy", "Summarize employee handbook")

User Query: {query}

Instructions:
- Return ONLY a valid JSON object matching the schema below. No other text, explanations, or code blocks.
{{
  "complexity": "COMPLEX"
}}"""

def parse_json_safely(content: Any, model: Type[BaseModel]) -> BaseModel:
    """Helper to locate, parse and validate JSON using Pydantic."""
    content_str = extract_text(content)
    match = re.search(r'\{.*\}', content_str, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in response.")
    parsed = json.loads(match.group(0))
    return model.model_validate(parsed)

def local_classify_complexity(query: str) -> str | None:
    """
    Fast deterministic rule-based complexity classification (<0.1 ms).
    Returns 'SIMPLE', 'MEDIUM', 'COMPLEX', or None if ambiguous.
    """
    if not query:
        return "SIMPLE"
    q = query.strip().lower()
    words = q.split()
    word_count = len(words)

    # 1. COMPLEX
    if any(k in q for k in ["compare", "comparison", "summarize", "summary", "list all", "all benefits", "eligibility conditions"]):
        return "COMPLEX"
    if word_count > 15:
        return "COMPLEX"

    # 2. SIMPLE
    if (q.startswith(("what is ", "what are ", "when is ", "where is ", "who is ", "how many ", "how much ", "does ", "can ", "is there ")) or
        any(k in q for k in ["timing", "timings", "hours", "bonus", "salary", "amount", "coverage", "allowance", "reimbursement", "limit"])):
        if word_count <= 10 and not any(k in q for k in ["and what are", "eligibility", "compare"]):
            return "SIMPLE"

    # 3. MEDIUM
    if any(k in q for k in ["benefits", "eligibility", "process", "explain", "how"]):
        return "MEDIUM"

    return None

def classify_complexity(query: str) -> str:
    """
    Classifies query complexity (SIMPLE, MEDIUM, COMPLEX) using fast local rule matching first (<0.1 ms),
    falling back to ChatGroq LLM for ambiguous queries.
    
    Args:
        query (str): The standalone user query.

    Returns:
        str: 'SIMPLE', 'MEDIUM', or 'COMPLEX'.
    """
    # 1. Try fast local classification
    local_complexity = local_classify_complexity(query)
    if local_complexity:
        logger.info(f"⚡ [FAST LOCAL COMPLEXITY] Classified complexity as '{local_complexity}' (<0.1ms)")
        return local_complexity

    # 2. Ambiguous query fallback to LLM
    logger.info(f"Local complexity ambiguous for query '{query}'. Falling back to LLM classifier...")
    def run_call():
        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_HELPER_MODEL,
            temperature=0.0
        )
        response = llm.invoke(COMPLEXITY_PROMPT.format(query=query))
        parsed = parse_json_safely(response.content.strip(), ComplexityResponse)
        val = parsed.complexity.upper()
        if val in {"SIMPLE", "MEDIUM", "COMPLEX"}:
            return val
        raise ValueError(f"Invalid complexity level classified: {val}")

    try:
        return run_call()
    except Exception as e:
        logger.warning(f"Complexity classification attempt 1 failed: {e}. Retrying once...")
        try:
            return run_call()
        except Exception as e2:
            logger.error(f"Complexity classification failed after retry: {e2}. Defaulting to 'MEDIUM'.")
            return "MEDIUM"
