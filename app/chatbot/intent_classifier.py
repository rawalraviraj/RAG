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

class IntentResponse(BaseModel):
    intent: str = Field(description="Query intent class: 'FACT_LOOKUP', 'SUMMARY', 'COMPARISON', 'LIST', 'YES_NO', or 'EXPLANATION'")

INTENT_PROMPT = """You are a highly precise user intent classifier.
Classify the user query into one of six intent categories:
1. FACT_LOOKUP: Direct lookup of specific names, numbers, or rules (e.g. "What is the insurance amount?")
2. SUMMARY: Request to summarize an entire topic or document (e.g. "Summarize HR policy")
3. COMPARISON: Request comparing policies, rules, or files (e.g. "Compare leave and insurance policy")
4. LIST: Request asking for a bulleted list of items (e.g. "List all employee benefits")
5. YES_NO: Simple yes/no confirmation queries (e.g. "Does it cover family?")
6. EXPLANATION: How-to guides, definitions, or procedural open-ended descriptions (e.g. "Tell me about finance", "How to index files")

User Query: {query}

Instructions:
- Return ONLY a valid JSON object matching the schema below. No other text, explanations, or code blocks.
{{
  "intent": "COMPARISON"
}}"""

def parse_json_safely(content: Any, model: Type[BaseModel]) -> BaseModel:
    """Helper to locate, parse and validate JSON using Pydantic."""
    content_str = extract_text(content)
    match = re.search(r'\{.*\}', content_str, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in response.")
    parsed = json.loads(match.group(0))
    return model.model_validate(parsed)

def local_classify_intent(query: str) -> str | None:
    """
    Fast deterministic rule-based intent classification (<0.1 ms).
    Returns valid intent string or None if ambiguous.
    """
    if not query:
        return "FACT_LOOKUP"
    q = query.strip().lower()

    # 1. COMPARISON
    if any(k in q for k in ["compare", "comparison", "difference between", " vs ", "versus", "differ"]):
        return "COMPARISON"

    # 2. SUMMARY
    if any(k in q for k in ["summarize", "summary of", "overview of", "brief on"]):
        return "SUMMARY"

    # 3. LIST
    if any(k in q for k in ["list all", "list of", "list the", "what are all", "enumerate"]):
        return "LIST"

    # 4. YES_NO
    if (q.startswith(("is ", "are ", "does ", "do ", "can ", "has ", "have ", "will ", "should ")) or
        "include " in q or "covered" in q or "carried forward" in q):
        if not q.startswith(("what ", "when ", "where ", "who ", "how ", "why ")):
            return "YES_NO"

    # 5. EXPLANATION
    if q.startswith(("explain ", "describe ", "how to ", "how do ", "how does ", "tell me about ")):
        return "EXPLANATION"

    # 6. FACT_LOOKUP
    if (q.startswith(("what is", "what are", "when is", "when are", "where is", "where are", "who is", "how much", "how many")) or
        any(k in q for k in ["timing", "timings", "hours", "bonus", "salary", "amount", "coverage", "allowance", "reimbursement", "policy", "limit", "code"])):
        return "FACT_LOOKUP"

    return None

def classify_intent(query: str) -> str:
    """
    Classifies query intent using fast local rule matching first (<0.1 ms),
    falling back to ChatGroq LLM for ambiguous queries.
    Validates output via Pydantic schema enums.

    Args:
        query (str): The standalone user query.

    Returns:
        str: Detected intent name.
    """
    # 1. Try fast local classification
    local_intent = local_classify_intent(query)
    if local_intent:
        logger.info(f"⚡ [FAST LOCAL INTENT] Classified intent as '{local_intent}' (<0.1ms)")
        return local_intent

    # 2. Ambiguous query fallback to LLM
    logger.info(f"Local intent ambiguous for query '{query}'. Falling back to LLM classifier...")
    def run_call():
        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_HELPER_MODEL,
            temperature=0.0
        )
        response = llm.invoke(INTENT_PROMPT.format(query=query))
        parsed = parse_json_safely(response.content.strip(), IntentResponse)
        val = parsed.intent.upper()
        valid_intents = {"FACT_LOOKUP", "SUMMARY", "COMPARISON", "LIST", "YES_NO", "EXPLANATION"}
        if val in valid_intents:
            return val
        raise ValueError(f"Invalid intent class classified: {val}")

    try:
        return run_call()
    except Exception as e:
        logger.warning(f"Intent classification attempt 1 failed: {e}. Retrying once...")
        try:
            return run_call()
        except Exception as e2:
            logger.error(f"Intent classification failed after retry: {e2}. Defaulting to 'FACT_LOOKUP'.")
            return "FACT_LOOKUP"
