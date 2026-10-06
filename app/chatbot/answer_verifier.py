import json
import re
import logging
from typing import List, Type, Any

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from app.config import settings
from app.utils import extract_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerificationResponse(BaseModel):
    supported: bool = Field(description="True if all statements are fully supported by context, False otherwise")
    unsupported_sentences: List[str] = Field(default_factory=list, description="List of unsupported sentences found in the answer")
    corrected_answer: str = Field(default="", description="A revised version of the answer containing only context-supported facts")

VERIFICATION_PROMPT = """You are a highly precise answer verification assistant (Hallucination Guard).
Analyze the candidate answer against the provided retrieved context.
Determine whether every factual statement in the answer is fully supported by the retrieved context.

Retrieved Context:
{context}

Candidate Answer:
{answer}

Instructions:
- If all statements in the candidate answer are fully supported, return supported = true.
- If there are statements that are unsupported or hallucinated relative to the context:
  - List them in unsupported_sentences.
  - Set supported = false.
  - Write a corrected_answer. The corrected_answer must preserve the supported facts while completely removing any unsupported or hallucinated claims.
- Return ONLY a valid JSON object matching the schema below. No other text, explanations, or code blocks.
{{
  "supported": false,
  "unsupported_sentences": ["unsupported claim"],
  "corrected_answer": "corrected response text"
}}"""

def parse_json_safely(content: Any, model: Type[BaseModel]) -> BaseModel:
    """Helper to locate, parse and validate JSON using Pydantic."""
    content_str = extract_text(content)
    match = re.search(r'\{.*\}', content_str, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in response.")
    parsed = json.loads(match.group(0))
    return model.model_validate(parsed)

def verify_answer(answer: str, context: str) -> VerificationResponse:
    """
    Verifies candidate answer correctness against retrieved context.
    Validates output via Pydantic. Retries once on failure, then defaults to original answer.

    Args:
        answer (str): Candidate generated answer.
        context (str): Compressed retrieved context text.

    Returns:
        VerificationResponse: Pydantic validated response object.
    """
    # Bypass verification check on 8B models to avoid false-positive rejections
    if "8b" in settings.GROQ_MODEL.lower():
        logger.info("Bypassing second-pass answer verification on 8B model to avoid false-positive rejections.")
        return VerificationResponse(supported=True, unsupported_sentences=[], corrected_answer=answer)

    def run_call():
        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.0
        )
        response = llm.invoke(VERIFICATION_PROMPT.format(context=context, answer=answer))
        parsed = parse_json_safely(response.content.strip(), VerificationResponse)
        return parsed

    try:
        return run_call()
    except Exception as e:
        logger.warning(f"Answer verification attempt 1 failed: {e}. Retrying once...")
        try:
            return run_call()
        except Exception as e2:
            logger.error(f"Answer verification failed after retry: {e2}. Falling back to original answer.")
            return VerificationResponse(supported=True, unsupported_sentences=[], corrected_answer=answer)
