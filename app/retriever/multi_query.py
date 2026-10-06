import re
import logging
from typing import List
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from app.config import settings
from app.utils import extract_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUERY_EXPANSION_TEMPLATE = """You are an AI assistant tasked with generating alternative search queries to improve retrieval.
For the user's original query, generate exactly {num_alternatives} alternative queries that are semantically similar but use different synonyms, keywords, or phrasings.
Format the output as a simple list of queries, one per line, with no numbers, prefixes, bullet points, introductions, or explanations.

Original query: {query}"""

def generate_alternative_queries(query: str, limit: int = 4) -> List[str]:
    """
    Generates alternative search queries using ChatGroq. Always prepends the original query.
    
    Args:
        query (str): The original user query.
        limit (int): Total number of queries desired (including original).

    Returns:
        List[str]: A list containing the original query and generated alternative queries.
    """
    if limit <= 1:
        return [query]
        
    num_alternatives = limit - 1
    try:
        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.0
        )
        prompt_str = QUERY_EXPANSION_TEMPLATE.format(num_alternatives=num_alternatives, query=query)
        response = llm.invoke(prompt_str)
        
        # Parse alternate queries from lines
        lines = extract_text(response).split("\n")
        queries = [query]
        for line in lines:
            line_clean = line.strip().lstrip("-*•").strip()
            # Strip numbering prefixes (e.g., "1. ")
            line_clean = re.sub(r'^\d+\.\s*', '', line_clean).strip()
            if line_clean and line_clean.lower() != query.lower():
                queries.append(line_clean)
                
        # Return unique queries capped at limit
        return list(dict.fromkeys(queries))[:limit]
    except Exception as e:
        logger.error(f"Failed to expand query: {e}. Falling back to original query only.")
        return [query]
