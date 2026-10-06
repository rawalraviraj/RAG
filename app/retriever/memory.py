import logging
import threading
import re
from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from app.config import settings
from app.utils import extract_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Thread-safe registry maintaining the history of the last 5 conversations.
    """
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_conversation(self, question: str, answer: str, sources: str, context: str) -> None:
        with self._lock:
            self.history.append({
                "question": question,
                "answer": answer,
                "sources": sources,
                "context": context
            })
            if len(self.history) > self.max_history:
                self.history.pop(0)
            logger.info(f"Added turn to conversation memory. Current history length: {len(self.history)}")

    def get_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.history)

    def clear(self) -> None:
        with self._lock:
            self.history.clear()
            logger.info("Conversation memory registry cleared.")

# Instantiate global memory tracker
memory = ConversationMemory()

CLASSIFY_PROMPT = """You are deciding whether a user question depends on the previous conversation history.

Conversation History:
{history_str}

Current Question:
{query}

Return ONLY one word: "FOLLOW_UP" or "STANDALONE". Do not provide any explanation, comments, or formatting."""

REWRITE_TEMPLATE = """You are a conversational search assistant.
Given the conversation history and the latest user query, rewrite the user query to be a standalone, search-engine friendly query that includes all necessary context (resolving pronouns like "it", "they", "those", "that policy", "those benefits", etc.).
Do not answer the query. Return ONLY the rewritten query text. No extra explanations, introductions, or formatting.

Conversation History:
{history_str}

Latest query: {query}"""

def is_follow_up_query(query: str, history: List[Dict[str, Any]]) -> bool:
    """
    Classifies the user query as either a follow-up to the conversation context
    or a standalone new topic query.
    """
    if not history:
        return False
        
    try:
        # Pre-filtering with simple heuristics for speed/cost efficiency
        pronouns = {
            "it", "its", "they", "them", "those", "that", "this", "same", 
            "again", "also", "there", "he", "she", "his", "her", "their", 
            "these", "former", "latter"
        }
        words = set(re.findall(r'\w+', query.lower()))
        
        # If the query contains pronoun references, classify as follow-up immediately
        if words.intersection(pronouns):
            logger.info("Query classified as FOLLOW_UP via heuristics (pronoun reference).")
            return True
            
        # Call lightweight LLM classification
        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.0
        )
        
        history_parts = []
        for turn in history:
            history_parts.append(f"User: {turn['question']}\nBot: {turn['answer']}")
        history_str = "\n\n".join(history_parts)
        
        prompt = CLASSIFY_PROMPT.format(history_str=history_str, query=query)
        response = llm.invoke(prompt)
        
        result = extract_text(response).upper()
        logger.info(f"Query classification result: {result} for query '{query}'")
        return "FOLLOW_UP" in result or "STANDALONE" not in result
        
    except Exception as e:
        logger.error(f"Failed to classify query: {e}. Defaulting to True (safe fallback).")
        return True

def resolve_query(query: str, history: List[Dict[str, Any]]) -> str:
    """
    Uses the conversation history to rewrite the query, resolving any follow-up pronouns.
    
    Args:
        query (str): The new user query.
        history (List[Dict[str, Any]]): List of previous dialogue turns.

    Returns:
        str: Clean standalone query.
    """
    if not history:
        return query
        
    # Check if the query is a follow-up
    if not is_follow_up_query(query, history):
        logger.info(f"Query classified as STANDALONE. Bypassing query resolution for: '{query}'")
        return query
        
    try:
        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.0
        )
        
        # Build clean history string for context
        history_parts = []
        for turn in history:
            history_parts.append(f"User: {turn['question']}\nBot: {turn['answer']}")
        history_str = "\n\n".join(history_parts)
        
        prompt = REWRITE_TEMPLATE.format(history_str=history_str, query=query)
        response = llm.invoke(prompt)
        
        rewritten = extract_text(response)
        logger.info(f"Query resolved from history: '{query}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.error(f"Failed to resolve query from history: {e}. Using original query.")
        return query
