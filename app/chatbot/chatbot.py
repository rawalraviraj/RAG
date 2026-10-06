import logging
from typing import List

# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.utils import extract_text
from app.chatbot.intent_classifier import classify_intent
from app.chatbot.answer_verifier import verify_answer
from app.retriever.context_builder import build_context

# Set up logging for the chatbot module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Initializing ChatGroq Primary LLM ({settings.GROQ_MODEL}) with fallback to Gemini ({settings.GEMINI_MODEL})")
primary_llm = ChatGroq(
    groq_api_key=settings.GROQ_API_KEY,
    model_name=settings.GROQ_MODEL,
    temperature=0.0,
    streaming=True
)

fallback_llm = ChatGoogleGenerativeAI(
    google_api_key=settings.GOOGLE_API_KEY,
    model=settings.GEMINI_MODEL,
    temperature=0.0,
    streaming=True
)

llm = primary_llm.with_fallbacks([fallback_llm])

# Intent formatting instructions lookup dictionary
INTENT_FORMAT_INSTRUCTIONS = {
    "FACT_LOOKUP": "Format your answer as a concise lookup response.",
    "SUMMARY": "Format your answer as a short structured summary.",
    "LIST": "Format your answer strictly as a bulleted list.",
    "COMPARISON": "Format your answer using a markdown comparison table.",
    "YES_NO": "Format your response starting with 'Yes' or 'No' as the first word, followed by an explanation.",
    "EXPLANATION": "Format your response as a clear explanation paragraph, using bullet points for detailed points if needed."
}

def answer_question(question: str, retrieved_docs: List[Document], intent: str = None) -> str:
    """
    Orchestrates the generation phase of the RAG pipeline.
    Classifies user intent, appends intent-aware formatting guidelines,
    generates response, verifies against context for hallucinations, and returns response.
    
    Args:
        question (str): The user query.
        retrieved_docs (List[Document]): Compressed matching candidate document chunks.
        intent (str, optional): Pre-computed query intent.

    Returns:
        str: Custom string object wrapped with intent and verification metadata.
    """
    # 1. Classify query intent if not provided
    if intent is None:
        intent = classify_intent(question)
    logger.info(f"Classified query intent: {intent}")
    
    # 2. Format context using context builder
    context_str = build_context(retrieved_docs)
    
    # 3. Incorporate intent formatting instruction into System Prompt template
    format_instruction = INTENT_FORMAT_INSTRUCTIONS.get(intent, "")
    
    # Core system template defining citation rules
    system_prompt = f"""You are a highly precise, context-grounded question-answering assistant.
Answer the user's question using ONLY the provided context below.
Every factual statement in your answer MUST cite its supporting document source inline using the document numbers, for example `[1]` or `[1][2]`.
If a statement is supported by multiple documents, list all applicable citations, e.g., `[1][2]`.
Never cite document numbers that were not provided in the context.
If the context does not contain the answer, reply exactly with: "I don't know based on the provided documents."
Do not make up facts or hallucinate any information.

{format_instruction}

Context:
{{context}}"""

    # Build Prompt Messages
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    messages = prompt_template.format_messages(context=context_str, question=question)
    
    # 4. Call LLM for generation
    logger.info("Invoking LLM for answer generation...")
    response = llm.invoke(messages)
    candidate_answer = extract_text(response)
    
    # 5. Answer Verification (Fast-path optimization)
    final_answer = candidate_answer
    is_verified = True
    unsupported = []

    if intent in ("SUMMARY", "COMPARISON", "EXPLANATION"):
        logger.info("Running Answer Verification Pass for complex intent...")
        verification = verify_answer(candidate_answer, context_str)
        is_verified = verification.supported
        if not verification.supported:
            logger.warning(f"Answer Verification Failed! Unsupported claim(s): {verification.unsupported_sentences}")
            final_answer = verification.corrected_answer
            unsupported = verification.unsupported_sentences
    else:
        logger.info("Fast path active: skipping verification LLM pass for simple fact lookup intent.")

    class ChatbotResponse(str):
        intent: str = ""
        verified: bool = True
        unsupported_sentences: List[str] = []

    res = ChatbotResponse(final_answer)
    res.intent = intent
    res.verified = is_verified
    res.unsupported_sentences = unsupported
    return res

def answer_question_stream(question: str, retrieved_docs: List[Document], intent: str = None):
    """
    Streams the generation phase of the RAG pipeline token-by-token.
    Note: Answer verification is skipped during streaming, as the UI handles it live.
    """
    if intent is None:
        intent = classify_intent(question)
    logger.info(f"Classified query intent for streaming: {intent}")
    
    context_str = build_context(retrieved_docs)
    format_instruction = INTENT_FORMAT_INSTRUCTIONS.get(intent, "")
    
    system_prompt = f"""You are a highly precise, context-grounded question-answering assistant.
Answer the user's question using ONLY the provided context below.
Every factual statement in your answer MUST cite its supporting document source inline using the document numbers, for example `[1]` or `[1][2]`.
If a statement is supported by multiple documents, list all applicable citations, e.g., `[1][2]`.
Never cite document numbers that were not provided in the context.
If the context does not contain the answer, reply exactly with: "I don't know based on the provided documents."
Do not make up facts or hallucinate any information.

{format_instruction}

Context:
{{context}}"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    messages = prompt_template.format_messages(context=context_str, question=question)
    
    logger.info("Invoking ChatGroq model for streaming generation...")
    return llm.stream(messages)


if __name__ == "__main__":
    from app.retriever.retriever import retrieve

    # Test question
    default_question = "What are the office timings?"
    try:
        user_input = input(f"Enter question [default: {default_question}]: ").strip()
        question = user_input if user_input else default_question
    except (KeyboardInterrupt, EOFError):
        question = default_question
        print(f"\nUsing default question: {question}")
        
    print(f"\nQuestion: {question}")
    
    try:
        # Retrieve Context
        docs = retrieve(question)
        
        # Generate Answer
        print("\n=== ANSWER ===")
        answer = answer_question(question, docs)
        print(answer)
        
        from app.chatbot.citation_tracker import track_and_filter_citations
        print("\n=== REFERENCES ===")
        ref_str, used, ignored = track_and_filter_citations(answer, docs)
        print(ref_str)
        print(f"Used: {used} | Ignored: {ignored}")
        
        retrieval_conf = getattr(docs, "confidence", 0.0)
        from app.chatbot.answer_formatter import get_answer_confidence_label
        print(f"\n=== CONFIDENCE ===")
        print(f"Answer Confidence: {get_answer_confidence_label(retrieval_conf)} ({retrieval_conf * 100:.1f}%)")
        print("======================\n")
        
    except Exception as e:
        logger.error(f"Chatbot pipeline execution failed: {e}")
