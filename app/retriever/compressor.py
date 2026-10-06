import logging
import copy
import numpy as np
from typing import List

# pyrefly: ignore [missing-import]
import tiktoken
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

from app.utils import extract_text
from app.config import settings
from app.embeddings import get_embedding_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMPRESSION_PROMPT = """You are an expert context compression assistant.
Given the user query and a document chunk, extract ONLY the facts, policies, numbers, dates, limits, names, or definitions from the chunk that are directly relevant to answering the query.
Remove all other unrelated sentences, filler text, and noise.
Do not change the meaning of the facts. Do not invent or summarize facts not present in the chunk.
If the chunk is completely irrelevant to the query, reply exactly with: EMPTY

User Query: {query}
Document Chunk:
{chunk}"""

def count_tokens(text: str) -> int:
    """
    Computes token counts using tiktoken with cl100k_base encoding.
    Falls back to word count approximation if tiktoken fails.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"tiktoken encoding failed: {e}. Falling back to word count.")
        return len(text.split())

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the cosine similarity between two numpy vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def compress_chunks(chunks: List[Document], query: str, relevant_documents: List[str] = None) -> List[Document]:
    """
    Compresses each chunk in the list relative to the query using ChatGroq.
    Discards any chunks returned as 'EMPTY', unless they belong to targeted
    routed documents or have extremely high semantic similarity.
    
    Args:
        chunks (List[Document]): Candidate chunks.
        query (str): Standalone resolved search query.
        relevant_documents (List[str]): List of document filenames routed for this query.

    Returns:
        List[Document]: The compressed list of document chunks.
    """
    if not chunks:
        return []
        
    embedding_model = get_embedding_model()
    query_emb = np.array(embedding_model.embed_query(query))
    
    # Calculate similarities for semantic check
    similarities = []
    for doc in chunks:
        doc_emb = np.array(embedding_model.embed_query(doc.page_content))
        sim = cosine_similarity(query_emb, doc_emb)
        similarities.append(sim)
        
    compressed_docs = []
    kept_by_doc = {}
    
    try:
        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.0
        )

        def _compress_single(idx_and_doc):
            idx, doc = idx_and_doc
            prompt = COMPRESSION_PROMPT.format(query=query, chunk=doc.page_content)
            response = llm.invoke(prompt)
            result = extract_text(response)
            return idx, doc, result

        from concurrent.futures import ThreadPoolExecutor
        workers = min(len(chunks), 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            llm_results = list(executor.map(_compress_single, enumerate(chunks)))

        # Preserve original document ordering after parallel execution
        llm_results.sort(key=lambda x: x[0])

        for idx, doc, result in llm_results:
            fn = doc.metadata.get("filename", "Unknown")

            # Check if LLM indicates the chunk is irrelevant
            if result.upper() == "EMPTY" or result.strip() == "":
                logger.info(f"LLM compression suggested discard for chunk from '{fn}' (Page {doc.metadata.get('page', 0)+1}).")

                # Semantic similarity check safeguard
                sim = similarities[idx]
                if sim >= 0.70:
                    logger.info(f"Safeguard triggered: Preserving chunk from '{fn}' due to high similarity ({sim:.2f}).")
                    doc_copy = copy.deepcopy(doc)
                    compressed_docs.append(doc_copy)
                    kept_by_doc.setdefault(fn, []).append(doc_copy)
                continue

            doc_copy = copy.deepcopy(doc)
            doc_copy.page_content = result
            compressed_docs.append(doc_copy)
            kept_by_doc.setdefault(fn, []).append(doc_copy)

    except Exception as e:
        logger.error(f"Failed during context compression: {e}. Falling back to original chunks.")
        return chunks
        
    # Safeguard: For every primary routed document, at least one chunk must survive compression
    if relevant_documents:
        for req_doc in relevant_documents:
            if req_doc not in kept_by_doc or not kept_by_doc[req_doc]:
                # Locate candidate index for this document
                candidate_indices = [
                    i for i, d in enumerate(chunks) 
                    if d.metadata.get("filename") == req_doc
                ]
                if candidate_indices:
                    # Pick highest similarity chunk
                    best_idx = max(candidate_indices, key=lambda i: similarities[i])
                    best_sim = similarities[best_idx]
                    logger.info(f"Safeguard triggered: Restoring highest-similarity chunk for routed doc '{req_doc}' (sim: {best_sim:.2f}) to prevent total discard.")
                    
                    doc_copy = copy.deepcopy(chunks[best_idx])
                    compressed_docs.append(doc_copy)
                    kept_by_doc.setdefault(req_doc, []).append(doc_copy)
                    
    return compressed_docs
