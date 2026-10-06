import logging
import threading
import re
from typing import List, Tuple

# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

from app.config import settings
from app.vector_store import get_qdrant_client

# Set up logging for the BM25 service
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Private singleton holder and lock for thread safety
_bm25_index: BM25Okapi | None = None
_documents: List[Document] = []
_lock = threading.Lock()

def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into a list of lowercase alphanumeric words.

    Args:
        text (str): The raw input string. 

    Returns:
        List[str]: A list of lowercased tokens.
    """
    return re.findall(r'\w+', text.lower())

def initialize_bm25_index() -> None:
    """
    Fetches all document chunks from Qdrant, tokenizes them, and builds
    the BM25 index in memory. This function is thread-safe and idempotent.
    """
    global _bm25_index, _documents
    with _lock:
        logger.info("Fetching all document chunks from Qdrant for BM25 indexing...")
        client = get_qdrant_client()
        
        offset = None
        records = []
        try:
            while True:
                scroll_records, offset = client.scroll(
                    collection_name=settings.COLLECTION_NAME,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )
                records.extend(scroll_records)
                if offset is None:
                    break
        except Exception as e:
            logger.error(f"Failed to scroll points from Qdrant: {e}")
            raise e
            
        logger.info(f"Retrieved {len(records)} chunks from Qdrant.")
        
        _documents = []
        tokenized_corpus = []
        for record in records:
            if record.payload and "page_content" in record.payload:
                content = record.payload["page_content"]
                metadata = record.payload.get("metadata", {})
                doc = Document(page_content=content, metadata=metadata)
                _documents.append(doc)
                tokenized_corpus.append(tokenize(content))
                
        if _documents:
            logger.info("Initializing BM25Okapi index...")
            _bm25_index = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index successfully initialized.")
        else:
            logger.warning("No documents found in Qdrant collection. BM25 index is empty.")
            _bm25_index = None

def retrieve_bm25(query: str, top_k: int = 5, relevant_documents: List[str] = None) -> List[Tuple[Document, float]]:
    """
    Retrieves the top K documents using BM25 keyword matching, optionally filtering
    by a list of relevant document filenames.

    Args:
        query (str): The search query text.
        top_k (int): Number of top results to return.
        relevant_documents (List[str]): Optional list of document filenames to search within.

    Returns:
        List[Tuple[Document, float]]: List of tuples of (Document, score).
    """
    global _bm25_index, _documents
    if _bm25_index is None:
        initialize_bm25_index()
        
    if _bm25_index is None or not _documents:
        return []
        
    tokenized_query = tokenize(query)
    
    # Dynamic corpus filtering for metadata-aware retrieval
    if relevant_documents:
        filtered_docs = []
        tokenized_corpus = []
        for doc in _documents:
            doc_filename = doc.metadata.get("filename")
            if doc_filename in relevant_documents:
                filtered_docs.append(doc)
                tokenized_corpus.append(tokenize(doc.page_content))
                
        if not filtered_docs:
            return []
            
        # Build a temporary BM25 index for the relevant subset
        temp_index = BM25Okapi(tokenized_corpus)
        scores = temp_index.get_scores(tokenized_query)
        scored_docs = list(zip(filtered_docs, scores))
    else:
        scores = _bm25_index.get_scores(tokenized_query)
        scored_docs = list(zip(_documents, scores))
    
    # Sort descending by score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]
