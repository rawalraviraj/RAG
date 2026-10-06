import os
import logging
import torch
from typing import List, Tuple

# Guarantee Hugging Face operates strictly offline using cached weights
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Cache dictionary for CrossEncoder instances by device
_reranker_models: dict = {}

def get_reranker(device: str = None) -> CrossEncoder:
    global _reranker_models
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested for reranker but unavailable. Falling back to CPU.")
        device = "cpu"

    if device not in _reranker_models:
        logger.info(f"Loading CrossEncoder re-ranker model on device: {device.upper()}...")
        _reranker_models[device] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device=device)
    return _reranker_models[device]

def rerank_documents(query: str, documents: List[Document], top_k: int = 5, device: str = None) -> List[Document]:
    if not documents:
        return []
    
    model = get_reranker(device=device)
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)
    
    # Attach scores and sort
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs[:top_k]]
