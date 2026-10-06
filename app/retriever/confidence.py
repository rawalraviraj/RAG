import numpy as np
from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
from app.embeddings import get_embedding_model

def compute_retrieval_confidence(query: str, selected_docs: List[Document]) -> float:
    """
    Computes a retrieval confidence score (0.0 to 1.0) based on the average cosine
    similarity of the selected documents to the query.
    
    Args:
        query (str): The search query.
        selected_docs (List[Document]): The final retrieved context documents.

    Returns:
        float: Normalized confidence score from 0.0 to 1.0.
    """
    if not selected_docs:
        return 0.0
        
    embedding_model = get_embedding_model()
    query_emb = np.array(embedding_model.embed_query(query))
    
    similarities = []
    for doc in selected_docs:
        doc_emb = np.array(embedding_model.embed_query(doc.page_content))
        norm_q = np.linalg.norm(query_emb)
        norm_d = np.linalg.norm(doc_emb)
        if norm_q > 0 and norm_d > 0:
            sim = float(np.dot(query_emb, doc_emb) / (norm_q * norm_d))
            similarities.append(sim)
            
    if not similarities:
        return 0.0
        
    avg_sim = float(np.mean(similarities))
    
    # Scale from range [0.3, 0.8] to standard [0.0, 1.0] confidence interval
    confidence = (avg_sim - 0.3) / 0.5
    confidence = max(0.0, min(1.0, confidence))
    return confidence
