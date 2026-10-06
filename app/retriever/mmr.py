import numpy as np
from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
from app.embeddings import get_embedding_model

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the cosine similarity between two numpy vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def maximum_marginal_relevance(
    query: str,
    candidate_docs: List[Document],
    top_k: int,
    lambda_param: float = 0.5
) -> List[Document]:
    """
    Selects top_k documents from candidate_docs using Maximum Marginal Relevance (MMR).
    Balances query relevance with diversity relative to already-selected documents.
    
    Args:
        query (str): The original search query.
        candidate_docs (List[Document]): Fused candidate documents.
        top_k (int): Number of final documents to select.
        lambda_param (float): Tuning parameter between relevance (1.0) and diversity (0.0).
        
    Returns:
        List[Document]: Diversified final list of documents.
    """
    if not candidate_docs:
        return []
    if len(candidate_docs) <= top_k:
        return candidate_docs
        
    embedding_model = get_embedding_model()
    
    # 1. Generate embeddings for query and candidate documents
    query_emb = np.array(embedding_model.embed_query(query))
    doc_embs = [np.array(embedding_model.embed_query(doc.page_content)) for doc in candidate_docs]
    
    # 2. Compute similarity to query for all candidates
    query_similarities = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embs]
    
    selected_indices: List[int] = []
    unselected_indices = list(range(len(candidate_docs)))
    
    # Start by selecting the document most similar to the query
    first_idx = int(np.argmax(query_similarities))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)
    
    while len(selected_indices) < top_k and unselected_indices:
        best_mmr_score = -float('inf')
        best_idx = -1
        
        for idx in unselected_indices:
            # Relevance: similarity to original query
            sim_to_query = query_similarities[idx]
            
            # Redundancy: max similarity to any document selected so far
            max_sim_to_selected = max(
                cosine_similarity(doc_embs[idx], doc_embs[sel_idx])
                for sel_idx in selected_indices
            )
            
            # MMR combined score
            mmr_score = lambda_param * sim_to_query - (1.0 - lambda_param) * max_sim_to_selected
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
                
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
        
    return [candidate_docs[idx] for idx in selected_indices]
