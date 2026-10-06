from typing import List, Dict, Tuple
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], 
    k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Combines multiple ranked lists of Documents using Reciprocal Rank Fusion (RRF).
    
    Args:
        ranked_lists (List[List[Document]]): List of ranked lists, where each list contains Documents.
        k (int): RRF parameter constant (default: 60).
        
    Returns:
        List[Tuple[Document, float]]: Merged list of (Document, RRF score) sorted descending.
    """
    rrf_scores: Dict[str, float] = {}
    doc_mapping: Dict[str, Document] = {}
    
    for doc_list in ranked_lists:
        for rank_idx, doc in enumerate(doc_list):
            # De-duplicate based on the page content text hash
            doc_id = hash(doc.page_content)
            doc_mapping[doc_id] = doc
            
            rank = rank_idx + 1  # 1-based ranking index
            score = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
            
    # Sort descending by the combined RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_mapping[doc_id], score) for doc_id, score in sorted_docs]
