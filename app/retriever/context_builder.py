from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

def build_context(docs: List[Document]) -> str:
    """
    Combines retrieved chunks intelligently. Removes duplicates (fused chunks are
    already unique), separates each chunk with clean structural delimiters,
    and returns a unified string.
    
    Args:
        docs (List[Document]): List of unique documents to construct context from.

    Returns:
        str: Delimited context string.
    """
    if not docs:
        return "[Empty Context]"
        
    context_parts = []
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", 0) + 1
        part = f"--- Document Chunk {idx + 1} | Source: {source} (Page {page}) ---\n{doc.page_content.strip()}"
        context_parts.append(part)
        
    return "\n\n".join(context_parts)
