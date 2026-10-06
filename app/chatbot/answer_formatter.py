from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

def format_sources(docs: List[Document]) -> str:
    """
    Generates a numbered reference footnote list corresponding to [1], [2], etc.
    Used for inline citations support.
    
    Args:
        docs (List[Document]): The final retrieved document list.

    Returns:
        str: A formatted references block.
    """
    if not docs:
        return "No sources available."
        
    lines = []
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", 0) + 1
        lines.append(f"[{idx + 1}] {source} (Page {page})")
        
    return "\n".join(lines)

def get_answer_confidence_label(retrieval_confidence: float) -> str:
    """
    Maps the retrieval confidence percentage to High, Medium, or Low labels.
    
    Args:
        retrieval_confidence (float): Retrieval confidence score in interval [0, 1].

    Returns:
        str: Confidence label (High, Medium, Low).
    """
    if retrieval_confidence >= 0.75:
        return "High"
    elif retrieval_confidence >= 0.40:
        return "Medium"
    else:
        return "Low"

import re

def clean_citation_brackets(text: str) -> str:
    """
    Strips inline citation markers like [1], [2], 【1】 and normalizes spacing.
    """
    if not text:
        return ""
    clean_text = re.sub(r'\s*[\(\[【]\d+[\]\)】]\s*', ' ', str(text))
    clean_text = re.sub(r'\s+([.,!?;:])', r'\1', clean_text)
    return re.sub(r' +', ' ', clean_text).strip()
