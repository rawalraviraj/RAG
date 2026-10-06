import re
import logging
from typing import List, Tuple
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def track_and_filter_citations(answer: str, docs: List[Document]) -> Tuple[str, List[str], List[str]]:
    """
    Parses the answer for inline citations like [1], [2], and returns:
    - The cleaned references string containing only used citations.
    - List of used citation tags (e.g. ["[1]"])
    - List of ignored citation tags (e.g. ["[2]"])
    
    Args:
        answer (str): The final verified chatbot answer.
        docs (List[Document]): The retrieved compressed context chunks.

    Returns:
        Tuple[str, List[str], List[str]]: References footnotes, used citation list, ignored citation list.
    """
    # Locate all integers wrapped in square brackets
    citation_numbers = re.findall(r'\[(\d+)\]', answer)
    used_indices = set(int(num) for num in citation_numbers)
    
    used_citations_list = []
    used_tags = []
    ignored_tags = []
    
    for idx, doc in enumerate(docs):
        citation_num = idx + 1
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", 0) + 1
        citation_str = f"[{citation_num}] {source} (Page {page})"
        
        if citation_num in used_indices:
            used_citations_list.append(citation_str)
            used_tags.append(f"[{citation_num}]")
        else:
            ignored_tags.append(f"[{citation_num}]")
            
    if not used_citations_list:
        references_str = "No references available."
    else:
        references_str = "\n".join(used_citations_list)
        
    # Log citation coverage statistics
    logger.info(f"Used Citations: {used_tags}")
    logger.info(f"Ignored Citations: {ignored_tags}")
    
    return references_str, used_tags, ignored_tags
