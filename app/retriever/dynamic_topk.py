def get_dynamic_top_k(query: str) -> int:
    """
    Analyzes the query and determines the optimal retrieval count.
    
    Heuristics:
        - If query asks for comparison/summarization/list -> return 8
        - If query is very short (<= 3 words) -> return 3
        - Otherwise -> return 5

    Args:
        query (str): The search query.

    Returns:
        int: Dynamic Top-K count (3, 5, or 8).
    """
    query_lower = query.lower()
    words = query_lower.split()
    
    comparison_keywords = {
        "compare", "comparison", "difference", "versus", "vs", "both", 
        "all", "list", "summarize", "table", "contrast", "distinguish"
    }
    
    is_comparison = any(kw in query_lower for kw in comparison_keywords)
    
    if is_comparison:
        return 8
    elif len(words) <= 3:
        return 3
    else:
        return 5
