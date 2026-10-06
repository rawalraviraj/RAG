from typing import Any

def extract_text(content: Any) -> str:
    """
    Safely extracts string text from an LLM response object or content property.
    Handles string, list of text parts, or dict parts returned by models like ChatGoogleGenerativeAI.
    Preserves raw token whitespace for streaming deltas.
    """
    if hasattr(content, "content"):
        content = content.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
            elif hasattr(item, "text") and isinstance(getattr(item, "text"), str):
                parts.append(getattr(item, "text"))
            elif hasattr(item, "content") and isinstance(getattr(item, "content"), str):
                parts.append(getattr(item, "content"))
        return "".join(parts)

    if content is None:
        return ""

    return str(content)
