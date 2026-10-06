import os
import logging
import torch
from typing import List

from app.config import settings

# Set up logger for the embeddings module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback mechanism for HuggingFaceEmbeddings import for robust environment compatibility
try:
    # pyrefly: ignore [missing-import]
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    logger.warning("langchain-huggingface package not found. Falling back to langchain-community.")
    # pyrefly: ignore [missing-import]
    from langchain_community.embeddings import HuggingFaceEmbeddings

import threading

# Private cache dictionary and lock for embedding model instances by device
_embedding_models: dict = {}
_lock = threading.Lock()

def get_embedding_model(device: str = None) -> HuggingFaceEmbeddings:
    """
    Initializes and returns a cached instance of the HuggingFaceEmbeddings model for the specified device.
    Supports dynamic switching between 'cuda' (GPU) and 'cpu'.
    This function is thread-safe.

    Args:
        device (str, optional): Target device ('cuda' or 'cpu'). Defaults to CUDA if available else CPU.

    Returns:
        HuggingFaceEmbeddings: The initialized embedding model instance for the target device.
    """
    global _embedding_models
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    if device not in _embedding_models:
        with _lock:
            if device not in _embedding_models:
                logger.info(f"Initializing embedding model ({settings.EMBEDDING_MODEL}) on device: {device.upper()}")
                _embedding_models[device] = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL,
                    model_kwargs={"device": device}
                )
    return _embedding_models[device]


if __name__ == "__main__":
    # 1. Load the embedding model
    model = get_embedding_model()
    
    # 2. Generate an embedding for the test sentence
    test_sentence = "India's national sport is Hockey."
    vector: List[float] = model.embed_query(test_sentence)
    
    # 3. Print verification info
    print(f"Embedding model name: {settings.EMBEDDING_MODEL}")
    print(f"Vector dimension: {len(vector)}")
    print(f"First 10 values of the vector: {vector[:10]}")
