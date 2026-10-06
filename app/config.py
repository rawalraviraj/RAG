import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure Hugging Face runs strictly in offline mode using local cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Determine root directory path and .env path dynamically
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

class Settings(BaseSettings):
    """
    Centralized configuration settings for the RAG application.
    Loads configurations from environment variables or from the root .env file.
    """
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_HELPER_MODEL: str = "openai/gpt-oss-20b"
    GOOGLE_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-3.6-flash"

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    COLLECTION_NAME: str = "rag_documents"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TTL: int = 86400  # 24 hours
    SEMANTIC_CACHE_THRESHOLD: float = 0.88
    ENABLE_SEMANTIC_CACHE: bool = True
    
    # Document Routing strategy: "DIRECT" (small collection fast path) or "LLM" (large collection LLM routing)
    ROUTER_STRATEGY: str = "DIRECT"
    ROUTER_LARGE_COLLECTION_THRESHOLD: int = 20
    
    LLAMA_CLOUD_API_KEY: str | None = None

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    TOP_K: int = 5

    DENSE_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate a single configuration object for the application
settings = Settings()

if __name__ == "__main__":
    print(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"Qdrant Host: {settings.QDRANT_HOST}")
    print(f"Qdrant Port: {settings.QDRANT_PORT}")
    print(f"Collection Name: {settings.COLLECTION_NAME}")
    print(f"Chunk Size: {settings.CHUNK_SIZE}")
    print(f"Chunk Overlap: {settings.CHUNK_OVERLAP}")
    print(f"Top K: {settings.TOP_K}")
    print(f"Selected Groq Model: {settings.GROQ_MODEL}")