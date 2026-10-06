import logging
import threading
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient

# pyrefly: ignore [missing-import]
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
# pyrefly: ignore [missing-import]
from langchain_qdrant import QdrantVectorStore

from app.config import settings
from app.embeddings import get_embedding_model

# Set up logging for the vector store module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch

# Private singleton holders and locks for thread safety
_qdrant_client: QdrantClient | None = None
_vector_stores: dict = {}
_client_lock = threading.Lock()
_store_lock = threading.Lock()

def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns a cached QdrantClient instance using the centralized configuration.
    Connects strictly to the standalone Qdrant server (http://localhost:6333).
    Does NOT fall back to embedded local disk storage. This function is thread-safe.

    Returns:
        QdrantClient: A connected Qdrant client instance.

    Raises:
        RuntimeError: If the Qdrant server at host:port is unreachable.
    """
    global _qdrant_client
    if _qdrant_client is None:
        with _client_lock:
            if _qdrant_client is None:
                if settings.QDRANT_URL and settings.QDRANT_API_KEY:
                    logger.info(f"Initializing QdrantClient with Qdrant Cloud URL ({settings.QDRANT_URL})")
                    _qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
                else:
                    target_url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
                    logger.info(f"Connecting to standalone Qdrant server at {target_url}...")
                    try:
                        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=5.0)
                        # Quick connection test
                        client.get_collections()
                        _qdrant_client = client
                        logger.info(f"Successfully connected to Qdrant server at {target_url}.")
                    except Exception as e:
                        err_msg = (
                            f"Could not connect to standalone Qdrant server at {target_url} ({e}). "
                            "Please make sure the Qdrant Docker container is running ('docker compose up -d qdrant')."
                        )
                        logger.error(err_msg)
                        raise RuntimeError(err_msg) from e
    return _qdrant_client

def ensure_payload_indices() -> None:
    """
    Ensures required payload indices (such as metadata.filename and metadata.source) exist in Qdrant.
    This prevents HTTP 400 'Index required but not found' errors during filtered vector searches.
    """
    client = get_qdrant_client()
    fields = ["metadata.filename", "metadata.source", "metadata.file_name", "filename", "source"]
    for field in fields:
        try:
            client.create_payload_index(
                collection_name=settings.COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info(f"Ensured payload index for field '{field}'.")
        except Exception as e:
            logger.debug(f"Payload index for field '{field}' skipped or existing: {e}")

def create_collection_if_not_exists() -> bool:
    """
    Checks if the configured collection exists in Qdrant. If it does not exist,
    creates a new collection configured for 384-dimension vectors with COSINE distance.
    Also guarantees that payload indices are present.

    Returns:
        bool: True if the collection was created, False if it already existed.
    """
    client = get_qdrant_client()
    
    logger.info(f"Checking if Qdrant collection '{settings.COLLECTION_NAME}' exists...")
    created = False
    if not client.collection_exists(collection_name=settings.COLLECTION_NAME):
        logger.info(f"Collection '{settings.COLLECTION_NAME}' does not exist. Creating collection...")
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{settings.COLLECTION_NAME}' created successfully.")
        created = True
    else:
        logger.info(f"Collection '{settings.COLLECTION_NAME}' already exists.")

    ensure_payload_indices()
    return created

def get_vector_store(device: str = None) -> QdrantVectorStore:
    """
    Ensures that the collection is created in Qdrant, then initializes, caches,
    and returns the LangChain QdrantVectorStore wrapper for the requested device.
    This function is thread-safe.

    Args:
        device (str, optional): Target hardware device ('cuda' or 'cpu').

    Returns:
        QdrantVectorStore: An initialized LangChain Qdrant vector store singleton.
    """
    global _vector_stores
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = device.lower()

    if device not in _vector_stores:
        with _store_lock:
            if device not in _vector_stores:
                # Guarantee the collection exists & payload indices are ready before initializing store
                create_collection_if_not_exists()

                client = get_qdrant_client()
                embedding_model = get_embedding_model(device=device)

                logger.info(f"Initializing LangChain QdrantVectorStore ({device.upper()}) for collection '{settings.COLLECTION_NAME}'...")
                _vector_stores[device] = QdrantVectorStore(
                    client=client,
                    collection_name=settings.COLLECTION_NAME,
                    embedding=embedding_model
                )
    return _vector_stores[device]

if __name__ == "__main__":
    print("----------------------------------------")
    print("Connecting to Qdrant...")
    try:
        # Establish connection
        client = get_qdrant_client()
        # Force a network request to check connection validity
        client.get_collections()
        print("Connected successfully.")
        print(f"Collection:\n{settings.COLLECTION_NAME}")

        # Check and create collection
        created = create_collection_if_not_exists()
        print("Status:")
        if created:
            print("Created Successfully")
            print("Vector Size:\n384")
            print("Distance:\nCOSINE")
        else:
            print("Collection already exists.")
        print("----------------------------------------")

        # Verify LangChain vector store initialization
        logger.info("Initializing vector store verification...")
        vector_store = get_vector_store()
        logger.info("Vector store verification successful.")

    except Exception as e:
        logger.error(f"Failed to connect or configure Qdrant: {e}")
        print("Connection failed.")
        print("----------------------------------------")
