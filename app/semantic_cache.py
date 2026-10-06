import json
import uuid
import logging
import re
import time
from datetime import datetime, timezone
import numpy as np
from typing import Optional, Dict, Any, List

import redis
from app.config import settings
from app.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    """
    Normalizes an incoming query string by:
    - Stripping leading/trailing whitespace
    - Lowercasing text
    - Collapsing multiple internal spaces to a single space
    - Stripping non-alphanumeric punctuation from the start and end
    """
    if not query:
        return ""
    text = query.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^[^\w]+|[^\w]+$', '', text)
    return text.strip()


def get_kb_identifier(target_documents: Optional[List[str]]) -> str:
    """
    Generates a deterministic knowledge_base/document-set identifier string.
    """
    if not target_documents:
        return "default"
    cleaned_docs = sorted([doc.strip() for doc in target_documents if doc.strip()])
    return ",".join(cleaned_docs) if cleaned_docs else "default"


class SemanticCache:
    """
    Redis-backed Semantic Cache for RAG pipeline.
    
    Features:
    - Query text normalization (lowercasing, whitespace collapsing, punctuation cleanup)
    - Vector similarity search via embedding model (BAAI/bge-small-en-v1.5)
    - Knowledge base / document set scoping to prevent cross-document cache contamination
    - Configurable similarity threshold (e.g., settings.SEMANTIC_CACHE_THRESHOLD = 0.88)
    - Full record payload storage (answer, original_query, normalized_query, knowledge_base, timestamp)
    """

    def __init__(self, threshold: Optional[float] = None, ttl: Optional[int] = None):
        self.threshold = threshold if threshold is not None else settings.SEMANTIC_CACHE_THRESHOLD
        self.ttl = ttl if ttl is not None else settings.REDIS_TTL
        self.redis_client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self):
        if not getattr(settings, "ENABLE_SEMANTIC_CACHE", True):
            logger.info("Semantic cache is disabled in settings.")
            return

        try:
            kwargs = {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "db": settings.REDIS_DB,
                "decode_responses": True,
                "socket_timeout": 3.0
            }
            if settings.REDIS_PASSWORD:
                kwargs["password"] = settings.REDIS_PASSWORD
            if settings.REDIS_SSL:
                kwargs["ssl"] = True
                kwargs["ssl_cert_reqs"] = None

            self.redis_client = redis.Redis(**kwargs)
            self.redis_client.ping()
            logger.info(f"Connected to Redis Semantic Cache at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis Semantic Cache: {e}. Cache fallback active.")
            self.redis_client = None

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get(
        self,
        query: str,
        target_documents: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Performs vector similarity search against cached queries for the same knowledge base.
        Returns cached response dict if similarity >= threshold, else None.
        """
        if not self.redis_client or not query:
            return None

        effective_threshold = threshold if threshold is not None else self.threshold
        normalized_q = normalize_query(query)
        kb_id = get_kb_identifier(target_documents)

        try:
            embedding_model = get_embedding_model(device=device)
            query_vector = embedding_model.embed_query(normalized_q)

            # Batch-fetch keys using non-blocking scan_iter and mget
            keys_batch = list(self.redis_client.scan_iter("semantic_cache:*", count=100))
            if not keys_batch:
                return None

            best_match: Optional[Dict[str, Any]] = None
            best_sim = -1.0

            raw_records = self.redis_client.mget(keys_batch)
            for raw_data in raw_records:
                if not raw_data:
                    continue

                try:
                    entry = json.loads(raw_data)
                except Exception:
                    continue

                # 1. Enforce strict Knowledge Base / Document Set matching
                cached_kb = entry.get("knowledge_base", "default")
                if cached_kb != kb_id:
                    continue

                # 2. Extract cached vector
                cached_vector = entry.get("embedding")
                if not cached_vector:
                    continue

                # 3. Fast-path check: if normalized query string matches exactly
                cached_norm_q = entry.get("normalized_query", "")
                if cached_norm_q == normalized_q:
                    sim = 1.0
                else:
                    sim = self._cosine_similarity(query_vector, cached_vector)

                if sim > best_sim:
                    best_sim = sim
                    best_match = entry

            if best_match and best_sim >= effective_threshold:
                logger.info(
                    f"⚡ [REDIS SEMANTIC CACHE HIT] Similarity: {best_sim:.4f} >= {effective_threshold} | KB: '{kb_id}'"
                )
                return {
                    "answer": best_match.get("answer", ""),
                    "references": best_match.get("references", ""),
                    "confidence": best_match.get("confidence", 1.0),
                    "intent": best_match.get("intent", "FACT_LOOKUP"),
                    "original_query": best_match.get("original_query", ""),
                    "normalized_query": best_match.get("normalized_query", ""),
                    "knowledge_base": best_match.get("knowledge_base", "default"),
                    "timestamp": best_match.get("timestamp", ""),
                    "similarity": round(best_sim, 4),
                    "query_vector": query_vector
                }
            else:
                logger.debug(
                    f"Cache miss for query '{normalized_q}' (best sim: {best_sim:.4f} vs threshold {effective_threshold})"
                )
                return {"cache_miss": True, "query_vector": query_vector}

        except Exception as e:
            logger.error(f"Error searching Redis Semantic Cache: {e}")

        return None

    def set(
        self,
        query: str,
        answer: str,
        references: str = "",
        confidence: float = 1.0,
        intent: str = "FACT_LOOKUP",
        target_documents: Optional[List[str]] = None,
        query_vector: Optional[List[float]] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Stores record in Redis Semantic Cache using optional precomputed query vector.
        """
        if not self.redis_client or not query or not answer:
            return False

        normalized_q = normalize_query(query)
        kb_id = get_kb_identifier(target_documents)

        try:
            if query_vector is None:
                embedding_model = get_embedding_model(device=device)
                query_vector = embedding_model.embed_query(normalized_q)

            cache_key = f"semantic_cache:{uuid.uuid4().hex}"
            payload = {
                "id": cache_key,
                "original_query": query,
                "normalized_query": normalized_q,
                "knowledge_base": kb_id,
                "embedding": query_vector,
                "answer": answer,
                "references": references,
                "confidence": confidence,
                "intent": intent,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            self.redis_client.set(
                name=cache_key,
                value=json.dumps(payload),
                ex=self.ttl
            )
            logger.info(f"💾 [REDIS SEMANTIC CACHE STORED] Key: {cache_key} | KB: '{kb_id}' | TTL: {self.ttl}s")
            return True
        except Exception as e:
            logger.error(f"Error writing to Redis Semantic Cache: {e}")
            return False

    def clear(self) -> bool:
        """Clears all semantic cache keys from Redis."""
        if not self.redis_client:
            return False
        try:
            keys = self.redis_client.keys("semantic_cache:*")
            if keys:
                self.redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} semantic cache keys from Redis.")
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis Semantic Cache: {e}")
            return False


# Global Singleton Instance
semantic_cache = SemanticCache()