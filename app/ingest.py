import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

# pyrefly: ignore [missing-import]
from qdrant_client import models
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

from app.config import settings
from app.document_loader import load_pdf
from app.text_splitter import split_documents
from app.vector_store import get_vector_store, get_qdrant_client

# Set up logging for the ingestion pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("indexed_files.json")
DATA_DIR = Path("data")

def calculate_sha256(file_path: Path) -> str:
    """Computes the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()

def load_index_registry() -> Dict[str, Any]:
    """Loads the index registry from indexed_files.json."""
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load index registry: {e}")
            return {}
    return {}

def save_index_registry(registry: Dict[str, Any]) -> None:
    """Saves the index registry to indexed_files.json."""
    try:
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save index registry: {e}")

def scan_data_folder() -> List[Path]:
    """Scans the DATA_DIR directory for every PDF file."""
    if not DATA_DIR.exists():
        logger.warning(f"Data directory '{DATA_DIR}' does not exist. Creating it.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return []
    
    return [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]

from typing import List, Dict, Any, Tuple, Callable, Optional

def process_pdf(pdf_path: Path, status_callback: Optional[Callable[[str, float, str], None]] = None) -> Tuple[List[Document], int]:
    """Loads a PDF and splits it into chunked Document objects, attaching running metadata."""
    if status_callback:
        status_callback("📄 Parsing PDF Pages", 0.25, f"Parsing text & layout from '{pdf_path.name}'...")
    docs = load_pdf(str(pdf_path))
    pages_count = len(docs)
    
    if status_callback:
        status_callback("✂️ Splitting Chunks", 0.50, f"Extracted {pages_count} page(s). Splitting into semantic text chunks...")
    chunks = split_documents(docs)
    filename = pdf_path.name
    
    for idx, chunk in enumerate(chunks):
        # Preserve existing metadata, set source to filename, page, and chunk_id
        chunk.metadata["source"] = filename
        chunk.metadata["filename"] = filename
        chunk.metadata["document_type"] = pdf_path.stem
        chunk.metadata["page"] = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["chunk_index"] = idx
        
        # Determine document title if available, otherwise fallback to stem name
        title = chunk.metadata.get("title")
        if not title or title.strip() == "" or title.lower() == "(anonymous)":
            title = pdf_path.stem.replace("_", " ").title()
        chunk.metadata["title"] = title
        
    return chunks, pages_count

def delete_pdf_vectors(filename: str) -> None:
    """Deletes old vectors from Qdrant associated with the given filename."""
    logger.info(f"Deleting old Qdrant vectors for file: {filename}")
    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=filename)
                    )
                ]
            )
        )
    except Exception as e:
        logger.error(f"Failed to delete Qdrant vectors for {filename}: {e}")

def index_pdf(pdf_path: Path, chunks: List[Document], status_callback: Optional[Callable[[str, float, str], None]] = None) -> int:
    """Generates embeddings and stores chunked documents in the Qdrant Vector Store."""
    if not chunks:
        logger.warning(f"No chunks to index for: {pdf_path}")
        return 0
        
    if status_callback:
        status_callback("🧠 Generating Vector Embeddings", 0.70, f"Embedding {len(chunks)} chunks & storing in Qdrant Vector Database...")
    vector_store = get_vector_store()
    logger.info(f"Uploading {len(chunks)} chunks to Qdrant for {pdf_path.name}...")
    vector_store.add_documents(chunks)
    return len(chunks)

def ingest_single_pdf(pdf_path: Path, status_callback: Optional[Callable[[str, float, str], None]] = None) -> Tuple[int, bool]:
    """
    Ingests a single PDF file if it has not been indexed yet (or if hash changed).
    Returns (num_chunks, was_newly_indexed).
    """
    filename = pdf_path.name
    if status_callback:
        status_callback("🔍 Checking Index Cache", 0.10, f"Computing SHA-256 hash for '{filename}'...")
    current_hash = calculate_sha256(pdf_path)
    registry = load_index_registry()

    if filename in registry:
        old_hash = registry[filename].get("hash")
        if old_hash == current_hash:
            logger.info(f"File '{filename}' is already indexed (hash match). Skipping re-embedding.")
            if status_callback:
                status_callback("⚡ Loaded from Cache", 1.00, f"File '{filename}' is ready instantly from cache!")
            return registry[filename].get("chunks", 0), False

    # Process and index new or updated file
    chunks, pages_count = process_pdf(pdf_path, status_callback=status_callback)
    num_chunks = index_pdf(pdf_path, chunks, status_callback=status_callback)
    
    registry[filename] = {
        "hash": current_hash,
        "chunks": num_chunks
    }
    save_index_registry(registry)
    return num_chunks, True

def run_auto_ingestion() -> Dict[str, Any]:
    """
    Main entry point for auto-indexing the `/data` folder.
    Scans, checks hashes, skips/updates/inserts, and returns execution stats.
    """
    # Safeguard: ensure connection is active and collection exists
    get_vector_store()
    
    print(f"Scanning {DATA_DIR} ...\n")
    pdf_files = scan_data_folder()
    print(f"Found {len(pdf_files)} PDFs\n")
    
    registry = load_index_registry()
    
    # Safeguard: if Qdrant collection is empty (e.g. fresh database), force re-indexing all documents
    try:
        q_client = get_qdrant_client()
        if q_client.count(collection_name=settings.COLLECTION_NAME).count == 0:
            logger.info("Vector database is empty. Forcing full re-indexing of all documents in /data.")
            registry = {}
    except Exception as e:
        logger.warning(f"Could not check Qdrant point count: {e}")

    new_registry = {}
    
    stats = {
        "total": len(pdf_files),
        "indexed": 0,
        "skipped": 0,
        "updated": 0,
        "total_chunks": 0
    }
    
    for pdf_path in pdf_files:
        filename = pdf_path.name
        current_hash = calculate_sha256(pdf_path)
        
        # Check index registry status
        if filename in registry:
            old_hash = registry[filename].get("hash")
            if old_hash == current_hash:
                print(f"{filename}")
                print("    Status: Already Indexed")
                print()
                stats["skipped"] += 1
                new_registry[filename] = registry[filename]
                continue
            else:
                print(f"{filename}")
                print("    Status: Updated → Reindexing")
                stats["updated"] += 1
                delete_pdf_vectors(filename)
        else:
            print(f"{filename}")
            stats["indexed"] += 1
            
        try:
            chunks, pages_count = process_pdf(pdf_path)
            num_chunks = index_pdf(pdf_path, chunks)
            
            print(f"    Pages: {pages_count}")
            print(f"    Chunks: {num_chunks}")
            if filename in registry:
                print("    Status: Reindexed")
            else:
                print("    Status: Indexed")
            print()
            
            new_registry[filename] = {
                "hash": current_hash,
                "chunks": num_chunks
            }
        except Exception as e:
            logger.error(f"Failed to process PDF {filename}: {e}")
            print(f"    Status: Failed ({e})\n")
            # Preserve old entry on failure
            if filename in registry:
                new_registry[filename] = registry[filename]
                
    # Calculate total chunks across all currently registered documents
    for filename, data in new_registry.items():
        stats["total_chunks"] += data.get("chunks", 0)
            
    save_index_registry(new_registry)
    
    print("----------------------------------\n")
    print(f"Total PDFs : {stats['total']}")
    print(f"Indexed    : {stats['indexed']}")
    print(f"Skipped    : {stats['skipped']}")
    print(f"Updated    : {stats['updated']}")
    print(f"Total Chunks : {stats['total_chunks']}")
    print()
    
    return stats

if __name__ == "__main__":
    try:
        run_auto_ingestion()
    except Exception as e:
        logger.error(f"Auto-ingestion failed: {e}")
