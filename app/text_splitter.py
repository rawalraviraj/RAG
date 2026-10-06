import logging
from typing import List

# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

# Set up logging for the text splitter module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks
    using RecursiveCharacterTextSplitter configured via app.config.settings.

    Args:
        documents (List[Document]): The input list of Document objects.

    Returns:
        List[Document]: The list of chunked Document objects.
    """
    logger.info(f"Initializing text splitter with CHUNK_SIZE={settings.CHUNK_SIZE}, CHUNK_OVERLAP={settings.CHUNK_OVERLAP}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    logger.info(f"Splitting {len(documents)} documents...")
    chunks = splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    from app.document_loader import load_pdf
    
    # Define target path for verification
    pdf_path = "data/sample.pdf"
    
    try:
        # 1. Load the PDF
        logger.info(f"Loading test document: {pdf_path}")
        original_docs = load_pdf(pdf_path)
        original_pages_count = len(original_docs)
        
        # If the PDF does not yield extractable characters, populate it with mock text for splitter verification
        has_text = any(doc.page_content.strip() for doc in original_docs)
        if not has_text and original_docs:
            logger.info("Loaded PDF has no extractable characters. Injecting text to verify splitting functionality.")
            original_docs[0].page_content = (
                "Retrieval-Augmented Generation (RAG) is a technique for localizing language model responses "
                "to external knowledge bases. This architecture integrates information retrieval with text generation. "
                "The ingestion layer extracts content from raw sources, slices it into smaller segments (chunking), "
                "embeds those segments into semantic vectors, and registers them in a database. "
                "During retrieval, the query vector finds the closest indexed vectors to supply context. "
            ) * 12
        
        # 2. Split documents
        logger.info("Executing split_documents...")
        chunks = split_documents(original_docs)
        
        # 3. Print verification statistics
        print("\nVerification Results:")
        print(f"Number of original pages: {original_pages_count}")
        print(f"Number of chunks: {len(chunks)}")
        print("----------------------------------------")
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk Index: {i + 1}")
            print(f"Chunk length: {len(chunk.page_content)}")
            print(f"Chunk metadata: {chunk.metadata}")
            preview = chunk.page_content[:200].replace("\n", " ")
            print(f"First 200 characters: {preview}...")
            print("----------------------------------------")
            
    except Exception as e:
        logger.error(f"Failed to execute text splitter test: {e}")
