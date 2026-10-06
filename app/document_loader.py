import os
import logging
from pathlib import Path
from typing import List

# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
import nest_asyncio
from app.config import settings

# Apply nest_asyncio to allow LlamaParse in Streamlit/Jupyter
nest_asyncio.apply()

# Set up logging for the document loader module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Loads a PDF file from the disk and returns a list of LangChain Document objects.
    Each page of the PDF is loaded as an individual Document object.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Document]: List of LangChain Document objects containing page content.

    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If the file path is not a PDF file.
        RuntimeError: If there's an error reading/parsing the PDF.
    """
    path = Path(pdf_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
        
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"The file is not a PDF document: {pdf_path}")

    # Check if we have LlamaParse API key
    if settings.LLAMA_CLOUD_API_KEY:
        try:
            from llama_parse import LlamaParse
            logger.info(f"Using LlamaParse for Advanced Parsing: {pdf_path}")
            
            parser = LlamaParse(
                api_key=settings.LLAMA_CLOUD_API_KEY,
                result_type="markdown",  # Get rich markdown with tables
                num_workers=2
            )
            parsed_docs = parser.load_data(str(path))
            
            # Convert LlamaIndex docs to LangChain docs
            return [
                Document(page_content=doc.text, metadata={"page": i, "source": path.name}) 
                for i, doc in enumerate(parsed_docs)
            ]
        except Exception as e:
            logger.error(f"LlamaParse failed, falling back to PyPDFLoader: {e}")
            
    # Fallback / Default
    try:
        logger.info(f"Using standard PyPDFLoader: {pdf_path}")
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        return docs
    except Exception as e:
        logger.error(f"Error loading PDF from {pdf_path}: {e}")
        raise RuntimeError(f"Failed to load and parse PDF file: {e}") from e

if __name__ == "__main__":
    # Define a default path pointing to our sample.pdf
    default_path = "data/sample.pdf"
    
    try:
        user_input = input(f"Enter PDF path [default: {default_path}]: ").strip()
        pdf_path = user_input if user_input else default_path
    except (KeyboardInterrupt, EOFError):
        pdf_path = default_path
        print(f"\nUsing default path: {pdf_path}")

    print(f"Loading: {pdf_path}")
    try:
        # Load documents
        documents = load_pdf(pdf_path)
        
        # Print number of pages
        print(f"Number of pages: {len(documents)}")
        print("----------------------------------------")
        
        # Iterate and print metadata/content preview for each page
        for i, doc in enumerate(documents):
            page_num = doc.metadata.get("page", i) + 1
            char_count = len(doc.page_content)
            preview = doc.page_content[:300].replace("\n", " ")
            
            print(f"Page Number: {page_num}")
            print(f"Character Count: {char_count}")
            print(f"First 300 characters: {preview}...")
            print("----------------------------------------")
            
    except Exception as e:
        logger.error(f"Test run failed: {e}")
