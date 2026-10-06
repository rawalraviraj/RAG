import sys
import logging
from app.retriever import retrieve
from app.chatbot import answer_question
from app.config import settings

# Configure logging. We set it to WARNING for the main terminal script to keep console output clean.
logging.basicConfig(level=logging.WARNING)

def run_rag_terminal() -> None:
    """
    Main interactive loop for the terminal-based RAG application.
    Prompts the user for questions, retrieves matching document chunks from Qdrant,
    generates responses from ChatGroq, and displays the results.
    """
    print("====================================")
    print("SYSTEM STARTUP")
    print("====================================")
    
    print("Loading Embedding Model...")
    from app.embeddings import get_embedding_model
    get_embedding_model()
    print("Embedding Model Ready.")
    
    print("Connecting Qdrant...")
    from app.vector_store import get_vector_store
    get_vector_store()
    print("Vector Store Ready.")
    
    # Automatically scan and index the data directory
    from app.ingest import run_auto_ingestion
    run_auto_ingestion()
    
    print("Initializing BM25 Index...")
    from app.bm25 import initialize_bm25_index
    initialize_bm25_index()
    print("BM25 Index Ready.")
    
    print("System Ready.")
    print()

    while True:
        try:
            # Prompt user for question input
            question = input("User: ").strip()
            
            if not question:
                continue
                
            # Exit conditions
            if question.lower() in ("exit", "quit"):
                print("\nExiting terminal session. Goodbye!")
                break
                
            print()
            print("====================================")
            print("QUESTION")
            print("====================================")
            print()
            print("Question:")
            print(question)
            print()
            
            print("Searching Vector DB...")
            print()
            
            # 1. Query the retriever module
            try:
                retrieved_chunks = retrieve(question)
            except Exception as e:
                print(f"[Retrieval Error] Failed to search vector database. Details: {e}")
                continue

            print(f"Retrieved {len(retrieved_chunks)} chunks.")
            print()

            # Display retrieved context chunks
            print("[Retrieved Chunks]")
            if retrieved_chunks:
                for i, doc in enumerate(retrieved_chunks):
                    source = doc.metadata.get("source", "Unknown Source")
                    page = doc.metadata.get("page", 0) + 1
                    content = doc.page_content.strip().replace("\n", " ")[:120]
                    print(f"  [{i + 1}] Source: {source} (Page {page})")
                    print(f"      Content preview: {content}...")
                    print("-" * 50)
            else:
                print("  [No matching chunks found in Qdrant]")
                print("-" * 50)
            print()
                
            # 2. Call the generator service with pre-retrieved chunks
            print("Generating Answer...")
            print()
            try:
                from app.chatbot.answer_formatter import get_answer_confidence_label, clean_citation_brackets
                retrieval_conf = getattr(retrieved_chunks, "confidence", 0.0)
                confidence_label = get_answer_confidence_label(retrieval_conf)
                
                intent = "FACT_LOOKUP"
                verified = True
                used_citations = []
                ignored_citations = []
                
                if confidence_label == "Low":
                    answer = "Insufficient information."
                    references_str = "No references available."
                else:
                    answer = answer_question(question, retrieved_chunks)
                    intent = getattr(answer, "intent", "FACT_LOOKUP")
                    verified = getattr(answer, "verified", True)
                    
                    from app.chatbot.citation_tracker import track_and_filter_citations
                    references_str, used_citations, ignored_citations = track_and_filter_citations(answer, retrieved_chunks)
                
                clean_answer = clean_citation_brackets(answer)

                print("Done.")
                print()
                print(f"Bot: {clean_answer}")
                print()
                
                print("References")
                print(references_str)
                print()
                print(f"Answer Confidence: {confidence_label} ({retrieval_conf * 100:.1f}%)")
                print()
                
                # Printing the dynamic Nice Logging Block (Logging)
                print("========================")
                print()
                print("Detected Intent:")
                print(intent)
                print()
                print("Complexity:")
                print(getattr(retrieved_chunks, "complexity", "MEDIUM"))
                print()
                print("Dynamic TopK:")
                print(getattr(retrieved_chunks, "top_k", 5))
                print()
                print("Expanded Queries:")
                print(getattr(retrieved_chunks, "expansion_limit", 3))
                print()
                print("Selected Documents:")
                sel_docs = getattr(retrieved_chunks, "relevant_documents", [])
                if sel_docs:
                    for d in sel_docs:
                        print(d)
                else:
                    print("All Documents")
                print()
                print("Retrieved Chunks:")
                print(len(getattr(retrieved_chunks, "original_chunks", retrieved_chunks)))
                print()
                print("Compressed Chunks:")
                print(len(retrieved_chunks))
                print()
                print("Answer Verification:")
                print("PASSED" if verified else "FAILED (Corrected)")
                print()
                print("Used Citations:")
                if used_citations:
                    for cite in used_citations:
                        print(cite)
                else:
                    print("None")
                print()
                print("Confidence:")
                print(f"{retrieval_conf * 100:.0f}%")
                print()
                print("========================")
                print()
                
                # Save the turn to Conversation Memory (Issue 7)
                from app.retriever.memory import memory
                from app.retriever.context_builder import build_context
                context_str = build_context(retrieved_chunks)
                memory.add_conversation(question, answer, references_str, context_str)
                
            except Exception as e:
                print(f"[Generation Error] Failed to generate answer. Details: {e}")
                
            print("\n" + "=" * 65)

        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"[Error] An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_rag_terminal()
