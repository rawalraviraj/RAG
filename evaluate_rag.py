import argparse
from app.config import settings

def main():
    """
    Automated RAG Evaluation Framework entrypoint.
    Supports running all tests, category filters, or ad-hoc query validation.
    Overrides Groq LLM model globally before modules are imported.
    """
    parser = argparse.ArgumentParser(description="Automated RAG Evaluation Framework")
    parser.add_argument(
        "--category", 
        type=str, 
        default=None, 
        help="Run only tests matching specific category: lookup, comparison, summary, list, yes_no, memory, switching"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default=None, 
        help="Evaluate a single ad-hoc question end-to-end"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="Override Groq model to use for testing (default: llama-3.1-8b-instant to bypass rate limits)"
    )
    
    args = parser.parse_args()
    
    # Override global Groq model config
    print(f"Setting evaluation LLM model to: '{args.model}'")
    settings.GROQ_MODEL = args.model
    
    # Delay imports of the actual retriever/chatbot components so that they pick up the overridden model
    from tests.evaluation_runner import run_evaluation
    run_evaluation(category_filter=args.category, single_query=args.query)

if __name__ == "__main__":
    main()
