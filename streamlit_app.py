import os
import time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st
import re
from pathlib import Path

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom Styling for polished UI
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }
    .stButton button { width: 100%; border-radius: 8px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# Imports
from app.utils import extract_text
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from app.ingest import run_auto_ingestion, ingest_single_pdf
from app.bm25 import initialize_bm25_index
from app.retriever import retrieve, run_stage1_parallel
from app.chatbot.chatbot import answer_question, answer_question_stream
from app.chatbot.answer_formatter import get_answer_confidence_label, clean_citation_brackets
from app.chatbot.citation_tracker import track_and_filter_citations
from app.semantic_cache import semantic_cache

@st.cache_resource
def init_system():
    get_embedding_model()
    get_vector_store()
    run_auto_ingestion()
    initialize_bm25_index()
    return True

with st.spinner("Initializing AI Engine & Vector Database..."):
    init_system()

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

import torch

# Hardware Acceleration: Automatically prefer GPU (CUDA) if available, fallback to CPU
device_mode = "cuda" if torch.cuda.is_available() else "cpu"

if "messages" not in st.session_state:
    st.session_state.messages = []

active_target_doc = None

# Sidebar Setup - Upload & Select PDF
st.sidebar.title("📄 Document Management")

existing_uploads = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")] if UPLOAD_DIR.exists() else []

if existing_uploads:
    st.sidebar.subheader("📂 Previously Uploaded PDFs")
    selected_existing = st.sidebar.selectbox(
        "Select an existing document:",
        ["-- Select Existing Document --"] + existing_uploads,
        index=0
    )
    if selected_existing != "-- Select Existing Document --":
        active_target_doc = selected_existing
        st.sidebar.success(f"Active Document: `{active_target_doc}`")
    st.sidebar.markdown("---")

st.sidebar.subheader("📤 Upload New Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_path = UPLOAD_DIR / uploaded_file.name
    active_target_doc = uploaded_file.name
    
    # Save and ingest uploaded PDF using persistent SHA256 caching
    if "processed_upload" not in st.session_state or st.session_state.processed_upload != uploaded_file.name:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        status_container = st.sidebar.status("🚀 Processing Document...", expanded=True)
        prog_bar = status_container.progress(0.0)
        step_msg_placeholder = status_container.empty()
        
        def update_progress(stage_title: str, progress_fraction: float, detail_text: str):
            prog_bar.progress(progress_fraction)
            step_msg_placeholder.markdown(f"**{stage_title}**\n\n_{detail_text}_")
            
        try:
            update_progress("📥 File Uploaded", 0.05, f"Saved '{uploaded_file.name}' locally.")
            num_chunks, newly_indexed = ingest_single_pdf(file_path, status_callback=update_progress)
            
            if newly_indexed:
                update_progress("🔍 Updating BM25 Index", 0.90, "Building sparse keyword search index...")
                initialize_bm25_index()
                update_progress("✅ Complete!", 1.00, f"Successfully indexed {num_chunks} chunks.")
                status_container.update(label=f"✅ Indexed {uploaded_file.name} ({num_chunks} chunks)", state="complete", expanded=False)
            else:
                update_progress("⚡ Cache Hit", 1.00, f"Ready instantly from index cache ({num_chunks} chunks).")
                status_container.update(label=f"⚡ Ready from Cache ({num_chunks} chunks)", state="complete", expanded=False)
                
            st.session_state.processed_upload = uploaded_file.name
        except Exception as e:
            status_container.update(label="❌ Document Processing Failed", state="error", expanded=True)
            st.sidebar.error(f"Error: {e}")
    else:
        st.sidebar.info(f"Active Document: `{uploaded_file.name}`")

# Sidebar Setup - Cache & History Management
st.sidebar.markdown("---")
st.sidebar.subheader("🧹 Cache & History Management")
if st.sidebar.button("Clear Redis Cache"):
    if semantic_cache.clear():
        st.sidebar.success("Cleared Redis cache successfully!")
    else:
        st.sidebar.warning("Redis cache is empty or disconnected.")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Main UI Header
st.markdown("<div class='main-header'>📤 Upload & Query Your PDF</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Upload a PDF document or select a previously uploaded file to ask questions instantly.</div>", unsafe_allow_html=True)

if not active_target_doc:
    st.info("👈 Please select a previously uploaded PDF or upload a new one from the sidebar to get started!")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("references") or msg.get("response_time") is not None:
            caption_parts = []
            if msg.get("response_time") is not None:
                dev_label = msg.get("device_label", "🚀 GPU Mode" if device_mode == "cuda" else "💻 CPU Mode")
                caption_parts.append(f"⏱️ Response Time: **{msg['response_time']:.2f} seconds** [{dev_label}]")
            if msg.get("references"):
                caption_parts.append(f"References: {msg['references']}")
            st.caption(" | ".join(caption_parts))

# Handle User Input
prompt = st.chat_input("Ask a question about your document...")

if prompt:
    if not active_target_doc:
        st.warning("⚠️ Please upload a PDF or select an active document from the sidebar first!")
    else:
        target_docs = [active_target_doc]

    st.session_state.messages.append({"role": "user", "content": prompt, "references": None, "response_time": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        
        # Check Redis Semantic Cache first using current device embedding
        cached_response = semantic_cache.get(prompt, target_documents=target_docs, device=device_mode)
        
        if cached_response and "answer" in cached_response and cached_response["answer"]:
            elapsed = time.time() - start_time
            answer_text = cached_response["answer"]
            references_str = cached_response["references"]
            sim_score = cached_response.get("similarity", 1.0)
            st.markdown(answer_text)
            
            mode_tag = f"Cache Hit ({'🚀 GPU Mode' if device_mode == 'cuda' else '💻 CPU Mode'})"
            st.caption(f"⚡ *[Redis Semantic Cache Hit ({sim_score*100:.1f}% match)]* | ⏱️ Response Time: **{elapsed:.3f} seconds** [{mode_tag}] | References: {references_str}")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "references": references_str,
                "response_time": elapsed,
                "device_label": mode_tag
            })
        else:
            q_vec = cached_response.get("query_vector") if cached_response else None
            spinner_msg = f"Searching knowledge base ({'🚀 GPU Mode' if device_mode == 'cuda' else '💻 CPU Mode'}) & generating answer..."
            
            with st.spinner(spinner_msg):
                try:
                    intent, resolved_query = run_stage1_parallel(prompt)
                    retrieved_chunks = retrieve(
                        prompt,
                        target_documents=target_docs,
                        query_vector=q_vec,
                        resolved_query=resolved_query,
                        device=device_mode
                    )
                    retrieval_conf = getattr(retrieved_chunks, "confidence", 0.0)
                    confidence_label = get_answer_confidence_label(retrieval_conf)
                    
                    if confidence_label == "Low":
                        answer_text = "Insufficient information to answer this question accurately from the provided documents."
                        references_str = "No references available."
                        st.markdown(answer_text)
                    else:
                        # Stream the response
                        stream = answer_question_stream(prompt, retrieved_chunks, intent=intent)
                        
                        # Yield text chunks directly for st.write_stream
                        def chunk_generator():
                            for chunk in stream:
                                text = extract_text(chunk)
                                if text:
                                    yield text
                                
                        answer_text = st.write_stream(chunk_generator())
                        
                        # Post-stream verification & citation formatting
                        references_str, _, _ = track_and_filter_citations(answer_text, retrieved_chunks)

                        # Clean inline citation brackets from the final text
                        answer_text = clean_citation_brackets(answer_text)

                        # Store in Redis Semantic Cache for future instant hits
                        semantic_cache.set(
                            query=prompt,
                            answer=answer_text,
                            references=references_str,
                            confidence=retrieval_conf,
                            target_documents=target_docs,
                            query_vector=q_vec,
                            device=device_mode
                        )

                    elapsed = time.time() - start_time
                    dev_label = "🚀 GPU Mode" if device_mode == "cuda" else "💻 CPU Mode"
                    
                    st.caption(f"⏱️ Response Time: **{elapsed:.2f} seconds** [{dev_label}] | References: {references_str}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "references": references_str,
                        "response_time": elapsed,
                        "device_label": dev_label
                    })
                except Exception as e:
                    st.error(f"An error occurred: {e}")

