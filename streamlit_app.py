"""
TechTrinetra7 — Advanced RAG for Enterprise Document Search
Streamlit application: multi-format upload, hybrid RAG chat, and a
premium dashboard that visualizes every stage of the pipeline.

Run:  streamlit run streamlit_app.py
"""

import streamlit as st

from theme import inject_css, hero
from document_loaders import load_many
from rag_pipeline import AdvancedRAGPipeline

st.set_page_config(
    page_title="Advanced RAG — Enterprise Document Search",
    page_icon="🧠",
    layout="wide",
)
inject_css()

# ----------------------------------------------------------------- session
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Used only in this session. Never stored or hardcoded.")
    st.caption("Models locked for cost control: **gpt-4o-mini** + **text-embedding-3-small**. Reranker runs locally ")

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Settings")
    chunk_size = st.slider("Chunk size", 300, 1500, 800, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 120, 10)
    top_k = st.slider("Top-K per retriever", 3, 15, 8, 1)
    final_k = st.slider("Final chunks after rerank", 2, 10, 5, 1)
    num_variants = st.slider("Query rewrite variants", 0, 5, 3, 1)

    st.markdown("---")
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF · DOCX · TXT · MD · JSON · CSV · PPTX",
        type=["pdf", "docx", "txt", "md", "json", "csv", "pptx"],
        accept_multiple_files=True,
    )

    build_clicked = st.button("🚀 Build Knowledge Base", use_container_width=True)

# ----------------------------------------------------------------- header
hero(
    "Advanced RAG — Enterprise Document Search",
    "Hybrid LangChain LCEL + LlamaIndex pipeline · BM25 + Dense + RRF Fusion + Local Reranking",
)

# ----------------------------------------------------------------- build
if build_clicked:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one document.")
    else:
        with st.spinner("Ingesting documents, chunking, and building hybrid indexes…"):
            files = [(f.name, f.read()) for f in uploaded_files]
            documents = load_many(files)
            pipeline = AdvancedRAGPipeline(
                api_key=api_key,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k_per_retriever=top_k,
                final_k=final_k,
                num_query_variants=num_variants,
            )
            chunks = pipeline.chunk_documents(documents)
            pipeline.build_indexes(chunks)
            st.session_state.pipeline = pipeline
            st.session_state.indexed = True
            st.session_state.chat_history = []
        st.success(f"Knowledge base ready — {len(documents)} document sections → {len(chunks)} chunks indexed.")

# ----------------------------------------------------------------- tabs
tab_chat, tab_dashboard = st.tabs(["💬 Search & Chat", "📊 Pipeline Dashboard"])

with tab_chat:
    if not st.session_state.indexed:
        st.info("Upload documents and click **Build Knowledge Base** in the sidebar to get started.")
    else:
        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(content)

        query = st.chat_input("Ask a question about your enterprise documents…")
        if query:
            st.session_state.chat_history.append(("user", query))
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Running hybrid retrieval, fusion, reranking…"):
                    trace = st.session_state.pipeline.run(query)
                    st.session_state.last_trace = trace
                st.markdown(trace.answer)

            st.session_state.chat_history.append(("assistant", trace.answer))

with tab_dashboard:
    if not st.session_state.last_trace:
        st.info("Ask a question in the **Search & Chat** tab to populate the live pipeline dashboard.")
    else:
        from dashboard import render_full_dashboard
        render_full_dashboard(
            st.session_state.last_trace,
            chunk_count=len(st.session_state.pipeline.chunks),
        )

st.markdown(
    '<div class="tt7-footer">TechTrinetra7 © 2026 · Advanced RAG Enterprise Document Search </div>',
    unsafe_allow_html=True,
)
