import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from generate import answer_query

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flynn Tech Docs AI",
    page_icon="🏗️",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏗️ Flynn Tech Docs AI")
st.caption("Ask questions about roofing specs, building codes, glazing standards, and safety manuals.")
st.divider()

# ── Session state for chat history ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

# ── Sidebar — source explorer ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document Corpus")
    docs = [
        ("doc1_roofing_installation_manual.pdf", "Roofing Installation Manual"),
        ("doc2_building_code_reference.pdf", "NBC Building Code Reference"),
        ("doc3_glazing_curtain_wall_spec.pdf", "Glazing & Curtain Wall Spec"),
        ("doc4_site_safety_manual.pdf", "Site Safety Manual"),
        ("doc5_tpo_product_datasheet.pdf", "TPO Product Data Sheet"),
    ]
    for filename, label in docs:
        st.markdown(f"📋 **{label}**")
        st.caption(filename)

    st.divider()

    st.header("💡 Example Queries")
    example_queries = [
        "What is the minimum TPO membrane thickness?",
        "Compare TPO and EPDM membranes",
        "What are the fall protection requirements?",
        "What roofing system for a building in Winnipeg?",
        "What is the minimum R-value for Climate Zone 7?",
        "What are the seam welding requirements for TPO?",
    ]
    for q in example_queries:
        if st.button(q, use_container_width=True):
            st.session_state.pending_query = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()

# ── Chat history display ───────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources Used"):
                for source in message["sources"]:
                    relevance_pct = int(source["relevance"] * 100)
                    st.markdown(
                        f"- **{source['filename']}**, Page {source['page']} "
                        f"— relevance: `{relevance_pct}%`"
                    )

# ── Handle example query button clicks ────────────────────────────────────────
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = answer_query(query)
        st.markdown(result["answer"])
        with st.expander("📚 Sources Used"):
            for source in result["sources"]:
                relevance_pct = int(source["relevance"] * 100)
                st.markdown(
                    f"- **{source['filename']}**, Page {source['page']} "
                    f"— relevance: `{relevance_pct}%`"
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
    st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a question about roofing, glazing, or safety..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = answer_query(query)
        st.markdown(result["answer"])
        with st.expander("📚 Sources Used"):
            for source in result["sources"]:
                relevance_pct = int(source["relevance"] * 100)
                st.markdown(
                    f"- **{source['filename']}**, Page {source['page']} "
                    f"— relevance: `{relevance_pct}%`"
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
    st.rerun()