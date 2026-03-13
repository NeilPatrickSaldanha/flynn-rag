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

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_source" not in st.session_state:
    st.session_state.selected_source = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document Corpus")
    docs_in_folder = sorted([
        f for f in os.listdir("docs") if f.endswith(".pdf")
    ])
    if docs_in_folder:
        for filename in docs_in_folder:
            label = filename.replace(".pdf", "").replace("_", " ").title()
            st.markdown(f"📋 **{label}**")
            st.caption(filename)
    else:
        st.caption("No documents loaded yet.")

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
        st.session_state.selected_source = None
        st.rerun()

    st.divider()
    st.header("📤 Upload a Document")
    st.caption("Upload any roofing, glazing, or safety PDF to query it instantly.")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type="pdf",
        key="pdf_uploader"
    )

    if uploaded_file is not None:
        if st.button("📥 Ingest Document", use_container_width=True):
            # Save to docs folder temporarily
            save_path = os.path.join("docs", uploaded_file.name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                from ingest import ingest_single_pdf
                success, result = ingest_single_pdf(save_path, uploaded_file.name)

            if success:
                st.success(f"Done. {result} chunks indexed from {uploaded_file.name}")
            else:
                st.error(result)

# ── Source popup modal ─────────────────────────────────────────────────────────
@st.dialog("Source Reference")
def show_source_dialog(source: dict):
    # Document badge row
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📋 {source['filename']}")
    with col2:
        st.markdown(
            f"<div style='background:#1f4e79;padding:6px 12px;border-radius:20px;"
            f"text-align:center;color:white;font-weight:bold;margin-top:8px'>"
            f"Page {source['page']}</div>",
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("**Cited passage:**")
    st.markdown(
        f"<div style='"
        f"background:#1a1a2e;"
        f"border-left: 4px solid #e67e22;"
        f"padding: 16px 20px;"
        f"border-radius: 6px;"
        f"font-size: 0.95rem;"
        f"line-height: 1.7;"
        f"color: #ecf0f1;"
        f"margin-top: 8px;"
        f"'>{source['content']}</div>",
        unsafe_allow_html=True
    )

    st.divider()
    st.caption("This passage was retrieved and used to generate the answer above.")


# ── Helper: build history for pipeline ────────────────────────────────────────
def get_pipeline_history() -> list[dict]:
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]


# ── Helper: render sources as clickable buttons ────────────────────────────────
def render_sources(sources: list[dict]):
    if not sources:
        return

    with st.expander("📚 Sources Used — click to view passage"):
        for i, source in enumerate(sources):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{source['filename']}**, Page {source['page']}")
            with col2:
                if st.button("View", key=f"src_{id(source)}_{i}"):
                    show_source_dialog(source)


# ── Helper: process and display a query ───────────────────────────────────────
def process_query(query: str):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            history = get_pipeline_history()[:-1]
            result = answer_query(query, history=history)

        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })


# ── Render existing chat history ───────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])

# ── Handle example query button clicks ────────────────────────────────────────
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")
    process_query(query)
    st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────────
# ── Chat input ─────────────────────────────────────────────────────────────────
col_input, col_mic = st.columns([6, 1])

with col_input:
    if query := st.chat_input("Ask a question about roofing, glazing, or safety..."):
        process_query(query)
        st.rerun()

with col_mic:
    st.markdown("<div style='padding-top: 8px'>", unsafe_allow_html=True)
    audio = st.audio_input("🎙️")
    st.markdown("</div>", unsafe_allow_html=True)

    if audio is not None:
        # Use the audio bytes as a key to prevent reprocessing same clip
        audio_key = hash(audio.getvalue())

        if st.session_state.get("last_audio_key") != audio_key:
            st.session_state.last_audio_key = audio_key

            with st.spinner("Transcribing..."):
                import tempfile
                from openai import OpenAI as OAI
                import os
                from dotenv import load_dotenv
                load_dotenv()

                oai = OAI(api_key=os.getenv("OPENAI_API_KEY"))

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio.getvalue())
                    tmp_path = tmp.name

                with open(tmp_path, "rb") as audio_file:
                    transcription = oai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )

                os.unlink(tmp_path)
                voice_query = transcription.text

            if voice_query:
                process_query(voice_query)
                st.rerun()