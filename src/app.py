import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components
import pyperclip
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client
from generate import answer_query

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flynn DocuMind",
    page_icon="🏗️",
    layout="wide"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* #12 — Hand cursor on all clickable elements */
button, [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"],
.stExpander summary, a, [role="button"],
[data-testid="stSidebarCollapseButton"] {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏗️ Flynn DocuMind")
st.caption("Ask questions about roofing specs, building codes, glazing standards, and safety manuals.")
st.divider()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_source" not in st.session_state:
    st.session_state.selected_source = None

if "processing" not in st.session_state:
    st.session_state.processing = False

# ── Document registry helpers ──────────────────────────────────────────────────
def load_document_registry() -> list[dict]:
    result = supabase.table("document_registry") \
        .select("*") \
        .order("uploaded_at", desc=False) \
        .execute()
    return result.data


def format_date(iso_str: str) -> str:
    if not iso_str:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y  %I:%M %p")
    except Exception:
        return iso_str


def delete_document(filename: str):
    supabase.table("documents").delete().eq("metadata->>filename", filename).execute()
    supabase.table("document_registry").delete().eq("filename", filename).execute()
    local_path = os.path.join("docs", filename)
    if os.path.exists(local_path):
        os.remove(local_path)


@st.dialog("Delete Document")
def confirm_delete_dialog(filename: str):
    label = filename.replace(".pdf", "").replace("_", " ").title()
    st.warning(f"Delete **{label}**? This removes all its indexed content and cannot be undone.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, delete", use_container_width=True, type="primary"):
            delete_document(filename)
            st.toast(f"'{label}' deleted.", icon="🗑️")
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document Corpus")

    registry = load_document_registry()

    # #3 — Search/filter documents
    search = st.text_input("🔍 Search documents", placeholder="Type to filter...").lower().strip()
    filtered = [
        doc for doc in registry
        if search in doc["filename"].lower()
    ] if search else registry

    if filtered:
        for doc in filtered:
            label = doc["filename"].replace(".pdf", "").replace("_", " ").title()
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.markdown(f"📋 **{label}**")
                # #10 — Show upload date
                st.caption(f"Uploaded: {format_date(doc['uploaded_at'])}  ·  {doc['chunk_count']} chunks")
            with col_del:
                # #11 — Delete button
                if st.button("🗑️", key=f"del_{doc['filename']}", help=f"Delete {label}"):
                    confirm_delete_dialog(doc["filename"])
    elif search:
        st.caption("No documents match your search.")
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
        is_busy = st.session_state.get("processing", False) or "pending_query" in st.session_state
        if st.button(q, use_container_width=True, disabled=is_busy):
            st.session_state.pending_query = q
            st.rerun()

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
            save_path = os.path.join("docs", uploaded_file.name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                from ingest import ingest_single_pdf
                success, result = ingest_single_pdf(save_path, uploaded_file.name)

            if success:
                st.success(f"Done. {result} chunks indexed from {uploaded_file.name}")
                st.rerun()  # registry reloads fresh on next render
            else:
                st.error(result)


# ── Source popup modal ─────────────────────────────────────────────────────────
@st.dialog("Source Reference")
def show_source_dialog(source: dict):
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
    if source.get("uploaded_at"):
        st.caption(f"Document uploaded: {format_date(source['uploaded_at'])}")
    st.caption("This passage was retrieved and used to generate the answer above.")


# ── Helper: scroll to top of latest message ───────────────────────────────────
def scroll_to_latest_answer():
    components.html("""
    <script>
        setTimeout(function() {
            const messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            if (messages.length > 0) {
                const last = messages[messages.length - 1];
                last.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 150);
    </script>
    """, height=0)


# ── Helper: pipeline history ───────────────────────────────────────────────────
def get_pipeline_history() -> list[dict]:
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]


# ── Helper: render sources ─────────────────────────────────────────────────────
def render_sources(sources: list[dict]):
    if not sources:
        return

    # #9 — Group passages by document so each source appears only once
    grouped: dict[str, list[dict]] = {}
    for source in sources:
        grouped.setdefault(source["filename"], []).append(source)

    st.markdown("**📚 Sources Used**")
    for filename, passages in grouped.items():
        label = filename.replace(".pdf", "").replace("_", " ").title()
        count = len(passages)
        passage_label = "passage" if count == 1 else "passages"
        with st.expander(f"📋 {label}  —  {count} {passage_label} referenced"):
            for i, source in enumerate(passages):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"Page **{source['page']}**")
                with col2:
                    if st.button("View", key=f"src_{filename}_{source['page']}_{i}"):
                        show_source_dialog(source)
                if i < count - 1:
                    st.divider()


# ── Helper: share buttons ──────────────────────────────────────────────────────
def render_share_buttons(question: str, answer: str):
    share_text = f"Q: {question}\n\nA: {answer}\n\nPowered by Flynn DocuMind"
    encoded = share_text.replace("\n", "%0A").replace(" ", "%20")

    whatsapp_url = f"https://wa.me/?text={encoded}"
    email_url = f"mailto:?subject=Flynn%20Tech%20Docs%20AI%20Answer&body={encoded}"

    col1, col2 = st.columns([5, 1])

    with col1:
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;gap:14px;margin-top:8px'>
                <span style='color:#888;font-size:0.8rem'>Share on:</span>
                <a href='{whatsapp_url}' target='_blank' title='Share on WhatsApp'
                   style='text-decoration:none;font-size:1.2rem'>📱</a>
                <a href='{email_url}' target='_blank' title='Share via Email'
                   style='text-decoration:none;font-size:1.2rem'>✉️</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        if st.button("📋 Copy", key=f"copy_{hash(answer)}"):
            try:
                pyperclip.copy(share_text)
                st.toast("Copied to clipboard!", icon="✅")
            except Exception:
                st.toast("Copy failed — try selecting the text manually", icon="⚠️")


# ── Helper: process query ──────────────────────────────────────────────────────
def process_query(query: str):
    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            history = get_pipeline_history()[:-1]
            result = answer_query(query, history=history)

            # #10 — Enrich sources with upload date from registry
            reg_lookup = {doc["filename"]: doc for doc in load_document_registry()}
            for source in result["sources"]:
                source["uploaded_at"] = reg_lookup.get(source["filename"], {}).get("uploaded_at", "")

        st.markdown(result["answer"])
        render_sources(result["sources"])
        render_share_buttons(query, result["answer"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "question": query
    })
    st.session_state.processing = False

    # #6 — Scroll to the top of the answer so the user reads from the beginning
    scroll_to_latest_answer()


# ── Render chat history ────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])
            if message.get("question"):
                render_share_buttons(message["question"], message["content"])

# ── Handle example query buttons ──────────────────────────────────────────────
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")
    process_query(query)
    st.rerun()

# ── Chat input + voice ─────────────────────────────────────────────────────────
is_busy = st.session_state.get("processing", False) or "pending_query" in st.session_state

if is_busy:
    st.info("⏳ Please wait — your question is being answered. Only one question can be processed at a time.")

col_input, col_mic = st.columns([6, 1])

with col_input:
    if query := st.chat_input(
        "Ask a question about roofing, glazing, or safety...",
        disabled=is_busy
    ):
        process_query(query)
        st.rerun()

with col_mic:
    audio = st.audio_input("🎙️")

    if audio is not None:
        audio_key = hash(audio.getvalue())

        if st.session_state.get("last_audio_key") != audio_key:
            st.session_state.last_audio_key = audio_key

            with st.spinner("Transcribing..."):
                import tempfile
                from openai import OpenAI as OAI
                load_dotenv = __import__('dotenv').load_dotenv
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