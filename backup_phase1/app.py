
import streamlit as st
from pathlib import Path
import hashlib
import shutil

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    UPLOAD_DIR,
    CHROMA_DIR
)

from modules.document_processor import process_document
from modules.embeddings import create_embeddings
from modules.vector_store import get_vector_store, add_chunks
from modules.rag_pipeline import answer_question
from modules.evaluation import simple_grounding_check


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NEXUS AI | Enterprise Knowledge",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: #0b1220;
        color: #e8eef9;
    }

    [data-testid="stSidebar"] {
        background: #101b2e;
    }

    .hero {
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg,#142746,#173e72);
        border: 1px solid #294b78;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2.25rem;
    }

    .hero p {
        color: #c5d8f4;
        margin: .35rem 0 0 0;
    }

    .metric {
        background: #121f34;
        border: 1px solid #263a57;
        border-radius: 14px;
        padding: 1rem;
    }

    .source {
        background: #13243b;
        border-left: 3px solid #58a6ff;
        padding: .75rem 1rem;
        border-radius: 8px;
        margin: .5rem 0;
    }

    .muted {
        color: #9fb1ca;
        font-size: .88rem;
    }

    div.stButton > button {
        border-radius: 9px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD LOCAL EMBEDDINGS AND VECTOR DATABASE
# --------------------------------------------------

@st.cache_resource(
    show_spinner="Loading local embedding model (first run may take a few minutes)..."
)
def load_components():

    embeddings = create_embeddings()

    vector_store = get_vector_store(
        embeddings,
        str(CHROMA_DIR)
    )

    return embeddings, vector_store


# --------------------------------------------------
# DOCUMENT COUNT
# --------------------------------------------------

def get_document_count():

    try:
        result = vector_store.get()

        return len(result.get("ids", []))

    except Exception:
        return 0


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🧠 NEXUS AI</h1>
    <p>Enterprise Knowledge Intelligence · Ask questions across your documents</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("## Workspace")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Knowledge Chat",
            "Document Library",
            "Evaluation"
        ],
        label_visibility="collapsed",
    )

    st.divider()


# --------------------------------------------------
# SYSTEM STATUS
# --------------------------------------------------

st.markdown("### System status")

if OPENROUTER_API_KEY and OPENROUTER_MODEL:

    st.success("OpenRouter API configuration detected")

    st.caption(
        f"Active model: {OPENROUTER_MODEL}"
    )

else:

    st.warning("OpenRouter API configuration missing")

    st.caption(
        "Please configure OPENROUTER_API_KEY and OPENROUTER_MODEL in your .env file."
    )

st.caption(
    "Embeddings: all-MiniLM-L6-v2 (local)"
)


# --------------------------------------------------
# INITIALIZE VECTOR DATABASE
# --------------------------------------------------

try:

    embeddings, vector_store = load_components()

except Exception as exc:

    st.error(
        f"Could not initialize the local vector database: {exc}"
    )

    st.info(
        "Check the installation steps in README.md, then restart the app."
    )

    st.stop()


chunk_count = get_document_count()


# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.subheader("Workspace overview")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f'<div class="metric"><div class="muted">Indexed chunks</div><h2>{chunk_count}</h2></div>',
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f'<div class="metric"><div class="muted">Files uploaded this session</div><h2>{len(st.session_state.uploaded_names)}</h2></div>',
            unsafe_allow_html=True
        )

    with c3:

        question_count = sum(
            1 for m in st.session_state.messages
            if m["role"] == "user"
        )

        st.markdown(
            f'<div class="metric"><div class="muted">Chat questions</div><h2>{question_count}</h2></div>',
            unsafe_allow_html=True
        )

    st.markdown("### Get started")

    st.write(
        "1. Open **Document Library** and upload one or more text-based PDF files."
    )

    st.write(
        "2. Wait for indexing to finish."
    )

    st.write(
        "3. Open **Knowledge Chat** and ask a question answered by your documents."
    )

    st.write(
        "4. Check the source references under each answer."
    )

    st.info(
        "Tip: Start with the included sample HR policy PDF. "
        "Scanned/image-only PDFs need OCR, which is not included in this MVP."
    )


# --------------------------------------------------
# KNOWLEDGE CHAT PAGE
# --------------------------------------------------

elif page == "Knowledge Chat":

    st.subheader("Knowledge Chat")

    st.write(
        "Ask questions across your uploaded enterprise documents. "
        "NEXUS AI retrieves relevant information and generates answers "
        "using your knowledge base."
    )

    # Chat controls
    col1, col2 = st.columns([5, 1])

    with col1:
        st.metric("Indexed text chunks", chunk_count)

    with col2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # Display previous conversation
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            # Display source references for assistant answers
            if message["role"] == "assistant":

                sources = message.get("sources", [])

                if sources:

                    with st.expander(
                        f"View sources ({len(sources)})"
                    ):

                        for index, source in enumerate(
                            sources, start=1
                        ):

                            st.markdown(
                                f"**Source {index}: "
                                f"{source.get('file', 'Unknown document')}**"
                            )

                            st.caption(
                                f"Page: {source.get('page', 'Not available')}"
                            )

                            st.write(
                                source.get("preview", "")
                            )

                            st.divider()

    # Question input
    question = st.chat_input(
        "Ask a question about your uploaded documents..."
    )

    if question:

        if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:

            st.error(
                "OpenRouter configuration is missing. "
                "Please check your .env file."
            )

        elif chunk_count == 0:

            st.warning(
                "Your knowledge base is empty. "
                "Please upload and index documents in Document Library first."
            )

        else:

            # Save user's question
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.spinner(
                "NEXUS AI is searching your documents and generating an answer..."
            ):

                try:

                    answer, sources = answer_question(
                        question=question,
                        vector_store=vector_store,
                        api_key=OPENROUTER_API_KEY,
                        model_name=OPENROUTER_MODEL,
                        top_k=4
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as exc:

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Unable to generate an answer: {exc}",
                        "sources": []
                    })

            st.rerun()



# --------------------------------------------------
# DOCUMENT LIBRARY PAGE
# --------------------------------------------------

elif page == "Document Library":

    st.subheader("Document library")

   
    st.write(
        "Upload PDF, Word, text and spreadsheet documents. "
        "Files are processed locally. Retrieved document text is sent "
        "to OpenRouter when you ask a question."
    )

    uploads = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, CSV and XLSX."
    )

    if st.button(
        "Process and index documents",
        type="primary",
        disabled=not uploads
    ):

        progress = st.progress(0)
        status = st.empty()
        successful = 0

        for index, uploaded in enumerate(uploads):

            safe_name = Path(uploaded.name).name

            digest = hashlib.sha256(
                uploaded.getvalue()
            ).hexdigest()[:16]

            saved_path = UPLOAD_DIR / f"{digest}_{safe_name}"

            saved_path.write_bytes(
                uploaded.getvalue()
            )

            try:

                status.write(
                    f"Processing **{safe_name}**..."
                )

                chunks, ids, unit_count, unit_label = process_document(
                    saved_path,
                    safe_name
                )

                added = add_chunks(
                    vector_store,
                    chunks,
                    ids
                )

                successful += 1

                if safe_name not in st.session_state.uploaded_names:
                    st.session_state.uploaded_names.append(safe_name)

                st.success(
                    f"{safe_name}: {unit_count} {unit_label} processed, "
                    f"{added} chunks indexed."
                )

            except Exception as exc:

                st.error(
                    f"{safe_name}: {exc}"
                )

            progress.progress(
                (index + 1) / len(uploads)
            )

        status.write(
            f"Finished. {successful} file(s) processed."
        )

        st.rerun()