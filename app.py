
import streamlit as st
from pathlib import Path
import hashlib
import shutil
from urllib.parse import urlparse

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    UPLOAD_DIR,
    CHROMA_DIR,
    DUCKDB_PATH
)

from modules.document_processor import process_document
from modules.embeddings import create_embeddings
from modules.vector_store import (
    get_vector_store,
    index_document,
    get_all_documents,
    delete_document,
    get_document_count
)
from modules.rag_pipeline import answer_question_with_diagnostics
from modules.web_ingestion import process_web_page

from modules.vector_store import (
    get_all_web_sources,
    index_web_source,
    delete_web_source
)
from modules.evaluation import simple_grounding_check
from modules.structured_data import (
    list_structured_sources,
    get_database_schemas,
    execute_persistent_query,
    index_structured_file,
    delete_structured_source,
    load_structured_file
)
from modules.nl_sql import (
    is_structured_question,
    generate_sql_from_question
)
from modules.research_report import (
    build_research_report,
    generate_markdown_report
)

from modules.report_export import (
    generate_pdf_report,
    generate_csv_report
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NEXUS AI | Enterprise Knowledge",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>

/* =========================================
   NEXUS AI - ENTERPRISE UI THEME
========================================= */

/* Main application background */
.stApp {
    background: #F1F5F9 !important;
    color: #14243B !important;
}

/* Main content */
[data-testid="stMain"] {
    background: #F1F5F9 !important;
    color: #14243B !important;
}



/* Main headings */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #14243B !important;
    font-weight: 750 !important;
}

/* =========================================
   SIDEBAR / WORKSPACE NAVIGATION
========================================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #14243B 0%,
        #1E3554 100%
    ) !important;
}

[data-testid="stSidebar"] * {
    color: #F8FAFC;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* Sidebar navigation text */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #E2E8F0 !important;
    font-weight: 600;
}

/* =========================================
   INPUT BOXES - FIX INVISIBLE TYPING
========================================= */

.stApp input,
.stApp textarea,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea {
    color: #14243B !important;
    -webkit-text-fill-color: #14243B !important;
    background-color: #FFFFFF !important;
    caret-color: #2563EB !important;
    border-radius: 9px !important;
}

/* Placeholder text */
.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
    opacity: 1 !important;
}

/* Input focus */
.stApp input:focus,
.stApp textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.15) !important;
}

/* Select boxes */
.stApp [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #14243B !important;
}

.stApp [data-baseweb="select"] * {
    color: #14243B !important;
}

/* =========================================
   DASHBOARD CARDS
========================================= */

[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 14px !important;
    box-shadow: 0 4px 14px rgba(15,23,42,0.05);
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #2563EB !important;
    font-weight: 750 !important;
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

/* =========================================
   BUTTONS
========================================= */

.stApp button[kind="primary"] {
    background: linear-gradient(
        90deg,
        #2563EB,
        #0D9488
    ) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 650 !important;
    transition: all 0.2s ease;
}

.stApp button[kind="primary"]:hover {
    box-shadow: 0 5px 15px rgba(37,99,235,0.25);
    transform: translateY(-1px);
}

/* =========================================
   DATAFRAMES AND TABLES
========================================= */

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
}

/* =========================================
   DIVIDERS AND STATUS
========================================= */

hr {
    border-color: #CBD5E1 !important;
}

[data-testid="stCaptionContainer"] {
    color: #64748B !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #2563EB;
    border-radius: 12px;
    padding: 10px;
}

/* Smooth appearance */
.stApp {
    transition: background 0.3s ease;
}

/* ==========================================
   NEXUS AI - CHAT VISIBILITY FIX
========================================== */

/* All chat message containers */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE5F0 !important;
    border-radius: 14px !important;
    padding: 18px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05) !important;
}

/* LLM answer text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #1E293B !important;
}

/* Headings inside AI responses */
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    color: #14243B !important;
    font-weight: 700 !important;
}

/* Markdown bold text */
[data-testid="stChatMessage"] strong {
    color: #14243B !important;
}

/* Code blocks */
[data-testid="stChatMessage"] pre {
    background: #14243B !important;
    border-radius: 10px !important;
    padding: 14px !important;
}

[data-testid="stChatMessage"] pre code {
    color: #E2E8F0 !important;
}

/* Inline code */
[data-testid="stChatMessage"] code {
    background: #EAF1FA !important;
    color: #1D4ED8 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea {
    color: #14243B !important;
    -webkit-text-fill-color: #14243B !important;
    caret-color: #2563EB !important;
}

/* Chat input placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
}

/* Chat avatar and spacing */
[data-testid="stChatMessageAvatar"] {
    border-radius: 50% !important;
}

/* ==========================================
   NEXUS AI - PROFESSIONAL CHAT INTERFACE
========================================== */

/* Chat message card */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE5F0 !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
    margin: 12px 0 !important;
    box-shadow: 0 4px 14px rgba(20,36,59,0.06) !important;
}

/* AI response content */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #1E293B !important;
    background: transparent !important;
    line-height: 1.75 !important;
}

/* All Markdown text */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
    color: #1E293B !important;
}

/* Response headings */
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    color: #14243B !important;
    font-weight: 700 !important;
}

/* Bold content */
[data-testid="stChatMessage"] strong {
    color: #14243B !important;
}

/* Tables */
[data-testid="stChatMessage"] table {
    color: #1E293B !important;
    border-collapse: collapse !important;
}

[data-testid="stChatMessage"] th {
    background: #EAF1FA !important;
    color: #14243B !important;
}

[data-testid="stChatMessage"] td {
    color: #1E293B !important;
}

/* Code blocks */
[data-testid="stChatMessage"] pre {
    background: #14243B !important;
    border-radius: 10px !important;
}

[data-testid="stChatMessage"] pre code {
    color: #F1F5F9 !important;
}

/* User and AI message avatar */
[data-testid="stChatMessageAvatar"] {
    border-radius: 50% !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #14243B !important;
    -webkit-text-fill-color: #14243B !important;
    caret-color: #2563EB !important;
}

/* Chat input placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
}

/* Chat input container */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 14px !important;
}



/* PHASE 8: VISUAL CONTRAST IMPROVEMENTS */

.metric {
    background: #FFFFFF !important;
    border: 1px solid #DCE5F0 !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    box-shadow: 0 4px 14px rgba(20,36,59,0.06) !important;
}

.metric .muted {
    color: #64748B !important;
    font-size: 14px !important;
}

.metric h2 {
    color: #2563EB !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 2.25rem !important;
    font-weight: 800 !important;
}

.hero p {
    color: #E2E8F0 !important;
}
</style>
""", unsafe_allow_html=True)




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

/* ==========================================
   NEXUS PREMIUM SIDEBAR VISIBILITY FIX
   ========================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #101D35 0%,
        #172B49 65%,
        #0C263C 100%
    ) !important;
}

/* Force all sidebar text to remain visible */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* NEXUS branding */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
    color: #FFFFFF !important;
}

/* Navigation labels - visible without hovering */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    color: #F1F5F9 !important;
    background: transparent !important;
    opacity: 1 !important;
    border-radius: 10px !important;
    padding: 11px 12px !important;
    margin-bottom: 5px !important;
    transition: all 0.2s ease-in-out;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label span,
section[data-testid="stSidebar"] div[role="radiogroup"] label p,
section[data-testid="stSidebar"] div[role="radiogroup"] label div {
    color: #F1F5F9 !important;
    opacity: 1 !important;
}

/* Hover effect */
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #29445F !important;
    transform: translateX(3px);
}

/* Selected module */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(
        90deg,
        #087F8C,
        #0D9DA4
    ) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
}

/* Sidebar dividers */
section[data-testid="stSidebar"] hr {
    border-color: #34445E !important;
}

/* ==========================================
   PREMIUM DASHBOARD VISUAL EFFECTS
   ========================================== */

/* Subtle dashboard background */
[data-testid="stMain"] {
    background: linear-gradient(
        135deg,
        #EAF0F7 0%,
        #F4F7FB 50%,
        #E8F1F6 100%
    ) !important;
}

/* Premium metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #DFE8F0 !important;
    border-radius: 15px !important;
    padding: 20px !important;
    box-shadow: 0 5px 18px rgba(20, 45, 75, 0.06) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(20, 45, 75, 0.12) !important;
}

/* Attractive buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        110deg,
        #087F8C,
        #0B9FA4
    ) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
    box-shadow: 0 4px 12px rgba(8, 127, 140, 0.18);
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(
        110deg,
        #066874,
        #087F8C
    ) !important;
    transform: translateY(-2px);
    box-shadow: 0 7px 18px rgba(8, 127, 140, 0.25);
}

/* Main headings */
[data-testid="stMain"] h1 {
    color: #142B47 !important;
    font-weight: 800 !important;
}

/* Polished expanders */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #DFE8F0 !important;
    border-radius: 12px !important;
}

/* Better upload area */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px dashed #80AAB9 !important;
    border-radius: 14px !important;
}

/* Smooth page appearance */
[data-testid="stMain"] .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}



/* PHASE 8: VISUAL CONTRAST IMPROVEMENTS */

.metric {
    background: #FFFFFF !important;
    border: 1px solid #DCE5F0 !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    box-shadow: 0 4px 14px rgba(20,36,59,0.06) !important;
}

.metric .muted {
    color: #64748B !important;
    font-size: 14px !important;
}

.metric h2 {
    color: #2563EB !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 2.25rem !important;
    font-weight: 800 !important;
}

.hero p {
    color: #E2E8F0 !important;
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
    <h1>NEXUS AI</h1>
    <p>Enterprise Knowledge Intelligence &middot; Ask questions across your documents</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# REUSABLE PAGE HEADER FUNCTION
# PASTE STEP 2 HERE
# --------------------------------------------------
def render_page_header(title, subtitle):
    st.markdown(f"""
    <div style="
        padding: 8px 0 22px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid #E2E8F0;
    ">
        <div style="
            font-size: 29px;
            font-weight: 800;
            color: #172B49;
            letter-spacing: -0.7px;
        ">{title}</div>
        <div style="
            font-size: 14px;
            color: #64748B;
            margin-top: 5px;
        ">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)



# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div style="padding: 12px 4px 22px 4px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="
                background:#087F8C;
                border-radius:12px;
                padding:10px;
                font-size:23px;
            ">&#10022;</div>
            <div>
                <div style="
                    font-size:23px;
                    font-weight:800;
                    color:#FFFFFF;
                    letter-spacing:1px;
                ">NEXUS AI</div>
                <div style="
                    font-size:10px;
                    color:#9FB6CE;
                    letter-spacing:2px;
                ">ENTERPRISE COPILOT 2.0</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        color:#8FA7C2;
        font-size:11px;
        font-weight:700;
        letter-spacing:1.5px;
        margin-bottom:12px;
    ">WORKSPACE NAVIGATION</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Knowledge Chat",
            "Document Library",
            "Structured Data",
            "Web Sources",
            "Evaluation",
        ],
        label_visibility="collapsed",
    )

    st.divider()

# --------------------------------------------------
# SYSTEM STATUS
# --------------------------------------------------

with st.sidebar:

    st.markdown("### System Status")

    if OPENROUTER_API_KEY and OPENROUTER_MODEL:

        st.success("API configuration ready")

        st.caption(f"Active model: {OPENROUTER_MODEL}")

    else:

        st.warning("API configuration missing")

        st.caption(
            "Configure OPENROUTER_API_KEY and OPENROUTER_MODEL in .env."
        )

    st.caption("Embeddings: all-MiniLM-L6-v2 (local)")


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


chunk_count = get_document_count(vector_store)

# Vector database status in sidebar
with st.sidebar:

    st.success("Vector database connected")

    st.caption(
        f"Indexed knowledge chunks: {chunk_count:,}"
    )


# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    render_page_header("Workspace Overview", "Monitor your knowledge workspace and system activity.")

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

    render_page_header("Knowledge Chat", "Ask questions across your documents, webpages and structured data.")

    st.write(
        "Ask questions across enterprise documents, webpages "
        "and structured Excel/CSV data."
    )

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

            # Display SQL results when available
            dataframe = message.get("dataframe")

            if dataframe is not None:
                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True
                )

                with st.expander("View executed SQL"):
                    st.code(message.get("sql", ""), language="sql")

                if message.get("truncated"):
                    st.warning("Results were limited to 500 rows.")
            
            # Display advanced retrieval diagnostics
            if message["role"] == "assistant":

                diagnostics = message.get("diagnostics")

                if diagnostics:

                    with st.expander(
                        "Retrieval Diagnostics",
                        expanded=False
                    ):

                        st.markdown("### Retrieval Overview")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Search Queries",
                            diagnostics.get("initial", {}).get(
                                "query_count", 0
                            )
                        )

                        col2.metric(
                            "Retrieval Hops",
                            diagnostics.get("hops_completed", 0)
                        )

                        col3.metric(
                            "Evidence Chunks",
                            diagnostics.get("retrieved_chunks", 0)
                        )

                        st.caption(
                            f"Strategy: {diagnostics.get('strategy', 'Not available')}"
                        )

                        st.caption(
                            f"Unique Sources: {diagnostics.get('unique_sources', 0)}"
                        )

                        st.markdown("### Initial Search Queries")

                        initial_queries = diagnostics.get(
                            "initial", {}
                        ).get("queries", [])

                        if initial_queries:

                            for index, item in enumerate(
                                initial_queries, start=1
                            ):
                                st.markdown(
                                    f"**Query {index}:** {item.get('query', '')}"
                                )

                                st.caption(
                                    f"Retrieved chunks: {item.get('results_found', 0)}"
                                )

                        else:
                            st.info("No initial query details available.")

                        st.markdown("### Follow-up Retrieval")

                        followup = diagnostics.get("followup", {})

                        if followup.get("attempted"):

                            st.write(
                                f"**Follow-up query:** "
                                f"{followup.get('query', '')}"
                            )

                            st.caption(
                                f"Follow-up results: "
                                f"{followup.get('results_found', 0)}"
                            )

                        else:
                            st.info(
                                "No follow-up retrieval was performed."
                            )

                        st.markdown("### Evidence Assessment")

                        st.caption(
                            diagnostics.get(
                                "evidence_sufficiency",
                                "Not assessed"
                            )
                        )

                        evidence = diagnostics.get("evidence", [])

                        if evidence:

                            for item in evidence:

                                st.markdown(
                                    f"**Evidence {item['number']}: "
                                    f"{item['source']}**"
                                )

                                location = []

                                if item.get("page") is not None:
                                    location.append(
                                        f"Page {item['page']}"
                                    )

                                if item.get("row") is not None:
                                    location.append(
                                        f"Row {item['row']}"
                                    )

                                if item.get("sheet"):
                                    location.append(
                                        f"Sheet {item['sheet']}"
                                    )

                                if location:
                                    st.caption(" | ".join(location))

                                st.write(item.get("preview", ""))

                                st.divider()
            
            # Phase 6: Claim-level Citation & Evidence Verification
            if message["role"] == "assistant":

                diagnostics = message.get("diagnostics") or {}
                citation_report = diagnostics.get("citation_report", {})

                if citation_report:
                    with st.expander(
                        "Claim & Evidence Verification",
                        expanded=False
                    ):
                        st.markdown("### Citation Overview")

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Total Claims",
                            citation_report.get("total_claims", 0)
                        )

                        col2.metric(
                            "Retrieved Sources",
                            citation_report.get("total_sources", 0)
                        )

                        st.caption(
                            "Evidence scores indicate lexical overlap, "
                            "not confirmed factual accuracy."
                        )

                        for claim in citation_report.get("claims", []):

                            status = claim.get(
                                "verification_status",
                                "unverified"
                            )

                            status_labels = {
                                "candidate_support": "Candidate support",
                                "insufficient_evidence": "Insufficient evidence",
                                "unverified": "Unverified"
                            }

                            score = float(
                                claim.get("best_evidence_score", 0)
                            )

                            score = max(0.0, min(1.0, score))

                            with st.expander(
                                f"{claim.get('claim_id', 'Claim')} | "
                                f"{status_labels.get(status, status)}"
                            ):

                                st.write(claim.get("claim", ""))

                                st.caption(
                                    f"Evidence relevance: {score:.1%}"
                                )

                                st.progress(score)

                                evidence_matches = claim.get(
                                    "evidence_matches", []
                                )

                                if evidence_matches:

                                    st.markdown("**Supporting evidence candidates**")

                                    for evidence in evidence_matches:

                                        st.markdown(
                                            f"**{evidence.get('source_id', 'Source')}**"
                                        )

                                        st.write(
                                            evidence.get("source_name", "Unknown")
                                        )

                                        st.caption(
                                            f"Relevance: "
                                            f"{evidence.get('relevance_score', 0):.1%}"
                                        )

                                        st.write(
                                            evidence.get("evidence", "")
                                        )

                                        st.divider()

                                else:
                                    st.warning(
                                        "No matching evidence was found."
                                    )


            # Display document and webpage sources
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

                            if source.get("source_type") == "web_page":

                                st.caption("Source type: Public webpage")

                                if source.get("url"):
                                    st.markdown(
                                        f"[Open original webpage]({source['url']})"
                                    )

                            else:

                                st.caption(
                                    f"Source type: Uploaded document | "
                                    f"Page: {source.get('page', 'Not available')}"
                                )

                                if source.get("row") != "Unknown":
                                    st.caption(
                                        f"Row: {source.get('row')} | "
                                        f"Sheet: {source.get('sheet', 'Not available')}"
                                    )

                            st.divider()
    
    # --------------------------------------------------
    # PHASE 7: RESEARCH REPORT EXPORTS
    # --------------------------------------------------

    messages = st.session_state.get("messages", [])

    assistant_index = next(
        (
            i for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "assistant"
        ),
        None
    )

    if assistant_index is not None:

        latest_assistant = messages[assistant_index]

        latest_question = next(
            (
                messages[i].get("content", "")
                for i in range(assistant_index - 1, -1, -1)
                if messages[i].get("role") == "user"
            ),
            ""
        )

        latest_answer = latest_assistant.get("content", "")

        latest_sources = latest_assistant.get("sources", [])

        latest_diagnostics = (
            latest_assistant.get("diagnostics") or {}
        )

        latest_citation_report = (
            latest_diagnostics.get("citation_report", {})
        )

        research_report = build_research_report(
            question=latest_question,
            answer=str(latest_answer),
            sources=latest_sources,
            citation_report=latest_citation_report
        )

        markdown_data = generate_markdown_report(research_report)

        pdf_data = generate_pdf_report(research_report)

        csv_data = generate_csv_report(research_report)

        with st.expander(
            "Download Research Report",
            expanded=False
        ):

            st.caption(
                "Export the latest assistant response, "
                "its evidence analysis and available source references."
            )

            col1, col2, col3 = st.columns(3)

            col1.download_button(
                label="Download Markdown",
                data=markdown_data,
                file_name="nexus_research_report.md",
                mime="text/markdown",
                key="nexus_download_markdown"
            )

            col2.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name="nexus_research_report.pdf",
                mime="application/pdf",
                key="nexus_download_pdf"
            )

            col3.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="nexus_claim_evidence.csv",
                mime="text/csv",
                key="nexus_download_csv"
            )

    # Question input
    question = st.chat_input(
        "Ask about documents, webpages, staff data or analytics..."
    )

    if question:

        if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:

            st.error(
                "OpenRouter configuration is missing. "
                "Please check your configuration."
            )

        else:

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            try:

                # Route structured-data questions to DuckDB
                schemas = get_database_schemas(DUCKDB_PATH)

                
                use_structured = (
                    is_structured_question(question) and bool(schemas)
                )
                result = None

                if use_structured:
                    try:
                        with st.spinner(
                            "NEXUS AI is generating SQL and analysing structured data..."
                        ):
                            result = generate_sql_from_question(
                                question=question,
                                database_path=DUCKDB_PATH,
                                api_key=OPENROUTER_API_KEY,
                                model_name=OPENROUTER_MODEL
                            )

                    except ValueError as exc:
                        if (
                            "available structured data cannot answer this question"
                            in str(exc).lower()
                        ):
                            use_structured = False
                        else:
                            raise

                if use_structured:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            "I analysed your structured data using DuckDB. "
                            f"Here are the results ({result['returned_rows']} rows returned)."
                        ),
                        "dataframe": result["dataframe"],
                        "sql": result["sql"],
                        "truncated": result["truncated"],
                        "sources": []
                    })


                   
                
                # Route other questions to document RAG
                elif chunk_count > 0:

                    with st.spinner(
                        "NEXUS AI is performing multi-hop retrieval and generating an answer..."
                    ):

                        answer, sources, diagnostics = (
                            answer_question_with_diagnostics(
                                question=question,
                                vector_store=vector_store,
                                api_key=OPENROUTER_API_KEY,
                                model_name=OPENROUTER_MODEL,
                                top_k=4
                            )
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "diagnostics": diagnostics
                    })

                else:

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            "No relevant indexed data is available. "
                            "Please index documents or structured files first."
                        ),
                        "sources": []
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

    render_page_header("Document Library", "Upload, process and manage your knowledge documents.")

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
        skipped = 0
        failed = 0
        partial_warnings = 0

        failed_files = []
        structured_warnings = []

        for index, uploaded in enumerate(uploads):

            safe_name = Path(uploaded.name).name
            file_bytes = uploaded.getvalue()

            digest = hashlib.sha256(
                file_bytes
            ).hexdigest()[:16]

            saved_path = UPLOAD_DIR / f"{digest}_{safe_name}"

            try:
                saved_path.write_bytes(file_bytes)

                status.write(
                    f"Processing **{safe_name}**..."
                )

                # Validate structured files BEFORE vector indexing.
                is_structured = saved_path.suffix.lower() in [
                    ".csv",
                    ".xlsx"
                ]

                if is_structured:
                    load_structured_file(
                        file_path=saved_path,
                        original_name=safe_name
                    )

                chunks, ids, unit_count, unit_label = process_document(
                    saved_path,
                    safe_name
                )

                result = index_document(
                    vector_store=vector_store,
                    chunks=chunks,
                    ids=ids,
                    source=safe_name,
                    file_id=digest
                )

                # Index structured data separately so a DuckDB error
                # does not incorrectly classify vector indexing as failed.
                structured_result = None
                structured_error = None

                if is_structured:
                    try:
                        structured_result = index_structured_file(
                            file_path=saved_path,
                            original_name=safe_name,
                            database_path=DUCKDB_PATH,
                            file_id=digest
                        )
                    except Exception as exc:
                        structured_error = str(exc)

                if result["status"] == "duplicate":
                    skipped += 1

                    st.info(
                        f"{safe_name}: Identical document already exists. "
                        "Vector indexing skipped."
                    )

                elif result["status"] == "replaced":
                    successful += 1

                    st.success(
                        f"{safe_name}: Updated successfully. "
                        f"{result['added']} new chunks indexed and "
                        f"{result['removed']} old chunks replaced."
                    )

                else:
                    successful += 1

                    st.success(
                        f"{safe_name}: {unit_count} {unit_label} processed, "
                        f"{result['added']} chunks indexed."
                    )

                if structured_result:
                    st.success(
                        f"DuckDB indexing complete: "
                        f"{structured_result['table_count']} table(s), "
                        f"{structured_result['row_count']} rows."
                    )

                if structured_error:
                    partial_warnings += 1

                    structured_warnings.append({
                        "file": safe_name,
                        "error": structured_error
                    })

                    st.warning(
                        f"{safe_name}: Vector indexing completed, "
                        "but structured-data indexing failed. "
                        "Check the warning details below."
                    )

                if safe_name not in st.session_state.uploaded_names:
                    st.session_state.uploaded_names.append(safe_name)

            except Exception as exc:
                failed += 1

                failed_files.append({
                    "file": safe_name,
                    "error": str(exc)
                })

                st.error(
                    f"{safe_name}: {exc}"
                )

            progress.progress(
                (index + 1) / len(uploads)
            )

        status.write("Document processing completed.")

        summary_col1, summary_col2, summary_col3, summary_col4 = (
            st.columns(4)
        )

        summary_col1.metric("Indexed / Updated", successful)
        summary_col2.metric("Duplicates Skipped", skipped)
        summary_col3.metric("Partial DB Warnings", partial_warnings)
        summary_col4.metric("Failed Files", failed)

        if successful > 0:
            st.success(
                f"Successfully indexed or updated {successful} file(s)."
            )

        if skipped > 0:
            st.info(
                f"{skipped} duplicate file(s) were skipped in vector indexing."
            )

        if partial_warnings > 0:
            st.warning(
                f"{partial_warnings} file(s) have incomplete structured-data indexing."
            )

            with st.expander("View structured-data warnings"):
                for warning in structured_warnings:
                    st.markdown(f"**{warning['file']}**")
                    st.caption(warning["error"])

        if failed > 0:
            st.warning(
                f"{failed} file(s) could not be processed."
            )

            with st.expander("View failed files and error details"):
                for failure in failed_files:
                    st.markdown(f"**{failure['file']}**")
                    st.caption(failure["error"])

        st.rerun()


    st.divider()

    render_page_header("Indexed Documents", "Inspect and manage documents stored in your knowledge base.")

    documents = get_all_documents(vector_store)

    st.metric(
        "Total documents",
        len(documents)
    )

    st.metric(
        "Total indexed chunks",
        get_document_count(vector_store)
    )

    
    if documents:

        for document in documents:

            with st.expander(
                f"{document['source']} "
                f"({document['file_type'].upper()})"
            ):

                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        f"**Chunks:** {document['chunk_count']}"
                    )

                with col2:
                    st.write(
                        f"**File ID:** {document['file_id']}"
                    )

                if st.button(
                    f"Delete {document['source']}",
                    key=f"delete_{document['source']}"
                ):

                    st.session_state[
                        f"confirm_delete_{document['file_id']}"
                    ] = True

                if st.session_state.get(
                    f"confirm_delete_{document['file_id']}",
                    False
                ):

                    st.warning(
                        f"Are you sure you want to delete {document['source']}? "
                        "This removes its indexed knowledge chunks and "
                        "any associated structured-data tables."
                    )

                    confirm_col, cancel_col = st.columns(2)

                    with confirm_col:

                        if st.button(
                            "Confirm deletion",
                            key=f"confirm_yes_{document['file_id']}",
                            type="primary"
                        ):

                            try:
                                # Remove associated DuckDB tables first.
                                structured_tables_removed = delete_structured_source(
                                    DUCKDB_PATH,
                                    document["source"]
                                )

                                # Remove document chunks from Chroma.
                                removed = delete_document(
                                    vector_store,
                                    document["source"]
                                )

                                st.session_state[
                                    f"confirm_delete_{document['file_id']}"
                                ] = False

                                st.success(
                                    f"Deleted {removed} vector chunks and "
                                    f"{structured_tables_removed} DuckDB tables."
                                )

                                st.rerun()

                            except Exception as exc:
                                st.error(
                                    f"Document deletion failed: {exc}"
                                )

                    with cancel_col:

                        if st.button(
                            "Cancel",
                            key=f"confirm_no_{document['file_id']}"
                        ):

                            st.session_state[
                                f"confirm_delete_{document['file_id']}"
                            ] = False

                            st.rerun()

    else:

        st.info("No indexed documents found.")

# --------------------------------------------------
# WEB SOURCES PAGE
# --------------------------------------------------

elif page == "Web Sources":

    render_page_header("Web Sources", "Ingest and manage knowledge from web pages.")

    st.write(
        "Connect public websites to your NEXUS AI knowledge base. "
        "Webpage content is extracted, cleaned, indexed and made "
        "searchable alongside your uploaded documents."
    )

    st.info(
        "Only public HTTP/HTTPS webpages are supported. "
        "Internal and private network destinations are blocked."
    )

    st.divider()

    st.markdown("### Add a website")

    website_url = st.text_input(
        "Website URL",
        placeholder="https://example.com/about",
        help="Enter the complete public webpage URL, including https://"
    )
    website_url_valid = False

    if website_url.strip():

        parsed_url = urlparse(website_url.strip())

        website_url_valid = (
            parsed_url.scheme in ("http", "https")
            and bool(parsed_url.netloc)
            and "." in parsed_url.netloc
        )

        if website_url_valid:
            st.caption("URL format looks valid.")
        else:
            st.warning(
                "Enter a valid public HTTP/HTTPS URL, "
                "for example https://example.com/about"
            )

    if st.button(
        "Fetch and index webpage",
        type="primary",
        disabled=not website_url_valid
    ):

        try:

            with st.spinner(
                "Fetching webpage, extracting content and creating embeddings..."
            ):

                chunks, ids, metadata = process_web_page(
                    website_url.strip()
                )

                result = index_web_source(
                    vector_store=vector_store,
                    chunks=chunks,
                    ids=ids,
                    url=metadata["url"],
                    file_id=metadata["file_id"]
                )

            if result["status"] == "unchanged":

                st.info(
                    "This webpage is already indexed and its content "
                    "has not changed."
                )

            elif result["status"] == "added":

                st.success(
                    f"Successfully indexed: {metadata['title']}"
                )

                st.write(
                    f"New chunks indexed: {result['added']}"
                )

            elif result["status"] == "refreshed":

                st.success(
                    f"Successfully refreshed: {metadata['title']}"
                )

                st.write(
                    f"New chunks: {result['added']} | "
                    f"Old chunks replaced: {result['removed']}"
                )

        except Exception as exc:

            st.error(
                f"Unable to index webpage: {exc}"
            )

    st.divider()

    st.markdown("### Indexed web sources")

    web_sources = get_all_web_sources(vector_store)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Indexed webpages",
            len(web_sources)
        )

    with col2:
        st.metric(
            "Total knowledge chunks",
            get_document_count(vector_store)
        )

    if web_sources:

        for index, source in enumerate(web_sources):

            title = source.get("title") or source["source"]

            with st.expander(
                f"{title} ({source['chunk_count']} chunks)"
            ):

                st.caption("Website URL")

                st.markdown(
                    f"[{source['url']}]({source['url']})"
                )

                st.write(
                    f"**Content ID:** {source.get('file_id', 'Unknown')}"
                )

                st.write(
                    f"**Last retrieved:** "
                    f"{source.get('retrieved_at', 'Unknown')}"
                )

                col_refresh, col_delete = st.columns(2)

                with col_refresh:

                    refresh_clicked = st.button(
                        "Refresh webpage",
                        key=f"refresh_web_{index}",
                        use_container_width=True
                    )

                with col_delete:

                    delete_clicked = st.button(
                        "Delete webpage",
                        key=f"delete_web_{index}",
                        use_container_width=True
                    )

                if refresh_clicked:

                    try:

                        with st.spinner("Refreshing webpage..."):

                            chunks, ids, metadata = process_web_page(
                                source["url"]
                            )

                            result = index_web_source(
                                vector_store=vector_store,
                                chunks=chunks,
                                ids=ids,
                                url=metadata["url"],
                                file_id=metadata["file_id"]
                            )

                        if result["status"] == "unchanged":

                            st.info("Webpage content has not changed.")

                        else:

                            st.success(
                                f"Webpage refreshed. "
                                f"{result['added']} chunks indexed."
                            )

                        st.rerun()

                    except Exception as exc:

                        st.error(
                            f"Refresh failed: {exc}"
                        )

                if delete_clicked:

                    st.session_state[
                        f"confirm_delete_web_{index}"
                    ] = True

                if st.session_state.get(
                    f"confirm_delete_web_{index}",
                    False
                ):

                    st.warning(
                        f"Are you sure you want to delete "
                        f"{source['url']}? This removes its indexed "
                        "knowledge chunks."
                    )

                    confirm_col, cancel_col = st.columns(2)

                    with confirm_col:

                        if st.button(
                            "Confirm deletion",
                            key=f"confirm_delete_web_yes_{index}",
                            type="primary"
                        ):

                            try:

                                removed = delete_web_source(
                                    vector_store,
                                    source["source"]
                                )

                                st.session_state[
                                    f"confirm_delete_web_{index}"
                                ] = False

                                st.success(
                                    f"Deleted webpage and "
                                    f"{removed} indexed chunks."
                                )

                                st.rerun()

                            except Exception as exc:

                                st.error(
                                    f"Unable to delete webpage: {exc}"
                                )

                    with cancel_col:

                        if st.button(
                            "Cancel",
                            key=f"confirm_delete_web_no_{index}"
                        ):

                            st.session_state[
                                f"confirm_delete_web_{index}"
                            ] = False

                            st.rerun()

    else:

        st.info(
            "No webpages indexed yet. Add a public website URL above."
        )

# --------------------------------------------------
# STRUCTURED DATA ANALYTICS PAGE
# --------------------------------------------------

elif page == "Structured Data":

    render_page_header("Structured Data Analytics", "Query and explore your structured business data.")

    st.write(
        "Explore indexed CSV and Excel files, inspect worksheet schemas, "
        "and execute read-only SQL analytics against your DuckDB database."
    )

    st.caption(f"Analytics database: {DUCKDB_PATH}")

    st.warning(
        "Some indexed spreadsheets contain personal staff information. "
        "Use aggregate queries for reporting and avoid displaying "
        "individual contact details or addresses."
    )

    st.divider()

    # --------------------------------------------------
    # LOAD STRUCTURED DATA INVENTORY
    # --------------------------------------------------

    try:
        structured_sources = list_structured_sources(DUCKDB_PATH)
        schemas = get_database_schemas(DUCKDB_PATH)

    except Exception as exc:
        st.error(f"Unable to load analytics database: {exc}")
        structured_sources = []
        schemas = {}

    if not structured_sources:

        st.info(
            "No structured files have been indexed yet. "
            "Upload CSV or Excel files through Document Library."
        )

    else:

        # --------------------------------------------------
        # SUMMARY METRICS
        # --------------------------------------------------

        total_files = len(
            set(source["source"] for source in structured_sources)
        )

        total_tables = len(structured_sources)

        total_rows = sum(
            source["row_count"] for source in structured_sources
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric("Structured files", total_files)

        with metric2:
            st.metric("Worksheet tables", total_tables)

        with metric3:
            st.metric("Total indexed rows", f"{total_rows:,}")

        st.divider()

        # --------------------------------------------------
        # SOURCE AND WORKSHEET INVENTORY
        # --------------------------------------------------

        st.markdown("### Indexed spreadsheet inventory")

        for source_name in sorted(
            set(item["source"] for item in structured_sources)
        ):

            source_tables = [
                item for item in structured_sources
                if item["source"] == source_name
            ]

            with st.expander(
                f"{source_name} ({len(source_tables)} worksheets)"
            ):

                for item in source_tables:

                    schema = schemas.get(item["table_name"], {})

                    st.markdown(
                        f"**Worksheet:** {item['sheet_name']}"
                    )

                    st.write(
                        f"Rows: {item['row_count']:,}"
                    )

                    st.code(
                        item["table_name"],
                        language="text"
                    )

                    columns = schema.get("columns", [])

                    if columns:

                        st.dataframe(
                            [
                                {
                                    "Column": column["name"],
                                    "Data type": column["dtype"]
                                }
                                for column in columns
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                    st.caption(
                        f"Indexed: {item['indexed_at']}"
                    )

                    st.divider()

        # --------------------------------------------------
        # SQL ANALYTICS WORKSPACE
        # --------------------------------------------------

        st.markdown("### SQL Analytics")

        st.write(
            "Select an indexed worksheet table, write a SQL SELECT query, "
            "and execute it against DuckDB."
        )

        table_names = list(schemas.keys())

        selected_table = st.selectbox(
            "Select worksheet table",
            options=table_names,
            format_func=lambda name: (
                f"{schemas[name]['source']} | "
                f"{schemas[name]['sheet_name']}"
            ),
            key="nexus_selected_analytics_table"
        )

        selected_schema = schemas[selected_table]

        st.caption(
            f"Source: {selected_schema['source']} | "
            f"Worksheet: {selected_schema['sheet_name']} | "
            f"Rows: {selected_schema['row_count']:,}"
        )

        st.markdown("**Available columns**")

        st.write(
            ", ".join(
                column["name"]
                for column in selected_schema["columns"]
            )
        )

        default_query = (
            f"SELECT COUNT(*) AS total_rows\n"
            f"FROM {selected_table};"
        )

        sql_query = st.text_area(
            "SQL query",
            value=default_query,
            height=160,
            key="nexus_sql_editor",
            help=(
                "Only one read-only SELECT query is permitted. "
                "Queries can access registered NEXUS tables only."
            )
        )

        if st.button(
            "Run SQL query",
            type="primary",
            key="nexus_run_sql"
        ):

            try:

                with st.spinner("Executing analytics query..."):

                    query_result = execute_persistent_query(
                        database_path=DUCKDB_PATH,
                        sql=sql_query,
                        max_rows=500
                    )

                st.session_state["nexus_last_sql_result"] = query_result
                st.session_state["nexus_last_sql_error"] = None

            except Exception as exc:

                st.session_state["nexus_last_sql_error"] = str(exc)
                st.session_state.pop(
                    "nexus_last_sql_result",
                    None
                )

        # --------------------------------------------------
        # DISPLAY QUERY RESULTS
        # --------------------------------------------------

        query_error = st.session_state.get(
            "nexus_last_sql_error"
        )

        query_result = st.session_state.get(
            "nexus_last_sql_result"
        )

        if query_error:

            st.error(f"Query failed: {query_error}")

        if query_result:

            result_df = query_result["dataframe"]

            st.divider()

            st.markdown("### Query Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Rows returned",
                    query_result["returned_rows"]
                )

            with result_col2:
                st.metric(
                    "Columns returned",
                    len(query_result["columns"])
                )

            if query_result["truncated"]:

                st.warning(
                    "Results were limited to 500 rows."
                )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            with st.expander("Executed SQL"):
                st.code(
                    query_result["sql"],
                    language="sql"
                )

            # --------------------------------------------------
            # BASIC CHART GENERATION
            # --------------------------------------------------

            numeric_columns = result_df.select_dtypes(
                include="number"
            ).columns.tolist()

            categorical_columns = [
                column
                for column in result_df.columns
                if column not in numeric_columns
            ]

            if numeric_columns and categorical_columns and len(result_df) > 0:

                st.divider()

                st.markdown("### Visual Analytics")

                chart_category = st.selectbox(
                    "Category",
                    categorical_columns,
                    key="nexus_chart_category"
                )

                chart_value = st.selectbox(
                    "Numeric value",
                    numeric_columns,
                    key="nexus_chart_value"
                )

                chart_data = result_df[
                    [chart_category, chart_value]
                ].copy()

                chart_data[chart_value] = (
                    chart_data[chart_value]
                    .apply(
                        lambda value: (
                            float(value)
                            if value is not None
                            else 0
                        )
                    )
                )

                chart_data = chart_data.dropna(
                    subset=[chart_category]
                )

                chart_data = chart_data.set_index(
                    chart_category
                )

                st.bar_chart(
                    chart_data[chart_value]
                )

            elif len(result_df) > 0:

                st.caption(
                    "This result does not contain both a category "
                    "and numeric column for a basic bar chart."
                )
        




# --------------------------------------------------
# EVALUATION PAGE
# --------------------------------------------------

elif page == "Evaluation":

    render_page_header(
        "RAG Evaluation",
        "Measure answer grounding against expected keywords and phrases."
    )

    st.write(
        "Evaluate whether an AI-generated answer contains the expected "
        "information. This is a transparent keyword-based smoke test, "
        "not a formal semantic or RAGAS evaluation."
    )

    st.divider()

    st.markdown("### Evaluation Workspace")

    answer_text = st.text_area(
        "AI-generated answer",
        placeholder="Paste the answer you want to evaluate...",
        height=220,
        key="nexus_eval_answer"
    )

    expected_terms_text = st.text_area(
        "Expected keywords or phrases",
        placeholder="Enter one expected term per line, for example:\n"
                    "machine learning\n"
                    "vector database\n"
                    "retrieval augmented generation",
        height=150,
        key="nexus_eval_expected_terms"
    )

    if st.button(
        "Run Evaluation",
        type="primary",
        key="nexus_run_evaluation"
    ):

        if not answer_text.strip():

            st.warning("Please enter an AI-generated answer.")

        elif not expected_terms_text.strip():

            st.warning("Please enter at least one expected keyword or phrase.")

        else:

            expected_terms = [
                term.strip()
                for term in expected_terms_text.splitlines()
                if term.strip()
            ]

            try:

                result = simple_grounding_check(
                    answer=answer_text,
                    expected_terms=expected_terms
                )

                st.session_state["nexus_eval_result"] = result

            except Exception as exc:

                st.error(f"Evaluation failed: {exc}")
                st.session_state.pop("nexus_eval_result", None)

    result = st.session_state.get("nexus_eval_result")

    if result:

        st.divider()

        st.markdown("### Evaluation Results")

        score = result["score"]

        if score is not None:

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "Grounding score",
                    f"{score * 100:.1f}%"
                )

            with metric2:
                st.metric(
                    "Matched terms",
                    len(result["matched"])
                )

            with metric3:
                st.metric(
                    "Missing terms",
                    len(result["missing"])
                )

            st.progress(score)

            if score == 1:

                st.success("All expected terms were found in the answer.")

            elif score >= 0.5:

                st.warning(
                    "Some expected terms were found, but the answer "
                    "is missing information."
                )

            else:

                st.error(
                    "Low keyword coverage. Most expected terms "
                    "were not found."
                )

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.markdown("#### Matched terms")

                if result["matched"]:
                    for term in result["matched"]:
                        st.success(term)
                else:
                    st.info("No expected terms matched.")

            with result_col2:

                st.markdown("#### Missing terms")

                if result["missing"]:
                    for term in result["missing"]:
                        st.error(term)
                else:
                    st.success("No missing terms.")

            st.caption(
                "Score = matched expected terms / total expected terms. "
                "This measures literal keyword presence only; it does not "
                "verify factual correctness or semantic relevance."
            )
