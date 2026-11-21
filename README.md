# NEXUS AI 2.0 — Enterprise Knowledge Intelligence

An enterprise-oriented Retrieval-Augmented Generation (RAG) application built with Python, Streamlit, LangChain, ChromaDB, DuckDB and OpenRouter.

NEXUS AI combines unstructured documents, structured business data and web content in one knowledge workspace.

## Key Features

### Document Intelligence
- Upload PDF, DOCX, TXT, CSV and XLSX files.
- Extract, clean and process content.
- Generate local embeddings using Sentence Transformers.
- Store document vectors in ChromaDB.
- Manage indexed documents.

### RAG Knowledge Chat
- Ask questions across indexed knowledge sources.
- Generate answers using OpenRouter through LangChain ChatOpenAI.
- Retrieve relevant chunks and provide source references.
- Inspect retrieval diagnostics and supporting evidence.

### Advanced Retrieval
- Semantic vector search.
- BM25 keyword retrieval.
- Reciprocal Rank Fusion (RRF).
- Multi-query retrieval.
- Multi-hop retrieval.

### Structured Data Intelligence
- Import CSV and Excel datasets.
- Clean data using Pandas.
- Persist structured data in DuckDB.
- Generate and execute natural-language-assisted SQL queries.
- Validate SQL and restrict execution to registered tables and read-only operations.

### Web Knowledge Ingestion
- Ingest supported webpages.
- Extract and index webpage content.
- Manage indexed web sources.

### Research and Reporting
- Generate research reports using retrieved evidence.
- Export reports in Markdown, PDF and CSV formats.

### Evaluation
- Inspect retrieval results and supporting evidence.
- Run expected-keyword coverage checks.
- Automated tests for core application modules.

## Technology Stack

| Layer | Technology |
|---|---|
| Application | Python, Streamlit |
| LLM | OpenRouter, LangChain ChatOpenAI |
| RAG | LangChain |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB |
| Retrieval | Semantic search, BM25, RRF |
| Structured database | DuckDB |
| Data processing | Pandas, OpenPyXL |
| SQL validation | SQLGlot |
| Document processing | PyPDF, python-docx |
| Web ingestion | Requests, BeautifulSoup |
| Reporting | ReportLab |
| Testing | Pytest |

## Architecture

```text
Documents / Structured Data / Web Sources
                  |
          Processing & Cleaning
                  |
        +---------+----------+
        |                    |
   Local Embeddings       DuckDB
        |
     ChromaDB
        |
   BM25 + Hybrid Retrieval
        |
 Multi-Query / Multi-Hop Retrieval
        |
   Retrieved Evidence
        |
    OpenRouter LLM
        |
   Answers + Sources
        |
   Streamlit Interface
```

## Requirements

- Windows 10/11 or compatible Python environment.
- Python 3.14.10 is the development version currently tested.
- Internet access for package installation, initial embedding-model download and OpenRouter API requests.
- OpenRouter API key and an available model identifier.

## Installation — Windows

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure OpenRouter

Create `.env` in the project root using `.env.example` as a template.

```env
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
OPENROUTER_MODEL=YOUR_OPENROUTER_MODEL_ID
```

Keep the actual key private. Never commit `.env` to Git.

### 4. Run NEXUS AI

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## Application Pages

- Workspace Overview
- Knowledge Chat
- Document Library
- Indexed Documents
- Web Sources
- Structured Data Analytics
- Evaluation

## Running Tests

Run the complete test suite:

```powershell
python -m pytest .\tests -v
```

Run with coverage:

```powershell
python -m pytest .\tests --cov=modules --cov-report=term-missing
```

### Development test baseline

The project previously achieved:
- 185 passing automated tests.
- 81% coverage across the `modules` package.

These are development results, not a guarantee of production readiness. Rerun tests after changes.

## Data Storage

- `chroma_db/` — persistent vector database.
- `uploads/` — uploaded documents.
- `data/` — structured database and sample data.

Back up these directories before moving or cleaning the project.

## Security Considerations

- Store credentials in `.env` or supported Streamlit secrets.
- Never commit API keys.
- SQL execution uses read-only validation and registered-table restrictions.
- Treat retrieved document and webpage content as untrusted input.
- Do not send confidential employer, customer, student, employee, medical or financial information to an external LLM without appropriate authorization.
- Review OpenRouter and model-provider data policies before sending sensitive content.

## Known Limitations

- Evaluation currently performs expected-keyword coverage checks. It is not a formal RAGAS or DeepEval benchmark.
- Lexical evidence matching does not independently establish factual correctness.
- Scanned PDFs may require OCR.
- Local ChromaDB and DuckDB storage are not a substitute for production-grade distributed infrastructure.
- Authentication, role-based access control, centralized monitoring and deployment hardening remain separate production considerations.

## Project Status

NEXUS AI 2.0 is an actively developed portfolio project. Core document processing, RAG, hybrid retrieval, structured-data analytics, web ingestion, reporting and automated tests have been implemented.

Remaining work focuses on end-to-end validation, security review, documentation and deployment preparation.

---

Built as a practical enterprise knowledge intelligence project.