# NEXUS AI — Enterprise Knowledge Intelligence

A beginner-friendly Retrieval-Augmented Generation (RAG) capstone MVP.

## Features
- Upload multiple text-based PDF files.
- Extract PDF text and split it into overlapping chunks.
- Generate embeddings locally with Sentence Transformers.
- Store and retrieve chunks using a persistent local ChromaDB collection.
- Use a Gemini model through LangChain to answer questions from retrieved context.
- Display source file names and page references.
- Includes a basic keyword smoke-test page.

## Architecture
PDF → PyPDFLoader → Text Splitter → Sentence Transformers Embeddings → ChromaDB
→ Similarity Search → Gemini LLM → Answer + Source References

## Requirements
- Windows 10/11
- Python 3.11 recommended
- Internet connection for package installation, first embedding-model download and Gemini API calls
- Google AI Studio API key (free tier availability and quotas depend on the account/model)

## Windows 11 setup

### 1. Install Python
Install Python 3.11 from https://www.python.org/downloads/ .
During installation, tick **Add Python to PATH**.

Open Command Prompt and verify:
```bat
py --version
```

### 2. Open the project folder
Extract the ZIP. In File Explorer, open the `NEXUS-AI` folder.
Click the address bar, type `cmd`, and press Enter.

### 3. Create and activate a virtual environment
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```
If PowerShell is used instead of Command Prompt:
```powershell
.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, use Command Prompt or follow Microsoft's execution-policy guidance. Do not change system-wide security settings just for this project.

### 4. Install dependencies
```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```
The first installation can take several minutes because the local embedding model dependencies are sizeable.

### 5. Create a Gemini API key
1. Visit https://aistudio.google.com/
2. Sign in and create an API key.
3. Copy `.env.example` and rename the copy to `.env`.
4. Open `.env` in Notepad and replace `PASTE_YOUR_API_KEY_HERE` with your key.
5. Save the file.

Never share your API key or commit `.env` to GitHub. Check Google's current free-tier model availability and rate limits in AI Studio before selecting a model. If the configured model is unavailable to your account, change `GEMINI_MODEL` in `.env` to a currently available Flash model.

### 6. Start NEXUS AI
```bat
streamlit run app.py
```
The browser should open automatically. If it does not, open the local URL printed in Command Prompt (usually http://localhost:8501).

## First demo
1. Open **Document Library**.
2. Upload `data/sample_documents/sample_hr_policy.pdf`.
3. Click **Process and index documents**.
4. Open **Knowledge Chat**.
5. Ask: `How many casual leave days are provided each year?`
6. Ask: `How much notice must an employee give before resignation?`
7. Expand **View source references** and check the file and page.
8. Ask: `What is the company's international relocation allowance?` The sample policy does not specify one; the assistant should say it cannot find sufficient information.

## Troubleshooting
- **`py` is not recognized:** reinstall Python and tick Add Python to PATH, then reopen Command Prompt.
- **`ModuleNotFoundError`:** activate `.venv` and run `pip install -r requirements.txt`.
- **Gemini authentication/model error:** verify `.env`, API key, model ID and AI Studio availability. Restart Streamlit after changing `.env`.
- **429 / quota error:** you may have reached a free-tier rate limit. Wait and retry later; do not repeatedly submit the same request.
- **No text extracted:** the PDF may be scanned. This MVP does not include OCR; use a text-selectable PDF.
- **First run is slow:** Sentence Transformers downloads its model once and caches it locally.
- **Port is busy:** run `streamlit run app.py --server.port 8502`.

## Evaluation limitation
The Evaluation page performs a simple expected-keyword coverage check. It is NOT RAGAS Faithfulness, Answer Relevance, DeepEval, or a formal benchmark. The assignment's full acceptance criteria require a larger, carefully labelled evaluation set and formal metrics.

## Privacy and API usage
- PDFs are stored in the local project folder.
- Text from retrieved chunks is sent to the Gemini API when you ask a question.
- Do not upload confidential employer, customer, student, employee, medical, financial or other sensitive records.
- Google's free-tier terms, model availability, rate limits and data-use rules can change. Review them before using the API.

## GitHub
Before pushing:
1. Confirm `.env` is ignored by `.gitignore`.
2. Never paste your key into source code, screenshots, README or commits.
3. Run `git status` and inspect files before committing.
