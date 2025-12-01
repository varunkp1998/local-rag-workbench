# RAG Workbench – Local

RAG Workbench is a compact, production-style Retrieval-Augmented Generation workspace
that runs entirely on your machine:

- Local semantic search using FAISS and sentence-transformer embeddings.
- Local LLM generation using Hugging Face `transformers` (no external APIs).
- Clean, two-step UX:
  - **Knowledge Base** view for ingestion.
  - **Assistant** view for querying and analysis.
- Multiple response modes:
  - **Standard** – concise answer.
  - **Explain & justify** – answer plus short reasoning and a confidence label.
  - **Ask me questions** – generates probing questions instead of an answer.

It is inspired by the classic “simple RAG from scratch” tutorials, but packaged as a
more universal, professional-feeling workbench.

## 1. Project Structure

```text
rag-workbench-local/
├─ backend/
│  ├─ app.py               # FastAPI app exposing /api/ingest and /api/query
│  ├─ rag_pipeline.py      # Core RAG pipeline (local LLM + FAISS)
│  └─ requirements.txt
├─ frontend/
│  ├─ index.html           # Single-page UI with sidebar navigation
│  ├─ style.css
│  └─ app.js
└─ README.md
```

## 2. Dependencies

- Python 3.9+.
- No Node/React – the frontend is plain HTML/CSS/JS.
- Internet connection only needed the first time to download:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - Your chosen local LLM (default `microsoft/phi-2`).

From `backend/`:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
# Install torch for your platform (example for CPU-only):
pip install torch torchvision torchaudio
```

Select a local model (optional, before running):

```bash
export LOCAL_LLM_NAME="microsoft/phi-2"
# or a lighter one, e.g.:
# export LOCAL_LLM_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

## 3. Running the Backend

In `backend/` with the virtual environment active:

```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The first run downloads the models. The pipeline persists:

- `faiss_index.index` – FAISS L2 index for embeddings.
- `faiss_metadata.pkl` – list of text chunks corresponding to vectors.

## 4. Running the Frontend

From `frontend/`:

```bash
python -m http.server 5500
```

Then open:

```text
http://localhost:5500/index.html
```

## 5. UX Overview

### Knowledge Base view

- Upload one or more `.txt` files.
- Press **Ingest** to:
  - Clean texts.
  - Split into overlapping chunks.
  - Embed with `all-MiniLM-L6-v2`.
  - Write embeddings into a FAISS `IndexFlatL2`.

For simplicity, each ingest operation replaces the previous index. This can
be extended to support multiple corpora or incremental updates.

### Assistant view

- Enter a natural-language query.
- Configure:
  - **Top-K** – number of chunks retrieved from FAISS.
  - **Mode**:
    - `Standard answer` – concise, grounded response.
    - `Explain & justify` – answer + short bullet reasoning + confidence (Low/Medium/High).
    - `Ask me questions` – outputs questions to explore the topic.
- Press **Run**.

You’ll see:

- The response in the **Answer** card.
- The active model and mode in a small meta line.
- A list of retrieved chunks on the right, with similarity distances.
- A collapsible **prompt view** showing the exact prompt sent to the LLM.

## 6. RAG Pipeline Internals

The core logic in `rag_pipeline.py`:

- `ingest_texts(raw_texts)`
  - Cleans raw text using regex.
  - Uses `RecursiveCharacterTextSplitter` for overlapping chunks.
  - Encodes chunks with `SentenceTransformer(all-MiniLM-L6-v2)`.
  - Builds a FAISS `IndexFlatL2` and stores chunks in a pickle file.

- `retrieve_chunks(query, top_k)`
  - Encodes the query.
  - Runs k-NN over FAISS.
  - Returns a ranked list of `{rank, text, distance}`.

- `generate_answer(question, top_k, mode)`
  - Retrieves chunks.
  - Builds a mode-specific prompt:
    - `standard`, `explain`, or `probe`.
  - Feeds the prompt into a local Hugging Face `pipeline("text-generation")`.
  - Trims any prompt echo.
  - Returns a JSON payload used directly by the frontend.

This design makes it easy to swap components (vector DB, LLM) while preserving
the overall flow.

## 7. Possible Extensions

- Add support for PDFs / HTML via a small pre-processing layer.
- Include per-chunk citation markers in the generated answer.
- Implement a history panel showing past questions and modes.
- Experiment with re-ranking on top of FAISS (e.g., using a cross-encoder).

The goal is to provide a small but realistic foundation for RAG experiments,
demos, or coursework that need a more polished, “tool-like” interface rather
than a purely educational prototype.
