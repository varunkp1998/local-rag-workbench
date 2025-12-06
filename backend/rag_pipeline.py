import os
import re
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama

# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.npy")
METADATA_PATH = os.path.join(BASE_DIR, "tfidf_metadata.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

# Local LLM model (GGUF)
LLAMA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "phi-3-mini-4k-instruct-q4.gguf",  # change if your filename differs
)

# LLM + retrieval config
LLAMA_CONTEXT_SIZE = 1536          # a bit smaller than 2048 for speed
LLAMA_THREADS = 6                  # tune to your CPU cores
LLAMA_BATCH = 128                  # batch size for llama.cpp

TFIDF_MAX_FEATURES = 8192          # more features = better retrieval
CHUNK_SIZE = 400                   # smaller chunks for tighter context
CHUNK_OVERLAP = 80                 # keep some overlap

MAX_TOKENS_STANDARD = 160
MAX_TOKENS_EXPLAIN = 240
MAX_TOKENS_PROBE = 160

# ---------------------------------------------------------
# Globals (lazy-loaded)
# ---------------------------------------------------------

_vectorizer: Optional[TfidfVectorizer] = None
_matrix: Optional[np.ndarray] = None
_text_chunks: List[str] = []
_llm: Optional[Llama] = None


# ---------------------------------------------------------
# Helpers: cleaning, chunking
# ---------------------------------------------------------

def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def _chunk_documents(
    documents: List[str],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = splitter.create_documents(documents)
    return [d.page_content for d in docs]


# ---------------------------------------------------------
# TF–IDF index + metadata
# ---------------------------------------------------------

def _save_matrix(matrix: np.ndarray) -> None:
    np.save(INDEX_MATRIX_PATH, matrix.astype("float32"))


def _load_matrix() -> np.ndarray:
    global _matrix
    if _matrix is not None:
        return _matrix
    if not os.path.exists(INDEX_MATRIX_PATH):
        raise RuntimeError("TF–IDF matrix not found. Ingest some data first.")
    _matrix = np.load(INDEX_MATRIX_PATH)
    return _matrix


def _save_metadata(text_chunks: List[str]) -> None:
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(text_chunks, f)


def _load_metadata() -> List[str]:
    global _text_chunks
    if _text_chunks:
        return _text_chunks
    if not os.path.exists(METADATA_PATH):
        raise RuntimeError("Metadata not found. Ingest some data first.")
    with open(METADATA_PATH, "rb") as f:
        _text_chunks = pickle.load(f)
    _text_chunks = list(_text_chunks)
    return _text_chunks


def _save_vectorizer(vec: TfidfVectorizer) -> None:
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vec, f)


def _load_vectorizer() -> TfidfVectorizer:
    global _vectorizer
    if _vectorizer is not None:
        return _vectorizer
    if not os.path.exists(VECTORIZER_PATH):
        raise RuntimeError("TF–IDF vectorizer not found. Ingest some data first.")
    with open(VECTORIZER_PATH, "rb") as f:
        _vectorizer = pickle.load(f)
    return _vectorizer


# ---------------------------------------------------------
# Public ingestion + retrieval
# ---------------------------------------------------------

def ingest_texts(raw_texts: List[str]) -> int:
    """
    Clean, chunk, fit TF–IDF, and persist everything.
    Called from /api/ingest.
    """
    cleaned = [_clean_text(t) for t in raw_texts if t and t.strip()]
    if not cleaned:
        raise ValueError("No non-empty documents provided.")

    chunks = _chunk_documents(cleaned)
    if not chunks:
        raise ValueError("Chunking produced no chunks; check your input.")

    # Fit TF–IDF on chunks
    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    X = vec.fit_transform(chunks)  # (num_chunks, num_features)

    dense = X.toarray().astype("float32")

    _save_matrix(dense)
    _save_metadata(chunks)
    _save_vectorizer(vec)

    # Reset in-memory cache so next query uses fresh artefacts
    global _matrix, _text_chunks, _vectorizer
    _matrix = None
    _text_chunks = []
    _vectorizer = None

    return len(chunks)


def retrieve_chunks(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Encode query with TF–IDF and compute cosine similarity against stored matrix.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Query must not be empty.")

    vec = _load_vectorizer()
    matrix = _load_matrix()
    text_chunks = _load_metadata()

    # Adaptive top_k: short queries use fewer chunks
    words = query.split()
    if len(words) < 8:
        top_k = min(top_k, 2)
    else:
        top_k = max(1, min(top_k, 8))  # clamp between 1 and 8

    q_vec = vec.transform([query]).toarray().astype("float32")[0]  # (features,)
    doc_norms = np.linalg.norm(matrix, axis=1) + 1e-8
    q_norm = np.linalg.norm(q_vec) + 1e-8

    sims = (matrix @ q_vec) / (doc_norms * q_norm)  # cosine similarity
    top_idx = np.argsort(sims)[::-1][:top_k]

    results: List[Dict[str, Any]] = []
    for rank, i in enumerate(top_idx, start=1):
        sim = float(sims[i])
        results.append(
            {
                "rank": rank,
                "text": text_chunks[i],
                "similarity": sim,
                "distance": float(1.0 - sim),  # for UI if it expects "distance"
            }
        )
    return results


# ---------------------------------------------------------
# Local LLM (llama.cpp)
# ---------------------------------------------------------

def _get_llm() -> Llama:
    """
    Lazily load and cache the local LLM.
    """
    global _llm
    if _llm is None:
        if not os.path.exists(LLAMA_MODEL_PATH):
            raise RuntimeError(
                f"LLM model file not found at {LLAMA_MODEL_PATH}. "
                f"Place your GGUF file there (e.g. phi-3-mini-4k-instruct-q4.gguf)."
            )
        _llm = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=LLAMA_CONTEXT_SIZE,
            n_threads=LLAMA_THREADS,
            n_batch=LLAMA_BATCH,
            verbose=False,
        )
    return _llm


def _generate_with_llama(prompt: str, mode: str) -> str:
    """
    Call the local LLM with mode-specific max_tokens.
    """
    llm = _get_llm()

    mode = (mode or "standard").lower()
    if mode == "explain":
        max_tokens = MAX_TOKENS_EXPLAIN
    elif mode == "probe":
        max_tokens = MAX_TOKENS_PROBE
    else:
        max_tokens = MAX_TOKENS_STANDARD

    result = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.15,
        top_p=0.85,
        stop=["</s>"],
    )
    return result["choices"][0]["text"].strip()


# ---------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------

def _build_context_block(context_chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for c in context_chunks:
        sim = c.get("similarity")
        sim_str = f"{sim:.4f}" if sim is not None else "n/a"
        parts.append(
            f"[Source {c['rank']} | similarity={sim_str}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _build_prompt(
    question: str,
    context_chunks: List[Dict[str, Any]],
    mode: str = "standard",
) -> str:
    mode = (mode or "standard").lower()
    context_block = _build_context_block(context_chunks)

    if mode == "explain":
        instructions = (
            "You are an analytical, cautious assistant.\n"
            "Using ONLY the information in the context, do three things:\n"
            "1) Provide a concise answer.\n"
            "2) Add 2-4 bullet points explaining which [Source N] tags support the answer.\n"
            "3) State an overallConfidence: Low, Medium, or High.\n"
            "If the answer is not clearly supported by the context, say you don't know "
            "and set overallConfidence to Low."
        )
    elif mode == "probe":
        instructions = (
            "You are an assistant that helps people think more deeply.\n"
            "Using ONLY the context, generate 4-6 thoughtful questions that someone could "
            "ask to better understand or stress-test the information.\n"
            "Do NOT answer the questions. Just list them, numbered."
        )
    else:
        instructions = (
            "You are a precise and cautious assistant.\n"
            "Use ONLY the information in the context below. "
            "If the context does not clearly support an answer, explicitly say "
            "\"I cannot answer this based on the provided documents.\" Do NOT guess.\n"
            "When you answer, reference which [Source N] tags you are using.\n"
            "Keep the answer focused and avoid repeating the question."
        )

    prompt = f"""
{instructions}

Context:
{context_block}

Question:
{question}

Response:
""".strip()
    return prompt


# ---------------------------------------------------------
# Grounding confidence helper
# ---------------------------------------------------------

def _estimate_grounding_confidence(chunks: List[Dict[str, Any]]) -> str:
    sims = [c.get("similarity") for c in chunks if c.get("similarity") is not None]
    if not sims:
        return "Unknown"
    avg_sim = float(np.mean(sims))
    if avg_sim >= 0.75:
        return "High"
    if avg_sim >= 0.5:
        return "Medium"
    return "Low"


# ---------------------------------------------------------
# Main RAG entrypoint (used by /api/query)
# ---------------------------------------------------------

def generate_answer(
    question: str,
    top_k: int = 4,
    mode: str = "standard",
) -> Dict[str, Any]:
    chunks = retrieve_chunks(question, top_k=top_k)
    if not chunks:
        return {
            "question": question,
            "answer": (
                "No relevant context is available yet. "
                "Please ingest some documents and try again."
            ),
            "chunks": [],
            "raw_prompt": "",
            "mode": mode,
            "model_name": os.path.basename(LLAMA_MODEL_PATH),
            "grounding_confidence": "Unknown",
        }

    prompt = _build_prompt(question, chunks, mode=mode)
    answer_text = _generate_with_llama(prompt, mode=mode)
    grounding_conf = _estimate_grounding_confidence(chunks)

    return {
        "question": question,
        "answer": answer_text,
        "chunks": chunks,
        "raw_prompt": prompt,
        "mode": mode,
        "model_name": os.path.basename(LLAMA_MODEL_PATH),
        "grounding_confidence": grounding_conf,
    }
