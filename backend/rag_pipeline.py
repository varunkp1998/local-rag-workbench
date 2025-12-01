import os
import re
import pickle
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "faiss_index.index")
METADATA_PATH = os.path.join(BASE_DIR, "..", "faiss_metadata.pkl")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL_NAME = os.getenv("LOCAL_LLM_NAME", "microsoft/phi-2")

_embedder = None
_faiss_index = None
_text_chunks: List[str] = []
_gen_pipeline = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def _get_generation_pipeline():
    global _gen_pipeline
    if _gen_pipeline is not None:
        return _gen_pipeline

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME)

    _gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return _gen_pipeline


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def _chunk_documents(documents: List[str],
                     chunk_size: int = 600,
                     chunk_overlap: int = 120) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = splitter.create_documents(documents)
    return [d.page_content for d in docs]


def _build_and_save_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, INDEX_PATH)
    return index


def _save_metadata(text_chunks: List[str]) -> None:
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(text_chunks, f)


def _load_index() -> faiss.IndexFlatL2:
    global _faiss_index
    if _faiss_index is None:
        if not os.path.exists(INDEX_PATH):
            raise RuntimeError("Vector index not found. Ingest some data first.")
        _faiss_index = faiss.read_index(INDEX_PATH)
    return _faiss_index


def _load_metadata() -> List[str]:
    global _text_chunks
    if _text_chunks:
        return _text_chunks
    if not os.path.exists(METADATA_PATH):
        raise RuntimeError("Metadata file not found. Ingest some data first.")
    with open(METADATA_PATH, "rb") as f:
        _text_chunks = pickle.load(f)
    _text_chunks = list(_text_chunks)
    return _text_chunks


def ingest_texts(raw_texts: List[str]) -> int:
    cleaned = [_clean_text(t) for t in raw_texts if t and t.strip()]
    if not cleaned:
        raise ValueError("No non-empty documents provided.")

    chunks = _chunk_documents(cleaned)
    if not chunks:
        raise ValueError("Chunking produced no chunks; check your input.")

    embedder = _get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    _build_and_save_index(np.array(embeddings))
    _save_metadata(chunks)

    global _faiss_index, _text_chunks
    _faiss_index = None
    _text_chunks = []

    return len(chunks)


def retrieve_chunks(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    index = _load_index()
    text_chunks = _load_metadata()
    embedder = _get_embedder()

    q_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, top_k)

    results: List[Dict[str, Any]] = []
    for rank, (i, d) in enumerate(zip(indices[0], distances[0]), start=1):
        if i < 0 or i >= len(text_chunks):
            continue
        results.append(
            {
                "rank": rank,
                "text": text_chunks[i],
                "distance": float(d),
            }
        )
    return results


def _build_context_block(context_chunks: List[Dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(
        f"[Source {c['rank']} | distance={c['distance']:.4f}]\n{c['text']}"
        for c in context_chunks
    )


def _build_prompt(question: str,
                  context_chunks: List[Dict[str, Any]],
                  mode: str = "standard") -> str:
    """
    Modes:
    - 'standard': concise answer.
    - 'explain': answer + brief, structured reasoning.
    - 'probe': generate probing questions instead of an answer.
    """
    mode = (mode or "standard").lower()
    context_block = _build_context_block(context_chunks)

    if mode == "explain":
        instructions = (
            "You are an analytical assistant.\n"
            "Using ONLY the information in the context, do three things:\n"
            "1) Provide a concise answer.\n"
            '2) Add 2-4 short bullet points explaining how you arrived at that answer.\n'
            "3) Indicate an overallConfidence level: Low, Medium, or High.\n"
            "If the answer is not clearly supported by the context, say you don't know."
        )
    elif mode == "probe":
        instructions = (
            "You are an assistant that helps people think more deeply.\n"
            "Using ONLY the context, generate 4-6 thoughtful questions that someone could\n"
            "ask to better understand or stress-test the information.\n"
            "Do NOT answer the questions. Just list them, numbered."
        )
    else:
        instructions = (
            "You are a precise assistant.\n"
            "Answer the question using ONLY the context below.\n"
            "Be concise (2-4 short paragraphs). If the context does not support a clear\n"
            "answer, say you don't know instead of guessing."
        )

    prompt = f"""
    {instructions}

    Context:
    {context_block}

    Question:
    {question}

    Response:
    """
    return re.sub(r"^\s+", "", prompt, flags=re.MULTILINE).strip()


def _trim_generation(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()


def generate_answer(question: str,
                    top_k: int = 4,
                    mode: str = "standard") -> Dict[str, Any]:
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
            "model_name": LOCAL_MODEL_NAME,
        }

    prompt = _build_prompt(question, chunks, mode=mode)
    gen = _get_generation_pipeline()
    outputs = gen(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )
    raw_text = outputs[0]["generated_text"]
    answer_text = _trim_generation(raw_text, prompt)

    return {
        "question": question,
        "answer": answer_text,
        "chunks": chunks,
        "raw_prompt": prompt,
        "mode": mode,
        "model_name": LOCAL_MODEL_NAME,
    }
