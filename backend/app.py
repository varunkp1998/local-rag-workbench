from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import ingest_texts, generate_answer

app = FastAPI(
    title="RAG Workbench (Local)",
    version="1.0.0",
    description="Local retrieval-augmented workbench with multiple analysis modes.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4
    mode: Optional[str] = "standard"  # 'standard', 'explain', 'probe'


@app.post("/api/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    texts: List[str] = []
    for f in files:
        raw_bytes = await f.read()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1", errors="ignore")
        texts.append(text)

    num_chunks = ingest_texts(texts)
    return {"status": "ok", "chunks": num_chunks}


@app.post("/api/query")
async def query(req: QuestionRequest):
    result = generate_answer(req.question, top_k=req.top_k, mode=req.mode or "standard")
    return result
