"""
Document QA System - Mock RAG API
Author: Juan Ruiz Alonso
Description: FastAPI mock deployment for a Document QA (RAG) system.
             Uses TF-IDF similarity search - no external API keys required.
"""

from __future__ import annotations

import io
import math
import re
import uuid
from collections import Counter
from datetime import datetime
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Document QA System",
    description=(
        "A Retrieval-Augmented Generation (RAG) API for document question-answering. "
        "Powered by TF-IDF similarity search - no external API keys required."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNK_SIZE = 400

class Chunk:
    def __init__(self, chunk_id: str, source: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.source = source
        self.text = text

_chunks: list[Chunk] = []

SAMPLE_DOCUMENTS: list[tuple[str, str]] = [
    ("intro_to_rag.txt",
     "Retrieval-Augmented Generation (RAG) is a natural language processing architecture that "
     "combines information retrieval with text generation. Instead of relying solely on a language "
     "model's parametric knowledge, RAG retrieves relevant documents from an external knowledge base "
     "at inference time and feeds them as context to the generator. "
     "The RAG pipeline consists of three main components: an indexing stage, a retrieval stage, and "
     "a generation stage. During indexing, documents are split into chunks, embedded into dense "
     "vector representations, and stored in a vector database such as ChromaDB, Pinecone, or FAISS. "
     "At query time, the user's question is embedded using the same encoder and the top-k most "
     "similar chunks are retrieved via approximate nearest-neighbour search. These chunks are "
     "concatenated with the original question and passed to a large language model (LLM) such as "
     "GPT-4 or Claude to synthesise a grounded, factual answer. "
     "RAG significantly reduces hallucinations compared to pure LLM generation because the model is "
     "anchored to retrieved source material. It also allows the knowledge base to be updated without "
     "retraining the underlying language model, making it highly suitable for enterprise use-cases "
     "where documents change frequently."),
    ("document_qa_systems.txt",
     "Document Question-Answering (Document QA) systems enable users to ask natural language "
     "questions about a corpus of documents and receive precise, evidence-backed answers. They are "
     "widely deployed in enterprise settings for customer support, legal research, medical literature "
     "review, and internal knowledge management. "
     "A typical Document QA system pipeline includes the following steps: "
     "1. Document ingestion - raw files (PDF, DOCX, TXT) are parsed and cleaned. "
     "2. Chunking - documents are split into overlapping or non-overlapping segments to fit within "
     "the context window of the embedding model. "
     "3. Embedding - each chunk is converted to a high-dimensional vector using a pre-trained "
     "sentence transformer or OpenAI's text-embedding-ada-002 model. "
     "4. Vector storage - embeddings are persisted in a vector database for fast similarity lookup. "
     "5. Retrieval - given a user query, the top-k most semantically similar chunks are fetched. "
     "6. Answer generation - an LLM synthesises a coherent answer from the retrieved chunks. "
     "Evaluation metrics for Document QA systems include Faithfulness (does the answer contradict "
     "the sources?), Answer Relevancy (does the answer address the question?), and Context Recall "
     "(are all relevant passages retrieved?). Frameworks like RAGAS and TruLens automate these "
     "evaluations. LangChain is a popular framework for building RAG pipelines. It provides "
     "abstractions for document loaders, text splitters, vector stores, retrievers, and LLM chains, "
     "enabling rapid prototyping of production-grade Document QA applications."),
    ("vector_databases.txt",
     "Vector databases are purpose-built storage systems optimised for high-dimensional embedding "
     "vectors. Unlike traditional relational databases that excel at exact-match queries, vector "
     "databases support approximate nearest-neighbour (ANN) search, which is the core operation "
     "required by RAG retrieval. "
     "Popular vector databases include: "
     "- ChromaDB: open-source, embeddable, ideal for local development and small-scale deployments. "
     "- Pinecone: fully managed cloud service with serverless and pod-based tiers. "
     "- Weaviate: open-source with hybrid search (vector + keyword BM25). "
     "- Qdrant: Rust-based, high performance, supports payload filtering. "
     "- FAISS: Facebook AI Similarity Search - an in-memory library, not a full database. "
     "- pgvector: PostgreSQL extension that adds vector similarity search. "
     "ChromaDB is used in this Document QA System because it requires zero infrastructure setup and "
     "can persist data to disk with a single line of configuration. It supports both cosine and L2 "
     "distance metrics and integrates natively with LangChain and LlamaIndex. "
     "When choosing a vector database for production, consider: query latency, throughput, index "
     "size limitations, hybrid search support, managed hosting availability, and pricing. For "
     "high-scale production workloads, Pinecone or Weaviate are strong choices; for developer "
     "prototypes, ChromaDB or FAISS are excellent starting points."),
]

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def _chunk_text(text: str, source: str) -> list[Chunk]:
    words = text.split()
    chunk_word_size = 80
    overlap = 20
    chunks: list[Chunk] = []
    i = 0
    while i < len(words):
        window = words[i : i + chunk_word_size]
        chunk_text = " ".join(window)
        chunks.append(Chunk(chunk_id=str(uuid.uuid4()), source=source, text=chunk_text))
        if i + chunk_word_size >= len(words):
            break
        i += chunk_word_size - overlap
    return chunks

def _preload_samples() -> None:
    for filename, content in SAMPLE_DOCUMENTS:
        new_chunks = _chunk_text(content, filename)
        _chunks.extend(new_chunks)

_preload_samples()

def _tfidf_scores(query: str, corpus_chunks: list[Chunk]) -> list[tuple[float, Chunk]]:
    if not corpus_chunks:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    num_docs = len(corpus_chunks)
    doc_freq: Counter[str] = Counter()
    tokenized_docs: list[list[str]] = []
    for chunk in corpus_chunks:
        tokens = _tokenize(chunk.text)
        tokenized_docs.append(tokens)
        for term in set(tokens):
            doc_freq[term] += 1
    def idf(term: str) -> float:
        df = doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((num_docs + 1) / (df + 1)) + 1.0
    def tf(term: str, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        return tokens.count(term) / len(tokens)
    def vec(tokens: list[str], vocab: set[str]) -> dict[str, float]:
        return {term: tf(term, tokens) * idf(term) for term in vocab}
    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in a)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
    query_token_set = set(query_tokens)
    vocab = query_token_set | {t for tokens in tokenized_docs for t in tokens}
    q_vec = vec(query_tokens, vocab)
    scored: list[tuple[float, Chunk]] = []
    for chunk, tokens in zip(corpus_chunks, tokenized_docs):
        d_vec = vec(tokens, vocab)
        score = cosine(q_vec, d_vec)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

_ANSWER_TEMPLATES = [
    "Based on the retrieved documents, {summary} The relevant passages indicate that {detail}.",
    "According to the ingested knowledge base, {summary} Specifically, the documents state: '{detail}'.",
    "The document corpus contains information relevant to your question. {summary} Furthermore, {detail}.",
]

def _build_answer(question: str, top_chunks: list[Chunk]) -> str:
    if not top_chunks:
        return "No relevant documents were found to answer your question. Please ingest documents first or refine your query."
    primary = top_chunks[0].text
    detail = primary[:300].strip()
    if len(primary) > 300:
        detail += "..."
    q_keywords = [t for t in _tokenize(question) if len(t) > 3]
    if q_keywords:
        kw_phrase = ", ".join(q_keywords[:4])
        summary = f"the documents discuss topics related to {kw_phrase}."
    else:
        summary = "the documents contain relevant information."
    template = _ANSWER_TEMPLATES[len(question) % len(_ANSWER_TEMPLATES)]
    return template.format(summary=summary, detail=detail)

class ServiceInfo(BaseModel):
    service: str = "Document QA System"
    version: str = "1.0.0"
    description: str = "Mock RAG API - TF-IDF retrieval, no API keys required"
    author: str = "Juan Ruiz Alonso"
    status: str = "healthy"
    docs: str = "/docs"
    endpoints: list[str] = Field(default=["GET  /", "GET  /health", "POST /api/v1/ingest", "POST /api/v1/query", "GET  /api/v1/documents/count", "DELETE /api/v1/documents", "GET  /api/v1/models"])

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    chunks_indexed: int
    version: str = "1.0.0"

class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    total_chunks: int

class Source(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float

class QueryRequest(BaseModel):
    question: str
    provider: str = "openai"
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    question: str
    answer: str
    provider: str
    sources: list[Source]
    model_used: str
    retrieval_method: str = "tfidf"

class DocumentCountResponse(BaseModel):
    total_chunks: int
    unique_sources: int
    sources: list[str]

class DeleteResponse(BaseModel):
    message: str
    chunks_deleted: int

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    available: bool
    note: str

class ModelsResponse(BaseModel):
    providers: list[ModelInfo]
    default: str
    note: str

@app.get("/", response_model=ServiceInfo, tags=["Health"])
async def root() -> ServiceInfo:
    return ServiceInfo()

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    return HealthResponse(timestamp=datetime.utcnow().isoformat() + "Z", chunks_indexed=len(_chunks))

@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    content_bytes: bytes = await file.read()
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="File must be UTF-8 encoded text.")
    if not text.strip():
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    filename = file.filename or "unknown.txt"
    new_chunks = _chunk_text(text, filename)
    _chunks.extend(new_chunks)
    return IngestResponse(message=f"Document '{filename}' successfully ingested.", filename=filename, chunks_created=len(new_chunks), total_chunks=len(_chunks))

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    scored = _tfidf_scores(request.question, _chunks)
    top_scored = [(score, chunk) for score, chunk in scored[: request.top_k] if score > 0.0]
    sources = [Source(chunk_id=chunk.chunk_id, source=chunk.source, text=chunk.text, score=round(score, 4)) for score, chunk in top_scored]
    top_chunks = [chunk for _, chunk in top_scored]
    answer = _build_answer(request.question, top_chunks)
    model_map = {"openai": "gpt-4o-mini (mocked)", "anthropic": "claude-3-haiku (mocked)", "mistral": "mistral-7b-instruct (mocked)", "ollama": "llama3.2 (mocked)"}
    model_used = model_map.get(request.provider.lower(), f"{request.provider} (mocked)")
    return QueryResponse(question=request.question, answer=answer, provider=request.provider, sources=sources, model_used=model_used)

@app.get("/api/v1/documents/count", response_model=DocumentCountResponse, tags=["Documents"])
async def document_count() -> DocumentCountResponse:
    unique_sources = sorted({chunk.source for chunk in _chunks})
    return DocumentCountResponse(total_chunks=len(_chunks), unique_sources=len(unique_sources), sources=unique_sources)

@app.delete("/api/v1/documents", response_model=DeleteResponse, tags=["Documents"])
async def clear_documents() -> DeleteResponse:
    deleted = len(_chunks)
    _chunks.clear()
    return DeleteResponse(message="All documents cleared from the knowledge base.", chunks_deleted=deleted)

@app.get("/api/v1/models", response_model=ModelsResponse, tags=["Models"])
async def list_models() -> ModelsResponse:
    providers = [
        ModelInfo(id="openai", name="OpenAI GPT-4o-mini", description="OpenAI's cost-efficient GPT-4 class model.", available=True, note="Mocked - no API key required."),
        ModelInfo(id="anthropic", name="Anthropic Claude 3 Haiku", description="Anthropic's fast Claude model.", available=True, note="Mocked - no API key required."),
        ModelInfo(id="mistral", name="Mistral 7B Instruct", description="Open-weight Mistral model.", available=True, note="Mocked - no API key required."),
        ModelInfo(id="ollama", name="Ollama (Local)", description="Run any model locally via Ollama.", available=True, note="Mocked - no API key required."),
    ]
    return ModelsResponse(providers=providers, default="openai", note="This is a mock deployment. All providers use TF-IDF retrieval with template-based answer generation.")

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("deploy.main:app", host="0.0.0.0", port=port, reload=False)