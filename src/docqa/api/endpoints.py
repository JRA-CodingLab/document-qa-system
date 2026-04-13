"""FastAPI application — REST endpoints for the Document QA System."""

from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from docqa.api.schemas import (
    HealthResponse,
    IngestResponse,
    LLMProviderEnum,
    ModelsResponse,
    ProviderInfo,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from docqa.ingestion.chunking import ChunkingConfig, chunk_documents
from docqa.ingestion.loaders import DocumentLoaderError, load_document
from docqa.llm.providers import LLMSettings, list_providers
from docqa.retrieval.chain import RAGChain
from docqa.vectordb.store import ChromaStore

# ── application state ──────────────────────────────────────────────────

_app_state: dict = {"vector_store": None}


def _get_vector_store() -> ChromaStore:
    store = _app_state.get("vector_store")
    if store is None:
        raise RuntimeError("Vector store not initialised")
    return store


# ── lifespan ──────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI):
    persist_dir = Path("data/chroma_db")
    persist_dir.mkdir(parents=True, exist_ok=True)
    _app_state["vector_store"] = ChromaStore(
        collection_name="documents",
        persist_directory=persist_dir,
    )
    yield
    _app_state["vector_store"] = None


# ── FastAPI app ───────────────────────────────────────────────────

app = FastAPI(
    title="Document QA System",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── health ────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", version="0.1.0")


# ── models ───────────────────────────────────────────────────────────


@app.get("/api/v1/models", tags=["Models"], response_model=ModelsResponse)
async def get_models():
    providers = [ProviderInfo(**p) for p in list_providers()]
    return ModelsResponse(providers=providers)


# ── documents ──────────────────────────────────────────────────────────


@app.get("/api/v1/documents/count", tags=["Documents"])
async def document_count():
    store = _get_vector_store()
    return {"count": store.count()}


@app.delete("/api/v1/documents", tags=["Documents"])
async def clear_documents():
    store = _get_vector_store()
    store.clear()
    return {"message": "All documents cleared"}


# ── ingest ────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}


@app.post("/api/v1/ingest", tags=["Documents"], response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Query(1000, ge=100, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
):
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be less than chunk_size",
        )

    temp_dir = Path("data/temp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    all_docs = []
    docs_processed = 0

    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}",
            )

        tmp_path = temp_dir / (upload.filename or "upload")
        try:
            with open(tmp_path, "wb") as fh:
                content = await upload.read()
                fh.write(content)
            loaded = load_document(tmp_path)
            all_docs.extend(loaded)
            docs_processed += 1
        except DocumentLoaderError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunk_documents(all_docs, config)

    store = _get_vector_store()
    store.add_documents(chunks)

    return IngestResponse(
        message="Documents ingested successfully",
        documents_processed=docs_processed,
        chunks_created=len(chunks),
    )


# ── query ─────────────────────────────────────────────────────────────


@app.post("/api/v1/query", tags=["Query"], response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    store = _get_vector_store()

    if store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents in the vector store",
        )

    llm_settings = None
    if request.provider is not None:
        llm_settings = LLMSettings(llm_provider=request.provider.value)

    chain = RAGChain(
        vector_store=store,
        llm_settings=llm_settings,
        k=request.k,
    )

    try:
        result = chain.invoke(request.question, use_history=request.use_history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [
        SourceDocument(content=s["content"], metadata=s["metadata"])
        for s in result.get("sources", [])
    ]
    return QueryResponse(answer=result["answer"], sources=sources)
