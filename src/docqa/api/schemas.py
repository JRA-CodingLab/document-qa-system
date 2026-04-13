"""Pydantic request / response models for the REST API."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────


class LLMProviderEnum(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# ── Requests ──────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    k: int = Field(4, ge=1, le=10, description="Documents to retrieve")
    use_history: bool = Field(False, description="Include conversation history")
    provider: Optional[LLMProviderEnum] = Field(
        None, description="Override the LLM provider"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What is the warranty policy?", "k": 4}
            ]
        }
    }


class IngestRequest(BaseModel):
    collection_name: str = Field("documents")
    chunk_size: int = Field(1000, ge=100, le=4000)
    chunk_overlap: int = Field(200, ge=0, le=1000)


# ── Responses ─────────────────────────────────────────────────────────


class SourceDocument(BaseModel):
    content: str
    metadata: Dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = Field(default_factory=list)


class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    version: str


class ProviderInfo(BaseModel):
    provider: str
    default_model: str
    requires_api_key: bool


class ModelsResponse(BaseModel):
    providers: List[ProviderInfo]
