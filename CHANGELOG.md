# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-13

### Added
- Document ingestion pipeline supporting PDF, Markdown, and plain text files
- Configurable text chunking with recursive character splitting
- ChromaDB vector store integration with OpenAI and HuggingFace embeddings
- Multi-provider LLM support (OpenAI, Anthropic, Ollama)
- RAG chain with conversation history and streaming support
- FastAPI REST API with document upload, querying, and management endpoints
- Streamlit web UI with chat interface and source citations
- Bilingual prompt templates (English and German)
- Language detection for automatic prompt selection
- MLflow-based evaluation pipeline with relevance and faithfulness metrics
- Comprehensive test suite (8 test modules)
- Full documentation (README, API reference, setup guide)
