# Document QA System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/JRA-CodingLab/document-qa-system/actions/workflows/ci.yml/badge.svg)](https://github.com/JRA-CodingLab/document-qa-system/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Retrieval-Augmented Generation system for document question-answering. Upload PDFs, Markdown, or text files, then ask natural-language questions and receive grounded answers with source citations.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Streamlit UI   ─────HTTP─────>  FastAPI Backend     │
│                                   │                  │
│                   ┌───────────────┼─────────────┐    │
│                   │ Ingestion     │  RAG Chain   │    │
│                   │  Loaders      │  Retrieval   │    │
│                   │  Chunking     │  LLM         │    │
│                   └───────────────┼─────────────┘    │
│                                   │                  │
│                   ┌───────────────┴──────────────┐   │
│                   │  ChromaDB Vector Store        │   │
│                   │  + Embeddings (OpenAI / HF)   │   │
│                   └──────────────────────────────┘   │
│                                                      │
│                   ┌──────────────────────────────┐   │
│                   │  Evaluation Pipeline (MLflow) │   │
│                   └──────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
# Create virtualenv
python -m venv .venv && source .venv/bin/activate

# Install with dev extras
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (required)
```

### 3. Run the API Server

```bash
uvicorn docqa.api.endpoints:app --reload --port 8000
```

### 4. Run the Web UI

```bash
streamlit run app/main.py
```

### 5. Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
document-qa-system/
├── src/docqa/           # Main package
│   ├── core/               # Shared config
│   ├── ingestion/          # Document loaders + chunking
│   ├── vectordb/           # Embeddings, ChromaDB store, language detection
│   ├── llm/                # LLM providers + prompt templates
│   ├── retrieval/          # RAG chain (retrieve + generate)
│   ├── api/                # FastAPI endpoints + Pydantic schemas
│   └── evaluation/         # Metrics, MLflow tracker, evaluation runner
├── tests/                  # Comprehensive test suite
├── app/                    # Streamlit web UI
├── data/
│   ├── sample_docs/        # Sample documents for testing
│   └── eval/               # Test questions for evaluation
├── docs/                   # Documentation
├── pyproject.toml          # Build & dependency config
└── .env.example            # Environment variable template
```

## Configuration

All settings are driven by environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `LLM_PROVIDER` | `openai` | `openai` / `anthropic` / `ollama` |
| `EMBEDDING_PROVIDER` | `openai` | `openai` / `huggingface` |
| `TEMPERATURE` | `0.0` | LLM sampling temperature |
| `MAX_TOKENS` | `1024` | Max output tokens |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/ingest` | Upload and index documents |
| `POST` | `/api/v1/query` | Ask a question |
| `GET` | `/api/v1/models` | List available LLM providers |
| `GET` | `/api/v1/documents/count` | Count indexed documents |
| `DELETE` | `/api/v1/documents` | Clear all documents |

## Evaluation

Run the built-in evaluation pipeline:

```bash
python -m docqa.evaluation.runner
```

This ingests sample documents, runs test questions through the pipeline, calculates relevance and faithfulness scores, and logs everything to MLflow.

## Technology Stack

- **Python 3.10+**
- **LangChain** — LLM orchestration
- **ChromaDB** — vector database
- **FastAPI** — REST API
- **Streamlit** — web UI
- **MLflow** — experiment tracking
- **OpenAI / Anthropic / Ollama** — LLM providers

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
