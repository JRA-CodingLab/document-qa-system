# Document QA System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/JRA-CodingLab/document-qa-system/actions/workflows/ci.yml/badge.svg)](https://github.com/JRA-CodingLab/document-qa-system/actions)
[![Deployed on Render](https://img.shields.io/badge/deployed-Render-46E3B7.svg)](https://document-qa-system.onrender.com/docs)

A Retrieval-Augmented Generation system for document question-answering. Upload PDFs, Markdown, or text files, then ask natural-language questions and receive grounded answers with source citations.

## 🚀 Live Demo

**API is deployed and publicly accessible:**

- 🔗 **Swagger UI:** [document-qa-system.onrender.com/docs](https://document-qa-system.onrender.com/docs)
- 📡 **Health Check:** [document-qa-system.onrender.com/health](https://document-qa-system.onrender.com/health)

**Try it now:**
```bash
curl -X POST https://document-qa-system.onrender.com/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG and how does it work?"}'
```

**Deployment Stack:** Docker + Render (free tier) + GitHub CI/CD auto-deploy

---

## Architecture

- **Ingestion:** Document loaders + text chunking (overlapping windows)
- **Retrieval:** TF-IDF / embedding-based vector similarity search
- **Generation:** LLM answer synthesis with source citations
- **Vector Store:** ChromaDB with OpenAI / HuggingFace embeddings
- **API:** FastAPI REST endpoints + Streamlit web UI
- **Evaluation:** MLflow-tracked metrics (faithfulness, relevancy, recall)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/ingest` | Upload and index documents |
| `POST` | `/api/v1/query` | Ask a question |
| `GET` | `/api/v1/models` | List available LLM providers |
| `GET` | `/api/v1/documents/count` | Count indexed documents |
| `DELETE` | `/api/v1/documents` | Clear all documents |

## Tech Stack

Python 3.10+ • LangChain • ChromaDB • FastAPI • Streamlit • MLflow • OpenAI / Anthropic / Ollama

## License

[MIT](LICENSE) © 2026 Juan Ruiz Alonso
