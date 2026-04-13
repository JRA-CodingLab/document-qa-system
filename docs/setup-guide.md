# Setup Guide

## Prerequisites

- Python 3.10 or newer
- An OpenAI API key (for default embeddings + LLM)
- Optional: Anthropic API key, or Ollama running locally

## Installation

```bash
# Clone and enter the project
cd build

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install the package with dev dependencies
pip install -e ".[dev]"

# (Optional) For HuggingFace local embeddings
pip install -e ".[huggingface]"
```

## Configuration

Copy the template and edit:

```bash
cp .env.example .env
```

At minimum, set:

```
OPENAI_API_KEY=sk-your-key-here
```

### Using Anthropic

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
LLM_PROVIDER=anthropic
```

### Using Ollama (local)

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

### Using HuggingFace Embeddings

```
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=intfloat/multilingual-e5-large
```

## Running

### API Server

```bash
uvicorn docqa.api.endpoints:app --reload --host 0.0.0.0 --port 8000
```

### Web UI

```bash
streamlit run app/main.py
```

The UI connects to the API at `http://localhost:8000` by default.

### Tests

```bash
pytest tests/ -v --cov=docqa
```

### Evaluation Pipeline

```bash
python -m docqa.evaluation.runner
```

View results in MLflow:

```bash
mlflow ui --port 5001
```

## Troubleshooting

- **"OPENAI_API_KEY is required"** — Set the key in your `.env` file.
- **ChromaDB errors** — Make sure `data/chroma_db/` is writable.
- **Streamlit says "Backend offline"** — Start the API server first.
- **HuggingFace import errors** — Install the `[huggingface]` extra.
