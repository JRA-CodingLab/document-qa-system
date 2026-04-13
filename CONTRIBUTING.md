# Contributing

Thank you for considering contributing to Document QA System! Here's how to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/JRA-CodingLab/document-qa-system.git
cd document-qa-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Code Standards

- **Formatter:** [Black](https://black.readthedocs.io/) (line-length 88)
- **Linter:** [Ruff](https://docs.astral.sh/ruff/)
- **Type checker:** [mypy](https://mypy-lang.org/)
- **Python:** 3.10+ with type hints throughout

Run checks before submitting:

```bash
black src/ tests/ app/
ruff check src/ tests/ app/
mypy src/
pytest tests/ -v --cov=docqa
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Write or update tests for your changes
3. Ensure all checks pass (formatting, linting, type checks, tests)
4. Update documentation if you changed any public API
5. Update `CHANGELOG.md` with a summary of your changes
6. Submit a pull request with a clear description of what and why

## Commit Messages

Use clear, descriptive commit messages:

```
feat: add hybrid search with BM25 + semantic retrieval
fix: handle empty PDF pages during ingestion
docs: update API reference with new query parameters
test: add edge case tests for language detection
```

## Reporting Issues

When reporting bugs, please include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

## Project Structure

```
src/docqa/
├── core/           # Shared configuration
├── ingestion/      # Document loading and chunking
├── vectordb/       # Embeddings, vector store, language detection
├── llm/            # LLM providers and prompt templates
├── retrieval/      # RAG chain implementation
├── api/            # FastAPI REST endpoints
└── evaluation/     # Metrics and experiment tracking
```

Each module has a corresponding test file in `tests/`.
