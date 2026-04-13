# API Reference

Base URL: `http://localhost:8000`

---

## Health

### `GET /health`

Returns server status.

**Response 200:**
```json
{"status": "healthy", "version": "0.1.0"}
```

---

## Documents

### `POST /api/v1/ingest`

Upload and index documents.

**Content-Type:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `files` | File[] | One or more files (.pdf, .md, .txt) |
| `chunk_size` | int (query) | Chunk size (100–4000, default 1000) |
| `chunk_overlap` | int (query) | Overlap (0–1000, default 200) |

**Response 200:**
```json
{
  "message": "Documents ingested successfully",
  "documents_processed": 2,
  "chunks_created": 15
}
```

**Errors:** 400 (unsupported type, bad params), 422 (missing files)

### `GET /api/v1/documents/count`

**Response 200:**
```json
{"count": 42}
```

### `DELETE /api/v1/documents`

Clear all indexed documents.

**Response 200:**
```json
{"message": "All documents cleared"}
```

---

## Query

### `POST /api/v1/query`

Ask a question about indexed documents.

**Request body:**
```json
{
  "question": "What is the warranty policy?",
  "k": 4,
  "use_history": false,
  "provider": "openai"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | Yes (min 1 char) | The question to ask |
| `k` | int | No (1–10, default 4) | Number of source docs |
| `use_history` | bool | No (default false) | Use conversation history |
| `provider` | string | No | Override LLM provider |

**Response 200:**
```json
{
  "answer": "The warranty covers...",
  "sources": [
    {
      "content": "According to section 3...",
      "metadata": {"source": "policy.pdf", "file_type": "pdf"}
    }
  ]
}
```

**Errors:** 400 (empty store), 422 (validation), 500 (LLM error)

---

## Models

### `GET /api/v1/models`

List available LLM providers.

**Response 200:**
```json
{
  "providers": [
    {"provider": "openai", "default_model": "gpt-4o-mini", "requires_api_key": true},
    {"provider": "anthropic", "default_model": "claude-3-5-sonnet-20241022", "requires_api_key": true},
    {"provider": "ollama", "default_model": "llama3.2", "requires_api_key": false}
  ]
}
```
