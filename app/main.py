"""Streamlit web UI for the Document QA System.

Communicates with the FastAPI backend over HTTP — does **not** import
any backend modules directly.
"""

from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://localhost:8000"
TIMEOUT = 30


def check_backend_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_available_models() -> list[str]:
    try:
        resp = requests.get(f"{API_BASE}/api/v1/models", timeout=5)
        data = resp.json()
        return [p["provider"] for p in data.get("providers", [])]
    except Exception:
        return ["openai"]


def get_document_count() -> int:
    try:
        resp = requests.get(f"{API_BASE}/api/v1/documents/count", timeout=5)
        return resp.json().get("count", 0)
    except Exception:
        return 0


def ingest_document(file_obj) -> dict:
    files = {"files": (file_obj.name, file_obj.getvalue(), "application/octet-stream")}
    resp = requests.post(f"{API_BASE}/api/v1/ingest", files=files, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def query_rag(question: str, provider: str) -> dict:
    payload = {"question": question, "provider": provider}
    resp = requests.post(f"{API_BASE}/api/v1/query", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def clear_all_documents() -> dict:
    resp = requests.delete(f"{API_BASE}/api/v1/documents", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "provider" not in st.session_state:
        st.session_state.provider = "openai"
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()


st.set_page_config(
    page_title="Document QA System",
    page_icon="\ud83d\udcda",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_state()

with st.sidebar:
    st.title("\ud83d\udcda RAG Assistant")
    if check_backend_health():
        st.success("\u2705 Backend connected")
    else:
        st.error("\u274c Backend offline")
        st.code("uvicorn docqa.api.endpoints:app --reload", language="bash")
        st.stop()

    models = get_available_models()
    st.session_state.provider = st.selectbox(
        "LLM Provider", models, index=models.index(st.session_state.provider) if st.session_state.provider in models else 0
    )

    st.subheader("\ud83d\udcc4 Upload Documents")
    uploaded = st.file_uploader("Choose files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    new_files_processed = False
    if uploaded:
        for f in uploaded:
            file_key = f"{f.name}_{f.size}"
            if file_key in st.session_state.processed_files:
                continue
            with st.spinner(f"Processing {f.name}\u2026"):
                try:
                    result = ingest_document(f)
                    st.success(f"\u2705 {f.name}: {result.get('chunks_created', 0)} chunks")
                except Exception as exc:
                    st.error(f"\u274c {f.name}: {exc}")
            st.session_state.processed_files.add(file_key)
            new_files_processed = True
    if new_files_processed:
        st.rerun()

    doc_count = get_document_count()
    st.metric("Indexed Documents", doc_count)

    if doc_count > 0:
        if st.button("\ud83d\uddd1\ufe0f Clear All Documents"):
            clear_all_documents()
            st.session_state.messages = []
            st.session_state.processed_files = set()
            st.rerun()

    with st.expander("\u2139\ufe0f Help"):
        st.markdown(
            "- Upload PDF, TXT, or MD files\n"
            "- Ask questions about the uploaded content\n"
            "- Chat history is kept for the current session only\n"
            "- Switch LLM providers using the dropdown above\n"
            "- Clear documents to start fresh"
        )
    st.caption("Built with FastAPI + LangChain + Streamlit")

st.title("\ud83d\udcac Ask Questions About Your Documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("\ud83d\udcc4 View Sources"):
                for src in msg["sources"]:
                    source_name = src.get("metadata", {}).get("source", "Unknown")
                    preview = src.get("content", "")[:300]
                    st.markdown(f"**{source_name}**")
                    st.text(preview)
                    st.divider()

if not st.session_state.messages:
    st.info("Upload documents in the sidebar, then ask questions here.")

if doc_count == 0:
    st.warning("Please upload documents first.")
else:
    if prompt := st.chat_input("Ask a question\u2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking\u2026"):
                try:
                    result = query_rag(prompt, st.session_state.provider)
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("\ud83d\udcc4 View Sources"):
                            for src in sources:
                                source_name = src.get("metadata", {}).get("source", "Unknown")
                                preview = src.get("content", "")[:300]
                                st.markdown(f"**{source_name}**")
                                st.text(preview)
                                st.divider()
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    err = f"Error: {exc}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
