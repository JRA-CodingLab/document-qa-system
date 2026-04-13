"""Prompt templates for the RAG pipeline (English + German)."""

from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

# ── English prompts ───────────────────────────────────────────────────

_EN_SYSTEM = (
    "You are a helpful document assistant. "
    "Answer the user's question using ONLY the provided context below. "
    "If the context does not contain enough information to answer, "
    'say "I don\'t have enough information to answer that question." '
    "Cite sources by document name when possible. "
    "Be concise and direct.\n\n"
    "Context:\n{context}"
)

_EN_RAG = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_EN_SYSTEM),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

_EN_CHAT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_EN_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

_EN_CONDENSE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Given the following conversation history and a follow-up "
            "question, rephrase the follow-up question as a standalone "
            "question that captures the full context."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# ── German prompts ────────────────────────────────────────────────────

_DE_SYSTEM = (
    "Du bist ein hilfreicher Dokumenten-Assistent. "
    "Beantworte die Frage des Nutzers ausschließlich anhand des "
    "bereitgestellten Kontexts. "
    "Wenn der Kontext nicht genügend Informationen enthält, sage "
    '"Ich habe nicht genug Informationen, um diese Frage zu beantworten." '
    "Zitiere Quellen nach Dokumentname, wenn möglich. "
    "Sei präzise und direkt.\n\n"
    "Kontext:\n{context}"
)

_DE_RAG = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_DE_SYSTEM),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

_DE_CHAT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_DE_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

_DE_CONDENSE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Formuliere anhand des folgenden Gesprächsverlaufs und einer "
            "Nachfrage die Nachfrage als eigenständige Frage um, die den "
            "gesamten Kontext erfasst."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# ── Lookup table ──────────────────────────────────────────────────────

_PROMPTS: Dict[str, Dict[str, ChatPromptTemplate]] = {
    "en": {
        "system": _EN_SYSTEM,
        "rag": _EN_RAG,
        "chat": _EN_CHAT,
        "condense": _EN_CONDENSE,
    },
    "de": {
        "system": _DE_SYSTEM,
        "rag": _DE_RAG,
        "chat": _DE_CHAT,
        "condense": _DE_CONDENSE,
    },
}


def get_prompts(language: str = "en") -> Dict[str, object]:
    """Return the prompt set for *language*, falling back to English."""
    return _PROMPTS.get(language, _PROMPTS["en"])


# ── Document formatting ──────────────────────────────────────────────


def format_documents(docs: List[Document]) -> str:
    """Format a list of documents into a numbered context string."""
    parts: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content.strip()
        parts.append(f"[Document {idx}] (Source: {source})\n{content}")
    return "\n\n---\n\n".join(parts)
