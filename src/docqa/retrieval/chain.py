"""RAG chain — combines vector retrieval with LLM generation."""

from __future__ import annotations

from typing import AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from docqa.llm.prompts import format_documents, get_prompts
from docqa.llm.providers import LLMSettings, get_llm
from docqa.vectordb.store import ChromaStore


class RAGChain:
    """End-to-end retrieval-augmented generation chain.

    Parameters
    ----------
    vector_store:
        ``ChromaStore`` used for document retrieval.
    llm_settings:
        LLM configuration; loaded from environment if *None*.
    k:
        Number of documents to retrieve per query.
    """

    def __init__(
        self,
        vector_store: ChromaStore,
        llm_settings: Optional[LLMSettings] = None,
        k: int = 4,
    ) -> None:
        self._vector_store = vector_store
        self._llm_settings = llm_settings
        self._k = k
        self._llm = None
        self._chat_history: List = []

    # ── lazy LLM ──────────────────────────────────────────────────

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(self._llm_settings)
        return self._llm

    # ── retrieval ─────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Document]:
        """Fetch the top-*k* documents for *query*."""
        return self._vector_store.similarity_search(query, k=self._k)

    # ── invoke ────────────────────────────────────────────────────

    def invoke(
        self,
        query: str,
        use_history: bool = False,
    ) -> Dict[str, object]:
        """Run the full RAG pipeline and return answer + sources."""
        docs = self.retrieve(query)
        context = format_documents(docs)
        prompts = get_prompts("en")

        if use_history and self._chat_history:
            prompt = prompts["chat"]
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke(
                {
                    "context": context,
                    "question": query,
                    "chat_history": self._chat_history,
                }
            )
        else:
            prompt = prompts["rag"]
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": query})

        if use_history:
            self._chat_history.append(HumanMessage(content=query))
            self._chat_history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "sources": self._format_sources(docs),
        }

    # ── streaming ─────────────────────────────────────────────────

    def stream(
        self,
        query: str,
        use_history: bool = False,
    ) -> Iterator[str]:
        """Yield answer chunks as they are generated."""
        docs = self.retrieve(query)
        context = format_documents(docs)
        prompts = get_prompts("en")

        if use_history and self._chat_history:
            prompt = prompts["chat"]
            chain = prompt | self.llm | StrOutputParser()
            invoke_input = {
                "context": context,
                "question": query,
                "chat_history": self._chat_history,
            }
        else:
            prompt = prompts["rag"]
            chain = prompt | self.llm | StrOutputParser()
            invoke_input = {"context": context, "question": query}

        full_response = ""
        for chunk in chain.stream(invoke_input):
            full_response += chunk
            yield chunk

        if use_history:
            self._chat_history.append(HumanMessage(content=query))
            self._chat_history.append(AIMessage(content=full_response))

    async def astream(
        self,
        query: str,
        use_history: bool = False,
    ) -> AsyncIterator[str]:
        """Async version of :pymeth:`stream`."""
        docs = self.retrieve(query)
        context = format_documents(docs)
        prompts = get_prompts("en")

        if use_history and self._chat_history:
            prompt = prompts["chat"]
            chain = prompt | self.llm | StrOutputParser()
            invoke_input = {
                "context": context,
                "question": query,
                "chat_history": self._chat_history,
            }
        else:
            prompt = prompts["rag"]
            chain = prompt | self.llm | StrOutputParser()
            invoke_input = {"context": context, "question": query}

        full_response = ""
        async for chunk in chain.astream(invoke_input):
            full_response += chunk
            yield chunk

        if use_history:
            self._chat_history.append(HumanMessage(content=query))
            self._chat_history.append(AIMessage(content=full_response))

    # ── history management ────────────────────────────────────────

    def clear_history(self) -> None:
        """Reset chat history."""
        self._chat_history = []

    @property
    def chat_history(self) -> List:
        """Return a *copy* of the internal chat history."""
        return list(self._chat_history)

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _format_sources(docs: List[Document]) -> List[Dict[str, object]]:
        sources = []
        for doc in docs:
            content = doc.page_content
            if len(content) > 200:
                content = content[:200] + "..."
            sources.append({"content": content, "metadata": doc.metadata})
        return sources
