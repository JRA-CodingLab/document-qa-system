"""Quality metrics for RAG responses."""

from __future__ import annotations

import string
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document

# Comprehensive English stop-word list used by faithfulness scoring.
_STOP_WORDS = frozenset(
    [
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "as", "is", "was",
        "are", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "shall", "can", "need", "dare", "ought", "used",
        "not", "no", "nor", "so", "yet", "both", "each", "few", "more",
        "most", "other", "some", "such", "than", "too", "very",
        "just", "about", "above", "after", "again", "all", "also",
        "any", "because", "before", "below", "between", "during",
        "even", "further", "here", "how", "into", "its", "itself",
        "let", "like", "me", "mine", "much", "must", "my", "myself",
        "new", "now", "off", "old", "once", "only", "our", "ours",
        "out", "over", "own", "same", "she", "her", "hers", "herself",
        "he", "him", "his", "himself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves",
        "this", "that", "these", "those", "then", "there", "therefore",
        "through", "under", "until", "up", "upon", "us", "we",
        "what", "when", "where", "which", "while", "who", "whom",
        "why", "you", "your", "yours", "yourself", "yourselves",
    ]
)


@dataclass
class EvaluationResult:
    """Container for a single evaluation outcome."""

    question: str
    answer: str
    expected_answer: Optional[str] = None
    sources: List[Dict] = field(default_factory=list)
    latency_seconds: float = 0.0
    relevance_score: Optional[float] = None
    faithfulness_score: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "expected_answer": self.expected_answer,
            "num_sources": len(self.sources),
            "latency_seconds": self.latency_seconds,
            "relevance_score": self.relevance_score,
            "faithfulness_score": self.faithfulness_score,
        }


# ── scoring functions ─────────────────────────────────────────────────


def calculate_relevance_score(
    question: str,
    retrieved_docs: List[Document],
    expected_keywords: Optional[List[str]] = None,
) -> float:
    """Score how relevant the retrieved documents are.

    Returns 0.0–1.0.  Without *expected_keywords* and at least one
    document retrieved the score is 1.0.
    """
    if not retrieved_docs:
        return 0.0
    if not expected_keywords:
        return 1.0

    combined = " ".join(d.page_content for d in retrieved_docs).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return hits / len(expected_keywords)


def calculate_faithfulness_score(
    answer: str,
    retrieved_docs: List[Document],
) -> float:
    """Score how well the *answer* is grounded in the *retrieved_docs*.

    Extracts significant (non-stop, len>2) words from the answer and
    checks how many appear in the source texts.
    """
    if not answer or not answer.strip():
        return 0.0
    if not retrieved_docs:
        return 0.0

    raw_words = answer.split()
    significant = [
        w.strip(string.punctuation).lower()
        for w in raw_words
        if len(w.strip(string.punctuation)) > 2
    ]
    significant = [w for w in significant if w and w not in _STOP_WORDS]

    if not significant:
        return 1.0

    combined = " ".join(d.page_content for d in retrieved_docs).lower()
    hits = sum(1 for w in significant if w in combined)
    return hits / len(significant)


# ── convenience ───────────────────────────────────────────────────────


def evaluate_response(
    question: str,
    answer: str,
    sources: List[Dict],
    latency: float,
    expected_answer: Optional[str] = None,
    expected_keywords: Optional[List[str]] = None,
) -> EvaluationResult:
    """Build an :class:`EvaluationResult` with computed scores."""
    docs = [
        Document(
            page_content=s.get("content", ""),
            metadata=s.get("metadata", {}),
        )
        for s in sources
    ]
    relevance = calculate_relevance_score(question, docs, expected_keywords)
    faithfulness = calculate_faithfulness_score(answer, docs)

    return EvaluationResult(
        question=question,
        answer=answer,
        expected_answer=expected_answer,
        sources=sources,
        latency_seconds=latency,
        relevance_score=relevance,
        faithfulness_score=faithfulness,
    )


def measure_latency(fn: Callable) -> Callable:
    """Decorator that returns ``(result, elapsed_seconds)``."""

    @wraps(fn)
    def wrapper(*args, **kwargs) -> Tuple:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        return result, elapsed

    return wrapper
