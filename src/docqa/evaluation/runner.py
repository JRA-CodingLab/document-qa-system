"""Evaluation runner — orchestrates test questions through the RAG chain."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from docqa.evaluation.metrics import EvaluationResult, evaluate_response
from docqa.evaluation.tracker import ExperimentTracker
from docqa.ingestion.chunking import ChunkingConfig, chunk_documents
from docqa.ingestion.loaders import load_directory
from docqa.retrieval.chain import RAGChain
from docqa.vectordb.store import ChromaStore


def load_test_questions(file_path: Path) -> List[Dict]:
    """Load test questions from a JSON file.

    Expected format::

        {"questions": [{"id": "q1", "question": "...", ...}, ...]}
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["questions"]


def run_evaluation(
    vector_store: ChromaStore,
    questions: List[Dict],
    tracker: Optional[ExperimentTracker] = None,
    k: int = 4,
) -> List[EvaluationResult]:
    """Run each question through the chain and collect metrics."""
    chain = RAGChain(vector_store=vector_store, k=k)
    results: List[EvaluationResult] = []

    for idx, q in enumerate(questions):
        question_text = q["question"]
        expected_keywords = q.get("expected_keywords")
        expected_answer = q.get("expected_answer")

        t0 = time.perf_counter()
        chain_result = chain.invoke(question_text)
        elapsed = time.perf_counter() - t0

        result = evaluate_response(
            question=question_text,
            answer=chain_result["answer"],
            sources=chain_result["sources"],
            latency=elapsed,
            expected_answer=expected_answer,
            expected_keywords=expected_keywords,
        )
        results.append(result)

        if tracker is not None:
            tracker.log_evaluation(result, step=idx)

    return results


def run_full_evaluation(
    documents_dir: Path,
    test_questions_path: Path,
    experiment_name: str = "rag-evaluation",
    run_name: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 4,
) -> List[EvaluationResult]:
    """End-to-end evaluation: ingest → query → score → track."""
    tracker = ExperimentTracker(experiment_name=experiment_name)
    tracker.start_run(run_name=run_name)

    try:
        # Log config
        tracker.log_params(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "k": k,
            }
        )

        # Ingest documents
        raw_docs = load_directory(documents_dir)
        config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunk_documents(raw_docs, config)
        tracker.log_params(
            {
                "document_count": len(raw_docs),
                "chunk_count": len(chunks),
            }
        )

        # Build in-memory vector store
        store = ChromaStore(collection_name="eval_collection")
        store.add_documents(chunks)

        # Load and run
        questions = load_test_questions(test_questions_path)
        results = run_evaluation(store, questions, tracker=tracker, k=k)

        tracker.log_batch_results(results)
        tracker.log_dict_artifact(
            {"results": [r.to_dict() for r in results]},
            "evaluation_results.json",
        )

        return results
    finally:
        tracker.end_run()


if __name__ == "__main__":
    sample_docs = Path("data/sample_docs")
    test_qs = Path("data/eval/test_questions.json")

    if not sample_docs.exists() or not test_qs.exists():
        print("Sample data not found. Run from the project root directory.")
        raise SystemExit(1)

    results = run_full_evaluation(
        documents_dir=sample_docs,
        test_questions_path=test_qs,
        run_name="sample-evaluation",
    )
    for r in results:
        print(
            f"Q: {r.question}\n"
            f"  Latency: {r.latency_seconds:.2f}s  "
            f"Relevance: {r.relevance_score:.2f}  "
            f"Faithfulness: {r.faithfulness_score:.2f}\n"
        )
