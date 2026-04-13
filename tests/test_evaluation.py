"""Tests for evaluation metrics, tracker, and runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from docqa.evaluation.metrics import (
    EvaluationResult,
    calculate_faithfulness_score,
    calculate_relevance_score,
    evaluate_response,
)
from docqa.evaluation.runner import load_test_questions
from docqa.evaluation.tracker import ExperimentTracker


class TestRelevanceScore:
    def test_all_keywords_match(self):
        docs = [Document(page_content="cats and dogs play together", metadata={})]
        score = calculate_relevance_score("test", docs, ["cats", "dogs"])
        assert score == 1.0

    def test_partial_match(self):
        docs = [Document(page_content="only cats here", metadata={})]
        score = calculate_relevance_score("test", docs, ["cats", "dogs"])
        assert score == 0.5

    def test_no_keywords_returns_1(self):
        docs = [Document(page_content="anything", metadata={})]
        score = calculate_relevance_score("test", docs, None)
        assert score == 1.0

    def test_empty_docs_returns_0(self):
        score = calculate_relevance_score("test", [], ["cats"])
        assert score == 0.0


class TestFaithfulnessScore:
    def test_grounded_answer(self):
        docs = [Document(page_content="Cats are very popular pets worldwide", metadata={})]
        score = calculate_faithfulness_score("Cats are popular pets", docs)
        assert score > 0.5

    def test_ungrounded_answer(self):
        docs = [Document(page_content="Cats are popular pets", metadata={})]
        score = calculate_faithfulness_score("Elephants live in Africa and roam", docs)
        assert score < 0.5

    def test_empty_answer(self):
        docs = [Document(page_content="something", metadata={})]
        assert calculate_faithfulness_score("", docs) == 0.0

    def test_empty_docs(self):
        assert calculate_faithfulness_score("answer", []) == 0.0


class TestEvaluationResult:
    def test_to_dict(self):
        r = EvaluationResult(
            question="q", answer="a",
            sources=[{"content": "c", "metadata": {}}],
            latency_seconds=0.5, relevance_score=0.9, faithfulness_score=0.8,
        )
        d = r.to_dict()
        assert d["question"] == "q"
        assert d["num_sources"] == 1
        assert d["latency_seconds"] == 0.5


class TestEvaluateResponse:
    def test_returns_result_with_all_fields(self):
        sources = [{"content": "cats are great", "metadata": {"source": "f.txt"}}]
        result = evaluate_response(
            question="What about cats?", answer="Cats are great",
            sources=sources, latency=0.1, expected_keywords=["cats"],
        )
        assert isinstance(result, EvaluationResult)
        assert result.relevance_score is not None
        assert result.faithfulness_score is not None


class TestLoadQuestions:
    def test_loads_json(self, tmp_path: Path):
        data = {"questions": [{"id": "q1", "question": "What?", "expected_keywords": ["what"]}]}
        p = tmp_path / "qs.json"
        p.write_text(json.dumps(data))
        qs = load_test_questions(p)
        assert len(qs) == 1
        assert qs[0]["id"] == "q1"


class TestExperimentTracker:
    @patch("docqa.evaluation.tracker.MlflowClient")
    @patch("docqa.evaluation.tracker.mlflow")
    def test_init_creates_experiment(self, mock_mlflow, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        mock_client_cls.return_value = mock_client
        tracker = ExperimentTracker("test-exp", tracking_uri="file:///tmp/mlruns")
        mock_mlflow.create_experiment.assert_called_once_with("test-exp")

    @patch("docqa.evaluation.tracker.MlflowClient")
    @patch("docqa.evaluation.tracker.mlflow")
    def test_start_end_run(self, mock_mlflow, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_mlflow.start_run.return_value = mock_run
        tracker = ExperimentTracker("exp")
        run_id = tracker.start_run(run_name="r1")
        assert run_id == "abc123"
        tracker.end_run()
        mock_mlflow.end_run.assert_called_once()

    @patch("docqa.evaluation.tracker.MlflowClient")
    @patch("docqa.evaluation.tracker.mlflow")
    def test_log_evaluation(self, mock_mlflow, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MagicMock()
        mock_client_cls.return_value = mock_client
        tracker = ExperimentTracker("exp")
        result = EvaluationResult(
            question="q", answer="a", sources=[],
            latency_seconds=0.5, relevance_score=0.9, faithfulness_score=0.8,
        )
        tracker.log_evaluation(result, step=0)
        mock_mlflow.log_metrics.assert_called_once()
