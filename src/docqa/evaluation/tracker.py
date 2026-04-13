"""MLflow experiment tracker for RAG evaluations."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from docqa.evaluation.metrics import EvaluationResult


class ExperimentTracker:
    """Track evaluation experiments with MLflow.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name (created if it doesn't exist).
    tracking_uri:
        MLflow tracking server URI.  Defaults to a local ``file://``
        store under ``./mlruns``.
    """

    def __init__(
        self,
        experiment_name: str = "rag-evaluation",
        tracking_uri: Optional[str] = None,
    ) -> None:
        if tracking_uri is None:
            tracking_uri = f"file://{Path('./mlruns').resolve()}"
        mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        self._active_run = None

    # ── run lifecycle ─────────────────────────────────────────────

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new MLflow run, returning its ID."""
        self._active_run = mlflow.start_run(run_name=run_name, tags=tags)
        return self._active_run.info.run_id

    def end_run(self) -> None:
        """End the current active run (if any)."""
        if self._active_run is not None:
            mlflow.end_run()
            self._active_run = None

    # ── logging helpers ───────────────────────────────────────────

    def log_params(self, params: Dict[str, object]) -> None:
        mlflow.log_params(params)

    def log_config(
        self,
        llm_provider: str,
        llm_model: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        k: int,
    ) -> None:
        """Log standard RAG configuration as MLflow params."""
        mlflow.log_params(
            {
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "k": k,
            }
        )

    def log_evaluation(
        self,
        result: EvaluationResult,
        step: Optional[int] = None,
    ) -> None:
        """Log metrics from a single :class:`EvaluationResult`."""
        metrics: Dict[str, float] = {
            "latency_seconds": result.latency_seconds,
            "num_sources": float(len(result.sources)),
        }
        if result.relevance_score is not None:
            metrics["relevance_score"] = result.relevance_score
        if result.faithfulness_score is not None:
            metrics["faithfulness_score"] = result.faithfulness_score
        mlflow.log_metrics(metrics, step=step)

    def log_batch_results(self, results: List[EvaluationResult]) -> None:
        """Log aggregate metrics across a batch of results."""
        if not results:
            return

        latencies = [r.latency_seconds for r in results]
        relevances = [r.relevance_score for r in results if r.relevance_score is not None]
        faithfulness = [
            r.faithfulness_score for r in results if r.faithfulness_score is not None
        ]

        metrics: Dict[str, float] = {
            "avg_latency_seconds": sum(latencies) / len(latencies),
            "total_queries": float(len(results)),
        }
        if relevances:
            metrics["avg_relevance_score"] = sum(relevances) / len(relevances)
        if faithfulness:
            metrics["avg_faithfulness_score"] = sum(faithfulness) / len(faithfulness)

        mlflow.log_metrics(metrics)

    # ── artifacts ─────────────────────────────────────────────────

    def log_artifact(
        self,
        file_path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        mlflow.log_artifact(file_path, artifact_path)

    def log_dict_artifact(self, data: Dict, filename: str) -> None:
        """Write *data* as JSON and log as an MLflow artifact."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        try:
            json.dump(data, tmp, indent=2, default=str)
            tmp.close()
            mlflow.log_artifact(tmp.name, "results")
        finally:
            os.unlink(tmp.name)

    # ── property ──────────────────────────────────────────────────

    @property
    def run_id(self) -> Optional[str]:
        if self._active_run is not None:
            return self._active_run.info.run_id
        return None
