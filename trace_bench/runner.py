from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

from trace_bench.adapters.llm import llm_from_config, LLMAdapter
from trace_bench.artifacts import (
    RunArtifacts,
    create_run_dir,
    write_config,
    write_metadata,
    write_env,
    append_metrics,
    append_summary,
    write_text,
)
from trace_bench.config import RunConfig
from trace_bench.tasks import load_trace_problem
from trace_bench.trainers.opto_algos import get_trainer


class Runner:
    def __init__(self, config: RunConfig, llm: Optional[LLMAdapter] = None) -> None:
        self.config = config
        self.config.ensure_run_id()
        self.llm = llm or llm_from_config(self.config.mode, self.config.llm)
        self.artifacts: Optional[RunArtifacts] = None

    def start(self) -> None:
        self.artifacts = create_run_dir(self.config.runs_root, self.config.run_id)
        self.artifacts.ensure()
        write_config(self.artifacts.run_dir / "config.yaml", self.config.to_dict())
        write_metadata(
            self.artifacts.run_dir / "run_metadata.json",
            {
                "run_id": self.config.run_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "mode": self.config.mode,
                "tasks": self.config.tasks,
                "trainers": self.config.trainers,
                "llm": self.config.llm,
            },
        )
        write_env(self.artifacts.run_dir / "env.txt", extra_keys=["TRACE_MODE", "TRACE_LITELLM_MODEL"])
        write_text(self.artifacts.run_dir / "mlflow_link.txt", "")
        write_text(self.artifacts.run_dir / "stdout.log", "")
        write_text(self.artifacts.run_dir / "stderr.log", "")

    def stop(self) -> None:
        return None

    def run_task(self, task_key: str, trainer_key: str) -> Dict[str, Any]:
        if self.artifacts is None:
            self.start()
        assert self.artifacts is not None

        problem = load_trace_problem(task_key, Path("LLM4AD") / "benchmark_tasks", eval_kwargs=self.config.timeouts)
        dry_run = self.config.mode == "stub" and self.config.llm.get("provider") == "stub"
        trainer = get_trainer(
            trainer_key,
            dry_run=dry_run,
            trainer_overrides={"threads": self.config.threads, **self.config.timeouts},
        )
        result = trainer.train_step(problem)

        metrics_row = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "task": task_key,
            "trainer": trainer_key,
            "status": result.get("status"),
            "score": result.get("score"),
            "elapsed": result.get("elapsed"),
        }
        append_metrics(
            self.artifacts.artifacts_dir / "metrics.csv",
            ["timestamp", "task", "trainer", "status", "score", "elapsed"],
            metrics_row,
        )
        append_summary(self.artifacts.artifacts_dir / "summary.jsonl", metrics_row)
        return metrics_row

    def run_matrix(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for task_key in self.config.tasks:
            for trainer_key in self.config.trainers:
                results.append(self.run_task(task_key, trainer_key))
        return results


__all__ = ["Runner"]
