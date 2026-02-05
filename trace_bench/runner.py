from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import os
import random

from trace_bench.artifacts import (
    RunArtifacts,
    append_event,
    append_results_csv,
    init_run_dir,
    write_config_snapshot,
    write_env_json,
    write_summary,
)
from trace_bench.config import RunConfig
from trace_bench.registry import discover_tasks, load_task_bundle


try:
    from opto.trace.nodes import ParameterNode
except Exception:  # pragma: no cover - only when opto is not available
    ParameterNode = object  # type: ignore


@dataclass
class RunSummary:
    run_id: str
    results: List[Dict[str, Any]]


def _normalize_tasks(tasks: List[Any], default_eval: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in tasks:
        if isinstance(item, str):
            normalized.append({"key": item, "eval_kwargs": dict(default_eval)})
        elif isinstance(item, dict):
            key = item.get("key") or item.get("task")
            if not key:
                raise ValueError(f"Task entry missing key: {item}")
            eval_kwargs = dict(default_eval)
            eval_kwargs.update(item.get("eval_kwargs", {}))
            normalized.append({"key": key, "eval_kwargs": eval_kwargs})
        else:
            raise ValueError(f"Unsupported task entry: {item}")
    return normalized


def _extract_response(model: Any, input_value: Any) -> Any:
    if isinstance(model, ParameterNode):
        return getattr(model, "data", model)
    if callable(model):
        output = model(input_value)
        return getattr(output, "data", output)
    return getattr(model, "data", model)


def _evaluate_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    dataset = bundle["train_dataset"]
    guide = bundle["guide"]
    inputs = dataset.get("inputs") or []
    infos = dataset.get("infos") or []
    if not inputs or not infos:
        return {"score": None, "feedback": "empty_dataset"}
    task_input = inputs[0]
    task_info = infos[0]
    response = _extract_response(bundle["param"], task_input)
    try:
        score, feedback = guide(task_input, response, task_info)
    except Exception as exc:
        return {"score": None, "feedback": f"eval_error: {exc}"}
    return {"score": score, "feedback": feedback}


def _resolve_algorithm(name: str):
    if name == "PrioritySearch":
        return "PrioritySearch"
    if name == "GEPA-Base":
        from opto.features.gepa.gepa_algorithms import GEPAAlgorithmBase
        return GEPAAlgorithmBase
    if name == "GEPA-UCB":
        from opto.features.gepa.gepa_algorithms import GEPAUCBSearch
        return GEPAUCBSearch
    if name == "GEPA-Beam":
        from opto.features.gepa.gepa_algorithms import GEPABeamPareto
        return GEPABeamPareto
    return name


def _default_trainer_kwargs(algo_name: str) -> Dict[str, Any]:
    if algo_name == "PrioritySearch":
        return dict(num_epochs=1, num_steps=1, num_batches=1, num_candidates=2, num_proposals=2)
    return dict(num_iters=1, num_search_iterations=1, train_batch_size=2, merge_every=2, pareto_subset_size=2)


def _train_bundle(bundle: Dict[str, Any], algo_name: str, trainer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    from opto import trainer as opto_trainer

    algo = _resolve_algorithm(algo_name)
    kwargs = _default_trainer_kwargs(algo_name)
    kwargs.update(trainer_kwargs)

    try:
        opto_trainer.train(
            model=bundle["param"],
            train_dataset=bundle["train_dataset"],
            algorithm=algo,
            guide=bundle["guide"],
            optimizer_kwargs=bundle.get("optimizer_kwargs", {}),
            **kwargs,
        )
        return {"status": "trained"}
    except Exception as exc:
        return {"status": "train_error", "error": str(exc)}


class BenchRunner:
    def __init__(self, config: RunConfig, tasks_root: str | Path = "LLM4AD/benchmark_tasks"):
        self.config = config
        self.tasks_root = Path(tasks_root)
        self.config.ensure_run_id()
        random.seed(self.config.seed)
        self.artifacts: Optional[RunArtifacts] = None

    def run(self) -> RunSummary:
        self.artifacts = init_run_dir(self.config.runs_dir, self.config.run_id)
        write_config_snapshot(self.artifacts.config_snapshot, self.config.snapshot())
        write_env_json(self.artifacts.env_json)

        tasks = _normalize_tasks(self.config.tasks, self.config.eval_kwargs)
        trainers = self.config.trainers or ["PrioritySearch"]

        results: List[Dict[str, Any]] = []
        for task in tasks:
            for trainer in trainers:
                results.append(self._run_task(task, trainer))

        write_summary(self.artifacts.summary_json, {"run_id": self.config.run_id, "results": results})
        return RunSummary(run_id=self.config.run_id, results=results)

    def _run_task(self, task: Dict[str, Any], trainer_name: str) -> Dict[str, Any]:
        assert self.artifacts is not None
        task_key = task["key"]
        eval_kwargs = task.get("eval_kwargs", {})
        bundle = load_task_bundle(task_key, self.tasks_root, eval_kwargs=eval_kwargs)

        start = datetime.utcnow().isoformat() + "Z"
        if self.config.mode == "stub":
            train_result = {"status": "skipped"}
        else:
            train_result = _train_bundle(bundle, trainer_name, self.config.trainer_kwargs)

        eval_result = _evaluate_bundle(bundle)
        row = {
            "timestamp": start,
            "task": task_key,
            "trainer": trainer_name,
            "status": train_result.get("status"),
            "score": eval_result.get("score"),
            "feedback": eval_result.get("feedback"),
        }
        append_results_csv(
            self.artifacts.results_csv,
            ["timestamp", "task", "trainer", "status", "score", "feedback"],
            row,
        )
        append_event(self.artifacts.events_jsonl, row)
        return row


__all__ = ["BenchRunner", "RunSummary"]
