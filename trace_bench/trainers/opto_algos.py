from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

from trace_bench.tasks import TraceProblem
from trace_bench.trainers.base import Trainer


def _get_param_data(param: Any) -> Any:
    for attr in ("data", "value", "get", "get_value"):
        if hasattr(param, attr):
            try:
                val = getattr(param, attr)
                return val() if callable(val) else val
            except Exception:
                pass
    return getattr(param, "__dict__", None)


def _baseline_eval(problem: TraceProblem) -> Dict[str, Any]:
    guide = problem.guide
    param = problem.param
    dataset = problem.train_dataset
    inputs = dataset.get("inputs", [])
    infos = dataset.get("infos", [])
    if not inputs or not infos:
        return {"score": None, "feedback": "no_dataset"}
    code = _get_param_data(param)
    try:
        score, feedback = guide("", code, infos[0])
    except Exception as exc:
        return {"score": None, "feedback": f"baseline_error: {exc}"}
    return {"score": score, "feedback": feedback}


@dataclass
class OptoAlgorithmTrainer(Trainer):
    algo_key: str
    dry_run: bool = False
    trainer_overrides: Optional[Dict[str, Any]] = None

    def train_step(self, problem: TraceProblem, **kwargs) -> Dict[str, Any]:
        start = time.time()
        baseline = _baseline_eval(problem)
        if self.dry_run:
            return {
                "status": "skipped",
                "score": baseline.get("score"),
                "feedback": baseline.get("feedback"),
                "elapsed": time.time() - start,
            }

        algo_cls = self._resolve_algo()
        if algo_cls is None:
            return {
                "status": "skipped",
                "score": baseline.get("score"),
                "feedback": "algorithm_unavailable",
                "elapsed": time.time() - start,
            }

        trained = self._try_opto_train(problem, algo_cls)
        if not trained:
            return {
                "status": "skipped",
                "score": baseline.get("score"),
                "feedback": "trainer_unavailable",
                "elapsed": time.time() - start,
            }

        final = _baseline_eval(problem)
        return {
            "status": "trained",
            "score": final.get("score"),
            "feedback": final.get("feedback"),
            "elapsed": time.time() - start,
        }

    def eval_step(self, problem: TraceProblem, **kwargs) -> Dict[str, Any]:
        result = _baseline_eval(problem)
        result["status"] = "evaluated"
        return result

    def save_checkpoint(self, path: str) -> None:
        return None

    def _resolve_algo(self):
        algo_map = {
            "PrioritySearch": ("opto.features.priority_search", "PrioritySearch"),
            "GEPA-Base": ("opto.features.gepa.gepa_algorithms", "GEPAAlgorithmBase"),
            "GEPA-UCB": ("opto.features.gepa.gepa_algorithms", "GEPAUCBSearch"),
            "GEPA-Beam": ("opto.features.gepa.gepa_algorithms", "GEPABeamPareto"),
        }
        if self.algo_key not in algo_map:
            return None
        module_name, cls_name = algo_map[self.algo_key]
        try:
            module = __import__(module_name, fromlist=[cls_name])
            return getattr(module, cls_name)
        except Exception:
            return None

    def _try_opto_train(self, problem: TraceProblem, algo_cls) -> bool:
        try:
            from opto import trainer as opto_trainer
        except Exception:
            return False

        if not hasattr(opto_trainer, "train"):
            return False

        overrides = dict(self.trainer_overrides or {})
        params: Dict[str, Any]
        if self.algo_key == "PrioritySearch":
            params = dict(
                guide=problem.guide,
                train_dataset=problem.train_dataset,
                score_range=problem.optimizer_kwargs.get("score_range", [-10, 10]),
                num_epochs=1,
                num_steps=overrides.get("ps_steps", 1),
                batch_size=1,
                num_batches=overrides.get("ps_batches", 1),
                verbose=False,
                num_candidates=overrides.get("ps_candidates", 2),
                num_proposals=overrides.get("ps_proposals", 2),
                memory_update_frequency=overrides.get("ps_mem_update", 2),
                optimizer_kwargs=problem.optimizer_kwargs,
                num_threads=overrides.get("threads", 1),
            )
        else:
            params = dict(
                guide=problem.guide,
                train_dataset=problem.train_dataset,
                validate_dataset=problem.train_dataset,
                num_iters=overrides.get("gepa_iters", 1),
                train_batch_size=overrides.get("gepa_train_bs", 2),
                merge_every=overrides.get("gepa_merge_every", 2),
                pareto_subset_size=overrides.get("gepa_pareto_subset", 2),
                num_threads=overrides.get("threads", 1),
                optimizer_kwargs=problem.optimizer_kwargs,
            )

        try:
            opto_trainer.train(model=problem.param, algorithm=algo_cls, **params)
            return True
        except Exception:
            return False


def get_trainer(algo_key: str, *, dry_run: bool, trainer_overrides: Optional[Dict[str, Any]] = None) -> Trainer:
    return OptoAlgorithmTrainer(algo_key=algo_key, dry_run=dry_run, trainer_overrides=trainer_overrides)


__all__ = ["OptoAlgorithmTrainer", "get_trainer"]
