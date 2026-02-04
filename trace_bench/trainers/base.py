from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from trace_bench.tasks import TraceProblem


class Trainer(ABC):
    @abstractmethod
    def train_step(self, problem: TraceProblem, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def eval_step(self, problem: TraceProblem, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError


__all__ = ["Trainer"]
