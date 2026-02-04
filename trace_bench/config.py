from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import uuid


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def _default_dict(value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return dict(value or {})


@dataclass
class RunConfig:
    run_id: Optional[str] = None
    runs_root: str = "runs"
    seed: int = 123
    mode: str = "stub"
    tasks: List[str] = field(default_factory=list)
    trainers: List[str] = field(default_factory=list)
    llm: Dict[str, Any] = field(default_factory=dict)
    timeouts: Dict[str, Any] = field(default_factory=dict)
    threads: int = 1
    artifact: Dict[str, Any] = field(default_factory=dict)
    mlflow: Dict[str, Any] = field(default_factory=dict)
    tensorboard: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        data = dict(data or {})
        return cls(
            run_id=data.get("run_id"),
            runs_root=data.get("runs_root", "runs"),
            seed=int(data.get("seed", 123)),
            mode=data.get("mode", "stub"),
            tasks=list(data.get("tasks", []) or []),
            trainers=list(data.get("trainers", []) or []),
            llm=_default_dict(data.get("llm")),
            timeouts=_default_dict(data.get("timeouts")),
            threads=int(data.get("threads", 1)),
            artifact=_default_dict(data.get("artifact")),
            mlflow=_default_dict(data.get("mlflow")),
            tensorboard=_default_dict(data.get("tensorboard")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "runs_root": self.runs_root,
            "seed": self.seed,
            "mode": self.mode,
            "tasks": self.tasks,
            "trainers": self.trainers,
            "llm": dict(self.llm),
            "timeouts": dict(self.timeouts),
            "threads": self.threads,
            "artifact": dict(self.artifact),
            "mlflow": dict(self.mlflow),
            "tensorboard": dict(self.tensorboard),
        }

    def ensure_run_id(self) -> str:
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
        return self.run_id


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    data = _load_yaml_or_json(config_path)
    return RunConfig.from_dict(data)


__all__ = ["RunConfig", "load_config"]
