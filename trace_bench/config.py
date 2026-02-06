from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import uuid


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = _load_text(path)
    # Prefer YAML if available
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("Config must be a mapping at top-level")
        return data
    except Exception:
        # Fallback to JSON for environments without PyYAML
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Config must be a mapping at top-level")
            return data
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse config {path}. Install PyYAML or use JSON syntax. Error: {exc}"
            )


def _as_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


@dataclass
class RunConfig:
    run_id: Optional[str] = None
    runs_dir: str = "runs"
    mode: str = "stub"
    seed: int = 123
    tasks: List[Any] = field(default_factory=list)
    trainers: List[Any] = field(default_factory=list)
    eval_kwargs: Dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        runs_dir = data.get("runs_dir", data.get("runs_root", "runs"))
        return cls(
            run_id=data.get("run_id"),
            runs_dir=runs_dir,
            mode=data.get("mode", "stub"),
            seed=int(data.get("seed", 123)),
            tasks=list(data.get("tasks", []) or []),
            trainers=list(data.get("trainers", []) or []),
            eval_kwargs=_as_dict(data.get("eval_kwargs")),
            trainer_kwargs=_as_dict(data.get("trainer_kwargs")),
        )

    def ensure_run_id(self) -> str:
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
        return self.run_id

    def snapshot(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "runs_dir": self.runs_dir,
            "mode": self.mode,
            "seed": self.seed,
            "tasks": self.tasks,
            "trainers": self.trainers,
            "eval_kwargs": dict(self.eval_kwargs),
            "trainer_kwargs": dict(self.trainer_kwargs),
        }


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    data = _load_yaml_or_json(config_path)
    return RunConfig.from_dict(data)


__all__ = ["RunConfig", "load_config"]
