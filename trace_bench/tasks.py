from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib.util
import json
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path(path: Path) -> None:
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_opto_importable() -> None:
    try:
        import opto  # noqa: F401
        return
    except Exception:
        pass
    repo_root = _repo_root()
    candidate = repo_root.parent / "OpenTrace"
    _ensure_sys_path(candidate)


def ensure_llm4ad_importable(tasks_dir: Path) -> None:
    repo_root = _repo_root()
    _ensure_sys_path(repo_root)
    _ensure_sys_path(tasks_dir.parent)


@dataclass
class TraceProblem:
    param: Any
    guide: Any
    train_dataset: Dict[str, Any]
    optimizer_kwargs: Dict[str, Any]
    metadata: Dict[str, Any]
    module: Any


def _load_index(tasks_dir: Path) -> List[Dict[str, Any]]:
    index_path = tasks_dir / "index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text(encoding="utf-8"))


def _build_task_map(tasks_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in _load_index(tasks_dir):
        key = entry.get("key")
        module = entry.get("module") or entry.get("wrapper")
        wrapper = entry.get("wrapper")
        if key and module:
            mapping[key] = module
        if module:
            mapping[module] = module
        if wrapper:
            mapping[wrapper] = wrapper
    return mapping


def list_task_keys(tasks_dir: str | Path) -> List[str]:
    tasks_path = Path(tasks_dir)
    index = _load_index(tasks_path)
    if index:
        return [e.get("key") for e in index if e.get("key")]
    return sorted([p.name for p in tasks_path.iterdir() if p.is_dir()])


def load_task_module(task_key: str, tasks_dir: str | Path):
    tasks_path = Path(tasks_dir)
    ensure_opto_importable()
    ensure_llm4ad_importable(tasks_path)

    mapping = _build_task_map(tasks_path)
    module_dir = mapping.get(task_key, task_key)
    module_path = tasks_path / module_dir / "__init__.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Task module not found: {module_path}")

    module_name = f"trace_bench_task_{module_dir}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_trace_problem(task_key: str, tasks_dir: str | Path, eval_kwargs: Optional[Dict[str, Any]] = None) -> TraceProblem:
    mod = load_task_module(task_key, tasks_dir)
    if not hasattr(mod, "build_trace_problem"):
        raise AttributeError(f"Task module {task_key} missing build_trace_problem")
    bundle = mod.build_trace_problem(**(eval_kwargs or {}))
    required = {"param", "guide", "train_dataset", "optimizer_kwargs", "metadata"}
    if not required.issubset(bundle.keys()):
        missing = required - set(bundle.keys())
        raise KeyError(f"Task bundle missing keys: {sorted(missing)}")
    return TraceProblem(
        param=bundle["param"],
        guide=bundle["guide"],
        train_dataset=bundle["train_dataset"],
        optimizer_kwargs=bundle["optimizer_kwargs"],
        metadata=bundle["metadata"],
        module=mod,
    )


__all__ = ["TraceProblem", "list_task_keys", "load_trace_problem", "load_task_module"]
