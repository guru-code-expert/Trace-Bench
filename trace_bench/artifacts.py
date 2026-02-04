from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import csv
import json
import os
from datetime import datetime
import uuid


@dataclass
class RunArtifacts:
    run_dir: Path
    artifacts_dir: Path
    raw_traces_dir: Path
    tensorboard_dir: Path

    def ensure(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.raw_traces_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)


def create_run_dir(runs_root: str, run_id: Optional[str] = None) -> RunArtifacts:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_id = run_id or str(uuid.uuid4())
    run_dir = Path(runs_root) / f"{ts}-{run_id}"
    artifacts_dir = run_dir / "artifacts"
    raw_traces_dir = artifacts_dir / "raw_traces"
    tensorboard_dir = run_dir / "tensorboard_dir"
    return RunArtifacts(run_dir, artifacts_dir, raw_traces_dir, tensorboard_dir)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_config(path: Path, config: Dict[str, Any]) -> None:
    # JSON is valid YAML; keeps us dependency-free.
    content = json.dumps(config, indent=2, sort_keys=True)
    write_text(path, content)


def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    content = json.dumps(metadata, indent=2, sort_keys=True)
    write_text(path, content)


def write_env(path: Path, extra_keys: Optional[List[str]] = None) -> None:
    keys = sorted(os.environ.keys())
    if extra_keys:
        for key in extra_keys:
            if key not in keys and key in os.environ:
                keys.append(key)
    lines = [f"{k}={os.environ.get(k, '')}" for k in keys]
    write_text(path, "\n".join(lines))


def append_metrics(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_summary(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = [
    "RunArtifacts",
    "create_run_dir",
    "write_config",
    "write_metadata",
    "write_env",
    "append_metrics",
    "append_summary",
    "write_text",
]
