from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import csv
import json
import os
import subprocess
from datetime import datetime


@dataclass
class RunArtifacts:
    run_dir: Path

    @property
    def config_snapshot(self) -> Path:
        return self.run_dir / "config.snapshot.yaml"

    @property
    def env_json(self) -> Path:
        return self.run_dir / "env.json"

    @property
    def results_csv(self) -> Path:
        return self.run_dir / "results.csv"

    @property
    def events_jsonl(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def summary_json(self) -> Path:
        return self.run_dir / "summary.json"


def init_run_dir(runs_dir: str, run_id: str) -> RunArtifacts:
    run_path = Path(runs_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(run_path)


def _dump_yaml_or_json(data: Dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore
        return yaml.safe_dump(data, sort_keys=False)
    except Exception:
        return json.dumps(data, indent=2, sort_keys=False)


def write_config_snapshot(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(_dump_yaml_or_json(data), encoding="utf-8")


def _git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        root = Path(__file__).resolve().parents[1]
        info["repo_root"] = str(root)
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
        info["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root).decode().strip()
        return info
    except Exception:
        return info


def write_env_json(path: Path) -> None:
    env = {k: os.environ.get(k, "") for k in sorted(os.environ.keys())}
    payload = {
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "env": env,
        "git": _git_info(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_results_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_event(path: Path, event: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


__all__ = [
    "RunArtifacts",
    "init_run_dir",
    "write_config_snapshot",
    "write_env_json",
    "append_results_csv",
    "append_event",
    "write_summary",
]
