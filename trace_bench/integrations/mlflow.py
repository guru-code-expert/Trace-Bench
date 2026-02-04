from __future__ import annotations

from pathlib import Path


def write_mlflow_link(path: str | Path, link: str) -> None:
    Path(path).write_text(link.strip() + "\n", encoding="utf-8")


__all__ = ["write_mlflow_link"]
