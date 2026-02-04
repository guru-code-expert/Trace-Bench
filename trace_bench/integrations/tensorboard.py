from __future__ import annotations

from pathlib import Path


def ensure_tensorboard_dir(path: str | Path) -> Path:
    tb_path = Path(path)
    tb_path.mkdir(parents=True, exist_ok=True)
    return tb_path


__all__ = ["ensure_tensorboard_dir"]
