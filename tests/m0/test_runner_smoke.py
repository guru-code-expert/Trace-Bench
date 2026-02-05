import os
from pathlib import Path
import importlib.util
import pytest

from trace_bench.config import load_config
from trace_bench.runner import BenchRunner


def test_runner_smoke(tmp_path):
    if importlib.util.find_spec("graphviz") is None:
        pytest.skip("graphviz not installed")

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    cfg = load_config("configs/smoke.yaml")
    cfg.runs_dir = str(tmp_path / "runs")

    runner = BenchRunner(cfg)
    summary = runner.run()

    assert summary.results
    run_dir = Path(cfg.runs_dir) / summary.run_id
    assert run_dir.exists()
    assert (run_dir / "config.snapshot.yaml").exists()
    assert (run_dir / "env.json").exists()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "events.jsonl").exists()
