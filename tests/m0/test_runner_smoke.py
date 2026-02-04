import os
from pathlib import Path

from trace_bench.config import load_config
from trace_bench.runner import Runner


def test_runner_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    cfg = load_config("configs/smoke.yaml")
    cfg.runs_root = str(tmp_path / "runs")
    runner = Runner(cfg)
    runner.start()
    results = runner.run_matrix()

    assert len(results) == 1
    run_dirs = list((tmp_path / "runs").iterdir())
    assert run_dirs, "run dir not created"
    run_dir = run_dirs[0]
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "artifacts" / "metrics.csv").exists()
