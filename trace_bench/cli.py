from __future__ import annotations

import argparse
from pathlib import Path
import sys

from trace_bench.config import load_config
from trace_bench.registry import discover_tasks, load_task_bundle
from trace_bench.runner import BenchRunner
from trace_bench.ui import launch_ui


def cmd_list_tasks(root: str, bench: str | None = None) -> int:
    specs = discover_tasks(root, bench=bench)
    for spec in specs:
        print(spec.key)
    return 0


def _task_in_bench(task_key: str, bench: str | None) -> bool:
    if not bench:
        return True
    if "veribench" in bench and task_key.startswith("veribench:"):
        return True
    if "examples" in bench and task_key.startswith("example:"):
        return True
    if "llm4ad" in bench and not task_key.startswith(("example:", "veribench:")):
        return True
    return False


def cmd_validate(config_path: str, root: str, bench: str | None = None) -> int:
    cfg = load_config(config_path)
    tasks_root = Path(root)
    errors = 0
    if bench:
        discover_tasks(tasks_root, bench=bench)
    for task in cfg.tasks:
        key = task.get("key") if isinstance(task, dict) else task
        if not _task_in_bench(key, bench):
            continue
        try:
            load_task_bundle(key, tasks_root, eval_kwargs=cfg.eval_kwargs)
            print(f"[OK] {key}")
        except Exception as exc:
            errors += 1
            print(f"[FAIL] {key}: {exc}")
    return 1 if errors else 0


def cmd_run(config_path: str, root: str, runs_dir: str | None = None) -> int:
    cfg = load_config(config_path)
    if runs_dir:
        cfg.runs_dir = runs_dir
    runner = BenchRunner(cfg, tasks_root=root)
    runner.run()
    return 0


def cmd_ui(runs_dir: str) -> int:
    return launch_ui(runs_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list-tasks", help="List discoverable tasks")
    list_p.add_argument("--root", default="LLM4AD/benchmark_tasks")
    list_p.add_argument("--bench", default=None, help="Bench selection: llm4ad,examples,veribench")

    val_p = sub.add_parser("validate", help="Validate tasks in config")
    val_p.add_argument("--config", required=True)
    val_p.add_argument("--root", default="LLM4AD/benchmark_tasks")
    val_p.add_argument("--bench", default=None, help="Bench selection: llm4ad,examples,veribench")

    run_p = sub.add_parser("run", help="Run a benchmark config")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--root", default="LLM4AD/benchmark_tasks")
    run_p.add_argument("--runs-dir", default=None)

    ui_p = sub.add_parser("ui", help="Launch Gradio UI (stub)")
    ui_p.add_argument("--runs-dir", default="runs")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "list-tasks":
        return cmd_list_tasks(args.root, args.bench)
    if args.cmd == "validate":
        return cmd_validate(args.config, args.root, args.bench)
    if args.cmd == "run":
        return cmd_run(args.config, args.root, args.runs_dir)
    if args.cmd == "ui":
        return cmd_ui(args.runs_dir)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
