from __future__ import annotations

import argparse
from pathlib import Path
import sys

from trace_bench.config import load_config
from trace_bench.registry import discover_tasks, load_task_bundle
from trace_bench.runner import BenchRunner
from trace_bench.ui import launch_ui


def cmd_list_tasks(root: str) -> int:
    specs = discover_tasks(root)
    for spec in specs:
        print(spec.key)
    return 0


def cmd_validate(config_path: str, root: str) -> int:
    cfg = load_config(config_path)
    tasks_root = Path(root)
    errors = 0
    for task in cfg.tasks:
        key = task.get("key") if isinstance(task, dict) else task
        try:
            load_task_bundle(key, tasks_root, eval_kwargs=cfg.eval_kwargs)
            print(f"[OK] {key}")
        except Exception as exc:
            errors += 1
            print(f"[FAIL] {key}: {exc}")
    return 1 if errors else 0


def cmd_run(config_path: str, root: str) -> int:
    cfg = load_config(config_path)
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

    val_p = sub.add_parser("validate", help="Validate tasks in config")
    val_p.add_argument("--config", required=True)
    val_p.add_argument("--root", default="LLM4AD/benchmark_tasks")

    run_p = sub.add_parser("run", help="Run a benchmark config")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--root", default="LLM4AD/benchmark_tasks")

    ui_p = sub.add_parser("ui", help="Launch Gradio UI (stub)")
    ui_p.add_argument("--runs-dir", default="runs")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "list-tasks":
        return cmd_list_tasks(args.root)
    if args.cmd == "validate":
        return cmd_validate(args.config, args.root)
    if args.cmd == "run":
        return cmd_run(args.config, args.root)
    if args.cmd == "ui":
        return cmd_ui(args.runs_dir)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
