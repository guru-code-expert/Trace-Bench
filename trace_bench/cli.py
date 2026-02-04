from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trace_bench.config import load_config
from trace_bench.runner import Runner
from trace_bench.tasks import list_task_keys, load_trace_problem


def cmd_run(config_path: str) -> int:
    config = load_config(config_path)
    runner = Runner(config)
    runner.start()
    runner.run_matrix()
    runner.stop()
    return 0


def cmd_list_tasks(tasks_dir: str) -> int:
    tasks = list_task_keys(tasks_dir)
    for key in tasks:
        print(key)
    return 0


def cmd_validate(config_path: str) -> int:
    config = load_config(config_path)
    tasks_dir = Path("LLM4AD") / "benchmark_tasks"
    errors = 0
    for task_key in config.tasks:
        try:
            load_trace_problem(task_key, tasks_dir, eval_kwargs=config.timeouts)
            print(f"[OK] {task_key}")
        except Exception as exc:
            errors += 1
            print(f"[FAIL] {task_key}: {exc}")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trace_bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a benchmark matrix")
    run_p.add_argument("--config", required=True, help="Path to config YAML")

    list_p = sub.add_parser("list-tasks", help="List available tasks")
    list_p.add_argument("--tasks", required=True, help="Tasks directory")

    val_p = sub.add_parser("validate", help="Validate tasks in config")
    val_p.add_argument("--config", required=True, help="Path to config YAML")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return cmd_run(args.config)
    if args.cmd == "list-tasks":
        return cmd_list_tasks(args.tasks)
    if args.cmd == "validate":
        return cmd_validate(args.config)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
