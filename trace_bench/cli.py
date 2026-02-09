from __future__ import annotations

import argparse
from pathlib import Path
import sys

from trace_bench.config import load_config
from trace_bench.matrix import expand_matrix
from trace_bench.registry import discover_tasks, discover_trainers, load_task_bundle
from trace_bench.runner import BenchRunner
from trace_bench.ui import launch_ui


def cmd_list_tasks(root: str, bench: str | None = None) -> int:
    specs = discover_tasks(root, bench=bench)
    for spec in specs:
        print(spec.id)
    return 0


def cmd_list_trainers() -> int:
    specs = discover_trainers()
    for spec in specs:
        status = "available" if spec.available else "unavailable"
        print(f"{spec.id}\t{status}")
    return 0


def _task_in_bench(task_key: str, bench: str | None) -> bool:
    if not bench:
        return True
    if ":" not in task_key:
        task_key = f"llm4ad:{task_key}"
    if "veribench" in bench and task_key.startswith("veribench:"):
        return True
    if "trace_examples" in bench and task_key.startswith("trace_examples:"):
        return True
    if "internal" in bench and task_key.startswith("internal:"):
        return True
    if "llm4ad" in bench and task_key.startswith("llm4ad:"):
        return True
    return False


def cmd_validate(config_path: str, root: str, bench: str | None = None, strict: bool = False) -> int:
    cfg = load_config(config_path)
    tasks_root = Path(root)
    errors = 0
    if bench:
        discover_tasks(tasks_root, bench=bench)
    trainers = discover_trainers()
    trainer_ids = {t.id for t in trainers if t.available}
    for trainer in cfg.trainers:
        if trainer.id not in trainer_ids:
            errors += 1
            print(f"[FAIL] trainer {trainer.id}: not available")

    for task in cfg.tasks:
        task_id = task.id
        if not _task_in_bench(task_id, bench):
            continue
        try:
            load_task_bundle(task_id, tasks_root, eval_kwargs=task.eval_kwargs)
            print(f"[OK] {task_id}")
        except NotImplementedError as exc:
            print(f"[SKIP] {task_id}: {exc}")
        except Exception as exc:
            errors += 1
            print(f"[FAIL] {task_id}: {exc}")

    if strict:
        jobs = expand_matrix(cfg)
        if not jobs:
            errors += 1
            print("[FAIL] matrix: no jobs expanded")
        else:
            print(f"\n[OK] matrix: {len(jobs)} jobs expanded deterministically")
            seen_trainers: set[str] = set()
            seen_tasks: set[str] = set()
            for job in jobs:
                seen_trainers.add(job.trainer_id)
                seen_tasks.add(job.task_id)
                print(f"  job {job.job_id}: {job.task_id} x {job.trainer_id} (seed={job.seed})")
            print(f"\n  tasks:    {sorted(seen_tasks)}")
            print(f"  trainers: {sorted(seen_trainers)}")
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
    list_p.add_argument("--bench", default=None, help="Bench selection: llm4ad,trace_examples,internal,veribench")

    list_t = sub.add_parser("list-trainers", help="List discoverable trainers")

    val_p = sub.add_parser("validate", help="Validate tasks in config")
    val_p.add_argument("--config", required=True)
    val_p.add_argument("--root", default="LLM4AD/benchmark_tasks")
    val_p.add_argument("--bench", default=None, help="Bench selection: llm4ad,trace_examples,internal,veribench")
    val_p.add_argument("--strict", action="store_true")

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
    if args.cmd == "list-trainers":
        return cmd_list_trainers()
    if args.cmd == "validate":
        return cmd_validate(args.config, args.root, args.bench, args.strict)
    if args.cmd == "run":
        return cmd_run(args.config, args.root, args.runs_dir)
    if args.cmd == "ui":
        return cmd_ui(args.runs_dir)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
