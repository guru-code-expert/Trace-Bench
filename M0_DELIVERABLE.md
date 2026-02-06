# M0 — Trace‑Bench Plan (Two Approaches + Explicit Acceptance)

**Executive Summary**
This M0 provides two concrete implementation approaches so you can choose the best path. Both keep VeriBench and OpenTrace examples **in scope**, define explicit acceptance targets, and specify MLflow/TensorBoard/Gradio wiring (not just stubs). The current codebase already includes a minimal runner/CLI and a Colab notebook; this update aligns the plan with the agreed scope and adds missing acceptance criteria, security constraints, and trainer-parameter configuration.

**Spec Alignment**
Primary source: `Trace-Bench UpWork specifications.md` (sections 1–9). This M0 is the formal plan for M1–M3 implementation and acceptance.

## 1) Two Plan Variants (Pick A or B)

**Plan A — Lean/Staged (Lower risk, faster M1)**
- M1 focuses on internal Trace examples + LLM4AD first.
- VeriBench discovery and smoke coverage land in M2.
- UI/MLflow/TB wiring lands in M3.

**Plan B — Compatibility‑First (Broader early coverage)**
- M1 includes internal examples + LLM4AD + VeriBench discovery + minimal smoke.
- M2 adds matrix runs + aggregation.
- M3 adds UI/MLflow/TB wiring.

**Key tradeoff**: Plan A reduces early risk and review load; Plan B maximizes breadth earlier at the cost of additional integration risk in M1.

## 2) Scope and Coverage Targets (Explicit)

**LLM4AD coverage**
- Total tasks: 60 (per spec).
- Acceptance target: **>=80% (>=48) functional in Real mode**.
- “Functional” means: a run completes with at least one training step and produces artifacts.
- “Optimizing” means: on a defined subset (10 tasks), the score improves vs baseline by a positive delta or changes from non‑finite to finite.
- Proposed improvement subset (tunable):
  - `circle_packing`
  - `optimization_knapsack_construct`
  - `optimization_bp_1d_construct`
  - `optimization_tsp_construct`
  - `optimization_cvrp_construct`
  - `optimization_jssp_construct`
  - `optimization_qap_construct`
  - `optimization_set_cover_construct`
  - `optimization_vrptw_construct`
  - `optimization_cflp_construct`

**VeriBench coverage**
- Acceptance target: **>=80% functional in Real mode**.
- Requires Trace team to provide task list/entrypoint (kept in scope).
- Validation command (to be finalized once entrypoint is known):
  - `trace-bench list-tasks --bench veribench`
  - `trace-bench validate --bench veribench --config configs/veribench_smoke.yaml`

**OpenTrace examples**
- Acceptance target: **100% smoke tests pass**.
- CI test requirement: import every script in `OpenTrace/examples/` and run `--help` for argparse scripts (no datasets/keys required).

## 3) Repository Grounding

- LLM4AD tasks: `LLM4AD/benchmark_tasks/*/__init__.py` with `build_trace_problem()`.
- LLM4AD loader: `LLM4AD/llm4ad_loader.py`.
- OpenTrace examples: `OpenTrace/examples/` (smoke import/--help required).
- VeriBench: in scope; entrypoint to be confirmed by Trace team.

Target compatibility: `OpenTrace@experimental` (commit `c0a0282b45a8ab0ec4f927611dd1847f4e15519f`).

## 4) Architecture (Minimal Surface)

**Files**
- `trace_bench/config.py` — RunConfig parsing/validation (YAML/JSON).
- `trace_bench/registry.py` — Task discovery + loaders (LLM4AD + examples + VeriBench later).
- `trace_bench/runner.py` — BenchRunner orchestration (stub + real modes).
- `trace_bench/artifacts.py` — Standardized run outputs (secure env allowlist).
- `trace_bench/cli.py` — `trace-bench` CLI + module entry.
- `trace_bench/ui.py` — Gradio UI (M3 wiring).
- `trace_bench/examples/*` — example tasks with proper Trace nodes/bundles.

**Public API**
```python
from trace_bench.config import RunConfig, load_config
from trace_bench.runner import BenchRunner

cfg = load_config("configs/smoke.yaml")
runner = BenchRunner(cfg)
summary = runner.run()
```

## 5) CLI and Config Schema (Trainer Params Supported)

**CLI**
```
trace-bench list-tasks --root LLM4AD/benchmark_tasks
trace-bench validate --config configs/smoke.yaml
trace-bench run --config configs/smoke.yaml --runs-dir /abs/path/runs
trace-bench ui --runs-dir /abs/path/runs
```

**RunConfig YAML** (per‑trainer params + optional optimizer/guide/logger overrides)
```yaml
runs_dir: /abs/path/runs
mode: real
seed: 123

tasks:
  - circle_packing
  - example:greeting_stub

trainers:
  - name: PrioritySearch
    params:
      num_steps: 1
      num_batches: 1
    optimizer: OptoPrimeV2
    optimizer_kwargs:
      memory_size: 10
    guide: LLMJudge
    guide_kwargs:
      temperature: 0.2
    logger: ConsoleLogger
    logger_kwargs: {}

  - name: GEPA-Base
    params:
      num_iters: 1
      pareto_subset_size: 2

eval_kwargs:
  timeout_seconds: 10
```

## 6) Standardized Run Artifacts + Security

Run directory layout:
```
runs/<run_id>/
  config.snapshot.yaml
  env.json
  results.csv
  events.jsonl
  summary.json
```

**Security requirement**
- `env.json` uses **allowlist + redaction** (no full environment dump).
- All secret‑like keys (contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`) are redacted.

## 7) MLflow + TensorBoard + Gradio (Specified Wiring)

**MLflow (M3)**
- Params logged per task × trainer:
  - `run_id`, `mode`, `seed`, `task`, `trainer`, `optimizer`, `guide`, `logger`.
- Metrics:
  - `eval_score`, `train_score`, `best_score`, `step`.
- Artifacts:
  - `config.snapshot.yaml`, `results.csv`, `events.jsonl`, `summary.json`, `env.json` (redacted).

**TensorBoard (M3)**
- Logdir: `runs/<run_id>/tb/`.
- Scalars:
  - `train/score`, `eval/score`, `train/step`, `train/batch`.

**Gradio UI (M3)**
- Screens:
  - Config selector (file picker + inline edit)
  - Launch run (mode, tasks, trainers)
  - Recent runs table
  - Run detail (summary, results table, MLflow run ID/URI, TB logdir + command)

## 8) Tests / CI

**Unit**
- Config parsing defaults.
- Artifact writer schema.

**Integration**
- Stub mode: deterministic end‑to‑end optimization loop (internal example).
- Real mode: minimal run with user key (Colab).

**OpenTrace examples smoke test**
- Import every `.py` under `OpenTrace/examples/`.
- For argparse scripts: `python <script> --help`.

## 9) Notebook Plan (Colab)

- Uses `bench_dir()` to persist `runs_dir` to Drive by default.
- Stub run uses internal example task.
- Real run uses the same internal example with API key.
- Optional LLM4AD smoke cell.
- CLI uses `--runs-dir` override to guarantee persistence.

## 10) Acceptance Criteria (Explicit)

**M0**
- Two plan variants delivered (A/B) with explicit acceptance criteria.
- Coverage targets stated for LLM4AD, VeriBench, OpenTrace examples.

**M1**
- `trace-bench run --config configs/smoke.yaml --runs-dir <abs>` works.
- Stub run completes with training (no status=skipped).
- Real run completes with user key.
- Colab `01_smoke_runner.ipynb` runs end‑to‑end.
- `pytest -q` passes.

**M2**
- Matrix run: multi‑task × multi‑trainer with aggregated summaries.
- LLM4AD: >=80% functional in Real mode; improvement subset achieved.
- VeriBench: >=80% functional in Real mode.

**M3**
- Gradio UI can launch runs and browse results.
- MLflow run ID/URI visible.
- TB logdir + open command visible.
- OpenTrace examples smoke tests pass (100%).

**M4**
- GEPA + Curriculum integrated (if inputs provided) and run on >=2 tasks in both modes.

## 11) Next Step
Pick Plan A or B, then finalize the M1 scope and acceptance set. Once approved, implementation proceeds in small PRs with CI‑verifiable outputs.
