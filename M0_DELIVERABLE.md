# M0 — Trace‑Bench Realignment (Minimal Runner + Examples + MLflow/Gradio Stubs)

**Executive Summary**
This M0 delivers a lean, reviewable runner/CLI aligned to the UpWork spec in `Trace-Bench UpWork specifications.md`. The focus is on a minimal, spec‑aligned artifact schema, proper Trace examples as pytest targets, and stubbed MLflow/Gradio UI. The code surface is intentionally small and reuses existing LLM4AD task structure and OpenTrace `experimental` APIs.

**Spec Alignment**
Primary source: `Trace-Bench UpWork specifications.md` (sections 1, 2, 5, 6, 8). This M0 maps to Milestone 0 and the first part of Milestone 1.

## 1) Scope (M0 Only)

**Included**
- Minimal runner + CLI with standardized outputs.
- Proper pytest suite (unit + smoke integration).
- Two Trace example tasks using `trace.node` and `@trace.bundle`.
- MLflow + Gradio stubs (no hard deps; documented behavior).
- Updated README with CLI usage and artifact schema.

**Excluded**
- VeriBench integration.
- GEPA/Curriculum migration (Milestone 4 in spec).
- Full UI/telemetry integration (stub only).

## 2) Repository Grounding

- LLM4AD tasks: `LLM4AD/benchmark_tasks/*/__init__.py` with `build_trace_problem()`.
- LLM4AD loader: `LLM4AD/llm4ad_loader.py`.
- OpenTrace `experimental` provides `opto.features` and `opto.trainer.train`.

Target compatibility: `OpenTrace@experimental` (commit `c0a0282b45a8ab0ec4f927611dd1847f4e15519f`).

## 3) Architecture (Minimal Surface)

**Files**
- `trace_bench/config.py` — RunConfig parsing/validation (YAML/JSON).
- `trace_bench/registry.py` — Task discovery + loaders (LLM4AD + examples).
- `trace_bench/runner.py` — BenchRunner orchestration.
- `trace_bench/artifacts.py` — Spec‑aligned run outputs.
- `trace_bench/cli.py` — `trace-bench` CLI + module entry.
- `trace_bench/ui.py` — Gradio/MLflow stubs.
- `trace_bench/examples/*` — example tasks with proper nodes/bundles.

**Public API**
```python
from trace_bench.config import RunConfig, load_config
from trace_bench.runner import BenchRunner

cfg = load_config("configs/smoke.yaml")
runner = BenchRunner(cfg)
summary = runner.run()
```

## 4) CLI (Spec‑aligned)

Commands required by spec:
```
trace-bench list-tasks --root LLM4AD/benchmark_tasks
trace-bench validate --config configs/smoke.yaml
trace-bench run --config configs/smoke.yaml
trace-bench ui --runs-dir runs
```

Module entrypoint also supported:
```
python -m trace_bench run --config configs/smoke.yaml
```

## 5) Standardized Run Artifacts (Spec)

Run directory layout:
```
runs/<run_id>/
  config.snapshot.yaml
  env.json
  results.csv
  events.jsonl
  summary.json
```

- `config.snapshot.yaml` uses YAML if available; falls back to JSON (still readable).
- `env.json` includes environment variables + git commit/branch if available.

## 6) Example Tasks (Proper Trace Nodes)

**Included**
- `example:greeting_stub` — uses `trace.node` + `@trace.bundle`.
- `example:train_single_node_stub` — uses `trace.node` + `@trace.bundle`.

These are registered in `trace_bench.registry` and discoverable via `trace-bench list-tasks`.

## 7) Tests (Pytest)

**Unit**
- Config parsing and defaults.

**Integration**
- Smoke run on `circle_packing` with stub mode.
- Example tasks load and return a valid bundle.

Run:
```
pytest -q
```

## 8) MLflow + Gradio (Stub)

- `trace_bench/ui.py` attempts to import `gradio`; prints install guidance if missing.
- MLflow presence is detected and displayed; no hard dependency.
- UI shows runs + config snapshot + results CSV.

## 9) Review Load Mitigation

- Total new code kept small (single runner path, no refactors).
- Three small PRs recommended (<300 LOC each).
- Standardized artifacts make verification deterministic.

## 10) Acceptance Criteria (M0)

- `trace-bench run --config configs/smoke.yaml` produces spec‑aligned run folder.
- `trace-bench list-tasks` shows LLM4AD + example stubs.
- `pytest -q` passes.
- Notebook `notebooks/01_smoke_runner.ipynb` runs in stub mode.

## 11) Next Step
If accepted, proceed to M1 (runner hardening + matrix runs + docs), followed by MLflow/Gradio integration and GEPA/Curriculum in M3/M4 per spec.
