# M0 — Trace‑Bench Harness (Technical Plan & Architecture)

**Executive Summary**
This M0 deliverable defines a minimal, low‑risk plan to productize Trace‑Bench into a reproducible benchmark harness. It inventories existing task and trainer assets, introduces a small `trace_bench` runner package with config‑driven execution, and specifies a deterministic StubLLM mode plus a LiteLLM‑backed real mode. It also standardizes run artifacts, adds a smoke config, tests, and a notebook so the Trace team can validate the harness quickly and consistently.

**Status Note**
OpenTrace `experimental` is required for full algorithm support (`opto.features.*`, `opto.trainer.loggers.TensorboardLogger`). It is not present locally. The plan below assumes `experimental` will be fetched and checked out before full runs.

## 1) Repository Inventory (Grounded Snapshot)

**Trace‑Bench top‑level**
- `Trace-Bench/LLM4AD/benchmark_tasks/*/__init__.py` — Self‑contained LLM4AD task modules with `build_trace_problem()`.
- `Trace-Bench/LLM4AD/llm4ad_loader.py` — Shared loader and `build_trace_problem_from_config()` implementation.
- `Trace-Bench/LLM4AD/trainers_benchmark.py` — CLI script that runs PrioritySearch/GEPA and writes CSV + TensorBoard logs.
- `Trace-Bench/LLM4AD/trainers_benchmark_tasks_validation.py` — Task loading and minimal optimization validation.
- `Trace-Bench/tests/test_lite_optimize_llm4ad.py` — Uses OptoPrime to optimize a couple of tasks.
- `Trace-Bench/setup.py` — Installs Trace‑Bench package (currently expects `opto` in env).

**OpenTrace (main branch locally)**
- `OpenTrace/opto/utils/llm.py` — `LLM` wrapper for LiteLLM/AutoGen backends.
- `OpenTrace/opto/trainer/*` — Base trainer algorithms, loaders, utils.
- `OpenTrace/opto/optimizers/*` — Optimizers used by algorithms.
- Missing in `main` (expected in `experimental`): `opto.features.*`, `opto.trainer.loggers.TensorboardLogger`, `opto.trainer.train` helper.

**Relevant task examples**
- `Trace-Bench/LLM4AD/benchmark_tasks/circle_packing/__init__.py` — Numpy‑only, fast, good smoke candidate.
- `Trace-Bench/LLM4AD/benchmark_tasks/online_bin_packing_local/__init__.py` — Numpy‑only, fast, good smoke candidate.
- `Trace-Bench/LLM4AD/benchmark_tasks/optimization_tsp_construct/__init__.py` — Numpy‑only, fast, good smoke candidate.

## 2) Gaps And Minimal Fixes

- **OpenTrace experimental not local**: Add setup step to fetch and checkout `experimental`.
- **Missing `opto.features` and `TensorboardLogger` in main**: Prefer `experimental`. Provide fallback in runner for “dry‑run” metrics when unavailable.
- **No unified runner/config**: Add a small `trace_bench` package with `RunConfig`, `Runner`, CLI, and artifact writer.
- **`opto.trainer.train` not present in main**: Add best‑effort wrapper in `trace_bench.trainers` that calls it if available, else falls back to dry‑run metrics.

## 3) Runner Architecture (Modules, Responsibilities, API)

**Module layout**
- `trace_bench/config.py` — Parse YAML/JSON config and validate defaults.
- `trace_bench/tasks.py` — Task registry, dynamic import, normalize `build_trace_problem()` bundle.
- `trace_bench/runner.py` — Orchestrate tasks/trainers, select LLM adapter, write artifacts.
- `trace_bench/trainers/base.py` — `Trainer` interface definition.
- `trace_bench/trainers/opto_algos.py` — Adapters for PrioritySearch/GEPA (best‑effort, dry‑run fallback).
- `trace_bench/adapters/llm.py` — `LLMAdapter`, `StubLLMAdapter`, `ProviderLLMAdapter`.
- `trace_bench/artifacts.py` — Run directory creation + metrics/metadata writers.
- `trace_bench/cli.py` — CLI entrypoints `run`, `list-tasks`, `validate`.
- `trace_bench/integrations/mlflow.py` — Stub for MLflow link writing.
- `trace_bench/integrations/tensorboard.py` — Stub for TensorBoard dir setup.

**Responsibilities and public APIs (one‑liners)**
- `RunConfig` loads `configs/*.yaml` and normalizes defaults.
- `load_trace_problem(task_key, tasks_dir)` returns `TraceProblem`.
- `Runner.run_matrix()` executes all tasks × trainers and logs artifacts.
- `LLMAdapter.generate(...)` normalizes LLM responses across stub/real.

## 4) API Signatures / Stubs (Python)

```python
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class RunConfig:
    run_id: str | None
    runs_root: str
    seed: int
    mode: str
    tasks: list[str]
    trainers: list[str]
    llm: dict
    timeouts: dict
    threads: int
    artifact: dict
    mlflow: dict
    tensorboard: dict

class Runner:
    def __init__(self, config: RunConfig, llm: "LLMAdapter" | None = None): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run_task(self, task_key: str, trainer_key: str) -> dict: ...
    def run_matrix(self) -> list[dict]: ...

class Trainer:
    def train_step(self, problem: "TraceProblem", **kwargs) -> dict: ...
    def eval_step(self, problem: "TraceProblem", **kwargs) -> dict: ...
    def save_checkpoint(self, path: str) -> None: ...

class LLMAdapter:
    def generate(self, messages: list[dict], **kwargs) -> str: ...
    def healthcheck(self) -> dict: ...

class StubLLMAdapter(LLMAdapter): ...
class ProviderLLMAdapter(LLMAdapter): ...

# Task interface expected from LLM4AD tasks
# build_trace_problem(**override_eval_kwargs) -> dict
```

## 5) Run Artifact Schema (Exact Layout + Examples)

```
runs/
  <timestamp>-<uuid>/
    config.yaml
    run_metadata.json
    mlflow_link.txt
    tensorboard_dir/
    artifacts/
      metrics.csv
      summary.jsonl
      raw_traces/
    stdout.log
    stderr.log
    env.txt
```

**Example `run_metadata.json`**
```json
{
  "run_id": "4c9cfe8a-bb0b-4d61-9d8a-4fb4e0c7c1a8",
  "created_at": "2026-02-04T12:00:00Z",
  "mode": "stub",
  "tasks": ["circle_packing"],
  "trainers": ["PrioritySearch"],
  "llm": {"provider": "stub", "seed": 123}
}
```

**Example `metrics.csv` header**
```
timestamp,task,trainer,status,score,elapsed
```

## 6) RunConfig YAML Example (`configs/smoke.yaml`)

```yaml
mode: stub
runs_root: runs
seed: 123
tasks:
  - circle_packing
trainers:
  - PrioritySearch
llm:
  provider: stub
  seed: 123
timeouts:
  task_seconds: 60
  eval_seconds: 20
threads: 1
```

Note: The repo `configs/smoke.yaml` uses JSON syntax (valid YAML) to keep parsing dependency‑free.

## 7) StubLLM & Real LLM Modes

**StubLLM**
- Deterministic RNG with seed + message hash.
- Fixed string output format `STUB_RESPONSE_<token>`.
- No network calls.

**Real LLM (LiteLLM via `opto.utils.llm.LLM`)**
- Uses env vars:
  - `TRACE_DEFAULT_LLM_BACKEND=LiteLLM`
  - `TRACE_LITELLM_MODEL=gpt-4o-mini`
  - `OPENAI_API_KEY=<key>`
  - `ANTHROPIC_API_KEY=<key>`
  - `AZURE_*` (if Azure via LiteLLM)

**Acceptance commands**
```
TRACE_MODE=stub python -m trace_bench.cli run --config configs/smoke.yaml
TRACE_MODE=real TRACE_DEFAULT_LLM_BACKEND=LiteLLM TRACE_LITELLM_MODEL=gpt-4o-mini OPENAI_API_KEY=... python -m trace_bench.cli run --config configs/smoke.yaml
```

## 8) Test Plan

**Pytest matrix**
- `tests/m0/test_config.py` — YAML/JSON config parsing.
- `tests/m0/test_stub_llm.py` — Stub determinism.
- `tests/m0/test_runner_smoke.py` — Single task run + artifact existence.

**Command**
```
pytest tests/m0 -q
```

**Example skeleton (already included in repo)**
```python
# tests/m0/test_stub_llm.py
from trace_bench.adapters.llm import StubLLMAdapter

def test_stub_llm_deterministic():
    messages = [{"role": "user", "content": "Hello"}]
    assert StubLLMAdapter(123).generate(messages) == StubLLMAdapter(123).generate(messages)
```

## 9) Notebook Plan

**Notebook:** `notebooks/01_smoke_runner.ipynb`

**Cell 1 (Install deps)**
```python
!pip install -q pyyaml
!pip install -q -e /content/OpenTrace
!pip install -q -e /content/Trace-Bench
```

**Cell 2 (Stub mode run)**
```python
import os
os.environ["TRACE_MODE"] = "stub"
!python -m trace_bench.cli run --config configs/smoke.yaml
```

**Cell 3 (Real mode)**
```python
import os
os.environ["TRACE_MODE"] = "real"
os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
os.environ["TRACE_LITELLM_MODEL"] = "gpt-4o-mini"
# os.environ["OPENAI_API_KEY"] = "<set in Colab secrets>"
# !python -m trace_bench.cli run --config configs/smoke.yaml
```

**Cell 4 (Inspect artifacts)**
```python
!ls -la runs
# TensorBoard: tensorboard --logdir runs/<run_id>/tensorboard_dir
# MLflow: open the link in runs/<run_id>/mlflow_link.txt
```

**Colab badge markdown**
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<ORG>/<REPO>/blob/main/notebooks/01_smoke_runner.ipynb)
```

## 10) Smoke Tasks & Acceptance Commands

**Smoke task set**
- `circle_packing` — Fast, numpy‑only, deterministic.
- `online_bin_packing_local` — Fast, numpy‑only.
- `optimization_tsp_construct` — Fast, numpy‑only.

**List tasks**
```
python -m trace_bench.cli list-tasks --tasks LLM4AD/benchmark_tasks
```

**Run smoke (stub)**
```
TRACE_MODE=stub python -m trace_bench.cli run --config configs/smoke.yaml
```

**Run smoke (real)**
```
TRACE_MODE=real TRACE_DEFAULT_LLM_BACKEND=LiteLLM TRACE_LITELLM_MODEL=gpt-4o-mini OPENAI_API_KEY=... python -m trace_bench.cli run --config configs/smoke.yaml
```

## 11) Deliverables & PR Plan

**Files to include in first PR**
- `trace_bench/__init__.py`
- `trace_bench/config.py`
- `trace_bench/tasks.py`
- `trace_bench/runner.py`
- `trace_bench/trainers/base.py`
- `trace_bench/trainers/opto_algos.py`
- `trace_bench/adapters/llm.py`
- `trace_bench/artifacts.py`
- `trace_bench/cli.py`
- `trace_bench/integrations/mlflow.py`
- `trace_bench/integrations/tensorboard.py`
- `configs/smoke.yaml`
- `tests/m0/test_config.py`
- `tests/m0/test_stub_llm.py`
- `tests/m0/test_runner_smoke.py`
- `notebooks/01_smoke_runner.ipynb`
- `M0_DELIVERABLE.md`

**Branch name**
- `m0/runner-foundation`

**Commit templates**
- `m0: add runner skeleton and configs`
- `m0: add stub llm and tests`
- `m0: add notebook and doc`

**Git commands**
```
git checkout -b m0/runner-foundation
git add trace_bench configs tests notebooks M0_DELIVERABLE.md
git commit -m "m0: add runner skeleton and configs"
git commit -m "m0: add stub llm and tests"
git commit -m "m0: add notebook and doc"
git push -u origin m0/runner-foundation
```

## 12) Time Estimate & Risks

**Estimate**
- 2–3 days to deliver M0 skeleton + tests + notebook.

**Key risks**
- OpenTrace `experimental` branch not present locally.
- `opto.features` and `TensorboardLogger` API differences between branches.
- Task modules with heavy imports or timeouts.

**Mitigations**
- Add a `dry_run` fallback in trainers when algorithms are unavailable.
- Keep smoke tasks numpy‑only.
- Validate tasks via `trace_bench.cli validate` before long runs.

## 13) Setup For Experimental Branch

```
cd ../OpenTrace
git fetch origin experimental
git checkout experimental
```

**Recommended next step**
If approved, I will open PR `m0/runner-foundation` implementing the skeleton and tests in ~2–3 days.
