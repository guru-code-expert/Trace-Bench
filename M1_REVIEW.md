# M1 Review Checklist (Runner + Canonical Artifacts)

## Acceptance Commands (run in order)

```bash
# List tasks (all benches) + trainers
trace-bench list-tasks --root LLM4AD/benchmark_tasks
trace-bench list-trainers

# Validate M1 config (strict)
trace-bench validate --config configs/m1_validation.yaml --strict

# Run M1 validation (stub by default)
trace-bench run --config configs/m1_validation.yaml --runs-dir runs

# Bounded 2x2 matrix smoke (4 jobs)
trace-bench run --config configs/m1_matrix_smoke.yaml --runs-dir runs

# Real smoke (requires OPENAI_API_KEY)
trace-bench run --config configs/smoke_real.yaml --runs-dir runs

# Tests (disable external plugin autoload)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Notebook

- `notebooks/01_m1_minimal_api.ipynb` (Open in Colab badge inside)

## Expected Artifacts

```
runs/<run_id>/
  meta/
    config.snapshot.yaml
    env.json              # allowlist + redaction
    git.json
    manifest.json
  jobs/<job_id>/
    job_meta.json
    results.json
    events.jsonl
    artifacts/
    tb/
  results.csv
  summary.json
```

## Dependencies (Required for Full Pass)

System:
- Graphviz (system package)

Python:
- `graphviz`, `pyyaml`, `pytest`, `numpy`, `matplotlib`, `litellm==1.75.0`

OpenTrace examples strict smoke (for 100% pass):
- `datasets`, `textgrad`, `dspy`, `autogen`, `python-dotenv`

## OpenTrace Examples Smoke (100% Pass)

Run strict smoke once optional deps are installed:
```bash
TRACE_BENCH_STRICT_EXAMPLES=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/m1/test_opentrace_examples_smoke.py
```

## Pending Inputs (Explicit)

- **VeriBench entrypoint/task list** from Trace team (CLI supports `--bench veribench`, but currently raises `NotImplementedError`).
- **Telemetry/MLflow integration module** for later milestones.
