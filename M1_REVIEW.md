# M1 Review Checklist (Runner + Standardized Outputs)

## Acceptance Commands (run in order)

```bash
# List tasks (LLM4AD + examples)
trace-bench list-tasks --root LLM4AD/benchmark_tasks

# Validate smoke config
trace-bench validate --config configs/smoke.yaml

# Stub smoke (deterministic, no keys)
trace-bench run --config configs/smoke.yaml --runs-dir runs

# Real smoke (requires OPENAI_API_KEY)
trace-bench run --config configs/smoke_real.yaml --runs-dir runs

# Tests (disable external plugin autoload)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Expected Artifacts

```
runs/<run_id>/
  config.snapshot.yaml
  env.json              # allowlist + redaction
  results.csv
  events.jsonl
  summary.json
  tb/
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
TRACE_BENCH_STRICT_EXAMPLES=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Pending Inputs (Explicit)

- **VeriBench entrypoint/task list** from Trace team (CLI supports `--bench veribench`, but currently raises `NotImplementedError`).
- **Telemetry/MLflow integration module** for later milestones.
