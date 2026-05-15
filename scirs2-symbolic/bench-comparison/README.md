# FSReD Benchmark Comparison: scirs2-symbolic vs PySR

## Rust-side (automated, runs in CI / locally)

```bash
cargo bench --bench fsred_bench -p scirs2-symbolic
```

Results are written to `target/fsred_results.json` (relative to the crate
root).  Each entry contains:

```json
{
  "name": "I.6.2a",
  "mse": 1.234e-05,
  "r_squared": 0.9987,
  "recovered": false
}
```

`recovered` is `true` when the best-found formula achieves MSE < 1e-3.

## PySR side (manual, NOT in CI)

**Prerequisites:**

- Python 3.11+
- Julia 1.10+
- PySR: `pip install pysr`

**Run:**

```bash
python bench-comparison/run_pysr.py
```

Output is written to `bench-comparison/pysr_results.json`.

**Compare:**

```bash
python bench-comparison/compare.py target/fsred_results.json bench-comparison/pysr_results.json
```

## Why the PySR side is not in CI

Julia 1.10+ is required by PySR but is not available in the standard CI
environment (GitHub Actions).  Running Julia setup in every CI job would add
≥ 5 min download time and ≥ 2 GB storage.  The Rust-only baseline is
sufficient for regression testing; the comparison against PySR is a
manual benchmarking exercise.
