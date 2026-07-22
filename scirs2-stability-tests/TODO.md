# scirs2-stability-tests TODO

## Status: v0.6.2 (2026-07-22)

## Purpose

Test-only workspace member (`publish = false`) providing shared infrastructure to validate the public API surface and numerical properties across all SciRS2 crates.

## Completed

### Compile-Fail Tests (`tests/compile_fail/`)
- [x] Trybuild-based regression tests asserting type errors and API misuse produce expected diagnostics
- [x] `compile_fail_tests.rs` runner

### API Stability Tests (`tests/api_stability.rs`)
- [x] Compile-pass tests for stable public APIs across `scirs2-core`, `scirs2-linalg`, `scirs2-stats`, `scirs2-io`, `scirs2-text`

### ML Property-Based Test Utilities (`src/ml_properties.rs`)
- [x] Convergence assertions for iterative solvers
- [x] Invariance checks (permutation, scaling, translation)
- [x] Idempotency / reproducibility validators
- [x] Gradient / Jacobian finite-difference verifiers

### Deterministic Data Generators (`src/data_generators.rs`)
- [x] Gaussian blobs and linearly separable datasets
- [x] Regression surfaces with configurable noise
- [x] Time-series with trend, seasonality, and outliers
- [x] Sparse graph adjacency structures

### Robustness / Accuracy Suites
- [x] `panic_resistance.rs` — NaN, Inf, empty, huge inputs (Wave 33: 45 stability tests via cargo-fuzz integration)
- [x] `accuracy_regression.rs` — numerical accuracy regression suite
- [x] `ml_property_tests.rs` — property-based correctness tests for ML algorithms

### Fuzz Targets (Wave 33, integrated)
- [x] 8 cargo-fuzz targets across linalg/stats/fft/signal entry points

### Numerical Benchmarks (`benches/numerical.rs`)
- [x] Criterion-based benchmarks tracking accuracy and throughput regressions

## v0.6.1 Quality Gate (verified 2026-07-15)

- 132 `#[test]` functions across `src/` and `tests/` (all passing via `cargo nextest run -p scirs2-stability-tests --all-features`); 0 doctests
- `todo!()`/`unimplemented!()` count: 0 real occurrences (the sole string hit is inside the trybuild `core_error_non_exhaustive.stderr` fixture — expected compiler-diagnostic text, not executable code)
- cargo check + clippy: clean
- Trybuild compile-fail harness: previously-tracked timeout/skip issue (`compile_fail_harness` vs `compile_fail_tests` nextest override collision, see MEMORY.md 2026-06-24 entry) is resolved — `.config/nextest.toml` now unions both binaries (`binary(compile_fail_tests) | binary(compile_fail_doc)`), and `compile_fail_tests` passes cleanly (~194s) with 0 skips
- Path-based dependencies on `scirs2-core`, `scirs2-linalg`, `scirs2-stats`, `scirs2-io`, `scirs2-text`

## Notes

- Crate is `publish = false` and is not distributed via crates.io.
- Used as the regression gate for SciRS2 API stability — any change that accidentally makes invalid code compile (or breaks the documented API surface) is caught here.
- Benchmarks integrate with the workspace `bench-regression.sh` flow.
