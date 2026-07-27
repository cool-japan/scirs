# scirs2-stability-tests

[![Stable](https://img.shields.io/badge/status-stable-brightgreen)]()
[![Version](https://img.shields.io/badge/version-0.6.3-green)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)]()

**API stability tests, compile-fail tests, and ML property-based validation for the SciRS2 scientific computing library.**

This crate is a test-only workspace member (`publish = false`) that provides shared infrastructure for validating the public API surface and numerical properties across all SciRS2 crates.

## What This Crate Provides

### Compile-Fail Tests (`tests/compile_fail/`)

Trybuild-based tests that assert type errors and API misuse produce the expected compiler diagnostics. These act as regression tests for the public API — if a change accidentally makes invalid code compile, CI catches it immediately.

### ML Property-Based Test Utilities (`src/ml_properties.rs`)

Reusable property checkers for machine-learning algorithms:
- Convergence assertions for iterative solvers
- Invariance checks (permutation, scaling, translation)
- Idempotency and reproducibility validators
- Gradient / Jacobian finite-difference verifiers

### Deterministic Data Generators (`src/data_generators.rs`)

Synthetic dataset factories for controlled, reproducible test scenarios:
- Gaussian blobs and linearly separable datasets
- Regression surfaces with configurable noise
- Time-series with trend, seasonality, and outliers
- Sparse graph adjacency structures

### Numerical Benchmarks (`benches/numerical.rs`)

Criterion-based benchmarks tracking numerical accuracy and throughput regressions across SciRS2 algorithms.

### Additional Test Suites (`tests/`)

| File                    | Purpose                                          |
|-------------------------|--------------------------------------------------|
| `api_stability.rs`      | Compile-pass tests for stable public APIs        |
| `compile_fail_tests.rs` | Trybuild runner for compile-fail assertions      |
| `ml_property_tests.rs`  | Property-based correctness tests for ML algos   |
| `panic_resistance.rs`   | Robustness tests: NaN, Inf, empty, huge inputs   |
| `accuracy_regression.rs`| Numerical accuracy regression suite             |

## Running the Tests

```bash
# Run all stability tests
cargo test -p scirs2-stability-tests --all-features

# Run only compile-fail tests
cargo test -p scirs2-stability-tests --test compile_fail_tests

# Run numerical benchmarks
cargo bench -p scirs2-stability-tests
```

## Dependencies

This crate depends on the following SciRS2 crates under test:

- `scirs2-core` — core types and utilities
- `scirs2-linalg` — linear algebra
- `scirs2-stats` — statistical functions
- `scirs2-io` — I/O utilities
- `scirs2-text` — NLP / tokenization

All dependencies are path-based (workspace-local) and no version is published to crates.io.

## Documentation

Full SciRS2 documentation: <https://github.com/cool-japan/scirs>

## License

Licensed under [Apache-2.0](LICENSE).

Copyright © COOLJAPAN OU (Team Kitasan)
