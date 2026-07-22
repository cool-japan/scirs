# scirs2-integration-tests TODO

## Status: v0.6.2 (2026-07-22)

## Purpose

Cross-crate integration tests for SciRS2 ecosystem.

## v0.3.3 Coverage

- autograd + neural integration
- linalg + sparse interop
- stats + optimize integration
- signal + fft pipeline
- vision + ndimage pipeline

## v0.4.0 Planned Tests

- [x] End-to-end ML pipeline (datasets -> neural -> optimize -> metrics) — implemented in v0.4.2 (`tests/integration/ml_pipeline.rs`)
- [x] Full signal analysis pipeline (io -> signal -> stats -> series) — implemented in v0.4.2 (`tests/integration/`)
- [x] Computer vision pipeline (io -> ndimage -> vision -> metrics) — implemented in v0.4.2 (`tests/integration/vision_pipeline.rs`)
- [x] Graph ML pipeline (graph -> neural -> metrics) — implemented in v0.4.2 (`tests/integration/`)
- [x] Scientific computing pipeline (integrate -> linalg -> sparse) — implemented in v0.4.2 (`tests/integration/`)
- [x] NLP pipeline (text -> neural -> metrics) — implemented in v0.4.2 (`tests/integration/`)

## v0.4.2 Wave 44 Additions

- [x] Compile-fail tests and cross-crate consistency tests — implemented in `tests/integration/numerical_crosscrate.rs` (16 tests: FFT convolution theorem, Parseval, round-trip, RFFT length, normal equations, rank-1 eigenvalue, solve known system, PCA variance ordering, SVD ordering, dense matvec, SPD eigenvalues, trace identity, circulant eigenvalues via FFT, and three error-handling tests)
- [x] Numerical validation test suite (statistical accuracy vs known reference values) — implemented in `tests/integration/numerical_validation.rs` (40 tests: special functions gamma/erf/j0/beta, Normal/Poisson/ChiSquare distributions, linalg solve/eigh/SVD, FFT Parseval/roundtrip/linearity, signal energy/spectrum, descriptive statistics mean/median/variance)

## Running Tests

cargo nextest run --all-features -p scirs2-integration-tests

## v0.6.2 Status (verified 2026-07-22)

- 251 `#[test]` functions across baseline scenarios + Wave 42/44 pipelines; 0 doctests; 0 `todo!()`/`unimplemented!()`
- Freshly re-run via `cargo nextest run --all-features -p scirs2-integration-tests` as part of the 0.6.2 workspace-wide all-features run: 246 passed, 5 skipped (0 failures). The 5 skips are all `#[ignore]`-by-design opt-in benchmarks (`neural_optimize::test_distributed_training_integration`, `performance::comprehensive_performance_benchmark`, `performance::test_cache_efficiency`, `performance::test_image_processing_pipeline_performance`, `performance::test_performance_scaling`) — not failures
- Pipelines covered: ML (`ml_pipeline.rs`), signal (`signal_pipeline.rs`), computer vision (`vision_pipeline.rs`), graph ML (`graph_pipeline.rs`), scientific computing (`scientific_pipeline.rs`), NLP (`nlp_pipeline.rs`)
- Cross-crate numerical: `numerical_crosscrate.rs` (16 tests), `numerical_validation.rs` (40 tests)
- Workspace version 0.6.2 confirmed; all dependencies path-based via `workspace = true`
