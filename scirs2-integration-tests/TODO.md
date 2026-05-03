# scirs2-integration-tests TODO

## Status: v0.4.3 Released (2026-05-03)

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

## v0.4.3 Status (2026-05-03)

- ~251 `#[test]` functions across baseline scenarios + Wave 42/44 pipelines
- All tests passing on cargo check + clippy clean workspace
- Pipelines covered: ML (`ml_pipeline.rs`), signal, computer vision (`vision_pipeline.rs`), graph ML, scientific computing, NLP
- Cross-crate numerical: `numerical_crosscrate.rs` (16 tests), `numerical_validation.rs` (40 tests)
- Workspace version 0.4.3 verified; all dependencies path-based via `workspace = true`
