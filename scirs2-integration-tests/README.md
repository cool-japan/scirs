# SciRS2 Integration Tests

[![Stable](https://img.shields.io/badge/status-stable-brightgreen)]()
[![Version](https://img.shields.io/badge/version-0.6.3-green)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)]()

Cross-crate integration test suite for the SciRS2 ecosystem, v0.6.3.

## Purpose

This crate verifies that SciRS2 sub-crates compose correctly at their boundaries. Unit tests within each sub-crate check isolated functionality; the tests here check data flow, API compatibility, type interoperability, and end-to-end numerical correctness across module boundaries.

All tests run with `--all-features` so that every optional sub-crate is exercised together.

---

## Integration Scenarios Tested (v0.6.3)

### autograd + neural

File: `tests/integration/neural_optimize.rs`

- Backward pass through a `Sequential` model using `scirs2-autograd` tape recording
- Gradient checks: numerical finite-difference vs. autograd derivatives agree to `1e-5`
- Training loop convergence on XOR and linear regression fixtures
- Second-order derivatives (Hessian-vector products) through dense layers

### linalg + sparse

File: `tests/integration/sparse_linalg.rs`

- Round-trip: build a sparse CSR matrix in `scirs2-sparse`, convert to dense `ndarray` array, and verify identical results from `scirs2-linalg` dense eigensolvers
- Sparse Cholesky from `scirs2-sparse` vs. dense Cholesky from `scirs2-linalg` on the same positive-definite matrix
- GMRES convergence with `scirs2-sparse` preconditioners applied to systems assembled with `scirs2-linalg` matrix functions

### stats + optimize

File: `tests/integration/stats_datasets.rs`

- Maximum-likelihood parameter estimation: optimize log-likelihood constructed from `scirs2-stats` distributions using `scirs2-optimize` L-BFGS-B
- Bayesian credible-interval workflow: NUTS sampler from `scirs2-stats` + posterior summarization
- Bootstrap confidence intervals via `scirs2-stats` resampling, compared against analytical results

### signal + fft

File: `tests/integration/fft_signal.rs`

- Butterworth filter design (`scirs2-signal`) → apply to synthetic sinusoid → `rfft` (`scirs2-fft`) → verify spectral peaks match expected frequencies
- STFT via `scirs2-signal` vs. manual windowed-FFT via `scirs2-fft`: bin-for-bin agreement
- Inverse FFT round-trip: signal → FFT → IFFT → recovered signal within tolerance

### vision + ndimage

File: `tests/integration/ndimage_vision.rs`

- Gaussian blur (`scirs2-ndimage`) → Harris corner detection (`scirs2-vision`): corner count stable to denoising
- Morphological opening (`scirs2-ndimage`) → contour extraction (`scirs2-vision`): shape area preserved
- SIFT keypoints (`scirs2-vision`) matched after affine warp constructed using `scirs2-ndimage` geometric transforms

---

## How to Run

### Recommended (nextest, parallel, fast feedback)

```bash
cargo nextest run --all-features -p scirs2-integration-tests
```

### Standard cargo test

```bash
cargo test --all-features --package scirs2-integration-tests
```

### Single scenario

```bash
# autograd + neural only
cargo nextest run --all-features -p scirs2-integration-tests neural

# linalg + sparse only
cargo nextest run --all-features -p scirs2-integration-tests sparse_linalg

# signal + fft only
cargo nextest run --all-features -p scirs2-integration-tests fft_signal
```

### With verbose output

```bash
cargo nextest run --all-features -p scirs2-integration-tests --no-capture
```

### Performance / ignored tests

```bash
cargo test --all-features --package scirs2-integration-tests -- --ignored
```

---

## Test Structure

```
scirs2-integration-tests/
├── Cargo.toml
└── tests/
    └── integration/
        ├── mod.rs                    # Module root
        ├── neural_optimize.rs        # autograd ↔ neural
        ├── sparse_linalg.rs          # linalg ↔ sparse
        ├── stats_datasets.rs         # stats ↔ optimize ↔ datasets
        ├── fft_signal.rs             # signal ↔ fft
        ├── ndimage_vision.rs         # vision ↔ ndimage
        ├── ml_pipeline.rs            # datasets → neural → optimize → metrics
        ├── vision_pipeline.rs        # io → ndimage → vision → metrics
        ├── graph_pipeline.rs         # graph → neural → metrics
        ├── scientific_pipeline.rs    # integrate → linalg → sparse
        ├── signal_pipeline.rs        # io → signal → stats → series
        ├── nlp_pipeline.rs           # text → neural → metrics
        ├── numerical_crosscrate.rs   # cross-crate numerical consistency (16 tests)
        ├── numerical_validation.rs   # numerical accuracy vs. known reference values (40 tests)
        ├── performance.rs            # Cross-crate perf benchmarks (ignored by default)
        ├── common/                   # Shared helpers (array builders, timing)
        └── fixtures/                 # Test data generators
```

---

## Adding New Tests

1. Choose the test file matching the module boundary being tested (or create a new file if the scenario crosses more than two modules).
2. Follow the no-`unwrap()` policy: use `?` or `expect("descriptive message")`.
3. Use `std::env::temp_dir()` for any temporary file I/O inside tests.
4. Mark slow tests with `#[ignore]` and document the expected runtime.
5. Use `approx::assert_relative_eq!` for floating-point comparisons — avoid raw `==`.

---

## Current Status (v0.6.3, verified 2026-07-27)

- Untouched by this release (no integration-tests-specific changes shipped in 0.6.3); the status below (verified 2026-07-22 as part of the 0.6.2 cycle) remains accurate since the crate is unchanged
- All five baseline integration scenarios above are implemented and passing
- v0.4.2 Wave 42/44 added 6 additional pipeline tests: ML pipeline, signal analysis, computer vision, graph ML, scientific computing, NLP — all green
- v0.4.2 Wave 44 added 16 numerical cross-crate consistency tests and 40 numerical validation tests
- 251 `#[test]` functions across the suite (246 run + 5 intentionally `#[ignore]`d performance/opt-in benchmarks); 0 doctests; 0 `todo!()`/`unimplemented!()`
- 246/246 pass via `cargo nextest run --all-features -p scirs2-integration-tests` (freshly re-verified 2026-07-22, as part of the 0.6.2 workspace-wide all-features run)
- Performance tests in `performance.rs` (plus a few long-running perf checks in other pipeline files) remain `#[ignore]`d by default and require explicit `--ignored` opt-in

---

## License

Apache-2.0

## Authors

COOLJAPAN OU (Team Kitasan)
