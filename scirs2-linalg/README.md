# scirs2-linalg

[![crates.io](https://img.shields.io/crates/v/scirs2-linalg)](https://crates.io/crates/scirs2-linalg)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-linalg)](https://docs.rs/scirs2-linalg)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)]()

**High-performance linear algebra for Rust, modeled after SciPy/NumPy linalg.**

`scirs2-linalg` provides a comprehensive linear algebra library with SciPy-compatible APIs, pure-Rust BLAS/LAPACK via OxiBLAS (no C or Fortran dependencies), SIMD acceleration, randomized methods, tensor decompositions, and iterative solvers suitable for large-scale scientific computing and machine learning.

**Tests:** 2018/2018 passing (default features), 2248/2248 passing (`--all-features`) as of 2026-07-22 baseline; not independently re-run for the 0.6.5 docs update, plus 4 new unit tests in `eigen/general.rs` for the general eigenvalue engine below.

**Fixed in 0.6.5:** added a real, convergence-checked general (non-symmetric) eigenvalue/Schur decomposition engine (`eigen/general.rs`: Householder→upper-Hessenberg reduction, then implicit double-shift Francis QR with deflation, per Golub & Van Loan), now shared by `decomposition::schur`, `lapack::eig`, and `eigen::advanced_precision_eig`'s non-symmetric path. Each of those three previously carried its own broken implementation: a fixed count of *unshifted* QR iterations with no convergence check (empirically verified to leave sub-diagonal entries around `1e-4` and eigenvalues off by 1-3% on a companion matrix with closely-spaced eigenvalues), or a placeholder only correct for already-diagonal input. `lstsq`/`pinv` (`compat.rs`) were already real (both delegate to the crate's genuine SVD-based solvers), so this completes the eig/schur/lstsq trio end-to-end. See [CHANGELOG.md](../CHANGELOG.md) `[0.6.5]` for full detail.

## Installation

```toml
[dependencies]
scirs2-linalg = "0.6.5"
```

With optional acceleration:

```toml
[dependencies]
scirs2-linalg = { version = "0.6.5", features = ["simd", "parallel"] }
```

## Features (v0.6.5)

### Core Decompositions
- LU (with partial/rook/complete pivoting), QR, SVD, Cholesky, LDL^T
- Eigendecomposition: `eig`, `eigh` (symmetric), generalized eigenproblem
- Schur decomposition (real and complex), QZ decomposition
- Polar decomposition, complete orthogonal decomposition
- Tall-and-Skinny QR (TSQR), LQ decomposition for wide matrices
- Randomized SVD (Halko, Martinsson, Tropp), Nystrom approximation

### Iterative Solvers
- GMRES (restarted, deflated, recycled/GCRO-DR style)
- Preconditioned Conjugate Gradient (PCG)
- BiCGStab (stabilized bi-conjugate gradient)
- MINRES, SYMMLQ
- Arnoldi iteration, Lanczos iteration (with thick restarts)
- SOR, SSOR, Gauss-Seidel, Jacobi

### Matrix Functions
- Matrix exponential `expm` (Pade approximant + scaling/squaring)
- Matrix logarithm `logm` (inverse scaling/squaring)
- Matrix square root `sqrtm` (Schur-based)
- Matrix sign function `signm`
- Matrix trigonometric functions (sin, cos, tan, sinh, cosh via Schur)
- Polar decomposition
- Matrix polynomial evaluation

### Control Theory
- Algebraic Riccati equations (CARE, DARE): Newton iteration, Hamiltonian Schur
- Lyapunov equations (continuous and discrete)
- Sylvester equations (Bartels-Stewart, Hessenberg-Schur)
- Controllability and observability Gramians

### Tensor Operations
- CP decomposition (Canonical Polyadic via ALS)
- Tucker decomposition (Higher-Order SVD / HOOI)
- Tensor contractions and mode-n products
- Einstein summation (`einsum`)
- Hierarchical Tucker (HT) decomposition
- Tensor-train format basics

### Randomized Linear Algebra
- Randomized SVD with power iteration and oversampling
- Nystrom extension for kernel matrices
- Randomized eigensolvers (subspace iteration)
- Sketching: CountSketch, Gaussian sketch, SRHT

### Structured and Specialized Matrices
- Toeplitz, Hankel, Circulant (FFT-based O(n log n) matvec)
- Cauchy matrix, companion matrix
- Banded matrices (tridiagonal, pentadiagonal), block tridiagonal
- Block diagonal, block sparse row
- Indefinite systems (symmetric indefinite factorization)

### Matrix Completion and Low-Rank
- Nuclear norm minimization via alternating projections
- Soft-impute algorithm for matrix completion
- CUR decomposition (column-row factorization)
- Sparse-dense hybrid operations

### Numerical Analysis
- Perturbation theory: condition number bounds, backward error analysis
- Numerical range (field of values) computation
- Matrix pencil problems (regular and singular pencils)
- Error analysis for linear systems and least squares

### ML / AI Support
- Scaled dot-product attention, multi-head attention
- Flash attention (memory-efficient)
- Sparse attention patterns
- Positional encodings: RoPE, ALiBi
- Quantization-aware matrix multiply (4-bit, 8-bit, 16-bit)
- Mixed-precision operations with iterative refinement
- Batch matrix operations for mini-batch processing

## Usage Examples

### Basic operations

```rust
use scirs2_linalg::{det, inv, solve, svd, eigh};
use scirs2_core::ndarray::array;

let a = array![[4.0_f64, 2.0], [2.0, 3.0]];
let b = array![6.0_f64, 7.0];

let d = det(&a.view(), None)?;
let a_inv = inv(&a.view(), None)?;
let x = solve(&a.view(), &b.view(), None)?;

let (u, s, vt) = svd(&a.view(), true, None)?;
let (eigenvals, eigenvecs) = eigh(&a.view(), None)?;
```

### Iterative solvers

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::iterative::{bicgstab, conjugate_gradient, gmres};

// GMRES for a general non-symmetric system: args are (a, b, x0, tol, max_iter, restart)
let a = array![[3.0_f64, 1.0], [1.0, 4.0]];
let b = array![5.0_f64, 6.0];
let result = gmres(&a, &b.view(), None, 1e-12, 50, 10)?;

// Conjugate Gradient for symmetric positive definite (optionally preconditioned via
// the trailing `Option`): args are (a, b, x0, tol, max_iter, preconditioner)
let a_spd = array![[4.0_f64, 1.0], [1.0, 3.0]];
let b_spd = array![1.0_f64, 2.0];
let result = conjugate_gradient(&a_spd, &b_spd.view(), None, 1e-12, 100, None)?;

// BiCGStab for non-symmetric: args are (a, b, x0, tol, max_iter)
let result = bicgstab(&a, &b.view(), None, 1e-12, 100)?;
```

All three return an `IterativeSolveResult<F>` bundling the solution vector with `iterations`,
`residual_norm`, and a `converged` flag.

### Matrix functions

```rust
use scirs2_linalg::matrix_functions::{expm, logm, sqrtm};

let exp_a = expm(&a.view(), None)?;       // workers: Option<usize>
let log_a = logm(&a.view())?;
let sqrt_a = sqrtm(&a.view(), 20, 1e-10)?; // max_iter, tol (Denman-Beavers iteration)
```

### Tensor decompositions

```rust
use scirs2_linalg::tensor::core::Tensor;
use scirs2_linalg::tensor::{cp_als, hooi, CPConfig};

let data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect();
let tensor = Tensor::new(data, vec![2, 3, 4])?;

// CP decomposition with 3 components, up to 200 ALS iterations
let cfg = CPConfig { max_iter: 200, ..Default::default() };
let cp = cp_als(&tensor, 3, &cfg)?;

// Tucker decomposition (HOOI) targeting multilinear rank [2, 2, 3]
let tucker = hooi(&tensor, &[2, 2, 3], 100)?;
```

### Control theory

```rust
use scirs2_linalg::control::{care_solve, dare_solve, lyapunov_continuous};

// Continuous algebraic Riccati equation: A^T X + X A - X B R^{-1} B^T X + Q = 0
let p = care_solve(&a.view(), &b.view(), &q.view(), &r.view())?;

// Discrete algebraic Riccati equation
let p_d = dare_solve(&a.view(), &b.view(), &q.view(), &r.view())?;

// Lyapunov equation: A X + X A^T = -Q
let x = lyapunov_continuous(&a.view(), &q.view())?;
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `linalg` | OxiBLAS pure-Rust BLAS/LAPACK backend (on by default) |
| `simd` | SIMD-accelerated kernels (AVX/AVX2/AVX-512/NEON) (on by default) |
| `parallel` | Multi-threaded operations via Rayon |
| `gpu` | GPU abstraction layer (delegates to `scirs2-core/gpu`) |
| `cuda` | Pure-Rust direct `oxicuda-*` CUDA path (NVIDIA-only, experimental) |
| `opencl` / `rocm` / `metal` / `vulkan` | Backend-specific GPU acceleration (experimental; each requires `gpu`) |
| `autograd` | Cross-crate integration with `scirs2-autograd` (n×n factorization backward passes: Cholesky/LU/QR/sqrtm/logm) |
| `symbolic` | Cross-crate integration with `scirs2-symbolic` (`det_symbolic`, `eigenvalues_symbolic_2x2`, `condition_number_symbolic`) |
| `python` | Python bindings (delegates to `scirs2-core/python`) |

`serde` is compiled in unconditionally as a dependency (not an optional Cargo feature of this crate).
Default features: `linalg` + `simd`.

## Links

- [API Documentation](https://docs.rs/scirs2-linalg)
- [SciRS2 Repository](https://github.com/cool-japan/scirs)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
