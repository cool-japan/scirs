# scirs2-linalg

[![crates.io](https://img.shields.io/crates/v/scirs2-linalg)](https://crates.io/crates/scirs2-linalg)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-linalg)](https://docs.rs/scirs2-linalg)

**High-performance linear algebra for Rust, modeled after SciPy/NumPy linalg.**

`scirs2-linalg` provides a comprehensive linear algebra library with SciPy-compatible APIs, pure-Rust BLAS/LAPACK via OxiBLAS (no C or Fortran dependencies), SIMD acceleration, randomized methods, tensor decompositions, and iterative solvers suitable for large-scale scientific computing and machine learning.

## Installation

```toml
[dependencies]
scirs2-linalg = "0.4.4"
```

With optional acceleration:

```toml
[dependencies]
scirs2-linalg = { version = "0.4.4", features = ["simd", "parallel"] }
```

## Features (v0.4.4)

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
use scirs2_linalg::iterative::{gmres, pcg, bicgstab};

// GMRES for a general non-symmetric system
let x = gmres(&a.view(), &b.view(), None, Some(30), Some(200), Some(1e-10))?;

// PCG for symmetric positive definite
let x = pcg(&a_spd.view(), &b.view(), None, Some(500), Some(1e-12))?;

// BiCGStab for non-symmetric
let x = bicgstab(&a.view(), &b.view(), None, Some(500), Some(1e-10))?;
```

### Matrix functions

```rust
use scirs2_linalg::matrix_functions::{expm, logm, sqrtm};

let exp_a = expm(&a.view())?;
let log_a = logm(&a.view())?;
let sqrt_a = sqrtm(&a.view())?;
```

### Tensor decompositions

```rust
use scirs2_linalg::tensor::{cp_als, tucker_hooi};

// CP decomposition with 5 components, 200 ALS iterations
let cp = cp_als(&tensor, 5, Some(200), Some(1e-8))?;

// Tucker decomposition with rank [3, 3, 3]
let tucker = tucker_hooi(&tensor, &[3, 3, 3], Some(100), Some(1e-8))?;
```

### Control theory

```rust
use scirs2_linalg::control::{solve_care, solve_dare, solve_lyapunov};

// Continuous algebraic Riccati equation: A^T X + X A - X B R^{-1} B^T X + Q = 0
let x = solve_care(&a.view(), &b.view(), &q.view(), &r.view())?;

// Discrete algebraic Riccati equation
let x = solve_dare(&a.view(), &b.view(), &q.view(), &r.view())?;

// Lyapunov equation: A X + X A^T + Q = 0
let x = solve_lyapunov(&a.view(), &q.view())?;
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `simd` | SIMD-accelerated kernels (AVX/AVX2/AVX-512/NEON) |
| `parallel` | Multi-threaded operations via Rayon |
| `gpu` | GPU acceleration (requires `scirs2-core` gpu feature) |
| `linalg` | Enable OxiBLAS pure-Rust BLAS/LAPACK backend |
| `serde` | Serialization for matrix types |

## Links

- [API Documentation](https://docs.rs/scirs2-linalg)
- [SciRS2 Repository](https://github.com/cool-japan/scirs)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
