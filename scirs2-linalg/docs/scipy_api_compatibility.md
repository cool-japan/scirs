# SciPy API Compatibility Guide

This document provides a comprehensive guide to the SciPy `scipy.linalg` API compatibility offered by `scirs2-linalg`. The compatibility layer allows for easy migration from SciPy code to Rust while maintaining the same function signatures and parameter names.

## Overview

The `scirs2_linalg::compat` module provides SciPy-compatible wrappers around the core `scirs2-linalg` functionality. This allows Python users familiar with SciPy to use similar APIs in Rust with minimal code changes.

## API Coverage

### ✅ Fully Implemented Functions

#### Basic Matrix Operations
| SciPy Function | scirs2-linalg Function | Description |
|---------------|------------------------|-------------|
| `scipy.linalg.det()` | `compat::det()` | Matrix determinant |
| `scipy.linalg.inv()` | `compat::inv()` | Matrix inverse |
| `scipy.linalg.pinv()` | `compat::pinv()` | Moore-Penrose pseudoinverse |
| `scipy.linalg.norm()` | `compat::norm()` | Matrix and vector norms |
| `scipy.linalg.cond()` | `compat::cond()` | Condition number |
| `scipy.linalg.matrix_rank()` | `compat::matrix_rank()` | Matrix rank |

#### Matrix Decompositions
| SciPy Function | scirs2-linalg Function | Description |
|---------------|------------------------|-------------|
| `scipy.linalg.lu()` | `compat::lu()` | LU decomposition |
| `scipy.linalg.qr()` | `compat::qr()` | QR decomposition |
| `scipy.linalg.svd()` | `compat::svd()` | Singular value decomposition |
| `scipy.linalg.cholesky()` | `compat::cholesky()` | Cholesky decomposition |
| `scipy.linalg.rq()` | `compat::rq()` | RQ decomposition |
| `scipy.linalg.polar()` | `compat::polar()` | Polar decomposition |

#### Eigenvalue Problems
| SciPy Function | scirs2-linalg Function | Description |
|---------------|------------------------|-------------|
| `scipy.linalg.eigh()` | `compat::eigh()` | Symmetric/Hermitian eigenvalues |
| `scipy.linalg.eigvals()` | `compat::eigvals()` | Eigenvalues only |
| `scipy.linalg.eigvalsh()` | `compat::eigvalsh()` | Eigenvalues of symmetric matrix |
| `scipy.linalg.eig_banded()` | `compat::eig_banded()` | Banded matrix eigenvalues |
| `scipy.linalg.eigh_tridiagonal()` | `compat::eigh_tridiagonal()` | Tridiagonal eigenvalues |

#### Linear System Solvers
| SciPy Function | scirs2-linalg Function | Description |
|---------------|------------------------|-------------|
| `scipy.linalg.solve()` | `compat::compat_solve()` | General linear system solver |
| `scipy.linalg.lstsq()` | `compat::lstsq()` | Least squares solver |
| `scipy.linalg.solve_triangular()` | `compat::solve_triangular()` | Triangular system solver |

#### Matrix Functions
| SciPy Function | scirs2-linalg Function | Description |
|---------------|------------------------|-------------|
| `scipy.linalg.expm()` | `compat::expm()` | Matrix exponential |
| `scipy.linalg.logm()` | `compat::logm()` | Matrix logarithm |
| `scipy.linalg.sqrtm()` | `compat::sqrtm()` | Matrix square root |
| `scipy.linalg.fractional_matrix_power()` | `compat::fractional_matrix_power()` | Fractional matrix power |
| `scipy.linalg.funm()` | `compat::funm()` | General matrix function |

#### Utility Functions
| SciPy Function | scirs2-linalg Function | Description |
|---------------|------------------------|-------------|
| `scipy.linalg.block_diag()` | `compat::block_diag()` | Block diagonal matrix |

### 🚧 Partially Implemented Functions

#### Matrix Functions (Basic Implementation)
| SciPy Function | scirs2-linalg Function | Status | Notes |
|---------------|------------------------|--------|-------|
| `scipy.linalg.cosm()` | `compat::cosm()` | Stub | Returns NotImplemented error |
| `scipy.linalg.sinm()` | `compat::sinm()` | Stub | Returns NotImplemented error |
| `scipy.linalg.tanm()` | `compat::tanm()` | Stub | Returns NotImplemented error |

#### General Eigenvalue Problem
| SciPy Function | scirs2-linalg Function | Status | Notes |
|---------------|------------------------|--------|-------|
| `scipy.linalg.eig()` | `compat::eig()` | Partial | No complex eigenvalue handling |

### ❌ Not Yet Implemented

#### Specialized Decompositions
| SciPy Function | Status | Priority |
|---------------|--------|----------|
| `scipy.linalg.schur()` | Not implemented | Medium |
| `scipy.linalg.qz()` | Not implemented | Low |
| `scipy.linalg.ordqz()` | Not implemented | Low |
| `scipy.linalg.orth()` | Not implemented | Medium |
| `scipy.linalg.null_space()` | Not implemented | Medium |

#### Specialized Solvers
| SciPy Function | Status | Priority |
|---------------|--------|----------|
| `scipy.linalg.solve_banded()` | Not implemented | High |
| `scipy.linalg.solveh_banded()` | Not implemented | Medium |
| `scipy.linalg.solve_toeplitz()` | Not implemented | Low |
| `scipy.linalg.solve_circulant()` | Not implemented | Low |
| `scipy.linalg.solve_discrete_are()` | Not implemented | Low |
| `scipy.linalg.solve_continuous_are()` | Not implemented | Low |
| `scipy.linalg.solve_discrete_lyapunov()` | Not implemented | Low |
| `scipy.linalg.solve_continuous_lyapunov()` | Not implemented | Low |

#### Matrix Analysis
| SciPy Function | Status | Priority |
|---------------|--------|----------|
| `scipy.linalg.LinAlgError` | Not implemented | Medium |
| `scipy.linalg.LinAlgWarning` | Not implemented | Low |

## Usage Examples

### Basic Matrix Operations

```rust
use ndarray::array;
use scirs2_linalg::compat;

let a = array![[4.0, 2.0], [2.0, 3.0]];

// Determinant (same as scipy.linalg.det(a))
let det = compat::det(&a.view(), false, true)?;

// Matrix inverse (same as scipy.linalg.inv(a))
let inv = compat::inv(&a.view(), false, true)?;

// Condition number (same as scipy.linalg.cond(a))
let cond = compat::cond(&a.view(), Some("2"))?;
```

### Matrix Decompositions

```rust
use ndarray::array;
use scirs2_linalg::compat;

let a = array![[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]];

// LU decomposition (same as scipy.linalg.lu(a))
let (p, l, u) = compat::lu(&a.view(), false, false, true, false)?;

// QR decomposition (same as scipy.linalg.qr(a))
let (q_opt, r) = compat::qr(&a.view(), false, None, "full", false, true)?;

// SVD (same as scipy.linalg.svd(a))
let (u_opt, s, vt_opt) = compat::svd(&a.view(), true, true, false, true, "gesdd")?;
```

### Matrix Norms

```rust
use ndarray::array;
use scirs2_linalg::compat;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let v = array![3.0, 4.0];

// Matrix norms (same as scipy.linalg.norm(a, ord=...))
let frobenius = compat::norm(&a.view(), Some("fro"), None, false, true)?;
let norm_1 = compat::norm(&a.view(), Some("1"), None, false, true)?;
let norm_inf = compat::norm(&a.view(), Some("inf"), None, false, true)?;

// Vector norms (same as scipy.linalg.norm(v, ord=...))
let v_norm_2 = compat::vector_norm(&v.view(), Some(2.0), true)?;
let v_norm_1 = compat::vector_norm(&v.view(), Some(1.0), true)?;
```

### Linear System Solving

```rust
use ndarray::array;
use scirs2_linalg::compat;

let a = array![[3.0, 1.0], [1.0, 2.0]];
let b = array![[4.0], [3.0]];

// General solve (same as scipy.linalg.solve(a, b))
let x = compat::compat_solve(&a.view(), &b.view(), false, false, false, true, None, false)?;

// Least squares (same as scipy.linalg.lstsq(a, b))
let (solution, residuals, rank, s) = compat::lstsq(&a.view(), &b.view(), None, false, false, true, None)?;
```

### Matrix Functions

```rust
use ndarray::array;
use scirs2_linalg::compat;

let a = array![[1.0, 0.1], [0.1, 1.0]];

// Matrix exponential (same as scipy.linalg.expm(a))
let exp_a = compat::expm(&a.view())?;

// Matrix square root (same as scipy.linalg.sqrtm(a))
let sqrt_a = compat::sqrtm(&a.view())?;

// General matrix function (same as scipy.linalg.funm(a, func))
let exp_via_funm = compat::funm(&a.view(), "exp", false)?;
```

## Parameter Compatibility

### Common Parameters

Most SciPy-compatible functions support the following common parameters:

- `check_finite`: Whether to check that input matrices contain only finite numbers
- `overwrite_a`, `overwrite_b`: Whether to allow overwriting input data (currently ignored)

### Error Handling

The compatibility layer maintains SciPy-style error handling:

- `LinalgError::ValueError`: Input validation errors (e.g., non-finite values)
- `LinalgError::NotImplemented`: Features not yet implemented
- `LinalgError::InvalidInput`: Invalid parameter combinations
- `LinalgError::DimensionError`: Matrix dimension mismatches

## Migration Guide

### From SciPy Python to scirs2-linalg Rust

1. **Import changes:**
   ```python
   # Python/SciPy
   from scipy.linalg import det, inv, svd
   ```
   ```rust
   // Rust/scirs2-linalg
   use scirs2_linalg::compat::{det, inv, svd};
   ```

2. **Function calls:**
   ```python
   # Python/SciPy
   result = det(matrix, check_finite=True)
   ```
   ```rust
   // Rust/scirs2-linalg
   let result = det(&matrix.view(), false, true)?;
   ```

3. **Error handling:**
   ```python
   # Python/SciPy
   try:
       result = inv(matrix)
   except LinAlgError:
       # Handle error
   ```
   ```rust
   // Rust/scirs2-linalg
   match inv(&matrix.view(), false, true) {
       Ok(result) => { /* Use result */ }
       Err(e) => { /* Handle error */ }
   }
   ```

## Performance Considerations

### Accelerated Implementations

When available, the compatibility layer automatically uses accelerated implementations:

- **BLAS/LAPACK**: For basic linear algebra operations
- **SIMD**: For vectorized operations when the `simd` feature is enabled
- **Parallel**: Multi-threaded operations for large matrices

### Memory Management

- Most functions avoid unnecessary allocations
- Input matrices are not modified unless explicitly requested
- Results are returned as owned arrays for safety

## Feature Flags

The SciPy compatibility layer respects the following feature flags:

- `blas`: Enables BLAS-accelerated operations
- `lapack`: Enables LAPACK-accelerated decompositions
- `simd`: Enables SIMD vectorization
- `autograd`: Enables automatic differentiation support

## Future Roadmap

### Short Term (Next Release)
- Implement `solve_banded()` for banded matrix systems
- Add `schur()` decomposition
- Implement trigonometric matrix functions (`cosm`, `sinm`, `tanm`)

### Medium Term
- Complete generalized eigenvalue problem support in `eig()`
- Add `orth()` and `null_space()` functions
- Implement specialized Lyapunov and Riccati equation solvers

### Long Term
- Full complex number support across all functions
- Advanced numerical methods for structured matrices
- GPU acceleration for supported operations

## Contributing

To contribute to SciPy API compatibility:

1. Check the "Not Yet Implemented" section for priority functions
2. Follow the existing function signature patterns
3. Add comprehensive tests with SciPy result comparisons
4. Update this documentation with the new functions

See `examples/scipy_compat_showcase.rs` for comprehensive usage examples and test patterns.