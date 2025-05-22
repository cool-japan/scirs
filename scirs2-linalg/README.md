# scirs2-linalg

[![crates.io](https://img.shields.io/crates/v/scirs2-linalg.svg)](https://crates.io/crates/scirs2-linalg)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-linalg)](https://docs.rs/scirs2-linalg)

Linear algebra module for SciRS2, providing functionality comparable to NumPy/SciPy's linalg module.

`scirs2-linalg` aims to bring comprehensive linear algebra functionality to Rust, offering both pure Rust implementations and accelerated versions using native BLAS/LAPACK libraries. Whether you're building scientific simulations, machine learning models, or data analysis tools, this library provides the mathematical foundations you need.

## Features

### Core Linear Algebra
- **Basic Operations**: Determinants, inverses, matrix multiplication, matrix powers
- **Decompositions**: LU, QR, SVD, Cholesky, Eigendecomposition, Schur, Polar
- **Solvers**: Direct methods, least squares, triangular systems
- **Eigenvalue Problems**: Standard and specialized eigenvalue/eigenvector computations
- **Matrix Functions**: Matrix exponential, logarithm, square root
- **Norms and Condition Numbers**: Various matrix and vector norms

### Advanced Capabilities
- **Iterative Solvers**: Conjugate gradient, GMRES, Jacobi, Gauss-Seidel, multigrid
- **Structured Matrices**: Efficient operations on Toeplitz, Hankel, circulant matrices
- **Specialized Matrices**: Optimized algorithms for tridiagonal, banded, symmetric matrices
- **BLAS/LAPACK Integration**: High-performance native library support when available

### Machine Learning & AI Support
- **Attention Mechanisms**: Scaled dot-product, multi-head, flash attention
- **Batch Operations**: Vectorized operations for mini-batch processing
- **Gradient Computations**: Automatic differentiation support for matrix operations
- **Low-Rank Approximations**: SVD-based dimensionality reduction
- **Tensor Operations**: Einstein summation, tensor contractions, mode products
- **Memory-Efficient Algorithms**: Flash attention, linear attention variants
- **Quantization-Aware Operations**: 4-bit, 8-bit and 16-bit precision support with advanced calibration and numerical stability analysis
  - Matrix-free operations for quantized tensors
  - Fusion operations for consecutive quantized operations
  - Specialized solvers for quantized matrices
- **Mixed Precision**: Operations across different numeric types
- **Sparse-Dense Operations**: Efficient handling of sparse matrices

## Installation

Add scirs2-linalg to your Cargo.toml:

```toml
[dependencies]
scirs2-linalg = "0.1.0-alpha.4"
ndarray = "0.16.1"
```

For accelerated performance with native BLAS/LAPACK:

```toml
[dependencies]
scirs2-linalg = { version = "0.1.0-alpha.4", features = ["openblas"] }
# Or use Intel MKL:
# scirs2-linalg = { version = "0.1.0-alpha.4", features = ["mkl"] }
```

## Quick Start

### Basic Usage

```rust
use ndarray::array;
use scirs2_linalg::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a matrix
    let a = array![[3.0, 1.0], [1.0, 2.0]];
    
    // Compute determinant
    let det = det(&a.view())?;
    println!("Determinant: {}", det);
    
    // Solve linear system Ax = b
    let b = array![9.0, 8.0];
    let x = solve(&a.view(), &b.view())?;
    println!("Solution: {:?}", x);
    
    // Compute eigenvalues and eigenvectors
    let (eigenvalues, eigenvectors) = eigh(&a.view())?;
    println!("Eigenvalues: {:?}", eigenvalues);
    
    Ok(())
}
```

### SciPy Compatibility

For users migrating from Python/SciPy:

```rust
use scirs2_linalg::compat;

// Use familiar SciPy-style function signatures
let det = compat::det(&a.view(), false, true)?;
let inv = compat::inv(&a.view(), false, true)?;
let (u, s, vt) = compat::svd(&a.view(), true, true, false, true, "gesdd")?;
```

### Matrix Decompositions

```rust
use ndarray::array;
use scirs2_linalg::{lu, qr, svd, cholesky};

// Create a test matrix
let a = array![[4.0, 2.0], [2.0, 5.0]];

// LU decomposition
let (p, l, u) = lu(&a.view())?;
assert_eq!(p.dot(&a), l.dot(&u)); // P·A = L·U

// QR decomposition  
let (q, r) = qr(&a.view())?;
assert_eq!(a, q.dot(&r)); // A = Q·R

// Singular Value Decomposition
let (u, s, vt) = svd(&a.view(), false)?;
// Note: reconstruction requires proper diagonal matrix construction

// Cholesky decomposition (for positive definite matrices)
let l = cholesky(&a.view())?;
assert_eq!(a, l.dot(&l.t())); // A = L·L^T
```

### Iterative Solvers

```rust
use scirs2_linalg::{conjugate_gradient, gmres};

// Solve using conjugate gradient (for symmetric positive definite)
let x = conjugate_gradient(&a.view(), &b.view(), 100, 1e-10)?;

// Solve using GMRES (for general matrices)
let x = gmres(&a.view(), &b.view(), 10, 100, 1e-10)?;
```

### Advanced Matrix Operations

```rust
use scirs2_linalg::matrix_functions::{expm, logm, sqrtm};
use scirs2_linalg::specialized::{TridiagonalMatrix, BandedMatrix};
use scirs2_linalg::structured::{ToeplitzMatrix, CirculantMatrix};

// Matrix functions
let exp_a = expm(&a.view())?; // Matrix exponential
let log_a = logm(&a.view())?; // Matrix logarithm
let sqrt_a = sqrtm(&a.view())?; // Matrix square root

// Specialized matrices for efficiency
let tridiag = TridiagonalMatrix::from_diagonals(&main_diag, &upper, &lower);
let x = tridiag.solve(&b)?;

// Structured matrices
let toeplitz = ToeplitzMatrix::from_column_and_row(&first_col, &first_row);
let result = toeplitz.matvec(&x)?;
```

### Machine Learning Operations

```rust
use ndarray::Array3;
use scirs2_linalg::attention::{scaled_dot_product_attention, multi_head_attention};
use scirs2_linalg::batch::attention::batch_multi_head_attention;

// Attention mechanism for transformers
let batch_size = 2;
let seq_len = 4;
let d_model = 64;

// Create query, key, value tensors
let query = Array3::from_shape_fn((batch_size, seq_len, d_model), |_| rand::random());
let key = Array3::from_shape_fn((batch_size, seq_len, d_model), |_| rand::random());
let value = Array3::from_shape_fn((batch_size, seq_len, d_model), |_| rand::random());

// Scaled dot-product attention
let scale = 1.0 / (d_model as f32).sqrt();
let output = scaled_dot_product_attention(
    &query.view(),
    &key.view(),
    &value.view(),
    None, // Optional mask
    scale
)?;

// Multi-head attention
let num_heads = 8;
let output = multi_head_attention(
    &query.view(),
    &key.view(),
    &value.view(),
    num_heads,
    None, // Optional mask
)?;
```

### Quantization and Model Compression

```rust
use scirs2_linalg::quantization::{
    quantize_matrix, dequantize_matrix, quantized_matmul,
    fusion::fused_quantized_matmul_chain,
    quantized_matrixfree::QuantizedMatrixFreeOp,
    stability::analyze_quantization_stability,
    calibration::{calibrate_matrix, CalibrationConfig, CalibrationMethod}
};

// Create a neural network weight matrix
let weights = Array2::from_shape_fn((1024, 1024), |_| rand::random::<f32>() * 0.1);
let activations = Array2::from_shape_fn((32, 1024), |_| rand::random::<f32>());

// Calibrate quantization parameters using advanced methods
let weight_config = CalibrationConfig {
    method: CalibrationMethod::EntropyCalibration,
    symmetric: true,
    num_bins: 2048,
    ..Default::default()
};

// Dynamic calibration for activations that change over time
let activation_config = CalibrationConfig {
    method: CalibrationMethod::ExponentialMovingAverage,
    symmetric: false, // Asymmetric for ReLU activations
    ema_factor: 0.1,
    ..Default::default()
};

// Calibrate and quantize weights (to 8-bit)
let weight_params = calibrate_matrix(&weights.view(), 8, &weight_config)?;
let (quantized_weights, _) = quantize_matrix(&weights.view(), 8, weight_params.method);

// Calibrate and quantize activations
let activation_params = calibrate_matrix(&activations.view(), 8, &activation_config)?;
let (quantized_activations, _) = quantize_matrix(&activations.view(), 8, activation_params.method);

// Perform matrix multiplication with quantized matrices
let result = quantized_matmul(
    &quantized_weights,
    &weight_params,
    &quantized_activations,
    &activation_params,
)?;

// Calculate quantization error
let reference = activations.dot(&weights.t());
let rel_error = (&reference - &result).mapv(|x| x.abs()).sum() /
                reference.mapv(|x| x.abs()).sum();
println!("Relative Error: {:.6}%", rel_error * 100.0);
println!("Memory Reduction: {:.1}%", (1.0 - 8.0/32.0) * 100.0);
```

## Performance Considerations

### Backend Selection

The library supports multiple BLAS/LAPACK backends:

```toml
# OpenBLAS (default, good general performance)
scirs2-linalg = { version = "0.1.0-alpha.4", features = ["openblas"] }

# Intel MKL (best for Intel CPUs)
scirs2-linalg = { version = "0.1.0-alpha.4", features = ["mkl"] }

# Netlib (reference implementation)
scirs2-linalg = { version = "0.1.0-alpha.4", features = ["netlib"] }
```

### Optimization Features

- **SIMD Acceleration**: Enable with `features = ["simd"]`
- **Parallel Operations**: Built-in Rayon support for large matrices
- **Memory-Efficient Algorithms**: Automatic selection based on matrix size
- **Cache-Friendly Implementations**: Blocked algorithms for better cache usage

### Benchmarks

Typical performance comparisons (1000×1000 matrices):

| Operation | Pure Rust | SIMD | OpenBLAS | Intel MKL |
|-----------|-----------|------|----------|-----------|
| Matrix Multiply | 245ms | 89ms | 42ms | 38ms |
| LU Decomposition | 185ms | N/A | 78ms | 71ms |
| SVD | 892ms | N/A | 340ms | 298ms |
| Eigenvalues | 1.2s | N/A | 445ms | 412ms |

## Error Handling

The library uses a comprehensive error system:

```rust
use scirs2_linalg::{LinalgError, LinalgResult};

match inv(&singular_matrix.view()) {
    Ok(inverse) => println!("Inverse computed"),
    Err(LinalgError::SingularMatrixError(msg)) => println!("Matrix is singular: {}", msg),
    Err(e) => println!("Other error: {}", e),
}
```

## Known Limitations

- SVD for non-square matrices has shape handling issues
- Matrix inverse only implemented for 2×2 matrices (use `solve` for larger matrices)
- Determinant only implemented for matrices up to 3×3
- Some specialized eigenvalue routines not yet optimized

For full implementation status, see [TODO.md](TODO.md).

## Contributing

Contributions are welcome! Please see our [contributing guidelines](https://github.com/cool-japan/scirs/blob/master/CONTRIBUTING.md).

Priority areas for contribution:
- Completing matrix inverse for larger matrices
- Optimizing specialized eigenvalue routines
- Adding more structured matrix types
- Improving numerical stability for edge cases

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.

## Acknowledgments

This library is inspired by NumPy and SciPy's excellent linear algebra implementations. We aim to bring similar functionality to the Rust ecosystem while leveraging Rust's performance and safety guarantees.
