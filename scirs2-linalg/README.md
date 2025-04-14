# scirs2-linalg

[![crates.io](https://img.shields.io/crates/v/scirs2-linalg.svg)](https://crates.io/crates/scirs2-linalg)
[![License](https://img.shields.io/crates/l/scirs2-linalg.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-linalg)](https://docs.rs/scirs2-linalg)

Linear algebra module for SciRS2, providing functionality comparable to NumPy/SciPy's linalg module.

## Features

- Core matrix operations (det, inv, solve, etc.)
- Matrix decompositions (LU, QR, SVD, Cholesky)
- Eigenvalue solvers
- BLAS and LAPACK interfaces
- Specialized operations for AI/ML:
  - Batch matrix operations
  - Gradient calculation utilities
  - Efficient matrix multiplication algorithms
  - Low-rank approximation techniques
  - Kronecker product optimization
  - Tensor contraction operations
  - Matrix-free operations for iterative solvers
  - Structured matrices (Toeplitz, circulant, etc.)
  - Attention mechanisms for transformer models
  - Memory-efficient attention algorithms (flash, linear)
  - Batched attention operations for high-throughput ML training
  - Quantization-aware linear algebra operations
  - Mixed-precision operations
  - Sparse-dense matrix operations

## Installation

Add scirs2-linalg to your Cargo.toml:

```toml
[dependencies]
scirs2-linalg = "0.1.0-alpha.1"
ndarray = "0.16.1"
```

## Usage

```rust
use ndarray::{array, Array2, Array3};
use scirs2_linalg::{solve, matrix_power, decomposition::svd};
use scirs2_linalg::attention::{scaled_dot_product_attention, multi_head_attention, AttentionConfig};
use scirs2_linalg::batch::attention::{batch_multi_head_attention, batch_flash_attention};

// Solve a linear system
let a = array![[1.0, 2.0], [3.0, 4.0]];
let b = array![5.0, 6.0];
let x = solve(&a.view(), &b.view()).unwrap();

// Compute matrix power
let a = array![[1.0, 2.0], [3.0, 4.0]];
let a_cubed = matrix_power(&a.view(), 3).unwrap();

// Perform SVD decomposition
let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let (u, s, vt) = svd(&a.view(), true).unwrap();

// Use attention mechanisms for transformer models
let batch_size = 2;
let seq_len = 4;
let d_model = 64;

// Create query, key, value matrices
let query = Array3::<f32>::ones((batch_size, seq_len, d_model));
let key = Array3::<f32>::ones((batch_size, seq_len, d_model));
let value = Array3::<f32>::ones((batch_size, seq_len, d_model));

// Compute attention
let scale = 1.0 / (d_model as f32).sqrt();
let output = scaled_dot_product_attention(
    &query.view(),
    &key.view(),
    &value.view(),
    None,
    scale
).unwrap();
```

## AI/ML Support

This module includes optimized implementations for deep learning and machine learning operations:

- Fast tensor contractions for neural networks
- Efficient convolution operations (im2col, col2im)
- Kronecker factorization for second-order optimization
- Batch operations for mini-batch processing
- Batched attention functions for efficient transformer training
- Matrix-free operations for large models
- Specialized matrices for efficient representation

## Performance

The module provides various backends for optimal performance:

- OpenBLAS (default)
- Intel MKL
- Netlib

Select a backend using the corresponding feature flag.

## License

Licensed under the Apache License, Version 2.0.