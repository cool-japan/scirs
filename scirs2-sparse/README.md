# SciRS2 Sparse

[![crates.io](https://img.shields.io/crates/v/scirs2-sparse.svg)](https://crates.io/crates/scirs2-sparse)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-sparse)](https://docs.rs/scirs2-sparse)

Sparse matrix module for the SciRS2 scientific computing library. This module provides various sparse matrix formats and operations for efficient handling of sparse data.

## Features

- **Sparse Matrix Formats**: Multiple sparse matrix formats (CSR, CSC, COO, DOK, LIL, DIA, BSR)
- **Format Conversions**: Utilities for converting between different sparse formats
- **Linear Algebra**: Operations optimized for sparse matrices including addition, multiplication, and spectral norms
- **Matrix Generation**: Utility functions for creating special matrices (diagonal, identity)
- **Matrix Norms**: Compute 1-norm, infinity norm, Frobenius norm, and spectral norm
- **Utility Functions**: Helper functions for working with sparse matrices

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-sparse = "0.1.0-alpha.2"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-sparse = { version = "0.1.0-alpha.2", features = ["parallel"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_sparse::{csr, csc, coo, convert, linalg};
use scirs2_core::error::CoreResult;
use ndarray::{array};

// Create and use a CSR matrix
fn csr_matrix_example() -> CoreResult<()> {
    // Create a CSR matrix from data, indices, and indptr
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let indices = vec![0, 1, 0, 2, 3];
    let indptr = vec![0, 2, 2, 4, 5];
    let shape = (4, 4);
    
    let csr_mat = csr::CsrMatrix::new(data, indices, indptr, shape)?;
    
    // Get a specific value
    let value = csr_mat.get(0, 1)?;
    println!("Value at (0, 1): {}", value);
    
    // Convert to dense matrix
    let dense = csr_mat.to_dense()?;
    println!("Dense matrix:\n{:?}", dense);
    
    // Matrix-vector multiplication
    let vector = vec![1.0, 2.0, 3.0, 4.0];
    let result = linalg::spmv(&csr_mat, &vector)?;
    println!("Matrix-vector product: {:?}", result);
    
    Ok(())
}

// Convert between different sparse formats
fn format_conversion_example() -> CoreResult<()> {
    // Create a dense matrix with some zeros
    let dense = array![
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 3.0],
        [0.0, 4.0, 0.0, 0.0],
        [5.0, 0.0, 6.0, 0.0]
    ];
    
    // Convert to COO format
    let coo_mat = convert::dense_to_coo(&dense)?;
    println!("COO format - data: {:?}, row: {:?}, col: {:?}", 
             coo_mat.data(), coo_mat.row(), coo_mat.col());
    
    // Convert COO to CSR
    let csr_mat = convert::coo_to_csr(&coo_mat)?;
    println!("CSR format - data: {:?}, indices: {:?}, indptr: {:?}", 
             csr_mat.data(), csr_mat.indices(), csr_mat.indptr());
    
    // Convert CSR to CSC
    let csc_mat = convert::csr_to_csc(&csr_mat)?;
    println!("CSC format - data: {:?}, indices: {:?}, indptr: {:?}", 
             csc_mat.data(), csc_mat.indices(), csc_mat.indptr());
    
    // Convert back to dense
    let dense_from_csc = csc_mat.to_dense()?;
    println!("Back to dense:\n{:?}", dense_from_csc);
    
    Ok(())
}

// Sparse linear algebra operations
fn sparse_linalg_example() -> CoreResult<()> {
    // Create two CSR matrices
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let indices_a = vec![0, 1, 1, 2];
    let indptr_a = vec![0, 2, 3, 4];
    let shape_a = (3, 3);
    
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let indices_b = vec![0, 1, 0, 2];
    let indptr_b = vec![0, 2, 3, 4];
    let shape_b = (3, 3);
    
    let csr_a = csr::CsrMatrix::new(data_a, indices_a, indptr_a, shape_a)?;
    let csr_b = csr::CsrMatrix::new(data_b, indices_b, indptr_b, shape_b)?;
    
    // Matrix addition
    let sum = linalg::add(&csr_a, &csr_b)?;
    println!("Matrix sum (CSR format):");
    println!("  data: {:?}", sum.data());
    println!("  indices: {:?}", sum.indices());
    println!("  indptr: {:?}", sum.indptr());
    
    // Matrix multiplication
    let product = linalg::matmul(&csr_a, &csr_b)?;
    println!("Matrix product (CSR format):");
    println!("  data: {:?}", product.data());
    println!("  indices: {:?}", product.indices());
    println!("  indptr: {:?}", product.indptr());
    
    Ok(())
}
```

## Components

### Sparse Matrix Formats

Various sparse matrix implementations:

```rust
use scirs2_sparse::{
    csr::CsrMatrix,         // Compressed Sparse Row format
    csc::CscMatrix,         // Compressed Sparse Column format
    coo::CooMatrix,         // COOrdinate format
    dia::DiaMatrix,         // DIAgonal format
    bsr::BsrMatrix,         // Block Sparse Row format
    lil::LilMatrix,         // List of Lists format
    dok::DokMatrix,         // Dictionary of Keys format
};
```

### Format Conversions

Functions for converting between formats:

```rust
use scirs2_sparse::convert::{
    dense_to_csr,           // Convert dense matrix to CSR
    dense_to_csc,           // Convert dense matrix to CSC
    dense_to_coo,           // Convert dense matrix to COO
    csr_to_csc,             // Convert CSR to CSC
    csc_to_csr,             // Convert CSC to CSR
    coo_to_csr,             // Convert COO to CSR
    coo_to_csc,             // Convert COO to CSC
    csr_to_coo,             // Convert CSR to COO
    csc_to_coo,             // Convert CSC to COO
    csr_to_bsr,             // Convert CSR to BSR
    bsr_to_csr,             // Convert BSR to CSR
    lil_to_csr,             // Convert LIL to CSR
    dok_to_csr,             // Convert DOK to CSR
};
```

### Linear Algebra

Sparse matrix operations:

```rust
use scirs2_sparse::linalg::{
    // Matrix-vector operations
    spmv,                   // Sparse matrix-vector multiplication
    
    // Matrix-matrix operations
    add,                    // Add two sparse matrices
    subtract,               // Subtract two sparse matrices
    multiply,               // Element-wise multiplication (Hadamard product)
    divide,                 // Element-wise division
    matmul,                 // Matrix multiplication
    transpose,              // Matrix transpose
    
    // Matrix functions
    diag_matrix,            // Create diagonal matrix from vector
    eye,                    // Create identity matrix
    
    // Matrix norms
    norm,                   // Compute matrix norm (1-norm, inf-norm, Frobenius, spectral)
    
    // Decompositions
    sparse_cholesky,        // Cholesky decomposition for SPD matrices
    sparse_lu,              // LU decomposition with partial pivoting
    sparse_ldlt,            // LDLT decomposition for symmetric matrices
    
    // Solvers
    spsolve,                // Solve linear system Ax = b
    sparse_direct_solve,    // Direct solver with different decomposition options
    sparse_lstsq,           // Solve least squares problem min ||Ax - b||₂
    sparse_cholesky_solve,  // Solve using Cholesky decomposition
    sparse_lu_solve,        // Solve using LU decomposition
    sparse_ldlt_solve,      // Solve using LDLT decomposition
};
```

### Utilities

Helper functions:

```rust
use scirs2_sparse::utils::{
    is_sparse,              // Check if a matrix should be stored as sparse
    density,                // Calculate density of a matrix
    find,                   // Find non-zero elements
    spdiags,                // Extract or set diagonals
    kron,                   // Kronecker product
    block_diag,             // Create block diagonal matrix
    hstack,                 // Stack matrices horizontally
    vstack,                 // Stack matrices vertically
};
```

## Performance Considerations

The sparse matrix implementations are optimized for performance:

- Memory-efficient storage for matrices with many zeros
- Specialized algorithms for sparse operations
- Leverages the `sprs` crate for core implementations
- Support for parallel operations when feature flags are enabled

Example of performance comparison:

```rust
use scirs2_sparse::{csr, linalg};
use ndarray::Array2;
use std::time::Instant;

// Create a large, sparse matrix (95% zeros)
let n = 1000;
let density = 0.05;
let sparse_mat = csr::random(n, n, density).unwrap();

// Convert to dense for comparison
let dense_mat = sparse_mat.to_dense().unwrap();

// Create a vector
let vec = vec![1.0; n];

// Measure sparse matrix-vector multiplication time
let start = Instant::now();
let sparse_result = linalg::spmv(&sparse_mat, &vec).unwrap();
let sparse_time = start.elapsed();

// Measure dense matrix-vector multiplication time
let start = Instant::now();
let dense_result = dense_mat.dot(&vec);
let dense_time = start.elapsed();

println!("Sparse multiplication time: {:?}", sparse_time);
println!("Dense multiplication time: {:?}", dense_time);
println!("Speedup: {:.2}x", dense_time.as_secs_f64() / sparse_time.as_secs_f64());
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
