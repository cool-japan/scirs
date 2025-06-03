# scirs2-autograd Linear Algebra API Documentation

## Overview

This document describes the comprehensive linear algebra operations implemented in scirs2-autograd. All operations support automatic differentiation (gradients).

## Basic Linear Algebra Operations

### Matrix Creation
- `eye(n, ctx)` - Create an n×n identity matrix
- `diag(v)` - Create a diagonal matrix from a vector
- `extract_diag(m)` - Extract diagonal elements from a matrix

### Matrix Properties
- `trace(m)` - Compute the trace (sum of diagonal elements)
- `frobenius_norm(m)` - Compute the Frobenius norm
- `determinant(m)` - Compute the determinant

### Scalar Operations
- `scalar_mul(m, scalar)` - Multiply matrix by scalar (also available as `tensor.scalar_mul(s)`)

## Matrix Operations

### Basic Operations
- `matrix_inverse(m)` - Compute matrix inverse
- `matrix_pseudo_inverse(m)` - Compute Moore-Penrose pseudo-inverse

### Matrix Functions
- `matrix_exp(m)` - Matrix exponential
- `matrix_log(m)` - Matrix logarithm
- `matrix_sqrt(m)` - Matrix square root
- `matrix_pow(m, p)` - Matrix power

## Matrix Decompositions

### Standard Decompositions
- `qr(m)` - QR decomposition (returns Q, R)
- `lu(m)` - LU decomposition with pivoting (returns L, U, P)
- `svd(m)` - Singular Value Decomposition (returns U, S, V)
- `cholesky(m)` - Cholesky decomposition for positive definite matrices

### Eigenvalue Decomposition
- `eigen(m)` - Full eigendecomposition (returns eigenvalues, eigenvectors)
- `eigenvalues(m)` - Compute only eigenvalues

## Linear System Solving

- `solve(A, b)` - Solve Ax = b for x
- `lstsq(A, b)` - Least squares solution for overdetermined systems

## Special Matrix Operations

### Triangular Operations
- `tril(m, k)` - Extract lower triangular part (k=0 for main diagonal)
- `triu(m, k)` - Extract upper triangular part (k=0 for main diagonal)
- `band_matrix(m, lower, upper)` - Extract band matrix

### Matrix Transformations
- `symmetrize(m)` - Make matrix symmetric: (M + M^T) / 2

## Gradient Support

All operations support automatic differentiation. Gradients flow through:
- Matrix inverse (using implicit differentiation)
- Decompositions (QR, LU, SVD, Cholesky)
- Eigenvalue computations
- Linear solvers
- Matrix functions

## Usage Examples

### Basic Usage
```rust
use scirs2_autograd as ag;
use ag::tensor_ops::*;
use ag::prelude::*;
use ndarray::array;

ag::run(|g| {
    let a = convert_to_tensor(array![[2.0, 1.0], [1.0, 3.0]], g);
    let inv = matrix_inverse(&a);
    let det = determinant(&a);
    println!("Inverse: {:?}", inv.eval(g).unwrap());
    println!("Determinant: {}", det.eval(g).unwrap()[[]]);
});
```

### Gradient Computation
```rust
ag::run(|g| {
    let a = variable(array![[2.0, 1.0], [1.0, 3.0]], g);
    let det = determinant(&a);
    let grads = grad(&[&det], &[&a]);
    println!("Gradient of det w.r.t. A: {:?}", grads[0].eval(g).unwrap());
});
```

### Solving Linear Systems
```rust
ag::run(|g| {
    let a = convert_to_tensor(array![[3.0, 1.0], [1.0, 2.0]], g);
    let b = convert_to_tensor(array![[9.0], [8.0]], g);
    let x = solve(&a, &b);
    println!("Solution: {:?}", x.eval(g).unwrap());
});
```

### Matrix Decompositions
```rust
ag::run(|g| {
    let a = convert_to_tensor(array![[1.0, 2.0], [3.0, 4.0]], g);
    let (q, r) = qr(&a);
    let (u, s, v) = svd(&a);
    println!("QR decomposition - Q: {:?}, R: {:?}", 
             q.eval(g).unwrap(), r.eval(g).unwrap());
});
```

## Implementation Details

### Performance Considerations
- Uses ndarray-linalg for efficient numerical computations
- Leverages BLAS/LAPACK when available
- Gradient computations use implicit differentiation for efficiency

### Numerical Stability
- LU decomposition uses partial pivoting
- QR uses modified Gram-Schmidt
- SVD handles rank-deficient matrices
- Cholesky checks for positive definiteness

### Error Handling
- Operations check matrix dimensions
- Appropriate errors for singular matrices
- Validation of positive definiteness for Cholesky
- Checks for convergence in iterative algorithms

## Testing

Comprehensive test suite includes:
- Unit tests for each operation
- Gradient verification tests
- Numerical accuracy tests
- Integration tests with complex pipelines

## Future Enhancements

Potential additions:
- Complex number support
- Sparse matrix operations
- Banded matrix solvers
- Iterative solvers for large systems
- Specialized eigenvalue algorithms
- Matrix functions via Padé approximation