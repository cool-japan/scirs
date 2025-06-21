# Enhanced Linear Algebra Features

## Overview
This document describes the enhanced linear algebra operations added to scirs2-autograd beyond the initial implementation.

## New Features

### 1. Matrix Norms (`src/tensor_ops/matrix_norms.rs`)
- **norm1**: 1-norm (maximum column sum)
- **norm2**: 2-norm/spectral norm (using power iteration)
- **norminf**: Infinity norm (maximum row sum)
- **normfro**: Frobenius norm

### 2. Symmetric Matrix Operations (`src/tensor_ops/symmetric_ops.rs`)
- **eigh**: Eigendecomposition for symmetric matrices
  - Uses Jacobi rotation method for numerical stability
  - Returns sorted eigenvalues and orthonormal eigenvectors
- **eigvalsh**: Eigenvalues only for symmetric matrices

### 3. Advanced Matrix Functions (`src/tensor_ops/matrix_ops.rs`)
- **expm2**: Matrix exponential using Padé approximation
  - Order 6 approximation for accuracy
  - Handles general matrices
- **expm3**: Matrix exponential using eigendecomposition
  - More efficient for diagonalizable matrices
  - Uses exp(A) = V * exp(D) * V^(-1)

### 4. Matrix Equation Solvers (`src/tensor_ops/matrix_solvers.rs`)
- **solve_sylvester**: Solves AX + XB = C
  - Uses Bartels-Stewart algorithm
  - Essential for control theory applications
- **solve_lyapunov**: Solves AX + XA^T = Q
  - Special case of Sylvester equation
  - Used in stability analysis
- **cholesky_solve**: Solves Ax = b for positive definite A
  - More efficient than general solve
  - Uses Cholesky decomposition

### 5. Special Decompositions (`src/tensor_ops/special_decompositions.rs`)
- **polar**: Polar decomposition A = UP
  - U is unitary/orthogonal
  - P is positive semidefinite
  - Uses iterative Newton method
- **schur**: Schur decomposition A = QTQ^T
  - Q is orthogonal
  - T is quasi-triangular
  - Real Schur form implementation

### 6. Advanced Tensor Operations (`src/tensor_ops/advanced_tensor_ops.rs`)
- **tensor_solve**: Solves tensor equations Ax = b
  - Handles higher-dimensional linear systems
  - Flattens and solves using matrix methods
- **einsum**: Einstein summation notation
  - Supports common contractions (matrix multiply, trace, etc.)
  - Flexible tensor operations

### 7. Enhanced Kronecker Product
- Moved to `advanced_tensor_ops.rs` and aliased as `kronecker_product`
- Full gradient support maintained

## Usage Examples

### Matrix Norms
```rust
let a = convert_to_tensor(array![[1.0, 2.0], [3.0, 4.0]], g);
let n1 = norm1(&a);      // Maximum column sum
let n2 = norm2(&a);      // Spectral norm
let ninf = norminf(&a);  // Maximum row sum
let nfro = normfro(&a);  // Frobenius norm
```

### Symmetric Eigendecomposition
```rust
let sym = convert_to_tensor(array![[4.0, 1.0], [1.0, 3.0]], g);
let (eigenvals, eigenvecs) = eigh(&sym);
```

### Matrix Exponential
```rust
let a = convert_to_tensor(array![[0.0, 1.0], [-1.0, 0.0]], g);
let exp_pade = expm2(&a);    // Padé approximation
let exp_eigen = expm3(&a);   // Eigendecomposition method
```

### Solving Equations
```rust
// Positive definite system
let a = convert_to_tensor(array![[4.0, 1.0], [1.0, 3.0]], g);
let b = convert_to_tensor(array![1.0, 2.0], g);
let x = cholesky_solve(&a, &b);

// Sylvester equation
let c = convert_to_tensor(array![[1.0, 1.0], [0.0, 1.0]], g);
let x = solve_sylvester(&a, &b, &c);
```

### Einstein Summation
```rust
let a = convert_to_tensor(array![[1.0, 2.0], [3.0, 4.0]], g);
let b = convert_to_tensor(array![[5.0, 6.0], [7.0, 8.0]], g);

// Matrix multiplication: C_ik = A_ij * B_jk
let c = einsum("ij,jk->ik", &[&a, &b]);

// Trace: sum of diagonal
let trace = einsum("ii->", &[&a]);

// Outer product: C_ij = a_i * b_j
let outer = einsum("i,j->ij", &[&a_vec, &b_vec]);
```

## Performance Considerations

1. **Matrix Norms**: O(n²) complexity, efficient implementations
2. **Symmetric Eigendecomposition**: O(n³) using Jacobi method, accurate for small/medium matrices
3. **Matrix Exponential**: 
   - Padé: O(n³), good for general matrices
   - Eigendecomposition: O(n³), better for diagonalizable matrices
4. **Equation Solvers**: O(n³) complexity, specialized for each equation type
5. **Polar Decomposition**: Iterative, typically converges in 5-10 iterations
6. **Einstein Summation**: Performance depends on contraction pattern

## Known Limitations

1. **Gradient Computation**: The base `grad` function still returns scalars instead of properly shaped gradients
2. **Numerical Accuracy**: Some algorithms use simplified implementations suitable for small to medium matrices
3. **Sparse Support**: Not yet implemented, all operations assume dense matrices
4. **GPU Acceleration**: Not available, CPU-only implementations

## Testing

All enhanced features have comprehensive unit tests in `tests/enhanced_linalg_test.rs`:
- 11 test cases covering all new operations
- Tests for correctness and edge cases
- Gradient computation tests (adapted for known issues)

## Future Enhancements

1. **High Priority**:
   - Fix gradient shape computation
   - Add Cholesky decomposition
   - Implement generalized eigenvalue problems

2. **Medium Priority**:
   - Add more matrix functions (logm improvements, funm)
   - Implement tensor decompositions (Tucker, CP)
   - Add sparse matrix support foundations

3. **Low Priority**:
   - GPU acceleration via compute shaders
   - Parallel implementations using rayon
   - Advanced numerical algorithms (LAPACK bindings)