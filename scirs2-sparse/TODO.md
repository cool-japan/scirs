# scirs2-sparse TODO

## Release Status: 0.1.0-alpha.6 (Final Alpha)

This is the final alpha release before the stable 1.0.0 release. The module provides comprehensive sparse matrix functionality with feature parity to SciPy's sparse module in key areas.

## Implemented Features ✅

### Core Sparse Matrix Formats
- **CSR (Compressed Sparse Row)** - Efficient row-wise operations
- **CSC (Compressed Sparse Column)** - Efficient column-wise operations  
- **COO (Coordinate)** - Simple triplet format, easy construction
- **DOK (Dictionary of Keys)** - Efficient incremental construction
- **LIL (List of Lists)** - Efficient row-wise incremental construction
- **DIA (Diagonal)** - Memory-efficient diagonal matrix storage
- **BSR (Block Sparse Row)** - Block-structured sparse matrices

### Matrix Operations
- Format conversions between all sparse formats
- Basic arithmetic operations (add, subtract, multiply, divide)
- Matrix multiplication (sparse-sparse, sparse-dense, sparse-vector)
- Transpose and conjugate transpose
- Element-wise operations (Hadamard product)
- Matrix norms (Frobenius, 1-norm, 2-norm, infinity norm)

### Construction Utilities
- Identity matrices (`eye`)
- Diagonal matrices (`diags`)
- Random sparse matrices (`random`)
- Kronecker products (`kron`)
- Kronecker sums (`kronsum`)
- Block matrices (`bmat`)
- Block diagonal matrices (`block_diag`)
- Matrix stacking (`hstack`, `vstack`)
- Triangular extraction (`tril`, `triu`)

### Specialized Formats
- Symmetric sparse matrices (SymCsrMatrix, SymCooMatrix)
- Enhanced index dtype handling with automatic optimization
- Safe index casting utilities

### Linear Algebra
- **Iterative Solvers**: CG, BiCG, BiCGSTAB, CGS, GMRES, LGMRES, MINRES, QMR
- **Preconditioners**: Jacobi, SSOR, Incomplete Cholesky (IC), Incomplete LU (ILU), SPAI
- **Matrix Functions**: Matrix exponential (expm), matrix powers, expm_multiply
- **Linear Operators**: Abstract linear operator interface with composition support

### Array-based API
- Modern array-focused API similar to SciPy 1.13+
- Support for NumPy-like array semantics
- Consistent element-wise and matrix multiplication operators

## Planned Features for 1.0.0 Release 📋

### High Priority
- **Graph Algorithms (csgraph module)**
  - Shortest path algorithms (Dijkstra, Bellman-Ford)
  - Connected components analysis
  - Graph traversal utilities
  - Laplacian matrix computation

- **Advanced Linear Algebra**
  - Eigenvalue solvers (eigs, eigsh)
  - Singular Value Decomposition (SVD)
  - Matrix decompositions (LU, Cholesky with pivoting)
  - One-norm estimator and advanced norm computations

- **Performance Optimizations**
  - SIMD acceleration for key operations
  - Parallel implementations using Rayon
  - Memory usage optimizations
  - GPU acceleration (optional CUDA/ROCm bindings)

### Medium Priority
- **Enhanced Linear Operators**
  - Operator composition (addition, multiplication)
  - Matrix-free operator implementations
  - Custom operator support

- **Specialized Formats**
  - Additional symmetric sparse formats
  - Banded matrix formats with specialized solvers
  - Block diagonal formats

- **Advanced Solvers**
  - Least squares solvers (LSQR, LSMR)
  - Additional Krylov methods (GCROT, TFQMR)
  - Algebraic Multigrid (AMG) preconditioners

## Migration Guide

For users upgrading from earlier alpha versions:

1. **Array API Migration**: Prefer `_array` variants over legacy matrix formats
2. **Import Changes**: Use `scirs2_sparse::*` instead of individual format imports
3. **Error Handling**: Update error handling to use the new `SparseResult` type
4. **Deprecated Features**: Remove usage of deprecated matrix-specific operators

## Testing and Quality Assurance

All implemented features have been thoroughly tested with:
- ✅ Unit tests for individual components
- ✅ Integration tests for complex workflows  
- ✅ Numerical accuracy tests against SciPy reference implementations
- ✅ Performance benchmarks and regression tests
- ✅ Memory leak detection and profiling
- ✅ Cross-platform compatibility testing

## Production Readiness

This alpha release is production-ready for:
- ✅ Core sparse matrix operations and formats
- ✅ Basic linear algebra computations
- ✅ Iterative solver applications
- ✅ Format conversions and data manipulation
- ✅ Integration with other scirs2 modules

**Note**: Advanced features like graph algorithms and specialized solvers are planned for the 1.0.0 stable release.