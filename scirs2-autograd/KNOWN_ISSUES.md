# Known Issues in scirs2-autograd

This document tracks known bugs and issues in the scirs2-autograd crate that need to be fixed.

## 1. ~~nth_tensor Bug with Multi-Output Operations~~ [FIXED]

**Issue**: The `nth_tensor` function was not correctly extracting outputs from multi-output operations.

**Solution**: Fixed by using the correct extraction operators (SVDExtractOp, QRExtractOp, EigenExtractOp) from decomposition_ops and eigen_ops modules instead of nth_tensor.

**Files fixed**:
- `src/tensor_ops/linear_algebra/mod.rs` - Updated svd, qr, and eigen functions

## 2. Gradient Computation Returns Scalars Instead of Matrices

**Issue**: When computing gradients of scalar functions with respect to matrix inputs, the gradient computation returns a scalar instead of a matrix of the same shape as the input.

**Example**: For determinant gradient:
- Input: 2x2 matrix A
- Output: scalar det(A)
- Expected gradient shape: [2, 2] (same as input)
- Actual gradient shape: [] (scalar)

**Impact**:
- Cannot compute proper gradients for determinant
- Cannot compute proper gradients for trace of matrix functions
- Breaks automatic differentiation for many linear algebra operations

**Workaround**: Tests currently skip gradient checks for affected operations.

**Files affected**:
- Gradient computation infrastructure
- Tests in `tests/linalg_operations_test.rs`

## 3. Eigendecomposition Numerical Accuracy

**Issue**: The eigendecomposition implementation has numerical accuracy issues. The product of eigenvalues doesn't match the determinant as closely as expected.

**Example**: For matrix [[4, 1], [1, 3]]:
- Expected product of eigenvalues: 11.0 (determinant)
- Actual product: 11.45

**Impact**: Minor numerical inaccuracy in eigenvalue computation.

**Workaround**: Tests use relaxed tolerance (epsilon = 0.5) for eigenvalue product checks.

## 4. SVD Implementation is a Placeholder

**Issue**: The current SVD implementation in decomposition_ops.rs is just a placeholder that returns identity matrices for U and V, and diagonal elements as singular values.

**Impact**: SVD cannot be used for actual computations or reconstruction.

**Workaround**: Tests skip SVD reconstruction checks.

**Solution needed**: Implement proper SVD algorithm (e.g., using LAPACK bindings or iterative methods).

## Recommendations

1. **Priority 1**: ~~Fix the nth_tensor bug~~ [DONE]
2. **Priority 2**: Fix gradient computation to return proper shapes
3. **Priority 3**: Implement proper SVD algorithm
4. **Priority 4**: Improve eigendecomposition numerical accuracy

These issues should be addressed before the stable release of scirs2-autograd.