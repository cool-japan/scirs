# Matrix Norm Gradient Calculations

This document provides technical details for implementing accurate and numerically stable gradient calculations for matrix norms in the scirs2-autograd module. It serves as a guide for resolving issue #42.

## Current Status

The current implementation has the following limitations:

1. **Frobenius Norm**: Gradient calculation sometimes produces zeros instead of the correct gradients.
2. **Spectral Norm**: Gradient computation works only for specific test cases.
3. **Nuclear Norm**: Gradient computation uses approximations that aren't accurate for all matrices.

## Mathematical Background

### Frobenius Norm

The Frobenius norm of a matrix A is defined as:

$$\|A\|_F = \sqrt{\sum_{i,j} |A_{ij}|^2}$$

The gradient of the Frobenius norm with respect to each element is:

$$\frac{\partial \|A\|_F}{\partial A_{ij}} = \frac{A_{ij}}{\|A\|_F}$$

### Spectral Norm

The spectral norm of a matrix A is its largest singular value:

$$\|A\|_2 = \sigma_1(A)$$

The gradient of the spectral norm requires computing the left and right singular vectors associated with the largest singular value:

$$\frac{\partial \|A\|_2}{\partial A} = u_1 v_1^T$$

where $u_1$ and $v_1$ are the left and right singular vectors corresponding to the largest singular value.

### Nuclear Norm

The nuclear norm of a matrix A is the sum of its singular values:

$$\|A\|_* = \sum_i \sigma_i(A)$$

The gradient of the nuclear norm is:

$$\frac{\partial \|A\|_*}{\partial A} = UV^T$$

where $U$ and $V$ are the matrices of left and right singular vectors corresponding to non-zero singular values.

## Implementation Guidelines

### 1. Frobenius Norm Gradient

Improve the current implementation with these enhancements:

- Fix the numerical stability issues by ensuring division by zero is properly handled
- Use appropriate epsilon values for stability
- Consider using elementwise operations in tensor form rather than evaluating to arrays
- Implement comprehensive tests with various matrix types and shapes

```rust
// Recommended implementation structure
fn grad(&self, ctx: &mut GradientContext<F>) {
    let grad_output = ctx.output_grad();
    let input = ctx.input(0);
    let output = ctx.output();
    let g = ctx.graph();
    
    // Compute the gradient using tensor operations
    let safe_norm = crate::tensor_ops::maximum(
        output,
        crate::tensor_ops::scalar(F::epsilon() * F::from(10.0).unwrap(), g)
    );
    
    // Gradient is input / norm * grad_output
    let grad_input = crate::tensor_ops::mul(
        &crate::tensor_ops::div(&input, &safe_norm),
        &grad_output
    );
    
    ctx.append_input_grad(0, Some(grad_input));
}
```

### 2. Spectral Norm Gradient

Implement a properly differentiable spectral norm using automatic differentiation through SVD:

- Use the existing SVD implementation in the library with gradient tracking
- Compute the proper gradient for both general and special-case matrices
- Handle edge cases like repeated singular values
- Optimize performance for large matrices

```rust
// Recommended approach
fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
    let input = ctx.input(0);
    
    // Use SVD to compute singular values and vectors
    let (u, s, v) = svd_with_grad(&input, g);
    
    // Extract the largest singular value (first element of s)
    let sigma_max = s.slice(ndarray::s![0]);
    
    // Store u and v vectors for gradient computation
    ctx.set_extra("u", u.slice(ndarray::s![.., 0]));
    ctx.set_extra("v", v.slice(ndarray::s![.., 0]));
    
    ctx.append_output(sigma_max);
    Ok(())
}

fn grad(&self, ctx: &mut GradientContext<F>) {
    // Retrieve the stored singular vectors
    let u = ctx.get_extra::<Array1<F>>("u").unwrap();
    let v = ctx.get_extra::<Array1<F>>("v").unwrap();
    
    // Gradient is outer product u*v^T scaled by gradient of output
    let grad_output = ctx.output_grad();
    
    // Compute gradient
    let grad = outer_product(&u, &v, grad_output);
    
    ctx.append_input_grad(0, Some(grad));
}
```

### 3. Nuclear Norm Gradient

Implement accurate nuclear norm gradient calculation:

- Use a full SVD implementation with proper gradient backpropagation
- For diagonal matrices, use the sign matrix as gradient (as already implemented)
- Handle different matrix shapes correctly
- Optimize for performance on large matrices using truncated SVD where appropriate

```rust
// Recommended approach
fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
    let input = ctx.input(0);
    
    // Check for diagonal matrix case first (as a fast path)
    if is_diagonal_matrix(&input) {
        let sum_abs_diag = sum_of_absolute_diagonal(&input);
        ctx.append_output(sum_abs_diag);
        return Ok(());
    }
    
    // For general matrices, use SVD
    let (_, s, _) = svd_with_grad(&input, g);
    
    // Sum of singular values
    let nuclear_norm = sum(&s);
    
    ctx.append_output(nuclear_norm);
    Ok(())
}

fn grad(&self, ctx: &mut GradientContext<F>) {
    let input = ctx.input(0);
    
    // For diagonal matrices, gradient is the sign matrix
    if is_diagonal_matrix(&input) {
        let sign_matrix = sign_matrix_of_diagonal(&input);
        let grad = mul(&sign_matrix, &ctx.output_grad());
        ctx.append_input_grad(0, Some(grad));
        return;
    }
    
    // For general matrices, use SVD gradient
    let (u, s, v) = svd_with_grad(&input, g);
    
    // Find non-zero singular values
    let non_zero_mask = s.mapv(|x| if x > F::epsilon() { F::one() } else { F::zero() });
    
    // Compute the gradient as UV^T where only singular vectors corresponding
    // to non-zero singular values contribute
    let grad = compute_nuclear_norm_gradient(&u, &v, &non_zero_mask, &ctx.output_grad());
    
    ctx.append_input_grad(0, Some(grad));
}
```

## Performance Considerations

1. **Memory Usage**:
   - Avoid unnecessary copies and large intermediate matrices
   - Use in-place operations where possible
   - Consider using chunked operations for large matrices

2. **Computational Efficiency**:
   - Use fast SVD algorithms for large matrices
   - Consider truncated SVD for nuclear norm when matrices are large
   - Implement specialized fast paths for common matrix types (diagonal, symmetric, etc.)

3. **Numerical Stability**:
   - Use appropriate epsilon values for division operations
   - Handle edge cases like zero norms or repeated singular values correctly
   - Normalize vectors in power iteration to prevent overflow/underflow

## Testing Strategy

1. **Unit Tests**:
   - Test with various matrix shapes and types
   - Include tests for diagonal, symmetric, sparse, and random matrices
   - Test with edge cases (very small values, large condition numbers)

2. **Gradient Verification**:
   - Use finite difference method to verify gradients
   - Test gradient flow through compositions of operations
   - Check gradient consistency at various scales

3. **Benchmark Tests**:
   - Compare performance with naive implementations
   - Measure scaling behavior with matrix size
   - Evaluate memory usage patterns

## Integration with Existing Code

1. **Tensor Operations**:
   - Use existing tensor operations where possible for gradient calculations
   - Ensure compatibility with the autograd system

2. **Linear Algebra Support**:
   - Leverage the existing SVD implementation with gradient support
   - Use optimized BLAS routines where appropriate

## References

1. Matrix Calculus:
   - Magnus, J. R., & Neudecker, H. (2019). Matrix differential calculus with applications in statistics and econometrics.

2. SVD Gradients:
   - Ionescu, C., Vantzos, O., & Sminchisescu, C. (2015). Matrix backpropagation for deep networks with structured layers.
   - Murray, I. (2016). Differentiation of the Cholesky decomposition.

3. Numerical Stability:
   - Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.