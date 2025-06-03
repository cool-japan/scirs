# scirs2-autograd Linear Algebra Quick Reference

## Basic Usage

```rust
use scirs2_autograd as ag;
use ag::tensor_ops::*;
use ag::prelude::*;

ag::run(|ctx| {
    // Your code here
});
```

## Matrix Creation
```rust
let identity = eye(3, ctx);                    // 3x3 identity matrix
let diag_mat = diag(&vector);                  // Diagonal matrix from vector
let diag_vec = extract_diag(&matrix);          // Extract diagonal elements
```

## Basic Operations
```rust
let tr = trace(&a);                            // Matrix trace
let det = determinant(&a);                     // Determinant
let norm = frobenius_norm(&a);                 // Frobenius norm
let scaled = scalar_mul(&a, 2.0);              // Scalar multiplication
// or: let scaled = a.scalar_mul(2.0);
```

## Element-wise vs Matrix Operations
```rust
let elem_inv = inv(&a);                        // Element-wise 1/x
let mat_inv = matrix_inverse(&a);              // Matrix inverse
let elem_sqrt = sqrt(&a);                      // Element-wise sqrt
let mat_sqrt = matrix_sqrt(&a);                // Matrix square root
```

## Matrix Decompositions
```rust
let (q, r) = qr(&a);                          // QR decomposition
let (l, u, p) = lu(&a);                       // LU with pivoting
let (u, s, v) = svd(&a);                      // Singular value decomposition
let chol = cholesky(&pd_matrix);              // Cholesky (positive definite)
let (vals, vecs) = eigen(&a);                 // Eigendecomposition
let vals_only = eigenvalues(&a);              // Eigenvalues only
```

## Linear System Solving
```rust
let x = solve(&a, &b);                        // Solve Ax = b
let x_ls = lstsq(&a, &b);                     // Least squares solution
```

## Matrix Functions
```rust
let exp_a = matrix_exp(&a);                   // Matrix exponential
let log_a = matrix_log(&a);                   // Matrix logarithm
let sqrt_a = matrix_sqrt(&a);                 // Matrix square root
let pow_a = matrix_pow(&a, 2.5);              // Matrix power
```

## Special Matrix Operations
```rust
let sym = symmetrize(&a);                     // Make symmetric: (A + A^T)/2
let lower = tril(&a, 0);                      // Lower triangular
let upper = triu(&a, 0);                      // Upper triangular
let band = band_matrix(&a, 1, 2);             // Band matrix
```

## Working with Gradients
```rust
let a = variable(array![[2.0, 1.0], [1.0, 3.0]], ctx);
let det = determinant(&a);
let grads = grad(&[&det], &[&a]);
let grad_a = &grads[0];
```

## Complete Example
```rust
use ndarray::array;
use scirs2_autograd as ag;
use ag::tensor_ops::*;
use ag::prelude::*;

fn main() {
    ag::run(|ctx| {
        // Create matrices
        let a = variable(array![[3.0, 1.0], [1.0, 2.0]], ctx);
        let b = convert_to_tensor(array![[5.0], [3.0]], ctx);
        
        // Solve linear system
        let x = solve(&a, &b);
        
        // Compute determinant
        let det = determinant(&a);
        
        // QR decomposition
        let (q, r) = qr(&a);
        
        // Create loss function
        let loss = sum_all(&square(&sub(&matmul(&a, &x), &b))) + square(&det);
        
        // Compute gradients
        let grads = grad(&[&loss], &[&a]);
        
        // Evaluate
        println!("Solution: {:?}", x.eval(ctx).unwrap());
        println!("Determinant: {:?}", det.eval(ctx).unwrap());
        println!("Gradient: {:?}", grads[0].eval(ctx).unwrap());
    });
}
```

## Performance Tips

1. Use Cholesky decomposition for positive definite matrices
2. Consider using specialized operations for diagonal/triangular matrices
3. For large matrices, some operations (SVD, eigen) may be expensive
4. Batch operations when possible to reduce graph construction overhead
5. Use BLAS features for better performance:
   ```toml
   scirs2-autograd = { version = "*", features = ["intel-mkl"] }
   ```

## Error Handling

All operations return `Result` types. Common errors:
- Non-square matrices for operations requiring square input
- Non-positive definite matrices for Cholesky
- Singular matrices for inverse operations
- Dimension mismatches for matrix operations

## Feature Flags

Include in Cargo.toml for optimized performance:
```toml
[dependencies]
scirs2-autograd = { version = "*", features = ["intel-mkl"] }
# or
scirs2-autograd = { version = "*", features = ["openblas"] }
# or (macOS)
scirs2-autograd = { version = "*", features = ["accelerate"] }
```