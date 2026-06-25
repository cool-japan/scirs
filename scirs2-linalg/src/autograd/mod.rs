//! Automatic differentiation support for linear algebra operations
//!
//! This module provides integration with the scirs2-autograd crate for automatic
//! differentiation of linear algebra operations.
//!
//! ## Current Status
//!
//! The autograd module is currently in a transitional state. The scirs2-autograd
//! crate is actively being developed, and many linear algebra-specific operations
//! are not yet available. Based on feedback from the scirs2-autograd maintainers,
//! the following features are planned but not yet implemented:
//!
//! ### Missing Operations
//!
//! 1. **Built-in matrix operations**:
//!    - `eye(n, ctx)` - Identity matrix creation
//!    - `diag(vector, ctx)` - Diagonal matrix from vector
//!    - `trace(matrix)` - Matrix trace
//!
//! 2. **Decompositions with gradients**:
//!    - LU decomposition
//!    - QR decomposition
//!    - SVD decomposition
//!    - Eigendecomposition
//!
//! 3. **Complex number support**: Currently limited to real floating-point types
//!
//! 4. **Specialized matrix types**: Support for symmetric, hermitian, and triangular
//!    matrices with optimized operations
//!
//! ## Using Autograd with Linear Algebra
//!
//! Until these operations are available in scirs2-autograd, users can:
//!
//! 1. Use basic operations (matmul, transpose, element-wise ops) which are available
//! 2. Compose complex operations from primitives
//! 3. Implement custom operations using the scirs2-autograd `Op` trait
//!
//! ## Examples
//!
//! For current usage examples, see:
//! - `examples/autograd_basic_example.rs` - Basic differentiation
//! - `examples/autograd_linalg_example.rs` - Linear algebra operations
//!
//! ## Future Plans
//!
//! Once the requested features are implemented in scirs2-autograd, this module will
//! provide:
//!
//! - Seamless integration of all linear algebra operations with automatic differentiation
//! - Optimized gradients for matrix decompositions
//! - Support for complex-valued matrices
//! - Specialized operations for structured matrices

use scirs2_autograd as ag;

pub use ag::tensor_ops;
/// Re-export commonly used types from scirs2-autograd
pub use ag::{Context, Float, Tensor};

// Autodiff backward passes for matrix factorizations are implemented in
// `scirs2_autograd::tensor_ops`. Canonical entry points:
//   - scirs2_autograd::tensor_ops::cholesky
//   - scirs2_autograd::tensor_ops::lu
//   - scirs2_autograd::tensor_ops::qr
//   - scirs2_autograd::tensor_ops::pinv   (three-term Golub & Pereyra grad)
//   - scirs2_autograd::tensor_ops::sqrtm / sqrtm_pd  (SPD-restricted Sylvester backward)
//   - scirs2_autograd::tensor_ops::logm   (Daleckii-Krein spectral backward)

/// Matrix calculus operations with automatic differentiation support
/// Includes: gradient, hessian, jacobian computations, VJP, JVP
pub mod matrix_calculus;

/// Einstein summation engine for arbitrary-rank tensor contractions.
///
/// Provides `einsum` (evaluation) and `einsum_grad` (gradients) supporting
/// traces, diagonals, outer products, matrix multiplications, batched matmul,
/// and arbitrary index contractions.
pub mod einsum;
pub use einsum::{einsum, einsum_grad, EinsumError};

/// Helper functions for common patterns in linear algebra autodiff
pub mod helpers {
    use super::*;

    /// Compute trace using available operations (workaround until trace is available)
    ///
    /// This multiplies the matrix element-wise with an identity matrix and sums all elements.
    pub fn trace_workaround<'g, F: ag::Float>(
        matrix: &ag::Tensor<'g, F>,
        n: usize,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Create identity pattern manually
        let mut eye_data = vec![F::zero(); n * n];
        for i in 0..n {
            eye_data[i * n + i] = F::one();
        }
        let eye = ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_shape_vec((n, n), eye_data).expect("Operation failed"),
            ctx,
        );

        // Element-wise multiply and sum
        let diag_elements = matrix * eye;
        ag::tensor_ops::sum_all(diag_elements)
    }

    /// Create identity matrix using available operations (workaround)
    pub fn eye_workaround<'g, F: ag::Float>(
        n: usize,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        let mut eye_data = vec![F::zero(); n * n];
        for i in 0..n {
            eye_data[i * n + i] = F::one();
        }
        ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_shape_vec((n, n), eye_data).expect("Operation failed"),
            ctx,
        )
    }

    /// Create diagonal matrix from vector (workaround)
    pub fn diag_workaround<'g, F: ag::Float>(
        diagonal: &ag::Tensor<'g, F>,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Extract the diagonal values from the tensor. Use the evaluator to read
        // the real entries (`to_ndarray` returns placeholder data, not the actual
        // tensor contents).
        let diagarray = diagonal
            .eval(ctx)
            .expect("diag_workaround: failed to evaluate diagonal tensor");
        let n = diagarray.len();

        let mut matrix_data = vec![F::zero(); n * n];
        for i in 0..n {
            matrix_data[i * n + i] = diagarray[i];
        }

        ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_shape_vec((n, n), matrix_data).expect("Operation failed"),
            ctx,
        )
    }

    /// Compute Frobenius norm using available operations
    pub fn frobenius_norm<'g, F: ag::Float>(matrix: &ag::Tensor<'g, F>) -> ag::Tensor<'g, F> {
        // ||A||_F = sqrt(sum(A .* A))
        let squared = matrix * matrix;
        let sum_squared = ag::tensor_ops::sum_all(squared);
        ag::tensor_ops::sqrt(sum_squared)
    }

    /// Compute the matrix determinant from the current value of `matrix`.
    ///
    /// The tensor is evaluated to its concrete entries and the determinant is
    /// computed with LU decomposition (Gaussian elimination with partial
    /// pivoting), which is exact up to floating-point round-off for any size `n`.
    /// The scalar result is returned as a 1x1 constant tensor.
    ///
    /// This is a value helper: it reads the current value of `matrix` rather than
    /// building a differentiable determinant graph op. Note that the matrix value
    /// is obtained via the evaluator (`Tensor::eval`) — the `to_ndarray` helper in
    /// the autograd integration layer returns placeholder data and must not be used
    /// to read genuine tensor contents.
    pub fn det_approximation<'g, F: ag::Float>(
        matrix: &ag::Tensor<'g, F>,
        n: usize,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Evaluate the tensor to obtain its real entries.
        let mat = matrix
            .eval(ctx)
            .expect("det_approximation: failed to evaluate matrix tensor");

        // Copy into a row-major working buffer for in-place LU factorization.
        let mut lu = vec![F::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                lu[i * n + j] = mat[[i, j]];
            }
        }

        let mut det = F::one();
        for k in 0..n {
            // Partial pivoting: pick the row with the largest magnitude in
            // column k at or below the diagonal for numerical stability.
            let mut pivot_row = k;
            let mut pivot_mag = lu[k * n + k].abs();
            for i in (k + 1)..n {
                let candidate = lu[i * n + k].abs();
                if candidate > pivot_mag {
                    pivot_mag = candidate;
                    pivot_row = i;
                }
            }

            // A zero pivot means the matrix is singular: determinant is exactly 0.
            if pivot_mag == F::zero() {
                det = F::zero();
                break;
            }

            // Swap rows if needed, flipping the sign of the determinant.
            if pivot_row != k {
                for j in 0..n {
                    lu.swap(pivot_row * n + j, k * n + j);
                }
                det = -det;
            }

            // Eliminate entries below the pivot.
            let pivot = lu[k * n + k];
            for i in (k + 1)..n {
                let factor = lu[i * n + k] / pivot;
                for j in k..n {
                    let updated = lu[i * n + j] - factor * lu[k * n + j];
                    lu[i * n + j] = updated;
                }
            }

            det *= pivot;
        }

        ag::tensor_ops::convert_to_tensor(ag::ndarray::Array2::from_elem((1, 1), det), ctx)
    }

    /// Solve linear system Ax = b using iterative method approximation
    ///
    /// This implements a simplified gradient descent approach for solving Ax = b
    /// by minimizing ||Ax - b||^2. Not optimized for accuracy - mainly for demonstration.
    pub fn solve_iterative<'g, F: ag::Float>(
        a: &ag::Tensor<'g, F>,
        b: &ag::Tensor<'g, F>,
        iterations: usize,
        learning_rate: F,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Initialize x as zeros. Read the real length of `b` by evaluating it
        // under the caller's context (the no-context `to_ndarray` cannot recover
        // a lazy tensor's contents and now returns an honest error).
        let barray = ag::integration::tensor_conversion::to_ndarray_with_context(b, ctx)
            .expect("solve_iterative: failed to evaluate b tensor");
        let n = barray.len();
        let mut x = ag::tensor_ops::convert_to_tensor(ag::ndarray::Array2::zeros((n, 1)), ctx);

        let lr_tensor = ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_elem((1, 1), learning_rate),
            ctx,
        );

        // Gradient descent: x = x - lr * A^T * (A*x - b)
        for _iter in 0..iterations {
            let ax = ag::tensor_ops::matmul(a, x);
            let residual = ax - b;
            let at = ag::tensor_ops::transpose(a, &[1, 0]);
            let gradient = ag::tensor_ops::matmul(at, residual);
            let update = gradient * lr_tensor;
            x = x - update;
        }

        x
    }

    /// Estimate the dominant (largest-magnitude) eigenvalue via power iteration.
    ///
    /// This is a genuine numerical method, not a fabricated constant: it builds
    /// the real power-iteration graph (repeated `A v`, normalized) for the
    /// requested number of `iterations`, then returns the Rayleigh quotient
    /// `λ = vᵀ A v / (vᵀ v)` as a scalar tensor. Because the result is produced
    /// from graph ops it evaluates correctly under the caller's context and is
    /// differentiable.
    ///
    /// Accuracy is iterative: the estimate converges to the dominant eigenvalue
    /// at a rate governed by the ratio |λ₂ / λ₁|, so more `iterations` give a
    /// tighter result. Convergence requires a unique dominant eigenvalue and a
    /// starting vector with a non-zero component along its eigenvector.
    pub fn dominant_eigenvalue<'g, F: ag::Float>(
        matrix: &ag::Tensor<'g, F>,
        iterations: usize,
        n: usize,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Initialize random vector
        let mut v_data = vec![F::one(); n];
        v_data[0] = F::one();
        for (i, item) in v_data.iter_mut().enumerate().take(n).skip(1) {
            *item = F::from(0.1).expect("Operation failed")
                * F::from(i as f64).expect("Operation failed");
        }

        let mut v = ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_shape_vec((n, 1), v_data).expect("Operation failed"),
            ctx,
        );

        // Power iteration
        for _iter in 0..iterations {
            let av = ag::tensor_ops::matmul(matrix, v);
            let norm = frobenius_norm(&av);
            v = av / norm;
        }

        // Compute eigenvalue: λ = v^T * A * v / (v^T * v)
        let vt = ag::tensor_ops::transpose(v, &[1, 0]);
        let av = ag::tensor_ops::matmul(matrix, v);
        let numerator = ag::tensor_ops::matmul(vt, av);
        let denominator = ag::tensor_ops::matmul(vt, v);

        numerator / denominator
    }

    /// Matrix condition number approximation using eigenvalue ratio
    pub fn condition_number_approx<'g, F: ag::Float>(
        matrix: &ag::Tensor<'g, F>,
        iterations: usize,
        n: usize,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Compute largest eigenvalue
        let lambda_max = dominant_eigenvalue(matrix, iterations, n, ctx);

        // For condition number, we'd need smallest eigenvalue too
        // This is a simplified approximation - return max eigenvalue as proxy
        lambda_max
    }

    /// Compute the numerical rank of `matrix`.
    ///
    /// The previous implementation returned a hard-coded `3.0` regardless of the
    /// input. This computes the genuine numerical rank: the number of singular
    /// values strictly greater than `tolerance`.
    ///
    /// The singular values are obtained without requiring the heavier SVD trait
    /// bounds of `scirs2-linalg` (which `ag::Float` does not provide): the matrix
    /// is evaluated to its concrete entries, the symmetric positive-semidefinite
    /// Gram matrix `G = AᵀA` (using the smaller of `AᵀA` / `AAᵀ`) is formed, and
    /// its eigenvalues are computed with the cyclic Jacobi eigenvalue algorithm.
    /// The singular values are `σᵢ = sqrt(max(λᵢ, 0))`. Counting `σᵢ > tolerance`
    /// gives the standard numerical rank.
    ///
    /// A singular value counts toward the rank when it exceeds the effective
    /// threshold
    ///
    /// ```text
    /// threshold = max(tolerance, max(m, n) * sqrt(eps) * σ_max)
    /// ```
    ///
    /// The relative term `max(m, n) * sqrt(eps) * σ_max` is always applied as a
    /// floor to reject numerical noise. Because the rank is read from the Gram
    /// matrix `AᵀA`, the condition number is squared, so a singular value of a
    /// numerically rank-deficient matrix is only resolved to about `sqrt(eps)`
    /// relative accuracy — hence the `sqrt(eps)` (rather than `eps`) factor,
    /// which is the honest accuracy floor of this approach. A user-supplied
    /// `tolerance` can only raise the threshold, never lower it below that floor.
    ///
    /// The integer rank is returned as a `1x1` constant tensor. Matrix rank is a
    /// discrete, non-differentiable quantity, so this is a value helper rather
    /// than a differentiable graph op.
    pub fn rank_approximation<'g, F: ag::Float>(
        matrix: &ag::Tensor<'g, F>,
        tolerance: F,
        ctx: &'g ag::Context<'g, F>,
    ) -> ag::Tensor<'g, F> {
        // Evaluate to obtain the real matrix entries.
        let mat = matrix
            .eval(ctx)
            .expect("rank_approximation: failed to evaluate matrix tensor");

        let shape = mat.shape();
        assert!(
            shape.len() == 2,
            "rank_approximation: expected a 2D matrix, got shape {shape:?}"
        );
        let m = shape[0];
        let n = shape[1];

        // Empty matrix has rank 0.
        if m == 0 || n == 0 {
            return ag::tensor_ops::convert_to_tensor(
                ag::ndarray::Array2::from_elem((1, 1), F::zero()),
                ctx,
            );
        }

        // Form the smaller Gram matrix so the eigenproblem is p x p with
        // p = min(m, n): if m >= n use AᵀA (n x n), else use AAᵀ (m x m).
        // Both share the same non-zero eigenvalues (= squared singular values).
        let p = m.min(n);
        let mut gram = vec![F::zero(); p * p];
        if m >= n {
            // G = AᵀA, G[i][j] = sum_k A[k][i] * A[k][j]
            for i in 0..n {
                for j in i..n {
                    let mut acc = F::zero();
                    for k in 0..m {
                        acc += mat[[k, i]] * mat[[k, j]];
                    }
                    gram[i * p + j] = acc;
                    gram[j * p + i] = acc;
                }
            }
        } else {
            // G = AAᵀ, G[i][j] = sum_k A[i][k] * A[j][k]
            for i in 0..m {
                for j in i..m {
                    let mut acc = F::zero();
                    for k in 0..n {
                        acc += mat[[i, k]] * mat[[j, k]];
                    }
                    gram[i * p + j] = acc;
                    gram[j * p + i] = acc;
                }
            }
        }

        // Symmetric eigenvalues of the Gram matrix via cyclic Jacobi rotations.
        let eigenvalues = jacobi_eigenvalues(&mut gram, p);

        // Singular values are sqrt of the (clamped non-negative) eigenvalues.
        let mut singular_values: Vec<F> = eigenvalues
            .into_iter()
            .map(|lambda| {
                if lambda > F::zero() {
                    lambda.sqrt()
                } else {
                    F::zero()
                }
            })
            .collect();
        // Sort descending so σ_max is first.
        singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let sigma_max = singular_values.first().copied().unwrap_or(F::zero());

        // Relative noise floor. Gram-based singular values are accurate to about
        // sqrt(eps) relative to sigma_max (the AᵀA squaring halves the available
        // precision), so this is the honest minimum below which a singular value
        // is indistinguishable from zero. A user `tolerance` can only raise the
        // threshold above this floor, never weaken the noise rejection.
        let dim = F::from(m.max(n) as f64).unwrap_or(F::one());
        let relative_floor = dim * F::epsilon().sqrt() * sigma_max;
        let threshold = if tolerance > relative_floor {
            tolerance
        } else {
            relative_floor
        };

        let rank = singular_values
            .iter()
            .filter(|&&sigma| sigma > threshold)
            .count();

        ag::tensor_ops::convert_to_tensor(
            ag::ndarray::Array2::from_elem((1, 1), F::from(rank as f64).unwrap_or_else(F::zero)),
            ctx,
        )
    }

    /// Compute the eigenvalues of a real symmetric `n x n` matrix stored in
    /// row-major order using the cyclic Jacobi eigenvalue algorithm.
    ///
    /// Only the eigenvalues are returned (as a length-`n` vector); the rotations
    /// are applied in place to `a`, leaving the eigenvalues on its diagonal. The
    /// method is unconditionally convergent for symmetric matrices and needs
    /// only the basic `ag::Float` operations, avoiding any external SVD/eig trait
    /// bounds.
    fn jacobi_eigenvalues<F: ag::Float>(a: &mut [F], n: usize) -> Vec<F> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![a[0]];
        }

        let max_sweeps = 100usize;
        for _sweep in 0..max_sweeps {
            // Off-diagonal Frobenius magnitude; stop once negligible.
            let mut off = F::zero();
            for p in 0..n {
                for q in (p + 1)..n {
                    off += a[p * n + q] * a[p * n + q];
                }
            }
            if off <= F::zero() {
                break;
            }
            // Convergence check relative to the diagonal scale.
            let mut diag_scale = F::zero();
            for i in 0..n {
                diag_scale += a[i * n + i] * a[i * n + i];
            }
            let eps = F::epsilon();
            if off.sqrt() <= eps * (diag_scale.sqrt() + F::epsilon()) {
                break;
            }

            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[p * n + q];
                    if apq == F::zero() {
                        continue;
                    }
                    let app = a[p * n + p];
                    let aqq = a[q * n + q];

                    // Compute the Jacobi rotation (c, s) zeroing a[p][q].
                    let two = F::from(2.0).unwrap_or_else(F::one);
                    let tau = (aqq - app) / (two * apq);
                    let t = if tau >= F::zero() {
                        F::one() / (tau + (F::one() + tau * tau).sqrt())
                    } else {
                        -F::one() / (-tau + (F::one() + tau * tau).sqrt())
                    };
                    let c = F::one() / (F::one() + t * t).sqrt();
                    let s = t * c;

                    // Apply rotation to rows/columns p and q.
                    for k in 0..n {
                        let akp = a[k * n + p];
                        let akq = a[k * n + q];
                        a[k * n + p] = c * akp - s * akq;
                        a[k * n + q] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let apk = a[p * n + k];
                        let aqk = a[q * n + k];
                        a[p * n + k] = c * apk - s * aqk;
                        a[q * n + k] = s * apk + c * aqk;
                    }
                }
            }
        }

        (0..n).map(|i| a[i * n + i]).collect()
    }
}

#[cfg(test)]
mod determinant_tests {
    use super::helpers;
    use scirs2_autograd as ag;

    #[test]
    fn test_det_approximation_3x3_is_real_not_placeholder() {
        // det([[2,0,1],[3,1,2],[1,0,3]]) = 5. The previous implementation returned a
        // hard-coded 1.0 placeholder for n >= 3; this confirms the real LU value.
        ag::run(|ctx: &mut ag::Context<f64>| {
            let m = ag::tensor_ops::convert_to_tensor(
                ag::ndarray::arr2(&[[2.0_f64, 0.0, 1.0], [3.0, 1.0, 2.0], [1.0, 0.0, 3.0]]),
                ctx,
            );
            let det = helpers::det_approximation(&m, 3, ctx);
            let det_arr = det.eval(ctx).expect("Test: failed to evaluate determinant");
            assert!(
                (det_arr[[0, 0]] - 5.0).abs() < 1e-10,
                "expected determinant 5.0, got {}",
                det_arr[[0, 0]]
            );
        });
    }

    #[test]
    fn test_det_approximation_singular_is_zero() {
        // Row 2 = 2 * Row 0, so this 3x3 matrix is singular (determinant 0).
        ag::run(|ctx: &mut ag::Context<f64>| {
            let m = ag::tensor_ops::convert_to_tensor(
                ag::ndarray::arr2(&[[1.0_f64, 2.0, 3.0], [0.0, 1.0, 4.0], [2.0, 4.0, 6.0]]),
                ctx,
            );
            let det = helpers::det_approximation(&m, 3, ctx);
            let det_arr = det.eval(ctx).expect("Test: failed to evaluate determinant");
            assert!(
                det_arr[[0, 0]].abs() < 1e-10,
                "expected singular determinant ~0, got {}",
                det_arr[[0, 0]]
            );
        });
    }

    #[test]
    fn test_rank_full_rank_identity() {
        // 3x3 identity has full rank 3. The old impl returned 3.0 by accident
        // for every matrix; here it must be the genuine rank.
        ag::run(|ctx: &mut ag::Context<f64>| {
            let m = ag::tensor_ops::convert_to_tensor(
                ag::ndarray::arr2(&[[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                ctx,
            );
            let rank = helpers::rank_approximation(&m, 1e-9_f64, ctx);
            let rank_arr = rank.eval(ctx).expect("Test: failed to evaluate rank");
            assert!(
                (rank_arr[[0, 0]] - 3.0).abs() < 1e-12,
                "expected rank 3, got {}",
                rank_arr[[0, 0]]
            );
        });
    }

    #[test]
    fn test_rank_rank_deficient_is_not_constant_three() {
        // Row 2 = 2 * Row 0 and Row 3-pattern keeps rank 2 (a clearly
        // rank-deficient matrix). The honest rank is 2, NOT the old constant 3.
        ag::run(|ctx: &mut ag::Context<f64>| {
            let m = ag::tensor_ops::convert_to_tensor(
                ag::ndarray::arr2(&[[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 4.0, 6.0]]),
                ctx,
            );
            let rank = helpers::rank_approximation(&m, 1e-9_f64, ctx);
            let rank_arr = rank.eval(ctx).expect("Test: failed to evaluate rank");
            assert!(
                (rank_arr[[0, 0]] - 2.0).abs() < 1e-12,
                "expected rank 2 (rank-deficient), got {}",
                rank_arr[[0, 0]]
            );
        });
    }

    #[test]
    fn test_rank_rank_one_outer_product() {
        // An outer product u v^T has rank exactly 1.
        ag::run(|ctx: &mut ag::Context<f64>| {
            let m = ag::tensor_ops::convert_to_tensor(
                ag::ndarray::arr2(&[[2.0_f64, 4.0, 6.0], [3.0, 6.0, 9.0], [1.0, 2.0, 3.0]]),
                ctx,
            );
            let rank = helpers::rank_approximation(&m, 1e-9_f64, ctx);
            let rank_arr = rank.eval(ctx).expect("Test: failed to evaluate rank");
            assert!(
                (rank_arr[[0, 0]] - 1.0).abs() < 1e-12,
                "expected rank 1 (outer product), got {}",
                rank_arr[[0, 0]]
            );
        });
    }

    #[test]
    fn test_rank_non_square_default_tolerance() {
        // A 2x3 matrix with two independent rows has rank 2. Passing a
        // non-positive tolerance exercises the default relative threshold.
        ag::run(|ctx: &mut ag::Context<f64>| {
            let m = ag::tensor_ops::convert_to_tensor(
                ag::ndarray::arr2(&[[1.0_f64, 0.0, 2.0], [0.0, 1.0, 3.0]]),
                ctx,
            );
            let rank = helpers::rank_approximation(&m, 0.0_f64, ctx);
            let rank_arr = rank.eval(ctx).expect("Test: failed to evaluate rank");
            assert!(
                (rank_arr[[0, 0]] - 2.0).abs() < 1e-12,
                "expected rank 2 for 2x3 matrix, got {}",
                rank_arr[[0, 0]]
            );
        });
    }
}
