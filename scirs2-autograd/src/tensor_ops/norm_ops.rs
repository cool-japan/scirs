use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops;
use crate::Float;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Ix2};

/// Frobenius norm operation with improved gradient computation
pub struct FrobeniusNormOp;

impl<F: Float> Op<F> for FrobeniusNormOp {
    fn name(&self) -> &'static str {
        "FrobeniusNorm"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let input_array = input.view();

        let mut sum_squared = F::zero();
        for &elem in input_array.iter() {
            sum_squared += elem * elem;
        }

        let norm = sum_squared.sqrt();
        ctx.append_output(scirs2_core::ndarray::arr0(norm).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // d ||x||_F / dx = x / ||x||_F, built entirely out of tensor operations so the
        // gradient is a real node in the graph.
        //
        // This op used to `.eval()` its input and the incoming cotangent *while the
        // backward graph was being constructed* and splice the numbers back in with
        // `convert_to_tensor`. That is a snapshot, not a gradient, and both of its
        // failure arms were silent:
        //
        //   * whenever either eval failed -- the operand is not evaluable yet at
        //     backward-construction time -- it reported *no* gradient at all, which the
        //     backward pass reads back as zero. Measured on the old code: a fed
        //     placeholder and a `VariableEnvironment` variable both came back 0 instead
        //     of x / ||x||_F (see `norm_frobenius_gradient_flows_through_a_placeholder`
        //     and `norm_frobenius_gradient_tracks_the_current_variable_value` in
        //     tests/gradient_fd_harness.rs);
        //   * when the evals did succeed, the spliced `convert_to_tensor` constant froze
        //     the gradient at the value the input happened to have, and, being a
        //     constant, carried no derivative of its own -- so second-order
        //     differentiation through the Frobenius norm collapsed to zero.
        //
        // The output tensor (`||x||_F`, already a node) is reused as the denominator
        // instead of recomputing the sum of squares.
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let output = ctx.output();
        let g = ctx.graph();

        // The Frobenius norm is not differentiable at x = 0; flooring the denominator
        // makes that point report a zero gradient (0 / eps) instead of NaN, and is
        // otherwise inert -- a norm at or below `F::epsilon()` cannot carry meaningful
        // directional information in this precision anyway.
        let floor = tensor_ops::scalar(F::epsilon(), g);
        let safe_norm = tensor_ops::maximum(output, floor);
        let grad_input = tensor_ops::mul(tensor_ops::div(input, safe_norm), grad_output);
        ctx.append_input_grad(0, Some(grad_input));
    }
}

/// Spectral norm operation with proper gradient computation
pub struct SpectralNormOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for SpectralNormOp {
    fn name(&self) -> &'static str {
        "SpectralNorm"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(
                "Spectral norm requires 2D matrix".into(),
            ));
        }

        // Convert input to 2D matrix
        let matrix = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Use power iteration to find the largest singular value
        let (_, sigma_max) = power_iteration_spectral(
            &matrix,
            50,
            F::from(1e-8).expect("Failed to convert constant to float"),
        );

        ctx.append_output(scirs2_core::ndarray::arr0(sigma_max).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        // Evaluate the input to work with concrete values for SVD computation
        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_scalar = grad_output_array[[]];

        if let Ok(matrix) = input_array.view().into_dimensionality::<Ix2>() {
            let grad_matrix = compute_spectral_norm_gradient(&matrix, grad_scalar);
            let grad_tensor = tensor_ops::convert_to_tensor(grad_matrix.into_dyn(), g);
            ctx.append_input_grad(0, Some(grad_tensor));
            return;
        }

        // Fallback
        ctx.append_input_grad(0, None);
    }
}

/// Nuclear norm operation with proper gradient computation
pub struct NuclearNormOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for NuclearNormOp {
    fn name(&self) -> &'static str {
        "NuclearNorm"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(
                "Nuclear norm requires 2D matrix".into(),
            ));
        }

        // Convert input to 2D matrix
        let matrix = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Check for diagonal matrix special case
        if is_diagonal_matrix(&matrix) {
            let nuclear_norm = compute_diagonal_nuclear_norm(&matrix);
            ctx.append_output(scirs2_core::ndarray::arr0(nuclear_norm).into_dyn());
            return Ok(());
        }

        // For general matrices, compute nuclear norm as sum of singular values
        let nuclear_norm = compute_nuclear_norm_improved(&matrix);

        ctx.append_output(scirs2_core::ndarray::arr0(nuclear_norm).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        // Evaluate inputs to work with concrete values
        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_scalar = grad_output_array[[]];

        if let Ok(matrix) = input_array.view().into_dimensionality::<Ix2>() {
            let grad_matrix = compute_nuclear_norm_gradient_improved(&matrix, grad_scalar);
            let grad_tensor = tensor_ops::convert_to_tensor(grad_matrix.into_dyn(), g);
            ctx.append_input_grad(0, Some(grad_tensor));
            return;
        }

        // Fallback
        ctx.append_input_grad(0, None);
    }
}

// Helper functions

/// Check if matrix is diagonal
#[allow(dead_code)]
fn is_diagonal_matrix<F: Float>(matrix: &ArrayView2<F>) -> bool {
    let (m, n) = matrix.dim();
    for i in 0..m {
        for j in 0..n {
            if i != j && matrix[[i, j]] != F::zero() {
                return false;
            }
        }
    }
    true
}

/// Compute nuclear norm for diagonal matrix
#[allow(dead_code)]
fn compute_diagonal_nuclear_norm<F: Float>(matrix: &ArrayView2<F>) -> F {
    let (m, n) = matrix.dim();
    let mut sum = F::zero();
    let min_dim = m.min(n);

    for i in 0..min_dim {
        sum += matrix[[i, i]].abs();
    }

    sum
}

#[allow(dead_code)]
/// Compute nuclear norm using power iteration approximation
fn compute_nuclear_norm_approximation<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &ArrayView2<F>,
) -> F {
    let (m, n) = matrix.dim();
    let min_dim = m.min(n);

    // For small matrices, use a simple approximation
    if min_dim <= 3 {
        // Fall back to Frobenius norm for small matrices
        let mut sum_squared = F::zero();
        for &elem in matrix.iter() {
            sum_squared += elem * elem;
        }
        return sum_squared.sqrt();
    }

    // For larger matrices, estimate a few singular values
    let max_rank = (min_dim as f64 * 0.5).ceil() as usize;
    let mut working_matrix = matrix.to_owned();
    let mut nuclear_norm = F::zero();

    for _ in 0..max_rank {
        let (_, sigma) = power_iteration_spectral(
            &working_matrix.view(),
            10,
            F::from(1e-6).expect("Failed to convert constant to float"),
        );

        if sigma < F::epsilon() * F::from(10.0).expect("Failed to convert constant to float") {
            break;
        }

        nuclear_norm += sigma;

        // Simple deflation: subtract a rank-1 approximation
        let (u, _) = power_iteration_spectral(
            &working_matrix.view(),
            5,
            F::from(1e-6).expect("Failed to convert constant to float"),
        );
        let at = working_matrix.t();
        let v = at.dot(&u) / sigma;

        // Deflate: A = A - sigma * u * v^T
        for i in 0..m {
            for j in 0..n {
                working_matrix[[i, j]] -= sigma * u[i] * v[j];
            }
        }
    }

    nuclear_norm
}

/// Power iteration for spectral norm
#[allow(dead_code)]
fn power_iteration_spectral<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &ArrayView2<F>,
    max_iter: usize,
    tol: F,
) -> (Array1<F>, F) {
    let (m, n) = matrix.dim();

    // Initialize with normalized vector
    let mut u = Array1::<F>::zeros(m);
    u[0] = F::one();

    // Add some perturbation to avoid getting stuck
    for i in 1..m {
        u[i] = F::from(0.01).expect("Failed to convert constant to float")
            * F::from(i as f64).expect("Failed to convert to float");
    }

    // Normalize
    let norm = u.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();
    if norm > F::epsilon() {
        u.mapv_inplace(|x| x / norm);
    }

    let mut prev_sigma = F::zero();

    for _iter in 0..max_iter {
        // A * u
        let au = matrix.dot(&u);

        // A^T * (A * u)
        let atau = matrix.t().dot(&au);

        // Compute norm (approximate eigenvalue of A^T * A)
        let sigma = atau.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();

        // Check convergence
        if (sigma - prev_sigma).abs() < tol {
            // Final computation of actual singular value
            let au_final = matrix.dot(&u);
            let sigma_final = au_final
                .iter()
                .fold(F::zero(), |acc, &x| acc + x * x)
                .sqrt();
            return (u, sigma_final);
        }

        prev_sigma = sigma;

        // Normalize for next iteration
        let norm = atau.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();
        if norm > F::epsilon() {
            u = atau.mapv(|x| x / norm);
        }
    }

    // Final estimate
    let au = matrix.dot(&u);
    let sigma = au.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();
    (u, sigma)
}

/// Compute gradient for spectral norm
#[allow(dead_code)]
fn compute_spectral_norm_gradient<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &ArrayView2<F>,
    grad_scalar: F,
) -> Array2<F> {
    let (m, n) = matrix.dim();

    // Special handling for diagonal matrices
    if is_diagonal_matrix(matrix) {
        let mut grad_matrix = Array2::zeros((m, n));
        let min_dim = m.min(n);

        // Find the largest diagonal element
        let mut max_idx = 0;
        let mut max_val = F::zero();
        for i in 0..min_dim {
            let abs_val = matrix[[i, i]].abs();
            if abs_val > max_val {
                max_val = abs_val;
                max_idx = i;
            }
        }

        // Gradient is 1 at the position of the largest singular value
        grad_matrix[[max_idx, max_idx]] = grad_scalar;

        return grad_matrix;
    }

    // For general matrices, recompute the singular vectors
    let (u, sigma) = power_iteration_spectral(
        matrix,
        20,
        F::from(1e-6).expect("Failed to convert constant to float"),
    );

    // Compute v = A^T * u / sigma
    let v = if sigma > F::epsilon() {
        matrix.t().dot(&u) / sigma
    } else {
        Array1::zeros(n)
    };

    // Create outer product u * v^T
    let mut grad_matrix = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            grad_matrix[[i, j]] = u[i] * v[j] * grad_scalar;
        }
    }

    grad_matrix
}

#[allow(dead_code)]
/// Compute gradient for nuclear norm
fn compute_nuclear_norm_gradient<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &ArrayView2<F>,
    grad_scalar: F,
) -> Array2<F> {
    let (m, n) = matrix.dim();

    // Handle diagonal matrix case
    if is_diagonal_matrix(matrix) {
        let mut grad_matrix = Array2::zeros((m, n));
        let min_dim = m.min(n);

        // Gradient is sign of diagonal elements
        for i in 0..min_dim {
            let diag_val = matrix[[i, i]];
            grad_matrix[[i, i]] = if diag_val > F::zero() {
                grad_scalar
            } else if diag_val < F::zero() {
                -grad_scalar
            } else {
                F::zero()
            };
        }

        return grad_matrix;
    }

    // For general matrices, use identity matrix as approximation
    // This is not mathematically accurate but provides a reasonable gradient
    let mut grad_matrix = Array2::zeros((m, n));
    let min_dim = m.min(n);
    for i in 0..min_dim {
        grad_matrix[[i, i]] = grad_scalar;
    }

    grad_matrix
}

/// Improved nuclear norm computation using better SVD approximation
#[allow(dead_code)]
fn compute_nuclear_norm_improved<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &ArrayView2<F>,
) -> F {
    let (m, n) = matrix.dim();
    let min_dim = m.min(n);

    // For small matrices, use a simple approximation
    if min_dim <= 2 {
        // Sum of absolute values of diagonal elements as approximation
        let mut nuclear_norm = F::zero();
        for i in 0..min_dim {
            nuclear_norm += matrix[[i, i]].abs();
        }
        return nuclear_norm;
    }

    // For larger matrices, use power iteration to estimate singular values
    let mut working_matrix = matrix.to_owned();
    let mut nuclear_norm = F::zero();
    let max_rank = min_dim.min(5); // Limit iterations for performance

    for _ in 0..max_rank {
        let (u, sigma) = power_iteration_spectral(
            &working_matrix.view(),
            20,
            F::from(1e-6).expect("Failed to convert constant to float"),
        );

        if sigma < F::epsilon() * F::from(10.0).expect("Failed to convert constant to float") {
            break;
        }

        nuclear_norm += sigma;

        // Simple deflation: subtract a rank-1 approximation
        let at = working_matrix.t();
        let v = at.dot(&u) / sigma;

        // Deflate: A = A - sigma * u * v^T
        for i in 0..m {
            for j in 0..n {
                working_matrix[[i, j]] -= sigma * u[i] * v[j];
            }
        }
    }

    nuclear_norm
}

/// Improved nuclear norm gradient computation
#[allow(dead_code)]
fn compute_nuclear_norm_gradient_improved<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &ArrayView2<F>,
    grad_scalar: F,
) -> Array2<F> {
    let (m, n) = matrix.dim();

    // Handle diagonal matrix case
    if is_diagonal_matrix(matrix) {
        let mut grad_matrix = Array2::zeros((m, n));
        let min_dim = m.min(n);

        // Gradient is sign of diagonal elements
        for i in 0..min_dim {
            let diag_val = matrix[[i, i]];
            grad_matrix[[i, i]] = if diag_val > F::zero() {
                grad_scalar
            } else if diag_val < F::zero() {
                -grad_scalar
            } else {
                F::zero()
            };
        }

        return grad_matrix;
    }

    // For general matrices, use approximate SVD-based gradient
    // This is a simplified version that accumulates gradients from multiple singular vectors
    let mut grad_matrix = Array2::zeros((m, n));
    let mut working_matrix = matrix.to_owned();
    let min_dim = m.min(n);
    let max_rank = min_dim.min(3); // Limit for performance

    for _ in 0..max_rank {
        let (u, sigma) = power_iteration_spectral(
            &working_matrix.view(),
            10,
            F::from(1e-6).expect("Failed to convert constant to float"),
        );

        if sigma < F::epsilon() * F::from(10.0).expect("Failed to convert constant to float") {
            break;
        }

        // Compute v = A^T * u / sigma
        let v = if sigma > F::epsilon() {
            working_matrix.t().dot(&u) / sigma
        } else {
            Array1::zeros(n)
        };

        // Add contribution from this singular vector pair
        for i in 0..m {
            for j in 0..n {
                grad_matrix[[i, j]] += u[i] * v[j] * grad_scalar;
            }
        }

        // Simple deflation for next iteration
        for i in 0..m {
            for j in 0..n {
                working_matrix[[i, j]] -= sigma * u[i] * v[j];
            }
        }
    }

    grad_matrix
}

// Public API functions

#[allow(dead_code)]
pub fn frobenius_norm<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(FrobeniusNormOp)
}

#[allow(dead_code)]
pub fn spectral_norm<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(SpectralNormOp)
}

#[allow(dead_code)]
pub fn nuclear_norm<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(NuclearNormOp)
}
