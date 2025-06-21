use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops;
use crate::Float;
use ndarray::{Array2, ArrayView2, Ix2};

/// Matrix 1-norm (maximum column sum)
pub struct Matrix1NormOp;

impl<F: Float> Op<F> for Matrix1NormOp {
    fn name(&self) -> &'static str {
        "Matrix1Norm"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(
                "Matrix 1-norm requires 2D matrix".into(),
            ));
        }

        let matrix = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        let norm = compute_matrix_1_norm(&matrix);
        ctx.append_output(ndarray::arr0(norm).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

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
            let grad_matrix = compute_matrix_1_norm_gradient(&matrix, grad_scalar);
            let grad_tensor = tensor_ops::convert_to_tensor(grad_matrix.into_dyn(), g);
            ctx.append_input_grad(0, Some(grad_tensor));
            return;
        }

        ctx.append_input_grad(0, None);
    }
}

/// Matrix infinity-norm (maximum row sum)
pub struct MatrixInfNormOp;

impl<F: Float> Op<F> for MatrixInfNormOp {
    fn name(&self) -> &'static str {
        "MatrixInfNorm"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(
                "Matrix infinity-norm requires 2D matrix".into(),
            ));
        }

        let matrix = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        let norm = compute_matrix_inf_norm(&matrix);
        ctx.append_output(ndarray::arr0(norm).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

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
            let grad_matrix = compute_matrix_inf_norm_gradient(&matrix, grad_scalar);
            let grad_tensor = tensor_ops::convert_to_tensor(grad_matrix.into_dyn(), g);
            ctx.append_input_grad(0, Some(grad_tensor));
            return;
        }

        ctx.append_input_grad(0, None);
    }
}

/// Matrix 2-norm (largest singular value, same as spectral norm)
/// This is an alias for spectral norm with consistent naming
pub struct Matrix2NormOp;

impl<F: Float + ndarray::ScalarOperand> Op<F> for Matrix2NormOp {
    fn name(&self) -> &'static str {
        "Matrix2Norm"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(
                "Matrix 2-norm requires 2D matrix".into(),
            ));
        }

        let matrix = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        let norm = compute_matrix_2_norm(&matrix);
        ctx.append_output(ndarray::arr0(norm).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Delegate to spectral norm gradient computation
        // since 2-norm is the same as spectral norm
        let grad_output = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

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
            let grad_matrix = compute_matrix_2_norm_gradient(&matrix, grad_scalar);
            let grad_tensor = tensor_ops::convert_to_tensor(grad_matrix.into_dyn(), g);
            ctx.append_input_grad(0, Some(grad_tensor));
            return;
        }

        ctx.append_input_grad(0, None);
    }
}

// Helper functions

/// Compute matrix 1-norm (maximum column sum)
fn compute_matrix_1_norm<F: Float>(matrix: &ArrayView2<F>) -> F {
    let (m, n) = matrix.dim();
    let mut max_col_sum = F::zero();

    for j in 0..n {
        let mut col_sum = F::zero();
        for i in 0..m {
            col_sum += matrix[[i, j]].abs();
        }
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }

    max_col_sum
}

/// Compute gradient for matrix 1-norm
fn compute_matrix_1_norm_gradient<F: Float>(matrix: &ArrayView2<F>, grad_scalar: F) -> Array2<F> {
    let (m, n) = matrix.dim();
    let mut grad_matrix = Array2::zeros((m, n));

    // Find column with maximum sum
    let mut max_col = 0;
    let mut max_col_sum = F::zero();

    for j in 0..n {
        let mut col_sum = F::zero();
        for i in 0..m {
            col_sum += matrix[[i, j]].abs();
        }
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
            max_col = j;
        }
    }

    // Gradient is sign of elements in the maximum column
    for i in 0..m {
        let elem = matrix[[i, max_col]];
        grad_matrix[[i, max_col]] = if elem > F::zero() {
            grad_scalar
        } else if elem < F::zero() {
            -grad_scalar
        } else {
            F::zero()
        };
    }

    grad_matrix
}

/// Compute matrix infinity-norm (maximum row sum)
fn compute_matrix_inf_norm<F: Float>(matrix: &ArrayView2<F>) -> F {
    let (m, n) = matrix.dim();
    let mut max_row_sum = F::zero();

    for i in 0..m {
        let mut row_sum = F::zero();
        for j in 0..n {
            row_sum += matrix[[i, j]].abs();
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    max_row_sum
}

/// Compute gradient for matrix infinity-norm
fn compute_matrix_inf_norm_gradient<F: Float>(matrix: &ArrayView2<F>, grad_scalar: F) -> Array2<F> {
    let (m, n) = matrix.dim();
    let mut grad_matrix = Array2::zeros((m, n));

    // Find row with maximum sum
    let mut max_row = 0;
    let mut max_row_sum = F::zero();

    for i in 0..m {
        let mut row_sum = F::zero();
        for j in 0..n {
            row_sum += matrix[[i, j]].abs();
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
            max_row = i;
        }
    }

    // Gradient is sign of elements in the maximum row
    for j in 0..n {
        let elem = matrix[[max_row, j]];
        grad_matrix[[max_row, j]] = if elem > F::zero() {
            grad_scalar
        } else if elem < F::zero() {
            -grad_scalar
        } else {
            F::zero()
        };
    }

    grad_matrix
}

/// Compute matrix 2-norm (largest singular value)
fn compute_matrix_2_norm<F: Float + ndarray::ScalarOperand>(matrix: &ArrayView2<F>) -> F {
    // Use power iteration to find the largest singular value
    let (_, sigma_max) = power_iteration_2norm(matrix, 50, F::from(1e-8).unwrap());
    sigma_max
}

/// Power iteration for 2-norm computation
fn power_iteration_2norm<F: Float + ndarray::ScalarOperand>(
    matrix: &ArrayView2<F>,
    max_iter: usize,
    tol: F,
) -> (ndarray::Array1<F>, F) {
    let (m, _n) = matrix.dim();

    // Initialize with normalized vector
    let mut u = ndarray::Array1::<F>::zeros(m);
    u[0] = F::one();

    // Add some perturbation to avoid getting stuck
    for i in 1..m {
        u[i] = F::from(0.01).unwrap() * F::from(i as f64).unwrap();
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

/// Compute gradient for matrix 2-norm
fn compute_matrix_2_norm_gradient<F: Float + ndarray::ScalarOperand>(
    matrix: &ArrayView2<F>,
    grad_scalar: F,
) -> Array2<F> {
    let (m, n) = matrix.dim();

    // Recompute the singular vectors
    let (u, sigma) = power_iteration_2norm(matrix, 20, F::from(1e-6).unwrap());

    // Compute v = A^T * u / sigma
    let v = if sigma > F::epsilon() {
        matrix.t().dot(&u) / sigma
    } else {
        ndarray::Array1::zeros(n)
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

// Public API functions

/// Compute the 1-norm of a matrix (maximum column sum)
pub fn norm1<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(Matrix1NormOp)
}

/// Compute the 2-norm of a matrix (largest singular value)
pub fn norm2<'g, F: Float + ndarray::ScalarOperand>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(Matrix2NormOp)
}

/// Compute the infinity-norm of a matrix (maximum row sum)
pub fn norminf<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(MatrixInfNormOp)
}

/// Compute the Frobenius norm of a matrix
/// This is an alias for the Frobenius norm in norm_ops.rs
pub fn normfro<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    crate::tensor_ops::norm_ops::frobenius_norm(matrix)
}
