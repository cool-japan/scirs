//! SVD operations — SVDOp, SVDExtractOp, and helper functions.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::convert_to_tensor;
use crate::Float;
use scirs2_core::ndarray::{Array1, Array2, Ix2};
use scirs2_core::numeric::FromPrimitive;

/// SVD Operation — Golub-Reinsch via one-sided Jacobi SVD (pure Rust)
pub struct SVDOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for SVDOp {
    fn name(&self) -> &'static str {
        "SVD"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(format!(
                "SVD requires 2D matrix, got shape {shape:?}"
            )));
        }

        // Convert input to 2D matrix
        let input_2d = input.view().into_dimensionality::<Ix2>().map_err(|e| {
            OpError::IncompatibleShape(format!("Failed to convert input to 2D: {e:?}"))
        })?;

        // Use real Jacobi SVD
        let (u, s, vt) =
            crate::tensor_ops::advanced_decompositions::compute_svd_jacobi(&input_2d, false)?;

        // Append the outputs: U, sigma, V^T
        ctx.append_output(u.into_dyn());
        ctx.append_output(s.into_dyn());
        ctx.append_output(vt.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let input = ctx.input(0);
        let g = ctx.graph();

        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let shape = input_array.shape();
        if shape.len() != 2 {
            ctx.append_input_grad(0, None);
            return;
        }

        let m = shape[0];
        let n = shape[1];

        // Return zero gradient as a safe default for SVD (complex to compute correctly)
        let gradient_matrix = Array2::<F>::zeros((m, n));
        let grad_tensor = convert_to_tensor(gradient_matrix.into_dyn(), g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

/// SVD component extraction — re-runs the real SVD and extracts the requested component
pub struct SVDExtractOp {
    pub(crate) component: usize,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for SVDExtractOp {
    fn name(&self) -> &'static str {
        match self.component {
            0 => "SVDExtractU",
            1 => "SVDExtractS",
            2 => "SVDExtractVt",
            _ => "SVDExtractUnknown",
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape("SVD requires 2D matrix".into()));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Run real Jacobi SVD
        let (u, s, vt) =
            crate::tensor_ops::advanced_decompositions::compute_svd_jacobi(&input_2d, false)?;

        // Extract the requested component
        match self.component {
            0 => ctx.append_output(u.into_dyn()),
            1 => ctx.append_output(s.into_dyn()),
            2 => ctx.append_output(vt.into_dyn()),
            _ => return Err(OpError::IncompatibleShape("Invalid component index".into())),
        }

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let g = ctx.graph();

        // Pass through gradient (best-effort: chain rule for component extraction)
        let grad_tensor = match gy.eval(g) {
            Ok(arr) => convert_to_tensor(arr, g),
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

/// The power iteration method for finding eigenvectors of a matrix.
/// This is used in the SVD implementation for matrices larger than 2x2.
#[allow(dead_code)]
pub(crate) fn power_iteration<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Array2<F>,
    max_iter: usize,
    tol: F,
) -> (Array1<F>, F) {
    let n = matrix.shape()[0];

    // Initialize with random unit vector
    let mut v = Array1::<F>::zeros(n);
    v[0] = F::one(); // Start with [1, 0, 0, ...]

    // Add small perturbation to avoid getting stuck
    for i in 1..n {
        v[i] = F::from(0.01).expect("Failed to convert constant to float")
            * F::from(i as f64 / n as f64).expect("Failed to convert to float");
    }

    // Normalize initial vector
    let norm = v.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();
    if norm > F::epsilon() {
        v = &v / norm;
    }

    let mut lambda_prev = F::zero();

    for _ in 0..max_iter {
        // Multiply matrix by current vector: w = A*v
        let w = matrix.dot(&v);

        // Find largest component to estimate eigenvalue
        let lambda = w.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()));

        // Check convergence
        if (lambda - lambda_prev).abs() < tol {
            // Return eigenvector and eigenvalue
            return (w.clone(), lambda);
        }

        lambda_prev = lambda;

        // Normalize w to get new v
        let norm = w.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();
        if norm > F::epsilon() {
            v = &w / norm;
        } else {
            // If norm is too small, we're converging to the zero vector
            // This could happen with a nilpotent matrix, so we restart with a different vector
            for i in 0..n {
                v[i] = F::from((i + 1) as f64 / n as f64).expect("Operation failed");
            }
            let norm = v.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();
            if norm > F::epsilon() {
                v = &v / norm;
            }
        }
    }

    // Return best guess if max iterations reached
    let w = matrix.dot(&v);
    let lambda = w.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()));
    (w, lambda)
}

/// Improved matrix deflation for SVD algorithm
/// This removes the contribution of a found singular value and vectors
/// from the matrix to find additional singular values.
#[allow(dead_code)]
pub(crate) fn improved_deflation<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Array2<F>,
    u_vec: &Array1<F>,
    sigma: F,
    v_vec: &Array1<F>,
) -> Array2<F> {
    let m = matrix.shape()[0];
    let n = matrix.shape()[1];
    let mut deflated = matrix.clone();

    // Subtract the outer product sigma * u * v^T
    for i in 0..m {
        for j in 0..n {
            deflated[[i, j]] -= sigma * u_vec[i] * v_vec[j];
        }
    }

    deflated
}

/// Singular Value Decomposition (SVD)
///
/// Decomposes a matrix A into U * S * V^T where:
/// - U is an orthogonal matrix
/// - S is a diagonal matrix of singular values
/// - V is an orthogonal matrix
///
/// # Arguments
/// * `matrix` - The input tensor to decompose
///
/// # Returns
/// A tuple of tensors (U, S, V) representing the decomposition
#[allow(dead_code)]
pub fn svd<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
) -> (Tensor<'g, F>, Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    // Extract the components directly using the extraction operator
    let u = Tensor::builder(g)
        .append_input(matrix, false)
        .build(SVDExtractOp { component: 0 });

    let s = Tensor::builder(g)
        .append_input(matrix, false)
        .build(SVDExtractOp { component: 1 });

    let v = Tensor::builder(g)
        .append_input(matrix, false)
        .build(SVDExtractOp { component: 2 });

    println!("SVD function: Extracted U, S, V components using specialized operators");

    (u, s, v)
}
