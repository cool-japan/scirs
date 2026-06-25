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
        // NOTE: This multi-output `Op::grad` is NOT the live gradient path.
        // The crate's reverse-mode engine (gradient.rs::compute_grad_for_input)
        // dispatches SVD gradients per component via `SVDExtractU/S/Vt`
        // → `SVDBackwardOp`, which implements the exact reduced-SVD VJP
        // (see tensor_ops::decomposition_backward::svd_backward).
        //
        // We deliberately return `None` (truly non-differentiable through THIS
        // op) instead of a fabricated zero gradient: a zero would silently and
        // incorrectly claim ∂loss/∂A = 0.  The honest VJP is available through
        // the public `svd()` API which builds `SVDExtractOp`s.
        ctx.append_input_grad(0, None);
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
        // Exact reduced-SVD VJP for this component, delivered via SVDBackwardOp
        // (recomputes the SVD of A and applies the analytic Townsend/Wan-Zhang
        // formula).  This mirrors the live gradient path in gradient.rs so a
        // direct `Op::grad` invocation is also correct — never a fabricated
        // pass-through of the (wrongly-shaped) component cotangent.
        let gy = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        let gx = crate::tensor::Tensor::builder(g)
            .append_input(input, false)
            .append_input(gy, false)
            .build(crate::tensor_ops::decomposition_ops::SVDBackwardOp {
                component: self.component,
            });
        ctx.append_input_grad(0, Some(gx));
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

    (u, s, v)
}

#[cfg(test)]
mod grad_tests {
    use crate::tensor_ops as T;
    use scirs2_core::ndarray::{array, Array2};

    /// d(Σ singular values)/dA via the autograd graph (SVDExtractS path).
    fn svd_s_grad(a: &Array2<f64>) -> Array2<f64> {
        crate::run(|g| {
            let av = T::variable(a.clone(), g);
            let (_u, s, _v) = super::svd(&av);
            let loss = T::sum_all(s);
            let grads = T::grad(&[&loss], &[&av]);
            grads[0]
                .eval(g)
                .expect("grad eval")
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .expect("2D")
                .to_owned()
        })
    }

    /// Reference Σ singular values via eigenvalues of AᵀA.
    fn sum_singular_values(a: &Array2<f64>) -> f64 {
        let ata = a.t().dot(a);
        let n = ata.nrows();
        // symmetric Jacobi eigenvalues
        let mut m = ata.clone();
        let mut iter = 0;
        loop {
            let mut p = 0;
            let mut q = 1;
            let mut mx = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    if m[[i, j]].abs() > mx {
                        mx = m[[i, j]].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            if mx < 1e-14 || iter > 200 {
                break;
            }
            iter += 1;
            let theta = 0.5 * (2.0 * m[[p, q]]).atan2(m[[p, p]] - m[[q, q]]);
            let (c, sn) = (theta.cos(), theta.sin());
            for i in 0..n {
                let mip = m[[i, p]];
                let miq = m[[i, q]];
                m[[i, p]] = c * mip + sn * miq;
                m[[i, q]] = -sn * mip + c * miq;
            }
            for i in 0..n {
                let mpi = m[[p, i]];
                let mqi = m[[q, i]];
                m[[p, i]] = c * mpi + sn * mqi;
                m[[q, i]] = -sn * mpi + c * mqi;
            }
        }
        (0..n).map(|i| m[[i, i]].max(0.0).sqrt()).sum()
    }

    #[test]
    fn svd_singular_value_gradient_matches_fd() {
        // Non-symmetric square matrix with distinct singular values.
        let a = array![[3.0_f64, 1.0, 0.0], [0.5, 2.5, 0.2], [0.1, 0.3, 1.7]];
        let analytic = svd_s_grad(&a);

        // The gradient must NOT be a pass-through of the (length-k) S cotangent
        // nor all-zero — verify against central FD of Σσ.
        let (m, n) = (a.nrows(), a.ncols());
        let h = 1e-6_f64;
        let mut numeric = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut ap = a.clone();
                let mut am = a.clone();
                ap[[i, j]] += h;
                am[[i, j]] -= h;
                numeric[[i, j]] = (sum_singular_values(&ap) - sum_singular_values(&am)) / (2.0 * h);
            }
        }
        let err = analytic
            .iter()
            .zip(numeric.iter())
            .fold(0.0_f64, |mx, (x, y)| (x - y).abs().max(mx));
        let max_g = analytic.iter().fold(0.0_f64, |mx, &x| x.abs().max(mx));
        assert!(
            max_g > 1e-6,
            "SVD singular-value gradient is all-zero (regression!)"
        );
        assert!(
            err < 1e-4,
            "svd_singular_value_gradient fd mismatch: err = {err}"
        );
    }
}
