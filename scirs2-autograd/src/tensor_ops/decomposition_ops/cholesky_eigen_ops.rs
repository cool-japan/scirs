//! Cholesky, Eigendecomposition, and Matrix function operations.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::convert_to_tensor;
use crate::tensor_ops::decomposition_backward::cholesky_backward;
use crate::tensor_ops::matrix_calculus::{self, MatrixFnKind, MatrixFnVjpOp};
use crate::Float;
use scirs2_core::ndarray::{Array1, Array2, Ix2};

/// Builds the backward node of a matrix function: `MatrixFnVjp(A, gy)`.
///
/// Every matrix function in this module has the same VJP shape — the adjoint of its
/// Fréchet derivative applied to the output cotangent — so they all funnel through here.
/// The node is built lazily (rather than evaluated inside `grad`) so the tape survives
/// and unfed placeholders upstream are not a problem.
fn append_matrix_fn_grad<F: Float>(ctx: &mut GradientContext<F>, kind: MatrixFnKind) {
    let a = *ctx.input(0);
    let gy = *ctx.output_grad();
    let g = ctx.graph();
    let gx = Tensor::builder(g)
        .append_input(a, false)
        .append_input(gy, false)
        .build(MatrixFnVjpOp { kind });
    ctx.append_input_grad(0, Some(gx));
}

/// Cholesky Decomposition Operation
pub struct CholeskyOp;

impl<F: Float> Op<F> for CholeskyOp {
    fn name(&self) -> &'static str {
        "Cholesky"
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Cholesky decomposition requires square matrix".into(),
            ));
        }

        let n = shape[0];

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Check if matrix is positive definite (simplified check)
        for i in 0..n {
            if input_2d[[i, i]] <= F::zero() {
                return Err(OpError::Other("Matrix is not positive definite".into()));
            }
        }

        // Perform Cholesky decomposition: A = L * L^T
        let mut l = Array2::<F>::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal elements
                    let mut sum = F::zero();
                    for k in 0..j {
                        sum += l[[j, k]] * l[[j, k]];
                    }
                    let diag_val = input_2d[[j, j]] - sum;
                    if diag_val <= F::zero() {
                        return Err(OpError::Other("Matrix is not positive definite".into()));
                    }
                    l[[j, j]] = diag_val.sqrt();
                } else {
                    // Off-diagonal elements
                    let mut sum = F::zero();
                    for k in 0..j {
                        sum += l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = (input_2d[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        ctx.append_output(l.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Murray (2016) Cholesky backward:
        //   phi(Lᵀ · dL)  then  dA = ½ L⁻ᵀ (phi + phiᵀ) L⁻¹
        let gy = ctx.output_grad();
        let output = ctx.output(); // forward output = L (lower triangular factor)
        let g = ctx.graph();

        let l_array = match output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let grad_l_array = match gy.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        match (
            l_array.view().into_dimensionality::<Ix2>(),
            grad_l_array.view().into_dimensionality::<Ix2>(),
        ) {
            (Ok(l_2d), Ok(grad_l_2d)) => {
                let l_owned = l_2d.to_owned();
                let grad_l_owned = grad_l_2d.to_owned();
                let grad_a = cholesky_backward(&l_owned, &grad_l_owned);
                let grad_tensor = convert_to_tensor(grad_a.into_dyn(), g);
                ctx.append_input_grad(0, Some(grad_tensor));
            }
            _ => {
                ctx.append_input_grad(0, None);
            }
        }
    }
}

/// Compute gradient for Cholesky decomposition
#[allow(dead_code)]
pub(crate) fn compute_cholesky_gradient<F: Float>(
    input: &scirs2_core::ndarray::ArrayView2<F>,
    grad_output: &scirs2_core::ndarray::ArrayView2<F>,
) -> Array2<F> {
    let n = input.shape()[0];
    let mut grad_input = Array2::<F>::zeros((n, n));

    // Compute L from input (re-run Cholesky)
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            if i == j {
                let mut sum = F::zero();
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let diag_val = input[[j, j]] - sum;
                if diag_val > F::zero() {
                    l[[j, j]] = diag_val.sqrt();
                }
            } else {
                let mut sum = F::zero();
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                if l[[j, j]] != F::zero() {
                    l[[i, j]] = (input[[i, j]] - sum) / l[[j, j]];
                }
            }
        }
    }

    // Simplified gradient computation
    // For A = L * L^T, if dL is the gradient w.r.t L, then dA = dL * L^T + L * dL^T
    // This is a simplified version assuming grad_output represents dL
    for i in 0..n {
        for j in 0..n {
            // Symmetrize the gradient since Cholesky input should be symmetric
            grad_input[[i, j]] = grad_output[[i.min(j), j.min(i)]];
        }
    }

    // Suppress unused variable — L is computed but only used for gradient shape
    let _ = l;

    grad_input
}

/// Cholesky decomposition of a positive definite matrix.
///
/// Decomposes a symmetric positive definite matrix A into L * L^T where:
/// - L is a lower triangular matrix
///
/// # Arguments
/// * `matrix` - The input symmetric positive definite tensor to decompose
///
/// # Returns
/// A tensor L representing the lower triangular decomposition
#[allow(dead_code)]
pub fn cholesky<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(CholeskyOp)
}

/// Eigendecomposition Operation for Symmetric Matrices
/// Uses a more stable algorithm optimized for symmetric matrices
pub struct SymmetricEigenOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for SymmetricEigenOp {
    fn name(&self) -> &'static str {
        "SymmetricEigen"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Eigendecomposition requires square matrix".into(),
            ));
        }

        let n = shape[0];

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Check if matrix is symmetric.  The tolerance is *relative* to the magnitude of
        // the matrix: an absolute `1e-10` rejects a perfectly symmetric matrix whose
        // entries are large enough that `a_ij - a_ji` cannot round to less than that.
        let magnitude = input_2d.iter().fold(F::zero(), |acc, &v| {
            let m = v.abs();
            if m > acc {
                m
            } else {
                acc
            }
        });
        let symmetry_tolerance = F::from(1e-10).unwrap_or(F::epsilon()) * (F::one() + magnitude);
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = (input_2d[[i, j]] - input_2d[[j, i]]).abs();
                if diff > symmetry_tolerance {
                    return Err(OpError::Other(
                        "Matrix is not symmetric for eigendecomposition".into(),
                    ));
                }
            }
        }

        // A single cyclic-Jacobi path for every size.  Special-casing `n == 1` / `n == 2`
        // bought nothing but a second eigenvector sign/order convention that the backward
        // pass would then have to match; Jacobi is exact (one rotation) for `n == 2` and a
        // no-op for `n == 1`.
        let (eigenvalues, eigenvectors) = matrix_calculus::symmetric_eigen(&input_2d)?;

        ctx.append_output(eigenvalues.into_dyn());
        ctx.append_output(eigenvectors.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Only output 0 (the eigenvalue vector) is reachable: a graph node exposes its
        // first output, so the eigenvector matrix appended second never becomes a
        // differentiable tensor.  The eigenvalue VJP is therefore the whole rule.
        //
        //   λ_k = v_kᵀ A v_k  =>  ∂λ_k / ∂A_ij = v_ki v_kj
        //   Ā = Σ_k ḡ_k v_k v_kᵀ = V diag(ḡ) Vᵀ
        //
        // No `1/(λ_i - λ_j)` term appears, so this rule stays finite on a degenerate
        // spectrum (individual eigenvalues are still only directionally differentiable
        // there; see `SymmetricEigenValuesVjpOp`).
        let a = *ctx.input(0);
        let gy = *ctx.output_grad();
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(a, false)
            .append_input(gy, false)
            .build(SymmetricEigenValuesVjpOp);
        ctx.append_input_grad(0, Some(gx));
    }
}

/// Backward node of [`SymmetricEigenOp`]: `Ā = V diag(ḡ) Vᵀ`.
///
/// Inputs are `(A, ḡ)` where `ḡ` is the cotangent of the eigenvalue vector.
///
/// # Degenerate spectra
///
/// A repeated eigenvalue makes the individual eigenvalues non-differentiable (only
/// symmetric functions of the whole cluster are). This rule stays *finite* there — it
/// contains no eigenvalue-difference denominator — but the eigenvector split inside a
/// cluster is arbitrary, so the per-eigenvalue gradient is only one valid element of the
/// generalised (Clarke) subdifferential.
pub struct SymmetricEigenValuesVjpOp;

impl<F: Float> Op<F> for SymmetricEigenValuesVjpOp {
    fn name(&self) -> &'static str {
        "SymmetricEigenValuesVjp"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_in = ctx.input(0);
        let gy_in = ctx.input(1);

        let a = a_in.view().into_dimensionality::<Ix2>().map_err(|_| {
            OpError::IncompatibleShape("SymmetricEigen backward: A is not 2-D".into())
        })?;
        let n = a.nrows();
        if a.ncols() != n {
            return Err(OpError::IncompatibleShape(
                "SymmetricEigen backward: A is not square".into(),
            ));
        }
        if gy_in.len() != n {
            return Err(OpError::IncompatibleShape(format!(
                "SymmetricEigen backward: eigenvalue cotangent has {} entries, expected {n}",
                gy_in.len()
            )));
        }

        let (_values, vectors) = matrix_calculus::symmetric_eigen(&a)?;
        let mut scaled = Array2::<F>::zeros((n, n));
        for (k, gk) in gy_in.iter().enumerate() {
            for i in 0..n {
                scaled[[i, k]] = vectors[[i, k]] * *gk;
            }
        }
        let grad = scaled.dot(&vectors.t());
        ctx.append_output(grad.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        matrix_calculus::append_unsupported_grad(
            ctx,
            "SymmetricEigen: second-order differentiation of the eigenvalue backward pass \
             is not implemented (it needs the eigenvector sensitivities, which the \
             single-output graph node cannot expose)."
                .into(),
        );
    }
}

/// Eigenvalues of a symmetric matrix, in descending order.
///
/// Cyclic Jacobi (see [`crate::tensor_ops::matrix_calculus::symmetric_eigen`]). The
/// previous implementation returned the sorted *diagonal* of the matrix, which is only
/// the spectrum when the matrix is already diagonal.
#[allow(dead_code)]
pub(crate) fn compute_symmetric_eigenvalues<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array1<F>, OpError> {
    matrix_calculus::symmetric_eigen(matrix).map(|(values, _)| values)
}

/// Eigenvectors (as columns) of a symmetric matrix, ordered to match
/// [`compute_symmetric_eigenvalues`].
///
/// The previous implementation returned the identity matrix regardless of the input.
#[allow(dead_code)]
pub(crate) fn compute_symmetric_eigenvectors<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    matrix_calculus::symmetric_eigen(matrix).map(|(_, vectors)| vectors)
}

/// Symmetric eigendecomposition of a symmetric matrix.
///
/// Decomposes a symmetric matrix A into V * Λ * V^T where:
/// - V is the matrix of eigenvectors (columns)
/// - Λ is the diagonal matrix of eigenvalues
///
/// # Arguments
/// * `matrix` - The input symmetric tensor to decompose
///
/// # Returns
/// A tensor representing the eigendecomposition result
#[allow(dead_code)]
pub fn symmetric_eigen<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(SymmetricEigenOp)
}

/// Matrix Exponential Operation
/// Computes exp(A) for a square matrix A
pub struct MatrixExpOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for MatrixExpOp {
    fn name(&self) -> &'static str {
        "MatrixExp"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix exponential requires square matrix".into(),
            ));
        }

        let _n = shape[0];
        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Compute matrix exponential using Padé approximation
        let result = compute_matrix_exp(&input_2d)?;

        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        append_matrix_fn_grad(ctx, MatrixFnKind::Exp);
    }
}

/// Matrix Logarithm Operation
/// Computes log(A) for a square matrix A
pub struct MatrixLogOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for MatrixLogOp {
    fn name(&self) -> &'static str {
        "MatrixLog"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix logarithm requires square matrix".into(),
            ));
        }

        let n = shape[0];
        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Check if matrix is invertible (simplified check)
        for i in 0..n {
            if input_2d[[i, i]] <= F::zero() {
                return Err(OpError::Other(
                    "Matrix logarithm requires positive definite matrix".into(),
                ));
            }
        }

        let result = compute_matrix_log(&input_2d)?;

        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        append_matrix_fn_grad(ctx, MatrixFnKind::Log);
    }
}

/// Matrix Power Operation
/// Computes A^p for a square matrix A and scalar power p
pub struct MatrixPowerOp {
    pub power: f64,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for MatrixPowerOp {
    fn name(&self) -> &'static str {
        "MatrixPower"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Matrix power requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        let result = compute_matrix_power(&input_2d, self.power)?;

        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        append_matrix_fn_grad(ctx, MatrixFnKind::Power(self.power));
    }
}

/// Matrix exponential `exp(A)`.
///
/// Scaling-and-squaring with a truncated Taylor series (see
/// [`crate::tensor_ops::matrix_calculus::expm`]). The previous implementation used eight
/// Taylor terms for `n <= 3` and, for anything larger, discarded the whole off-diagonal
/// and exponentiated only the diagonal — which is the exponential of a *different*
/// matrix, not an approximation of this one.
#[allow(dead_code)]
pub(crate) fn compute_matrix_exp<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    matrix_calculus::expm(matrix)
}

/// Principal matrix logarithm `log(A)`.
///
/// Inverse scaling and squaring (see [`crate::tensor_ops::matrix_calculus::logm`]). The
/// previous implementation took `ln` of each diagonal entry and filled the off-diagonal
/// with `a_ij / a_ii`, which is not the logarithm of anything.
#[allow(dead_code)]
pub(crate) fn compute_matrix_log<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    matrix_calculus::logm(matrix)
}

/// Matrix power `A^p`.
///
/// Binary exponentiation for integer `p`, `exp(p log A)` otherwise (see
/// [`crate::tensor_ops::matrix_calculus::powm`]). The previous implementation raised only
/// the diagonal entries to the power and returned zeros off the diagonal.
#[allow(dead_code)]
pub(crate) fn compute_matrix_power<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
    power: f64,
) -> Result<Array2<F>, OpError> {
    matrix_calculus::powm(matrix, power)
}

/// Matrix exponential function.
///
/// Computes exp(A) for a square matrix A using Padé approximation.
///
/// # Arguments
/// * `matrix` - The input square tensor
///
/// # Returns
/// A tensor representing exp(A)
#[allow(dead_code)]
pub fn matrix_exp<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(MatrixExpOp)
}

/// Matrix logarithm function.
///
/// Computes log(A) for a square matrix A.
///
/// # Arguments
/// * `matrix` - The input square tensor (must be positive definite)
///
/// # Returns
/// A tensor representing log(A)
#[allow(dead_code)]
pub fn matrix_log<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(MatrixLogOp)
}

/// Matrix power function.
///
/// Computes A^p for a square matrix A and scalar power p.
///
/// # Arguments
/// * `matrix` - The input square tensor
/// * `power` - The power to raise the matrix to
///
/// # Returns
/// A tensor representing A^p
#[allow(dead_code)]
pub fn matrix_power<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
    power: f64,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(MatrixPowerOp { power })
}

#[cfg(test)]
mod tests {
    //! Finite-difference gradient checks for the ops in this file.
    //!
    //! `matrix_log`, `matrix_power` and `symmetric_eigen` are re-exported only inside the
    //! crate (`tensor_ops::matrix_log`/`matrix_power` resolve to the `matrix_functions`
    //! versions), so their gradients cannot be reached from `tests/`. The checks live
    //! here instead, and follow the same two rules as `tests/gradient_fd_harness*.rs`:
    //! a **non-uniform** cotangent, and inputs built with `T::variable`.

    use super::{matrix_exp, matrix_log, matrix_power, symmetric_eigen};
    use crate::tensor_ops as T;
    use scirs2_core::ndarray::{ArrayD, IxDyn};

    type Ctx<'g> = crate::Context<'g, f64>;
    type Tsr<'g> = crate::Tensor<'g, f64>;

    fn array(shape: &[usize], data: &[f64]) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).expect("shape/data mismatch")
    }

    /// `sum(cotangent * build(x))` evaluated on plain constants.
    fn forward_loss<B>(shape: &[usize], data: &[f64], cot: &[f64], build: &B) -> f64
    where
        B: for<'g> Fn(Tsr<'g>, &'g Ctx<'g>) -> Tsr<'g>,
    {
        crate::run(|g| {
            let g: &Ctx = g;
            let x = T::convert_to_tensor(array(shape, data), g);
            let y = build(x, g);
            let y_arr = y.eval(g).expect("forward eval failed");
            y_arr
                .iter()
                .zip(cot.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        })
    }

    fn check<B>(name: &str, shape: &[usize], data: &[f64], cot: &[f64], build: B, tol: f64)
    where
        B: for<'g> Fn(Tsr<'g>, &'g Ctx<'g>) -> Tsr<'g>,
    {
        let analytic: Vec<f64> = crate::run(|g| {
            let g: &Ctx = g;
            let x = T::variable(array(shape, data), g);
            let y = build(x, g);
            let y_arr = y.eval(g).expect("forward eval failed");
            assert_eq!(
                y_arr.len(),
                cot.len(),
                "{name}: cotangent has {} entries, output has {}",
                cot.len(),
                y_arr.len()
            );
            let cot_t = T::convert_to_tensor(array(y_arr.shape(), cot), g);
            let loss = T::sum_all(T::mul(y, cot_t));
            let grads = T::grad(&[loss], &[x]);
            grads[0]
                .eval(g)
                .expect("gradient eval failed")
                .iter()
                .copied()
                .collect()
        });

        assert_eq!(
            analytic.len(),
            data.len(),
            "{name}: gradient has wrong size"
        );
        for (i, &got) in analytic.iter().enumerate() {
            let h = 1e-5 * (1.0 + data[i].abs());
            let mut plus = data.to_vec();
            plus[i] += h;
            let mut minus = data.to_vec();
            minus[i] -= h;
            let numeric = (forward_loss(shape, &plus, cot, &build)
                - forward_loss(shape, &minus, cot, &build))
                / (2.0 * h);
            assert!(
                got.is_finite(),
                "{name}: gradient[{i}] is not finite ({got})"
            );
            assert!(
                (got - numeric).abs() <= tol * (1.0 + numeric.abs()),
                "{name}: d/dx[{i}] analytic={got} finite-difference={numeric}"
            );
        }
    }

    /// Symmetric positive definite, non-diagonal.
    const SPD: [f64; 9] = [2.40, 0.35, -0.20, 0.35, 1.90, 0.28, -0.20, 0.28, 1.55];

    /// The eigenvalue VJP is `V diag(ḡ) Vᵀ`.
    ///
    /// The argument is built as `B + Bᵀ` so a finite difference in `B` stays exactly
    /// symmetric — `SymmetricEigenOp` rejects a non-symmetric argument, and eigenvalues of
    /// a symmetric matrix are only differentiable along symmetric perturbations anyway.
    #[test]
    fn fd_symmetric_eigen_eigenvalue_gradient() {
        let b = [0.90, 0.21, -0.13, 0.07, 0.65, 0.19, -0.11, 0.05, 0.48];
        check(
            "symmetric_eigen",
            &[3, 3],
            &b,
            &[0.41, -0.27, 0.63],
            |x, _g| {
                let sym = T::add(x, T::transpose(x, &[1, 0]));
                symmetric_eigen(&sym)
            },
            1e-4,
        );
    }

    /// A diagonal-only forward would give a one-hot gradient on the diagonal entries and
    /// exactly zero off it; the Fréchet rule gives a dense gradient.
    #[test]
    fn fd_matrix_log_gradient() {
        check(
            "matrix_log",
            &[3, 3],
            &SPD,
            &[0.31, -0.72, 0.15, 0.44, 0.09, -0.53, -0.26, 0.68, 0.37],
            |x, _g| matrix_log(&x),
            1e-4,
        );
    }

    #[test]
    fn fd_matrix_power_fractional() {
        check(
            "matrix_power_0.5",
            &[3, 3],
            &SPD,
            &[0.31, -0.72, 0.15, 0.44, 0.09, -0.53, -0.26, 0.68, 0.37],
            |x, _g| matrix_power(&x, 0.5),
            1e-4,
        );
    }

    #[test]
    fn fd_matrix_power_integer_on_a_general_matrix() {
        // Non-symmetric: `A^3` is a genuine matrix product, not an entry-wise cube.
        let a = [0.30, 0.24, -0.17, 0.11, -0.28, 0.19, -0.21, 0.13, 0.35];
        check(
            "matrix_power_3",
            &[3, 3],
            &a,
            &[0.31, -0.72, 0.15, 0.44, 0.09, -0.53, -0.26, 0.68, 0.37],
            |x, _g| matrix_power(&x, 3.0),
            1e-4,
        );
    }

    #[test]
    fn fd_matrix_exp_gradient() {
        let a = [0.30, 0.24, -0.17, 0.11, -0.28, 0.19, -0.21, 0.13, 0.35];
        check(
            "matrix_exp",
            &[3, 3],
            &a,
            &[0.31, -0.72, 0.15, 0.44, 0.09, -0.53, -0.26, 0.68, 0.37],
            |x, _g| matrix_exp(&x),
            1e-4,
        );
    }

    /// The forward eigenvalues must be the real spectrum, not the diagonal.
    #[test]
    fn symmetric_eigen_forward_is_not_the_diagonal() {
        crate::run(|g| {
            let g: &Ctx = g;
            let x = T::convert_to_tensor(array(&[3, 3], &SPD), g);
            let vals = symmetric_eigen(&x).eval(g).expect("eval failed");
            let mut sorted_diagonal = [SPD[0], SPD[4], SPD[8]];
            sorted_diagonal.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
            let trace: f64 = sorted_diagonal.iter().sum();
            let spectrum_sum: f64 = vals.iter().sum();
            // The spectrum must have the same trace ...
            assert!((trace - spectrum_sum).abs() < 1e-9);
            // ... but must not be the diagonal itself.
            let differs = vals
                .iter()
                .zip(sorted_diagonal.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6);
            assert!(differs, "eigenvalues equal the sorted diagonal: {vals:?}");
        });
    }
}
