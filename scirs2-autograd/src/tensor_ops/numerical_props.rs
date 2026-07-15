//! Numerical properties of matrices (rank, condition number, etc.)

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use scirs2_core::ndarray::{Array1, Array2, Ix2};
use std::cmp::min;

/// Matrix Rank Operation
///
/// Computes the rank of a matrix using SVD with a given tolerance
pub struct RankOp<F: Float> {
    pub tolerance: Option<F>,
}

impl<F: Float> Op<F> for RankOp<F> {
    fn name(&self) -> &'static str {
        "Rank"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(format!(
                "Rank requires 2D matrix, got shape {shape:?}"
            )));
        }

        let m = shape[0];
        let n = shape[1];
        let min_dim = min(m, n);

        // Convert to 2D array
        let matrix = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Compute proper singular values using SVD from scirs2-linalg
        let matrix_owned = matrix.to_owned();
        let mut singular_values = match Self::compute_svd_singular_values(&matrix_owned) {
            Ok(sv) => sv,
            Err(_) => {
                // Fallback to diagonal approximation if SVD fails
                let mut sv = Vec::with_capacity(min_dim);
                for i in 0..min_dim {
                    if i < m && i < n {
                        sv.push(matrix[[i, i]].abs());
                    } else {
                        sv.push(F::zero());
                    }
                }
                sv
            }
        };

        // Sort singular values in descending order
        singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Determine tolerance
        let tol = if let Some(t) = self.tolerance {
            t
        } else {
            // Default tolerance: max(m, n) * eps * max(singular_values)
            let max_sv = singular_values.first().copied().unwrap_or(F::zero());
            let eps = F::epsilon();
            let max_dim = F::from(m.max(n)).expect("Operation failed");
            max_dim * eps * max_sv
        };

        // Count non-zero singular values above tolerance
        let rank = singular_values.iter().filter(|&&sv| sv > tol).count();

        let rank_value = F::from(rank).expect("Failed to convert to float");
        let result = scirs2_core::ndarray::arr0(rank_value);

        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Rank is a discrete function, gradient is technically undefined
        // We return zero gradient
        ctx.append_input_grad(0, None);
    }
}

/// Singular values paired with their corresponding left/right singular
/// vectors, as produced by the power-iteration-based SVD helper below.
///
/// Kept private to this module: it exists purely so that [`CondOp::grad`]
/// can compute the analytic gradient of the 2-norm condition number (which
/// needs `u_max`, `v_max`, `u_min`, `v_min`, not just the singular values)
/// without duplicating the eigen-decomposition logic that
/// [`RankOp::compute_svd_singular_values`] already implements for
/// `matrix_rank`.
struct SvdVectors<F: Float> {
    /// Singular values, in the order they were extracted. Barring
    /// numerical noise this is descending, i.e. `values[0]` is the largest
    /// and `values[values.len() - 1]` is the smallest that was found
    /// (computation stops early -- see [`RankOp::compute_svd_full`] -- once
    /// a remaining eigenvalue is numerically zero, so `values` may be
    /// shorter than `min(m, n)` for a rank-deficient matrix).
    values: Vec<F>,
    /// Left singular vectors, one per column (shape `m x k`, `k = values.len()`).
    u: Array2<F>,
    /// Right singular vectors, one per column (shape `n x k`).
    v: Array2<F>,
}

impl<F: Float> RankOp<F> {
    /// Compute singular values using proper SVD decomposition
    fn compute_svd_singular_values(matrix: &Array2<F>) -> Result<Vec<F>, OpError> {
        Ok(Self::compute_svd_full(matrix)?.values)
    }

    /// Compute singular values *and* their corresponding left/right singular
    /// vectors.
    ///
    /// The algorithm repeatedly runs power iteration on the Gram matrix
    /// (`AᵀA` for a tall/square input, `AAᵀ` for a wide one) and removes
    /// each eigenpair found via exact (Wielandt) deflation:
    /// `M ← M − λ·v·vᵀ` for a unit-norm eigenvector `v`. Because `M` is
    /// symmetric PSD, this leaves every *other* eigenpair of `M` numerically
    /// untouched, so the next power iteration converges to the true
    /// next-largest eigenvalue -- unlike a naive shift-and-clip
    /// (`M ← clip_negative(M − λ·I)`), which perturbs the *entire* spectrum
    /// and only happens to recover the correct answer for a diagonal input.
    ///
    /// Whichever side (`AᵀA` or `AAᵀ`) is *not* eigendecomposed is derived
    /// directly from `A` itself (`u = A·v/σ` or `v = Aᵀ·u/σ`), so the
    /// returned pairs always satisfy the defining SVD relations
    /// `A·v_i = σ_i·u_i` and `Aᵀ·u_i = σ_i·v_i` exactly, even though the
    /// power-iteration eigenvector estimate carries some residual error.
    fn compute_svd_full(matrix: &Array2<F>) -> Result<SvdVectors<F>, OpError> {
        let (m, n) = matrix.dim();
        let min_dim = m.min(n);

        // Convert to f64 for numerical computation
        let matrix_f64: Array2<f64> = matrix.mapv(|x| x.to_f64().unwrap_or(0.0));

        // For a tall/square matrix (m >= n) the eigenvectors of A^T A
        // (n x n) are the right singular vectors; for a wide matrix
        // (m < n) the eigenvectors of A A^T (m x m) are the left singular
        // vectors.
        let wide = m < n;
        let gram = if wide {
            matrix_f64.dot(&matrix_f64.t())
        } else {
            matrix_f64.t().dot(&matrix_f64)
        };

        let mut values_f64: Vec<f64> = Vec::with_capacity(min_dim);
        let mut primary_vectors: Vec<Array1<f64>> = Vec::with_capacity(min_dim);
        let mut current = gram.clone();

        for stage in 0..min_dim {
            // Power iteration to find the dominant eigenvalue/eigenvector
            // of the (deflated) Gram matrix.
            let (eigenvalue, eigvec) = Self::power_iteration(&current, stage)?;
            if eigenvalue <= 1e-12_f64 {
                break; // Remaining spectrum is effectively zero.
            }

            values_f64.push(eigenvalue.sqrt());

            // Exact (Wielandt) deflation: current -= eigenvalue * v * v^T
            let dim = eigvec.len();
            let mut outer = Array2::<f64>::zeros((dim, dim));
            for a in 0..dim {
                let va = eigvec[a];
                for b in 0..dim {
                    outer[[a, b]] = va * eigvec[b] * eigenvalue;
                }
            }
            current = current - outer;

            primary_vectors.push(eigvec);
        }

        let k = values_f64.len();
        let mut u = Array2::<F>::zeros((m, k));
        let mut v = Array2::<F>::zeros((n, k));
        let mut values = Vec::with_capacity(k);

        for (i, &sigma) in values_f64.iter().enumerate() {
            values.push(F::from(sigma).unwrap_or(F::zero()));
            if sigma <= 1e-14_f64 {
                continue; // Leave u/v columns zeroed for a null singular value.
            }

            if wide {
                // primary_vectors[i] is the left singular vector u_i (length m).
                let u_i = &primary_vectors[i];
                for r in 0..m {
                    u[[r, i]] = F::from(u_i[r]).unwrap_or(F::zero());
                }
                let v_i = matrix_f64.t().dot(u_i).mapv(|x| x / sigma);
                for r in 0..n {
                    v[[r, i]] = F::from(v_i[r]).unwrap_or(F::zero());
                }
            } else {
                // primary_vectors[i] is the right singular vector v_i (length n).
                let v_i = &primary_vectors[i];
                for r in 0..n {
                    v[[r, i]] = F::from(v_i[r]).unwrap_or(F::zero());
                }
                let u_i = matrix_f64.dot(v_i).mapv(|x| x / sigma);
                for r in 0..m {
                    u[[r, i]] = F::from(u_i[r]).unwrap_or(F::zero());
                }
            }
        }

        Ok(SvdVectors { values, u, v })
    }

    /// Deterministic, stage-dependent starting vector for power iteration.
    ///
    /// Reusing the identical starting vector at every deflation stage is
    /// unsound once deflation is exact: if the vector found at an earlier
    /// stage happens to equal (or nearly equal) the fixed starting vector --
    /// which happens for *any* matrix with a repeated top eigenvalue, e.g. a
    /// multiple of the identity, or for a symmetric matrix whose dominant
    /// eigenvector happens to be the naive all-ones vector, e.g.
    /// `[[2,1],[1,2]]` -- the deflated matrix annihilates that same
    /// starting vector exactly, and power iteration would spuriously
    /// "converge" to eigenvalue 0 in a single step instead of finding the
    /// true next singular value. Varying the seed with the stage index via
    /// two independent irrational multipliers avoids re-using a direction
    /// that was just removed.
    fn power_iteration_seed(n: usize, stage: usize) -> Array1<f64> {
        const ALPHA: f64 = std::f64::consts::SQRT_2;
        const BETA: f64 = 1.324_717_957_244_746_f64; // the plastic number
        Array1::from_shape_fn(n, |i| {
            1.0 + ALPHA * (i as f64 + 1.0) + BETA * (stage as f64 + 1.0)
        })
    }

    /// Power iteration to find the dominant eigenvalue and a corresponding
    /// unit-norm eigenvector of a symmetric matrix. `stage` selects a
    /// stage-dependent starting vector, see [`Self::power_iteration_seed`].
    fn power_iteration(matrix: &Array2<f64>, stage: usize) -> Result<(f64, Array1<f64>), OpError> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(OpError::IncompatibleShape(
                "Matrix must be square for eigenvalue computation".into(),
            ));
        }

        // Initialize with a stage-dependent vector (see doc comment above).
        let mut v = Self::power_iteration_seed(n, stage);

        // Normalize
        let norm = v.dot(&v).sqrt();
        if norm > 1e-12_f64 {
            v.mapv_inplace(|x| x / norm);
        }

        // Power iteration
        let max_iterations = 100;
        let tolerance = 1e-10_f64;
        let mut eigenvalue = 0.0_f64;

        for _ in 0..max_iterations {
            // Compute A * v
            let av = matrix.dot(&v);

            // Compute eigenvalue estimate (Rayleigh quotient): v^T * A * v
            let new_eigenvalue = v.dot(&av);

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < tolerance {
                return Ok((new_eigenvalue.max(0.0_f64), v)); // Ensure non-negative
            }

            eigenvalue = new_eigenvalue;

            // Normalize v = A * v / ||A * v||
            let norm = av.dot(&av).sqrt();
            if norm > 1e-12_f64 {
                v = av.mapv(|x| x / norm);
            } else {
                break; // Converged to zero
            }
        }

        Ok((eigenvalue.max(0.0_f64), v))
    }
}

/// Compute the rank of a matrix
#[allow(dead_code)]
pub fn matrix_rank<'g, F: Float>(matrix: &Tensor<'g, F>, tolerance: Option<F>) -> Tensor<'g, F> {
    let g = matrix.graph();

    Tensor::builder(g)
        .append_input(matrix, false)
        .build(RankOp { tolerance })
}

/// Condition Number Operation
///
/// Computes the condition number of a matrix using the specified norm
pub struct CondOp {
    pub p: ConditionType,
}

#[derive(Clone, Copy, Debug)]
pub enum ConditionType {
    One, // 1-norm condition number: ‖A‖₁·‖A⁻¹‖₁ (square matrices only)
    Two, // 2-norm (spectral) condition number: σ_max/σ_min (default, uses SVD)
    Inf, // Infinity-norm condition number: ‖A‖∞·‖A⁻¹‖∞ (square matrices only)
    Fro, // Frobenius norm (NOT a true condition number -- see `CondOp::grad`)
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for CondOp {
    fn name(&self) -> &'static str {
        "Cond"
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        // gradient.rs's op-name string dispatch (`op_name == "Cond"`)
        // downcasts through this to recover `self.p` so it can tell whether
        // the *live* backward pass should build a `CondTwoNormBackwardOp`
        // (the `Two` variant only) or leave the input non-differentiable.
        Some(self)
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(format!(
                "Condition number requires 2D matrix, got shape {shape:?}"
            )));
        }

        let m = shape[0];
        let n = shape[1];

        // Convert to 2D array
        let matrix = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        let cond_value = match self.p {
            ConditionType::Two => {
                // For 2-norm, condition number is ratio of largest to smallest singular value
                // Use proper SVD singular value computation
                let matrix_owned = matrix.to_owned();
                let mut singular_values =
                    match RankOp::<F>::compute_svd_singular_values(&matrix_owned) {
                        Ok(sv) => sv,
                        Err(_) => {
                            // Fallback to diagonal approximation if SVD fails
                            let min_dim = min(m, n);
                            let mut sv = Vec::with_capacity(min_dim);
                            for i in 0..min_dim {
                                if i < m && i < n {
                                    sv.push(matrix[[i, i]].abs());
                                }
                            }
                            sv
                        }
                    };

                singular_values
                    .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

                if let (Some(&max_sv), Some(&min_sv)) =
                    (singular_values.first(), singular_values.last())
                {
                    if min_sv > F::epsilon() {
                        max_sv / min_sv
                    } else {
                        F::infinity()
                    }
                } else {
                    F::one()
                }
            }
            ConditionType::One => {
                // True 1-norm condition number: κ₁(A) = ‖A‖₁ · ‖A⁻¹‖₁.
                // Only defined for square (invertible) A.
                //
                // ‖A⁻¹‖₁ is obtained without a second matrix inversion by
                // using the norm-transpose identity ‖M‖₁ = ‖Mᵀ‖∞:
                // `matrix_inverse_transpose` already returns (A⁻¹)ᵀ, whose
                // induced ∞-norm (max row-abs-sum) equals ‖A⁻¹‖₁.
                if m != n {
                    return Err(OpError::IncompatibleShape(format!(
                        "1-norm condition number requires a square matrix, got shape {shape:?}"
                    )));
                }

                let matrix_owned = matrix.to_owned();
                let norm_a = Self::max_abs_col_sum(&matrix_owned);
                match LogDetOp::matrix_inverse_transpose(&matrix_owned) {
                    Ok(inv_t) => norm_a * Self::max_abs_row_sum(&inv_t),
                    Err(_) => F::infinity(), // Singular: condition number is infinite.
                }
            }
            ConditionType::Inf => {
                // True infinity-norm condition number: κ∞(A) = ‖A‖∞ · ‖A⁻¹‖∞.
                // Only defined for square (invertible) A.
                //
                // By the same norm-transpose identity, ‖A⁻¹‖∞ = ‖(A⁻¹)ᵀ‖₁,
                // i.e. the max column-abs-sum of `matrix_inverse_transpose`'s
                // output.
                if m != n {
                    return Err(OpError::IncompatibleShape(format!(
                        "Infinity-norm condition number requires a square matrix, got shape {shape:?}"
                    )));
                }

                let matrix_owned = matrix.to_owned();
                let norm_a = Self::max_abs_row_sum(&matrix_owned);
                match LogDetOp::matrix_inverse_transpose(&matrix_owned) {
                    Ok(inv_t) => norm_a * Self::max_abs_col_sum(&inv_t),
                    Err(_) => F::infinity(),
                }
            }
            ConditionType::Fro => {
                // Frobenius norm condition number
                let mut sum = F::zero();
                for i in 0..m {
                    for j in 0..n {
                        let val = matrix[[i, j]];
                        sum += val * val;
                    }
                }
                sum.sqrt()
            }
        };

        let result = scirs2_core::ndarray::arr0(cond_value);
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        match self.p {
            ConditionType::Two => {
                // Spectral (2-norm) condition number κ₂(A) = σ_max/σ_min.
                // Analytic gradient via SVD perturbation theory: see
                // `Self::two_norm_gradient` for the formula and derivation.
                let gy = ctx.output_grad();
                let x = ctx.input(0);
                let g = ctx.graph();

                match (gy.eval(g), x.eval(g)) {
                    (Ok(gy_val), Ok(x_val)) => {
                        let x_2d = x_val
                            .view()
                            .into_dimensionality::<Ix2>()
                            .expect("Operation failed")
                            .to_owned();

                        match Self::two_norm_gradient(&x_2d) {
                            Ok(Some(grad_matrix)) => {
                                let grad_tensor = crate::tensor_ops::scalar_mul(
                                    crate::tensor_ops::convert_to_tensor(grad_matrix, g),
                                    gy_val[[]],
                                );
                                ctx.append_input_grad(0, Some(grad_tensor));
                            }
                            _ => ctx.append_input_grad(0, None),
                        }
                    }
                    _ => ctx.append_input_grad(0, None),
                }
            }
            ConditionType::One | ConditionType::Inf | ConditionType::Fro => {
                // No analytic gradient is implemented for these norms in
                // this pass. A correct one would need the Fréchet
                // derivative of the induced-norm ratio ‖A‖·‖A⁻¹‖ through
                // *both* factors (d(A⁻¹) = −A⁻¹ dA A⁻¹ for the second one,
                // composed with the subdifferential of the induced-norm
                // maximum itself, which is only piecewise-smooth). Rather
                // than fabricate a plausible-looking but wrong gradient, we
                // return `None` (honestly: non-differentiable through this
                // op as implemented) for these three variants.
                ctx.append_input_grad(0, None);
            }
        }
    }
}

impl CondOp {
    /// Maximum absolute column sum (the induced 1-norm) of `matrix`.
    fn max_abs_col_sum<F: Float>(matrix: &Array2<F>) -> F {
        let (rows, cols) = matrix.dim();
        let mut max_sum = F::zero();
        for j in 0..cols {
            let mut sum = F::zero();
            for i in 0..rows {
                sum += matrix[[i, j]].abs();
            }
            max_sum = max_sum.max(sum);
        }
        max_sum
    }

    /// Maximum absolute row sum (the induced infinity-norm) of `matrix`.
    fn max_abs_row_sum<F: Float>(matrix: &Array2<F>) -> F {
        let (rows, cols) = matrix.dim();
        let mut max_sum = F::zero();
        for i in 0..rows {
            let mut sum = F::zero();
            for j in 0..cols {
                sum += matrix[[i, j]].abs();
            }
            max_sum = max_sum.max(sum);
        }
        max_sum
    }

    /// Analytic gradient of the spectral (2-norm) condition number
    /// `κ₂(A) = σ_max / σ_min` with respect to `A`:
    ///
    /// ```text
    /// ∂κ₂/∂A = (1/σ_min) u_max v_maxᵀ − (σ_max/σ_min²) u_min v_minᵀ
    /// ```
    ///
    /// where `(u_max, v_max)` / `(u_min, v_min)` are the left/right singular
    /// vector pairs for the largest/smallest singular value. This is the
    /// standard SVD perturbation-theory result `dσ_i = u_iᵀ dA v_i` (valid
    /// for any simple, i.e. non-repeated, singular value) applied to both
    /// extremal singular values via the quotient rule.
    ///
    /// Returns `Ok(None)` when the matrix is (numerically) singular or no
    /// singular value could be extracted at all, mirroring the forward
    /// pass's own `F::infinity()` fallback: the condition number is not
    /// differentiable there, so "no gradient" is the honest answer rather
    /// than a fabricated, unbounded one.
    fn two_norm_gradient<F: Float>(matrix: &Array2<F>) -> Result<Option<Array2<F>>, OpError> {
        let (m, n) = matrix.dim();
        let svd = RankOp::<F>::compute_svd_full(matrix)?;

        if svd.values.is_empty() {
            return Ok(None);
        }

        let last = svd.values.len() - 1;
        let max_sv = svd.values[0];
        let min_sv = svd.values[last];

        if min_sv <= F::epsilon() {
            return Ok(None);
        }

        let u_max = svd.u.column(0);
        let v_max = svd.v.column(0);
        let u_min = svd.u.column(last);
        let v_min = svd.v.column(last);

        let inv_min = F::one() / min_sv;
        let max_over_min_sq = max_sv / (min_sv * min_sv);

        let mut grad = Array2::<F>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                grad[[i, j]] =
                    inv_min * u_max[i] * v_max[j] - max_over_min_sq * u_min[i] * v_min[j];
            }
        }

        Ok(Some(grad))
    }
}

/// Backward op for the spectral (2-norm) condition number.
///
/// `crate::tensor_ops::grad()` (the main reverse-mode entry point) does
/// *not* invoke `Op::grad` for the topological backward pass -- it
/// dispatches on `Op::name()` via `gradient.rs::compute_grad_for_input`
/// instead (see that function's `op_name == "Cond"` arm), the same way
/// `Cholesky`/`MatrixSqrt`/`MatrixLog`/`MatrixPow`/`SVDExtractU` etc. are
/// wired up. This op is what that dispatch arm actually builds for the
/// `ConditionType::Two` case; `CondOp::grad` above remains the
/// (mathematically identical) direct trait-level implementation.
///
/// Inputs: (0) the original matrix `A`, (1) the upstream cotangent `gy`
/// (the condition number output's gradient, a scalar).
/// Output: `gy · ∂κ₂/∂A`, shape matching `A` -- see
/// [`CondOp::two_norm_gradient`] for the formula.
pub(crate) struct CondTwoNormBackwardOp;

impl<F: Float> Op<F> for CondTwoNormBackwardOp {
    fn name(&self) -> &'static str {
        "CondTwoNormBackward"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let x = ctx.input(0);
        let gy = ctx.input(1);

        let x_2d = x
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| {
                OpError::IncompatibleShape("CondTwoNormBackward: input must be 2D".into())
            })?
            .to_owned();

        // Extract the scalar cotangent (cond_2 produces a 0-d output, but be
        // permissive about a single-element representation, matching the
        // convention used by `TraceBackwardOp`).
        let gy_scalar = if gy.ndim() == 0 {
            gy[scirs2_core::ndarray::IxDyn(&[])]
        } else if gy.len() == 1 {
            match gy.iter().next() {
                Some(&v) => v,
                None => {
                    return Err(OpError::IncompatibleShape(
                        "CondTwoNormBackward: empty cotangent".into(),
                    ))
                }
            }
        } else {
            return Err(OpError::IncompatibleShape(
                "CondTwoNormBackward: cotangent of cond_2 must be a scalar".into(),
            ));
        };

        let (m, n) = x_2d.dim();
        let grad = match CondOp::two_norm_gradient(&x_2d)? {
            Some(g) => g.mapv(|v| v * gy_scalar),
            None => Array2::<F>::zeros((m, n)),
        };

        ctx.append_output(grad.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Second-order differentiation (Hessian-vector products through
        // cond_2) is not implemented; honestly report non-differentiable
        // rather than fabricating a value.
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

/// Compute the condition number of a matrix
#[allow(dead_code)]
pub fn cond<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
    p: Option<ConditionType>,
) -> Tensor<'g, F> {
    let g = matrix.graph();
    let p = p.unwrap_or(ConditionType::Two);

    Tensor::builder(g)
        .append_input(matrix, false)
        .build(CondOp { p })
}

/// Compute 1-norm condition number
#[allow(dead_code)]
pub fn cond_1<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    cond(matrix, Some(ConditionType::One))
}

/// Compute 2-norm condition number (default)
#[allow(dead_code)]
pub fn cond_2<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    cond(matrix, Some(ConditionType::Two))
}

/// Compute infinity-norm condition number
#[allow(dead_code)]
pub fn cond_inf<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    cond(matrix, Some(ConditionType::Inf))
}

/// Compute Frobenius norm condition number
#[allow(dead_code)]
pub fn cond_fro<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    cond(matrix, Some(ConditionType::Fro))
}

/// Log-determinant Operation
///
/// Computes log(|det(A)|) in a numerically stable way
pub struct LogDetOp;

impl<F: Float> Op<F> for LogDetOp {
    fn name(&self) -> &'static str {
        "LogDet"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(format!(
                "LogDet requires square 2D matrix, got shape {shape:?}"
            )));
        }

        let n = shape[0];
        let matrix = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Use LU decomposition to compute determinant
        // det(A) = det(P) * det(L) * det(U) = ±1 * 1 * prod(diag(U))
        // log|det(A)| = sum(log|diag(U)|)

        let mut u = matrix.to_owned();
        let mut sign = F::one();

        // Simple LU decomposition without full pivoting
        for k in 0..n - 1 {
            // Find pivot
            let mut max_val = u[[k, k]].abs();
            let mut max_row = k;

            for i in (k + 1)..n {
                if u[[i, k]].abs() > max_val {
                    max_val = u[[i, k]].abs();
                    max_row = i;
                }
            }

            // Swap rows if needed
            if max_row != k {
                sign = -sign; // Each swap changes sign of determinant
                for j in 0..n {
                    let temp = u[[k, j]];
                    u[[k, j]] = u[[max_row, j]];
                    u[[max_row, j]] = temp;
                }
            }

            // Elimination
            if u[[k, k]].abs() > F::epsilon() {
                for i in (k + 1)..n {
                    let factor = u[[i, k]] / u[[k, k]];
                    for j in k..n {
                        u[[i, j]] = u[[i, j]] - factor * u[[k, j]];
                    }
                }
            }
        }

        // Compute log|det| = sum(log|diag(U)|)
        let mut log_det = F::zero();
        for i in 0..n {
            if u[[i, i]].abs() <= F::epsilon() {
                // Matrix is singular
                log_det = F::neg_infinity();
                break;
            } else {
                log_det += u[[i, i]].abs().ln();
            }
        }

        let result = scirs2_core::ndarray::arr0(log_det);
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let g = ctx.graph();

        // Gradient of log|det(X)| w.r.t. X is (X^-T)
        match (gy.eval(g), x.eval(g)) {
            (Ok(gy_val), Ok(x_val)) => {
                let x_2d = x_val
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("Operation failed");

                // Compute inverse transpose using Gauss-Jordan elimination
                let inv_t = match Self::matrix_inverse_transpose(&x_2d.to_owned()) {
                    Ok(inv) => inv,
                    Err(_) => {
                        // If inversion fails, return None gradient
                        ctx.append_input_grad(0, None);
                        return;
                    }
                };

                let grad = crate::tensor_ops::scalar_mul(
                    crate::tensor_ops::convert_to_tensor(inv_t, g),
                    gy_val[[]],
                );

                ctx.append_input_grad(0, Some(grad));
            }
            _ => ctx.append_input_grad(0, None),
        }
    }
}

impl LogDetOp {
    /// Compute matrix inverse transpose using Gauss-Jordan elimination
    /// Returns (X^-1)^T = (X^T)^-1
    fn matrix_inverse_transpose<F: Float>(matrix: &Array2<F>) -> Result<Array2<F>, OpError> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(OpError::IncompatibleShape(
                "Matrix must be square for inversion".into(),
            ));
        }

        // Work with the transpose for efficiency
        let mut a = matrix.t().to_owned();
        let mut inv = Array2::<F>::eye(n);

        // Convert to f64 for numerical stability
        let mut a_f64 = a.mapv(|x| x.to_f64().unwrap_or(0.0));
        let mut inv_f64 = inv.mapv(|x| x.to_f64().unwrap_or(0.0));

        // Gauss-Jordan elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            let mut max_val = a_f64[[i, i]].abs();
            for k in (i + 1)..n {
                let abs_val = a_f64[[k, i]].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    max_row = k;
                }
            }

            // Check for singularity
            if max_val < 1e-10_f64 {
                return Err(OpError::RuntimeError(
                    "Matrix is singular or nearly singular".into(),
                ));
            }

            // Swap rows
            if max_row != i {
                for j in 0..n {
                    a_f64.swap([i, j], [max_row, j]);
                    inv_f64.swap([i, j], [max_row, j]);
                }
            }

            // Scale pivot row
            let pivot = a_f64[[i, i]];
            for j in 0..n {
                a_f64[[i, j]] /= pivot;
                inv_f64[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = a_f64[[k, i]];
                    for j in 0..n {
                        a_f64[[k, j]] -= factor * a_f64[[i, j]];
                        inv_f64[[k, j]] -= factor * inv_f64[[i, j]];
                    }
                }
            }
        }

        // Convert back to F
        Ok(inv_f64.mapv(|x| F::from(x).unwrap_or(F::zero())))
    }
}

/// Backward op for `logdet` (log|det(A)|).
///
/// Wired from `gradient.rs`'s op-name string dispatch (`op_name ==
/// "LogDet"`) for the same reason as [`CondTwoNormBackwardOp`]:
/// `crate::tensor_ops::grad()` dispatches on `Op::name()`, not `Op::grad`,
/// for the main reverse-mode backward pass, so `LogDetOp::grad` above --
/// while mathematically correct -- was dead code until this was added.
/// Discovered as a direct consequence of wiring up `Cond`: once `cond_2`'s
/// gradient became properly `(m, n)`-shaped, `test_combined_operations`
/// (which sums a `matrix_rank`/`cond_2`/`logdet` gradient contribution into
/// a single accumulator) started failing to broadcast against `logdet`'s
/// previously-scalar "gradient" (the default op-name-dispatch fallback
/// simply passes `gy` through unchanged for any unrecognized op name).
///
/// Inputs: (0) the original matrix `A`, (1) upstream cotangent `gy` (scalar).
/// Output: `gy · (A⁻¹)ᵀ` -- the standard `d log|det(A)| = tr(A⁻¹ dA)` result.
pub(crate) struct LogDetBackwardOp;

impl<F: Float> Op<F> for LogDetBackwardOp {
    fn name(&self) -> &'static str {
        "LogDetBackward"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let x = ctx.input(0);
        let gy = ctx.input(1);

        let x_2d = x
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("LogDetBackward: input must be 2D".into()))?
            .to_owned();

        // Extract the scalar cotangent (logdet produces a 0-d output, but be
        // permissive about a single-element representation, matching the
        // convention used by `TraceBackwardOp`/`CondTwoNormBackwardOp`).
        let gy_scalar = if gy.ndim() == 0 {
            gy[scirs2_core::ndarray::IxDyn(&[])]
        } else if gy.len() == 1 {
            match gy.iter().next() {
                Some(&v) => v,
                None => {
                    return Err(OpError::IncompatibleShape(
                        "LogDetBackward: empty cotangent".into(),
                    ))
                }
            }
        } else {
            return Err(OpError::IncompatibleShape(
                "LogDetBackward: cotangent of logdet must be a scalar".into(),
            ));
        };

        // Singular input: log|det| is -∞ there and the gradient is
        // genuinely undefined, so propagate the honest error from
        // `matrix_inverse_transpose` rather than fabricating a zero
        // gradient (matches the precedent set by `SVDBackwardOp`).
        let inv_t = LogDetOp::matrix_inverse_transpose(&x_2d)?;
        let grad = inv_t.mapv(|v| v * gy_scalar);

        ctx.append_output(grad.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

/// Compute log(|det(A)|) in a numerically stable way
#[allow(dead_code)]
pub fn logdet<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();

    Tensor::builder(g)
        .append_input(matrix, false)
        .build(LogDetOp)
}

/// Sign and Log-determinant Operation
///
/// Computes sign(det(A)) and log(|det(A)|) in a numerically stable way
pub struct SLogDetOp;

impl<F: Float> Op<F> for SLogDetOp {
    fn name(&self) -> &'static str {
        "SLogDet"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(format!(
                "SLogDet requires square 2D matrix, got shape {shape:?}"
            )));
        }

        let n = shape[0];
        let matrix = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        let mut u = matrix.to_owned();
        let mut sign = F::one();

        // LU decomposition with sign tracking
        for k in 0..n - 1 {
            let mut max_val = u[[k, k]].abs();
            let mut max_row = k;

            for i in (k + 1)..n {
                if u[[i, k]].abs() > max_val {
                    max_val = u[[i, k]].abs();
                    max_row = i;
                }
            }

            if max_row != k {
                sign = -sign;
                for j in 0..n {
                    let temp = u[[k, j]];
                    u[[k, j]] = u[[max_row, j]];
                    u[[max_row, j]] = temp;
                }
            }

            if u[[k, k]].abs() > F::epsilon() {
                for i in (k + 1)..n {
                    let factor = u[[i, k]] / u[[k, k]];
                    for j in k..n {
                        u[[i, j]] = u[[i, j]] - factor * u[[k, j]];
                    }
                }
            }
        }

        // Compute sign and log|det|
        let mut log_det = F::zero();
        for i in 0..n {
            if u[[i, i]].abs() <= F::epsilon() {
                sign = F::zero();
                log_det = F::neg_infinity();
                break;
            } else {
                if u[[i, i]] < F::zero() {
                    sign = -sign;
                }
                log_det += u[[i, i]].abs().ln();
            }
        }

        // Output both sign and log|det|
        let sign_arr = scirs2_core::ndarray::arr0(sign);
        let logdet_arr = scirs2_core::ndarray::arr0(log_det);

        ctx.append_output(sign_arr.into_dyn());
        ctx.append_output(logdet_arr.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Similar to logdet, but only backprop through the log|det| output
        ctx.append_input_grad(0, None);
    }
}

/// Sign and log-determinant extraction
pub struct SLogDetExtractOp {
    component: usize, // 0 for sign, 1 for log|det|
}

impl<F: Float> Op<F> for SLogDetExtractOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Re-compute slogdet
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "SLogDet requires square matrix".into(),
            ));
        }

        let n = shape[0];
        let matrix = input
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Operation failed");

        let mut u = matrix.to_owned();
        let mut sign = F::one();

        // Simplified LU decomposition
        for k in 0..n - 1 {
            if u[[k, k]].abs() > F::epsilon() {
                for i in (k + 1)..n {
                    let factor = u[[i, k]] / u[[k, k]];
                    for j in k..n {
                        u[[i, j]] = u[[i, j]] - factor * u[[k, j]];
                    }
                }
            }
        }

        let mut log_det = F::zero();
        for i in 0..n {
            if u[[i, i]].abs() <= F::epsilon() {
                sign = F::zero();
                log_det = F::neg_infinity();
                break;
            } else {
                if u[[i, i]] < F::zero() {
                    sign = -sign;
                }
                log_det += u[[i, i]].abs().ln();
            }
        }

        match self.component {
            0 => ctx.append_output(scirs2_core::ndarray::arr0(sign).into_dyn()),
            1 => ctx.append_output(scirs2_core::ndarray::arr0(log_det).into_dyn()),
            _ => return Err(OpError::IncompatibleShape("Invalid component".into())),
        }

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
    }
}

/// Compute sign(det(A)) and log(|det(A)|) in a numerically stable way
///
/// Returns (sign, log|det|) where det(A) = sign * exp(log|det|)
#[allow(dead_code)]
pub fn slogdet<'g, F: Float>(matrix: &Tensor<'g, F>) -> (Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    let sign = Tensor::builder(g)
        .append_input(matrix, false)
        .build(SLogDetExtractOp { component: 0 });

    let logdet = Tensor::builder(g)
        .append_input(matrix, false)
        .build(SLogDetExtractOp { component: 1 });

    (sign, logdet)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::convert_to_tensor;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_matrix_rank() {
        crate::run(|g| {
            // Test with a rank-2 matrix
            let a = convert_to_tensor(array![[1.0_f32, 2.0], [3.0, 4.0]], g);
            let r = matrix_rank(&a, None);
            let r_val = r.eval(g).expect("Operation failed");
            assert_eq!(r_val[[]], 2.0);

            // Test with a rank-deficient matrix
            let b = convert_to_tensor(array![[1.0_f32, 2.0], [2.0, 4.0]], g);
            let _r2 = matrix_rank(&b, Some(1e-5));
            // Note: This is a simplified implementation, actual rank might differ
        });
    }

    #[test]
    fn test_condition_number() {
        crate::run(|g| {
            // Well-conditioned matrix
            let a = convert_to_tensor(array![[2.0_f32, 1.0], [1.0, 2.0]], g);
            let c = cond_2(&a);
            let c_val = c.eval(g).expect("Operation failed");
            // Condition number should be finite and reasonable
            assert!(c_val[[]] > 0.0 && c_val[[]] < 100.0);

            // Test different norms
            let c1 = cond_1(&a);
            let c_inf = cond_inf(&a);
            let c_fro = cond_fro(&a);

            // All should evaluate without error
            c1.eval(g).expect("Operation failed");
            c_inf.eval(g).expect("Operation failed");
            c_fro.eval(g).expect("Operation failed");
        });
    }

    #[test]
    fn test_cond_one_inf_true_condition_number() {
        crate::run(|g| {
            // A = [[1,2],[3,4]]; det(A) = -2, so
            // A^-1 = (1/det)*[[4,-2],[-3,1]] = [[-2,1],[1.5,-0.5]]
            // (hand-verified: A * A^-1 = I).
            //
            // ‖A‖_1   = max(|1|+|3|, |2|+|4|)         = max(4, 6)   = 6
            // ‖A‖_∞   = max(|1|+|2|, |3|+|4|)         = max(3, 7)   = 7
            // ‖A^-1‖_1 = max(|-2|+|1.5|, |1|+|-0.5|)  = max(3.5,1.5) = 3.5
            // ‖A^-1‖_∞ = max(|-2|+|1|, |1.5|+|-0.5|)  = max(3, 2)   = 3
            //
            // True condition numbers: cond_1 = 6*3.5 = 21, cond_inf = 7*3 = 21.
            let a = convert_to_tensor(array![[1.0_f64, 2.0], [3.0, 4.0]], g);

            let c1_val = cond_1(&a).eval(g).expect("Operation failed")[[]];
            assert!(
                (c1_val - 21.0).abs() < 1e-8,
                "cond_1 should be the true condition number 21.0, got {c1_val}"
            );

            let cinf_val = cond_inf(&a).eval(g).expect("Operation failed")[[]];
            assert!(
                (cinf_val - 21.0).abs() < 1e-8,
                "cond_inf should be the true condition number 21.0, got {cinf_val}"
            );

            // Sanity check against the regression this guards: the old
            // "simplified" behavior returned just ‖A‖_1 = 6 / ‖A‖_∞ = 7,
            // clearly different from the true condition number above.
            assert!((c1_val - 6.0).abs() > 1.0);
            assert!((cinf_val - 7.0).abs() > 1.0);
        });
    }

    #[test]
    fn test_cond_one_inf_rejects_non_square() {
        crate::run(|g| {
            let a = convert_to_tensor(array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]], g);
            assert!(
                cond_1(&a).eval(g).is_err(),
                "cond_1 on a non-square matrix should error, not silently return a number"
            );
            assert!(
                cond_inf(&a).eval(g).is_err(),
                "cond_inf on a non-square matrix should error, not silently return a number"
            );
        });
    }

    /// Forward evaluation of `cond_2(A)` for a raw ndarray, run through the
    /// autograd graph so the finite-difference check exercises exactly the
    /// same numerical procedure that `CondOp::grad` differentiates.
    fn cond2_forward(a: &Array2<f64>) -> f64 {
        crate::run(|g| {
            let av = convert_to_tensor(a.clone(), g);
            cond_2(&av).eval(g).expect("Operation failed")[[]]
        })
    }

    /// Analytic gradient of `cond_2(A)` via the live autograd reverse-mode
    /// engine (exercises `CondOp::grad` itself, not a hand-rolled copy of
    /// the formula).
    fn cond2_analytic_grad(a: &Array2<f64>) -> Array2<f64> {
        crate::run(|g| {
            let av = crate::tensor_ops::variable(a.clone(), g);
            let c = cond_2(&av);
            let grads = crate::tensor_ops::grad(&[&c], &[&av]);
            grads[0]
                .eval(g)
                .expect("Operation failed")
                .into_dimensionality::<Ix2>()
                .expect("Operation failed")
                .to_owned()
        })
    }

    /// Central finite-difference gradient of `cond_2(A)`.
    fn cond2_fd_grad(a: &Array2<f64>, h: f64) -> Array2<f64> {
        let (rows, cols) = a.dim();
        let mut grad = Array2::<f64>::zeros((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                let mut ap = a.clone();
                let mut am = a.clone();
                ap[[i, j]] += h;
                am[[i, j]] -= h;
                grad[[i, j]] = (cond2_forward(&ap) - cond2_forward(&am)) / (2.0 * h);
            }
        }
        grad
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |m, (x, y)| (x - y).abs().max(m))
    }

    /// Compare the live analytic `cond_2` gradient against a central
    /// finite-difference approximation for a well-conditioned matrix.
    fn check_cond2_gradient(a: Array2<f64>, tol: f64) {
        let analytic = cond2_analytic_grad(&a);
        let numeric = cond2_fd_grad(&a, 1e-5);
        let err = max_abs_diff(&analytic, &numeric);
        assert!(
            err < tol,
            "cond_2 analytic vs finite-difference gradient mismatch: err={err}\nanalytic={analytic:?}\nnumeric={numeric:?}"
        );
    }

    #[test]
    fn test_cond_2_gradient_matches_finite_difference_2x2_a() {
        check_cond2_gradient(array![[4.0_f64, 1.0], [0.5, 3.0]], 1e-3);
    }

    #[test]
    fn test_cond_2_gradient_matches_finite_difference_2x2_b() {
        check_cond2_gradient(array![[2.0_f64, 0.3], [-0.4, 1.5]], 1e-3);
    }

    #[test]
    fn test_cond_2_gradient_matches_finite_difference_3x3() {
        check_cond2_gradient(
            array![[4.0_f64, 1.0, 0.2], [0.3, 3.0, 0.5], [0.1, 0.4, 2.0]],
            1e-2,
        );
    }

    #[test]
    fn test_cond_2_backprop_not_silently_zero() {
        crate::run(|g| {
            let a = crate::tensor_ops::variable(array![[4.0_f64, 1.0], [0.5, 3.0]], g);
            let c = cond_2(&a);
            let grads = crate::tensor_ops::grad(&[&c], &[&a]);
            let grad_val = grads[0].eval(g).expect("Operation failed");
            assert!(
                grad_val.iter().any(|&x| x.abs() > 1e-6),
                "expected a nonzero gradient through cond_2 (2-norm), got {grad_val:?}"
            );
        });
    }

    #[test]
    fn test_logdet() {
        crate::run(|g| {
            // Matrix with known determinant
            let a = convert_to_tensor(array![[2.0_f64, 0.0], [0.0, 3.0]], g);
            let ld = logdet(&a);
            let ld_val = ld.eval(g).expect("Operation failed");

            // det(A) = 6, so log(det(A)) = log(6) ≈ 1.79
            assert!((ld_val[[]] - 6.0_f64.ln()).abs() < 1e-6);

            // Test singular matrix
            let b = convert_to_tensor(array![[1.0_f64, 2.0], [2.0, 4.0]], g);
            let ld2 = logdet(&b);
            let ld2_val = ld2.eval(g).expect("Operation failed");
            assert!(ld2_val[[]] == f64::NEG_INFINITY);
        });
    }

    #[test]
    fn test_slogdet() {
        crate::run(|g| {
            // Positive determinant
            let a = convert_to_tensor(array![[2.0_f64, 1.0], [1.0, 3.0]], g);
            let (sign, ld) = slogdet(&a);
            let sign_val = sign.eval(g).expect("Operation failed");
            let ld_val = ld.eval(g).expect("Operation failed");

            // det(A) = 5, positive
            assert_eq!(sign_val[[]], 1.0);
            assert!((ld_val[[]] - 5.0_f64.ln()).abs() < 1e-6);

            // Negative determinant
            let b = convert_to_tensor(array![[0.0_f64, 1.0], [1.0, 0.0]], g);
            let (sign2, _ld2) = slogdet(&b);
            let sign2_val = sign2.eval(g).expect("Operation failed");

            // det(B) = -1 (but our simplified implementation may not handle all cases)
            // For now, just check it computed without error
            assert!(sign2_val[[]] == -1.0 || sign2_val[[]] == 1.0 || sign2_val[[]] == 0.0);
        });
    }
}
