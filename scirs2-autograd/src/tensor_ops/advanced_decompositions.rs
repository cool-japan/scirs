use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::matrix_calculus;
use crate::Float;
use scirs2_core::ndarray::{s, Array1, Array2, Ix2};
use scirs2_core::numeric::FromPrimitive;

// Type aliases to reduce complexity
type SVDResult<F> = Result<(Array2<F>, Array1<F>, Array2<F>), OpError>;
#[allow(dead_code)]
type QRResult<F> = Result<(Array2<F>, Array2<F>), OpError>;
type QRPivotResult<F> = Result<(Array2<F>, Array2<F>, Array1<F>), OpError>;
#[allow(dead_code)]
type EigenResult<F> = Result<(Array2<F>, Array2<F>, Array1<F>), OpError>;

/// Improved SVD using Jacobi algorithm for better numerical stability
pub struct JacobiSVDOp {
    full_matrices: bool,
}

/// Extraction operator for Jacobi SVD components
pub struct SVDJacobiExtractOp {
    component: usize,
    full_matrices: bool,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for SVDJacobiExtractOp {
    fn name(&self) -> &'static str {
        match self.component {
            0 => "SVDJacobiExtractU",
            1 => "SVDJacobiExtractS",
            2 => "SVDJacobiExtractVt",
            _ => "SVDJacobiExtractUnknown",
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Re-run the decomposition on the original matrix and select one factor.
        //
        // This node used to unconditionally return
        // `Err("SVD extraction should be handled by parent op")`, and no parent op ever
        // did: `svd_jacobi()` returned three tensors none of which could be evaluated.
        // Each component now recomputes the factorisation from the matrix it is given
        // (the same design `decomposition_ops::SVDExtractOp` uses), which costs three
        // decompositions but is correct and needs no multi-output plumbing — a graph node
        // exposes only its first output.
        let input = ctx.input(0);
        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Jacobi SVD requires a 2-D matrix".into()))?;
        let (u, sigma, vt) = compute_svd_jacobi(&input_2d, self.full_matrices)?;
        match self.component {
            0 => ctx.append_output(u.into_dyn()),
            1 => ctx.append_output(sigma.into_dyn()),
            2 => ctx.append_output(vt.into_dyn()),
            other => {
                return Err(OpError::Other(format!(
                    "Jacobi SVD: component index {other} is out of range (expected 0, 1 or 2)"
                )))
            }
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        if self.full_matrices {
            // The extra `m - k` columns of a full `U` span the orthogonal complement of
            // the range of `A` and are only determined up to an arbitrary rotation, so
            // they have no well-defined derivative.
            matrix_calculus::append_unsupported_grad(
                ctx,
                "svd_jacobi(full_matrices = true): the silent columns of a full U/Vt are \
                 determined only up to a rotation of the null space and have no unique \
                 derivative. Use full_matrices = false for a differentiable SVD."
                    .into(),
            );
            return;
        }
        // Reduced SVD: the exact analytic VJP is already implemented in
        // `decomposition_ops::SVDBackwardOp` (Townsend / Wan-Zhang), which recomputes the
        // decomposition of A and applies the formula for the requested component.
        let gy = *ctx.output_grad();
        let input = *ctx.input(0);
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(input, false)
            .append_input(gy, false)
            .build(crate::tensor_ops::decomposition_ops::SVDBackwardOp {
                component: self.component,
            });
        ctx.append_input_grad(0, Some(gx));
    }
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for JacobiSVDOp {
    fn name(&self) -> &'static str {
        "JacobiSVD"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape("SVD requires 2D matrix".into()));
        }

        let _m = shape[0];
        let _n = shape[1];
        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute SVD using Jacobi algorithm
        let (u, s, vt) = compute_svd_jacobi(&input_2d, self.full_matrices)?;

        // Append outputs
        ctx.append_output(u.into_dyn());
        ctx.append_output(s.into_dyn());
        ctx.append_output(vt.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // This node carries three outputs but a graph node exposes only its first, so it
        // is never the tensor a user differentiates: `svd_jacobi()` builds one
        // `SVDJacobiExtractOp` per factor and those carry the VJP.  Returning `None`
        // (rather than a fabricated zero or a pass-through) keeps that honest.
        ctx.append_input_grad(0, None);
    }
}

/// Randomized SVD for large matrices
pub struct RandomizedSVDOp {
    rank: usize,
    oversampling: usize,
    n_iter: usize,
}

/// Extraction operator for Randomized SVD components
pub struct RandomizedSVDExtractOp {
    component: usize,
    rank: usize,
    oversampling: usize,
    n_iter: usize,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F>
    for RandomizedSVDExtractOp
{
    fn name(&self) -> &'static str {
        match self.component {
            0 => "RandomizedSVDExtractU",
            1 => "RandomizedSVDExtractS",
            2 => "RandomizedSVDExtractVt",
            _ => "RandomizedSVDExtractUnknown",
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // As for `SVDJacobiExtractOp`: recompute and select, instead of the previous
        // unconditional "should be handled by parent op" error that made
        // `randomized_svd()` impossible to evaluate.
        let input = ctx.input(0);
        let input_2d = input.view().into_dimensionality::<Ix2>().map_err(|_| {
            OpError::IncompatibleShape("randomized SVD requires a 2-D matrix".into())
        })?;
        let (u, sigma, vt) =
            compute_randomized_svd(&input_2d, self.rank, self.oversampling, self.n_iter)?;
        match self.component {
            0 => ctx.append_output(u.into_dyn()),
            1 => ctx.append_output(sigma.into_dyn()),
            2 => ctx.append_output(vt.into_dyn()),
            other => {
                return Err(OpError::Other(format!(
                    "randomized SVD: component index {other} is out of range"
                )))
            }
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // A randomised range finder produces a rank-`rank` *approximation* whose factors
        // depend on the sketch as well as on A.  The exact-SVD VJP is therefore not the
        // derivative of this map, and pretending otherwise would report a gradient that
        // is wrong by an amount nobody can bound.  `svd_jacobi(m, false)` is the
        // differentiable entry point.
        matrix_calculus::append_unsupported_grad(
            ctx,
            "randomized_svd: this is a sketched, rank-truncated approximation; its \
             derivative is not the exact SVD VJP and is not implemented. Use \
             svd_jacobi(matrix, false) when a gradient is required."
                .into(),
        );
    }
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for RandomizedSVDOp {
    fn name(&self) -> &'static str {
        "RandomizedSVD"
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
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute randomized SVD
        let (u, s, vt) =
            compute_randomized_svd(&input_2d, self.rank, self.oversampling, self.n_iter)?;

        ctx.append_output(u.into_dyn());
        ctx.append_output(s.into_dyn());
        ctx.append_output(vt.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
    }
}

/// Generalized eigenvalue problem: Ax = λBx
pub struct GeneralizedEigenOp;

/// Extraction operator for Generalized Eigen components
pub struct GeneralizedEigenExtractOp {
    component: usize,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F>
    for GeneralizedEigenExtractOp
{
    fn name(&self) -> &'static str {
        match self.component {
            0 => "GeneralizedEigenExtractValues",
            1 => "GeneralizedEigenExtractVectors",
            _ => "GeneralizedEigenExtractUnknown",
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Inputs are `(A, B)`; the node recomputes the decomposition and selects one
        // component.  Previously this always returned an error, so `generalized_eigen()`
        // produced two tensors that could not be evaluated at all.
        let a_in = ctx.input(0);
        let b_in = ctx.input(1);
        let a = a_in.view().into_dimensionality::<Ix2>().map_err(|_| {
            OpError::IncompatibleShape("generalized eigenproblem: A is not 2-D".into())
        })?;
        let b = b_in.view().into_dimensionality::<Ix2>().map_err(|_| {
            OpError::IncompatibleShape("generalized eigenproblem: B is not 2-D".into())
        })?;
        let (values, vectors) = compute_generalized_eigen(&a, &b)?;
        match self.component {
            0 => ctx.append_output(values.into_dyn()),
            1 => ctx.append_output(vectors.into_dyn()),
            other => {
                return Err(OpError::Other(format!(
                    "generalized eigenproblem: component index {other} is out of range"
                )))
            }
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        if self.component != 0 {
            matrix_calculus::append_unsupported_grad(
                ctx,
                "generalized_eigen: the eigenvector VJP is not implemented (it needs the \
                 1/(lambda_i - lambda_j) coupling matrix and a rule for the arbitrary \
                 scale of each eigenvector). The eigenvalue output is differentiable."
                    .into(),
            );
            return;
        }
        // First-order perturbation of `A v = lambda B v` with B-orthonormal eigenvectors
        // (`v_kᵀ B v_k = 1`) gives `dlambda_k = v_kᵀ (dA - lambda_k dB) v_k`, hence
        //
        //     Ā = V diag(ḡ) Vᵀ            B̄ = -V diag(ḡ · lambda) Vᵀ
        let a = *ctx.input(0);
        let b = *ctx.input(1);
        let gy = *ctx.output_grad();
        let g = ctx.graph();
        let grad_a = Tensor::builder(g)
            .append_input(a, false)
            .append_input(b, false)
            .append_input(gy, false)
            .build(GeneralizedEigenValuesVjpOp { wrt_b: false });
        let grad_b = Tensor::builder(g)
            .append_input(a, false)
            .append_input(b, false)
            .append_input(gy, false)
            .build(GeneralizedEigenValuesVjpOp { wrt_b: true });
        ctx.append_input_grad(0, Some(grad_a));
        ctx.append_input_grad(1, Some(grad_b));
    }
}

/// Backward node of the generalized-eigenvalue extraction.
///
/// Inputs are `(A, B, ḡ)`; the output is `Ā` (`wrt_b = false`) or `B̄` (`wrt_b = true`).
pub struct GeneralizedEigenValuesVjpOp {
    wrt_b: bool,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F>
    for GeneralizedEigenValuesVjpOp
{
    fn name(&self) -> &'static str {
        "GeneralizedEigenValuesVjp"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_in = ctx.input(0);
        let b_in = ctx.input(1);
        let gy_in = ctx.input(2);
        let a = a_in.view().into_dimensionality::<Ix2>().map_err(|_| {
            OpError::IncompatibleShape("generalized eigen backward: A is not 2-D".into())
        })?;
        let b = b_in.view().into_dimensionality::<Ix2>().map_err(|_| {
            OpError::IncompatibleShape("generalized eigen backward: B is not 2-D".into())
        })?;
        let n = a.nrows();
        if gy_in.len() != n {
            return Err(OpError::IncompatibleShape(
                "generalized eigen backward: cotangent length does not match the problem size"
                    .into(),
            ));
        }

        let (values, vectors) = compute_generalized_eigen(&a, &b)?;
        let mut scaled = Array2::<F>::zeros((n, n));
        for (k, gk) in gy_in.iter().enumerate() {
            let weight = if self.wrt_b { -(*gk * values[k]) } else { *gk };
            for i in 0..n {
                scaled[[i, k]] = vectors[[i, k]] * weight;
            }
        }
        ctx.append_output(scaled.dot(&vectors.t()).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        matrix_calculus::append_unsupported_grad(
            ctx,
            "generalized_eigen: second-order differentiation is not implemented".into(),
        );
    }
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for GeneralizedEigenOp {
    fn name(&self) -> &'static str {
        "GeneralizedEigen"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        if a.shape() != b.shape() || a.shape().len() != 2 {
            return Err(OpError::IncompatibleShape(
                "Generalized eigenvalue problem requires two square matrices of same size".into(),
            ));
        }

        let n = a.shape()[0];
        if n != a.shape()[1] {
            return Err(OpError::IncompatibleShape("Matrices must be square".into()));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;
        let b_2d = b
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert B to 2D".into()))?;

        // Compute generalized eigenvalues and eigenvectors
        let (eigenvalues, eigenvectors) = compute_generalized_eigen(&a_2d, &b_2d)?;

        ctx.append_output(eigenvalues.into_dyn());
        ctx.append_output(eigenvectors.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

/// QR decomposition with column pivoting for better numerical stability
pub struct QRPivotOp;

/// Extraction operator for QR Pivot components
pub struct QRPivotExtractOp {
    component: usize,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for QRPivotExtractOp {
    fn name(&self) -> &'static str {
        match self.component {
            0 => "QRPivotExtractQ",
            1 => "QRPivotExtractR",
            2 => "QRPivotExtractP",
            _ => "QRPivotExtractUnknown",
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Recompute-and-select, replacing the previous unconditional error that made
        // `qr_pivot()` return three unevaluable tensors.
        let input = ctx.input(0);
        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("pivoted QR requires a 2-D matrix".into()))?;
        let (q, r, p) = compute_qr_pivot(&input_2d)?;
        match self.component {
            0 => ctx.append_output(q.into_dyn()),
            1 => ctx.append_output(r.into_dyn()),
            2 => ctx.append_output(p.into_dyn()),
            other => {
                return Err(OpError::Other(format!(
                    "pivoted QR: component index {other} is out of range"
                )))
            }
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        if self.component == 2 {
            // The permutation is an integer vector chosen by a discrete argmax over
            // column norms: piecewise constant, so its derivative is zero wherever it is
            // defined.  `None` is the honest answer and keeps a shape-mismatched edge out
            // of the accumulation.
            ctx.append_input_grad(0, None);
            return;
        }
        matrix_calculus::append_unsupported_grad(
            ctx,
            "qr_pivot: the Q/R vector-Jacobian products are not implemented. The factors \
             satisfy A P = Q R, so their VJP is the unpivoted QR rule composed with the \
             column permutation; use qr() (decomposition_ops) for a differentiable QR."
                .into(),
        );
    }
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for QRPivotOp {
    fn name(&self) -> &'static str {
        "QRPivot"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(
                "QR decomposition requires 2D matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D".into()))?;

        // Compute QR with column pivoting
        let (q, r, p) = compute_qr_pivot(&input_2d)?;

        ctx.append_output(q.into_dyn());
        ctx.append_output(r.into_dyn());
        ctx.append_output(p.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
    }
}

// Helper functions

/// Compute SVD using one-sided Jacobi algorithm.
///
/// For an m×n matrix A, computes (U, σ, V^T) such that A = U diag(σ) V^T,
/// where k = min(m, n):
///
/// - U is m×m orthogonal (if full_matrices=true) or m×k (if false)
/// - σ is a length-k vector of non-negative singular values in descending order
/// - V^T is k×n (if full_matrices=false) or n×n (if true)
///
/// Algorithm: one-sided Jacobi — iterates over column pairs (i,j) and applies
/// Givens rotations to columns of A to annihilate A[:,i]·A[:,j].  Accumulates
/// V.  After convergence σ_i = ‖A[:,i]‖ and U[:,i] = A[:,i]/σ_i.
#[allow(dead_code)]
pub(crate) fn compute_svd_jacobi<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
    full_matrices: bool,
) -> SVDResult<F> {
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let k = m.min(n);

    // Work on A (m×n copy) and accumulate V (n×n)
    let mut a = matrix.to_owned();
    // If m > n, we only need the first n columns; keep all m rows.
    let mut v = Array2::<F>::eye(n);

    let max_sweeps = 30;
    let tol = F::epsilon() * F::from_f64(10.0).unwrap_or_else(|| F::from(10.0).unwrap_or(F::one()));

    // One-sided Jacobi: sweep over all column pairs
    'outer: for _sweep in 0..max_sweeps {
        let mut converged = true;

        for i in 0..n {
            for j in (i + 1)..n {
                // Compute entries of A[:,i]^T A[:,j] sub-matrix
                let aii = col_dot_f(&a.view(), i, i, m);
                let aij = col_dot_f(&a.view(), i, j, m);
                let ajj = col_dot_f(&a.view(), j, j, m);

                // Off-diagonal mass; skip if already small
                if aij.abs() <= tol * (aii * ajj).sqrt() {
                    continue;
                }

                converged = false;

                // Compute Jacobi rotation angle for the symmetric 2×2
                // [[aii, aij],[aij, ajj]]
                let tau = (ajj - aii) / (F::from(2.0).unwrap_or(F::one()) * aij);

                let t = if tau >= F::zero() {
                    F::one() / (tau + (F::one() + tau * tau).sqrt())
                } else {
                    -F::one() / (-tau + (F::one() + tau * tau).sqrt())
                };

                let cos = F::one() / (F::one() + t * t).sqrt();
                let sin = t * cos;

                // Apply rotation to columns i and j of A
                for row in 0..m {
                    let ai = a[[row, i]];
                    let aj = a[[row, j]];
                    a[[row, i]] = cos * ai - sin * aj;
                    a[[row, j]] = sin * ai + cos * aj;
                }

                // Accumulate V
                for row in 0..n {
                    let vi = v[[row, i]];
                    let vj = v[[row, j]];
                    v[[row, i]] = cos * vi - sin * vj;
                    v[[row, j]] = sin * vi + cos * vj;
                }
            }
        }

        if converged {
            break 'outer;
        }
    }

    // Extract singular values as column norms for ALL n columns, then sort.
    // For wide matrices (m < n), only k=m singular values are nonzero but we
    // need to scan all n columns to find the k largest.
    let mut all_sigma = Array1::<F>::zeros(n);
    for i in 0..n {
        all_sigma[i] = col_dot_f(&a.view(), i, i, m).sqrt();
    }

    // Sort all n indices by descending sigma, then take the top k
    let mut all_indices: Vec<usize> = (0..n).collect();
    all_indices.sort_by(|&ia, &ib| {
        all_sigma[ib]
            .partial_cmp(&all_sigma[ia])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top k indices
    let indices: Vec<usize> = all_indices.into_iter().take(k).collect();

    // Build U from the selected k columns
    let mut u_mat = Array2::<F>::zeros((m, k));
    for (new_i, &old_i) in indices.iter().enumerate() {
        let sigma_val = all_sigma[old_i];
        if sigma_val > F::epsilon() {
            for row in 0..m {
                u_mat[[row, new_i]] = a[[row, old_i]] / sigma_val;
            }
        } else if new_i < m {
            // Zero singular value: leave as zero (the gram-schmidt complement will fix full U)
        }
    }

    let sigma_sorted = Array1::from_iter(indices.iter().map(|&idx| all_sigma[idx]));

    // u_mat is already built in sorted order; just wrap it appropriately
    let mut u_sorted = if full_matrices {
        // Pad to m×m with identity basis for extra columns
        let mut uf = Array2::<F>::zeros((m, m));
        for new_i in 0..k {
            for row in 0..m {
                uf[[row, new_i]] = u_mat[[row, new_i]];
            }
        }
        // Fill remaining with standard basis (Gram-Schmidt will fix orthogonality)
        for new_i in k..m {
            if new_i < m {
                uf[[new_i, new_i]] = F::one();
            }
        }
        uf
    } else {
        u_mat
    };

    // Ensure U columns are orthonormal (Gram-Schmidt polish for full case)
    if full_matrices && m > k {
        gram_schmidt_complement(&mut u_sorted, k, m);
    }

    // Reorder V columns, then transpose to get V^T
    let vt_sorted = if full_matrices {
        let mut vt = Array2::<F>::zeros((n, n));
        for (new_i, &old_i) in indices.iter().enumerate() {
            for col in 0..n {
                vt[[new_i, col]] = v[[col, old_i]];
            }
        }
        vt
    } else {
        let mut vt = Array2::<F>::zeros((k, n));
        for (new_i, &old_i) in indices.iter().enumerate() {
            for col in 0..n {
                vt[[new_i, col]] = v[[col, old_i]];
            }
        }
        vt
    };

    Ok((u_sorted, sigma_sorted, vt_sorted))
}

/// Dot product of column i and column j of matrix a (using first m rows).
#[inline]
fn col_dot_f<F: Float>(a: &scirs2_core::ndarray::ArrayView2<F>, i: usize, j: usize, m: usize) -> F {
    let mut s = F::zero();
    for row in 0..m {
        s += a[[row, i]] * a[[row, j]];
    }
    s
}

/// Extend the first `k` orthonormal columns of `u` (m×m) to a full orthonormal
/// basis using Gram-Schmidt for columns k..m.
fn gram_schmidt_complement<F: Float>(u: &mut Array2<F>, k: usize, m: usize) {
    // Generate candidate vectors; for each try basis vectors e_0, e_1, …
    let mut filled = k;
    let mut candidate = 0usize;
    while filled < m && candidate < m {
        // Build e_candidate
        let mut v = Array1::<F>::zeros(m);
        v[candidate] = F::one();
        candidate += 1;

        // Gram-Schmidt against all already-accepted columns
        for j in 0..filled {
            let mut dot = F::zero();
            for i in 0..m {
                dot += u[[i, j]] * v[i];
            }
            for i in 0..m {
                v[i] -= dot * u[[i, j]];
            }
        }

        // Normalize
        let norm = v.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();
        if norm > F::epsilon() {
            for i in 0..m {
                u[[i, filled]] = v[i] / norm;
            }
            filled += 1;
        }
    }
}

/// Compute randomized SVD for large matrices
#[allow(dead_code)]
fn compute_randomized_svd<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
    rank: usize,
    oversampling: usize,
    n_iter: usize,
) -> SVDResult<F> {
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let l = (rank + oversampling).min(n.min(m));

    // Generate random Gaussian matrix
    let mut omega = Array2::<F>::zeros((n, l));
    for i in 0..n {
        for j in 0..l {
            // Simple pseudo-random number (not cryptographically secure)
            let val = F::from((i * l + j) % 7).expect("Operation failed")
                / F::from(7.0).expect("Failed to convert constant to float")
                - F::from(0.5).expect("Failed to convert constant to float");
            omega[[i, j]] = val;
        }
    }

    // Power iteration for better approximation
    let mut q = matrix.dot(&omega);

    for _ in 0..n_iter {
        q = orthogonalize_qr(&q)?;
        q = matrix.t().dot(&q);
        q = orthogonalize_qr(&q)?;
        q = matrix.dot(&q);
    }

    let q = orthogonalize_qr(&q)?;

    // Project matrix onto Q subspace
    let b = q.t().dot(matrix);

    // Compute SVD of smaller matrix B
    let (u_b, s, vt) = compute_svd_jacobi(&b.view(), false)?;

    // Recover full U
    let u = q.dot(&u_b);

    // Truncate to requested rank
    let u_truncated = u.slice(s![.., ..rank]).to_owned();
    let s_truncated = s.slice(s![..rank]).to_owned();
    let vt_truncated = vt.slice(s![..rank, ..]).to_owned();

    Ok((u_truncated, s_truncated, vt_truncated))
}

/// Compute generalized eigenvalue problem
#[allow(dead_code)]
/// Symmetric-definite generalized eigenproblem `A v = lambda B v`.
///
/// Returns `(values, vectors)` with the eigenvalues in descending order and the
/// eigenvectors as **B-orthonormal columns**: `Vᵀ B V = I` and `Vᵀ A V = diag(values)`.
/// That normalisation is what makes the eigenvalue derivative
/// `dlambda_k = v_kᵀ (dA - lambda_k dB) v_k` valid, and it is the convention LAPACK's
/// `sygv` uses.
///
/// The reduction is the textbook one: factor `B = L Lᵀ` (Cholesky), form the *symmetric*
/// matrix `C = L^-1 A L^-ᵀ`, diagonalise it, and map the eigenvectors back with
/// `V = L^-ᵀ Y`.
///
/// # Errors
///
/// Rejects a non-symmetric `A`, and a `B` that is not symmetric positive definite. The
/// previous implementation formed the *non-symmetric* product `B^-1 A` and ran an
/// unshifted QR iteration on it, returning the accumulated `Q` as "eigenvectors" — those
/// are Schur vectors, not eigenvectors, unless `B^-1 A` happens to be symmetric, so the
/// second output was simply the wrong matrix for a general input. Failing loudly is
/// better than returning a plausible-looking basis that does not diagonalise anything.
fn compute_generalized_eigen<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    let n = a.nrows();
    if a.ncols() != n || b.nrows() != n || b.ncols() != n {
        return Err(OpError::IncompatibleShape(
            "generalized eigenproblem: A and B must be square and the same size".into(),
        ));
    }
    if !matrix_calculus::is_symmetric(a) {
        return Err(OpError::Other(
            "generalized eigenproblem: A must be symmetric (only the symmetric-definite \
             problem is implemented; the general case needs the QZ algorithm)"
                .into(),
        ));
    }
    if !matrix_calculus::is_symmetric(b) {
        return Err(OpError::Other(
            "generalized eigenproblem: B must be symmetric positive definite".into(),
        ));
    }

    // Cholesky factor of B (lower triangular), with a positive-definiteness check.
    let mut l = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = (b[[i, j]] + b[[j, i]]) / F::from(2.0).unwrap_or_else(F::one);
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= F::zero() {
                    return Err(OpError::Other(
                        "generalized eigenproblem: B is not positive definite".into(),
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    let l_inv = matrix_calculus::inverse(&l.view())?;
    // C = L^-1 A L^-T, symmetric by construction.
    let c = l_inv.dot(a).dot(&l_inv.t());
    let (values, y) = matrix_calculus::symmetric_eigen(&c.view())?;
    // V = L^-T Y satisfies Vᵀ B V = Yᵀ L^-1 (L Lᵀ) L^-T Y = Yᵀ Y = I.
    let vectors = l_inv.t().dot(&y);
    Ok((values, vectors))
}

/// QR decomposition with column pivoting
#[allow(dead_code)]
fn compute_qr_pivot<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> QRPivotResult<F> {
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let k = m.min(n);

    let mut a = matrix.to_owned();
    let mut q = Array2::<F>::eye(m);
    let mut perm: Vec<usize> = (0..n).collect();

    // Column norms for pivoting
    let mut col_norms = Array1::<F>::zeros(n);
    for j in 0..n {
        let col = a.slice(s![.., j]);
        col_norms[j] = col.dot(&col).sqrt();
    }

    for i in 0..k {
        // Find pivot column
        let (pivot_idx, _) = col_norms
            .slice(s![i..])
            .indexed_iter()
            .max_by(|(_, &a), (_, &b)| a.abs().partial_cmp(&b.abs()).expect("Operation failed"))
            .expect("Failed to perform decomposition");
        let pivot_col = i + pivot_idx;

        // Swap columns
        if pivot_col != i {
            perm.swap(i, pivot_col);
            for row in 0..m {
                a.swap((row, i), (row, pivot_col));
            }
            col_norms.swap(i, pivot_col);
        }

        // Compute Householder reflector
        if i < m {
            let col = a.slice(s![i.., i]).to_owned();
            let (v, beta) = householder_vector(&col.view())?;

            // Apply Householder transformation
            if beta.abs() > F::epsilon() {
                // Update A
                for j in i..n {
                    let col = a.slice(s![i.., j]).to_owned();
                    let dot_product = v.dot(&col);
                    let update = v.mapv(|x| x * beta * dot_product);
                    for k in 0..m - i {
                        a[[i + k, j]] -= update[k];
                    }
                }

                // Update Q
                for j in 0..m {
                    let col = q.slice(s![j, i..]).to_owned();
                    let dot_product = col.dot(&v);
                    for k in 0..m - i {
                        q[[j, i + k]] -= beta * dot_product * v[k];
                    }
                }

                // Update column norms
                for j in (i + 1)..n {
                    let col = a.slice(s![i + 1.., j]);
                    col_norms[j] = col.dot(&col).sqrt();
                }
            }
        }
    }

    // Extract R
    let r = a.slice(s![..k, ..]).to_owned();

    // Convert permutation to array
    let p = Array1::from_vec(
        perm.iter()
            .map(|&i| F::from(i).expect("Failed to convert to float"))
            .collect(),
    );

    Ok((q, r, p))
}

// Utility functions

#[allow(dead_code)]
fn householder_vector<F: Float>(
    x: &scirs2_core::ndarray::ArrayView1<F>,
) -> Result<(Array1<F>, F), OpError> {
    let n = x.len();
    if n == 0 {
        return Err(OpError::IncompatibleShape("Empty vector".into()));
    }

    let mut v = x.to_owned();
    let norm_x = x.dot(x).sqrt();

    if norm_x < F::epsilon() {
        v[0] = F::one();
        return Ok((v, F::zero()));
    }

    let sign = if x[0] >= F::zero() {
        F::one()
    } else {
        -F::one()
    };
    v[0] += sign * norm_x;

    let norm_v_sq = v.dot(&v);
    let beta = F::from(2.0).expect("Failed to convert constant to float") / norm_v_sq;

    Ok((v, beta))
}

#[allow(dead_code)]
fn householder_matrix<F: Float>(v: &Array1<F>, size: usize) -> Array2<F> {
    let beta = F::from(2.0).expect("Failed to convert constant to float") / v.dot(v);
    let mut h = Array2::<F>::eye(size);

    for i in 0..size {
        for j in 0..size {
            h[[i, j]] -= beta * v[i] * v[j];
        }
    }

    h
}

#[allow(dead_code)]
fn compute_givens_rotation<F: Float>(a: F, b: F, c: F) -> (F, F) {
    if b.abs() < F::epsilon() {
        return (F::one(), F::zero());
    }

    let tau = (c - a) / (F::from(2.0).expect("Failed to convert constant to float") * b);
    let t = if tau >= F::zero() {
        F::one() / (tau + (F::one() + tau * tau).sqrt())
    } else {
        -F::one() / (-tau + (F::one() + tau * tau).sqrt())
    };

    let cos = F::one() / (F::one() + t * t).sqrt();
    let sin = t * cos;

    (cos, sin)
}

#[allow(dead_code)]
fn apply_givens_left<F: Float>(matrix: &mut Array2<F>, i: usize, j: usize, cos: F, sin: F) {
    let n = matrix.shape()[1];
    for k in 0..n {
        let ai = matrix[[i, k]];
        let aj = matrix[[j, k]];
        matrix[[i, k]] = cos * ai - sin * aj;
        matrix[[j, k]] = sin * ai + cos * aj;
    }
}

#[allow(dead_code)]
fn apply_givens_right<F: Float>(matrix: &mut Array2<F>, i: usize, j: usize, cos: F, sin: F) {
    let m = matrix.shape()[0];
    for k in 0..m {
        let ai = matrix[[k, i]];
        let aj = matrix[[k, j]];
        matrix[[k, i]] = cos * ai - sin * aj;
        matrix[[k, j]] = sin * ai + cos * aj;
    }
}

#[allow(dead_code)]
fn orthogonalize_qr<F: Float + scirs2_core::ndarray::ScalarOperand>(
    a: &Array2<F>,
) -> Result<Array2<F>, OpError> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let mut q = a.to_owned();

    // Modified Gram-Schmidt
    for j in 0..n {
        let mut col = q.slice_mut(s![.., j]);
        let norm = col.dot(&col).sqrt();

        if norm < F::epsilon() {
            return Err(OpError::Other("Matrix is rank deficient".into()));
        }

        col.mapv_inplace(|x| x / norm);

        for k in (j + 1)..n {
            let dot_product = q.slice(s![.., j]).dot(&q.slice(s![.., k]));
            let q_col_j = q.slice(s![.., j]).to_owned();
            for i in 0..m {
                q[[i, k]] -= dot_product * q_col_j[i];
            }
        }
    }

    Ok(q)
}

#[allow(dead_code)]
fn compute_matrix_inverse<F: Float>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<Array2<F>, OpError> {
    let n = matrix.shape()[0];
    let mut a = matrix.to_owned();
    let mut inv = Array2::<F>::eye(n);

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        if a[[max_row, i]].abs() < F::epsilon() {
            return Err(OpError::IncompatibleShape("Matrix is singular".into()));
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                a.swap((i, j), (max_row, j));
                inv.swap((i, j), (max_row, j));
            }
        }

        // Scale pivot row
        let pivot = a[[i, i]];
        for j in 0..n {
            a[[i, j]] /= pivot;
            inv[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = a[[k, i]];
                for j in 0..n {
                    let a_ij = a[[i, j]];
                    let inv_ij = inv[[i, j]];
                    a[[k, j]] -= factor * a_ij;
                    inv[[k, j]] -= factor * inv_ij;
                }
            }
        }
    }

    Ok(inv)
}

#[allow(dead_code)]
fn compute_eigen_iterative<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<(Array1<F>, Array2<F>), OpError> {
    let n = matrix.shape()[0];
    let max_iter = 100;
    let tol = F::epsilon() * F::from(10.0).expect("Failed to convert constant to float");

    // QR algorithm with shifts
    let mut a = matrix.to_owned();
    let mut q_total = Array2::<F>::eye(n);

    for _ in 0..max_iter {
        // Wilkinson shift
        let a_nn = a[[n - 1, n - 1]];
        let a_nm1 = if n > 1 { a[[n - 2, n - 1]] } else { F::zero() };
        let a_nm1nm1 = if n > 1 { a[[n - 2, n - 2]] } else { F::zero() };

        let delta = (a_nm1nm1 - a_nn) / F::from(2.0).expect("Failed to convert constant to float");
        let sign = if delta >= F::zero() {
            F::one()
        } else {
            -F::one()
        };
        let mu =
            a_nn - sign * a_nm1 * a_nm1 / (delta.abs() + (delta * delta + a_nm1 * a_nm1).sqrt());

        // Shifted QR step
        for i in 0..n {
            a[[i, i]] -= mu;
        }

        // QR decomposition
        let (q, r) = compute_qr_simple(&a.view())?;
        a = r.dot(&q);

        for i in 0..n {
            a[[i, i]] += mu;
        }

        q_total = q_total.dot(&q);

        // Check convergence
        let mut converged = true;
        for i in 0..n - 1 {
            if a[[i + 1, i]].abs() > tol {
                converged = false;
                break;
            }
        }

        if converged {
            break;
        }
    }

    // Extract eigenvalues
    let mut eigenvalues = Array1::<F>::zeros(n);
    for i in 0..n {
        eigenvalues[i] = a[[i, i]];
    }

    Ok((eigenvalues, q_total))
}

#[allow(dead_code)]
fn compute_qr_simple<F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &scirs2_core::ndarray::ArrayView2<F>,
) -> Result<(Array2<F>, Array2<F>), OpError> {
    let (m, n) = (matrix.shape()[0], matrix.shape()[1]);
    let k = m.min(n);

    let mut q = Array2::<F>::eye(m);
    let mut r = matrix.to_owned();

    for j in 0..k {
        let col = r.slice(s![j.., j]).to_owned();
        let (v, beta) = householder_vector(&col.view())?;

        if beta.abs() > F::epsilon() {
            // Apply to R
            for col_idx in j..n {
                let col = r.slice(s![j.., col_idx]).to_owned();
                let dot_product = v.dot(&col);
                for row_idx in 0..(m - j) {
                    r[[j + row_idx, col_idx]] -= beta * dot_product * v[row_idx];
                }
            }

            // Apply to Q
            for row_idx in 0..m {
                let row = q.slice(s![row_idx, j..]).to_owned();
                let dot_product = row.dot(&v);
                for col_idx in 0..(m - j) {
                    q[[row_idx, j + col_idx]] -= beta * dot_product * v[col_idx];
                }
            }
        }
    }

    Ok((q, r))
}

// Public API functions

/// Compute SVD using improved Jacobi algorithm
#[allow(dead_code)]
pub fn svd_jacobi<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
    full_matrices: bool,
) -> (Tensor<'g, F>, Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    // Each component reads the *original matrix*, not a shared parent node: a graph node
    // exposes only its first output, so a multi-output parent cannot hand U, S and Vt to
    // three separate consumers.  Wiring the extractors to the parent is what made every
    // one of these tensors fail to evaluate.
    let component = |index: usize| {
        Tensor::builder(g)
            .append_input(matrix, false)
            .build(SVDJacobiExtractOp {
                component: index,
                full_matrices,
            })
    };

    (component(0), component(1), component(2))
}

/// Compute randomized SVD for large matrices
#[allow(dead_code)]
pub fn randomized_svd<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    matrix: &Tensor<'g, F>,
    rank: usize,
    oversampling: usize,
    n_iter: usize,
) -> (Tensor<'g, F>, Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    let component = |index: usize| {
        Tensor::builder(g)
            .append_input(matrix, false)
            .build(RandomizedSVDExtractOp {
                component: index,
                rank,
                oversampling,
                n_iter,
            })
    };

    (component(0), component(1), component(2))
}

/// Solve generalized eigenvalue problem Ax = λBx
#[allow(dead_code)]
pub fn generalized_eigen<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
) -> (Tensor<'g, F>, Tensor<'g, F>) {
    let g = a.graph();

    let component = |index: usize| {
        Tensor::builder(g)
            .append_input(a, false)
            .append_input(b, false)
            .build(GeneralizedEigenExtractOp { component: index })
    };

    (component(0), component(1))
}

/// QR decomposition with column pivoting
#[allow(dead_code)]
pub fn qr_pivot<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> (Tensor<'g, F>, Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    let component = |index: usize| {
        Tensor::builder(g)
            .append_input(matrix, false)
            .build(QRPivotExtractOp { component: index })
    };

    (component(0), component(1), component(2))
}
