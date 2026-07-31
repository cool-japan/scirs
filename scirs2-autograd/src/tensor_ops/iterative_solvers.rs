use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::matrix_calculus;
use crate::Float;
use scirs2_core::ndarray::{Array1, Array2, Ix1, Ix2};
use scirs2_core::numeric::FromPrimitive;

/// Which iterative algorithm a solver node (and its backward node) runs.
#[derive(Clone, Copy)]
pub enum SolverKind {
    /// Conjugate gradient (symmetric positive definite `A`).
    ConjugateGradient,
    /// Restarted GMRES (general `A`), carrying the restart length.
    Gmres { restart: usize },
    /// Biconjugate gradient stabilised (general `A`).
    BiCgStab,
    /// Preconditioned conjugate gradient, carrying the preconditioner choice.
    Pcg { preconditioner: PreconditionerType },
}

impl SolverKind {
    /// Runs the algorithm on `A x = b`.
    fn solve<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
        self,
        a: &scirs2_core::ndarray::ArrayView2<F>,
        b: &scirs2_core::ndarray::ArrayView1<F>,
        max_iter: usize,
        tolerance: Option<f64>,
    ) -> Result<Array1<F>, OpError> {
        match self {
            SolverKind::ConjugateGradient => conjugate_gradient(a, b, max_iter, tolerance),
            SolverKind::Gmres { restart } => gmres(a, b, max_iter, restart, tolerance),
            SolverKind::BiCgStab => bicgstab(a, b, max_iter, tolerance),
            SolverKind::Pcg { preconditioner } => {
                let m = build_preconditioner(a, preconditioner)?;
                pcg(a, b, &m, max_iter, tolerance)
            }
        }
    }
}

/// Backward node of a linear solve: `y = solve(Aᵀ, gy)`.
///
/// `x = A^-1 b` is defined implicitly by `A x = b`. Differentiating that relation gives
/// `A dx = db - dA x`, so for an output cotangent `gy`
///
/// ```text
///   b̄ = A^-ᵀ gy          (solve the transposed system)
///   Ā = -b̄ xᵀ            (outer product)
/// ```
///
/// This node computes `b̄` by running the *same* iterative algorithm on `Aᵀ`, which is
/// what makes the rule exact for the solver actually used rather than for some idealised
/// direct solve.
pub struct SolveTransposeOp {
    kind: SolverKind,
    max_iter: usize,
    tolerance: Option<f64>,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for SolveTransposeOp {
    fn name(&self) -> &'static str {
        "IterativeSolveTranspose"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_in = ctx.input(0);
        let gy_in = ctx.input(1);

        let a_2d = a_in
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("solver backward: A is not 2-D".into()))?;
        if a_2d.nrows() != a_2d.ncols() {
            return Err(OpError::IncompatibleShape(
                "solver backward: A is not square".into(),
            ));
        }
        let gy_1d = gy_in.view().into_dimensionality::<Ix1>().map_err(|_| {
            OpError::IncompatibleShape("solver backward: the cotangent is not 1-D".into())
        })?;
        if gy_1d.len() != a_2d.nrows() {
            return Err(OpError::IncompatibleShape(
                "solver backward: cotangent length does not match the system size".into(),
            ));
        }

        let at = a_2d.t().to_owned();
        let y = self
            .kind
            .solve(&at.view(), &gy_1d, self.max_iter, self.tolerance)?;
        ctx.append_output(y.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        matrix_calculus::append_unsupported_grad(
            ctx,
            "iterative linear solve: second-order differentiation is not implemented (it \
             requires differentiating through the transposed solve itself)."
                .into(),
        );
    }
}

/// `-u vᵀ`, the outer product appearing in the linear-solve VJP.
pub struct NegOuterProductOp;

impl<F: Float> Op<F> for NegOuterProductOp {
    fn name(&self) -> &'static str {
        "NegOuterProduct"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let u_in = ctx.input(0);
        let v_in = ctx.input(1);
        let u = u_in.view().into_dimensionality::<Ix1>().map_err(|_| {
            OpError::IncompatibleShape("outer product: the first operand is not 1-D".into())
        })?;
        let v = v_in.view().into_dimensionality::<Ix1>().map_err(|_| {
            OpError::IncompatibleShape("outer product: the second operand is not 1-D".into())
        })?;
        let mut out = Array2::<F>::zeros((u.len(), v.len()));
        for i in 0..u.len() {
            for j in 0..v.len() {
                out[[i, j]] = -(u[i] * v[j]);
            }
        }
        ctx.append_output(out.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // With `Y = -u vᵀ`: `ū = -G v` and `v̄ = -Gᵀ u`.  Both matrix-vector products are
        // expressed as a broadcast multiply followed by a row sum so that the rule is
        // built out of ops whose own VJPs are already covered.
        let u = *ctx.input(0);
        let v = *ctx.input(1);
        let gy = *ctx.output_grad();
        let neg_gy = crate::tensor_ops::neg(gy);
        let neg_gy_t = crate::tensor_ops::transpose(neg_gy, &[1, 0]);
        let grad_u = crate::tensor_ops::reduce_sum(crate::tensor_ops::mul(neg_gy, v), &[1], false);
        let grad_v =
            crate::tensor_ops::reduce_sum(crate::tensor_ops::mul(neg_gy_t, u), &[1], false);
        ctx.append_input_grad(0, Some(grad_u));
        ctx.append_input_grad(1, Some(grad_v));
    }
}

/// Emits the implicit-function VJP of `x = solve(A, b)` for every solver in this module.
fn append_linear_solve_grad<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    ctx: &mut GradientContext<F>,
    kind: SolverKind,
    max_iter: usize,
    tolerance: Option<f64>,
) {
    let a = *ctx.input(0);
    let x = *ctx.output();
    let gy = *ctx.output_grad();
    let g = ctx.graph();

    let grad_b = Tensor::builder(g)
        .append_input(a, false)
        .append_input(gy, false)
        .build(SolveTransposeOp {
            kind,
            max_iter,
            tolerance,
        });
    let grad_a = Tensor::builder(g)
        .append_input(grad_b, false)
        .append_input(x, false)
        .build(NegOuterProductOp);

    ctx.append_input_grad(0, Some(grad_a));
    ctx.append_input_grad(1, Some(grad_b));
}

/// Conjugate Gradient solver for symmetric positive definite systems
pub struct ConjugateGradientOp {
    max_iter: usize,
    tolerance: Option<f64>,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for ConjugateGradientOp {
    fn name(&self) -> &'static str {
        "ConjugateGradient"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let ashape = a.shape();
        let bshape = b.shape();

        if ashape.len() != 2 || ashape[0] != ashape[1] {
            return Err(OpError::IncompatibleShape(
                "CG requires square matrix".into(),
            ));
        }

        if bshape.len() != 1 || bshape[0] != ashape[0] {
            return Err(OpError::IncompatibleShape(
                "Incompatible dimensions for Ax=b".into(),
            ));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;
        let b_1d = b
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 1D".into()))?;

        // Solve using conjugate gradient
        let x = conjugate_gradient(&a_2d, &b_1d, self.max_iter, self.tolerance)?;

        ctx.append_output(x.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        append_linear_solve_grad(
            ctx,
            SolverKind::ConjugateGradient,
            self.max_iter,
            self.tolerance,
        );
    }
}

/// GMRES (Generalized Minimal RESidual) solver for general linear systems
pub struct GMRESOp {
    max_iter: usize,
    restart: usize,
    tolerance: Option<f64>,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for GMRESOp {
    fn name(&self) -> &'static str {
        "GMRES"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let ashape = a.shape();
        let bshape = b.shape();

        if ashape.len() != 2 || ashape[0] != ashape[1] {
            return Err(OpError::IncompatibleShape(
                "GMRES requires square matrix".into(),
            ));
        }

        if bshape.len() != 1 || bshape[0] != ashape[0] {
            return Err(OpError::IncompatibleShape(
                "Incompatible dimensions for Ax=b".into(),
            ));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;
        let b_1d = b
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 1D".into()))?;

        // Solve using GMRES
        let x = gmres(&a_2d, &b_1d, self.max_iter, self.restart, self.tolerance)?;

        ctx.append_output(x.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        append_linear_solve_grad(
            ctx,
            SolverKind::Gmres {
                restart: self.restart,
            },
            self.max_iter,
            self.tolerance,
        );
    }
}

/// BiCGSTAB (Biconjugate Gradient Stabilized) solver
pub struct BiCGSTABOp {
    max_iter: usize,
    tolerance: Option<f64>,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for BiCGSTABOp {
    fn name(&self) -> &'static str {
        "BiCGSTAB"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let ashape = a.shape();
        let bshape = b.shape();

        if ashape.len() != 2 || ashape[0] != ashape[1] {
            return Err(OpError::IncompatibleShape(
                "BiCGSTAB requires square matrix".into(),
            ));
        }

        if bshape.len() != 1 || bshape[0] != ashape[0] {
            return Err(OpError::IncompatibleShape(
                "Incompatible dimensions for Ax=b".into(),
            ));
        }

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;
        let b_1d = b
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 1D".into()))?;

        // Solve using BiCGSTAB
        let x = bicgstab(&a_2d, &b_1d, self.max_iter, self.tolerance)?;

        ctx.append_output(x.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        append_linear_solve_grad(ctx, SolverKind::BiCgStab, self.max_iter, self.tolerance);
    }
}

/// Preconditioned Conjugate Gradient solver
pub struct PCGOp {
    max_iter: usize,
    tolerance: Option<f64>,
    preconditioner: PreconditionerType,
}

#[derive(Clone, Copy)]
pub enum PreconditionerType {
    None,
    Jacobi,
    IncompleteCholesky,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive> Op<F> for PCGOp {
    fn name(&self) -> &'static str {
        "PCG"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        let a_2d = a
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert A to 2D".into()))?;
        let b_1d = b
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert b to 1D".into()))?;

        // Build preconditioner
        let preconditioner = build_preconditioner(&a_2d, self.preconditioner)?;

        // Solve using PCG
        let x = pcg(&a_2d, &b_1d, &preconditioner, self.max_iter, self.tolerance)?;

        ctx.append_output(x.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        append_linear_solve_grad(
            ctx,
            SolverKind::Pcg {
                preconditioner: self.preconditioner,
            },
            self.max_iter,
            self.tolerance,
        );
    }
}

// Helper functions

/// Conjugate Gradient implementation
#[allow(dead_code)]
fn conjugate_gradient<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView1<F>,
    max_iter: usize,
    tolerance: Option<f64>,
) -> Result<Array1<F>, OpError> {
    let n = b.len();
    let tol = tolerance
        .map(|t| F::from(t).expect("Failed to convert to float"))
        .unwrap_or_else(|| {
            F::epsilon() * F::from(10.0).expect("Failed to convert constant to float")
        });

    // Initial guess x = 0
    let mut x = Array1::<F>::zeros(n);

    // r = b - Ax = b (since x = 0)
    let mut r = b.to_owned();
    let mut p = r.clone();
    let mut rsold = r.dot(&r);

    for _ in 0..max_iter {
        let ap = a.dot(&p);
        let alpha = rsold / p.dot(&ap);

        x = &x + &p.mapv(|v| alpha * v);
        r = &r - &ap.mapv(|v| alpha * v);

        let rsnew = r.dot(&r);

        if rsnew.sqrt() < tol {
            break;
        }

        let beta = rsnew / rsold;
        p = &r + &p.mapv(|v| beta * v);

        rsold = rsnew;
    }

    Ok(x)
}

/// GMRES implementation
#[allow(dead_code)]
fn gmres<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView1<F>,
    max_iter: usize,
    restart: usize,
    tolerance: Option<f64>,
) -> Result<Array1<F>, OpError> {
    let n = b.len();
    let tol = tolerance
        .map(|t| F::from(t).expect("Failed to convert to float"))
        .unwrap_or_else(|| {
            F::epsilon() * F::from(10.0).expect("Failed to convert constant to float")
        });

    let mut x = Array1::<F>::zeros(n);
    let m = restart.min(n);

    for _ in 0..max_iter {
        let r = b - &a.dot(&x);
        let rnorm = r.dot(&r).sqrt();

        if rnorm < tol {
            break;
        }

        // Arnoldi process
        let mut v = vec![Array1::<F>::zeros(n); m + 1];
        let mut h = Array2::<F>::zeros((m + 1, m));

        v[0] = &r / rnorm;

        let mut j = 0;
        while j < m {
            let mut w = a.dot(&v[j]);

            // Modified Gram-Schmidt
            for i in 0..=j {
                h[[i, j]] = w.dot(&v[i]);
                w = &w - &v[i].mapv(|val| h[[i, j]] * val);
            }

            h[[j + 1, j]] = w.dot(&w).sqrt();

            if h[[j + 1, j]].abs() < F::epsilon() {
                break;
            }

            v[j + 1] = w / h[[j + 1, j]];
            j += 1;
        }

        // Solve least squares problem
        let beta = rnorm;
        let mut e1 = Array1::<F>::zeros(j + 1);
        e1[0] = beta;

        let y = solve_least_squares(
            &h.slice(scirs2_core::ndarray::s![..j + 1, ..j]).to_owned(),
            &e1,
        )?;

        // Update solution
        for i in 0..j {
            x = &x + &v[i].mapv(|val| y[i] * val);
        }
    }

    Ok(x)
}

/// BiCGSTAB implementation
#[allow(dead_code)]
fn bicgstab<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView1<F>,
    max_iter: usize,
    tolerance: Option<f64>,
) -> Result<Array1<F>, OpError> {
    let n = b.len();
    let tol = tolerance
        .map(|t| F::from(t).expect("Failed to convert to float"))
        .unwrap_or_else(|| {
            F::epsilon() * F::from(10.0).expect("Failed to convert constant to float")
        });

    let mut x = Array1::<F>::zeros(n);
    let mut r = b - &a.dot(&x);
    let r0 = r.clone();

    let mut rho = F::one();
    let mut alpha = F::one();
    let mut omega = F::one();

    let mut v = Array1::<F>::zeros(n);
    let mut p = Array1::<F>::zeros(n);

    for _ in 0..max_iter {
        let rho_new = r0.dot(&r);

        if rho_new.abs() < F::epsilon() {
            break;
        }

        let beta = (rho_new / rho) * (alpha / omega);
        p = &r + &(&p - &v.mapv(|val| omega * val)).mapv(|val| beta * val);

        v = a.dot(&p);
        alpha = rho_new / r0.dot(&v);

        let s = &r - &v.mapv(|val| alpha * val);

        if s.dot(&s).sqrt() < tol {
            x = &x + &p.mapv(|v| alpha * v);
            break;
        }

        let t = a.dot(&s);
        omega = t.dot(&s) / t.dot(&t);

        x = &x + &p.mapv(|val| alpha * val) + &s.mapv(|val| omega * val);
        r = &s - &t.mapv(|val| omega * val);

        if r.dot(&r).sqrt() < tol {
            break;
        }

        rho = rho_new;
    }

    Ok(x)
}

/// Preconditioned Conjugate Gradient
#[allow(dead_code)]
fn pcg<F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView1<F>,
    m_inv: &Array2<F>,
    max_iter: usize,
    tolerance: Option<f64>,
) -> Result<Array1<F>, OpError> {
    let n = b.len();
    let tol = tolerance
        .map(|t| F::from(t).expect("Failed to convert to float"))
        .unwrap_or_else(|| {
            F::epsilon() * F::from(10.0).expect("Failed to convert constant to float")
        });

    let mut x = Array1::<F>::zeros(n);
    let mut r = b.to_owned();
    let mut z = m_inv.dot(&r);
    let mut p = z.clone();
    let mut rzold = r.dot(&z);

    for _ in 0..max_iter {
        let ap = a.dot(&p);
        let alpha = rzold / p.dot(&ap);

        x = &x + &p.mapv(|v| alpha * v);
        r = &r - &ap.mapv(|v| alpha * v);

        if r.dot(&r).sqrt() < tol {
            break;
        }

        z = m_inv.dot(&r);
        let rznew = r.dot(&z);
        let beta = rznew / rzold;
        p = &z + &p.mapv(|val| beta * val);

        rzold = rznew;
    }

    Ok(x)
}

/// Build preconditioner
#[allow(dead_code)]
fn build_preconditioner<F: Float + scirs2_core::ndarray::ScalarOperand>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    preconditioner_type: PreconditionerType,
) -> Result<Array2<F>, OpError> {
    let n = a.shape()[0];

    match preconditioner_type {
        PreconditionerType::None => Ok(Array2::<F>::eye(n)),
        PreconditionerType::Jacobi => {
            // Diagonal preconditioner
            let mut m_inv = Array2::<F>::zeros((n, n));
            for i in 0..n {
                if a[[i, i]].abs() > F::epsilon() {
                    m_inv[[i, i]] = F::one() / a[[i, i]];
                } else {
                    m_inv[[i, i]] = F::one();
                }
            }
            Ok(m_inv)
        }
        PreconditionerType::IncompleteCholesky => {
            // Simplified incomplete Cholesky
            let mut l = Array2::<F>::zeros((n, n));

            for i in 0..n {
                for j in 0..=i {
                    if a[[i, j]].abs() > F::epsilon() {
                        let mut sum = a[[i, j]];
                        for k in 0..j {
                            sum -= l[[i, k]] * l[[j, k]];
                        }

                        if i == j {
                            if sum > F::epsilon() {
                                l[[i, j]] = sum.sqrt();
                            } else {
                                l[[i, j]] = F::one();
                            }
                        } else {
                            l[[i, j]] = sum / l[[j, j]];
                        }
                    }
                }
            }

            // Return L^{-1}L^{-T} approximation
            // For simplicity, use diagonal approximation
            let mut m_inv = Array2::<F>::zeros((n, n));
            for i in 0..n {
                if l[[i, i]].abs() > F::epsilon() {
                    m_inv[[i, i]] = F::one() / (l[[i, i]] * l[[i, i]]);
                } else {
                    m_inv[[i, i]] = F::one();
                }
            }
            Ok(m_inv)
        }
    }
}

/// Solve least squares problem
#[allow(dead_code)]
fn solve_least_squares<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, OpError> {
    // Use normal equations: A^T A x = A^T b
    let at = a.t();
    let ata = at.dot(a);
    let atb = at.dot(b);

    // Solve using Cholesky decomposition (since A^T A is positive definite)
    solve_cholesky(&ata.view(), &atb.view())
}

/// Solve using Cholesky decomposition
#[allow(dead_code)]
fn solve_cholesky<F: Float>(
    a: &scirs2_core::ndarray::ArrayView2<F>,
    b: &scirs2_core::ndarray::ArrayView1<F>,
) -> Result<Array1<F>, OpError> {
    let n = a.shape()[0];
    let mut l = Array2::<F>::zeros((n, n));

    // Cholesky decomposition
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }

            if i == j {
                if sum < F::epsilon() {
                    return Err(OpError::Other("Matrix not positive definite".into()));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = Array1::<F>::zeros(n);
    for i in 0..n {
        y[i] = b[i];
        for j in 0..i {
            let y_j = y[j];
            y[i] -= l[[i, j]] * y_j;
        }
        y[i] /= l[[i, i]];
    }

    // Back substitution: L^T x = y
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in (i + 1)..n {
            let x_j = x[j];
            x[i] -= l[[j, i]] * x_j;
        }
        x[i] /= l[[i, i]];
    }

    Ok(x)
}

/// Outer product of two vectors
#[allow(dead_code)]
fn outer_product<F: Float>(
    u: &scirs2_core::ndarray::ArrayView1<F>,
    v: &scirs2_core::ndarray::ArrayView1<F>,
) -> Array2<F> {
    let m = u.len();
    let n = v.len();
    let mut result = Array2::<F>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = u[i] * v[j];
        }
    }

    result
}

// Public API functions

/// Solve Ax = b using Conjugate Gradient (for symmetric positive definite A)
#[allow(dead_code)]
pub fn conjugate_gradient_solve<
    'g,
    F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive,
>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
    max_iter: usize,
    tolerance: Option<f64>,
) -> Tensor<'g, F> {
    let g = a.graph();

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(ConjugateGradientOp {
            max_iter,
            tolerance,
        })
}

/// Solve Ax = b using GMRES (for general matrices)
#[allow(dead_code)]
pub fn gmres_solve<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
    max_iter: usize,
    restart: usize,
    tolerance: Option<f64>,
) -> Tensor<'g, F> {
    let g = a.graph();

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(GMRESOp {
            max_iter,
            restart,
            tolerance,
        })
}

/// Solve Ax = b using BiCGSTAB (for general matrices)
#[allow(dead_code)]
pub fn bicgstab_solve<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
    max_iter: usize,
    tolerance: Option<f64>,
) -> Tensor<'g, F> {
    let g = a.graph();

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(BiCGSTABOp {
            max_iter,
            tolerance,
        })
}

/// Solve Ax = b using Preconditioned Conjugate Gradient
#[allow(dead_code)]
pub fn pcg_solve<'g, F: Float + scirs2_core::ndarray::ScalarOperand + FromPrimitive>(
    a: &Tensor<'g, F>,
    b: &Tensor<'g, F>,
    max_iter: usize,
    tolerance: Option<f64>,
    preconditioner: PreconditionerType,
) -> Tensor<'g, F> {
    let g = a.graph();

    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(PCGOp {
            max_iter,
            tolerance,
            preconditioner,
        })
}
