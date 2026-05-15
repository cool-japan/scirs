//! `mle::derive` — symbolic MLE estimator factory.
//!
//! # Overview
//!
//! [`fn@derive`] accepts a parametric pdf `f(x; θ)` expressed as a [`LoweredOp`]
//! and a list of parameter Var-ids.  It:
//!
//! 1. Builds the symbolic log-likelihood `ℓ(θ) = Σᵢ ln(pdf(xᵢ; θ))` using a
//!    balanced add-tree for depth O(log n).
//! 2. Differentiates w.r.t. each parameter to get the score equations
//!    `∂ℓ/∂θⱼ = 0`.
//! 3. Attempts a closed-form solve via [`fn@scirs2_symbolic::cas::solve_system`].
//! 4. Returns an [`Estimator`] that can either evaluate the closed-form
//!    solutions directly or fall back to Newton's method.
//!
//! # Variable convention
//!
//! - `Var(data_var)` .. `Var(data_var + n_samples - 1)` — data sample slots.
//! - `Var(params[j])` — the j-th parameter.
//!
//! Var-id ranges must **not** overlap (checked at runtime).
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "symbolic")]
//! # {
//! use scirs2_symbolic::eml::LoweredOp;
//! use scirs2_stats::mle::{derive, Estimator};
//!
//! // Exponential pdf: λ·exp(-λ·x)
//! // Var(0)=λ (param), Var(1)=x (data_var=1, n_samples=5)
//! let lambda = LoweredOp::Var(0);
//! let x = LoweredOp::Var(1);
//! let pdf = LoweredOp::Mul(
//!     Box::new(lambda.clone()),
//!     Box::new(LoweredOp::Exp(Box::new(LoweredOp::Neg(Box::new(
//!         LoweredOp::Mul(Box::new(lambda), Box::new(x)),
//!     ))))),
//! );
//! let data = vec![0.5, 0.25, 0.4, 0.6, 0.3];
//! let est = derive(&pdf, &[0], 1, data.len()).expect("derive");
//! # }
//! ```

use scirs2_core::ndarray::ArrayView1;
use scirs2_symbolic::{
    cas::{grad_canonical, solve_system, SystemSolveError},
    eml::{eval_real, EvalCtx, LoweredOp},
};

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// Error produced by [`fn@derive`].
#[derive(Debug, Clone)]
pub enum DeriveError {
    /// The `params` slice was empty.
    EmptyParams,
    /// `n_samples` was zero.
    ZeroSamples,
    /// The Var-id ranges for data and parameters overlap.
    VarIdCollision {
        /// The Var-id that is claimed by both a data slot and a parameter.
        id: usize,
    },
    /// An internal error that should not normally occur.
    InternalError(String),
}

impl std::fmt::Display for DeriveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeriveError::EmptyParams => write!(f, "params slice must not be empty"),
            DeriveError::ZeroSamples => write!(f, "n_samples must be ≥ 1"),
            DeriveError::VarIdCollision { id } => write!(
                f,
                "Var({id}) is used by both a data slot and a parameter — ids must not overlap"
            ),
            DeriveError::InternalError(msg) => write!(f, "internal error in mle::derive: {msg}"),
        }
    }
}

impl std::error::Error for DeriveError {}

/// Error produced by [`Estimator::fit`].
#[derive(Debug, Clone)]
pub enum FitError {
    /// The data length does not match `n_samples` the estimator was built for.
    DataLengthMismatch {
        /// Expected number of samples.
        expected: usize,
        /// Actual length supplied.
        got: usize,
    },
    /// The numerical Newton fallback failed to converge or encountered a NaN.
    NumericalFailed(String),
    /// No estimator path is available (should not occur in practice).
    NoEstimator,
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::DataLengthMismatch { expected, got } => write!(
                f,
                "data length mismatch: estimator built for {expected} samples, got {got}"
            ),
            FitError::NumericalFailed(msg) => write!(f, "Newton MLE failed: {msg}"),
            FitError::NoEstimator => write!(f, "no estimator path available"),
        }
    }
}

impl std::error::Error for FitError {}

// ─────────────────────────────────────────────────────────────────────────────
// Public struct
// ─────────────────────────────────────────────────────────────────────────────

/// A symbolic MLE estimator produced by [`fn@derive`].
///
/// Contains the score equations and optionally a closed-form solution for each
/// parameter. Use [`Estimator::fit`] to evaluate the estimator on real data.
#[derive(Debug)]
pub struct Estimator {
    /// Closed-form estimator expressions — one per parameter.
    ///
    /// `None` when no closed form was found (e.g. transcendental score equations).
    pub closed_form: Option<Vec<LoweredOp>>,

    /// Score equations `∂ℓ/∂θⱼ` (not set to zero — these are the derivatives;
    /// setting them to zero is what `derive` and `solve_system` do).
    pub score_equations: Vec<LoweredOp>,

    /// `true` when the closed-form solver could not reduce to explicit formulas
    /// and `fit()` will use Newton's method.
    pub falls_back_to_numeric: bool,

    /// Var ids for each parameter, in the same order as `params` passed to `derive`.
    pub(crate) params: Vec<usize>,

    /// Base Var id for the first data sample.
    pub(crate) data_var: usize,

    /// Number of data sample Var slots (Var(data_var) .. Var(data_var + n_samples - 1)).
    pub(crate) n_samples: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Derive MLE estimators for a parametric pdf.
///
/// # Arguments
///
/// - `pdf`: Symbolic pdf expression.  Data occupies `Var(data_var)` ..
///   `Var(data_var + n_samples - 1)` and parameters occupy `Var(params[j])`.
/// - `params`: Var ids for the parameters to estimate.
/// - `data_var`: Base Var id for data samples.
/// - `n_samples`: Number of data sample slots.
///
/// # Errors
///
/// - [`DeriveError::EmptyParams`] if `params` is empty.
/// - [`DeriveError::ZeroSamples`] if `n_samples == 0`.
/// - [`DeriveError::VarIdCollision`] if any parameter Var id falls inside the
///   data Var range `[data_var, data_var + n_samples)`.
pub fn derive(
    pdf: &LoweredOp,
    params: &[usize],
    data_var: usize,
    n_samples: usize,
) -> Result<Estimator, DeriveError> {
    if params.is_empty() {
        return Err(DeriveError::EmptyParams);
    }
    if n_samples == 0 {
        return Err(DeriveError::ZeroSamples);
    }

    // Guard: no Var-id overlap between data range and parameters
    for &p in params {
        if p >= data_var && p < data_var + n_samples {
            return Err(DeriveError::VarIdCollision { id: p });
        }
    }

    // ── Step 1: Build symbolic log-likelihood ──────────────────────────────
    // ℓ(θ) = Σᵢ ln(pdf(xᵢ; θ))
    // Substitute Var(data_var) → Var(data_var + i) in pdf for each sample i.
    let log_terms: Vec<LoweredOp> = (0..n_samples)
        .map(|i| {
            let pdf_i = substitute_var(pdf, data_var, data_var + i);
            LoweredOp::Ln(Box::new(pdf_i))
        })
        .collect();
    let log_likelihood = balanced_sum(&log_terms);

    // ── Step 2: Score equations ∂ℓ/∂θⱼ ────────────────────────────────────
    let score_equations: Vec<LoweredOp> = params
        .iter()
        .map(|&p| grad_canonical(&log_likelihood, p))
        .collect();

    // ── Step 3: Attempt closed-form solve via solve_system ─────────────────
    // Solve: score_j = 0 for all j.
    let eqs: Vec<(LoweredOp, LoweredOp)> = score_equations
        .iter()
        .map(|s| (s.clone(), LoweredOp::Const(0.0)))
        .collect();

    let (closed_form, falls_back_to_numeric) = match solve_system(&eqs, params) {
        Ok(result) if !result.solutions.is_empty() => {
            let sol = &result.solutions[0];
            let cf: Vec<LoweredOp> = params
                .iter()
                .map(|&p| sol.get(&p).cloned().unwrap_or(LoweredOp::Const(f64::NAN)))
                .collect();
            (Some(cf), false)
        }
        Ok(_) => (None, true), // empty solutions = underdetermined or inconsistent
        Err(SystemSolveError::CannotEliminateTranscendental) => (None, true),
        Err(SystemSolveError::EmptyVars | SystemSolveError::EmptyEquations) => (None, true),
        Err(SystemSolveError::InternalError(msg)) => {
            return Err(DeriveError::InternalError(msg));
        }
    };

    Ok(Estimator {
        closed_form,
        score_equations,
        falls_back_to_numeric,
        params: params.to_vec(),
        data_var,
        n_samples,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Estimator impl
// ─────────────────────────────────────────────────────────────────────────────

impl Estimator {
    /// Fit the estimator to data.
    ///
    /// If `closed_form` is available, substitutes the data Var bindings into
    /// each closed-form expression and evaluates numerically.  Otherwise,
    /// runs Newton's method on the score equations with a finite-difference
    /// Hessian approximation.
    ///
    /// # Errors
    ///
    /// - [`FitError::DataLengthMismatch`] if `data.len() != n_samples`.
    /// - [`FitError::NumericalFailed`] if evaluation or Newton iteration fails.
    pub fn fit(&self, data: ArrayView1<f64>) -> Result<Vec<f64>, FitError> {
        if data.len() != self.n_samples {
            return Err(FitError::DataLengthMismatch {
                expected: self.n_samples,
                got: data.len(),
            });
        }

        if let Some(ref cf) = self.closed_form {
            self.fit_closed_form(data, cf)
        } else {
            self.fit_newton(data)
        }
    }

    /// Evaluate each closed-form expression at the given data bindings.
    fn fit_closed_form(
        &self,
        data: ArrayView1<f64>,
        cf: &[LoweredOp],
    ) -> Result<Vec<f64>, FitError> {
        // Build a dense binding vector large enough for all Var ids.
        // We need slots for data vars and possibly the parameter Var slots
        // (which may appear in the closed-form expressions).
        let max_data_id = self.data_var + self.n_samples;
        let max_param_id = self.params.iter().copied().max().unwrap_or(0) + 1;
        let binding_len = max_data_id.max(max_param_id);
        let mut bindings = vec![0.0f64; binding_len];

        // Fill data slots
        for (i, &xi) in data.iter().enumerate() {
            let slot = self.data_var + i;
            if slot < bindings.len() {
                bindings[slot] = xi;
            }
        }

        let ctx = EvalCtx::new(&bindings);
        let mut estimates = Vec::with_capacity(cf.len());
        for expr in cf {
            let v = eval_real(expr, &ctx)
                .map_err(|e| FitError::NumericalFailed(format!("closed-form eval: {e}")))?;
            estimates.push(v);
        }
        Ok(estimates)
    }

    /// Newton's method fallback with finite-difference Hessian.
    fn fit_newton(&self, data: ArrayView1<f64>) -> Result<Vec<f64>, FitError> {
        let n = self.params.len();

        // Build base bindings (data slots are fixed, param slots vary).
        let max_data_id = self.data_var + self.n_samples;
        let max_param_id = self.params.iter().copied().max().unwrap_or(0) + 1;
        let binding_len = max_data_id.max(max_param_id);

        let mut base_bindings = vec![0.0f64; binding_len];
        for (i, &xi) in data.iter().enumerate() {
            let slot = self.data_var + i;
            if slot < base_bindings.len() {
                base_bindings[slot] = xi;
            }
        }

        // Initial guess: 0.5 for all parameters (safe for Bernoulli p, Normal σ).
        let mut theta: Vec<f64> = vec![0.5; n];
        let max_iter = 200_usize;
        let eps = 1e-5_f64; // finite-difference step
        let tol = 1e-8_f64;

        for _iter in 0..max_iter {
            // Build full bindings for current theta
            let mut bindings = base_bindings.clone();
            for (k, &p) in self.params.iter().enumerate() {
                if p < bindings.len() {
                    bindings[p] = theta[k];
                }
            }
            let ctx = EvalCtx::new(&bindings);

            // Evaluate score at current theta
            let mut score = Vec::with_capacity(n);
            for s in &self.score_equations {
                let sv = eval_real(s, &ctx)
                    .map_err(|e| FitError::NumericalFailed(format!("score eval: {e}")))?;
                if !sv.is_finite() {
                    return Err(FitError::NumericalFailed(format!(
                        "non-finite score at iteration {_iter}: {sv}"
                    )));
                }
                score.push(sv);
            }

            // Check convergence
            let norm: f64 = score.iter().map(|s| s * s).sum::<f64>().sqrt();
            if norm < tol {
                break;
            }

            // Finite-difference Jacobian of score (= Hessian of log-likelihood)
            let mut hessian = vec![vec![0.0f64; n]; n];
            for j in 0..n {
                let param_id = self.params[j];

                // Forward: theta[j] + eps
                let mut bindings_plus = base_bindings.clone();
                for (k, &p) in self.params.iter().enumerate() {
                    if p < bindings_plus.len() {
                        bindings_plus[p] = if k == j { theta[k] + eps } else { theta[k] };
                    }
                }
                // Backward: theta[j] - eps
                let mut bindings_minus = base_bindings.clone();
                for (k, &p) in self.params.iter().enumerate() {
                    if p < bindings_minus.len() {
                        bindings_minus[p] = if k == j { theta[k] - eps } else { theta[k] };
                    }
                }
                let _ = param_id; // used via the loop above

                let ctx_plus = EvalCtx::new(&bindings_plus);
                let ctx_minus = EvalCtx::new(&bindings_minus);

                for i in 0..n {
                    let sp = eval_real(&self.score_equations[i], &ctx_plus).unwrap_or(f64::NAN);
                    let sm = eval_real(&self.score_equations[i], &ctx_minus).unwrap_or(f64::NAN);
                    if !sp.is_finite() || !sm.is_finite() {
                        return Err(FitError::NumericalFailed(format!(
                            "non-finite finite-difference at H[{i}][{j}]"
                        )));
                    }
                    hessian[i][j] = (sp - sm) / (2.0 * eps);
                }
            }

            // Newton step: theta -= H^{-1} * score
            let delta = solve_linear_system_f64(&hessian, &score)
                .map_err(|e| FitError::NumericalFailed(format!("Hessian solve: {e}")))?;

            for k in 0..n {
                theta[k] -= delta[k];
            }
        }

        Ok(theta)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Substitute `Var(old_id)` with `Var(new_id)` throughout an expression.
///
/// Iterative work-stack traversal — safe for deep trees.
fn substitute_var(op: &LoweredOp, old_id: usize, new_id: usize) -> LoweredOp {
    if old_id == new_id {
        return op.clone();
    }

    enum Frame<'a> {
        Enter(&'a LoweredOp),
        Build(&'a LoweredOp),
    }

    let mut frame_stack: Vec<Frame> = vec![Frame::Enter(op)];
    let mut result_stack: Vec<LoweredOp> = Vec::new();

    while let Some(frame) = frame_stack.pop() {
        match frame {
            Frame::Enter(node) => match node {
                LoweredOp::Const(_) | LoweredOp::Var(_) => {
                    frame_stack.push(Frame::Build(node));
                }
                LoweredOp::Add(a, b)
                | LoweredOp::Sub(a, b)
                | LoweredOp::Mul(a, b)
                | LoweredOp::Div(a, b)
                | LoweredOp::Pow(a, b) => {
                    frame_stack.push(Frame::Build(node));
                    frame_stack.push(Frame::Enter(b));
                    frame_stack.push(Frame::Enter(a));
                }
                LoweredOp::Neg(c)
                | LoweredOp::Exp(c)
                | LoweredOp::Ln(c)
                | LoweredOp::Sin(c)
                | LoweredOp::Cos(c)
                | LoweredOp::Tan(c)
                | LoweredOp::Sinh(c)
                | LoweredOp::Cosh(c)
                | LoweredOp::Tanh(c)
                | LoweredOp::Arcsin(c)
                | LoweredOp::Arccos(c)
                | LoweredOp::Arctan(c)
                | LoweredOp::Arcsinh(c)
                | LoweredOp::Arccosh(c)
                | LoweredOp::Arctanh(c)
                | LoweredOp::Sqrt(c)
                | LoweredOp::Abs(c) => {
                    frame_stack.push(Frame::Build(node));
                    frame_stack.push(Frame::Enter(c));
                }
            },
            Frame::Build(node) => {
                let built = match node {
                    LoweredOp::Const(c) => LoweredOp::Const(*c),
                    LoweredOp::Var(v) => {
                        if *v == old_id {
                            LoweredOp::Var(new_id)
                        } else {
                            LoweredOp::Var(*v)
                        }
                    }
                    LoweredOp::Add(_, _) => {
                        let b = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        let a = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Add(Box::new(a), Box::new(b))
                    }
                    LoweredOp::Sub(_, _) => {
                        let b = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        let a = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Sub(Box::new(a), Box::new(b))
                    }
                    LoweredOp::Mul(_, _) => {
                        let b = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        let a = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Mul(Box::new(a), Box::new(b))
                    }
                    LoweredOp::Div(_, _) => {
                        let b = result_stack.pop().unwrap_or(LoweredOp::Const(1.0));
                        let a = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Div(Box::new(a), Box::new(b))
                    }
                    LoweredOp::Pow(_, _) => {
                        let b = result_stack.pop().unwrap_or(LoweredOp::Const(1.0));
                        let a = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Pow(Box::new(a), Box::new(b))
                    }
                    LoweredOp::Neg(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Neg(Box::new(c))
                    }
                    LoweredOp::Exp(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Exp(Box::new(c))
                    }
                    LoweredOp::Ln(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Ln(Box::new(c))
                    }
                    LoweredOp::Sin(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Sin(Box::new(c))
                    }
                    LoweredOp::Cos(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Cos(Box::new(c))
                    }
                    LoweredOp::Tan(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Tan(Box::new(c))
                    }
                    LoweredOp::Sinh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Sinh(Box::new(c))
                    }
                    LoweredOp::Cosh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Cosh(Box::new(c))
                    }
                    LoweredOp::Tanh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Tanh(Box::new(c))
                    }
                    LoweredOp::Arcsin(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arcsin(Box::new(c))
                    }
                    LoweredOp::Arccos(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arccos(Box::new(c))
                    }
                    LoweredOp::Arctan(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arctan(Box::new(c))
                    }
                    LoweredOp::Arcsinh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arcsinh(Box::new(c))
                    }
                    LoweredOp::Arccosh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arccosh(Box::new(c))
                    }
                    LoweredOp::Arctanh(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Arctanh(Box::new(c))
                    }
                    LoweredOp::Sqrt(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Sqrt(Box::new(c))
                    }
                    LoweredOp::Abs(_) => {
                        let c = result_stack.pop().unwrap_or(LoweredOp::Const(0.0));
                        LoweredOp::Abs(Box::new(c))
                    }
                };
                result_stack.push(built);
            }
        }
    }

    result_stack.pop().unwrap_or(LoweredOp::Const(0.0))
}

/// Build a balanced binary Add tree from a slice of LoweredOp.
///
/// Empty slice → `Const(0)`.  Single element → that element.  Otherwise
/// recurse on halves.
fn balanced_sum(ops: &[LoweredOp]) -> LoweredOp {
    match ops.len() {
        0 => LoweredOp::Const(0.0),
        1 => ops[0].clone(),
        _ => {
            let mid = ops.len() / 2;
            let left = balanced_sum(&ops[..mid]);
            let right = balanced_sum(&ops[mid..]);
            LoweredOp::Add(Box::new(left), Box::new(right))
        }
    }
}

/// Gaussian elimination (partial pivot) to solve Ax = b for small dense
/// systems.  Returns the solution vector.
///
/// # Errors
///
/// Returns `Err` with a description if the system is singular or degenerate.
fn solve_linear_system_f64(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, String> {
    let n = b.len();
    if a.len() != n {
        return Err(format!("matrix rows ({}) != rhs length ({})", a.len(), n));
    }

    // Build augmented matrix [A | b]
    let mut mat: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot row (largest absolute value in column)
        let pivot = (col..n)
            .max_by(|&i, &j| {
                mat[i][col]
                    .abs()
                    .partial_cmp(&mat[j][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| "empty pivot range".to_string())?;

        mat.swap(col, pivot);

        let diag = mat[col][col];
        if diag.abs() < 1e-14 {
            return Err(format!("singular matrix at column {col}"));
        }

        // Normalize pivot row
        let inv_diag = 1.0 / diag;
        for k in col..=n {
            mat[col][k] *= inv_diag;
        }

        // Eliminate column in all other rows
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = mat[row][col];
            if factor.abs() < 1e-15 {
                continue;
            }
            for k in col..=n {
                let v = mat[col][k];
                mat[row][k] -= factor * v;
            }
        }
    }

    Ok(mat.iter().map(|row| row[n]).collect())
}
