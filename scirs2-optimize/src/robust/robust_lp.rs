//! Robust Linear Programming
//!
//! This module implements robust counterparts of linear programs under various
//! uncertainty models. A standard LP:
//!
//! ```text
//! min  cᵀ x
//! s.t. A x ≤ b,  x ∈ X
//! ```
//!
//! becomes a *robust LP* when the data (c, A, b) are uncertain:
//!
//! ```text
//! min  max_{(c,A,b) ∈ U}  cᵀ x
//! s.t. A x ≤ b  for all  (A, b) ∈ U_constraints
//! ```
//!
//! # Uncertainty Models
//!
//! - **Box uncertainty** (`box_robust_lp`): Each coefficient independently perturbed
//!   within ±δ. The robust counterpart is an LP of comparable size.
//! - **Ellipsoidal uncertainty** (`ellipsoidal_robust_lp`): Coefficients perturbed inside
//!   an ellipsoid. The robust counterpart is a Second-Order Cone Program (SOCP).
//! - **Worst-case objective** (`robust_objective`): Evaluate worst-case objective value
//!   under any supported uncertainty set without full reformulation.
//!
//! # Reformulation Approach
//!
//! Both reformulations are solved via projected gradient descent on the resulting
//! tractable problem (no external LP/SOCP solver is needed).
//!
//! # References
//!
//! - Ben-Tal, A. & Nemirovski, A. (1998). "Robust convex optimization". *Mathematics of Operations Research*.
//! - Ben-Tal, A. & Nemirovski, A. (1999). "Robust solutions of uncertain linear programs". *Operations Research Letters*.
//! - Lobo, M.S. et al. (1998). "Applications of second-order cone programming". *Linear Algebra and its Applications*.
//! - Bertsimas, D. & Sim, M. (2004). "The price of robustness". *Operations Research*.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ─── Robust LP problem definition ─────────────────────────────────────────────

/// A linear program with uncertain data.
///
/// Nominal problem:
/// ```text
/// min  cᵀ x
/// s.t. A x ≤ b
///      lb ≤ x ≤ ub   (optional bounds)
/// ```
#[derive(Debug, Clone)]
pub struct RobustLP {
    /// Nominal objective coefficient vector c (n-vector).
    pub c: Array1<f64>,
    /// Nominal constraint matrix A (m × n).
    pub a_matrix: Array2<f64>,
    /// Nominal right-hand side b (m-vector).
    pub b_rhs: Array1<f64>,
    /// Optional lower bounds on x (length n; use f64::NEG_INFINITY for unbounded).
    pub lb: Option<Array1<f64>>,
    /// Optional upper bounds on x (length n; use f64::INFINITY for unbounded).
    pub ub: Option<Array1<f64>>,
    /// Uncertainty in objective coefficients c: perturbation radius per coordinate.
    pub c_uncertainty: Option<Array1<f64>>,
    /// Uncertainty in constraint matrix A: perturbation radius per entry (m × n).
    pub a_uncertainty: Option<Array2<f64>>,
    /// Uncertainty in right-hand side b: perturbation radius per constraint.
    pub b_uncertainty: Option<Array1<f64>>,
}

impl RobustLP {
    /// Create a new robust LP with no uncertainty (reduces to nominal LP).
    ///
    /// # Arguments
    ///
    /// * `c`        – objective vector (n)
    /// * `a_matrix` – constraint matrix (m × n)
    /// * `b_rhs`    – right-hand side (m)
    pub fn new(c: Array1<f64>, a_matrix: Array2<f64>, b_rhs: Array1<f64>) -> OptimizeResult<Self> {
        let n = c.len();
        let (m, nc) = (a_matrix.shape()[0], a_matrix.shape()[1]);
        if nc != n {
            return Err(OptimizeError::ValueError(format!(
                "A has {} columns but c has length {}",
                nc, n
            )));
        }
        if b_rhs.len() != m {
            return Err(OptimizeError::ValueError(format!(
                "b has length {} but A has {} rows",
                b_rhs.len(),
                m
            )));
        }
        Ok(Self {
            c,
            a_matrix,
            b_rhs,
            lb: None,
            ub: None,
            c_uncertainty: None,
            a_uncertainty: None,
            b_uncertainty: None,
        })
    }

    /// Set box uncertainty on the objective: c̃ ∈ [c - δ_c, c + δ_c].
    pub fn with_c_uncertainty(mut self, delta_c: Array1<f64>) -> OptimizeResult<Self> {
        if delta_c.len() != self.c.len() {
            return Err(OptimizeError::ValueError(format!(
                "delta_c has length {} but c has length {}",
                delta_c.len(),
                self.c.len()
            )));
        }
        self.c_uncertainty = Some(delta_c);
        Ok(self)
    }

    /// Set box uncertainty on constraints: Ã_ij ∈ [A_ij - δ_A_ij, A_ij + δ_A_ij].
    pub fn with_a_uncertainty(mut self, delta_a: Array2<f64>) -> OptimizeResult<Self> {
        if delta_a.shape() != self.a_matrix.shape() {
            return Err(OptimizeError::ValueError(format!(
                "delta_A shape {:?} does not match A shape {:?}",
                delta_a.shape(),
                self.a_matrix.shape()
            )));
        }
        self.a_uncertainty = Some(delta_a);
        Ok(self)
    }

    /// Set box uncertainty on the RHS: b̃_i ∈ [b_i - δ_b_i, b_i + δ_b_i].
    pub fn with_b_uncertainty(mut self, delta_b: Array1<f64>) -> OptimizeResult<Self> {
        if delta_b.len() != self.b_rhs.len() {
            return Err(OptimizeError::ValueError(format!(
                "delta_b has length {} but b has length {}",
                delta_b.len(),
                self.b_rhs.len()
            )));
        }
        self.b_uncertainty = Some(delta_b);
        Ok(self)
    }

    /// Set variable bounds.
    pub fn with_bounds(mut self, lb: Array1<f64>, ub: Array1<f64>) -> OptimizeResult<Self> {
        let n = self.c.len();
        if lb.len() != n || ub.len() != n {
            return Err(OptimizeError::ValueError(format!(
                "bounds have length {}/{} but n={}",
                lb.len(),
                ub.len(),
                n
            )));
        }
        self.lb = Some(lb);
        self.ub = Some(ub);
        Ok(self)
    }

    /// Number of variables.
    pub fn n_vars(&self) -> usize {
        self.c.len()
    }

    /// Number of constraints.
    pub fn n_constraints(&self) -> usize {
        self.b_rhs.len()
    }
}

/// Result of a robust LP solve.
#[derive(Debug, Clone)]
pub struct RobustLPResult {
    /// Optimal solution vector x*.
    pub x: Array1<f64>,
    /// Robust objective value (worst-case cost cᵀ x under uncertainty).
    pub fun: f64,
    /// Nominal objective value cᵀ x (without uncertainty penalty).
    pub nominal_fun: f64,
    /// Robust constraint slack: min_i (b_i - (Ax)_i) (positive = feasible).
    pub constraint_slack: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Status message.
    pub message: String,
}

/// Configuration for robust LP solvers.
#[derive(Debug, Clone)]
pub struct RobustLPConfig {
    /// Maximum projected gradient iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Initial step size for projected gradient descent.
    pub step_size: f64,
    /// Step size reduction factor (Armijo backtracking).
    pub step_reduction: f64,
    /// Penalty weight for constraint violations.
    pub constraint_penalty: f64,
}

impl Default for RobustLPConfig {
    fn default() -> Self {
        Self {
            max_iter: 5_000,
            tol: 1e-6,
            step_size: 1e-2,
            step_reduction: 0.5,
            constraint_penalty: 100.0,
        }
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Project x onto the box [lb, ub].
fn project_box(x: &Array1<f64>, lb: &Option<Array1<f64>>, ub: &Option<Array1<f64>>) -> Array1<f64> {
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let lo = lb.as_ref().map(|b| b[i]).unwrap_or(f64::NEG_INFINITY);
            let hi = ub.as_ref().map(|b| b[i]).unwrap_or(f64::INFINITY);
            xi.max(lo).min(hi)
        })
        .collect()
}

/// Evaluate Ax - b (constraint residual). Positive entries are violated.
fn constraint_residual(
    a: &ArrayView2<f64>,
    x: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
) -> Array1<f64> {
    let m = b.len();
    let n = x.len();
    let mut r = Array1::<f64>::zeros(m);
    for i in 0..m {
        let ax_i: f64 = (0..n).map(|j| a[[i, j]] * x[j]).sum();
        r[i] = ax_i - b[i];
    }
    r
}

/// Gradient of the penalized objective:
///   φ(x) = c̃ᵀ x + penalty * Σ_i max(A_row_i · x - b_i, 0)²
/// where c̃ is the worst-case objective coefficient.
fn penalized_gradient(
    x: &Array1<f64>,
    c_worst: &Array1<f64>,
    a: &Array2<f64>,
    b: &Array1<f64>,
    penalty: f64,
) -> Array1<f64> {
    let n = x.len();
    let m = b.len();

    // Gradient of c̃ᵀ x
    let mut grad = c_worst.clone();

    // Gradient of penalty for violated constraints
    for i in 0..m {
        let ax_i: f64 = (0..n).map(|j| a[[i, j]] * x[j]).sum();
        let viol = ax_i - b[i];
        if viol > 0.0 {
            // d/dx [ penalty * viol² ] = 2 * penalty * viol * A[i, :]
            for j in 0..n {
                grad[j] += 2.0 * penalty * viol * a[[i, j]];
            }
        }
    }
    grad
}

// ─── Box Robust LP ────────────────────────────────────────────────────────────

/// Solve a robust LP with box uncertainty via projected gradient descent.
///
/// **Box uncertainty model**:
/// - Objective: c̃ ∈ [c - δ_c, c + δ_c] → worst-case cost = cᵀ x + δ_cᵀ |x|
/// - Constraints: Ã x ≤ b̃ for all Ã ∈ [A - δ_A, A + δ_A], b̃ ∈ [b - δ_b, b + δ_b]
///   → Robust constraint: A x + δ_A |x| ≤ b - δ_b (Ben-Tal & Nemirovski 1999)
///
/// The robust LP with box uncertainty is again an LP (only slightly larger).
/// We solve it via penalized projected gradient descent.
///
/// # Arguments
///
/// * `problem` – robust LP instance (with box uncertainty fields set)
/// * `x0`      – feasible initial point
/// * `config`  – solver configuration
///
/// # Returns
///
/// [`RobustLPResult`] with robust-optimal solution.
pub fn box_robust_lp(
    problem: &RobustLP,
    x0: &ArrayView1<f64>,
    config: &RobustLPConfig,
) -> OptimizeResult<RobustLPResult> {
    let n = problem.n_vars();
    let m = problem.n_constraints();
    if x0.len() != n {
        return Err(OptimizeError::ValueError(format!(
            "x0 has length {} but problem has {} variables",
            x0.len(),
            n
        )));
    }

    // Worst-case objective coefficient: c̃ = c + δ_c * sign(x)
    // We use absolute value reformulation: worst-case cᵀ x = cᵀ x + δ_cᵀ |x|
    // Gradient: d/dx [c̃ᵀ x] = c + δ_c * sign(x)
    let delta_c = problem
        .c_uncertainty
        .clone()
        .unwrap_or_else(|| Array1::zeros(n));

    // Robust RHS: b̃ = b - δ_b (tighten constraints)
    let delta_b = problem
        .b_uncertainty
        .clone()
        .unwrap_or_else(|| Array1::zeros(m));
    let b_robust: Array1<f64> = problem
        .b_rhs
        .iter()
        .zip(delta_b.iter())
        .map(|(&bi, &dbi)| bi - dbi.abs())
        .collect();

    // Robust A: Ã = A, but add row-wise perturbation to RHS (absorption into b_robust)
    // Constraint: (A + δ_A * sign(x)) x ≤ b → A x + δ_A |x| ≤ b
    let delta_a = problem
        .a_uncertainty
        .clone()
        .unwrap_or_else(|| Array2::zeros((m, n)));

    let mut x = x0.to_owned();
    let mut converged = false;
    let mut step = config.step_size;

    for iter in 0..config.max_iter {
        // Worst-case objective: c̃(x) = c + δ_c * sign(x)
        let c_worst: Array1<f64> = problem
            .c
            .iter()
            .zip(delta_c.iter())
            .zip(x.iter())
            .map(|((&ci, &dci), &xi)| ci + dci * xi.signum())
            .collect();

        // Robust constraint: A x + δ_A |x| ≤ b_robust
        // Incorporate δ_A |x| into effective RHS reduction
        let b_effective: Array1<f64> = (0..m)
            .map(|i| {
                let reduction: f64 = (0..n).map(|j| delta_a[[i, j]] * x[j].abs()).sum();
                b_robust[i] - reduction
            })
            .collect();

        let grad = penalized_gradient(
            &x,
            &c_worst,
            &problem.a_matrix,
            &b_effective,
            config.constraint_penalty,
        );

        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.tol {
            converged = true;
            break;
        }

        // Backtracking line search
        let obj_curr: f64 = problem
            .c
            .iter()
            .zip(x.iter())
            .map(|(&ci, &xi)| ci * xi)
            .sum::<f64>()
            + delta_c
                .iter()
                .zip(x.iter())
                .map(|(&di, &xi)| di * xi.abs())
                .sum::<f64>();

        let mut accepted = false;
        for _ in 0..20 {
            let x_new: Array1<f64> = x
                .iter()
                .zip(grad.iter())
                .map(|(&xi, &gi)| xi - step * gi)
                .collect();
            let x_proj = project_box(&x_new, &problem.lb, &problem.ub);

            let obj_new: f64 = problem
                .c
                .iter()
                .zip(x_proj.iter())
                .map(|(&ci, &xi)| ci * xi)
                .sum::<f64>()
                + delta_c
                    .iter()
                    .zip(x_proj.iter())
                    .map(|(&di, &xi)| di * xi.abs())
                    .sum::<f64>();

            if obj_new <= obj_curr - 1e-4 * step * grad_norm * grad_norm {
                x = x_proj;
                accepted = true;
                break;
            }
            step *= config.step_reduction;
        }

        if !accepted {
            // Take a small step anyway to avoid stagnation
            let x_new: Array1<f64> = x
                .iter()
                .zip(grad.iter())
                .map(|(&xi, &gi)| xi - step * gi)
                .collect();
            x = project_box(&x_new, &problem.lb, &problem.ub);
        }

        // Reset step size for next iteration (with mild increase)
        if iter % 100 == 99 {
            step = (step * 1.1).min(config.step_size);
        }
    }

    // Compute result metrics
    let nominal_fun: f64 = problem
        .c
        .iter()
        .zip(x.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum();
    let robust_fun: f64 = nominal_fun
        + delta_c
            .iter()
            .zip(x.iter())
            .map(|(&di, &xi)| di * xi.abs())
            .sum::<f64>();

    let residual = constraint_residual(&problem.a_matrix.view(), &x.view(), &problem.b_rhs.view());
    let constraint_slack = -residual.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok(RobustLPResult {
        x,
        fun: robust_fun,
        nominal_fun,
        constraint_slack,
        n_iter: config.max_iter,
        converged,
        message: if converged {
            "Box robust LP converged".to_string()
        } else {
            "Box robust LP reached maximum iterations".to_string()
        },
    })
}

// ─── Ellipsoidal Robust LP (SOCP Reformulation) ──────────────────────────────

/// Solve a robust LP with ellipsoidal uncertainty on the objective.
///
/// **Ellipsoidal uncertainty model** (Ben-Tal & Nemirovski 1998):
///
/// The cost vector satisfies c̃ ∈ E = {c + Σ^{1/2} ξ : ‖ξ‖₂ ≤ ρ}, where
/// Σ is a positive-semidefinite covariance matrix and ρ is the radius.
///
/// Worst-case objective:
/// ```text
/// max_{c̃ ∈ E} c̃ᵀ x = cᵀ x + ρ ‖Σ^{1/2} x‖₂
/// ```
///
/// This is the SOCP reformulation: we minimize cᵀ x + ρ ‖Σ^{1/2} x‖₂
/// subject to the constraints, which is a convex (non-smooth) problem.
///
/// # Arguments
///
/// * `problem`          – robust LP instance
/// * `c_covariance`     – PSD covariance matrix Σ for objective uncertainty (n×n)
/// * `ellipsoid_radius` – radius ρ of the ellipsoidal uncertainty set
/// * `x0`               – initial point
/// * `config`           – solver configuration
///
/// # Returns
///
/// [`RobustLPResult`] with the robust-optimal solution under ellipsoidal uncertainty.
///
/// # References
///
/// Ben-Tal & Nemirovski (1999), "Robust solutions of uncertain linear programs".
pub fn ellipsoidal_robust_lp(
    problem: &RobustLP,
    c_covariance: &ArrayView2<f64>,
    ellipsoid_radius: f64,
    x0: &ArrayView1<f64>,
    config: &RobustLPConfig,
) -> OptimizeResult<RobustLPResult> {
    let n = problem.n_vars();
    let m = problem.n_constraints();
    if x0.len() != n {
        return Err(OptimizeError::ValueError(format!(
            "x0 has length {} but problem has {} variables",
            x0.len(),
            n
        )));
    }
    if c_covariance.shape() != [n, n] {
        return Err(OptimizeError::ValueError(format!(
            "c_covariance shape {:?} != [{n},{n}]",
            c_covariance.shape()
        )));
    }
    if ellipsoid_radius < 0.0 {
        return Err(OptimizeError::ValueError(
            "ellipsoid_radius must be non-negative".to_string(),
        ));
    }

    // Compute Cholesky factor L of Σ: Σ = L Lᵀ
    let l_chol = cholesky_lower_triangular(c_covariance)?;

    let mut x = x0.to_owned();
    let mut converged = false;
    let h = 1e-7;
    let step = config.step_size;

    for _ in 0..config.max_iter {
        // Worst-case objective: F(x) = cᵀ x + ρ ‖L x‖₂
        // ∇F(x) = c + ρ Lᵀ (L x / ‖L x‖₂)
        let lx = mat_vec_mul_lower(&l_chol, &x.view());
        let lx_norm = l2_norm_vec(&lx);

        let socp_grad: Array1<f64> = if lx_norm > h {
            // ∇(ρ ‖L x‖₂) = ρ Lᵀ (L x / ‖L x‖₂)
            let lx_normalized: Array1<f64> = lx.iter().map(|&v| v / lx_norm).collect();
            let lt_v = mat_vec_mul_lower_transpose(&l_chol, &lx_normalized.view());
            problem
                .c
                .iter()
                .zip(lt_v.iter())
                .map(|(&ci, &li)| ci + ellipsoid_radius * li)
                .collect()
        } else {
            // At the origin, subgradient of ‖L x‖₂ is any vector in the unit ball
            problem.c.clone()
        };

        // Add constraint penalty gradient
        let residual =
            constraint_residual(&problem.a_matrix.view(), &x.view(), &problem.b_rhs.view());
        let mut full_grad = socp_grad;
        for i in 0..m {
            if residual[i] > 0.0 {
                for j in 0..n {
                    full_grad[j] +=
                        2.0 * config.constraint_penalty * residual[i] * problem.a_matrix[[i, j]];
                }
            }
        }

        let grad_norm = l2_norm_vec(&full_grad);
        if grad_norm < config.tol {
            converged = true;
            break;
        }

        let x_new: Array1<f64> = x
            .iter()
            .zip(full_grad.iter())
            .map(|(&xi, &gi)| xi - step * gi)
            .collect();
        x = project_box(&x_new, &problem.lb, &problem.ub);
    }

    let nominal_fun: f64 = problem
        .c
        .iter()
        .zip(x.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum();
    let lx = mat_vec_mul_lower(&l_chol, &x.view());
    let robust_fun = nominal_fun + ellipsoid_radius * l2_norm_vec(&lx);

    let residual = constraint_residual(&problem.a_matrix.view(), &x.view(), &problem.b_rhs.view());
    let constraint_slack = -residual.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok(RobustLPResult {
        x,
        fun: robust_fun,
        nominal_fun,
        constraint_slack,
        n_iter: config.max_iter,
        converged,
        message: if converged {
            "Ellipsoidal robust LP converged".to_string()
        } else {
            "Ellipsoidal robust LP reached maximum iterations".to_string()
        },
    })
}

// ─── Robust Objective Evaluation ─────────────────────────────────────────────

/// Evaluate the worst-case objective value of cᵀ x under the given uncertainty model.
///
/// Supports:
/// - **Box**: worst-case = cᵀ x + δ_cᵀ |x|
/// - **Ellipsoidal**: worst-case = cᵀ x + ρ ‖Σ^{1/2} x‖₂
/// - **Budget** (Bertsimas-Sim): worst-case = cᵀ x + top-Γ perturbation
///
/// # Arguments
///
/// * `c`     – nominal objective vector
/// * `x`     – solution to evaluate
/// * `model` – uncertainty model specification
///
/// # Returns
///
/// Worst-case objective value.
pub fn robust_objective(
    c: &ArrayView1<f64>,
    x: &ArrayView1<f64>,
    model: &ObjectiveUncertaintyModel,
) -> OptimizeResult<f64> {
    let n = c.len();
    if x.len() != n {
        return Err(OptimizeError::ValueError(format!(
            "c has length {} but x has length {}",
            n,
            x.len()
        )));
    }

    let nominal: f64 = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();

    let penalty = match model {
        ObjectiveUncertaintyModel::Box { delta } => {
            if delta.len() != n {
                return Err(OptimizeError::ValueError(format!(
                    "delta has length {} but n={}",
                    delta.len(),
                    n
                )));
            }
            // max perturbation = δ_i * |x_i|
            delta
                .iter()
                .zip(x.iter())
                .map(|(&di, &xi)| di * xi.abs())
                .sum::<f64>()
        }
        ObjectiveUncertaintyModel::Ellipsoidal { covariance, radius } => {
            if covariance.shape() != [n, n] {
                return Err(OptimizeError::ValueError(format!(
                    "covariance shape {:?} != [{n},{n}]",
                    covariance.shape()
                )));
            }
            // penalty = ρ * ‖Σ^{1/2} x‖₂ = ρ * sqrt(xᵀ Σ x)
            let sigma_x: f64 = (0..n)
                .map(|i| {
                    let row: f64 = (0..n).map(|j| covariance[[i, j]] * x[j]).sum();
                    row * x[i]
                })
                .sum();
            radius * sigma_x.sqrt()
        }
        ObjectiveUncertaintyModel::Budget { delta, budget } => {
            if delta.len() != n {
                return Err(OptimizeError::ValueError(format!(
                    "delta has length {} but n={}",
                    delta.len(),
                    n
                )));
            }
            // Bertsimas-Sim: sort δ_i |x_i| descending, take sum of top Γ entries
            let mut perturbations: Vec<f64> = delta
                .iter()
                .zip(x.iter())
                .map(|(&di, &xi)| di * xi.abs())
                .collect();
            perturbations.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            let gamma = (*budget as usize).min(n);
            let floor_gamma = budget.floor() as usize;
            let frac = budget - budget.floor();

            // Integer part: sum of top floor(Γ) terms
            let int_sum: f64 = perturbations.iter().take(floor_gamma).sum();
            // Fractional part: add fraction of next term
            let frac_sum = if floor_gamma < perturbations.len() && gamma <= perturbations.len() {
                frac * perturbations[floor_gamma]
            } else {
                0.0
            };
            int_sum + frac_sum
        }
    };

    Ok(nominal + penalty)
}

/// Uncertainty model for the objective vector c in cᵀ x.
#[derive(Debug, Clone)]
pub enum ObjectiveUncertaintyModel {
    /// Box uncertainty: c̃ ∈ [c - δ, c + δ], worst-case = cᵀ x + δᵀ |x|.
    Box {
        /// Per-component perturbation radius δ_i ≥ 0.
        delta: Array1<f64>,
    },
    /// Ellipsoidal uncertainty: c̃ = c + Σ^{1/2} ξ, ‖ξ‖₂ ≤ ρ.
    /// Worst-case = cᵀ x + ρ ‖Σ^{1/2} x‖₂ = cᵀ x + ρ √(xᵀ Σ x).
    Ellipsoidal {
        /// PSD covariance matrix Σ (n×n).
        covariance: Array2<f64>,
        /// Ellipsoid radius ρ.
        radius: f64,
    },
    /// Bertsimas-Sim budgeted uncertainty: at most Γ components can deviate.
    /// Worst-case = cᵀ x + Σ_top_Γ δ_i |x_i|.
    Budget {
        /// Per-component radius δ_i.
        delta: Array1<f64>,
        /// Budget parameter Γ ∈ [0, n].
        budget: f64,
    },
}

// ─── Linear algebra helpers ──────────────────────────────────────────────────

/// Lower-triangular Cholesky factor of a symmetric PSD matrix.
fn cholesky_lower_triangular(a: &ArrayView2<f64>) -> OptimizeResult<Array2<f64>> {
    let n = a.shape()[0];
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = 0.0;
            for p in 0..j {
                s += l[[i, p]] * l[[j, p]];
            }
            if i == j {
                let diag = a[[i, i]] - s;
                l[[i, j]] = if diag < 0.0 {
                    diag.abs().sqrt().max(1e-12)
                } else {
                    diag.sqrt()
                };
            } else {
                let ljj = l[[j, j]];
                l[[i, j]] = if ljj.abs() < 1e-14 {
                    0.0
                } else {
                    (a[[i, j]] - s) / ljj
                };
            }
        }
    }
    Ok(l)
}

/// Lower-triangular matrix-vector product: y = L x.
fn mat_vec_mul_lower(l: &Array2<f64>, x: &ArrayView1<f64>) -> Array1<f64> {
    let n = x.len();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in 0..=i {
            y[i] += l[[i, j]] * x[j];
        }
    }
    y
}

/// Transpose lower-triangular matrix-vector product: y = Lᵀ x.
fn mat_vec_mul_lower_transpose(l: &Array2<f64>, x: &ArrayView1<f64>) -> Array1<f64> {
    let n = x.len();
    let mut y = Array1::<f64>::zeros(n);
    for j in 0..n {
        for i in j..n {
            y[j] += l[[i, j]] * x[i];
        }
    }
    y
}

/// L2 norm of a vector.
fn l2_norm_vec(v: &Array1<f64>) -> f64 {
    v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    fn simple_lp() -> OptimizeResult<RobustLP> {
        // min -x[0] - x[1]  s.t. x[0]+x[1] ≤ 1, x[0] ≥ 0, x[1] ≥ 0
        // Optimal: x=(0.5, 0.5), obj=-1
        let c = array![-1.0, -1.0];
        let a = array![[1.0, 1.0]];
        let b = array![1.0];
        RobustLP::new(c, a, b)
    }

    #[test]
    fn test_robust_lp_new_dimension_mismatch() {
        let c = array![1.0, 2.0];
        let a = array![[1.0, 2.0, 3.0]]; // wrong columns
        let b = array![1.0];
        assert!(RobustLP::new(c, a, b).is_err());
    }

    #[test]
    fn test_box_robust_lp_no_uncertainty() {
        let problem = simple_lp()
            .expect("failed to create problem")
            .with_bounds(array![0.0, 0.0], array![1.0, 1.0])
            .expect("unexpected None or Err");
        let x0 = array![0.0, 0.0];
        let config = RobustLPConfig {
            max_iter: 3_000,
            step_size: 1e-2,
            constraint_penalty: 50.0,
            ..Default::default()
        };
        let result = box_robust_lp(&problem, &x0.view(), &config).expect("failed to create result");
        // Without uncertainty, robust = nominal
        assert!(
            result.nominal_fun < 0.0,
            "nominal fun should be negative (minimizing -x)"
        );
    }

    #[test]
    fn test_box_robust_lp_with_uncertainty() {
        let problem = simple_lp()
            .expect("unexpected None or Err")
            .with_c_uncertainty(array![0.1, 0.1])
            .expect("unexpected None or Err")
            .with_bounds(array![0.0, 0.0], array![1.0, 1.0])
            .expect("unexpected None or Err");
        let x0 = array![0.1, 0.1];
        let config = RobustLPConfig {
            max_iter: 2_000,
            step_size: 5e-3,
            constraint_penalty: 50.0,
            ..Default::default()
        };
        let result = box_robust_lp(&problem, &x0.view(), &config).expect("failed to create result");
        // Robust objective should be ≥ nominal (uncertainty adds penalty)
        assert!(
            result.fun >= result.nominal_fun - 1e-9,
            "robust obj {} should be ≥ nominal obj {}",
            result.fun,
            result.nominal_fun
        );
    }

    #[test]
    fn test_ellipsoidal_robust_lp() {
        let problem = simple_lp()
            .expect("unexpected None or Err")
            .with_bounds(array![0.0, 0.0], array![1.0, 1.0])
            .expect("unexpected None or Err");
        let cov = Array2::<f64>::eye(2) * 0.01;
        let x0 = array![0.1, 0.1];
        let config = RobustLPConfig {
            max_iter: 2_000,
            step_size: 5e-3,
            constraint_penalty: 50.0,
            ..Default::default()
        };
        let result = ellipsoidal_robust_lp(&problem, &cov.view(), 0.5, &x0.view(), &config)
            .expect("unexpected None or Err");
        // Result should be a valid solution
        assert!(result.x.len() == 2);
        assert!(result.fun.is_finite());
    }

    #[test]
    fn test_robust_objective_box() {
        let c = array![1.0, 2.0, -1.0];
        let x = array![1.0, -1.0, 2.0];
        let model = ObjectiveUncertaintyModel::Box {
            delta: array![0.1, 0.1, 0.1],
        };
        let obj = robust_objective(&c.view(), &x.view(), &model).expect("failed to create obj");
        // nominal = 1*1 + 2*(-1) + (-1)*2 = 1 - 2 - 2 = -3
        // penalty = 0.1*1 + 0.1*1 + 0.1*2 = 0.4
        assert!((obj - (-3.0 + 0.4)).abs() < 1e-9, "box robust obj={obj}");
    }

    #[test]
    fn test_robust_objective_ellipsoidal() {
        let c = array![1.0, 0.0];
        let x = array![1.0, 0.0];
        let cov = Array2::<f64>::eye(2);
        let model = ObjectiveUncertaintyModel::Ellipsoidal {
            covariance: cov,
            radius: 1.0,
        };
        let obj = robust_objective(&c.view(), &x.view(), &model).expect("failed to create obj");
        // nominal = 1, penalty = 1 * sqrt(1^2) = 1 → total = 2
        assert!((obj - 2.0).abs() < 1e-9, "ellipsoidal robust obj={obj}");
    }

    #[test]
    fn test_robust_objective_budget() {
        let c = array![1.0, 1.0, 1.0];
        let x = array![1.0, 1.0, 1.0]; // all positive
        let model = ObjectiveUncertaintyModel::Budget {
            delta: array![0.5, 0.3, 0.2],
            budget: 2.0, // only top-2 perturb
        };
        let obj = robust_objective(&c.view(), &x.view(), &model).expect("failed to create obj");
        // nominal = 3; top-2 perturbations: 0.5 + 0.3 = 0.8
        assert!((obj - 3.8).abs() < 1e-9, "budget robust obj={obj}");
    }

    #[test]
    fn test_robust_objective_dimension_error() {
        let c = array![1.0, 2.0];
        let x = array![1.0]; // wrong length
        let model = ObjectiveUncertaintyModel::Box {
            delta: array![0.1, 0.1],
        };
        assert!(robust_objective(&c.view(), &x.view(), &model).is_err());
    }
}
