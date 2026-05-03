//! Robust Optimization and Distributionally Robust Optimization
//!
//! This module provides robust optimization methods that hedge against uncertainty
//! in problem data. These algorithms are essential for risk-aware decision making
//! under uncertainty.
//!
//! # Uncertainty Models
//!
//! - **Box uncertainty**: Each parameter perturbed independently within ±δ
//! - **Ellipsoidal uncertainty**: Parameters perturbed inside an ellipsoid defined by a covariance matrix
//! - **Polyhedral uncertainty**: Parameters perturbed within a polytope (Aξ ≤ b)
//! - **Budgeted uncertainty**: Bertsimas-Sim model with total budget Γ
//!
//! # Algorithms
//!
//! - [`box_robust`]: Worst-case evaluation over box uncertainty set
//! - [`ellipsoidal_robust`]: Worst-case evaluation over ellipsoidal uncertainty set
//! - [`distributionally_robust_cvar`]: CVaR-based distributionally robust objective
//! - [`saa_solve`]: Sample Average Approximation (Kleywegt-Shapiro-Homem-de-Mello)
//!
//! # Sub-modules
//!
//! - [`minimax`]: Minimax optimization (subgradient, bundle, Nesterov smoothing, fictitious play)
//! - [`robust_lp`]: Robust linear programming (box, ellipsoidal, budgeted uncertainty)
//! - [`worst_case`]: Worst-case analysis, AARC, scenario approach, Wasserstein DRO
//!
//! # References
//!
//! - Ben-Tal, A., El Ghaoui, L., & Nemirovski, A. (2009). *Robust Optimization*.
//! - Bertsimas, D. & Sim, M. (2004). "The price of robustness". *Operations Research*.
//! - Shapiro, A., Dentcheva, D., & Ruszczyński, A. (2014). *Lectures on Stochastic Programming*.

pub mod minimax;
pub mod robust_lp;
pub mod worst_case;

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// High-level configuration for robust optimization.
///
/// Wraps the uncertainty set type and robustness parameter in a single
/// convenient struct for use with high-level robust optimization APIs.
#[derive(Debug, Clone)]
pub struct RobustConfig {
    /// Type of uncertainty set.
    pub uncertainty_type: UncertaintyType,
    /// Robustness parameter ρ: controls the size / conservativeness of the uncertainty set.
    /// Interpretation depends on `uncertainty_type`:
    /// - Box: ρ is the uniform box radius (all δ_i = ρ)
    /// - Ellipsoidal: ρ is the ellipsoid radius
    /// - Budgeted: ρ is the budget Γ
    pub robustness_parameter: f64,
    /// Whether to use CVaR (true) or worst-case (false) as the robustness criterion.
    pub use_cvar: bool,
    /// CVaR confidence level α ∈ (0,1) (used when `use_cvar = true`).
    pub cvar_alpha: f64,
    /// Number of inner samples for sampling-based solvers.
    pub n_inner_samples: usize,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            uncertainty_type: UncertaintyType::Box,
            robustness_parameter: 0.1,
            use_cvar: false,
            cvar_alpha: 0.95,
            n_inner_samples: 200,
        }
    }
}

/// Type of uncertainty set used in robust optimization.
#[derive(Debug, Clone, PartialEq)]
pub enum UncertaintyType {
    /// Axis-aligned box: each parameter can vary independently within ±ρ.
    Box,
    /// Ellipsoidal: parameters vary inside an ellipsoid with radius ρ.
    Ellipsoidal,
    /// Polyhedral: parameters vary inside a polytope (custom A, b).
    Polyhedral,
    /// Bertsimas-Sim budgeted: at most ρ (= Γ) parameters deviate simultaneously.
    Budgeted,
}

/// Describes the geometry of an uncertainty set for robust optimization.
#[derive(Debug, Clone)]
pub enum UncertaintySet {
    /// Axis-aligned box: ‖ξ‖∞ ≤ δ, i.e. |ξ_i| ≤ delta_i for each component.
    Box {
        /// Per-component radius (length must equal problem dimension).
        delta: Array1<f64>,
    },
    /// Ellipsoidal set: {ξ : ξᵀ Σ⁻¹ ξ ≤ ρ²} where Σ is a positive-definite covariance matrix.
    Ellipsoidal {
        /// Ellipsoid centre (shift from nominal).
        center: Array1<f64>,
        /// Covariance / shape matrix Σ (n×n positive definite).
        covariance: Array2<f64>,
        /// Ellipsoid radius ρ.
        radius: f64,
    },
    /// Polyhedral set: {ξ : A ξ ≤ b}.
    Polyhedral {
        /// Constraint matrix A (m×n).
        a_matrix: Array2<f64>,
        /// Right-hand side b (m-vector).
        b_vector: Array1<f64>,
    },
    /// Bertsimas-Sim budgeted uncertainty:
    /// at most Γ of the n parameters can deviate by their full radius δ_i.
    BudgetedUncertainty {
        /// Per-component radius.
        delta: Array1<f64>,
        /// Budget parameter Γ ∈ [0, n].
        budget: f64,
    },
}

/// Configuration for a robust optimization problem.
///
/// Wraps a nominal objective together with an uncertainty set; the robust problem
/// seeks the solution that minimises the *worst-case* objective over the uncertainty set.
#[derive(Debug, Clone)]
pub struct RobustProblem {
    /// Description of the uncertainty set.
    pub uncertainty_set: UncertaintySet,
    /// Number of auxiliary inner maximisation samples used by sampling-based solvers.
    pub n_inner_samples: usize,
    /// Absolute tolerance for the inner maximisation.
    pub inner_tol: f64,
}

impl Default for RobustProblem {
    fn default() -> Self {
        Self {
            uncertainty_set: UncertaintySet::Box {
                delta: Array1::zeros(0),
            },
            n_inner_samples: 200,
            inner_tol: 1e-8,
        }
    }
}

/// Evaluate the worst-case objective over an *axis-aligned box* uncertainty set.
///
/// For each dimension i, the perturbed parameter x̃_i is chosen from {x_i - δ_i, x_i + δ_i}
/// to maximise f.  The full worst-case is approximated by multi-start local search over the
/// 2n vertex candidates plus random interior samples.
///
/// # Arguments
///
/// * `f`     – objective function (lower is better; we find the *maximum*)
/// * `x`     – nominal parameter vector (length n)
/// * `delta` – per-component box radius (length n)
///
/// # Returns
///
/// The worst-case value max_{ξ : |ξ_i| ≤ δ_i} f(x + ξ).
pub fn box_robust<F>(f: &F, x: &ArrayView1<f64>, delta: &ArrayView1<f64>) -> OptimizeResult<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    if delta.len() != n {
        return Err(OptimizeError::ValueError(format!(
            "delta length {} does not match x length {}",
            delta.len(),
            n
        )));
    }

    // Check all deltas non-negative
    for i in 0..n {
        if delta[i] < 0.0 {
            return Err(OptimizeError::ValueError(format!(
                "delta[{}] = {} is negative",
                i, delta[i]
            )));
        }
    }

    // Evaluate at all 2n vertices (corner-optimality of linear functions guarantees
    // the worst case is attained at a vertex for linear f; for general f we use this
    // as a strong heuristic starting set plus random interior samples).
    let mut worst = f64::NEG_INFINITY;

    // --- vertex evaluation ------------------------------------------------
    let mut perturbed = x.to_owned();
    for i in 0..n {
        // +δ_i direction
        perturbed[i] = x[i] + delta[i];
        let val_pos = f(&perturbed.view());
        if val_pos > worst {
            worst = val_pos;
        }
        // -δ_i direction
        perturbed[i] = x[i] - delta[i];
        let val_neg = f(&perturbed.view());
        if val_neg > worst {
            worst = val_neg;
        }
        // restore
        perturbed[i] = x[i];
    }

    // --- random interior samples ------------------------------------------
    // Use a deterministic quasi-random sequence (uniform grid per dimension) to
    // avoid external RNG state dependencies while still covering the interior.
    let n_grid: usize = 5.min(100 / n.max(1));
    let steps = if n_grid < 2 { 1.0 } else { n_grid as f64 - 1.0 };
    let mut buf = x.to_owned();
    // Iterate over a coarse grid (product grid capped to n_grid points per dim)
    let total = n_grid.pow(n.min(8) as u32); // cap at dim 8 to avoid explosion
    for sample_idx in 0..total.min(200) {
        let mut idx = sample_idx;
        for i in 0..n {
            let dim_idx = idx % n_grid;
            idx /= n_grid;
            let t = if n_grid <= 1 {
                0.0
            } else {
                (dim_idx as f64 / steps) * 2.0 - 1.0 // in [-1, 1]
            };
            buf[i] = x[i] + t * delta[i];
        }
        let val = f(&buf.view());
        if val > worst {
            worst = val;
        }
    }

    Ok(worst)
}

/// Evaluate the worst-case objective over an *ellipsoidal* uncertainty set.
///
/// The uncertainty set is E = {x + Σ^{1/2} ξ : ‖ξ‖₂ ≤ ρ}.
/// For general (non-linear) f, the worst-case direction is approximated by
/// gradient-based power iteration followed by projected gradient ascent.
///
/// # Arguments
///
/// * `f`          – objective function
/// * `x`          – nominal parameter vector (n-vector)
/// * `center`     – shift of ellipsoid centre relative to x (n-vector; often zero)
/// * `covariance` – positive-definite shape matrix Σ (n×n)
/// * `radius`     – ellipsoid radius ρ
///
/// # Returns
///
/// Worst-case value max_{ξ : ξᵀ Σ⁻¹ ξ ≤ ρ²} f(x + center + ξ).
pub fn ellipsoidal_robust<F>(
    f: &F,
    x: &ArrayView1<f64>,
    center: &ArrayView1<f64>,
    covariance: &Array2<f64>,
    radius: f64,
) -> OptimizeResult<f64>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    if center.len() != n {
        return Err(OptimizeError::ValueError(format!(
            "center length {} != x length {}",
            center.len(),
            n
        )));
    }
    if covariance.shape() != [n, n] {
        return Err(OptimizeError::ValueError(format!(
            "covariance shape {:?} != [{n}, {n}]",
            covariance.shape()
        )));
    }
    if radius < 0.0 {
        return Err(OptimizeError::ValueError(
            "radius must be non-negative".to_string(),
        ));
    }

    // Nominal shifted point
    let x_shifted: Array1<f64> = x
        .iter()
        .zip(center.iter())
        .map(|(&xi, &ci)| xi + ci)
        .collect();

    // Cholesky factor L such that Σ = L Lᵀ  (approximate via diagonal if needed)
    let chol = cholesky_lower(covariance)?;

    // Projected gradient ascent on ξ ∈ {‖ξ‖₂ ≤ ρ} (in the whitened space η = L⁻¹ ξ)
    // f(x + center + L η) subject to ‖η‖₂ ≤ ρ
    let h = 1e-5; // finite-difference step
    let step_size = 0.1 * radius;
    let max_iter = 200;

    // Start from multiple candidate directions
    let mut best_val = f(&x_shifted.view());

    let start_dirs: Vec<Array1<f64>> = {
        let mut dirs = Vec::with_capacity(2 * n + 1);
        // zero perturbation (nominal)
        dirs.push(Array1::zeros(n));
        // ± unit vectors in η space
        for i in 0..n {
            let mut v = Array1::zeros(n);
            v[i] = radius;
            dirs.push(v.clone());
            v[i] = -radius;
            dirs.push(v);
        }
        dirs
    };

    for init_eta in start_dirs {
        let mut eta = project_onto_ball(&init_eta.view(), radius);

        for _ in 0..max_iter {
            // Compute ξ = L η
            let xi = mat_vec_lower(&chol, &eta.view());
            // Compute perturbed point
            let x_pert: Array1<f64> = x_shifted
                .iter()
                .zip(xi.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            // Finite-difference gradient w.r.t. η
            let mut grad_eta = Array1::<f64>::zeros(n);
            for j in 0..n {
                let mut eta_fwd = eta.clone();
                eta_fwd[j] += h;
                let xi_fwd = mat_vec_lower(&chol, &eta_fwd.view());
                let x_fwd: Array1<f64> = x_shifted
                    .iter()
                    .zip(xi_fwd.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                grad_eta[j] = (f(&x_fwd.view()) - f(&x_pert.view())) / h;
            }

            // Gradient ascent step + projection
            let eta_new: Array1<f64> = eta
                .iter()
                .zip(grad_eta.iter())
                .map(|(&e, &g)| e + step_size * g)
                .collect();
            eta = project_onto_ball(&eta_new.view(), radius);

            let xi_new = mat_vec_lower(&chol, &eta.view());
            let x_new: Array1<f64> = x_shifted
                .iter()
                .zip(xi_new.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let val = f(&x_new.view());
            if val > best_val {
                best_val = val;
            }
        }
    }

    Ok(best_val)
}

/// Distributionally robust CVaR (Conditional Value-at-Risk) objective.
///
/// Computes the empirical CVaR at level α over a set of discrete scenarios,
/// implementing the Rockafellar-Uryasev linear programming formula:
///
/// CVaR_α(f(x, ξ)) = min_{t} { t + (1/(1-α)) E[max(f(x,ξ) - t, 0)] }
///
/// The minimisation over t is performed by exact line search over the sorted
/// scenario losses (the optimal t is always a scenario value).
///
/// # Arguments
///
/// * `f`         – scenario loss function: (x, scenario) → loss value
/// * `x`         – decision variable
/// * `scenarios` – slice of scenario parameter vectors (each has length equal to scenario_dim)
/// * `alpha`     – CVaR confidence level ∈ (0, 1) (typical values: 0.9, 0.95, 0.99)
///
/// # Returns
///
/// The CVaR_α value at x.
pub fn distributionally_robust_cvar<F>(
    f: &F,
    x: &ArrayView1<f64>,
    scenarios: &[Array1<f64>],
    alpha: f64,
) -> OptimizeResult<f64>
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    if scenarios.is_empty() {
        return Err(OptimizeError::ValueError(
            "scenarios must be non-empty".to_string(),
        ));
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(OptimizeError::ValueError(format!(
            "alpha must be in (0,1), got {}",
            alpha
        )));
    }

    // Evaluate all scenario losses
    let mut losses: Vec<f64> = scenarios.iter().map(|s| f(x, &s.view())).collect();

    losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = losses.len();

    // Rockafellar-Uryasev: CVaR_α = min_t { t + 1/(1-α) * mean(max(loss - t, 0)) }
    // The minimum is attained at t = VaR_α (the α-quantile).
    // We evaluate at all n candidate t values (each scenario loss) and pick the best.
    let scale = 1.0 / ((1.0 - alpha) * n as f64);

    let best_cvar = losses
        .iter()
        .map(|&t| {
            let excess: f64 = losses.iter().map(|&l| (l - t).max(0.0)).sum();
            t + scale * excess
        })
        .fold(f64::INFINITY, f64::min);

    Ok(best_cvar)
}

/// Result of a Sample Average Approximation solve.
#[derive(Debug, Clone)]
pub struct SaaResult {
    /// Approximate optimal solution.
    pub x: Array1<f64>,
    /// Optimal SAA objective value (average loss over samples).
    pub fun: f64,
    /// Number of SAA iterations performed.
    pub n_iter: usize,
    /// Whether the SAA problem converged.
    pub success: bool,
    /// Status message.
    pub message: String,
}

/// Options for the SAA solver.
#[derive(Debug, Clone)]
pub struct SaaConfig {
    /// Total number of Monte-Carlo samples.
    pub n_samples: usize,
    /// Maximum SAA outer iterations (re-sampling rounds).
    pub max_outer_iter: usize,
    /// Inner optimisation tolerance.
    pub tol: f64,
    /// Inner optimisation maximum iterations.
    pub inner_max_iter: usize,
    /// Step size for gradient descent inner solve.
    pub step_size: f64,
}

impl Default for SaaConfig {
    fn default() -> Self {
        Self {
            n_samples: 500,
            max_outer_iter: 10,
            tol: 1e-6,
            inner_max_iter: 500,
            step_size: 1e-3,
        }
    }
}

/// Solve a stochastic program via Sample Average Approximation (SAA).
///
/// The stochastic program is:
///   min_x E_{ξ}[f(x, ξ)]
///
/// SAA replaces the expectation with a sample average:
///   min_x (1/N) Σ_i f(x, ξ_i)
///
/// The inner deterministic problem is solved with projected gradient descent
/// using finite-difference gradients.
///
/// # Arguments
///
/// * `f`                – per-sample loss: (x, sample) → f64
/// * `sample_generator` – draws one Monte-Carlo sample ξ ~ P (called n_samples times)
/// * `x0`               – starting point
/// * `config`           – SAA configuration
///
/// # Returns
///
/// [`SaaResult`] containing the approximate minimiser.
pub fn saa_solve<F, G>(
    f: &F,
    sample_generator: &mut G,
    x0: &ArrayView1<f64>,
    config: &SaaConfig,
) -> OptimizeResult<SaaResult>
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    G: FnMut() -> Array1<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }

    let mut x = x0.to_owned();
    let h = 1e-5; // finite-difference step
    let mut converged = false;
    let mut outer_iter = 0;

    for outer in 0..config.max_outer_iter {
        outer_iter = outer + 1;

        // Draw fresh samples
        let samples: Vec<Array1<f64>> = (0..config.n_samples).map(|_| sample_generator()).collect();

        // Inner gradient descent on SAA objective
        for _ in 0..config.inner_max_iter {
            // Compute SAA gradient via finite differences
            let f_x: f64 = samples.iter().map(|s| f(&x.view(), &s.view())).sum::<f64>()
                / config.n_samples as f64;

            let mut grad = Array1::<f64>::zeros(n);
            for j in 0..n {
                let mut x_fwd = x.clone();
                x_fwd[j] += h;
                let f_fwd: f64 = samples
                    .iter()
                    .map(|s| f(&x_fwd.view(), &s.view()))
                    .sum::<f64>()
                    / config.n_samples as f64;
                grad[j] = (f_fwd - f_x) / h;
            }

            let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < config.tol {
                converged = true;
                break;
            }

            // Gradient descent step
            for j in 0..n {
                x[j] -= config.step_size * grad[j];
            }
        }

        if converged {
            break;
        }
    }

    // Final objective value
    let final_samples: Vec<Array1<f64>> = (0..config.n_samples.min(100))
        .map(|_| sample_generator())
        .collect();
    let fun = if final_samples.is_empty() {
        0.0
    } else {
        final_samples
            .iter()
            .map(|s| f(&x.view(), &s.view()))
            .sum::<f64>()
            / final_samples.len() as f64
    };

    Ok(SaaResult {
        x,
        fun,
        n_iter: outer_iter,
        success: converged,
        message: if converged {
            "SAA converged".to_string()
        } else {
            "SAA reached maximum outer iterations".to_string()
        },
    })
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Project a vector onto the L2-ball of given radius.
fn project_onto_ball(v: &ArrayView1<f64>, radius: f64) -> Array1<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm <= radius || norm == 0.0 {
        v.to_owned()
    } else {
        v.mapv(|x| x * radius / norm)
    }
}

/// Lower-triangular matrix–vector product: y = L x.
fn mat_vec_lower(l: &Array2<f64>, x: &ArrayView1<f64>) -> Array1<f64> {
    let n = x.len();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in 0..=i {
            y[i] += l[[i, j]] * x[j];
        }
    }
    y
}

/// Incomplete (lower-triangular) Cholesky of a symmetric positive-definite matrix.
/// Returns the lower-triangular factor L such that A ≈ L Lᵀ.
/// Falls back to the diagonal square root if the matrix is not positive definite.
fn cholesky_lower(a: &Array2<f64>) -> OptimizeResult<Array2<f64>> {
    let n = a.shape()[0];
    let mut l = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diag = a[[i, i]] - sum;
                if diag < 0.0 {
                    // Fall back: use absolute value (matrix is near-singular)
                    l[[i, j]] = diag.abs().sqrt().max(1e-12);
                } else {
                    l[[i, j]] = diag.sqrt();
                }
            } else {
                let ljj = l[[j, j]];
                if ljj.abs() < 1e-14 {
                    l[[i, j]] = 0.0;
                } else {
                    l[[i, j]] = (a[[i, j]] - sum) / ljj;
                }
            }
        }
    }
    Ok(l)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    fn quadratic(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    #[test]
    fn test_box_robust_basic() {
        // f(x) = x₀² + x₁²; worst case over box |ξ| ≤ 1 around (0,0) should be 2.0
        let x = array![0.0, 0.0];
        let delta = array![1.0, 1.0];
        let val = box_robust(&quadratic, &x.view(), &delta.view()).expect("failed to create val");
        assert!((val - 2.0).abs() < 1e-9, "expected 2.0, got {val}");
    }

    #[test]
    fn test_box_robust_shifted() {
        // f(x) = x²; worst case over box |ξ| ≤ 0.5 around x=1 should be (1.5)²=2.25
        let x = array![1.0];
        let delta = array![0.5];
        let val = box_robust(&|v: &ArrayView1<f64>| v[0] * v[0], &x.view(), &delta.view())
            .expect("unexpected None or Err");
        assert!((val - 2.25).abs() < 1e-9, "expected 2.25, got {val}");
    }

    #[test]
    fn test_box_robust_bad_delta() {
        let x = array![1.0];
        let delta = array![-0.1];
        assert!(box_robust(&quadratic, &x.view(), &delta.view()).is_err());
    }

    #[test]
    fn test_ellipsoidal_robust_identity() {
        // Ellipsoidal with Σ = I, ρ=1: worst case of f(x)=‖x‖² around x=0 is ρ²=1
        let x = array![0.0, 0.0];
        let center = array![0.0, 0.0];
        let cov = Array2::<f64>::eye(2);
        let val = ellipsoidal_robust(&quadratic, &x.view(), &center.view(), &cov, 1.0)
            .expect("unexpected None or Err");
        // worst case: move distance 1 in any direction → ‖x+ξ‖²= 1
        assert!((val - 1.0).abs() < 0.05, "expected ~1.0, got {val}");
    }

    #[test]
    fn test_cvar_basic() {
        // 5 scenarios with losses [0,1,2,3,4]; CVaR_{0.8} = mean of top 20% = 4.0
        let x = array![0.0];
        let scenarios: Vec<Array1<f64>> = (0..5).map(|i| array![i as f64]).collect();
        let f = |_x: &ArrayView1<f64>, s: &ArrayView1<f64>| s[0];
        let cvar = distributionally_robust_cvar(&f, &x.view(), &scenarios, 0.8)
            .expect("failed to create cvar");
        assert!((cvar - 4.0).abs() < 1e-9, "expected 4.0, got {cvar}");
    }

    #[test]
    fn test_cvar_alpha_error() {
        let x = array![0.0];
        let scenarios = vec![array![1.0]];
        let f = |_: &ArrayView1<f64>, s: &ArrayView1<f64>| s[0];
        assert!(distributionally_robust_cvar(&f, &x.view(), &scenarios, 0.0).is_err());
        assert!(distributionally_robust_cvar(&f, &x.view(), &scenarios, 1.0).is_err());
    }

    #[test]
    fn test_saa_quadratic() {
        // min_x E[(x - ξ)²] where ξ ~ Uniform[0, 2]; optimum at x* = E[ξ] = 1
        let f = |x: &ArrayView1<f64>, xi: &ArrayView1<f64>| {
            let diff = x[0] - xi[0];
            diff * diff
        };
        let mut rng_state = 42u64;
        let mut sample_gen = || {
            // Simple LCG pseudo-random in [0, 2]
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Take upper 32 bits to get a full u32-range value, then normalize to [0, 2]
            let t = ((rng_state >> 32) as u32 as f64) / (u32::MAX as f64) * 2.0;
            array![t]
        };
        let x0 = array![0.0];
        let config = SaaConfig {
            n_samples: 200,
            max_outer_iter: 5,
            tol: 1e-4,
            inner_max_iter: 200,
            step_size: 5e-3,
        };
        let result =
            saa_solve(&f, &mut sample_gen, &x0.view(), &config).expect("failed to create result");
        assert!(
            (result.x[0] - 1.0).abs() < 0.15,
            "expected x* ≈ 1.0, got {}",
            result.x[0]
        );
    }
}
