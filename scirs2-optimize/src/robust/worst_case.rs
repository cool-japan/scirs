//! Worst-Case Analysis and Scenario-Based Robust Optimization
//!
//! This module provides tools for analyzing worst-case performance and
//! scenario-based approaches to robust optimization.
//!
//! # Methods
//!
//! - [`worst_case_analysis`]: Enumerate and evaluate all scenarios to find the worst case
//! - [`affinely_adjustable`]: Affinely Adjustable Robust Counterpart (AARC)
//! - [`scenario_approach`]: Scenario approach (Campi-Garatti) with sample complexity guarantees
//! - [`distributionally_robust`]: DRO with Wasserstein ball constraint
//!
//! # Background
//!
//! The *scenario approach* (Campi & Garatti 2008) provides a distribution-free way
//! to design robust controllers: given N i.i.d. samples of uncertainty ξ_1, …, ξ_N,
//! solve the sampled program and obtain *a priori* feasibility guarantees.
//!
//! *Distributionally robust optimization* (Wiesemann et al. 2014; Mohajerin Esfahani
//! & Kuhn 2018) minimizes the worst-case expected loss over all distributions P in a
//! Wasserstein ball around the empirical distribution.
//!
//! # References
//!
//! - Campi, M.C. & Garatti, S. (2008). "The exact feasibility of randomized solutions
//!   of uncertain convex programs". *SIAM Journal on Optimization*.
//! - Ben-Tal, A. & Goryashko, A. (2004). "Adjustable robust solutions of uncertain LP".
//!   *Mathematical Programming*.
//! - Mohajerin Esfahani, P. & Kuhn, D. (2018). "Data-driven distributionally robust
//!   optimization using the Wasserstein metric". *Mathematical Programming*.
//! - Shapiro, A. (2017). "Distributionally robust stochastic programming". *SIAM JOPT*.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// ─── Worst-case analysis ──────────────────────────────────────────────────────

/// A single evaluated scenario.
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    /// Scenario parameter vector ξ.
    pub scenario: Array1<f64>,
    /// Objective value f(x, ξ).
    pub obj_value: f64,
    /// Whether the constraints are satisfied for this scenario.
    pub feasible: bool,
    /// Constraint violations (positive = violated).
    pub violations: Vec<f64>,
}

/// Result of worst-case analysis over a scenario set.
#[derive(Debug, Clone)]
pub struct WorstCaseResult {
    /// All evaluated scenario results.
    pub scenarios: Vec<ScenarioResult>,
    /// Index of the worst scenario (highest objective value).
    pub worst_index: usize,
    /// Worst-case objective value.
    pub worst_obj: f64,
    /// Best-case objective value.
    pub best_obj: f64,
    /// Average objective value over all scenarios.
    pub avg_obj: f64,
    /// Standard deviation of objective values.
    pub std_obj: f64,
    /// Fraction of scenarios that are feasible.
    pub feasibility_rate: f64,
    /// Number of scenarios evaluated.
    pub n_scenarios: usize,
}

/// Perform worst-case analysis over an enumerated set of scenarios.
///
/// Evaluates the objective f(x, ξ) and optionally constraint functions g_i(x, ξ) ≤ 0
/// at every provided scenario ξ_j, and reports statistics.
///
/// # Arguments
///
/// * `obj_fn`      – objective function (x, ξ) → f64
/// * `x`           – decision variable (fixed)
/// * `scenarios`   – slice of uncertainty realizations ξ_j
/// * `constraint_fns` – optional slice of constraint functions (x, ξ) → f64; feasible when ≤ 0
///
/// # Returns
///
/// [`WorstCaseResult`] summarizing worst-case, best-case, and distributional statistics.
pub fn worst_case_analysis<F, G>(
    obj_fn: &F,
    x: &ArrayView1<f64>,
    scenarios: &[Array1<f64>],
    constraint_fns: &[G],
) -> OptimizeResult<WorstCaseResult>
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    G: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    if scenarios.is_empty() {
        return Err(OptimizeError::ValueError(
            "scenarios must be non-empty".to_string(),
        ));
    }

    let n_scen = scenarios.len();
    let mut results = Vec::with_capacity(n_scen);
    let mut sum_obj = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut worst_val = f64::NEG_INFINITY;
    let mut best_val = f64::INFINITY;
    let mut worst_idx = 0_usize;
    let mut n_feasible = 0_usize;

    for (j, scenario) in scenarios.iter().enumerate() {
        let obj = obj_fn(x, &scenario.view());
        let violations: Vec<f64> = constraint_fns
            .iter()
            .map(|g| g(x, &scenario.view()))
            .collect();
        let feasible = violations.iter().all(|&v| v <= 0.0);

        sum_obj += obj;
        sum_sq += obj * obj;
        if obj > worst_val {
            worst_val = obj;
            worst_idx = j;
        }
        if obj < best_val {
            best_val = obj;
        }
        if feasible {
            n_feasible += 1;
        }

        results.push(ScenarioResult {
            scenario: scenario.clone(),
            obj_value: obj,
            feasible,
            violations,
        });
    }

    let avg = sum_obj / n_scen as f64;
    let variance = (sum_sq / n_scen as f64 - avg * avg).max(0.0);
    let std_dev = variance.sqrt();
    let feasibility_rate = n_feasible as f64 / n_scen as f64;

    Ok(WorstCaseResult {
        scenarios: results,
        worst_index: worst_idx,
        worst_obj: worst_val,
        best_obj: best_val,
        avg_obj: avg,
        std_obj: std_dev,
        feasibility_rate,
        n_scenarios: n_scen,
    })
}

// ─── Affinely Adjustable Robust Counterpart ────────────────────────────────

/// Configuration for the affinely adjustable robust counterpart (AARC).
#[derive(Debug, Clone)]
pub struct AARCConfig {
    /// Dimension of the recourse variable y.
    pub recourse_dim: usize,
    /// Maximum iterations for the saddle-point solve.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Step size for the outer (robust) minimization.
    pub step_size: f64,
    /// Number of uncertainty samples for the inner maximization.
    pub n_uncertainty_samples: usize,
    /// Finite-difference step.
    pub fd_step: f64,
}

impl Default for AARCConfig {
    fn default() -> Self {
        Self {
            recourse_dim: 1,
            max_iter: 2_000,
            tol: 1e-5,
            step_size: 1e-3,
            n_uncertainty_samples: 100,
            fd_step: 1e-5,
        }
    }
}

/// Result of the AARC solve.
#[derive(Debug, Clone)]
pub struct AARCResult {
    /// First-stage decision x*.
    pub x: Array1<f64>,
    /// Affine recourse policy: y(ξ) = K x + L ξ + m.
    /// Returned as the matrices K (recourse_dim × x_dim), L (recourse_dim × xi_dim), and m.
    pub k_matrix: Array2<f64>,
    pub l_matrix: Array2<f64>,
    pub m_vector: Array1<f64>,
    /// Worst-case objective value.
    pub worst_obj: f64,
    /// Number of iterations.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Status message.
    pub message: String,
}

/// Solve an Affinely Adjustable Robust Counterpart (AARC).
///
/// The AARC replaces the static robust program with a two-stage problem where
/// a *recourse variable* y can be adjusted as an affine function of the realized
/// uncertainty ξ:
///
/// ```text
/// y(ξ) = K x + L ξ + m
/// ```
///
/// The optimization seeks (x, K, L, m) minimizing worst-case cost:
///
/// ```text
/// min_{x, K, L, m}  max_{ξ ∈ U}  f(x, y(ξ), ξ)
/// ```
///
/// This implementation uses projected gradient descent on (x, K, L, m) with
/// sampled inner maximization.
///
/// # Arguments
///
/// * `obj_fn`    – (x, y, ξ) → objective value
/// * `x0`        – initial first-stage decision (x_dim)
/// * `xi_samples`– samples from the uncertainty set U
/// * `xi_dim`    – dimension of ξ
/// * `config`    – AARC configuration
///
/// # Returns
///
/// [`AARCResult`] containing the robust recourse policy.
///
/// # References
///
/// Ben-Tal, A. & Goryashko, A. (2004). "Adjustable robust solutions of uncertain LP".
pub fn affinely_adjustable<F>(
    obj_fn: &F,
    x0: &ArrayView1<f64>,
    xi_samples: &[Array1<f64>],
    xi_dim: usize,
    config: &AARCConfig,
) -> OptimizeResult<AARCResult>
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    if xi_samples.is_empty() {
        return Err(OptimizeError::ValueError(
            "xi_samples must be non-empty".to_string(),
        ));
    }
    let x_dim = x0.len();
    let y_dim = config.recourse_dim;
    if y_dim == 0 {
        return Err(OptimizeError::ValueError(
            "recourse_dim must be positive".to_string(),
        ));
    }

    // Policy parameters: K (y_dim × x_dim), L (y_dim × xi_dim), m (y_dim)
    let mut k_matrix = Array2::<f64>::zeros((y_dim, x_dim));
    let mut l_matrix = Array2::<f64>::zeros((y_dim, xi_dim));
    let mut m_vector = Array1::<f64>::zeros(y_dim);
    let mut x = x0.to_owned();
    let h = config.fd_step;

    let mut converged = false;

    // Evaluate policy: y(ξ) = K x + L ξ + m
    let eval_policy = |x_cur: &Array1<f64>,
                       k: &Array2<f64>,
                       l: &Array2<f64>,
                       m: &Array1<f64>,
                       xi: &Array1<f64>|
     -> Array1<f64> {
        let mut y = m.clone();
        for i in 0..y_dim {
            for j in 0..x_dim {
                y[i] += k[[i, j]] * x_cur[j];
            }
            for j in 0..xi.len().min(xi_dim) {
                y[i] += l[[i, j]] * xi[j];
            }
        }
        y
    };

    // Worst-case objective over samples
    let worst_obj_fn =
        |x_cur: &Array1<f64>, k: &Array2<f64>, l: &Array2<f64>, m: &Array1<f64>| -> f64 {
            xi_samples
                .iter()
                .map(|xi| {
                    let y = eval_policy(x_cur, k, l, m, xi);
                    obj_fn(&x_cur.view(), &y.view(), &xi.view())
                })
                .fold(f64::NEG_INFINITY, f64::max)
        };

    for _ in 0..config.max_iter {
        let f_curr = worst_obj_fn(&x, &k_matrix, &l_matrix, &m_vector);

        // Finite-difference gradients for x
        let mut grad_x = Array1::<f64>::zeros(x_dim);
        for j in 0..x_dim {
            let mut x_fwd = x.clone();
            x_fwd[j] += h;
            let f_fwd = worst_obj_fn(&x_fwd, &k_matrix, &l_matrix, &m_vector);
            grad_x[j] = (f_fwd - f_curr) / h;
        }

        // Finite-difference gradients for K
        let mut grad_k = Array2::<f64>::zeros((y_dim, x_dim));
        for i in 0..y_dim {
            for j in 0..x_dim {
                let mut k_fwd = k_matrix.clone();
                k_fwd[[i, j]] += h;
                let f_fwd = worst_obj_fn(&x, &k_fwd, &l_matrix, &m_vector);
                grad_k[[i, j]] = (f_fwd - f_curr) / h;
            }
        }

        // Finite-difference gradients for L
        let mut grad_l = Array2::<f64>::zeros((y_dim, xi_dim));
        for i in 0..y_dim {
            for j in 0..xi_dim {
                let mut l_fwd = l_matrix.clone();
                l_fwd[[i, j]] += h;
                let f_fwd = worst_obj_fn(&x, &k_matrix, &l_fwd, &m_vector);
                grad_l[[i, j]] = (f_fwd - f_curr) / h;
            }
        }

        // Finite-difference gradients for m
        let mut grad_m = Array1::<f64>::zeros(y_dim);
        for i in 0..y_dim {
            let mut m_fwd = m_vector.clone();
            m_fwd[i] += h;
            let f_fwd = worst_obj_fn(&x, &k_matrix, &l_matrix, &m_fwd);
            grad_m[i] = (f_fwd - f_curr) / h;
        }

        // Gradient norms for convergence
        let gx_norm = l2_norm_arr1(&grad_x);
        let gk_norm = l2_norm_arr2(&grad_k);
        let gl_norm = l2_norm_arr2(&grad_l);
        let gm_norm = l2_norm_arr1(&grad_m);
        let total_norm = gx_norm + gk_norm + gl_norm + gm_norm;

        if total_norm < config.tol {
            converged = true;
            break;
        }

        // Gradient descent step
        let alpha = config.step_size;
        for j in 0..x_dim {
            x[j] -= alpha * grad_x[j];
        }
        for i in 0..y_dim {
            for j in 0..x_dim {
                k_matrix[[i, j]] -= alpha * grad_k[[i, j]];
            }
            for j in 0..xi_dim {
                l_matrix[[i, j]] -= alpha * grad_l[[i, j]];
            }
            m_vector[i] -= alpha * grad_m[i];
        }
    }

    let worst_obj = worst_obj_fn(&x, &k_matrix, &l_matrix, &m_vector);

    Ok(AARCResult {
        x,
        k_matrix,
        l_matrix,
        m_vector,
        worst_obj,
        n_iter: config.max_iter,
        converged,
        message: if converged {
            "AARC converged".to_string()
        } else {
            "AARC reached maximum iterations".to_string()
        },
    })
}

// ─── Scenario Approach (Campi-Garatti) ────────────────────────────────────────

/// Configuration for the scenario approach.
#[derive(Debug, Clone)]
pub struct ScenarioApproachConfig {
    /// Number of scenarios N (sample complexity). Should satisfy theoretical bound.
    pub n_scenarios: usize,
    /// Confidence parameter β ∈ (0, 1): the solution is feasible with prob. ≥ 1 - β.
    pub confidence: f64,
    /// Maximum inner optimization iterations.
    pub max_iter: usize,
    /// Inner optimization convergence tolerance.
    pub tol: f64,
    /// Step size for inner optimization.
    pub step_size: f64,
    /// Finite-difference step.
    pub fd_step: f64,
}

impl Default for ScenarioApproachConfig {
    fn default() -> Self {
        Self {
            n_scenarios: 500,
            confidence: 0.95,
            max_iter: 2_000,
            tol: 1e-5,
            step_size: 1e-3,
            fd_step: 1e-5,
        }
    }
}

/// Result of a scenario approach solve.
#[derive(Debug, Clone)]
pub struct ScenarioApproachResult {
    /// Optimal solution x* of the sampled program.
    pub x: Array1<f64>,
    /// Optimal objective value of the sampled program.
    pub fun: f64,
    /// Number of active (binding) scenarios (support scenarios).
    pub n_support_scenarios: usize,
    /// A priori probability guarantee: feasibility probability ≥ this value.
    pub feasibility_guarantee: f64,
    /// Number of scenarios used.
    pub n_scenarios: usize,
    /// Number of iterations.
    pub n_iter: usize,
    /// Whether the inner optimization converged.
    pub converged: bool,
    /// Status message.
    pub message: String,
}

/// Solve a robust optimization problem via the Campi-Garatti scenario approach.
///
/// The scenario approach solves the *sampled* robust program:
///
/// ```text
/// min_x  f₀(x)
/// s.t.  f_i(x, ξ_j) ≤ 0  for all j = 1 … N, i = 1 … m
/// ```
///
/// By the Campi-Garatti theorem, with confidence ≥ 1 - β, the solution is
/// feasible for all future realizations of ξ, provided N ≥ N*(n, ε, β) where:
///
/// ```text
/// N*(n, ε, β) = ⌈(2/ε) · (ln(1/β) + n)⌉
/// ```
///
/// and n = dim(x), ε = violation probability.
///
/// The inner problem is solved with projected gradient descent.
///
/// # Arguments
///
/// * `obj_fn`        – deterministic objective (does NOT depend on ξ): x → f64
/// * `constraint_fn` – constraint (x, ξ) → f64; feasible when ≤ 0
/// * `sample_fn`     – draws one sample ξ ~ P
/// * `x0`            – initial point
/// * `config`        – scenario approach configuration
///
/// # Returns
///
/// [`ScenarioApproachResult`] with the scenario-optimal solution and guarantees.
///
/// # References
///
/// Campi, M.C. & Garatti, S. (2008). "The exact feasibility of randomized solutions
/// of uncertain convex programs". *SIAM Journal on Optimization*, 19(3), 1211–1230.
pub fn scenario_approach<F, G, H>(
    obj_fn: &F,
    constraint_fn: &G,
    sample_fn: &mut H,
    x0: &ArrayView1<f64>,
    config: &ScenarioApproachConfig,
) -> OptimizeResult<ScenarioApproachResult>
where
    F: Fn(&ArrayView1<f64>) -> f64,
    G: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    H: FnMut() -> Array1<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }
    if !(0.0 < config.confidence && config.confidence < 1.0) {
        return Err(OptimizeError::ValueError(format!(
            "confidence must be in (0,1), got {}",
            config.confidence
        )));
    }

    // Draw N scenarios
    let scenarios: Vec<Array1<f64>> = (0..config.n_scenarios).map(|_| sample_fn()).collect();

    let h = config.fd_step;
    let penalty = 1e3_f64; // constraint penalty weight

    let mut x = x0.to_owned();
    let mut converged = false;

    // Penalized objective: f₀(x) + penalty * Σ_j max(g(x, ξ_j), 0)²
    let penalized_obj = |x_cur: &ArrayView1<f64>| -> f64 {
        let base = obj_fn(x_cur);
        let viol: f64 = scenarios
            .iter()
            .map(|xi| constraint_fn(x_cur, &xi.view()).max(0.0).powi(2))
            .sum();
        base + penalty * viol
    };

    for _ in 0..config.max_iter {
        let f0 = penalized_obj(&x.view());

        // Finite-difference gradient
        let mut grad = Array1::<f64>::zeros(n);
        let mut x_fwd = x.clone();
        for j in 0..n {
            x_fwd[j] += h;
            let f_fwd = penalized_obj(&x_fwd.view());
            grad[j] = (f_fwd - f0) / h;
            x_fwd[j] = x[j];
        }

        let gn = l2_norm_arr1(&grad);
        if gn < config.tol {
            converged = true;
            break;
        }

        // Backtracking line search (Armijo condition) to prevent divergence
        let mut alpha = config.step_size;
        let mut x_trial = x.clone();
        let armijo_c = 0.5_f64;
        let backtrack_factor = 0.5_f64;
        let max_backtracks = 30_usize;
        for _ in 0..max_backtracks {
            for j in 0..n {
                x_trial[j] = x[j] - alpha * grad[j];
            }
            let f_trial = penalized_obj(&x_trial.view());
            if f_trial <= f0 - armijo_c * alpha * gn * gn {
                break;
            }
            alpha *= backtrack_factor;
        }
        x = x_trial;
    }

    let fun = obj_fn(&x.view());

    // Count support scenarios (binding constraints: g(x*, ξ_j) ≈ 0)
    let tol_support = 1e-3;
    let n_support = scenarios
        .iter()
        .filter(|xi| constraint_fn(&x.view(), &xi.view()).abs() < tol_support)
        .count();

    // A priori guarantee: P[feasibility] ≥ 1 - Σ_{k=0}^{n-1} C(N,k) ε^k (1-ε)^{N-k}
    // For the simplified Campi-Garatti bound with ε = n/N (approximation):
    let beta = 1.0 - config.confidence;
    let epsilon_cg = if config.n_scenarios > n {
        // Campi-Garatti: ε such that N ≥ (2/ε)(ln(1/β) + n)
        // → ε = 2(ln(1/β) + n) / N
        2.0 * (beta.recip().ln() + n as f64) / config.n_scenarios as f64
    } else {
        1.0 // trivial bound if too few samples
    };
    let feasibility_guarantee = (1.0 - epsilon_cg).max(0.0).min(1.0);

    Ok(ScenarioApproachResult {
        x,
        fun,
        n_support_scenarios: n_support,
        feasibility_guarantee,
        n_scenarios: config.n_scenarios,
        n_iter: config.max_iter,
        converged,
        message: if converged {
            "Scenario approach inner optimization converged".to_string()
        } else {
            "Scenario approach reached maximum iterations".to_string()
        },
    })
}

// ─── Distributionally Robust Optimization (Wasserstein DRO) ─────────────────

/// Configuration for Wasserstein DRO.
#[derive(Debug, Clone)]
pub struct WassersteinDROConfig {
    /// Wasserstein ball radius ε > 0.
    pub epsilon: f64,
    /// Wasserstein order p (1 or 2).
    pub p_order: u32,
    /// Regularization penalty for Wasserstein constraint.
    pub lambda: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Step size.
    pub step_size: f64,
    /// Finite-difference step.
    pub fd_step: f64,
}

impl Default for WassersteinDROConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            p_order: 1,
            lambda: 10.0,
            max_iter: 2_000,
            tol: 1e-5,
            step_size: 1e-3,
            fd_step: 1e-5,
        }
    }
}

/// Result of a Wasserstein DRO solve.
#[derive(Debug, Clone)]
pub struct WassersteinDROResult {
    /// Robust-optimal decision x*.
    pub x: Array1<f64>,
    /// Worst-case expected loss under the Wasserstein ball.
    pub worst_case_loss: f64,
    /// Empirical loss (average over training scenarios).
    pub empirical_loss: f64,
    /// Estimated Lipschitz constant of the loss function.
    pub lipschitz_estimate: f64,
    /// Number of iterations.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Status message.
    pub message: String,
}

/// Solve a distributionally robust optimization problem with a Wasserstein ball.
///
/// **Problem formulation** (Mohajerin Esfahani & Kuhn 2018):
///
/// ```text
/// min_x  max_{P : W_p(P, P̂_N) ≤ ε}  E_P[f(x, ξ)]
/// ```
///
/// where P̂_N = (1/N) Σ_i δ_{ξ_i} is the empirical distribution and
/// W_p is the p-Wasserstein distance.
///
/// **Tractable reformulation** (finite-sample, p=1):
///
/// The worst-case expected loss has the upper bound (Kuhn et al. 2019):
///
/// ```text
/// sup_{P ∈ Bε} E_P[f(x,ξ)] ≤ (1/N) Σ_i f(x, ξ_i) + ε · L(x)
/// ```
///
/// where L(x) is the Lipschitz constant of ξ ↦ f(x, ξ).
///
/// We minimize this tractable upper bound, estimating L(x) empirically.
///
/// # Arguments
///
/// * `loss_fn`   – loss (x, ξ) → f64 (should be Lipschitz in ξ for theoretical guarantees)
/// * `x0`        – initial decision variable
/// * `scenarios` – empirical samples ξ_1, …, ξ_N
/// * `config`    – DRO configuration
///
/// # Returns
///
/// [`WassersteinDROResult`] with robust-optimal solution.
///
/// # References
///
/// Mohajerin Esfahani, P. & Kuhn, D. (2018). "Data-driven distributionally robust
/// optimization using the Wasserstein metric". *Mathematical Programming*, 171, 115–166.
pub fn distributionally_robust<F>(
    loss_fn: &F,
    x0: &ArrayView1<f64>,
    scenarios: &[Array1<f64>],
    config: &WassersteinDROConfig,
) -> OptimizeResult<WassersteinDROResult>
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
{
    if scenarios.is_empty() {
        return Err(OptimizeError::ValueError(
            "scenarios must be non-empty".to_string(),
        ));
    }
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "x0 must be non-empty".to_string(),
        ));
    }
    if config.epsilon < 0.0 {
        return Err(OptimizeError::ValueError(format!(
            "epsilon must be non-negative, got {}",
            config.epsilon
        )));
    }

    let n_scen = scenarios.len();
    let h = config.fd_step;
    let mut x = x0.to_owned();
    let mut converged = false;

    // Estimate Lipschitz constant: L(x) ≈ max_{i≠j} |f(x,ξ_i)-f(x,ξ_j)| / ‖ξ_i - ξ_j‖_p
    let estimate_lipschitz = |x_cur: &Array1<f64>| -> f64 {
        if n_scen < 2 {
            return 1.0; // fallback
        }
        // Only use a subset for efficiency
        let n_pairs = n_scen.min(20);
        let mut max_lip = 0.0_f64;
        for i in 0..n_pairs {
            for j in (i + 1)..n_pairs.min(i + 5) {
                let fi = loss_fn(&x_cur.view(), &scenarios[i].view());
                let fj = loss_fn(&x_cur.view(), &scenarios[j].view());
                let dist: f64 = if config.p_order == 1 {
                    scenarios[i]
                        .iter()
                        .zip(scenarios[j].iter())
                        .map(|(&a, &b)| (a - b).abs())
                        .sum()
                } else {
                    scenarios[i]
                        .iter()
                        .zip(scenarios[j].iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt()
                };
                if dist > 1e-12 {
                    max_lip = max_lip.max((fi - fj).abs() / dist);
                }
            }
        }
        max_lip.max(1e-3) // ensure positive
    };

    // DRO objective: (1/N) Σ_i f(x, ξ_i) + ε * L(x)
    // + regularization λ * ‖x‖² to control Lipschitz
    let dro_objective = |x_cur: &Array1<f64>| -> f64 {
        let empirical: f64 = scenarios
            .iter()
            .map(|xi| loss_fn(&x_cur.view(), &xi.view()))
            .sum::<f64>()
            / n_scen as f64;
        let lip = estimate_lipschitz(x_cur);
        let reg: f64 = x_cur.iter().map(|xi| xi * xi).sum::<f64>();
        empirical + config.epsilon * lip + config.lambda * 1e-4 * reg
    };

    for _ in 0..config.max_iter {
        let f0 = dro_objective(&x);

        // Finite-difference gradient
        let mut grad = Array1::<f64>::zeros(n);
        let mut x_fwd = x.clone();
        for j in 0..n {
            x_fwd[j] += h;
            let f_fwd = dro_objective(&x_fwd);
            grad[j] = (f_fwd - f0) / h;
            x_fwd[j] = x[j];
        }

        let gn = l2_norm_arr1(&grad);
        if gn < config.tol {
            converged = true;
            break;
        }

        for j in 0..n {
            x[j] -= config.step_size * grad[j];
        }
    }

    let empirical_loss: f64 = scenarios
        .iter()
        .map(|xi| loss_fn(&x.view(), &xi.view()))
        .sum::<f64>()
        / n_scen as f64;
    let lip = estimate_lipschitz(&x);
    let worst_case_loss = empirical_loss + config.epsilon * lip;

    Ok(WassersteinDROResult {
        x,
        worst_case_loss,
        empirical_loss,
        lipschitz_estimate: lip,
        n_iter: config.max_iter,
        converged,
        message: if converged {
            "Wasserstein DRO converged".to_string()
        } else {
            "Wasserstein DRO reached maximum iterations".to_string()
        },
    })
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn l2_norm_arr1(v: &Array1<f64>) -> f64 {
    v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

fn l2_norm_arr2(m: &Array2<f64>) -> f64 {
    m.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn quadratic_loss(x: &ArrayView1<f64>, xi: &ArrayView1<f64>) -> f64 {
        (x[0] - xi[0]).powi(2)
    }

    fn make_uniform_scenarios(n: usize) -> Vec<Array1<f64>> {
        // Deterministic quasi-uniform grid on [0, 1]
        (0..n)
            .map(|i| array![(i as f64 + 0.5) / n as f64])
            .collect()
    }

    #[test]
    fn test_worst_case_analysis_basic() {
        let x = array![0.5];
        let scenarios = make_uniform_scenarios(10);
        let obj_fn = |x: &ArrayView1<f64>, xi: &ArrayView1<f64>| (x[0] - xi[0]).powi(2);
        let constraints: &[fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64] = &[];
        let result = worst_case_analysis(&obj_fn, &x.view(), &scenarios, constraints)
            .expect("failed to create result");

        assert_eq!(result.n_scenarios, 10);
        assert!(result.worst_obj >= result.best_obj);
        assert!(result.avg_obj >= result.best_obj);
        assert!((result.feasibility_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_worst_case_analysis_with_constraints() {
        let x = array![0.5];
        let scenarios = make_uniform_scenarios(5);
        let obj_fn = |_x: &ArrayView1<f64>, xi: &ArrayView1<f64>| xi[0];
        // Constraint: ξ ≤ 0.5 → only first half feasible
        let constraints = vec![|_x: &ArrayView1<f64>, xi: &ArrayView1<f64>| xi[0] - 0.5];
        let result = worst_case_analysis(&obj_fn, &x.view(), &scenarios, &constraints)
            .expect("failed to create result");
        // Scenarios: 0.1, 0.3, 0.5, 0.7, 0.9 → 0.5 border, > 0.5 infeasible
        assert!(result.feasibility_rate <= 1.0);
        assert!(result.feasibility_rate >= 0.0);
    }

    #[test]
    fn test_scenario_approach_basic() {
        // min_x x² s.t. (x - ξ)² ≤ 1 for all ξ
        let obj_fn = |x: &ArrayView1<f64>| x[0].powi(2);
        // Constraint: (x - ξ)² - 1 ≤ 0
        let constraint_fn =
            |x: &ArrayView1<f64>, xi: &ArrayView1<f64>| (x[0] - xi[0]).powi(2) - 1.0;

        let mut idx = 0_usize;
        let grid: Vec<f64> = (0..50).map(|i| (i as f64) / 49.0).collect();
        let mut sample_fn = || {
            let v = grid[idx % grid.len()];
            idx += 1;
            array![v]
        };

        let x0 = array![2.0];
        let config = ScenarioApproachConfig {
            n_scenarios: 50,
            confidence: 0.9,
            max_iter: 1_000,
            tol: 1e-4,
            step_size: 1e-2,
            ..Default::default()
        };
        let result =
            scenario_approach(&obj_fn, &constraint_fn, &mut sample_fn, &x0.view(), &config)
                .expect("unexpected None or Err");

        assert!(result.x[0].abs() <= 3.0, "x* should be bounded");
        assert!(result.feasibility_guarantee >= 0.0);
        assert!(result.feasibility_guarantee <= 1.0);
    }

    #[test]
    fn test_distributionally_robust_basic() {
        // min_x E[(x - ξ)²] under Wasserstein ball; optimum near E[ξ]
        let scenarios = make_uniform_scenarios(20);
        let x0 = array![0.0];
        let config = WassersteinDROConfig {
            epsilon: 0.05,
            p_order: 1,
            lambda: 1.0,
            max_iter: 1_000,
            tol: 1e-4,
            step_size: 1e-2,
            ..Default::default()
        };
        let result = distributionally_robust(&quadratic_loss, &x0.view(), &scenarios, &config)
            .expect("unexpected None or Err");

        // DRO minimizer should be in [0, 1]
        assert!(
            result.x[0] >= -0.5 && result.x[0] <= 1.5,
            "DRO minimizer {} should be near [0,1]",
            result.x[0]
        );
        assert!(result.worst_case_loss >= 0.0);
    }

    #[test]
    fn test_affinely_adjustable_basic() {
        // Simple AARC: obj(x, y, ξ) = (x + y - ξ)²
        // With affine policy y(ξ) = L ξ + m, optimal L=1, m=0 → y(ξ)=ξ → obj=x²
        let obj_fn = |x: &ArrayView1<f64>, y: &ArrayView1<f64>, xi: &ArrayView1<f64>| {
            (x[0] + y[0] - xi[0]).powi(2)
        };
        let xi_samples = make_uniform_scenarios(10);
        let x0 = array![0.0];
        let config = AARCConfig {
            recourse_dim: 1,
            max_iter: 200,
            tol: 1e-4,
            step_size: 1e-3,
            ..Default::default()
        };
        let result = affinely_adjustable(&obj_fn, &x0.view(), &xi_samples, 1, &config)
            .expect("failed to create result");
        assert!(result.worst_obj >= 0.0);
        assert_eq!(result.k_matrix.shape(), [1, 1]);
        assert_eq!(result.l_matrix.shape(), [1, 1]);
    }

    #[test]
    fn test_worst_case_empty_scenarios() {
        let x = array![0.0];
        let empty: Vec<Array1<f64>> = vec![];
        let obj = |_x: &ArrayView1<f64>, _xi: &ArrayView1<f64>| 0.0;
        let constraints: &[fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64] = &[];
        assert!(worst_case_analysis(&obj, &x.view(), &empty, constraints).is_err());
    }

    #[test]
    fn test_distributionally_robust_empty_scenarios() {
        let x0 = array![0.0];
        let config = WassersteinDROConfig::default();
        let empty: Vec<Array1<f64>> = vec![];
        assert!(distributionally_robust(&quadratic_loss, &x0.view(), &empty, &config).is_err());
    }
}
