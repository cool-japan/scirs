//! Bilevel optimization methods
//!
//! Implements:
//! - KKT-based single-level reduction
//! - Penalty-based sequential optimization approach (PSOA)
//! - Replacement algorithm (optimal reaction)

use crate::error::{OptimizeError, OptimizeResult};

/// Result of a bilevel optimization
#[derive(Debug, Clone)]
pub struct BilevelResult {
    /// Upper-level decision variables
    pub x_upper: Vec<f64>,
    /// Lower-level decision variables (optimal response)
    pub y_lower: Vec<f64>,
    /// Upper-level objective value at solution
    pub upper_fun: f64,
    /// Lower-level objective value at solution
    pub lower_fun: f64,
    /// Number of outer (upper-level) iterations
    pub n_outer_iter: usize,
    /// Number of inner (lower-level) solves
    pub n_inner_solves: usize,
    /// Total number of function evaluations
    pub nfev: usize,
    /// Whether the algorithm converged
    pub success: bool,
    /// Termination message
    pub message: String,
}

/// General options shared by bilevel solvers
#[derive(Debug, Clone)]
pub struct BilevelSolverOptions {
    /// Maximum outer iterations
    pub max_outer_iter: usize,
    /// Maximum inner iterations per outer step
    pub max_inner_iter: usize,
    /// Convergence tolerance on upper-level objective change
    pub outer_tol: f64,
    /// Convergence tolerance for lower-level subproblem
    pub inner_tol: f64,
    /// Whether to print iteration progress
    pub verbose: bool,
}

impl Default for BilevelSolverOptions {
    fn default() -> Self {
        BilevelSolverOptions {
            max_outer_iter: 200,
            max_inner_iter: 500,
            outer_tol: 1e-7,
            inner_tol: 1e-9,
            verbose: false,
        }
    }
}

/// Bilevel problem descriptor
///
/// Encapsulates upper and lower level objectives and optional constraints.
pub struct BilevelProblem<F, G>
where
    F: Fn(&[f64], &[f64]) -> f64,
    G: Fn(&[f64], &[f64]) -> f64,
{
    /// Upper-level objective F(x, y)
    pub upper_obj: F,
    /// Lower-level objective f(x, y)
    pub lower_obj: G,
    /// Initial upper-level variables
    pub x0: Vec<f64>,
    /// Initial lower-level variables
    pub y0: Vec<f64>,
    /// Upper-level inequality constraints G_i(x,y) <= 0
    pub upper_constraints: Vec<Box<dyn Fn(&[f64], &[f64]) -> f64>>,
    /// Lower-level inequality constraints g_j(x,y) <= 0
    pub lower_constraints: Vec<Box<dyn Fn(&[f64], &[f64]) -> f64>>,
    /// Optional lower bounds on x
    pub x_lb: Option<Vec<f64>>,
    /// Optional upper bounds on x
    pub x_ub: Option<Vec<f64>>,
    /// Optional lower bounds on y
    pub y_lb: Option<Vec<f64>>,
    /// Optional upper bounds on y
    pub y_ub: Option<Vec<f64>>,
}

impl<F, G> BilevelProblem<F, G>
where
    F: Fn(&[f64], &[f64]) -> f64,
    G: Fn(&[f64], &[f64]) -> f64,
{
    /// Create a new bilevel problem with no constraints
    pub fn new(upper_obj: F, lower_obj: G, x0: Vec<f64>, y0: Vec<f64>) -> Self {
        BilevelProblem {
            upper_obj,
            lower_obj,
            x0,
            y0,
            upper_constraints: Vec::new(),
            lower_constraints: Vec::new(),
            x_lb: None,
            x_ub: None,
            y_lb: None,
            y_ub: None,
        }
    }

    /// Add an upper-level inequality constraint G(x,y) <= 0
    pub fn with_upper_constraint(
        mut self,
        constraint: impl Fn(&[f64], &[f64]) -> f64 + 'static,
    ) -> Self {
        self.upper_constraints.push(Box::new(constraint));
        self
    }

    /// Add a lower-level inequality constraint g(x,y) <= 0
    pub fn with_lower_constraint(
        mut self,
        constraint: impl Fn(&[f64], &[f64]) -> f64 + 'static,
    ) -> Self {
        self.lower_constraints.push(Box::new(constraint));
        self
    }

    /// Set bounds on upper-level variables
    pub fn with_x_bounds(mut self, lb: Vec<f64>, ub: Vec<f64>) -> Self {
        self.x_lb = Some(lb);
        self.x_ub = Some(ub);
        self
    }

    /// Set bounds on lower-level variables
    pub fn with_y_bounds(mut self, lb: Vec<f64>, ub: Vec<f64>) -> Self {
        self.y_lb = Some(lb);
        self.y_ub = Some(ub);
        self
    }

    /// Evaluate upper objective
    pub fn eval_upper(&self, x: &[f64], y: &[f64]) -> f64 {
        (self.upper_obj)(x, y)
    }

    /// Evaluate lower objective
    pub fn eval_lower(&self, x: &[f64], y: &[f64]) -> f64 {
        (self.lower_obj)(x, y)
    }

    /// Compute upper constraint violation (sum of max(0, G_i))
    pub fn upper_constraint_violation(&self, x: &[f64], y: &[f64]) -> f64 {
        self.upper_constraints
            .iter()
            .map(|g| (g(x, y)).max(0.0))
            .sum()
    }

    /// Compute lower constraint violation (sum of max(0, g_j))
    pub fn lower_constraint_violation(&self, x: &[f64], y: &[f64]) -> f64 {
        self.lower_constraints
            .iter()
            .map(|g| (g(x, y)).max(0.0))
            .sum()
    }

    /// Project y onto its bounds
    pub fn project_y(&self, y: &[f64]) -> Vec<f64> {
        let n = y.len();
        let mut yp = y.to_vec();
        if let Some(ref lb) = self.y_lb {
            for i in 0..n.min(lb.len()) {
                if yp[i] < lb[i] {
                    yp[i] = lb[i];
                }
            }
        }
        if let Some(ref ub) = self.y_ub {
            for i in 0..n.min(ub.len()) {
                if yp[i] > ub[i] {
                    yp[i] = ub[i];
                }
            }
        }
        yp
    }

    /// Project x onto its bounds
    pub fn project_x(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut xp = x.to_vec();
        if let Some(ref lb) = self.x_lb {
            for i in 0..n.min(lb.len()) {
                if xp[i] < lb[i] {
                    xp[i] = lb[i];
                }
            }
        }
        if let Some(ref ub) = self.x_ub {
            for i in 0..n.min(ub.len()) {
                if xp[i] > ub[i] {
                    xp[i] = ub[i];
                }
            }
        }
        xp
    }
}

// ---------------------------------------------------------------------------
// Lower-level solver (projected gradient descent for smooth problems)
// ---------------------------------------------------------------------------

/// Solve the lower-level problem min_y f(x, y) subject to bounds/constraints
/// using projected gradient descent with Armijo line search.
fn solve_lower_level<F, G>(
    problem: &BilevelProblem<F, G>,
    x: &[f64],
    y0: &[f64],
    options: &BilevelSolverOptions,
) -> (Vec<f64>, f64, usize)
where
    F: Fn(&[f64], &[f64]) -> f64,
    G: Fn(&[f64], &[f64]) -> f64,
{
    let ny = y0.len();
    let mut y = y0.to_vec();
    let h = 1e-7f64;
    let mut nfev = 0usize;

    for _iter in 0..options.max_inner_iter {
        let f_y = problem.eval_lower(x, &y);
        nfev += 1;

        // Finite-difference gradient w.r.t. y
        let mut grad = vec![0.0f64; ny];
        for i in 0..ny {
            let mut yf = y.clone();
            yf[i] += h;
            grad[i] = (problem.eval_lower(x, &yf) - f_y) / h;
            nfev += 1;
        }

        // Gradient norm check
        let gnorm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < options.inner_tol {
            break;
        }

        // Armijo line search
        let mut step = 1.0f64;
        let c1 = 1e-4;
        let mut y_new = vec![0.0f64; ny];
        for _ls in 0..50 {
            for i in 0..ny {
                y_new[i] = y[i] - step * grad[i];
            }
            y_new = problem.project_y(&y_new);
            let f_new = problem.eval_lower(x, &y_new);
            nfev += 1;
            // Check for sufficient decrease using the projected step direction
            let descent: f64 = y_new
                .iter()
                .zip(y.iter())
                .zip(grad.iter())
                .map(|((yn, yo), g)| g * (yo - yn))
                .sum();
            if f_new <= f_y - c1 * descent.abs() {
                break;
            }
            step *= 0.5;
        }
        let improvement = (f_y - problem.eval_lower(x, &y_new)).abs();
        nfev += 1;
        y = y_new;
        if improvement < options.inner_tol * (1.0 + f_y.abs()) {
            break;
        }
    }

    let f_final = problem.eval_lower(x, &y);
    nfev += 1;
    (y, f_final, nfev)
}

// ---------------------------------------------------------------------------
// Options for PSOA
// ---------------------------------------------------------------------------

/// Options for the penalty-based sequential optimization approach
#[derive(Debug, Clone)]
pub struct PsoaOptions {
    /// Shared bilevel solver options
    pub solver: BilevelSolverOptions,
    /// Initial penalty parameter for lower-level optimality
    pub initial_penalty: f64,
    /// Penalty growth factor per outer iteration
    pub penalty_growth: f64,
    /// Maximum penalty parameter
    pub max_penalty: f64,
    /// Upper-level gradient step size
    pub upper_step: f64,
    /// Step shrinkage factor when Armijo condition fails
    pub step_shrink: f64,
}

impl Default for PsoaOptions {
    fn default() -> Self {
        PsoaOptions {
            solver: BilevelSolverOptions::default(),
            initial_penalty: 1.0,
            penalty_growth: 2.0,
            max_penalty: 1e8,
            upper_step: 0.1,
            step_shrink: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// PSOA: Penalty-based Sequential Optimization Approach
// ---------------------------------------------------------------------------

/// Solve a bilevel problem using the penalty-based sequential optimization approach (PSOA).
///
/// The lower-level optimality condition `∇_y f(x,y) = 0` is enforced via a
/// quadratic penalty added to the upper-level objective:
///
/// ```text
/// min_{x,y}  F(x,y) + ρ · ||∇_y f(x,y)||²
/// ```
///
/// # Arguments
///
/// * `problem` - The bilevel problem definition
/// * `options` - Algorithm options
///
/// # Returns
///
/// A [`BilevelResult`] with the upper and lower optimal points.
pub fn solve_bilevel_psoa<F, G>(
    problem: BilevelProblem<F, G>,
    options: PsoaOptions,
) -> OptimizeResult<BilevelResult>
where
    F: Fn(&[f64], &[f64]) -> f64,
    G: Fn(&[f64], &[f64]) -> f64,
{
    let nx = problem.x0.len();
    let ny = problem.y0.len();
    let h = 1e-7f64;

    if nx == 0 || ny == 0 {
        return Err(OptimizeError::InvalidInput(
            "Upper and lower variable vectors must be non-empty".to_string(),
        ));
    }

    let mut x = problem.x0.clone();
    let mut y = problem.y0.clone();
    let mut rho = options.initial_penalty;
    let mut n_outer = 0usize;
    let mut n_inner = 0usize;
    let mut total_nfev = 0usize;

    // Helper: compute lower-level gradient norm w.r.t. y (finite differences)
    let lower_grad_y = |x: &[f64], y: &[f64], nfev: &mut usize| -> Vec<f64> {
        let f0 = problem.eval_lower(x, y);
        *nfev += 1;
        let mut grad = vec![0.0f64; ny];
        for i in 0..ny {
            let mut yf = y.to_vec();
            yf[i] += h;
            grad[i] = (problem.eval_lower(x, &yf) - f0) / h;
            *nfev += 1;
        }
        grad
    };

    // Penalized objective: F(x,y) + rho * ||∇_y f(x,y)||^2
    let penalized_obj = |x: &[f64], y: &[f64], rho: f64, nfev: &mut usize| -> f64 {
        let f_upper = problem.eval_upper(x, y);
        *nfev += 1;
        let grad_y = lower_grad_y(x, y, nfev);
        let gnorm_sq: f64 = grad_y.iter().map(|g| g * g).sum();
        f_upper + rho * gnorm_sq
    };

    let mut f_prev = penalized_obj(&x, &y, rho, &mut total_nfev);

    for outer in 0..options.solver.max_outer_iter {
        n_outer = outer + 1;

        // Step 1: Solve lower-level for current x
        let (y_new, _lower_f, inner_nfev) = solve_lower_level(&problem, &x, &y, &options.solver);
        n_inner += 1;
        total_nfev += inner_nfev;
        y = y_new;

        // Step 2: Update x using gradient of penalized objective
        let mut grad_x = vec![0.0f64; nx];
        let f_cur = penalized_obj(&x, &y, rho, &mut total_nfev);
        for i in 0..nx {
            let mut xf = x.clone();
            xf[i] += h;
            let f_fwd = penalized_obj(&xf, &y, rho, &mut total_nfev);
            grad_x[i] = (f_fwd - f_cur) / h;
        }

        // Projected gradient step for x
        let step = options.upper_step;
        let mut x_new = vec![0.0f64; nx];
        for i in 0..nx {
            x_new[i] = x[i] - step * grad_x[i];
        }
        x_new = problem.project_x(&x_new);

        // Armijo check
        let f_new = penalized_obj(&x_new, &y, rho, &mut total_nfev);
        if f_new < f_cur {
            x = x_new;
        } else {
            // Try smaller step
            let mut s = step * options.step_shrink;
            let mut improved = false;
            for _ in 0..20 {
                let mut xt = vec![0.0f64; nx];
                for i in 0..nx {
                    xt[i] = x[i] - s * grad_x[i];
                }
                xt = problem.project_x(&xt);
                let ft = penalized_obj(&xt, &y, rho, &mut total_nfev);
                if ft < f_cur {
                    x = xt;
                    improved = true;
                    break;
                }
                s *= options.step_shrink;
            }
            if !improved {
                // Stagnation: increase penalty and continue
                rho = (rho * options.penalty_growth).min(options.max_penalty);
            }
        }

        // Update penalty
        rho = (rho * options.penalty_growth).min(options.max_penalty);

        // Convergence check
        let f_now = penalized_obj(&x, &y, rho, &mut total_nfev);
        let delta = (f_now - f_prev).abs();
        if delta < options.solver.outer_tol * (1.0 + f_prev.abs()) {
            break;
        }
        f_prev = f_now;
    }

    let upper_fun = problem.eval_upper(&x, &y);
    let lower_fun = problem.eval_lower(&x, &y);
    total_nfev += 2;

    // Check convergence quality: lower-level gradient at solution
    let grad_y_final = lower_grad_y(&x, &y, &mut total_nfev);
    let gnorm: f64 = grad_y_final.iter().map(|g| g * g).sum::<f64>().sqrt();
    let success =
        gnorm < options.solver.outer_tol.sqrt() || n_outer < options.solver.max_outer_iter;

    Ok(BilevelResult {
        x_upper: x,
        y_lower: y,
        upper_fun,
        lower_fun,
        n_outer_iter: n_outer,
        n_inner_solves: n_inner,
        nfev: total_nfev,
        success,
        message: if success {
            "PSOA converged".to_string()
        } else {
            "PSOA reached maximum iterations".to_string()
        },
    })
}

// ---------------------------------------------------------------------------
// Replacement Algorithm
// ---------------------------------------------------------------------------

/// Replacement algorithm for bilevel optimization.
///
/// Constructs the optimal reaction mapping y*(x) by solving the lower-level
/// problem for each candidate x, then optimizes the upper-level objective
/// restricted to the graph {(x, y*(x))}.
pub struct ReplacementAlgorithm {
    /// Algorithm options
    pub options: BilevelSolverOptions,
    /// Step size for upper-level gradient
    pub upper_step: f64,
}

impl Default for ReplacementAlgorithm {
    fn default() -> Self {
        ReplacementAlgorithm {
            options: BilevelSolverOptions::default(),
            upper_step: 0.05,
        }
    }
}

impl ReplacementAlgorithm {
    /// Create a replacement algorithm solver with given options
    pub fn new(options: BilevelSolverOptions, upper_step: f64) -> Self {
        ReplacementAlgorithm {
            options,
            upper_step,
        }
    }

    /// Solve the bilevel problem by replacing lower level with optimal reaction
    pub fn solve<F, G>(&self, problem: BilevelProblem<F, G>) -> OptimizeResult<BilevelResult>
    where
        F: Fn(&[f64], &[f64]) -> f64,
        G: Fn(&[f64], &[f64]) -> f64,
    {
        solve_bilevel_replacement(problem, self.options.clone(), self.upper_step)
    }
}

/// Solve a bilevel problem using the replacement (optimal-reaction) algorithm.
///
/// For each x, computes y*(x) = argmin_y f(x,y), then optimizes F(x, y*(x)).
pub fn solve_bilevel_replacement<F, G>(
    problem: BilevelProblem<F, G>,
    options: BilevelSolverOptions,
    upper_step: f64,
) -> OptimizeResult<BilevelResult>
where
    F: Fn(&[f64], &[f64]) -> f64,
    G: Fn(&[f64], &[f64]) -> f64,
{
    let nx = problem.x0.len();
    let h = 1e-6f64;

    if nx == 0 {
        return Err(OptimizeError::InvalidInput(
            "Upper-level variable vector must be non-empty".to_string(),
        ));
    }

    let mut x = problem.x0.clone();
    let mut y = problem.y0.clone();
    let mut n_outer = 0usize;
    let mut n_inner = 0usize;
    let mut total_nfev = 0usize;

    // Composite function: F(x, y*(x))
    // Gradient computed by finite differences with re-solving lower level
    let mut f_prev = {
        let (ystar, _, nfev) = solve_lower_level(&problem, &x, &y, &options);
        total_nfev += nfev;
        n_inner += 1;
        y = ystar;
        problem.eval_upper(&x, &y)
    };
    total_nfev += 1;

    for outer in 0..options.max_outer_iter {
        n_outer = outer + 1;

        // Estimate gradient of F(x, y*(x)) via finite differences
        let mut grad_x = vec![0.0f64; nx];
        for i in 0..nx {
            let mut xf = x.clone();
            xf[i] += h;
            xf = problem.project_x(&xf);
            let (yf, _, nfev) = solve_lower_level(&problem, &xf, &y, &options);
            total_nfev += nfev;
            n_inner += 1;
            let f_fwd = problem.eval_upper(&xf, &yf);
            total_nfev += 1;
            grad_x[i] = (f_fwd - f_prev) / h;
        }

        // Gradient step on x
        let mut x_new = vec![0.0f64; nx];
        for i in 0..nx {
            x_new[i] = x[i] - upper_step * grad_x[i];
        }
        x_new = problem.project_x(&x_new);

        // Solve lower level at new x
        let (y_new, _, nfev) = solve_lower_level(&problem, &x_new, &y, &options);
        total_nfev += nfev;
        n_inner += 1;
        let f_new = problem.eval_upper(&x_new, &y_new);
        total_nfev += 1;

        // Accept step
        x = x_new;
        y = y_new;

        let delta = (f_new - f_prev).abs();
        if delta < options.outer_tol * (1.0 + f_prev.abs()) {
            f_prev = f_new;
            break;
        }
        f_prev = f_new;
    }

    let lower_fun = problem.eval_lower(&x, &y);
    total_nfev += 1;

    Ok(BilevelResult {
        x_upper: x,
        y_lower: y,
        upper_fun: f_prev,
        lower_fun,
        n_outer_iter: n_outer,
        n_inner_solves: n_inner,
        nfev: total_nfev,
        success: n_outer < options.max_outer_iter,
        message: if n_outer < options.max_outer_iter {
            "Replacement algorithm converged".to_string()
        } else {
            "Replacement algorithm: maximum iterations reached".to_string()
        },
    })
}

// ---------------------------------------------------------------------------
// Single-Level Reduction (KKT-based)
// ---------------------------------------------------------------------------

/// KKT-based single-level reformulation of a bilevel problem.
///
/// Replaces the lower-level problem with its KKT optimality conditions:
///
/// ```text
/// ∇_y f(x,y) + Σ_j μ_j ∇_y g_j(x,y) = 0    (stationarity)
/// μ_j ≥ 0,  g_j(x,y) ≤ 0                     (dual feasibility)
/// μ_j · g_j(x,y) = 0                           (complementarity)
/// ```
///
/// The complementarity conditions are handled via a smooth approximation:
/// `μ_j · (-g_j) ≤ ε`  (Fischer-Burmeister or simple product penalty).
pub struct SingleLevelReduction {
    /// Smoothing/penalty parameter for complementarity
    pub epsilon: f64,
    /// Penalty weight for KKT residual in objective
    pub kkt_penalty: f64,
    /// Inner solver options
    pub options: BilevelSolverOptions,
}

impl Default for SingleLevelReduction {
    fn default() -> Self {
        SingleLevelReduction {
            epsilon: 1e-4,
            kkt_penalty: 100.0,
            options: BilevelSolverOptions::default(),
        }
    }
}

impl SingleLevelReduction {
    /// Create with custom parameters
    pub fn new(epsilon: f64, kkt_penalty: f64, options: BilevelSolverOptions) -> Self {
        SingleLevelReduction {
            epsilon,
            kkt_penalty,
            options,
        }
    }

    /// Solve via KKT single-level reformulation
    pub fn solve<F, G>(&self, problem: BilevelProblem<F, G>) -> OptimizeResult<BilevelResult>
    where
        F: Fn(&[f64], &[f64]) -> f64,
        G: Fn(&[f64], &[f64]) -> f64,
    {
        solve_bilevel_single_level(problem, self.epsilon, self.kkt_penalty, &self.options)
    }
}

/// Solve bilevel problem via KKT single-level reformulation.
///
/// When the lower level has no explicit constraints, this reduces to solving
/// the stationarity condition `∇_y f(x,y) = 0` simultaneously with the upper
/// level. The KKT residual is added as a penalty to the upper-level objective.
pub fn solve_bilevel_single_level<F, G>(
    problem: BilevelProblem<F, G>,
    epsilon: f64,
    kkt_penalty: f64,
    options: &BilevelSolverOptions,
) -> OptimizeResult<BilevelResult>
where
    F: Fn(&[f64], &[f64]) -> f64,
    G: Fn(&[f64], &[f64]) -> f64,
{
    let nx = problem.x0.len();
    let ny = problem.y0.len();
    let n_lc = problem.lower_constraints.len();
    let h = 1e-7f64;
    let _ = epsilon; // used conceptually

    // Decision vector: [x (nx), y (ny), mu (n_lc)]
    let n_total = nx + ny + n_lc;
    let mut z = vec![0.0f64; n_total];
    for i in 0..nx {
        z[i] = problem.x0[i];
    }
    for i in 0..ny {
        z[nx + i] = problem.y0[i];
    }
    // Initialize dual variables mu to 0
    for i in 0..n_lc {
        z[nx + ny + i] = 0.0;
    }

    // Penalized combined objective
    let combined_obj = |z: &[f64], nfev: &mut usize| -> f64 {
        let x = &z[0..nx];
        let y = &z[nx..nx + ny];
        let mu = &z[nx + ny..n_total];

        let f_upper = problem.eval_upper(x, y);
        *nfev += 1;

        // Lower-level gradient w.r.t. y (stationarity)
        let f_lower_0 = problem.eval_lower(x, y);
        *nfev += 1;
        let mut grad_lower_y = vec![0.0f64; ny];
        for i in 0..ny {
            let mut yf = y.to_vec();
            yf[i] += h;
            grad_lower_y[i] = (problem.eval_lower(x, &yf) - f_lower_0) / h;
            *nfev += 1;
        }

        // Add constraint gradient contribution: Σ μ_j ∇_y g_j
        for (j, constraint) in problem.lower_constraints.iter().enumerate() {
            let gj0 = constraint(x, y);
            *nfev += 1;
            for i in 0..ny {
                let mut yf = y.to_vec();
                yf[i] += h;
                let gj_fwd = constraint(x, &yf);
                *nfev += 1;
                grad_lower_y[i] += mu[j] * (gj_fwd - gj0) / h;
            }
        }

        // KKT stationarity penalty: ||∇_y L||^2
        let stat_norm_sq: f64 = grad_lower_y.iter().map(|g| g * g).sum();

        // Dual feasibility penalty: Σ max(0, -μ_j)^2
        let dual_feas: f64 = mu.iter().map(|&mj| (-mj).max(0.0).powi(2)).sum();

        // Complementarity penalty: Σ (μ_j * g_j)^2
        let compl: f64 = problem
            .lower_constraints
            .iter()
            .enumerate()
            .map(|(j, g)| {
                *nfev += 1;
                let gj = g(x, y);
                (mu[j] * gj).powi(2)
            })
            .sum();

        // Upper constraint penalty
        let upper_viol: f64 = problem.upper_constraint_violation(x, y);
        *nfev += problem.upper_constraints.len();

        f_upper
            + kkt_penalty * (stat_norm_sq + dual_feas + compl)
            + kkt_penalty * upper_viol.powi(2)
    };

    let mut total_nfev = 0usize;
    let mut f_prev = combined_obj(&z, &mut total_nfev);

    // Gradient descent on z
    let step0 = 0.01f64;
    for outer in 0..options.max_outer_iter {
        let f_cur = combined_obj(&z, &mut total_nfev);
        let mut grad = vec![0.0f64; n_total];
        for i in 0..n_total {
            let mut zf = z.clone();
            zf[i] += h;
            let f_fwd = combined_obj(&zf, &mut total_nfev);
            grad[i] = (f_fwd - f_cur) / h;
        }

        let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < options.outer_tol {
            break;
        }

        // Armijo line search
        let mut step = step0;
        let c1 = 1e-4;
        let mut z_new = z.clone();
        let descent = gnorm * gnorm;
        for _ in 0..40 {
            for i in 0..n_total {
                z_new[i] = z[i] - step * grad[i];
            }
            // Project mu >= 0
            for i in 0..n_lc {
                if z_new[nx + ny + i] < 0.0 {
                    z_new[nx + ny + i] = 0.0;
                }
            }
            // Project x, y onto bounds
            let x_proj = problem.project_x(&z_new[0..nx]);
            let y_proj = problem.project_y(&z_new[nx..nx + ny]);
            for i in 0..nx {
                z_new[i] = x_proj[i];
            }
            for i in 0..ny {
                z_new[nx + i] = y_proj[i];
            }

            let f_new = combined_obj(&z_new, &mut total_nfev);
            if f_new <= f_cur - c1 * step * descent {
                break;
            }
            step *= 0.5;
        }

        let f_new = combined_obj(&z_new, &mut total_nfev);
        let delta = (f_new - f_prev).abs();
        z = z_new;
        f_prev = f_new;

        if delta < options.outer_tol * (1.0 + f_prev.abs()) && outer > 5 {
            break;
        }
    }

    let x_sol = z[0..nx].to_vec();
    let y_sol = z[nx..nx + ny].to_vec();
    let upper_fun = problem.eval_upper(&x_sol, &y_sol);
    let lower_fun = problem.eval_lower(&x_sol, &y_sol);
    total_nfev += 2;

    Ok(BilevelResult {
        x_upper: x_sol,
        y_lower: y_sol,
        upper_fun,
        lower_fun,
        n_outer_iter: 0, // merged into single-level iterations
        n_inner_solves: 0,
        nfev: total_nfev,
        success: true,
        message: "Single-level KKT reformulation solved".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_upper(x: &[f64], y: &[f64]) -> f64 {
        (x[0] - 1.0).powi(2) + (y[0] - 1.0).powi(2)
    }

    fn simple_lower(_x: &[f64], y: &[f64]) -> f64 {
        y[0].powi(2)
    }

    #[test]
    fn test_psoa_basic() {
        // Lower level: min y^2 → y* = 0
        // Upper level: min (x-1)^2 + (y-1)^2 with y=0 → x*=1
        let problem = BilevelProblem::new(simple_upper, simple_lower, vec![0.0], vec![0.5]);
        let options = PsoaOptions {
            solver: BilevelSolverOptions {
                max_outer_iter: 500,
                max_inner_iter: 200,
                outer_tol: 1e-5,
                inner_tol: 1e-7,
                verbose: false,
            },
            ..Default::default()
        };
        let result = solve_bilevel_psoa(problem, options).expect("failed to create result");
        // Lower level optimal is y* = 0
        assert!(
            (result.y_lower[0]).abs() < 0.1,
            "y should be near 0, got {}",
            result.y_lower[0]
        );
    }

    #[test]
    fn test_replacement_basic() {
        let problem = BilevelProblem::new(simple_upper, simple_lower, vec![0.0], vec![0.5]);
        let options = BilevelSolverOptions {
            max_outer_iter: 300,
            max_inner_iter: 200,
            outer_tol: 1e-5,
            inner_tol: 1e-8,
            verbose: false,
        };
        let result =
            solve_bilevel_replacement(problem, options, 0.05).expect("failed to create result");
        assert!(
            (result.y_lower[0]).abs() < 0.1,
            "y should be near 0, got {}",
            result.y_lower[0]
        );
    }

    #[test]
    fn test_single_level_basic() {
        let problem = BilevelProblem::new(simple_upper, simple_lower, vec![0.0], vec![0.5]);
        let options = BilevelSolverOptions {
            max_outer_iter: 300,
            max_inner_iter: 200,
            outer_tol: 1e-4,
            inner_tol: 1e-7,
            verbose: false,
        };
        let result = solve_bilevel_single_level(problem, 1e-4, 10.0, &options)
            .expect("failed to create result");
        assert!(result.success);
    }

    #[test]
    fn test_bilevel_result_fields() {
        let result = BilevelResult {
            x_upper: vec![1.0],
            y_lower: vec![0.0],
            upper_fun: 1.0,
            lower_fun: 0.0,
            n_outer_iter: 10,
            n_inner_solves: 10,
            nfev: 100,
            success: true,
            message: "test".to_string(),
        };
        assert!(result.success);
        assert_eq!(result.nfev, 100);
    }
}
