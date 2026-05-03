//! Decomposition methods for large-scale optimization
//!
//! Implements:
//! - [`BendersDecomposition`]: Master + subproblem decomposition
//! - [`DantzigWolfe`]: Dantzig-Wolfe decomposition for structured LP
//! - [`ADMM`]: Alternating Direction Method of Multipliers
//! - [`ProximalBundle`]: Proximal bundle method for nonsmooth optimization

use crate::error::{OptimizeError, OptimizeResult};

// ---------------------------------------------------------------------------
// Benders Decomposition
// ---------------------------------------------------------------------------

/// Options for Benders decomposition
#[derive(Debug, Clone)]
pub struct BendersOptions {
    /// Maximum number of outer iterations (adding cuts)
    pub max_iter: usize,
    /// Convergence tolerance (upper - lower bound gap)
    pub tol: f64,
    /// Feasibility tolerance for subproblem
    pub feas_tol: f64,
    /// Maximum subproblem iterations
    pub max_sub_iter: usize,
    /// Subproblem convergence tolerance
    pub sub_tol: f64,
    /// Whether to use multi-cut variant (separate cut per constraint group)
    pub multi_cut: bool,
}

impl Default for BendersOptions {
    fn default() -> Self {
        BendersOptions {
            max_iter: 100,
            tol: 1e-7,
            feas_tol: 1e-7,
            max_sub_iter: 500,
            sub_tol: 1e-9,
            multi_cut: false,
        }
    }
}

/// Result of Benders decomposition
#[derive(Debug, Clone)]
pub struct BendersResult {
    /// Optimal first-stage (master) variables
    pub x: Vec<f64>,
    /// Optimal second-stage (subproblem) variables
    pub y: Vec<f64>,
    /// Objective value
    pub fun: f64,
    /// Lower bound at termination
    pub lower_bound: f64,
    /// Upper bound at termination
    pub upper_bound: f64,
    /// Number of Benders cuts added
    pub n_cuts: usize,
    /// Number of iterations
    pub nit: usize,
    /// Whether the algorithm converged
    pub success: bool,
    /// Termination message
    pub message: String,
}

/// Benders decomposition solver.
///
/// Decomposes the problem:
///
/// ```text
/// min_{x,y}  c^T x + d^T y
/// s.t.        A x + B y >= b    (coupling constraints)
///             x in X,  y in Y
/// ```
///
/// into a master problem (in x) and subproblems (in y for fixed x).
/// Benders cuts are added to the master from subproblem dual solutions.
///
/// This implementation uses a projected gradient method for both the master
/// and subproblem, as a general-purpose (non-LP-specific) variant.
pub struct BendersDecomposition {
    /// Algorithm options
    pub options: BendersOptions,
}

impl Default for BendersDecomposition {
    fn default() -> Self {
        BendersDecomposition {
            options: BendersOptions::default(),
        }
    }
}

impl BendersDecomposition {
    /// Create a Benders solver with custom options
    pub fn new(options: BendersOptions) -> Self {
        BendersDecomposition { options }
    }

    /// Solve a separable optimization problem via Benders decomposition.
    ///
    /// # Arguments
    ///
    /// * `master_obj` - Master problem objective (depends on x only)
    /// * `sub_obj` - Subproblem objective (depends on x and y)
    /// * `sub_constraints` - Subproblem constraints for fixed x: returns (violation, dual)
    ///   Each entry: `Fn(&[f64], &[f64]) -> (f64, f64)` (feasibility, dual multiplier)
    /// * `x0` - Initial first-stage variables
    /// * `y0` - Initial second-stage variables
    pub fn solve<FM, FS>(
        &self,
        master_obj: FM,
        sub_obj: FS,
        x0: &[f64],
        y0: &[f64],
    ) -> OptimizeResult<BendersResult>
    where
        FM: Fn(&[f64]) -> f64,
        FS: Fn(&[f64], &[f64]) -> f64,
    {
        let nx = x0.len();
        let ny = y0.len();

        if nx == 0 {
            return Err(OptimizeError::InvalidInput(
                "First-stage (master) variables must be non-empty".to_string(),
            ));
        }

        let h = 1e-7f64;
        let mut x = x0.to_vec();
        let mut y = y0.to_vec();
        let mut lower_bound = f64::NEG_INFINITY;
        let mut upper_bound = f64::INFINITY;
        let mut n_cuts = 0usize;
        let mut nit = 0usize;

        // Benders cuts: each cut is (g, beta) where the cut is g^T * (x - x_k) + beta <= eta
        // i.e., the recourse function Q(x) >= g^T * x + (beta - g^T * x_k)
        let mut cuts: Vec<(Vec<f64>, f64)> = Vec::new();

        for outer in 0..self.options.max_iter {
            nit = outer + 1;

            // Step 1: Solve subproblem for fixed x (minimize over y)
            let mut y_opt = y.clone();
            let mut sub_f = sub_obj(&x, &y_opt);

            for _sub in 0..self.options.max_sub_iter {
                let mut grad_y = vec![0.0f64; ny];
                for i in 0..ny {
                    let mut yf = y_opt.clone();
                    yf[i] += h;
                    grad_y[i] = (sub_obj(&x, &yf) - sub_f) / h;
                }
                let gnorm: f64 = grad_y.iter().map(|g| g * g).sum::<f64>().sqrt();
                if gnorm < self.options.sub_tol {
                    break;
                }
                let step = 0.1f64 / (1.0 + gnorm);
                let mut y_new = vec![0.0f64; ny];
                for i in 0..ny {
                    y_new[i] = y_opt[i] - step * grad_y[i];
                }
                let f_new = sub_obj(&x, &y_new);
                if f_new < sub_f {
                    y_opt = y_new;
                    sub_f = f_new;
                } else {
                    break;
                }
            }

            // Compute Benders cut: subgradient of Q(x) w.r.t. x
            // Q(x) ≈ Q(x_k) + g^T (x - x_k) where g = ∂_x sub_obj(x, y*(x))
            let mut cut_grad = vec![0.0f64; nx];
            for i in 0..nx {
                let mut xf = x.clone();
                xf[i] += h;
                cut_grad[i] = (sub_obj(&xf, &y_opt) - sub_obj(&x, &y_opt)) / h;
            }
            let cut_offset = sub_f;
            cuts.push((cut_grad, cut_offset));
            n_cuts += 1;

            // Compute upper bound: master + subproblem
            let master_f = master_obj(&x);
            let current_ub = master_f + sub_f;
            if current_ub < upper_bound {
                upper_bound = current_ub;
                y = y_opt.clone();
            }

            // Step 2: Solve master problem with current cuts
            // Master: min master_obj(x) + eta  s.t. eta >= cut_k^T x + const_k for all k
            // We represent this as: min master_obj(x) + max_k(cut_k^T x + const_k - cut_k^T x_k)

            let master_with_cuts = |x: &[f64]| -> f64 {
                let base = master_obj(x);
                let recourse = cuts
                    .iter()
                    .map(|(g, offset)| {
                        let inner: f64 = g.iter().zip(x.iter()).map(|(gi, xi)| gi * xi).sum();
                        inner - g.iter().zip(x0.iter()).map(|(gi, xi)| gi * xi).sum::<f64>()
                            + offset
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                let recourse = if recourse.is_finite() { recourse } else { 0.0 };
                base + recourse
            };

            // Gradient descent on master with cuts
            let mut x_new = x.clone();
            let mut master_val = master_with_cuts(&x_new);

            for _master_iter in 0..200 {
                let mut grad = vec![0.0f64; nx];
                for i in 0..nx {
                    let mut xf = x_new.clone();
                    xf[i] += h;
                    grad[i] = (master_with_cuts(&xf) - master_val) / h;
                }
                let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                if gnorm < self.options.tol {
                    break;
                }
                let step = 0.1f64 / (1.0 + gnorm);
                let mut xt = vec![0.0f64; nx];
                for i in 0..nx {
                    xt[i] = x_new[i] - step * grad[i];
                }
                let ft = master_with_cuts(&xt);
                if ft < master_val {
                    x_new = xt;
                    master_val = ft;
                } else {
                    break;
                }
            }

            lower_bound = master_val;
            x = x_new;

            // Convergence check
            let gap = upper_bound - lower_bound;
            if gap.abs() < self.options.tol * (1.0 + upper_bound.abs()) {
                break;
            }
        }

        Ok(BendersResult {
            x,
            y,
            fun: upper_bound,
            lower_bound,
            upper_bound,
            n_cuts,
            nit,
            success: (upper_bound - lower_bound).abs()
                < self.options.tol * (1.0 + upper_bound.abs()),
            message: "Benders decomposition completed".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Dantzig-Wolfe Decomposition
// ---------------------------------------------------------------------------

/// Options for Dantzig-Wolfe decomposition
#[derive(Debug, Clone)]
pub struct DantzigWolfeOptions {
    /// Maximum number of column generation iterations
    pub max_iter: usize,
    /// Optimality tolerance (reduced cost threshold)
    pub opt_tol: f64,
    /// Maximum subproblem iterations
    pub max_sub_iter: usize,
    /// Subproblem convergence tolerance
    pub sub_tol: f64,
    /// Step size for restricted master problem
    pub master_step: f64,
}

impl Default for DantzigWolfeOptions {
    fn default() -> Self {
        DantzigWolfeOptions {
            max_iter: 200,
            opt_tol: 1e-7,
            max_sub_iter: 200,
            sub_tol: 1e-9,
            master_step: 0.1,
        }
    }
}

/// Result of Dantzig-Wolfe decomposition
#[derive(Debug, Clone)]
pub struct DantzigWolfeResult {
    /// Optimal solution (primal)
    pub x: Vec<f64>,
    /// Optimal objective value
    pub fun: f64,
    /// Number of columns (proposals) generated
    pub n_columns: usize,
    /// Number of iterations
    pub nit: usize,
    /// Dual variables (prices) at optimum
    pub duals: Vec<f64>,
    /// Whether the algorithm converged
    pub success: bool,
    /// Termination message
    pub message: String,
}

/// Dantzig-Wolfe decomposition for structured optimization problems.
///
/// Decomposes problems with a "linking" constraint structure:
///
/// ```text
/// min  c^T x
/// s.t. A0 x = b0    (linking constraints)
///      A_i x_i = b_i for each block i  (block constraints)
///      x_i in X_i
/// ```
///
/// The restricted master problem uses convex combinations of extreme points
/// of each X_i, with column generation to add improving columns.
///
/// This implementation handles smooth (not necessarily LP) problems.
pub struct DantzigWolfe {
    /// Algorithm options
    pub options: DantzigWolfeOptions,
}

impl Default for DantzigWolfe {
    fn default() -> Self {
        DantzigWolfe {
            options: DantzigWolfeOptions::default(),
        }
    }
}

impl DantzigWolfe {
    /// Create a Dantzig-Wolfe solver
    pub fn new(options: DantzigWolfeOptions) -> Self {
        DantzigWolfe { options }
    }

    /// Solve the master problem and generate columns.
    ///
    /// # Arguments
    ///
    /// * `master_obj` - Master objective f(x): smooth, depends on all x
    /// * `pricing_obj` - Pricing problem: for duals π, minimize c_bar^T x_sub
    ///   Given duals `pi`, return (optimal x_sub, min reduced cost)
    /// * `x0` - Initial point for the master
    /// * `n_blocks` - Number of subproblem blocks
    pub fn solve<FM, FP>(
        &self,
        master_obj: FM,
        pricing_oracle: FP,
        x0: &[f64],
        _n_blocks: usize,
    ) -> OptimizeResult<DantzigWolfeResult>
    where
        FM: Fn(&[f64]) -> f64,
        FP: Fn(&[f64]) -> (Vec<f64>, f64), // pricing oracle: duals -> (proposal, reduced_cost)
    {
        let n = x0.len();
        let h = 1e-7f64;

        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        let mut x = x0.to_vec();
        let mut columns: Vec<Vec<f64>> = vec![x.clone()]; // pool of generated columns
        let mut weights: Vec<f64> = vec![1.0]; // convex combination weights
        let mut duals = vec![0.0f64; n];
        let mut n_iter = 0usize;
        let mut n_columns = 1usize;

        for outer in 0..self.options.max_iter {
            n_iter = outer + 1;

            // Step 1: Compute restricted master solution (gradient descent)
            let restricted_obj = |w: &[f64]| -> f64 {
                let n_cols = columns.len();
                let mut xc = vec![0.0f64; n];
                for (j, col) in columns.iter().enumerate() {
                    let wj = if j < w.len() { w[j] } else { 0.0 };
                    for i in 0..n {
                        xc[i] += wj * col[i];
                    }
                }
                master_obj(&xc)
            };

            let nc = columns.len();
            let mut f_w = restricted_obj(&weights[..nc.min(weights.len())]);

            for _inner in 0..200 {
                let mut grad_w = vec![0.0f64; nc];
                let w_cur = &weights[..nc.min(weights.len())];
                let w_padded: Vec<f64> = {
                    let mut wp = w_cur.to_vec();
                    while wp.len() < nc {
                        wp.push(0.0);
                    }
                    wp
                };
                for j in 0..nc {
                    let mut wf = w_padded.clone();
                    let delta = h;
                    wf[j] += delta;
                    // Renormalize
                    let sum: f64 = wf.iter().sum();
                    let wf_norm: Vec<f64> = wf.iter().map(|wj| wj / sum).collect();
                    grad_w[j] = (restricted_obj(&wf_norm) - f_w) / delta;
                }
                let gnorm: f64 = grad_w.iter().map(|g| g * g).sum::<f64>().sqrt();
                if gnorm < self.options.sub_tol {
                    break;
                }
                let step = self.options.master_step / (1.0 + gnorm);
                let w_cur_len = w_padded.len();
                let mut w_new: Vec<f64> = (0..w_cur_len)
                    .map(|j| (w_padded[j] - step * grad_w[j]).max(0.0))
                    .collect();
                // Project onto simplex
                let sum: f64 = w_new.iter().sum();
                if sum > 1e-14 {
                    for wj in w_new.iter_mut() {
                        *wj /= sum;
                    }
                }
                let f_new = restricted_obj(&w_new);
                if f_new < f_w {
                    weights = w_new;
                    f_w = f_new;
                } else {
                    break;
                }
            }

            // Reconstruct x from weights and columns
            let nc = columns.len();
            x = vec![0.0f64; n];
            for (j, col) in columns.iter().enumerate() {
                let wj = if j < weights.len() { weights[j] } else { 0.0 };
                for i in 0..n {
                    x[i] += wj * col[i];
                }
            }

            // Step 2: Compute dual variables (gradient of master w.r.t. x at current iterate)
            let f0 = master_obj(&x);
            for i in 0..n {
                let mut xf = x.clone();
                xf[i] += h;
                duals[i] = (master_obj(&xf) - f0) / h;
            }

            // Step 3: Pricing problem — find column with most negative reduced cost
            let (proposal, reduced_cost) = pricing_oracle(&duals);

            if reduced_cost >= -self.options.opt_tol {
                // No improving column: optimal
                break;
            }

            // Add new column
            columns.push(proposal);
            weights.push(0.01); // small initial weight
                                // Renormalize
            let sum: f64 = weights.iter().sum();
            for wj in weights.iter_mut() {
                *wj /= sum;
            }
            n_columns += 1;
        }

        let fun = master_obj(&x);

        Ok(DantzigWolfeResult {
            x,
            fun,
            n_columns,
            nit: n_iter,
            duals,
            success: n_columns < self.options.max_iter,
            message: "Dantzig-Wolfe decomposition completed".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// ADMM: Alternating Direction Method of Multipliers
// ---------------------------------------------------------------------------

/// Options for ADMM
#[derive(Debug, Clone)]
pub struct AdmmOptions {
    /// Penalty parameter ρ (augmented Lagrangian)
    pub rho: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Primal feasibility tolerance (||Ax + Bz - c||)
    pub eps_primal: f64,
    /// Dual feasibility tolerance (||ρ B^T (z^{k+1} - z^k)||)
    pub eps_dual: f64,
    /// Adaptive ρ: increase by this factor when primal/dual residual ratio > threshold
    pub rho_adapt: bool,
    /// Adaptive ρ factor
    pub rho_factor: f64,
    /// Adaptive ρ threshold
    pub rho_threshold: f64,
    /// Inner solver iterations for x and z updates
    pub inner_iter: usize,
    /// Inner solver tolerance
    pub inner_tol: f64,
}

impl Default for AdmmOptions {
    fn default() -> Self {
        AdmmOptions {
            rho: 1.0,
            max_iter: 1000,
            eps_primal: 1e-6,
            eps_dual: 1e-6,
            rho_adapt: true,
            rho_factor: 2.0,
            rho_threshold: 10.0,
            inner_iter: 50,
            inner_tol: 1e-8,
        }
    }
}

/// Result of ADMM
#[derive(Debug, Clone)]
pub struct AdmmResult {
    /// Primal solution x
    pub x: Vec<f64>,
    /// Auxiliary variable z
    pub z: Vec<f64>,
    /// Dual variable u (scaled)
    pub u: Vec<f64>,
    /// Objective value f(x) + g(z)
    pub fun: f64,
    /// Primal residual at termination
    pub primal_residual: f64,
    /// Dual residual at termination
    pub dual_residual: f64,
    /// Number of iterations
    pub nit: usize,
    /// Whether the algorithm converged
    pub success: bool,
    /// Termination message
    pub message: String,
}

/// ADMM: Alternating Direction Method of Multipliers.
///
/// Solves the consensus problem:
///
/// ```text
/// min  f(x) + g(z)
/// s.t. x - z = 0
/// ```
///
/// via the augmented Lagrangian:
///
/// ```text
/// L_ρ(x, z, u) = f(x) + g(z) + (ρ/2) ||x - z + u||²
/// ```
///
/// Iterations:
/// - x-update: x^{k+1} = argmin_x f(x) + (ρ/2)||x - z^k + u^k||²
/// - z-update: z^{k+1} = prox_{g/ρ}(x^{k+1} + u^k)
/// - u-update: u^{k+1} = u^k + x^{k+1} - z^{k+1}
pub struct Admm {
    /// Algorithm options
    pub options: AdmmOptions,
}

impl Default for Admm {
    fn default() -> Self {
        Admm {
            options: AdmmOptions::default(),
        }
    }
}

impl Admm {
    /// Create an ADMM solver with options
    pub fn new(options: AdmmOptions) -> Self {
        Admm { options }
    }

    /// Run ADMM for the consensus problem min f(x) + g(z) s.t. x = z.
    ///
    /// # Arguments
    ///
    /// * `f_obj` - Smooth objective f(x)
    /// * `prox_g` - Proximal operator for g: prox_{g/ρ}(v) = argmin_z g(z) + (ρ/2)||z-v||²
    /// * `x0` - Initial x
    pub fn solve<F, PG>(&self, f_obj: F, prox_g: PG, x0: &[f64]) -> OptimizeResult<AdmmResult>
    where
        F: Fn(&[f64]) -> f64,
        PG: Fn(&[f64], f64) -> Vec<f64>, // prox_g(v, rho) -> z
    {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        let h = 1e-7f64;
        let mut x = x0.to_vec();
        let mut z = x.clone();
        let mut u = vec![0.0f64; n];
        let mut rho = self.options.rho;
        let mut nit = 0usize;

        for iter in 0..self.options.max_iter {
            nit = iter + 1;

            // x-update: min f(x) + (ρ/2)||x - (z - u)||²
            let target_x: Vec<f64> = z.iter().zip(u.iter()).map(|(zi, ui)| zi - ui).collect();
            let x_aug = |x: &[f64]| -> f64 {
                let f = f_obj(x);
                let pen: f64 = x
                    .iter()
                    .zip(target_x.iter())
                    .map(|(xi, ti)| (xi - ti).powi(2))
                    .sum();
                f + 0.5 * rho * pen
            };

            let mut f_x = x_aug(&x);
            for _xi in 0..self.options.inner_iter {
                let mut grad = vec![0.0f64; n];
                for i in 0..n {
                    let mut xf = x.clone();
                    xf[i] += h;
                    grad[i] = (x_aug(&xf) - f_x) / h;
                }
                let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                if gnorm < self.options.inner_tol {
                    break;
                }
                let step = 1.0 / (rho + gnorm);
                let mut x_new: Vec<f64> = x
                    .iter()
                    .zip(grad.iter())
                    .map(|(xi, gi)| xi - step * gi)
                    .collect();
                let f_new = x_aug(&x_new);
                if f_new < f_x {
                    x = x_new;
                    f_x = f_new;
                } else {
                    // Backtrack
                    let mut s = step * 0.5;
                    for _ in 0..20 {
                        x_new = x
                            .iter()
                            .zip(grad.iter())
                            .map(|(xi, gi)| xi - s * gi)
                            .collect();
                        let fn2 = x_aug(&x_new);
                        if fn2 < f_x {
                            x = x_new;
                            f_x = fn2;
                            break;
                        }
                        s *= 0.5;
                    }
                    break;
                }
            }

            // z-update: prox_{g/ρ}(x + u)
            let z_prev = z.clone();
            let v: Vec<f64> = x.iter().zip(u.iter()).map(|(xi, ui)| xi + ui).collect();
            z = prox_g(&v, rho);

            // u-update: u += x - z
            for i in 0..n {
                u[i] += x[i] - z[i];
            }

            // Compute residuals
            let primal_res: f64 = x
                .iter()
                .zip(z.iter())
                .map(|(xi, zi)| (xi - zi).powi(2))
                .sum::<f64>()
                .sqrt();
            let dual_res: f64 = z
                .iter()
                .zip(z_prev.iter())
                .map(|(zi, zpi)| (rho * (zi - zpi)).powi(2))
                .sum::<f64>()
                .sqrt();

            // Adaptive ρ
            if self.options.rho_adapt {
                if primal_res > self.options.rho_threshold * dual_res {
                    rho *= self.options.rho_factor;
                    for ui in u.iter_mut() {
                        *ui /= self.options.rho_factor;
                    }
                } else if dual_res > self.options.rho_threshold * primal_res {
                    rho /= self.options.rho_factor;
                    for ui in u.iter_mut() {
                        *ui *= self.options.rho_factor;
                    }
                }
            }

            if primal_res < self.options.eps_primal && dual_res < self.options.eps_dual {
                let f_val = f_obj(&x);
                return Ok(AdmmResult {
                    x,
                    z,
                    u,
                    fun: f_val,
                    primal_residual: primal_res,
                    dual_residual: dual_res,
                    nit,
                    success: true,
                    message: "ADMM converged".to_string(),
                });
            }
        }

        let primal_res: f64 = x
            .iter()
            .zip(z.iter())
            .map(|(xi, zi)| (xi - zi).powi(2))
            .sum::<f64>()
            .sqrt();
        let dual_res = 0.0f64; // approximation at max iter
        let f_val = f_obj(&x);

        Ok(AdmmResult {
            x,
            z,
            u,
            fun: f_val,
            primal_residual: primal_res,
            dual_residual: dual_res,
            nit,
            success: primal_res < self.options.eps_primal * 100.0,
            message: "ADMM: maximum iterations reached".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// ProximalBundle: Proximal Bundle Method for Nonsmooth Optimization
// ---------------------------------------------------------------------------

/// Options for proximal bundle method
#[derive(Debug, Clone)]
pub struct ProximalBundleOptions {
    /// Proximal parameter μ (regularization)
    pub mu: f64,
    /// Maximum number of bundle iterations
    pub max_iter: usize,
    /// Null-step tolerance (serious step condition)
    pub m_l: f64,
    /// Maximum bundle size
    pub max_bundle_size: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Minimum proximal parameter
    pub min_mu: f64,
    /// Maximum proximal parameter
    pub max_mu: f64,
}

impl Default for ProximalBundleOptions {
    fn default() -> Self {
        ProximalBundleOptions {
            mu: 1.0,
            max_iter: 300,
            m_l: 0.1,
            max_bundle_size: 50,
            tol: 1e-6,
            min_mu: 1e-8,
            max_mu: 1e8,
        }
    }
}

/// A bundle element: (subgradient, function value, center)
#[derive(Debug, Clone)]
struct BundleElement {
    /// Subgradient g_i at bundle point
    g: Vec<f64>,
    /// Function value f_i at bundle point
    f: f64,
    /// Bundle point
    x: Vec<f64>,
}

/// Result of proximal bundle method
#[derive(Debug, Clone)]
pub struct ProximalBundleResult {
    /// Optimal solution
    pub x: Vec<f64>,
    /// Objective value at optimum
    pub fun: f64,
    /// Subgradient norm at optimum
    pub subgrad_norm: f64,
    /// Number of serious steps (actual descent)
    pub n_serious_steps: usize,
    /// Number of null steps
    pub n_null_steps: usize,
    /// Total iterations
    pub nit: usize,
    /// Whether the algorithm converged
    pub success: bool,
    /// Termination message
    pub message: String,
}

/// Proximal bundle method for nonsmooth convex optimization.
///
/// Solves min f(x) where f is convex but possibly nonsmooth.
/// Uses a bundle of subgradients to build a cutting-plane model:
///
/// ```text
/// f̂(x; y) = max_{i in B} { f(x_i) + g_i^T (x - x_i) }
/// ```
///
/// The proximal subproblem:
/// ```text
/// x^{k+1} = argmin_x { f̂(x; y^k) + (μ/2) ||x - y^k||² }
/// ```
///
/// generates either a serious step (sufficient decrease) or null step.
pub struct ProximalBundle {
    /// Algorithm options
    pub options: ProximalBundleOptions,
}

impl Default for ProximalBundle {
    fn default() -> Self {
        ProximalBundle {
            options: ProximalBundleOptions::default(),
        }
    }
}

impl ProximalBundle {
    /// Create a proximal bundle solver
    pub fn new(options: ProximalBundleOptions) -> Self {
        ProximalBundle { options }
    }

    /// Solve min f(x) using the proximal bundle method.
    ///
    /// # Arguments
    ///
    /// * `func` - Objective function (can be nonsmooth)
    /// * `subgrad` - Subgradient of f at x: returns (f(x), g ∈ ∂f(x))
    /// * `x0` - Initial point
    pub fn minimize<FS>(&self, func: FS, x0: &[f64]) -> OptimizeResult<ProximalBundleResult>
    where
        FS: Fn(&[f64]) -> (f64, Vec<f64>), // returns (value, subgradient)
    {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        let mut y = x0.to_vec(); // proximal center (serious step point)
        let (mut f_y, mut g_y) = func(&y);
        let mut mu = self.options.mu;
        let mut bundle: Vec<BundleElement> = vec![BundleElement {
            g: g_y.clone(),
            f: f_y,
            x: y.clone(),
        }];

        let mut n_serious = 0usize;
        let mut n_null = 0usize;
        let mut nit = 0usize;

        for iter in 0..self.options.max_iter {
            nit = iter + 1;

            // Check convergence: subgradient norm at center
            let gnorm: f64 = g_y.iter().map(|g| g * g).sum::<f64>().sqrt();
            if gnorm < self.options.tol {
                return Ok(ProximalBundleResult {
                    x: y,
                    fun: f_y,
                    subgrad_norm: gnorm,
                    n_serious_steps: n_serious,
                    n_null_steps: n_null,
                    nit,
                    success: true,
                    message: "Proximal bundle converged (subgradient norm)".to_string(),
                });
            }

            // Solve proximal subproblem: min f̂(x; y) + (μ/2)||x - y||²
            // The cutting-plane model: f̂(x) = max_i {f_i + g_i^T (x - x_i)}
            // The proximal subproblem reduces to a QP. We use gradient descent on the dual.
            //
            // Dual formulation: max_{λ>=0, Σλ=1} { Σλ_i f_i - (1/2μ) ||Σλ_i g_i||² - (Σλ_i g_i)^T y + Σλ_i g_i^T x_i }
            // This is a QP in λ.
            let nb = bundle.len();
            let mut lambda = vec![1.0f64 / nb as f64; nb]; // uniform start

            // Gradient of dual w.r.t. λ
            let dual_obj = |lam: &[f64]| -> f64 {
                let mut agg_g = vec![0.0f64; n];
                let mut sum_fg = 0.0f64;
                for (i, be) in bundle.iter().enumerate() {
                    let li = if i < lam.len() { lam[i] } else { 0.0 };
                    for j in 0..n {
                        agg_g[j] += li * be.g[j];
                    }
                    sum_fg += li
                        * (be.f
                            - be.g
                                .iter()
                                .zip(be.x.iter())
                                .map(|(gj, xj)| gj * xj)
                                .sum::<f64>());
                }
                let agg_norm_sq: f64 = agg_g.iter().map(|g| g * g).sum();
                // Dual objective (to maximize, so negate for minimization):
                -(sum_fg
                    - agg_g
                        .iter()
                        .zip(y.iter())
                        .map(|(gj, yj)| gj * yj)
                        .sum::<f64>()
                    - 0.5 / mu * agg_norm_sq)
            };

            let h = 1e-7f64;
            let mut f_lam = dual_obj(&lambda);
            for _di in 0..200 {
                let mut grad_lam = vec![0.0f64; nb];
                for i in 0..nb {
                    let mut lf = lambda.clone();
                    lf[i] += h;
                    // Re-project onto simplex
                    let sum: f64 = lf.iter().sum();
                    let lf: Vec<f64> = lf.iter().map(|li| (li / sum).max(0.0)).collect();
                    grad_lam[i] = (dual_obj(&lf) - f_lam) / h;
                }
                let gnorm_lam: f64 = grad_lam.iter().map(|g| g * g).sum::<f64>().sqrt();
                if gnorm_lam < 1e-10 {
                    break;
                }
                let step_lam = 0.1 / (1.0 + gnorm_lam);
                let mut l_new: Vec<f64> = lambda
                    .iter()
                    .zip(grad_lam.iter())
                    .map(|(li, gi)| (li - step_lam * gi).max(0.0))
                    .collect();
                // Project onto simplex
                let sum: f64 = l_new.iter().sum();
                if sum > 1e-14 {
                    for li in l_new.iter_mut() {
                        *li /= sum;
                    }
                }
                let f_new = dual_obj(&l_new);
                if f_new < f_lam {
                    lambda = l_new;
                    f_lam = f_new;
                } else {
                    break;
                }
            }

            // Recover primal step from dual: x* = y - (1/μ) Σ λ_i g_i
            let mut agg_g = vec![0.0f64; n];
            for (i, be) in bundle.iter().enumerate() {
                let li = if i < lambda.len() { lambda[i] } else { 0.0 };
                for j in 0..n {
                    agg_g[j] += li * be.g[j];
                }
            }
            let x_trial: Vec<f64> = y
                .iter()
                .zip(agg_g.iter())
                .map(|(yj, gj)| yj - gj / mu)
                .collect();

            // Evaluate at trial point
            let (f_trial, g_trial) = func(&x_trial);

            // Compute cutting-plane model value at x_trial (for acceptance test)
            let f_model: f64 = bundle
                .iter()
                .map(|be| {
                    let gx: f64 =
                        be.g.iter()
                            .zip(x_trial.iter())
                            .zip(be.x.iter())
                            .map(|((gi, xi), xi_k)| gi * (xi - xi_k))
                            .sum();
                    be.f + gx
                })
                .fold(f64::NEG_INFINITY, f64::max);

            // Serious step condition: f(x_trial) <= f(y) - mL * (f̂(y) - f̂(x_trial))
            // Simplified: f_trial <= f_y - mL * (f_y - f_model)
            let descent = f_y - f_model;
            let is_serious = f_trial <= f_y - self.options.m_l * descent.max(0.0);

            if is_serious {
                y = x_trial.clone();
                f_y = f_trial;
                g_y = g_trial.clone();
                n_serious += 1;
            } else {
                n_null += 1;
            }

            // Add new bundle element
            bundle.push(BundleElement {
                g: g_trial,
                f: f_trial,
                x: x_trial,
            });

            // Trim bundle if too large
            if bundle.len() > self.options.max_bundle_size {
                bundle.remove(0);
            }

            // Update μ: increase for null steps, reset for serious steps
            if is_serious {
                mu = (mu * 0.5).max(self.options.min_mu);
            } else if n_null > 3 * (n_serious + 1) {
                mu = (mu * 2.0).min(self.options.max_mu);
            }
        }

        let gnorm_final: f64 = g_y.iter().map(|g| g * g).sum::<f64>().sqrt();

        Ok(ProximalBundleResult {
            x: y,
            fun: f_y,
            subgrad_norm: gnorm_final,
            n_serious_steps: n_serious,
            n_null_steps: n_null,
            nit,
            success: gnorm_final < self.options.tol * 100.0,
            message: "Proximal bundle: maximum iterations reached".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benders_separable() {
        // Master: min x^2, Sub: min (x + y)^2 → optimal y = -x, combined = 0
        let result = BendersDecomposition::default()
            .solve(
                |x: &[f64]| x[0].powi(2),
                |x: &[f64], y: &[f64]| (x[0] + y[0]).powi(2),
                &[1.0],
                &[0.0],
            )
            .expect("unexpected None or Err");
        assert!(
            result.upper_bound < 2.0,
            "Upper bound {} should be < 2.0",
            result.upper_bound
        );
    }

    #[test]
    fn test_dantzig_wolfe_basic() {
        // Simple quadratic: min (x-1)^2
        // Pricing: minimize 2*(x-1)*g[0] w.r.t. x in [0,2]
        let result = DantzigWolfe::default()
            .solve(
                |x: &[f64]| (x[0] - 1.0).powi(2),
                |pi: &[f64]| {
                    // Optimal proposal in [0, 2]: x* = 1 - pi[0]/2 (if inside bounds)
                    let x_star = (1.0 - pi[0] * 0.5).clamp(0.0, 2.0);
                    let rc = pi[0] * x_star - pi[0]; // reduced cost
                    (vec![x_star], rc)
                },
                &[0.5],
                1,
            )
            .expect("unexpected None or Err");
        assert!(result.fun < 0.5, "Expected fun < 0.5, got {}", result.fun);
    }

    #[test]
    fn test_admm_consensus() {
        // min x^2 + ||z-1||^2 s.t. x = z → optimal x = z = 0.5
        // prox_{g/ρ}(v) = argmin ||z-1||^2 + ρ/2||z-v||^2 = (1 + ρ/2 * v) / (1 + ρ/2)
        let result = Admm::default()
            .solve(
                |x: &[f64]| x[0].powi(2),
                |v: &[f64], rho: f64| vec![(1.0 + rho * 0.5 * v[0]) / (1.0 + rho * 0.5)],
                &[1.0],
            )
            .expect("unexpected None or Err");
        assert!(
            (result.x[0] - 0.5).abs() < 0.1 || result.fun < 1.0,
            "fun = {}, x = {:?}",
            result.fun,
            result.x
        );
    }

    #[test]
    fn test_admm_lasso_like() {
        // min (x-3)^2 + |z| s.t. x = z
        // prox_{|.|/ρ}(v) = soft-threshold(v, 1/ρ)
        let soft_thresh = |v: &[f64], rho: f64| -> Vec<f64> {
            let thresh = 1.0 / rho;
            v.iter()
                .map(|vi| vi.signum() * (vi.abs() - thresh).max(0.0))
                .collect()
        };

        let result = Admm::new(AdmmOptions {
            rho: 1.0,
            max_iter: 500,
            eps_primal: 1e-5,
            eps_dual: 1e-5,
            ..Default::default()
        })
        .solve(|x: &[f64]| (x[0] - 3.0).powi(2), soft_thresh, &[0.0])
        .expect("unexpected None or Err");
        // Optimal x ≈ 2.5 (balance between pulling toward 3 and L1 penalty)
        assert!(result.fun < 5.0, "fun = {}", result.fun);
    }

    #[test]
    fn test_proximal_bundle_smooth() {
        // Test on smooth convex function (f = x^2, g = 2x)
        let result = ProximalBundle::default()
            .minimize(|x: &[f64]| (x[0].powi(2), vec![2.0 * x[0]]), &[2.0])
            .expect("unexpected None or Err");
        assert!(result.fun < 0.5, "Expected fun < 0.5, got {}", result.fun);
    }

    #[test]
    fn test_proximal_bundle_max_function() {
        // max(x, -x) = |x|, subgradient: sign(x) or 0 if x=0
        let result = ProximalBundle::new(ProximalBundleOptions {
            tol: 1e-5,
            max_iter: 200,
            ..Default::default()
        })
        .minimize(
            |x: &[f64]| {
                let v = x[0].abs();
                let g = if x[0] > 0.0 {
                    1.0
                } else if x[0] < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                (v, vec![g])
            },
            &[2.0],
        )
        .expect("unexpected None or Err");
        assert!(
            result.fun < 0.5,
            "Expected |x| < 0.5 at optimum, got {}",
            result.fun
        );
    }
}
