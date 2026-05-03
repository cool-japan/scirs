//! Newton-CG (Inexact Newton method with CG inner solver) for unconstrained optimization
//!
//! This module implements the Newton-CG method, which solves the Newton equation
//! `H * p = -g` inexactly using the Conjugate Gradient (CG) method. This avoids
//! forming the full Hessian by instead using Hessian-vector products (HVPs),
//! which can be computed efficiently via finite differences or automatic differentiation.
//!
//! ## References
//!
//! - Nocedal & Wright, "Numerical Optimization", 2nd ed., §7.1 (inexact Newton methods)
//! - Nash, "A survey of truncated-Newton methods", 2000
//! - Steihaug, "The conjugate gradient method and trust regions in large scale optimization", 1983

use crate::error::OptimizeError;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::finite_difference_gradient;
use crate::unconstrained::Bounds;
use scirs2_core::ndarray::{Array1, ArrayView1};

// ─── Traits ──────────────────────────────────────────────────────────────────

/// Trait for Hessian-vector product oracles.
///
/// Implementors provide the product `H(x) * v` for a given point `x` and
/// direction vector `v`, without ever materialising the full Hessian matrix.
pub trait HVPFunction: Send + Sync {
    /// Compute the Hessian-vector product `H(x) * v`.
    ///
    /// # Arguments
    /// * `x` – current iterate (length `n`)
    /// * `v` – direction vector (length `n`)
    ///
    /// # Returns
    /// `H(x) * v` as a `Vec<f64>` of length `n`.
    fn hvp(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, OptimizeError>;
}

// ─── Finite-difference HVP ───────────────────────────────────────────────────

/// Finite-difference Hessian-vector product approximation.
///
/// Uses the second-order central difference formula:
/// ```text
/// H(x) * v ≈ (∇f(x + ε v) - ∇f(x - ε v)) / (2 ε)
/// ```
/// which requires two gradient evaluations per HVP call and has error O(ε²).
pub struct FiniteDiffHVP<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Send + Sync,
{
    /// Objective function
    pub fun: F,
    /// Finite-difference step size (default: √(machine epsilon) ≈ 1.49e-8)
    pub step: f64,
}

impl<F> FiniteDiffHVP<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Send + Sync,
{
    /// Create a new finite-difference HVP oracle.
    pub fn new(fun: F, step: f64) -> Self {
        Self { fun, step }
    }

    /// Create with the default step size `sqrt(eps_mach)`.
    pub fn with_default_step(fun: F) -> Self {
        Self {
            fun,
            step: f64::EPSILON.sqrt(),
        }
    }
}

impl<F> HVPFunction for FiniteDiffHVP<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Send + Sync,
{
    fn hvp(&self, x: &[f64], v: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let n = x.len();
        if v.len() != n {
            return Err(OptimizeError::ValueError(
                "x and v must have the same length".to_string(),
            ));
        }

        // Normalise v to get a unit direction; scale result back afterwards.
        let v_norm = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
        if v_norm < 1e-300 {
            return Ok(vec![0.0; n]);
        }

        // ε proportional to ‖x‖ for scale-invariance.
        // The outer perturbation must be large enough so that the amplification
        // of inner-gradient rounding errors (≈ eps_mach / (2·eps)) stays below
        // the desired HVP tolerance. Using eps_mach^{1/3} (≈ 6e-6) introduces
        // rounding-dominated errors of O(eps_mach / eps) ≈ 7e-4 for x≈1.
        // A value of 1e-3·‖x‖ gives truncation error O(eps²) ≈ 1e-6 for smooth f
        // while keeping rounding amplification at O(eps_mach / (1e-3)) ≈ 2e-13.
        let x_norm = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt().max(1.0);
        let outer_step = 1e-3; // large enough to avoid rounding amplification
        let eps = outer_step * x_norm;

        let mut x_plus = Array1::from(x.to_vec());
        let mut x_minus = Array1::from(x.to_vec());
        for i in 0..n {
            x_plus[i] += eps * v[i] / v_norm;
            x_minus[i] -= eps * v[i] / v_norm;
        }

        // Compute directional gradient via finite differences on f
        // H v ≈ (g(x + ε v̂) - g(x - ε v̂)) / (2ε)  ← uses gradient differences
        // But here we do it directly on f via the second-order central diff:
        // (d²f/dα²)|_{α=0} in direction v̂ ≈ (f(x+εv̂) - 2f(x) + f(x-εv̂)) / ε²
        // We can recover H v by evaluating directional 2nd derivative but that
        // only gives a scalar. For the full HVP we use the gradient-difference formula.

        // Gradient at x + ε v̂
        let grad_plus = finite_diff_grad_internal(&self.fun, &x_plus.view(), self.step)?;
        // Gradient at x - ε v̂
        let grad_minus = finite_diff_grad_internal(&self.fun, &x_minus.view(), self.step)?;

        let hvp: Vec<f64> = grad_plus
            .iter()
            .zip(grad_minus.iter())
            .map(|(gp, gm)| v_norm * (gp - gm) / (2.0 * eps))
            .collect();

        Ok(hvp)
    }
}

/// Internal gradient computation using central differences.
fn finite_diff_grad_internal<F>(
    fun: &F,
    x: &ArrayView1<f64>,
    step: f64,
) -> Result<Vec<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut x_tmp = x.to_owned();

    for i in 0..n {
        let h = step * (1.0 + x[i].abs());
        let orig = x_tmp[i];

        x_tmp[i] = orig + h;
        let fp = fun(&x_tmp.view());
        x_tmp[i] = orig - h;
        let fm = fun(&x_tmp.view());
        x_tmp[i] = orig;

        if !fp.is_finite() || !fm.is_finite() {
            return Err(OptimizeError::ComputationError(
                "Non-finite function value in HVP gradient computation".to_string(),
            ));
        }
        grad[i] = (fp - fm) / (2.0 * h);
    }

    Ok(grad)
}

// ─── Truncated Conjugate Gradient ────────────────────────────────────────────

/// Options for the Truncated CG solver used inside Newton-CG.
#[derive(Debug, Clone)]
pub struct TruncatedCGOptions {
    /// Forcing term η for the relative residual stopping criterion:
    /// stop when ‖r‖ ≤ η ‖g‖. Typical values: 1e-4 … 0.5.
    pub eta: f64,
    /// Maximum number of CG iterations (default: n, the problem dimension)
    pub max_iter: Option<usize>,
    /// Minimum step-length threshold; terminate if ‖p‖ < threshold (stagnation guard)
    pub min_step_norm: f64,
    /// Whether to use Steihaug's trust-region CG termination (negative curvature detection)
    pub use_negative_curvature_detection: bool,
    /// Trust-region radius (only meaningful when `use_negative_curvature_detection = true`)
    pub trust_radius: f64,
}

impl Default for TruncatedCGOptions {
    fn default() -> Self {
        Self {
            // eta = 0.5 stops CG after 1 iteration for well-scaled problems because
            // the relative residual tolerance (eta * ‖g‖) is met immediately.
            // Using eta = 0.1 allows CG to reach the exact Newton direction in the
            // typical case where n ≤ 10, and still terminates early for large systems.
            eta: 0.1,
            max_iter: None,
            min_step_norm: 1e-12,
            use_negative_curvature_detection: true,
            trust_radius: 1e10,
        }
    }
}

/// Result from `TruncatedCG::solve`.
#[derive(Debug, Clone)]
pub struct TruncatedCGResult {
    /// The Newton step direction (solution to H p = -g)
    pub step: Array1<f64>,
    /// Final residual norm
    pub residual_norm: f64,
    /// Number of HVP matrix-vector products performed
    pub num_hvp: usize,
    /// Whether CG converged to requested tolerance
    pub converged: bool,
    /// Whether negative curvature was encountered
    pub hit_negative_curvature: bool,
}

/// Truncated Conjugate Gradient solver for the Newton equation `H p = -g`.
///
/// Implements Steihaug's algorithm (Algorithm 7.2 in Nocedal & Wright) which
/// additionally monitors for negative curvature and trust-region boundary hits.
pub struct TruncatedCG {
    /// Configuration
    pub options: TruncatedCGOptions,
}

impl TruncatedCG {
    /// Create with given options.
    pub fn new(options: TruncatedCGOptions) -> Self {
        Self { options }
    }

    /// Create with defaults.
    pub fn default_solver() -> Self {
        Self {
            options: TruncatedCGOptions::default(),
        }
    }

    /// Solve `H p = -g` approximately using CG.
    ///
    /// The HVP oracle `hvp_fn` provides `H * v` without materialising `H`.
    ///
    /// # Arguments
    /// * `x`      – current iterate (needed by the HVP oracle)
    /// * `g`      – gradient at `x`
    /// * `hvp_fn` – HVP oracle
    pub fn solve<H: HVPFunction>(
        &self,
        x: &[f64],
        g: &[f64],
        hvp_fn: &H,
    ) -> Result<TruncatedCGResult, OptimizeError> {
        let n = g.len();
        let max_iter = self.options.max_iter.unwrap_or(n.max(1));
        let g_norm = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();

        // Trivial case
        if g_norm < 1e-300 {
            return Ok(TruncatedCGResult {
                step: Array1::zeros(n),
                residual_norm: 0.0,
                num_hvp: 0,
                converged: true,
                hit_negative_curvature: false,
            });
        }

        let tol = self.options.eta * g_norm;
        let delta = self.options.trust_radius;

        // Initialise: p = 0, r = -g (residual of H p + g = 0), d = r
        let mut p = vec![0.0f64; n];
        let mut r: Vec<f64> = g.iter().map(|gi| -gi).collect();
        let mut d = r.clone(); // search direction = residual (steepest descent)

        let mut r_dot_r: f64 = r.iter().map(|ri| ri * ri).sum();
        let mut num_hvp = 0;
        let mut hit_neg_curv = false;

        for _ in 0..max_iter {
            // Compute H * d
            let hd = hvp_fn.hvp(x, &d)?;
            num_hvp += 1;

            // Curvature: d^T H d
            let d_hd: f64 = d.iter().zip(hd.iter()).map(|(di, hdi)| di * hdi).sum();

            if self.options.use_negative_curvature_detection && d_hd <= 0.0 {
                // Negative or zero curvature – return along d towards trust boundary
                hit_neg_curv = true;
                if delta < 1e10 {
                    // Step to trust-region boundary in direction d
                    let d_norm_sq: f64 = d.iter().map(|di| di * di).sum();
                    let p_dot_d: f64 = p.iter().zip(d.iter()).map(|(pi, di)| pi * di).sum();
                    let p_norm_sq: f64 = p.iter().map(|pi| pi * pi).sum();
                    // solve ‖p + τ d‖ = δ for τ ≥ 0
                    let disc = p_dot_d * p_dot_d + d_norm_sq * (delta * delta - p_norm_sq);
                    let tau = if disc >= 0.0 {
                        (-p_dot_d + disc.sqrt()) / d_norm_sq.max(1e-300)
                    } else {
                        0.0
                    };
                    for i in 0..n {
                        p[i] += tau * d[i];
                    }
                }
                break;
            }

            let alpha = r_dot_r / d_hd.max(1e-300);

            // Tentative new iterate
            let p_new: Vec<f64> = p
                .iter()
                .zip(d.iter())
                .map(|(pi, di)| pi + alpha * di)
                .collect();

            // Trust-region boundary check
            let p_new_norm_sq: f64 = p_new.iter().map(|pi| pi * pi).sum();
            if self.options.use_negative_curvature_detection && p_new_norm_sq >= delta * delta {
                // Step to trust boundary
                let d_norm_sq: f64 = d.iter().map(|di| di * di).sum();
                let p_dot_d: f64 = p.iter().zip(d.iter()).map(|(pi, di)| pi * di).sum();
                let p_norm_sq: f64 = p.iter().map(|pi| pi * pi).sum();
                let disc = p_dot_d * p_dot_d + d_norm_sq * (delta * delta - p_norm_sq);
                let tau = if disc >= 0.0 {
                    (-p_dot_d + disc.sqrt()) / d_norm_sq.max(1e-300)
                } else {
                    0.0
                };
                for i in 0..n {
                    p[i] += tau * d[i];
                }
                break;
            }

            p = p_new;

            // Update residual: r_new = r - alpha * H d
            let mut r_new = vec![0.0f64; n];
            for i in 0..n {
                r_new[i] = r[i] - alpha * hd[i];
            }

            let r_new_dot_r_new: f64 = r_new.iter().map(|ri| ri * ri).sum();
            let r_norm = r_new_dot_r_new.sqrt();

            if r_norm <= tol {
                r = r_new;
                r_dot_r = r_new_dot_r_new;
                break;
            }

            let beta = r_new_dot_r_new / r_dot_r.max(1e-300);
            for i in 0..n {
                d[i] = r_new[i] + beta * d[i];
            }
            r = r_new;
            r_dot_r = r_new_dot_r_new;
        }

        let residual_norm = r.iter().map(|ri| ri * ri).sum::<f64>().sqrt();
        let converged = residual_norm <= tol || !hit_neg_curv;

        Ok(TruncatedCGResult {
            step: Array1::from(p),
            residual_norm,
            num_hvp,
            converged,
            hit_negative_curvature: hit_neg_curv,
        })
    }
}

// ─── NewtonCG convergence result ─────────────────────────────────────────────

/// Convergence result returned by `NewtonCG::minimize`.
#[derive(Debug, Clone)]
pub struct NewtonCGResult {
    /// Solution vector
    pub x: Array1<f64>,
    /// Objective function value at solution
    pub f_val: f64,
    /// Gradient at solution
    pub gradient: Array1<f64>,
    /// Gradient norm at each outer iteration (convergence history)
    pub grad_norm_history: Vec<f64>,
    /// Number of outer Newton iterations
    pub n_iter: usize,
    /// Total number of function evaluations
    pub n_fev: usize,
    /// Total number of HVP evaluations
    pub n_hvp: usize,
    /// Whether the method converged
    pub converged: bool,
    /// Termination message
    pub message: String,
}

// ─── NewtonCG struct ─────────────────────────────────────────────────────────

/// Options for the Newton-CG solver.
#[derive(Debug, Clone)]
pub struct NewtonCGOptions {
    /// Gradient tolerance for convergence (outer loop)
    pub gtol: f64,
    /// Function value tolerance for convergence (outer loop)
    pub ftol: f64,
    /// Maximum number of outer Newton iterations
    pub max_iter: usize,
    /// Finite-difference step size for gradient/HVP computation
    pub eps: f64,
    /// Options forwarded to the inner Truncated CG solver
    pub cg_options: TruncatedCGOptions,
    /// Whether to use a line search after each Newton step
    pub use_line_search: bool,
    /// Line-search Armijo sufficient-decrease constant
    pub c1: f64,
    /// Backtracking reduction factor
    pub backtrack_rho: f64,
    /// Box bounds (optional)
    pub bounds: Option<Bounds>,
}

impl Default for NewtonCGOptions {
    fn default() -> Self {
        Self {
            gtol: 1e-5,
            ftol: 1e-8,
            max_iter: 200,
            eps: f64::EPSILON.sqrt(),
            cg_options: TruncatedCGOptions::default(),
            use_line_search: true,
            c1: 1e-4,
            backtrack_rho: 0.5,
            bounds: None,
        }
    }
}

/// Inexact Newton method with Conjugate Gradient inner solver.
///
/// At each outer iteration the Newton equation
/// ```text
/// H(xₖ) pₖ = -∇f(xₖ)
/// ```
/// is solved approximately by the Truncated CG method. The resulting step is
/// then accepted or rejected via a backtracking line search.
pub struct NewtonCG {
    /// Configuration
    pub options: NewtonCGOptions,
}

impl NewtonCG {
    /// Create with given options.
    pub fn new(options: NewtonCGOptions) -> Self {
        Self { options }
    }

    /// Create with defaults.
    pub fn default_solver() -> Self {
        Self {
            options: NewtonCGOptions::default(),
        }
    }

    /// Minimise a function using the Newton-CG method.
    ///
    /// # Arguments
    /// * `fun`    – objective function `f: ℝⁿ → ℝ`
    /// * `x0`     – initial iterate
    /// * `hvp_fn` – HVP oracle (can be a `FiniteDiffHVP`)
    pub fn minimize<F, H>(
        &self,
        mut fun: F,
        x0: &[f64],
        hvp_fn: &H,
    ) -> Result<NewtonCGResult, OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> f64,
        H: HVPFunction,
    {
        let n = x0.len();
        let opts = &self.options;
        let cg_solver = TruncatedCG::new(opts.cg_options.clone());

        let mut x = Array1::from(x0.to_vec());
        if let Some(ref b) = opts.bounds {
            b.project(x.as_slice_mut().ok_or_else(|| {
                OptimizeError::ComputationError("Cannot get mutable slice".to_string())
            })?);
        }

        let mut n_fev = 0usize;
        let mut n_hvp = 0usize;

        let mut f = {
            n_fev += 1;
            fun(&x.view())
        };

        let mut g = {
            let grad = finite_difference_gradient(&mut fun, &x.view(), opts.eps)?;
            n_fev += 2 * n;
            grad
        };

        let mut grad_norm_history = Vec::new();
        let mut iter = 0usize;
        let mut converged = false;
        let mut message = "Maximum iterations reached".to_string();

        loop {
            let g_norm = g.dot(&g).sqrt();
            grad_norm_history.push(g_norm);

            if g_norm <= opts.gtol {
                converged = true;
                message = "Gradient norm below tolerance".to_string();
                break;
            }

            if iter >= opts.max_iter {
                break;
            }

            // Solve Newton equation with Truncated CG
            let x_slice = x.as_slice().ok_or_else(|| {
                OptimizeError::ComputationError("Cannot get slice from x".to_string())
            })?;
            let g_slice = g.as_slice().ok_or_else(|| {
                OptimizeError::ComputationError("Cannot get slice from g".to_string())
            })?;

            let cg_result = cg_solver.solve(x_slice, g_slice, hvp_fn)?;
            n_hvp += cg_result.num_hvp;

            let mut step = cg_result.step;

            // Project step to respect bounds (if any)
            if let Some(ref b) = opts.bounds {
                project_step_bounds(&mut step, &x, b);
            }

            // Determine step length via backtracking line search
            let step_norm = step.dot(&step).sqrt();
            if step_norm < opts.cg_options.min_step_norm {
                converged = true;
                message = "Step size too small, converged".to_string();
                break;
            }

            let slope = g.dot(&step);
            let alpha = if opts.use_line_search {
                backtrack_line_search(
                    &mut fun,
                    &x,
                    &step,
                    f,
                    slope,
                    opts.c1,
                    opts.backtrack_rho,
                    opts.bounds.as_ref(),
                    &mut n_fev,
                )
            } else {
                1.0
            };

            // Update iterate
            let mut x_new = &x + &(alpha * &step);
            if let Some(ref b) = opts.bounds {
                b.project(x_new.as_slice_mut().ok_or_else(|| {
                    OptimizeError::ComputationError("Cannot get mutable slice".to_string())
                })?);
            }

            let f_new = {
                n_fev += 1;
                fun(&x_new.view())
            };

            let g_new = {
                let grad = finite_difference_gradient(&mut fun, &x_new.view(), opts.eps)?;
                n_fev += 2 * n;
                grad
            };

            // Check function-value convergence
            if (f - f_new).abs() < opts.ftol * (1.0 + f.abs()) {
                x = x_new;
                f = f_new;
                g = g_new;
                converged = true;
                message = "Function value change below tolerance".to_string();
                break;
            }

            x = x_new;
            f = f_new;
            g = g_new;
            iter += 1;
        }

        Ok(NewtonCGResult {
            x,
            f_val: f,
            gradient: g,
            grad_norm_history,
            n_iter: iter,
            n_fev,
            n_hvp,
            converged,
            message,
        })
    }
}

// ─── DampedNewton ─────────────────────────────────────────────────────────────

/// Newton method with step damping and a backtracking line search.
///
/// Unlike `NewtonCG`, this variant assembles the **full finite-difference Hessian**
/// and solves the linear system exactly (via Cholesky with a diagonal regularisation
/// fall-back). It is therefore only suitable for small-to-medium problems.
pub struct DampedNewton {
    /// Convergence tolerance on the gradient norm
    pub gtol: f64,
    /// Maximum outer iterations
    pub max_iter: usize,
    /// Finite-difference step for Hessian and gradient
    pub eps: f64,
    /// Levenberg–Marquardt regularisation: add λI to the Hessian when it is
    /// not sufficiently positive definite (λ starts here)
    pub lambda_init: f64,
    /// Armijo constant for backtracking line search
    pub c1: f64,
    /// Backtracking reduction factor
    pub rho: f64,
    /// Box bounds
    pub bounds: Option<Bounds>,
}

impl Default for DampedNewton {
    fn default() -> Self {
        Self {
            gtol: 1e-5,
            max_iter: 200,
            eps: f64::EPSILON.sqrt(),
            lambda_init: 1e-4,
            c1: 1e-4,
            rho: 0.5,
            bounds: None,
        }
    }
}

impl DampedNewton {
    /// Minimise a function.
    pub fn minimize<F>(&self, mut fun: F, x0: &[f64]) -> Result<NewtonCGResult, OptimizeError>
    where
        F: FnMut(&ArrayView1<f64>) -> f64,
    {
        let n = x0.len();
        let mut x = Array1::from(x0.to_vec());

        if let Some(ref b) = self.bounds {
            b.project(x.as_slice_mut().ok_or_else(|| {
                OptimizeError::ComputationError("Cannot get mutable slice".to_string())
            })?);
        }

        let mut n_fev = 0usize;
        let mut f = {
            n_fev += 1;
            fun(&x.view())
        };
        let mut g = {
            let grad = finite_difference_gradient(&mut fun, &x.view(), self.eps)?;
            n_fev += 2 * n;
            grad
        };

        let mut grad_norm_history = Vec::new();
        let mut iter = 0usize;
        let mut converged = false;
        let mut message = "Maximum iterations reached".to_string();

        while iter < self.max_iter {
            let g_norm = g.dot(&g).sqrt();
            grad_norm_history.push(g_norm);
            if g_norm <= self.gtol {
                converged = true;
                message = "Gradient norm below tolerance".to_string();
                break;
            }

            // Build finite-difference Hessian
            let hess = build_finite_diff_hessian(&mut fun, &x.view(), self.eps, &mut n_fev)?;

            // Regularised Cholesky solve: (H + λI) p = -g
            let mut step = regularised_solve(&hess, g.as_slice().unwrap_or(&[]), self.lambda_init);

            if let Some(ref b) = self.bounds {
                project_step_bounds(&mut step, &x, b);
            }

            // Backtracking line search
            let slope = g.dot(&step);
            let alpha = backtrack_line_search(
                &mut fun,
                &x,
                &step,
                f,
                slope,
                self.c1,
                self.rho,
                self.bounds.as_ref(),
                &mut n_fev,
            );

            let mut x_new = &x + &(alpha * &step);
            if let Some(ref b) = self.bounds {
                b.project(x_new.as_slice_mut().ok_or_else(|| {
                    OptimizeError::ComputationError("Cannot get mutable slice".to_string())
                })?);
            }

            let f_new = {
                n_fev += 1;
                fun(&x_new.view())
            };
            let g_new = {
                let grad = finite_difference_gradient(&mut fun, &x_new.view(), self.eps)?;
                n_fev += 2 * n;
                grad
            };

            x = x_new;
            f = f_new;
            g = g_new;
            iter += 1;
        }

        Ok(NewtonCGResult {
            x,
            f_val: f,
            gradient: g,
            grad_norm_history,
            n_iter: iter,
            n_fev,
            n_hvp: 0,
            converged,
            message,
        })
    }
}

// ─── Helper functions ─────────────────────────────────────────────────────────

/// Backtracking Armijo line search (shared by Newton-CG and DampedNewton).
fn backtrack_line_search<F>(
    fun: &mut F,
    x: &Array1<f64>,
    step: &Array1<f64>,
    f0: f64,
    slope: f64,
    c1: f64,
    rho: f64,
    bounds: Option<&Bounds>,
    n_fev: &mut usize,
) -> f64
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    // If slope is non-negative we have a bad direction; return a tiny step
    if slope >= 0.0 {
        return 1e-14;
    }

    let mut alpha = 1.0;
    let max_steps = 60;

    for _ in 0..max_steps {
        let mut x_trial = x + alpha * step;
        if let Some(b) = bounds {
            if let Some(s) = x_trial.as_slice_mut() {
                b.project(s);
            }
        }
        *n_fev += 1;
        let f_trial = fun(&x_trial.view());

        if f_trial <= f0 + c1 * alpha * slope {
            return alpha;
        }
        alpha *= rho;
        if alpha < 1e-14 {
            break;
        }
    }
    alpha
}

/// Project a step direction so that it respects box constraints at the boundary.
fn project_step_bounds(step: &mut Array1<f64>, x: &Array1<f64>, bounds: &Bounds) {
    for i in 0..x.len() {
        if let Some(lb) = bounds.lower[i] {
            if (x[i] - lb).abs() < 1e-12 && step[i] < 0.0 {
                step[i] = 0.0;
            }
        }
        if let Some(ub) = bounds.upper[i] {
            if (x[i] - ub).abs() < 1e-12 && step[i] > 0.0 {
                step[i] = 0.0;
            }
        }
    }
}

/// Build a finite-difference Hessian (for `DampedNewton`).
fn build_finite_diff_hessian<F>(
    fun: &mut F,
    x: &ArrayView1<f64>,
    step: f64,
    n_fev: &mut usize,
) -> Result<Vec<Vec<f64>>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
{
    let n = x.len();
    let mut hess = vec![vec![0.0f64; n]; n];
    let mut x_tmp = x.to_owned();

    *n_fev += 1;
    let f0 = fun(&x.view());

    for i in 0..n {
        let hi = step * (1.0 + x[i].abs());

        x_tmp[i] = x[i] + hi;
        *n_fev += 1;
        let fp = fun(&x_tmp.view());
        x_tmp[i] = x[i] - hi;
        *n_fev += 1;
        let fm = fun(&x_tmp.view());
        x_tmp[i] = x[i];

        if !fp.is_finite() || !fm.is_finite() {
            return Err(OptimizeError::ComputationError(
                "Non-finite value during Hessian computation".to_string(),
            ));
        }
        hess[i][i] = (fp - 2.0 * f0 + fm) / (hi * hi);

        for j in (i + 1)..n {
            let hj = step * (1.0 + x[j].abs());

            x_tmp[i] = x[i] + hi;
            x_tmp[j] = x[j] + hj;
            *n_fev += 1;
            let fpp = fun(&x_tmp.view());

            x_tmp[i] = x[i] + hi;
            x_tmp[j] = x[j] - hj;
            *n_fev += 1;
            let fpm = fun(&x_tmp.view());

            x_tmp[i] = x[i] - hi;
            x_tmp[j] = x[j] + hj;
            *n_fev += 1;
            let fmp = fun(&x_tmp.view());

            x_tmp[i] = x[i] - hi;
            x_tmp[j] = x[j] - hj;
            *n_fev += 1;
            let fmm = fun(&x_tmp.view());

            x_tmp[i] = x[i];
            x_tmp[j] = x[j];

            let val = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            hess[i][j] = val;
            hess[j][i] = val;
        }
    }

    Ok(hess)
}

/// Solve (H + λI) p = -g via Gauss elimination with adaptive regularisation.
///
/// Increases λ geometrically if the initial system is numerically singular,
/// providing a fallback to steepest descent when H is far from positive definite.
fn regularised_solve(hess: &[Vec<f64>], g: &[f64], lambda_init: f64) -> Array1<f64> {
    let n = g.len();
    if n == 0 {
        return Array1::zeros(0);
    }

    let mut lambda = lambda_init;

    // Try up to 10 times, doubling λ on each failure
    for attempt in 0..10 {
        let _ = attempt;
        let mut a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = hess[i].clone();
                row[i] += lambda;
                row
            })
            .collect();
        let mut b: Vec<f64> = g.iter().map(|gi| -gi).collect();

        if let Some(sol) = gaussian_elimination(&mut a, &mut b) {
            return Array1::from(sol);
        }
        lambda *= 10.0;
    }

    // Absolute fallback: steepest descent direction
    Array1::from(g.iter().map(|gi| -gi).collect::<Vec<_>>())
}

/// Simple Gaussian elimination with partial pivoting. Returns `None` if singular.
fn gaussian_elimination(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            b[row] -= factor * b[col];
            for k in col..n {
                let v = a[col][k];
                a[row][k] -= factor * v;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-14 {
            return None;
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Wrapper that integrates `NewtonCG` into the standard `OptimizeResult` API.
pub fn minimize_newton_cg_advanced<F>(
    fun: F,
    x0: &[f64],
    options: Option<NewtonCGOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Send + Sync + Clone,
{
    let opts = options.unwrap_or_default();
    let eps = opts.eps;
    let fun_clone = fun.clone();
    let hvp = FiniteDiffHVP::new(fun_clone, eps);
    let solver = NewtonCG::new(opts);
    let result = solver.minimize(fun, x0, &hvp)?;

    Ok(OptimizeResult {
        x: result.x.clone(),
        fun: result.f_val,
        nit: result.n_iter,
        func_evals: result.n_fev,
        nfev: result.n_fev,
        success: result.converged,
        message: result.message,
        jacobian: Some(result.gradient),
        hessian: None,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    fn quadratic(x: &ArrayView1<f64>) -> f64 {
        x[0] * x[0] + 4.0 * x[1] * x[1]
    }

    #[test]
    fn test_finite_diff_hvp_quadratic() {
        // f(x) = x₀² + 4 x₁²  => H = diag(2, 8)
        let hvp_oracle = FiniteDiffHVP::with_default_step(quadratic);
        let x = vec![1.0, 1.0];
        let v = vec![1.0, 0.0];
        let hv = hvp_oracle.hvp(&x, &v).expect("HVP failed");
        assert_abs_diff_eq!(hv[0], 2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(hv[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_truncated_cg_quadratic() {
        // H = diag(2, 8), g = (2, 8) at (1,1) => optimal step = (-1, -1)
        let hvp_oracle = FiniteDiffHVP::with_default_step(quadratic);
        let x = vec![1.0, 1.0];
        let g = vec![2.0, 8.0];
        let cg = TruncatedCG::default_solver();
        let result = cg.solve(&x, &g, &hvp_oracle).expect("CG failed");
        assert_abs_diff_eq!(result.step[0], -1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.step[1], -1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_newton_cg_quadratic() {
        let hvp = FiniteDiffHVP::with_default_step(quadratic);
        let solver = NewtonCG::default_solver();
        let result = solver
            .minimize(quadratic, &[2.0, 1.0], &hvp)
            .expect("NewtonCG failed");
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_newton_cg_rosenbrock() {
        let hvp = FiniteDiffHVP::with_default_step(rosenbrock);
        let mut opts = NewtonCGOptions::default();
        opts.max_iter = 300;
        opts.gtol = 1e-4;
        let solver = NewtonCG::new(opts);
        let result = solver
            .minimize(rosenbrock, &[0.5, 0.5], &hvp)
            .expect("Newton-CG failed");
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_damped_newton_quadratic() {
        let solver = DampedNewton::default();
        let result = solver
            .minimize(quadratic, &[2.0, 1.0])
            .expect("DampedNewton failed");
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_newton_cg_with_bounds() {
        // Minimum of (x+1)² + (y+1)² is at (-1,-1) but bounds = [0,∞)²
        // So constrained minimum should be at (0,0)
        let f = |x: &ArrayView1<f64>| (x[0] + 1.0).powi(2) + (x[1] + 1.0).powi(2);
        let hvp = FiniteDiffHVP::with_default_step(f);
        let mut opts = NewtonCGOptions::default();
        opts.bounds = Some(Bounds::new(&[(Some(0.0), None), (Some(0.0), None)]));
        let solver = NewtonCG::new(opts);
        let result = solver.minimize(f, &[0.5, 0.5], &hvp).expect("failed");
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 0.1);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 0.1);
    }
}
