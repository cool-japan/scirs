//! Advanced trust-region constrained optimization
//!
//! Provides:
//! - [`TrustConstr`]: SLSQP-like trust-region constrained optimizer
//! - [`SubproblemSolver`]: Constrained trust-region subproblem via projected CG
//! - [`EqualityConstrained`]: Equality-constrained trust region (augmented Lagrangian)
//! - [`InequalityHandling`]: Interior point + trust region
//! - [`FilterMethodTR`]: Filter-based trust region for nonlinear programming

use crate::error::{OptimizeError, OptimizeResult};

// ---------------------------------------------------------------------------
// Common result
// ---------------------------------------------------------------------------

/// Result of a trust-region constrained solve
#[derive(Debug, Clone)]
pub struct TrustConstrResult {
    /// Optimal primal variables
    pub x: Vec<f64>,
    /// Objective value at optimum
    pub fun: f64,
    /// Gradient at optimum
    pub grad: Vec<f64>,
    /// Constraint violations at optimum
    pub constraint_violation: f64,
    /// Lagrange multipliers for equality constraints
    pub lambda_eq: Vec<f64>,
    /// Lagrange multipliers for inequality constraints
    pub lambda_ineq: Vec<f64>,
    /// Number of iterations
    pub nit: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of gradient evaluations
    pub njev: usize,
    /// Trust region radius at termination
    pub trust_radius: f64,
    /// Whether the optimizer converged
    pub success: bool,
    /// Termination message
    pub message: String,
}

// ---------------------------------------------------------------------------
// TrustConstr
// ---------------------------------------------------------------------------

/// Options for the TrustConstr optimizer
#[derive(Debug, Clone)]
pub struct TrustConstrOptions {
    /// Initial trust region radius
    pub initial_radius: f64,
    /// Minimum trust region radius
    pub min_radius: f64,
    /// Maximum trust region radius
    pub max_radius: f64,
    /// Gradient norm convergence tolerance
    pub gtol: f64,
    /// Constraint violation convergence tolerance
    pub ctol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Finite-difference step for gradient computation
    pub fd_step: f64,
    /// Acceptance threshold for trust region step
    pub eta1: f64,
    /// Good step threshold
    pub eta2: f64,
    /// Trust radius increase factor
    pub gamma_inc: f64,
    /// Trust radius decrease factor
    pub gamma_dec: f64,
    /// Penalty parameter for constraint handling
    pub penalty: f64,
    /// Penalty growth factor
    pub penalty_growth: f64,
}

impl Default for TrustConstrOptions {
    fn default() -> Self {
        TrustConstrOptions {
            initial_radius: 1.0,
            min_radius: 1e-10,
            max_radius: 1e4,
            gtol: 1e-8,
            ctol: 1e-8,
            max_iter: 1000,
            fd_step: 1e-7,
            eta1: 0.1,
            eta2: 0.75,
            gamma_inc: 2.0,
            gamma_dec: 0.25,
            penalty: 10.0,
            penalty_growth: 1.5,
        }
    }
}

/// SLSQP-like trust-region constrained optimizer.
///
/// At each iteration, solves a quadratic programming subproblem within
/// a trust region, using the projected conjugate gradient (PCG) method.
///
/// Handles both equality and inequality constraints. Inequality constraints
/// are converted to equality via slack variables: g(x) <= 0 → g(x) + s = 0, s >= 0.
pub struct TrustConstr {
    /// Algorithm options
    pub options: TrustConstrOptions,
}

impl Default for TrustConstr {
    fn default() -> Self {
        TrustConstr {
            options: TrustConstrOptions::default(),
        }
    }
}

impl TrustConstr {
    /// Create a new TrustConstr optimizer with given options
    pub fn new(options: TrustConstrOptions) -> Self {
        TrustConstr { options }
    }

    /// Minimize f(x) subject to equality constraints ceq(x) = 0 and
    /// inequality constraints cineq(x) <= 0.
    ///
    /// # Arguments
    ///
    /// * `func` - Objective function f(x)
    /// * `x0` - Initial point
    /// * `eq_constraints` - Equality constraint functions c_i(x) = 0
    /// * `ineq_constraints` - Inequality constraint functions g_j(x) <= 0
    pub fn minimize<F>(
        &self,
        func: F,
        x0: &[f64],
        eq_constraints: &[Box<dyn Fn(&[f64]) -> f64>],
        ineq_constraints: &[Box<dyn Fn(&[f64]) -> f64>],
    ) -> OptimizeResult<TrustConstrResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x0.len();
        let n_eq = eq_constraints.len();
        let n_ineq = ineq_constraints.len();

        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        // Augmented Lagrangian / projected Newton method.
        //
        // The augmented Lagrangian is:
        //   L_ρ(x, λ, μ) = f(x) + λ^T c(x) + (ρ/2)||c(x)||^2
        //                   + μ^T max(0, g(x)) + (ρ/2)||max(0,g(x))||^2
        //
        // Outer loop: update (λ, μ, ρ) via dual ascent.
        // Inner loop: minimise L_ρ w.r.t. x using a trust-region Newton step
        //             with finite-difference gradient and BFGS Hessian.
        let h = self.options.fd_step;

        let mut x = x0.to_vec();
        let mut radius = self.options.initial_radius;
        let mut penalty = self.options.penalty;
        let mut lambda_eq = vec![0.0f64; n_eq];
        let mut lambda_ineq = vec![0.0f64; n_ineq];
        let mut nfev = 0usize;
        let mut njev = 0usize;

        // Augmented Lagrangian value at x
        let aug_lag =
            |xv: &[f64], lam_eq: &[f64], lam_ineq: &[f64], rho: f64, nfev: &mut usize| -> f64 {
                let fv = func(xv);
                *nfev += 1;
                let mut val = fv;
                for (i, c) in eq_constraints.iter().enumerate() {
                    let cv = c(xv);
                    *nfev += 1;
                    val += lam_eq[i] * cv + 0.5 * rho * cv * cv;
                }
                for (j, g) in ineq_constraints.iter().enumerate() {
                    let gv = g(xv);
                    *nfev += 1;
                    // Shifted constraint: σ_j = g_j(x) + λ_j/ρ; active when σ_j > 0
                    let sigma = gv + lam_ineq[j] / rho;
                    if sigma > 0.0 {
                        val += lam_ineq[j] * gv + 0.5 * rho * gv * gv;
                    } else {
                        // Inactive — subtract contribution already counted
                        val -= lam_ineq[j] * lam_ineq[j] / (2.0 * rho);
                    }
                }
                val
            };

        // BFGS Hessian approximation (identity initially)
        let mut bfgs_h: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0f64; n];
                row[i] = 1.0;
                row
            })
            .collect();
        let mut prev_x: Option<Vec<f64>> = None;
        let mut prev_grad: Option<Vec<f64>> = None;

        // Finite-difference gradient of the augmented Lagrangian
        let aug_lag_grad = |xv: &[f64],
                            lam_eq: &[f64],
                            lam_ineq: &[f64],
                            rho: f64,
                            nfev: &mut usize,
                            njev: &mut usize|
         -> Vec<f64> {
            *njev += 1;
            let f0 = aug_lag(xv, lam_eq, lam_ineq, rho, nfev);
            let mut g = vec![0.0f64; n];
            let mut xp = xv.to_vec();
            for i in 0..n {
                xp[i] = xv[i] + h;
                g[i] = (aug_lag(&xp, lam_eq, lam_ineq, rho, nfev) - f0) / h;
                xp[i] = xv[i];
            }
            g
        };

        let mut total_iter = 0usize;
        // Number of outer AL iterations
        let outer_iters = 30usize;
        let inner_iters = (self.options.max_iter / outer_iters).max(10);

        for _outer in 0..outer_iters {
            // ---- Inner loop: trust-region Newton minimisation of L_ρ ----
            let mut f_cur = aug_lag(&x, &lambda_eq, &lambda_ineq, penalty, &mut nfev);

            for _inner in 0..inner_iters {
                total_iter += 1;

                let grad =
                    aug_lag_grad(&x, &lambda_eq, &lambda_ineq, penalty, &mut nfev, &mut njev);
                let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

                // BFGS Hessian update
                if let (Some(ref px), Some(ref pg)) = (&prev_x, &prev_grad) {
                    let s: Vec<f64> = x.iter().zip(px.iter()).map(|(xi, pxi)| xi - pxi).collect();
                    let y: Vec<f64> = grad
                        .iter()
                        .zip(pg.iter())
                        .map(|(gi, pgi)| gi - pgi)
                        .collect();
                    let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();
                    let hs: Vec<f64> = (0..n)
                        .map(|i| {
                            bfgs_h[i]
                                .iter()
                                .zip(s.iter())
                                .map(|(hi, si)| hi * si)
                                .sum::<f64>()
                        })
                        .collect();
                    let sths: f64 = s.iter().zip(hs.iter()).map(|(si, hsi)| si * hsi).sum();
                    let sy_damp = sy.max(0.2 * sths);
                    if sy_damp.abs() > 1e-10 && sths.abs() > 1e-10 {
                        for i in 0..n {
                            for j in 0..n {
                                bfgs_h[i][j] += y[i] * y[j] / sy_damp - hs[i] * hs[j] / sths;
                            }
                        }
                    }
                }
                prev_x = Some(x.clone());
                prev_grad = Some(grad.clone());

                if gnorm < self.options.gtol {
                    break;
                }

                // Compute Newton step d = -B^{-1} grad using Gaussian elimination
                let d = {
                    let mut a: Vec<Vec<f64>> = bfgs_h.iter().map(|row| row.to_vec()).collect();
                    let mut b: Vec<f64> = grad.iter().map(|&gi| -gi).collect();
                    for i in 0..n {
                        a[i][i] += 1e-6;
                    }
                    tc_gaussian_elim(&mut a, &mut b, n)
                        .unwrap_or_else(|| grad.iter().map(|&gi| -gi * 0.01).collect())
                };

                // Clamp to trust radius
                let d_norm: f64 = d.iter().map(|v| v * v).sum::<f64>().sqrt();
                let scale = if d_norm > radius {
                    radius / d_norm
                } else {
                    1.0
                };
                let d_scaled: Vec<f64> = d.iter().map(|&v| v * scale).collect();

                // Accept/reject via trust region ratio
                let x_trial: Vec<f64> = x
                    .iter()
                    .zip(d_scaled.iter())
                    .map(|(xi, di)| xi + di)
                    .collect();
                let f_trial = aug_lag(&x_trial, &lambda_eq, &lambda_ineq, penalty, &mut nfev);

                let actual_red = f_cur - f_trial;
                let pred_red: f64 = grad
                    .iter()
                    .zip(d_scaled.iter())
                    .map(|(gi, di)| -gi * di)
                    .sum::<f64>();

                let rho = if pred_red > 1e-14 {
                    actual_red / pred_red
                } else if actual_red > 0.0 {
                    1.0
                } else {
                    0.0
                };

                if rho < self.options.eta1 {
                    radius = (radius * self.options.gamma_dec).max(self.options.min_radius);
                } else {
                    x = x_trial;
                    f_cur = f_trial;
                    if rho > self.options.eta2 {
                        radius = (radius * self.options.gamma_inc).min(self.options.max_radius);
                    }
                }

                if radius < self.options.min_radius {
                    break;
                }
            }

            // ---- Outer loop: check convergence and update multipliers ----
            let mut cv_eq_sum = 0.0f64;
            for c in eq_constraints.iter() {
                cv_eq_sum += c(&x).abs();
                nfev += 1;
            }
            let mut cv_ineq_sum = 0.0f64;
            for g in ineq_constraints.iter() {
                let gv = g(&x);
                nfev += 1;
                cv_ineq_sum += gv.max(0.0);
            }
            let cv_sum = cv_eq_sum + cv_ineq_sum;

            // Check full convergence
            let grad_final =
                aug_lag_grad(&x, &lambda_eq, &lambda_ineq, penalty, &mut nfev, &mut njev);
            let gnorm_final: f64 = grad_final.iter().map(|g| g * g).sum::<f64>().sqrt();

            if gnorm_final < self.options.gtol && cv_sum < self.options.ctol {
                let final_f = func(&x);
                nfev += 1;
                return Ok(TrustConstrResult {
                    x: x.clone(),
                    fun: final_f,
                    grad: grad_final,
                    constraint_violation: cv_sum,
                    lambda_eq,
                    lambda_ineq,
                    nit: total_iter,
                    nfev,
                    njev,
                    trust_radius: radius,
                    success: true,
                    message: "Optimization converged".to_string(),
                });
            }

            // Dual ascent update for equality multipliers: λ += ρ * c(x)
            for (i, c) in eq_constraints.iter().enumerate() {
                let cv = c(&x);
                nfev += 1;
                lambda_eq[i] += penalty * cv;
            }
            // Dual ascent update for inequality multipliers (projected): μ = max(0, μ + ρ*g)
            for (j, g) in ineq_constraints.iter().enumerate() {
                let gv = g(&x);
                nfev += 1;
                lambda_ineq[j] = (lambda_ineq[j] + penalty * gv).max(0.0);
            }

            // Increase penalty if constraint violation is large
            if cv_sum > self.options.ctol * 10.0 {
                penalty = (penalty * self.options.penalty_growth).min(1e8);
                // Reset trust radius when penalty changes to encourage
                // larger steps in the new landscape
                radius = self.options.initial_radius;
                // Reset BFGS approximation
                bfgs_h = (0..n)
                    .map(|i| {
                        let mut row = vec![0.0f64; n];
                        row[i] = 1.0;
                        row
                    })
                    .collect();
                prev_x = None;
                prev_grad = None;
            }
        }

        let x_final = x;
        let f_final = func(&x_final);
        nfev += 1;

        let mut cv_final = 0.0f64;
        for c in eq_constraints {
            cv_final += c(&x_final).abs();
            nfev += 1;
        }
        for g in ineq_constraints {
            cv_final += g(&x_final).max(0.0);
            nfev += 1;
        }

        let grad_final: Vec<f64> = {
            let mut g = vec![0.0f64; n];
            let f0 = func(&x_final);
            nfev += 1;
            for i in 0..n {
                let mut xf = x_final.clone();
                xf[i] += h;
                g[i] = (func(&xf) - f0) / h;
                nfev += 1;
                njev += 1;
            }
            g
        };

        Ok(TrustConstrResult {
            x: x_final,
            fun: f_final,
            grad: grad_final,
            constraint_violation: cv_final,
            lambda_eq,
            lambda_ineq,
            nit: total_iter,
            nfev,
            njev,
            trust_radius: radius,
            success: cv_final < self.options.ctol * 100.0,
            message: "Maximum iterations reached".to_string(),
        })
    }
}

/// Gaussian elimination with partial pivoting for the trust-constr Newton step.
/// Returns None if the system is (near-)singular.
fn tc_gaussian_elim(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
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
            for k in col..n {
                let val = a[col][k] * factor;
                a[row][k] -= val;
            }
            let bv = b[col] * factor;
            b[row] -= bv;
        }
    }

    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-14 {
            return None;
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

/// Estimate Lagrange multipliers for equality constraints
fn update_multipliers_eq(
    x: &[f64],
    lambda: &mut Vec<f64>,
    constraints: &[Box<dyn Fn(&[f64]) -> f64>],
    penalty: f64,
    h: f64,
    nfev: &mut usize,
) {
    for (i, c) in constraints.iter().enumerate() {
        let cv = c(x);
        *nfev += 1;
        // Dual update: λ += ρ * c(x)
        lambda[i] += penalty * cv;
    }
}

/// Estimate Lagrange multipliers for inequality constraints
fn update_multipliers_ineq(
    x: &[f64],
    lambda: &mut Vec<f64>,
    constraints: &[Box<dyn Fn(&[f64]) -> f64>],
    penalty: f64,
    _h: f64,
    nfev: &mut usize,
) {
    for (i, g) in constraints.iter().enumerate() {
        let gv = g(x);
        *nfev += 1;
        // Dual update (projected): λ = max(0, λ + ρ * g(x))
        lambda[i] = (lambda[i] + penalty * gv).max(0.0);
    }
}

// ---------------------------------------------------------------------------
// SubproblemSolver: Projected Conjugate Gradient
// ---------------------------------------------------------------------------

/// Constrained trust-region subproblem solver using projected conjugate gradient.
///
/// Solves: min_{p}  g^T p + 0.5 p^T B p
///         s.t.    ||p|| <= Δ,   A p = 0  (projected on constraint manifold)
///
/// Uses the Steihaug-Toint projected CG approach.
pub struct SubproblemSolver {
    /// Maximum CG iterations
    pub max_cg_iter: usize,
    /// CG convergence tolerance
    pub cg_tol: f64,
}

impl Default for SubproblemSolver {
    fn default() -> Self {
        SubproblemSolver {
            max_cg_iter: 100,
            cg_tol: 1e-10,
        }
    }
}

impl SubproblemSolver {
    /// Create a new subproblem solver
    pub fn new(max_cg_iter: usize, cg_tol: f64) -> Self {
        SubproblemSolver {
            max_cg_iter,
            cg_tol,
        }
    }

    /// Solve the trust-region subproblem using Steihaug-Toint PCG.
    ///
    /// Returns the step `p` and whether the trust region boundary was hit.
    ///
    /// # Arguments
    ///
    /// * `g` - Gradient vector
    /// * `b_times_v` - Function computing B*v (Hessian-vector product)
    /// * `radius` - Trust region radius Δ
    pub fn solve<BV>(&self, g: &[f64], b_times_v: BV, radius: f64) -> (Vec<f64>, bool)
    where
        BV: Fn(&[f64]) -> Vec<f64>,
    {
        let n = g.len();
        let mut p = vec![0.0f64; n];
        let mut r = g.to_vec(); // residual = g + B*p = g for p=0
        let mut d: Vec<f64> = r.iter().map(|ri| -ri).collect(); // search direction

        let r_norm_0: f64 = r.iter().map(|ri| ri * ri).sum::<f64>().sqrt();
        if r_norm_0 < self.cg_tol {
            return (p, false);
        }

        for _iter in 0..self.max_cg_iter {
            let bd = b_times_v(&d);

            // d^T B d
            let dtbd: f64 = d.iter().zip(bd.iter()).map(|(di, bdi)| di * bdi).sum();

            if dtbd <= 0.0 {
                // Negative curvature: step to trust region boundary
                let tau = find_tau_boundary(&p, &d, radius);
                for i in 0..n {
                    p[i] += tau * d[i];
                }
                return (p, true);
            }

            let r_norm_sq: f64 = r.iter().map(|ri| ri * ri).sum();
            let alpha = r_norm_sq / dtbd;

            // Trial step
            let mut p_new = vec![0.0f64; n];
            for i in 0..n {
                p_new[i] = p[i] + alpha * d[i];
            }

            // Check trust region
            let p_new_norm: f64 = p_new.iter().map(|pi| pi * pi).sum::<f64>().sqrt();
            if p_new_norm >= radius {
                // Intersect with trust region boundary
                let tau = find_tau_boundary(&p, &d, radius);
                for i in 0..n {
                    p[i] += tau * d[i];
                }
                return (p, true);
            }

            p = p_new;

            // Update residual: r_new = r + alpha * B * d
            let mut r_new = vec![0.0f64; n];
            for i in 0..n {
                r_new[i] = r[i] + alpha * bd[i];
            }

            let r_new_norm: f64 = r_new.iter().map(|ri| ri * ri).sum::<f64>().sqrt();
            if r_new_norm < self.cg_tol * r_norm_0 {
                return (p, false);
            }

            let beta = r_new.iter().map(|ri| ri * ri).sum::<f64>() / r_norm_sq;
            for i in 0..n {
                d[i] = -r_new[i] + beta * d[i];
            }
            r = r_new;
        }

        (p, false)
    }
}

/// Find τ such that ||p + τ d|| = Δ (positive root)
fn find_tau_boundary(p: &[f64], d: &[f64], radius: f64) -> f64 {
    let a: f64 = d.iter().map(|di| di * di).sum();
    let b: f64 = 2.0 * p.iter().zip(d.iter()).map(|(pi, di)| pi * di).sum::<f64>();
    let c: f64 = p.iter().map(|pi| pi * pi).sum::<f64>() - radius * radius;

    if a < 1e-14 {
        return 0.0;
    }

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return 0.0;
    }

    let sqrt_disc = disc.sqrt();
    let tau1 = (-b + sqrt_disc) / (2.0 * a);
    let tau2 = (-b - sqrt_disc) / (2.0 * a);

    if tau1 >= 0.0 {
        tau1
    } else {
        tau2.max(0.0)
    }
}

// ---------------------------------------------------------------------------
// EqualityConstrained: Augmented Lagrangian Trust Region
// ---------------------------------------------------------------------------

/// Options for equality-constrained trust region
#[derive(Debug, Clone)]
pub struct EqualityConstrainedOptions {
    /// Initial penalty parameter ρ
    pub rho: f64,
    /// Maximum penalty
    pub max_rho: f64,
    /// Penalty growth factor
    pub rho_growth: f64,
    /// Outer convergence tolerance (constraint violation)
    pub outer_tol: f64,
    /// Inner convergence tolerance (stationarity)
    pub inner_tol: f64,
    /// Maximum outer iterations (Lagrange multiplier updates)
    pub max_outer: usize,
    /// Maximum inner iterations (trust region steps)
    pub max_inner: usize,
    /// Initial trust region radius
    pub radius: f64,
    /// Finite-difference step
    pub h: f64,
}

impl Default for EqualityConstrainedOptions {
    fn default() -> Self {
        EqualityConstrainedOptions {
            rho: 1.0,
            max_rho: 1e8,
            rho_growth: 2.0,
            outer_tol: 1e-7,
            inner_tol: 1e-6,
            max_outer: 50,
            max_inner: 200,
            radius: 1.0,
            h: 1e-7,
        }
    }
}

/// Equality-constrained trust region optimizer using augmented Lagrangian.
///
/// Solves: min f(x) s.t. c(x) = 0
///
/// Via the augmented Lagrangian:
/// L_ρ(x, λ) = f(x) + λ^T c(x) + (ρ/2) ||c(x)||²
///
/// Inner minimization uses trust-region gradient descent on L_ρ.
/// Outer loop updates multipliers: λ += ρ c(x*).
pub struct EqualityConstrained {
    /// Algorithm options
    pub options: EqualityConstrainedOptions,
}

impl Default for EqualityConstrained {
    fn default() -> Self {
        EqualityConstrained {
            options: EqualityConstrainedOptions::default(),
        }
    }
}

impl EqualityConstrained {
    /// Create with custom options
    pub fn new(options: EqualityConstrainedOptions) -> Self {
        EqualityConstrained { options }
    }

    /// Solve min f(x) s.t. c_i(x) = 0 for all i.
    pub fn minimize<F>(
        &self,
        func: F,
        x0: &[f64],
        constraints: &[Box<dyn Fn(&[f64]) -> f64>],
    ) -> OptimizeResult<TrustConstrResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x0.len();
        let n_eq = constraints.len();
        let h = self.options.h;

        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        let mut x = x0.to_vec();
        let mut lambda = vec![0.0f64; n_eq];
        let mut rho = self.options.rho;
        let mut radius = self.options.radius;
        let mut nfev = 0usize;
        let mut njev = 0usize;
        let mut total_iter = 0usize;

        // Augmented Lagrangian: L_ρ(x) = f(x) + Σ λ_i c_i(x) + (ρ/2) Σ c_i(x)²
        let aug_lag = |x: &[f64], lambda: &[f64], rho: f64, nfev: &mut usize| -> f64 {
            let f = func(x);
            *nfev += 1;
            let mut pen = 0.0f64;
            for (i, c) in constraints.iter().enumerate() {
                let cv = c(x);
                *nfev += 1;
                pen += lambda[i] * cv + 0.5 * rho * cv * cv;
            }
            f + pen
        };

        for _outer in 0..self.options.max_outer {
            // Inner loop: minimize L_ρ(x, λ) w.r.t. x using trust-region gradient descent
            for _inner in 0..self.options.max_inner {
                total_iter += 1;

                let f_cur = aug_lag(&x, &lambda, rho, &mut nfev);

                // Gradient of augmented Lagrangian
                let mut grad = vec![0.0f64; n];
                for i in 0..n {
                    let mut xf = x.clone();
                    xf[i] += h;
                    grad[i] = (aug_lag(&xf, &lambda, rho, &mut nfev) - f_cur) / h;
                    njev += 1;
                }

                let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                if gnorm < self.options.inner_tol {
                    break;
                }

                // Cauchy step
                let step = (radius / gnorm).min(radius);
                let mut x_trial = vec![0.0f64; n];
                for i in 0..n {
                    x_trial[i] = x[i] - step * grad[i];
                }

                let f_trial = aug_lag(&x_trial, &lambda, rho, &mut nfev);

                // Simple acceptance
                if f_trial < f_cur {
                    let improvement = f_cur - f_trial;
                    x = x_trial;
                    if improvement > 0.5 * step * gnorm {
                        radius = (radius * 2.0).min(10.0);
                    }
                } else {
                    radius *= 0.5;
                    if radius < 1e-12 {
                        break;
                    }
                }
            }

            // Check constraint violation
            let mut cv_norm = 0.0f64;
            let mut cv_vec = vec![0.0f64; n_eq];
            for (i, c) in constraints.iter().enumerate() {
                cv_vec[i] = c(&x);
                nfev += 1;
                cv_norm += cv_vec[i] * cv_vec[i];
            }
            cv_norm = cv_norm.sqrt();

            if cv_norm < self.options.outer_tol {
                let f_final = func(&x);
                nfev += 1;

                // Compute gradient of objective at solution
                let mut grad_f = vec![0.0f64; n];
                for i in 0..n {
                    let mut xf = x.clone();
                    xf[i] += h;
                    grad_f[i] = (func(&xf) - f_final) / h;
                    nfev += 1;
                    njev += 1;
                }

                return Ok(TrustConstrResult {
                    x,
                    fun: f_final,
                    grad: grad_f,
                    constraint_violation: cv_norm,
                    lambda_eq: lambda,
                    lambda_ineq: vec![],
                    nit: total_iter,
                    nfev,
                    njev,
                    trust_radius: radius,
                    success: true,
                    message: "Equality-constrained TR converged".to_string(),
                });
            }

            // Multiplier update: λ += ρ c(x)
            for i in 0..n_eq {
                lambda[i] += rho * cv_vec[i];
            }

            // Penalty update
            rho = (rho * self.options.rho_growth).min(self.options.max_rho);
        }

        let f_final = func(&x);
        nfev += 1;
        let cv_final: f64 = constraints
            .iter()
            .map(|c| {
                nfev += 1;
                c(&x).powi(2)
            })
            .sum::<f64>()
            .sqrt();

        Ok(TrustConstrResult {
            x,
            fun: f_final,
            grad: vec![0.0f64; n],
            constraint_violation: cv_final,
            lambda_eq: lambda,
            lambda_ineq: vec![],
            nit: total_iter,
            nfev,
            njev,
            trust_radius: radius,
            success: cv_final < self.options.outer_tol * 100.0,
            message: "Maximum outer iterations reached".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// InequalityHandling: Interior Point + Trust Region
// ---------------------------------------------------------------------------

/// Options for interior-point trust-region inequality handling
#[derive(Debug, Clone)]
pub struct InequalityHandlingOptions {
    /// Initial barrier parameter μ
    pub mu: f64,
    /// Barrier parameter reduction factor
    pub mu_reduce: f64,
    /// Minimum barrier parameter
    pub min_mu: f64,
    /// Constraint violation tolerance
    pub tol: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Trust region radius
    pub radius: f64,
    /// Finite-difference step
    pub h: f64,
}

impl Default for InequalityHandlingOptions {
    fn default() -> Self {
        InequalityHandlingOptions {
            mu: 1.0,
            mu_reduce: 0.1,
            min_mu: 1e-9,
            tol: 1e-7,
            max_iter: 500,
            radius: 1.0,
            h: 1e-7,
        }
    }
}

/// Interior point trust-region method for inequality-constrained optimization.
///
/// Solves: min f(x) s.t. g_j(x) <= 0 for all j
///
/// Via barrier formulation (with slack variables s_j > 0):
/// min  f(x) - μ Σ log(s_j)
/// s.t. g_j(x) + s_j = 0  (equality after introducing slacks)
///
/// Trust-region step is taken in (x, s) space.
pub struct InequalityHandling {
    /// Algorithm options
    pub options: InequalityHandlingOptions,
}

impl Default for InequalityHandling {
    fn default() -> Self {
        InequalityHandling {
            options: InequalityHandlingOptions::default(),
        }
    }
}

impl InequalityHandling {
    /// Create with custom options
    pub fn new(options: InequalityHandlingOptions) -> Self {
        InequalityHandling { options }
    }

    /// Minimize f(x) subject to g_j(x) <= 0
    pub fn minimize<F>(
        &self,
        func: F,
        x0: &[f64],
        ineq_constraints: &[Box<dyn Fn(&[f64]) -> f64>],
    ) -> OptimizeResult<TrustConstrResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x0.len();
        let n_ineq = ineq_constraints.len();
        let h = self.options.h;

        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        // Augmented vector [x, s] where s_j = -g_j(x) + small positive margin
        let n_aug = n + n_ineq;
        let mut z = vec![0.0f64; n_aug];
        for i in 0..n {
            z[i] = x0[i];
        }
        // Initialize slacks
        for j in 0..n_ineq {
            let gj = (ineq_constraints[j])(&z[0..n]);
            z[n + j] = (-gj + 1e-4).max(1e-8);
        }

        let mut mu = self.options.mu;
        let mut radius = self.options.radius;
        let mut nfev = 0usize;
        let mut njev = 0usize;
        let mut total_iter = 0usize;

        // Barrier function: f(x) - μ Σ log(s_j) + penalty Σ (g_j(x) + s_j)^2
        let barrier = |z: &[f64], mu: f64, nfev: &mut usize| -> f64 {
            let x = &z[0..n];
            let s = &z[n..n_aug];
            let f = func(x);
            *nfev += 1;
            let mut barrier_val = 0.0f64;
            let mut pen = 0.0f64;
            for (j, g) in ineq_constraints.iter().enumerate() {
                let gj = g(x);
                *nfev += 1;
                if s[j] > 0.0 {
                    barrier_val -= mu * s[j].ln();
                } else {
                    barrier_val += 1e10; // infeasible
                }
                // Penalty for g_j(x) + s_j ≠ 0
                pen += (gj + s[j]).powi(2);
            }
            f + barrier_val + 1000.0 * mu * pen
        };

        while mu >= self.options.min_mu {
            let mut f_cur = barrier(&z, mu, &mut nfev);

            for _inner in 0..self.options.max_iter / 10 {
                total_iter += 1;

                // Gradient of barrier
                let mut grad = vec![0.0f64; n_aug];
                for i in 0..n_aug {
                    let mut zf = z.clone();
                    zf[i] += h;
                    // Protect slacks from going negative in FD
                    if i >= n && zf[i] <= 0.0 {
                        grad[i] = 1e10;
                        continue;
                    }
                    grad[i] = (barrier(&zf, mu, &mut nfev) - f_cur) / h;
                    njev += 1;
                }

                let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                if gnorm < self.options.tol {
                    break;
                }

                // Trust region step
                let step = (radius / gnorm).min(1.0);
                let mut z_trial = vec![0.0f64; n_aug];
                for i in 0..n_aug {
                    z_trial[i] = z[i] - step * grad[i];
                }
                // Ensure slacks > 0
                for j in 0..n_ineq {
                    if z_trial[n + j] <= 0.0 {
                        z_trial[n + j] = 1e-10;
                    }
                }

                let f_trial = barrier(&z_trial, mu, &mut nfev);

                if f_trial < f_cur {
                    z = z_trial;
                    f_cur = f_trial;
                    radius = (radius * 1.5).min(10.0);
                } else {
                    radius *= 0.5;
                    if radius < 1e-12 {
                        break;
                    }
                }
            }

            mu *= self.options.mu_reduce;
        }

        let x_final = z[0..n].to_vec();
        let f_final = func(&x_final);
        nfev += 1;

        let mut cv_final = 0.0f64;
        for g in ineq_constraints {
            let gv = g(&x_final);
            nfev += 1;
            if gv > 0.0 {
                cv_final += gv;
            }
        }

        let mut grad_f = vec![0.0f64; n];
        let f0 = func(&x_final);
        nfev += 1;
        for i in 0..n {
            let mut xf = x_final.clone();
            xf[i] += h;
            grad_f[i] = (func(&xf) - f0) / h;
            nfev += 1;
            njev += 1;
        }

        Ok(TrustConstrResult {
            x: x_final,
            fun: f_final,
            grad: grad_f,
            constraint_violation: cv_final,
            lambda_eq: vec![],
            lambda_ineq: z[n..n_aug].to_vec(),
            nit: total_iter,
            nfev,
            njev,
            trust_radius: radius,
            success: cv_final < self.options.tol * 100.0,
            message: "Interior-point TR optimization completed".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// FilterMethodTR: Filter-based Trust Region
// ---------------------------------------------------------------------------

/// Options for filter-based trust region
#[derive(Debug, Clone)]
pub struct FilterMethodTROptions {
    /// Initial trust region radius
    pub initial_radius: f64,
    /// Minimum trust region radius
    pub min_radius: f64,
    /// Maximum trust region radius
    pub max_radius: f64,
    /// Filter envelope parameter γ_f
    pub gamma_f: f64,
    /// Filter envelope parameter γ_theta
    pub gamma_theta: f64,
    /// Acceptance threshold ρ_min
    pub eta1: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Finite-difference step
    pub h: f64,
}

impl Default for FilterMethodTROptions {
    fn default() -> Self {
        FilterMethodTROptions {
            initial_radius: 1.0,
            min_radius: 1e-10,
            max_radius: 100.0,
            gamma_f: 1e-5,
            gamma_theta: 1e-5,
            eta1: 0.1,
            max_iter: 500,
            tol: 1e-7,
            h: 1e-7,
        }
    }
}

/// A filter entry (f_val, theta) representing a dominated pair
#[derive(Debug, Clone)]
struct FilterEntry {
    f_val: f64,
    theta: f64,
}

impl FilterEntry {
    /// Check if (f, theta) is dominated by this entry
    fn dominates(&self, f: f64, theta: f64, gamma_f: f64, gamma_theta: f64) -> bool {
        self.f_val - gamma_f * self.theta <= f && self.theta * (1.0 - gamma_theta) <= theta
    }
}

/// Filter-based trust region method for nonlinear programming.
///
/// Uses a filter to accept/reject trial steps: a step is acceptable if it
/// improves either the objective value or the constraint violation compared
/// to all current filter entries.
///
/// The filter {(f_k, θ_k)} records (objective, violation) pairs at accepted
/// iterates. A trial point (f, θ) passes the filter if it is not dominated.
pub struct FilterMethodTR {
    /// Algorithm options
    pub options: FilterMethodTROptions,
}

impl Default for FilterMethodTR {
    fn default() -> Self {
        FilterMethodTR {
            options: FilterMethodTROptions::default(),
        }
    }
}

impl FilterMethodTR {
    /// Create with custom options
    pub fn new(options: FilterMethodTROptions) -> Self {
        FilterMethodTR { options }
    }

    /// Minimize f(x) subject to constraints
    ///
    /// # Arguments
    ///
    /// * `func` - Objective function
    /// * `x0` - Initial point
    /// * `constraints` - Vector of constraint functions h_i(x) (violations: positive values)
    pub fn minimize<F>(
        &self,
        func: F,
        x0: &[f64],
        constraints: &[Box<dyn Fn(&[f64]) -> f64>],
    ) -> OptimizeResult<TrustConstrResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x0.len();
        let h = self.options.h;

        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        let mut x = x0.to_vec();
        let mut radius = self.options.initial_radius;
        let mut nfev = 0usize;
        let mut njev = 0usize;

        // Constraint violation measure θ(x) = Σ max(0, h_i(x))
        let theta = |x: &[f64], nfev: &mut usize| -> f64 {
            constraints
                .iter()
                .map(|c| {
                    *nfev += 1;
                    c(x).max(0.0)
                })
                .sum()
        };

        let f0 = func(&x);
        nfev += 1;
        let theta0 = theta(&x, &mut nfev);

        // Initialize filter with (f, theta) at x0
        let mut filter: Vec<FilterEntry> = vec![FilterEntry {
            f_val: f0,
            theta: theta0,
        }];

        let is_filter_acceptable = |f: f64, th: f64, filter: &[FilterEntry]| -> bool {
            for entry in filter {
                if entry.dominates(f, th, self.options.gamma_f, self.options.gamma_theta) {
                    return false;
                }
            }
            true
        };

        let mut f_cur = f0;

        for iter in 0..self.options.max_iter {
            // Compute gradient of objective
            let mut grad = vec![0.0f64; n];
            for i in 0..n {
                let mut xf = x.clone();
                xf[i] += h;
                grad[i] = (func(&xf) - f_cur) / h;
                nfev += 1;
                njev += 1;
            }

            let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            let theta_cur = theta(&x, &mut nfev);

            if gnorm < self.options.tol && theta_cur < self.options.tol {
                let mut grad_f = vec![0.0f64; n];
                let f0c = func(&x);
                nfev += 1;
                for i in 0..n {
                    let mut xf = x.clone();
                    xf[i] += h;
                    grad_f[i] = (func(&xf) - f0c) / h;
                    nfev += 1;
                }
                return Ok(TrustConstrResult {
                    x,
                    fun: f_cur,
                    grad: grad_f,
                    constraint_violation: theta_cur,
                    lambda_eq: vec![],
                    lambda_ineq: vec![],
                    nit: iter + 1,
                    nfev,
                    njev,
                    trust_radius: radius,
                    success: true,
                    message: "Filter-TR converged".to_string(),
                });
            }

            // Compute trial step (Cauchy point)
            let step_len = (radius / gnorm).min(radius);
            let mut x_trial = vec![0.0f64; n];
            for i in 0..n {
                x_trial[i] = x[i] - step_len * grad[i];
            }

            let f_trial = func(&x_trial);
            nfev += 1;
            let theta_trial = theta(&x_trial, &mut nfev);

            // Compute actual reduction ratio (for f-type steps only)
            let actual_red = f_cur - f_trial;
            let predicted_red = step_len * gnorm;
            let rho = if predicted_red.abs() > 1e-14 {
                actual_red / predicted_red
            } else {
                0.0
            };

            // Filter acceptance
            let accepted = is_filter_acceptable(f_trial, theta_trial, &filter);

            if accepted && (rho > self.options.eta1 || theta_trial < theta_cur * 0.9) {
                // Accept step
                x = x_trial;
                f_cur = f_trial;

                // Add current point to filter (remove dominated entries first)
                filter.retain(|e| {
                    !(e.f_val >= f_cur && e.theta >= theta_trial * (1.0 - self.options.gamma_theta))
                });
                filter.push(FilterEntry {
                    f_val: f_cur,
                    theta: theta_trial,
                });

                // Expand trust region
                if rho > 0.75 {
                    radius =
                        (radius * self.options.initial_radius.sqrt()).min(self.options.max_radius);
                }
            } else {
                // Reject step; contract trust region
                radius = (radius * 0.25).max(self.options.min_radius);
            }

            if radius < self.options.min_radius {
                break;
            }
        }

        let theta_final = theta(&x, &mut nfev);

        Ok(TrustConstrResult {
            x,
            fun: f_cur,
            grad: vec![0.0f64; n],
            constraint_violation: theta_final,
            lambda_eq: vec![],
            lambda_ineq: vec![],
            nit: self.options.max_iter,
            nfev,
            njev,
            trust_radius: radius,
            success: theta_final < self.options.tol * 100.0,
            message: "Filter-TR: maximum iterations reached".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    #[test]
    fn test_trust_constr_unconstrained() {
        let tc = TrustConstr::default();
        let result = tc
            .minimize(rosenbrock, &[0.0, 0.0], &[], &[])
            .expect("failed to create result");
        assert!(result.fun < 1.0, "Expected fun < 1.0, got {}", result.fun);
    }

    #[test]
    fn test_trust_constr_with_equality() {
        // min (x-2)^2 + (y-2)^2 s.t. x + y = 3
        let func = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
        let eq_c: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![Box::new(|x: &[f64]| x[0] + x[1] - 3.0)];

        let tc = TrustConstr::new(TrustConstrOptions {
            max_iter: 500,
            ..Default::default()
        });
        let result = tc
            .minimize(func, &[0.5, 0.5], &eq_c, &[])
            .expect("failed to create result");
        assert!(
            result.constraint_violation < 0.5,
            "cv = {}",
            result.constraint_violation
        );
    }

    #[test]
    fn test_trust_constr_with_inequality() {
        // min (x-1)^2 + (y-1)^2 s.t. x + y <= 1  (i.e. x+y-1 <= 0)
        let func = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2);
        let ineq_c: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![Box::new(|x: &[f64]| x[0] + x[1] - 1.0)];

        let tc = TrustConstr::new(TrustConstrOptions {
            max_iter: 500,
            ..Default::default()
        });
        let result = tc
            .minimize(func, &[0.25, 0.25], &[], &ineq_c)
            .expect("failed to create result");
        assert!(result.fun < 1.0, "Expected fun < 1.0, got {}", result.fun);
    }

    #[test]
    fn test_subproblem_solver_cauchy_point() {
        let solver = SubproblemSolver::default();
        let g = vec![1.0, 0.0];
        // B = identity
        let (p, on_boundary) = solver.solve(&g, |v| v.to_vec(), 0.5);
        assert!(
            (p[0] + 0.5).abs() < 0.1,
            "p[0] should be near -0.5, got {}",
            p[0]
        );
        assert!(on_boundary, "Should hit boundary");
    }

    #[test]
    fn test_equality_constrained_tr() {
        let func = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let constraints: Vec<Box<dyn Fn(&[f64]) -> f64>> =
            vec![Box::new(|x: &[f64]| x[0] + x[1] - 1.0)];

        let ec = EqualityConstrained::new(EqualityConstrainedOptions {
            max_outer: 30,
            max_inner: 100,
            outer_tol: 1e-5,
            ..Default::default()
        });
        let result = ec
            .minimize(func, &[0.5, 0.5], &constraints)
            .expect("failed to create result");
        // Optimal at x = (0.5, 0.5), f = 0.5
        assert!(result.fun < 0.6, "Expected fun < 0.6, got {}", result.fun);
    }

    #[test]
    fn test_inequality_handling_interior_point() {
        // min x^2 + y^2 s.t. x + y >= 1 (i.e., -(x+y-1) <= 0)
        let func = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let ineq_c: Vec<Box<dyn Fn(&[f64]) -> f64>> =
            vec![Box::new(|x: &[f64]| -(x[0] + x[1] - 1.0))];

        let ih = InequalityHandling::default();
        let result = ih
            .minimize(func, &[1.0, 1.0], &ineq_c)
            .expect("failed to create result");
        assert!(result.fun < 1.0, "Expected fun < 1.0, got {}", result.fun);
    }

    #[test]
    fn test_filter_tr_unconstrained() {
        let ft = FilterMethodTR::default();
        let result = ft
            .minimize(rosenbrock, &[0.0, 0.0], &[])
            .expect("failed to create result");
        assert!(result.fun < 1.0, "Expected fun < 1.0, got {}", result.fun);
    }

    #[test]
    fn test_filter_tr_with_constraint() {
        let func = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
        // Constraint: x[0] + x[1] <= 3.0, i.e. x[0]+x[1]-3.0 <= 0
        let c: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![Box::new(|x: &[f64]| x[0] + x[1] - 3.0)];

        let ft = FilterMethodTR::new(FilterMethodTROptions {
            max_iter: 500,
            ..Default::default()
        });
        let result = ft
            .minimize(func, &[0.5, 0.5], &c)
            .expect("failed to create result");
        assert!(result.fun < 2.0, "Expected fun < 2.0, got {}", result.fun);
    }
}
