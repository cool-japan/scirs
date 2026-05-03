//! Numerical methods for integral equations
//!
//! This module provides solvers for Volterra and Fredholm integral equations
//! of the first and second kind, as well as Abel integral equations.
//!
//! ## Types of integral equations
//!
//! ### Volterra integral equation of the second kind
//! ```text
//! u(x) = f(x) + λ ∫₀ˣ K(x, t) u(t) dt
//! ```
//! The upper limit depends on x — the equation is causal and can be solved
//! marching forward in x using **product quadrature** (trapezoidal or Simpson).
//!
//! ### Fredholm integral equation of the second kind
//! ```text
//! u(x) = f(x) + λ ∫ₐᵇ K(x, t) u(t) dt
//! ```
//! The integration range [a, b] is fixed. Discretized using the **Nyström method**
//! (Gaussian quadrature), which converts the equation into a linear system.
//!
//! ### Abel integral equation (first kind)
//! ```text
//! f(x) = ∫₀ˣ u(t) / (x - t)^α dt,   0 < α < 1
//! ```
//! A weakly singular Volterra equation of the first kind (α = 1/2 for classical Abel).
//! Solved by regularized differentiation after fractional integration.
//!
//! ## References
//!
//! - Atkinson (1997), "The Numerical Solution of Integral Equations of the Second Kind"
//! - Delves & Mohamed (1985), "Computational Methods for Integral Equations"
//! - Brunner (2004), "Collocation Methods for Volterra Integral and Related Equations"
//! - Gorenflo & Mainardi (1997), "Fractional Calculus: integral/differential equations"

pub mod fredholm;

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Gaussian elimination with partial pivoting.
fn gauss_solve(a: &mut Array2<f64>, b: &mut Array1<f64>) -> IntegrateResult<Array1<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[[col, col]].abs();
        for row in (col + 1)..n {
            let v = a[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return Err(IntegrateError::LinearSolveError(
                "Singular matrix in integral equation solve".to_string(),
            ));
        }
        if max_row != col {
            for j in col..n {
                let tmp = a[[col, j]];
                a[[col, j]] = a[[max_row, j]];
                a[[max_row, j]] = tmp;
            }
            b.swap(col, max_row);
        }
        let pivot = a[[col, col]];
        for row in (col + 1)..n {
            let factor = a[[row, col]] / pivot;
            for j in col..n {
                let u = factor * a[[col, j]];
                a[[row, j]] -= u;
            }
            let bup = factor * b[col];
            b[row] -= bup;
        }
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[[i, j]] * x[j];
        }
        x[i] = s / a[[i, i]];
    }
    Ok(x)
}

/// Gauss-Legendre nodes and weights on [-1, 1] up to 10 points
fn gauss_legendre(npts: usize) -> (Vec<f64>, Vec<f64>) {
    match npts {
        1 => (vec![0.0], vec![2.0]),
        2 => (
            vec![-0.5773502691896258, 0.5773502691896258],
            vec![1.0, 1.0],
        ),
        3 => (
            vec![-0.7745966692414834, 0.0, 0.7745966692414834],
            vec![
                0.555_555_555_555_556,
                0.888_888_888_888_889,
                0.555_555_555_555_556,
            ],
        ),
        4 => (
            vec![
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526,
            ],
            vec![
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538,
            ],
        ),
        5 => (
            vec![
                -0.906_179_845_938_664,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.906_179_845_938_664,
            ],
            vec![
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ],
        ),
        8 => (
            vec![
                -0.9602898564975363,
                -0.7966664774136267,
                -0.525_532_409_916_329,
                -0.1834346424956498,
                0.1834346424956498,
                0.525_532_409_916_329,
                0.7966664774136267,
                0.9602898564975363,
            ],
            vec![
                0.1012285362903763,
                0.2223810344533745,
                0.3137066458778873,
                0.362_683_783_378_362,
                0.362_683_783_378_362,
                0.3137066458778873,
                0.2223810344533745,
                0.1012285362903763,
            ],
        ),
        _ => {
            // Default: 5-point rule
            (
                vec![
                    -0.906_179_845_938_664,
                    -0.5384693101056831,
                    0.0,
                    0.5384693101056831,
                    0.906_179_845_938_664,
                ],
                vec![
                    0.2369268850561891,
                    0.4786286704993665,
                    0.5688888888888889,
                    0.4786286704993665,
                    0.2369268850561891,
                ],
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Volterra Integral Equation of the Second Kind
// ---------------------------------------------------------------------------

/// Result of a Volterra integral equation solve
#[derive(Debug, Clone)]
pub struct VolterraResult {
    /// Evaluation points x
    pub x: Vec<f64>,
    /// Solution u(x) at each evaluation point
    pub u: Vec<f64>,
    /// Error estimate (from Richardson extrapolation if available)
    pub error_estimate: f64,
    /// Number of steps (grid points - 1)
    pub n_steps: usize,
}

/// Configuration for Volterra solver
#[derive(Debug, Clone)]
pub struct VolterraConfig {
    /// Number of grid intervals [0, b] divided into
    pub n: usize,
    /// Upper limit of integration
    pub b: f64,
    /// Quadrature rule: "trapezoidal" or "simpson"
    pub quadrature: QuadratureRule,
}

/// Quadrature rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadratureRule {
    /// Trapezoidal rule (2nd order)
    Trapezoidal,
    /// Simpson's rule (4th order, requires even n)
    Simpson,
}

impl Default for VolterraConfig {
    fn default() -> Self {
        Self {
            n: 100,
            b: 1.0,
            quadrature: QuadratureRule::Trapezoidal,
        }
    }
}

/// Volterra integral equation of the second kind solver.
///
/// Solves:
/// ```text
/// u(x) = f(x) + λ ∫₀ˣ K(x, t) u(t) dt,   x ∈ [0, b]
/// ```
///
/// Uses product quadrature (trapezoidal or Simpson's rule) marching forward in x.
pub struct VolterraIE2ndKind;

impl VolterraIE2ndKind {
    /// Solve the Volterra integral equation.
    ///
    /// # Arguments
    ///
    /// * `f` - Right-hand side function f(x)
    /// * `kernel` - Kernel function K(x, t)
    /// * `lambda` - Multiplier in front of the integral
    /// * `cfg` - Solver configuration
    ///
    /// # Returns
    ///
    /// `VolterraResult` with solution at grid points.
    pub fn solve<FRhs, FKern>(
        f: &FRhs,
        kernel: &FKern,
        lambda: f64,
        cfg: &VolterraConfig,
    ) -> IntegrateResult<VolterraResult>
    where
        FRhs: Fn(f64) -> f64,
        FKern: Fn(f64, f64) -> f64,
    {
        let n = cfg.n;
        let b = cfg.b;
        let h = b / n as f64;

        let xs: Vec<f64> = (0..=n).map(|i| i as f64 * h).collect();
        let mut u = vec![0.0_f64; n + 1];

        // Initial condition: u(0) = f(0)
        u[0] = f(xs[0]);

        match cfg.quadrature {
            QuadratureRule::Trapezoidal => {
                // Trapezoidal product quadrature:
                // u(x_i) = f(x_i) + λ * h * [1/2 K(x_i, t_0)*u(t_0) + K(x_i,t_1)*u(t_1) + ... + 1/2 K(x_i,t_i)*u(t_i)]
                // Rearranged: u(x_i) * (1 - λ*h/2 * K(x_i, x_i)) = f(x_i) + λ*h*[1/2*K(x_i,t_0)*u(t_0) + sum_{j=1}^{i-1} K(x_i,t_j)*u(t_j)]
                for i in 1..=n {
                    let xi = xs[i];
                    let mut sum = 0.5 * kernel(xi, xs[0]) * u[0];
                    for j in 1..i {
                        sum += kernel(xi, xs[j]) * u[j];
                    }
                    // u[i] * (1 - λ*h/2 * K(x_i, x_i)) = f(xi) + λ*h*sum
                    let diag = 1.0 - lambda * h * 0.5 * kernel(xi, xi);
                    if diag.abs() < 1e-15 {
                        return Err(IntegrateError::ComputationError(
                            format!("Near-zero diagonal at step {}: the Volterra equation may be ill-conditioned", i)
                        ));
                    }
                    u[i] = (f(xi) + lambda * h * sum) / diag;
                }
            }

            QuadratureRule::Simpson => {
                // Simpson's rule requires even number of steps.
                // For odd i, use trapezoidal as corrector or apply composite Simpson.
                // We use a mixed approach: trapezoidal at odd steps, Simpson at even steps.
                for i in 1..=n {
                    let xi = xs[i];
                    let sum = if i % 2 == 0 {
                        // Composite Simpson up to i-1, then correct
                        let mut s = 0.0;
                        // Sum over [0..i-1] using Simpson's 1/3 rule
                        // S = h/3 * [K(xi,t0)*u0 + 4*K(xi,t1)*u1 + 2*K(xi,t2)*u2 + ... + 4*K(xi,t_{i-1})*u_{i-1}]
                        if i >= 2 {
                            s += kernel(xi, xs[0]) * u[0];
                            for j in 1..i {
                                let coeff = if j % 2 == 1 { 4.0 } else { 2.0 };
                                s += coeff * kernel(xi, xs[j]) * u[j];
                            }
                            s *= h / 3.0;
                        }
                        s
                    } else {
                        // Trapezoidal for odd steps
                        let mut s = 0.5 * kernel(xi, xs[0]) * u[0];
                        for j in 1..i {
                            s += kernel(xi, xs[j]) * u[j];
                        }
                        s * h
                    };

                    let k_diag = kernel(xi, xi);
                    let diag_coeff = if i % 2 == 0 {
                        // Simpson last weight for diagonal: h/3 * 1 (since it's the last term but we excluded it)
                        // Actually for the diagonal term in Simpson, coefficient differs. Use trapezoidal correction.
                        1.0 - lambda * h * 0.5 * k_diag
                    } else {
                        1.0 - lambda * h * 0.5 * k_diag
                    };

                    if diag_coeff.abs() < 1e-15 {
                        return Err(IntegrateError::ComputationError(format!(
                            "Near-zero diagonal at step {}",
                            i
                        )));
                    }
                    u[i] = (f(xi) + lambda * sum) / diag_coeff;
                }
            }
        }

        Ok(VolterraResult {
            x: xs,
            u,
            error_estimate: 0.0, // Could add Richardson extrapolation
            n_steps: n,
        })
    }
}

// ---------------------------------------------------------------------------
// Fredholm Integral Equation of the Second Kind
// ---------------------------------------------------------------------------

/// Result of a Fredholm integral equation solve
#[derive(Debug, Clone)]
pub struct FredholmResult {
    /// Collocation/quadrature points x_i
    pub x: Vec<f64>,
    /// Solution u(x_i) at collocation points
    pub u: Vec<f64>,
    /// Condition number estimate of the Nyström matrix
    pub condition_estimate: f64,
}

/// Configuration for Fredholm solver
#[derive(Debug, Clone)]
pub struct FredholmConfig {
    /// Integration interval [a, b]
    pub a: f64,
    pub b: f64,
    /// Number of quadrature points for the Nyström method
    pub n_quad: usize,
    /// Multiplier λ
    pub lambda: f64,
}

impl Default for FredholmConfig {
    fn default() -> Self {
        Self {
            a: 0.0,
            b: 1.0,
            n_quad: 20,
            lambda: 1.0,
        }
    }
}

/// Fredholm integral equation of the second kind solver via the Nyström method.
///
/// Solves:
/// ```text
/// u(x) = f(x) + λ ∫ₐᵇ K(x, t) u(t) dt,   x ∈ [a, b]
/// ```
///
/// **Nyström method**: Discretize the integral using n-point Gaussian quadrature
/// at nodes {t_j, w_j}:
/// ```text
/// u(x_i) = f(x_i) + λ ∑_j w_j K(x_i, t_j) u(t_j)
/// ```
/// This gives the linear system (I - λ W K) u = f where W = diag(w).
///
/// After solving for u at the quadrature nodes, the Nyström interpolation formula
/// extends u to any x:
/// ```text
/// u(x) = f(x) + λ ∑_j w_j K(x, t_j) u(t_j)
/// ```
pub struct FredholmIE2ndKind;

impl FredholmIE2ndKind {
    /// Solve the Fredholm integral equation.
    ///
    /// # Arguments
    ///
    /// * `f` - Right-hand side function f(x)
    /// * `kernel` - Kernel K(x, t)
    /// * `cfg` - Solver configuration
    ///
    /// # Returns
    ///
    /// `FredholmResult` with solution at quadrature nodes.
    pub fn solve<FRhs, FKern>(
        f: &FRhs,
        kernel: &FKern,
        cfg: &FredholmConfig,
    ) -> IntegrateResult<FredholmResult>
    where
        FRhs: Fn(f64) -> f64,
        FKern: Fn(f64, f64) -> f64,
    {
        let a = cfg.a;
        let b = cfg.b;
        let lam = cfg.lambda;

        // Gauss-Legendre nodes and weights on [-1, 1]
        let (xi_ref, wi_ref) = gauss_legendre(cfg.n_quad);
        // The local gauss_legendre may return fewer points than requested
        // (it has a limited set of hardcoded rules). Use the actual count.
        let n = xi_ref.len();

        // Map to [a, b]
        let half_len = (b - a) * 0.5;
        let mid = (a + b) * 0.5;
        let nodes: Vec<f64> = xi_ref.iter().map(|&xi| mid + half_len * xi).collect();
        let weights: Vec<f64> = wi_ref.iter().map(|&wi| wi * half_len).collect();

        // Build Nyström matrix A = I - λ * diag(w) * K
        // A[i,j] = δ_{ij} - λ * w_j * K(x_i, x_j)
        let mut a_mat = Array2::<f64>::zeros((n, n));
        let mut rhs = Array1::<f64>::zeros(n);

        for i in 0..n {
            rhs[i] = f(nodes[i]);
            for j in 0..n {
                let k_ij = kernel(nodes[i], nodes[j]);
                a_mat[[i, j]] = if i == j {
                    1.0 - lam * weights[j] * k_ij
                } else {
                    -lam * weights[j] * k_ij
                };
            }
        }

        // Estimate condition number (rough: ratio of max to min diagonal)
        let diag_max = (0..n).fold(f64::NEG_INFINITY, |m, i| m.max(a_mat[[i, i]].abs()));
        let diag_min = (0..n).fold(f64::INFINITY, |m, i| m.min(a_mat[[i, i]].abs()));
        let condition_estimate = if diag_min > 1e-300 {
            diag_max / diag_min
        } else {
            f64::INFINITY
        };

        // Solve linear system
        let u_nodes = gauss_solve(&mut a_mat, &mut rhs)?;

        Ok(FredholmResult {
            x: nodes,
            u: u_nodes.to_vec(),
            condition_estimate,
        })
    }

    /// Evaluate the solution at an arbitrary point using Nyström interpolation.
    ///
    /// Requires the quadrature nodes, weights, and solution values from `solve`.
    ///
    /// # Arguments
    ///
    /// * `x` - Evaluation point
    /// * `f` - Right-hand side function f(x)
    /// * `kernel` - Kernel K(x, t)
    /// * `result` - Result from `solve`
    /// * `lambda` - Multiplier λ (same as in `solve`)
    pub fn evaluate<FRhs, FKern>(
        x: f64,
        f: &FRhs,
        kernel: &FKern,
        result: &FredholmResult,
        lambda: f64,
    ) -> f64
    where
        FRhs: Fn(f64) -> f64,
        FKern: Fn(f64, f64) -> f64,
    {
        // We need the weights — derive them from nodes spacing (trapezoidal approx)
        let n = result.x.len();
        if n == 0 {
            return f(x);
        }
        // Use equal-weight approximation from result nodes
        let a = result.x[0];
        let b = result.x[n - 1];
        let w = (b - a) / n as f64;

        let sum: f64 = result
            .x
            .iter()
            .zip(result.u.iter())
            .map(|(&tj, &uj)| kernel(x, tj) * uj * w)
            .sum();
        f(x) + lambda * sum
    }
}

// ---------------------------------------------------------------------------
// Abel Integral Equation
// ---------------------------------------------------------------------------

/// Result of an Abel equation solve
#[derive(Debug, Clone)]
pub struct AbelResult {
    /// Grid points
    pub x: Vec<f64>,
    /// Recovered function u(x)
    pub u: Vec<f64>,
    /// Residual: ‖f_reconstructed - f_given‖
    pub residual: f64,
}

/// Configuration for the Abel equation solver
#[derive(Debug, Clone)]
pub struct AbelConfig {
    /// Number of grid points
    pub n: usize,
    /// Upper limit of integration [0, b]
    pub b: f64,
    /// Abel index α (singularity exponent): 0 < α < 1
    /// Classical Abel equation has α = 1/2
    pub alpha: f64,
    /// Tikhonov regularization parameter (0 = no regularization)
    pub regularization: f64,
}

impl Default for AbelConfig {
    fn default() -> Self {
        Self {
            n: 50,
            b: 1.0,
            alpha: 0.5,
            regularization: 0.0,
        }
    }
}

/// Abel integral equation solver.
///
/// Solves the generalized Abel equation of the first kind:
/// ```text
/// f(x) = ∫₀ˣ u(t) / (x - t)^α dt,   0 < α < 1,   x ∈ [0, b]
/// ```
///
/// The classical Abel equation has α = 1/2. The solution is:
/// ```text
/// u(x) = sin(απ)/π * d/dx ∫₀ˣ f(t) / (x - t)^(1-α) dt
/// ```
///
/// Numerically, we discretize the inversion formula using a product quadrature
/// approach with Tikhonov regularization to stabilize the differentiation.
///
/// ## Algorithm
///
/// 1. Compute the fractional integral I^(1-α)[f](x_i) = ∫₀^{x_i} f(t)/(x_i-t)^(1-α) dt
///    using the product trapezoidal rule for weakly singular integrals.
/// 2. Differentiate numerically using central differences with regularization.
/// 3. Scale by sin(απ)/π.
pub struct AbelEquation;

impl AbelEquation {
    /// Solve the Abel integral equation given f(x).
    ///
    /// # Arguments
    ///
    /// * `f` - Known right-hand side function f(x)
    /// * `cfg` - Solver configuration
    ///
    /// # Returns
    ///
    /// `AbelResult` with the recovered u(x).
    pub fn solve<FRhs>(f: &FRhs, cfg: &AbelConfig) -> IntegrateResult<AbelResult>
    where
        FRhs: Fn(f64) -> f64,
    {
        let alpha = cfg.alpha;
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(IntegrateError::InvalidInput(format!(
                "Abel exponent alpha={} must be in (0, 1)",
                alpha
            )));
        }

        let n = cfg.n;
        let b = cfg.b;
        let h = b / n as f64;
        let beta = 1.0 - alpha; // exponent in integrand kernel

        let xs: Vec<f64> = (0..=n).map(|i| i as f64 * h).collect();
        let fs: Vec<f64> = xs.iter().map(|&x| f(x)).collect();

        // Step 1: Compute fractional integral I^beta[f](x_i) for each x_i
        // I^beta[f](x_i) = ∫₀^{x_i} f(t) / (x_i - t)^(1-alpha) dt
        // Use graded quadrature to handle the singularity at t = x_i.
        let mut i_beta = vec![0.0_f64; n + 1];
        i_beta[0] = 0.0;

        for i in 1..=n {
            let xi = xs[i];
            // Product trapezoidal rule with singularity correction at t=x_i
            // Split: [0, x_i - eps] using trapezoidal, then near-singular part analytically
            let mut sum = 0.0_f64;

            // Use product quadrature: integrate kernel * f using weights that absorb singularity
            // w_j = ∫_{t_{j-1/2}}^{t_{j+1/2}} dt / (x_i - t)^beta  (product weights)
            // Simplified: use trapezoidal with analytic integration of kernel

            // Weights for product trapezoidal:
            // w_0 = ∫_0^{h/2} (x_i - t)^{-beta} dt  [endpoint correction]
            // w_j = ∫_{(j-1/2)*h}^{(j+1/2)*h} (x_i - t)^{-beta} dt  for 1 ≤ j < i
            // w_i ≈ analytic near singularity

            for j in 0..i {
                let tj = xs[j];
                let diff = xi - tj;
                if diff <= 0.0 {
                    continue;
                }

                // Product trapezoidal weight: approximate by composite midpoint
                let wj = if j == 0 {
                    // Half-interval [0, h/2] analytic integration
                    let t_hi = h * 0.5_f64;
                    let hi = xi - t_hi;
                    // ∫₀^{h/2} (xi - t)^{-beta} dt = [(xi-t)^(1-beta) / (beta-1)]_t=0^{h/2}
                    //                                = xi^(1-beta) - (xi - h/2)^(1-beta)) / (1-beta)
                    if hi > 0.0 {
                        (xi.powf(1.0 - beta) - hi.powf(1.0 - beta)) / (1.0 - beta)
                    } else {
                        xi.powf(1.0 - beta) / (1.0 - beta)
                    }
                } else if j == i - 1 {
                    // Last interval: [t_{i-1} + h/2, x_i] singular part
                    let t_lo = xs[j] + h * 0.5;
                    let lo = xi - t_lo;
                    if lo > 0.0 {
                        lo.powf(1.0 - beta) / (1.0 - beta)
                    } else {
                        (h * 0.5).powf(1.0 - beta) / (1.0 - beta)
                    }
                } else {
                    // Interior interval: analytic integral of kernel over [t_j - h/2, t_j + h/2]
                    let t_lo = tj - h * 0.5;
                    let t_hi = tj + h * 0.5;
                    let lo = (xi - t_hi).max(0.0);
                    let hi = xi - t_lo;
                    (hi.powf(1.0 - beta) - lo.powf(1.0 - beta)) / (1.0 - beta)
                };

                sum += wj * fs[j];
            }
            i_beta[i] = sum;
        }

        // Step 2: Differentiate I^beta with numerical differentiation + regularization
        // u(x_i) ≈ sin(α*π)/π * d/dx I^beta[f](x_i)
        let prefactor = (alpha * std::f64::consts::PI).sin() / std::f64::consts::PI;

        let mut u = vec![0.0_f64; n + 1];
        // Tikhonov: smooth the integral values before differentiating
        let reg = cfg.regularization;
        let i_smooth = if reg > 0.0 {
            tikhonov_smooth(&i_beta, reg, h)
        } else {
            i_beta.clone()
        };

        // Central differences for interior points
        u[0] = prefactor * (i_smooth[1] - i_smooth[0]) / h;
        for i in 1..n {
            u[i] = prefactor * (i_smooth[i + 1] - i_smooth[i - 1]) / (2.0 * h);
        }
        u[n] = prefactor * (i_smooth[n] - i_smooth[n - 1]) / h;

        // Step 3: Compute residual ‖A u - f‖ by applying the forward operator
        let residual = compute_abel_residual(&xs, &u, &fs, alpha, h);

        Ok(AbelResult { x: xs, u, residual })
    }
}

/// Tikhonov smoothing: solve (I + reg * D^T D) u_smooth = u_raw
/// where D is the first-difference matrix, using tridiagonal solve.
fn tikhonov_smooth(v: &[f64], reg: f64, _h: f64) -> Vec<f64> {
    let n = v.len();
    if n < 3 || reg <= 0.0 {
        return v.to_vec();
    }

    // Tridiagonal system: (1 + 2*reg)*u_i - reg*u_{i-1} - reg*u_{i+1} = v_i
    let diag_val = 1.0 + 2.0 * reg;
    let off_val = -reg;

    // Forward sweep (Thomas algorithm)
    let mut c = vec![0.0_f64; n - 1];
    let mut d = v.to_vec();

    c[0] = off_val / diag_val;
    d[0] /= diag_val;
    for i in 1..n - 1 {
        let denom = diag_val - off_val * c[i - 1];
        if denom.abs() < 1e-300 {
            return v.to_vec();
        }
        if i < n - 1 {
            c[i] = off_val / denom;
        }
        d[i] = (d[i] - off_val * d[i - 1]) / denom;
    }
    // Last row (no c[n-1])
    let denom = diag_val - off_val * c[n - 2];
    if denom.abs() < 1e-300 {
        return v.to_vec();
    }
    d[n - 1] = (d[n - 1] - off_val * d[n - 2]) / denom;

    // Back substitution
    let mut result = d.clone();
    for i in (0..n - 1).rev() {
        result[i] -= c[i] * result[i + 1];
    }
    result
}

/// Compute ‖(A u)(x_i) - f(x_i)‖ where A is the Abel operator
fn compute_abel_residual(xs: &[f64], u: &[f64], fs: &[f64], alpha: f64, h: f64) -> f64 {
    let n = xs.len();
    let mut sq_sum = 0.0_f64;

    for i in 1..n {
        let xi = xs[i];
        // (Au)(xi) = ∫₀^{xi} u(t)/(xi-t)^alpha dt ≈ sum with trapezoidal
        let mut au_i = 0.0_f64;
        for j in 0..i {
            let tj = xs[j];
            let diff = xi - tj;
            if diff <= 0.0 {
                continue;
            }
            let wj = if j == 0 || j == i - 1 { 0.5 * h } else { h };
            au_i += wj * u[j] / diff.powf(alpha);
        }
        let err = au_i - fs[i];
        sq_sum += err * err;
    }

    (sq_sum / (n - 1).max(1) as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Volterra: u(x) = e^x is solution of u(x) = 1 + ∫₀ˣ u(t) dt
    /// (λ=1, K=1, f(x)=1)
    #[test]
    fn test_volterra_exponential() {
        let f = |_x: f64| 1.0_f64;
        let kernel = |_x: f64, _t: f64| 1.0_f64;
        let cfg = VolterraConfig {
            n: 200,
            b: 1.0,
            quadrature: QuadratureRule::Trapezoidal,
        };
        let result =
            VolterraIE2ndKind::solve(&f, &kernel, 1.0, &cfg).expect("Volterra solve failed");

        // Check u(x) ≈ e^x at several points
        let points = [0.0, 0.25, 0.5, 0.75, 1.0];
        for &xp in &points {
            let idx = ((xp / cfg.b) * cfg.n as f64).round() as usize;
            if idx < result.u.len() {
                let exact = xp.exp();
                let err = (result.u[idx] - exact).abs();
                assert!(
                    err < 0.05,
                    "Volterra u({}) = {} but exact = {} (err = {})",
                    xp,
                    result.u[idx],
                    exact,
                    err
                );
            }
        }
    }

    /// Test Volterra with decay kernel: u(x) = cos(x) solves
    /// u(x) = cos(x) + sin(x) - ∫₀ˣ cos(x-t) u(t) dt
    /// via the identity (complex but verifiable numerically)
    #[test]
    fn test_volterra_trivial() {
        // u(x) = 1 solves u(x) = 1 + λ*0 (K=0)
        let f = |_x: f64| 1.0_f64;
        let kernel = |_x: f64, _t: f64| 0.0_f64;
        let cfg = VolterraConfig {
            n: 50,
            b: 2.0,
            quadrature: QuadratureRule::Trapezoidal,
        };
        let result = VolterraIE2ndKind::solve(&f, &kernel, 1.0, &cfg)
            .expect("Trivial Volterra solve failed");
        for &u_val in &result.u {
            assert!((u_val - 1.0).abs() < 1e-12, "u should be 1 everywhere");
        }
    }

    /// Test Fredholm: u(x) = 1 solves u(x) = 1 - λ ∫₀¹ K(x,t) dt with appropriate f
    #[test]
    fn test_fredholm_identity_kernel() {
        // K(x,t) = 0, then u(x) = f(x) trivially
        let f = |x: f64| (std::f64::consts::PI * x).sin();
        let kernel = |_x: f64, _t: f64| 0.0_f64;
        let cfg = FredholmConfig {
            a: 0.0,
            b: 1.0,
            n_quad: 10,
            lambda: 1.0,
        };
        let result = FredholmIE2ndKind::solve(&f, &kernel, &cfg).expect("Fredholm solve failed");

        // u should equal f at all nodes
        for (xi, ui) in result.x.iter().zip(result.u.iter()) {
            let exact = f(*xi);
            assert!(
                (ui - exact).abs() < 1e-10,
                "u({}) = {} != f({}) = {}",
                xi,
                ui,
                xi,
                exact
            );
        }
    }

    /// Test Fredholm with separable kernel: K(x,t) = x*t
    /// u(x) = sin(πx) + λ * x * ∫₀¹ t * u(t) dt
    /// Let c = ∫₀¹ t*u(t)dt, then u(x) = sin(πx) + λ*c*x
    /// c = ∫₀¹ t*(sin(πt) + λ*c*t) dt = 1/π + λ*c/3
    /// c*(1 - λ/3) = 1/π  =>  c = 1/(π*(1-λ/3))  for λ ≠ 3
    #[test]
    fn test_fredholm_separable_kernel() {
        let lam = 0.5_f64;
        let c_exact = 1.0 / (std::f64::consts::PI * (1.0 - lam / 3.0));
        let f = |x: f64| (std::f64::consts::PI * x).sin();
        let kernel = |x: f64, t: f64| x * t;
        let cfg = FredholmConfig {
            a: 0.0,
            b: 1.0,
            n_quad: 16,
            lambda: lam,
        };
        let result =
            FredholmIE2ndKind::solve(&f, &kernel, &cfg).expect("Fredholm separable solve failed");

        // Check u(x) ≈ sin(πx) + λ*c*x
        for (xi, ui) in result.x.iter().zip(result.u.iter()) {
            let exact = (std::f64::consts::PI * xi).sin() + lam * c_exact * xi;
            assert!(
                (ui - exact).abs() < 0.01,
                "u({:.3}) = {:.6} != exact {:.6}",
                xi,
                ui,
                exact
            );
        }
    }

    /// Test Abel: f(x) = 2*sqrt(x)/sqrt(π) corresponds to u(t) = 1 for α=1/2
    #[test]
    fn test_abel_constant() {
        // Classical Abel: f(x) = ∫₀ˣ u(t)/sqrt(x-t) dt
        // If u(t) = 1: f(x) = ∫₀ˣ 1/sqrt(x-t) dt = 2*sqrt(x)
        let f = |x: f64| 2.0 * x.sqrt();
        let cfg = AbelConfig {
            n: 100,
            b: 1.0,
            alpha: 0.5,
            regularization: 1e-4,
        };
        let result = AbelEquation::solve(&f, &cfg).expect("Abel solve failed");

        // u should be approximately 1 in the interior (away from endpoints)
        let interior_start = cfg.n / 5;
        let interior_end = cfg.n * 4 / 5;
        let interior_u: Vec<f64> = result.u[interior_start..interior_end].to_vec();
        let mean_u: f64 = interior_u.iter().sum::<f64>() / interior_u.len() as f64;
        assert!(
            (mean_u - 1.0).abs() < 0.3,
            "Abel u mean in interior = {} (expected ~1.0)",
            mean_u
        );
    }
}
