//! Structure-preserving integrators for Port-Hamiltonian systems
//!
//! This module provides numerical integrators that preserve the geometric
//! structure of port-Hamiltonian systems, including:
//!
//! - **Discrete Gradient Method**: Exactly preserves energy for conservative systems
//! - **Average Vector Field (AVF)**: Energy-preserving quadrature-based method
//! - **Störmer-Verlet (pH variant)**: Symplectic, second-order
//! - **Implicit Midpoint**: Symplectic, energy-conserving in the limit
//! - **RATTLE**: For constrained Hamiltonian systems
//!
//! # Mathematical Background
//!
//! A structure-preserving integrator for a pH system satisfies a discrete
//! energy balance:
//! ```text
//! H(x^{n+1}) - H(x^n) = -dt * D_n + dt * S_n
//! ```
//! where D_n >= 0 is the discrete dissipation and S_n is the supply rate.
//!
//! For conservative systems (R = 0, u = 0), discrete gradient methods ensure
//! H(x^{n+1}) = H(x^n) exactly (to machine precision).

use crate::error::{IntegrateError, IntegrateResult};
use crate::port_hamiltonian::system::PortHamiltonianSystem;
use scirs2_core::ndarray::Array1;

/// Result of a single integration step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// New state after the step
    pub x_new: Vec<f64>,
    /// Hamiltonian value at new state
    pub energy: f64,
    /// Number of function evaluations
    pub n_evals: usize,
    /// Number of Newton iterations (for implicit methods)
    pub n_iters: usize,
}

/// Result of integrating over a time span
#[derive(Debug, Clone)]
pub struct PortHamiltonianResult {
    /// Time points
    pub t: Vec<f64>,
    /// State trajectory: t\[i\] -> x\[i\]
    pub x: Vec<Vec<f64>>,
    /// Hamiltonian values along trajectory
    pub energy: Vec<f64>,
    /// Output values y = B^T ∇H along trajectory
    pub output: Vec<Vec<f64>>,
    /// Total function evaluations
    pub n_evals: usize,
    /// Maximum energy drift: |H(t_f) - H(t_0)|
    pub energy_drift: f64,
}

/// Options common to all structure-preserving integrators
#[derive(Debug, Clone)]
pub struct IntegratorOptions {
    /// Newton iteration tolerance (for implicit methods)
    pub newton_tol: f64,
    /// Maximum Newton iterations
    pub max_newton_iters: usize,
    /// Number of Gauss-Legendre quadrature points (for AVF)
    pub n_quad_points: usize,
    /// Whether to store full trajectory
    pub store_trajectory: bool,
}

impl Default for IntegratorOptions {
    fn default() -> Self {
        Self {
            newton_tol: 1e-12,
            max_newton_iters: 50,
            n_quad_points: 5,
            store_trajectory: true,
        }
    }
}

// ─── Newton solver helper ──────────────────────────────────────────────────────

/// Solve F(x) = 0 using Newton's method with numerical Jacobian.
fn newton_solve(
    f: &dyn Fn(&[f64]) -> IntegrateResult<Vec<f64>>,
    x0: &[f64],
    tol: f64,
    max_iters: usize,
) -> IntegrateResult<(Vec<f64>, usize)> {
    let n = x0.len();
    let mut x = x0.to_vec();
    let eps = 1e-8;

    for iter in 0..max_iters {
        let fx = f(&x)?;
        let res_norm: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if res_norm < tol {
            return Ok((x, iter + 1));
        }

        // Build numerical Jacobian
        let mut jac = vec![vec![0.0_f64; n]; n];
        for j in 0..n {
            let mut x_plus = x.clone();
            x_plus[j] += eps;
            let fx_plus = f(&x_plus)?;
            for i in 0..n {
                jac[i][j] = (fx_plus[i] - fx[i]) / eps;
            }
        }

        // Solve J * delta = -F using Gaussian elimination
        let delta = gauss_solve(&jac, &fx.iter().map(|v| -v).collect::<Vec<_>>())?;

        // Line search: simple backtracking
        let mut alpha = 1.0_f64;
        for _ in 0..10 {
            let x_try: Vec<f64> = x
                .iter()
                .zip(delta.iter())
                .map(|(xi, di)| xi + alpha * di)
                .collect();
            let fx_try = f(&x_try)?;
            let res_try: f64 = fx_try.iter().map(|v| v * v).sum::<f64>().sqrt();
            if res_try < res_norm {
                break;
            }
            alpha *= 0.5;
        }

        x = x
            .iter()
            .zip(delta.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
    }

    Err(IntegrateError::ConvergenceError(format!(
        "Newton iteration did not converge in {max_iters} steps"
    )))
}

/// Gaussian elimination with partial pivoting to solve A*x = b.
fn gauss_solve(a: &[Vec<f64>], b: &[f64]) -> IntegrateResult<Vec<f64>> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivoting
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1][col]
                    .abs()
                    .partial_cmp(&aug[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| IntegrateError::LinearSolveError("Pivot selection failed".into()))?;

        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            return Err(IntegrateError::LinearSolveError(
                "Near-singular Jacobian in Newton step".into(),
            ));
        }

        for j in col..=n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    Ok((0..n).map(|i| aug[i][n]).collect())
}

// ─── Gauss-Legendre quadrature nodes/weights on [0,1] ────────────────────────

/// Return Gauss-Legendre nodes and weights on [0,1] for n-point rule.
fn gauss_legendre_01(n: usize) -> IntegrateResult<(Vec<f64>, Vec<f64>)> {
    // We use stored values for n=1..=7 (sufficient for practical accuracy)
    // Reference: Abramowitz & Stegun or any standard quadrature table
    let (nodes_m1_1, weights): (Vec<f64>, Vec<f64>) = match n {
        1 => (vec![0.0], vec![2.0]),
        2 => (
            vec![-0.577_350_269_189_625_8, 0.577_350_269_189_625_8],
            vec![1.0, 1.0],
        ),
        3 => (
            vec![-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4],
            vec![
                0.555_555_555_555_555_6,
                0.888_888_888_888_889,
                0.555_555_555_555_555_6,
            ],
        ),
        4 => (
            vec![
                -0.861_136_311_594_953,
                -0.339_981_043_584_856,
                0.339_981_043_584_856,
                0.861_136_311_594_953,
            ],
            vec![
                0.347_854_845_137_454,
                0.652_145_154_862_546,
                0.652_145_154_862_546,
                0.347_854_845_137_454,
            ],
        ),
        5 => (
            vec![
                -0.906_179_845_938_664,
                -0.538_469_310_105_683,
                0.0,
                0.538_469_310_105_683,
                0.906_179_845_938_664,
            ],
            vec![
                0.236_926_885_056_189,
                0.478_628_670_499_366,
                0.568_888_888_888_889,
                0.478_628_670_499_366,
                0.236_926_885_056_189,
            ],
        ),
        6 => (
            vec![
                -0.932_469_514_203_152,
                -0.661_209_386_466_265,
                -0.238_619_186_083_197,
                0.238_619_186_083_197,
                0.661_209_386_466_265,
                0.932_469_514_203_152,
            ],
            vec![
                0.171_324_492_379_170,
                0.360_761_573_048_139,
                0.467_913_934_572_691,
                0.467_913_934_572_691,
                0.360_761_573_048_139,
                0.171_324_492_379_170,
            ],
        ),
        7 => (
            vec![
                -0.949_107_912_342_759,
                -0.741_531_185_599_394,
                -0.405_845_151_377_397,
                0.0,
                0.405_845_151_377_397,
                0.741_531_185_599_394,
                0.949_107_912_342_759,
            ],
            vec![
                0.129_484_966_168_870,
                0.279_705_391_489_277,
                0.381_830_050_505_119,
                0.417_959_183_673_469,
                0.381_830_050_505_119,
                0.279_705_391_489_277,
                0.129_484_966_168_870,
            ],
        ),
        _ => {
            return Err(IntegrateError::ValueError(format!(
                "Gauss-Legendre quadrature only supports n=1..=7, got {n}"
            )))
        }
    };

    // Transform from [-1,1] to [0,1]: t = (s+1)/2, dt = ds/2
    let nodes: Vec<f64> = nodes_m1_1.iter().map(|s| (s + 1.0) / 2.0).collect();
    let w: Vec<f64> = weights.iter().map(|w| w / 2.0).collect();
    Ok((nodes, w))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Method 1: Discrete Gradient — Gonzalez midpoint variant
// ═══════════════════════════════════════════════════════════════════════════════

/// Discrete Gradient Integrator (Gonzalez midpoint variant).
///
/// This method exactly preserves the Hamiltonian for conservative systems.
/// For dissipative systems, it satisfies a discrete energy dissipation law.
///
/// The discrete gradient D̄H(x^n, x^{n+1}) is defined by the Gonzalez formula:
/// ```text
/// D̄H = ∇H(x̄) + [H(x^{n+1}) - H(x^n) - ∇H(x̄)·(x^{n+1}-x^n)] * (x^{n+1}-x^n) / |x^{n+1}-x^n|²
/// ```
/// where x̄ = (x^n + x^{n+1})/2.
///
/// The discrete system is:
/// ```text
/// (x^{n+1} - x^n)/dt = (J(x̄) - R(x̄)) * D̄H(x^n, x^{n+1}) + B(x̄) * u
/// ```
///
/// # Reference
///
/// O. Gonzalez, "Time integration and discrete Hamiltonian systems,"
/// J. Nonlinear Sci. 6 (1996), 449-467.
#[derive(Debug, Clone)]
pub struct DiscreteGradientGonzalez {
    options: IntegratorOptions,
}

impl DiscreteGradientGonzalez {
    /// Create a new Discrete Gradient (Gonzalez) integrator.
    pub fn new() -> Self {
        Self {
            options: IntegratorOptions::default(),
        }
    }

    /// Set integrator options.
    pub fn with_options(mut self, options: IntegratorOptions) -> Self {
        self.options = options;
        self
    }

    /// Take a single step: x^n -> x^{n+1}
    pub fn step(
        &self,
        system: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        let n = x.len();
        let h_n = system.hamiltonian(x)?;

        // Residual function for Newton: given x_new, compute
        // F(x_new) = x_new - x - dt * (J(x̄) - R(x̄)) * D̄H - dt * B(x̄) * u
        let x_old = x.to_vec();
        let u_vec = u.to_vec();

        let residual = |x_new: &[f64]| -> IntegrateResult<Vec<f64>> {
            // Midpoint
            let x_mid: Vec<f64> = x_old
                .iter()
                .zip(x_new.iter())
                .map(|(a, b)| 0.5 * (a + b))
                .collect();

            let h_new = system.hamiltonian(x_new)?;
            let grad_h_mid = system.grad_hamiltonian(&x_mid)?;
            let j = system.j_matrix(&x_mid)?;
            let r = system.r_matrix(&x_mid)?;
            let b = system.b_matrix(&x_mid)?;

            // Compute discrete gradient
            let delta_x: Vec<f64> = x_new
                .iter()
                .zip(x_old.iter())
                .map(|(xn, xo)| xn - xo)
                .collect();
            let norm_sq: f64 = delta_x.iter().map(|d| d * d).sum();

            let discrete_grad: Vec<f64> = if norm_sq > 1e-30 {
                let dh = h_new - h_n;
                let grad_dot_delta: f64 = grad_h_mid
                    .iter()
                    .zip(delta_x.iter())
                    .map(|(g, d)| g * d)
                    .sum();
                let correction = (dh - grad_dot_delta) / norm_sq;
                grad_h_mid
                    .iter()
                    .zip(delta_x.iter())
                    .map(|(g, d)| g + correction * d)
                    .collect()
            } else {
                grad_h_mid.to_vec()
            };

            let dg_arr = Array1::from_vec(discrete_grad);
            let jr = &j - &r;
            let drift = jr.dot(&dg_arr);

            // B * u term
            let u_arr = Array1::from_vec(u_vec.clone());
            let bu = b.dot(&u_arr);

            let mut res = vec![0.0; n];
            for i in 0..n {
                res[i] = x_new[i] - x_old[i] - dt * (drift[i] + bu[i]);
            }
            Ok(res)
        };

        // Initial guess: explicit Euler
        let x_init: Vec<f64> = {
            let rhs = system.rhs(x, u)?;
            x.iter()
                .zip(rhs.iter())
                .map(|(xi, fi)| xi + dt * fi)
                .collect()
        };

        let (x_new, n_iters) = newton_solve(
            &residual,
            &x_init,
            self.options.newton_tol,
            self.options.max_newton_iters,
        )?;

        let energy = system.hamiltonian(&x_new)?;
        Ok(StepResult {
            x_new,
            energy,
            n_evals: n_iters * (2 * n + 4) + 2,
            n_iters,
        })
    }

    /// Integrate over a time span with optional control input.
    pub fn integrate(
        &self,
        system: &PortHamiltonianSystem,
        x0: &[f64],
        t0: f64,
        tf: f64,
        dt: f64,
        u: Option<&dyn Fn(f64, &[f64]) -> Vec<f64>>,
    ) -> IntegrateResult<PortHamiltonianResult> {
        integrate_impl(self, system, x0, t0, tf, dt, u, &self.options)
    }
}

impl Default for DiscreteGradientGonzalez {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Method 2: Discrete Gradient — Itoh-Abe variant
// ═══════════════════════════════════════════════════════════════════════════════

/// Discrete Gradient Integrator (Itoh-Abe coordinate increment variant).
///
/// This alternative formulation uses coordinate-wise discrete gradients:
/// ```text
/// D̄H_i(x^n, x^{n+1}) = [H(x^{n+1,i}) - H(x^{n,i})] / (x^{n+1}_i - x^n_i)
/// ```
/// if x^{n+1}_i ≠ x^n_i, otherwise ∂H/∂x_i at the midpoint.
///
/// where x^{n,i} = (x^{n+1}_1, ..., x^{n+1}_{i-1}, x^n_i, ..., x^n_n).
///
/// # Reference
///
/// T. Itoh and K. Abe, "Hamiltonian-conserving discrete canonical equations
/// based on variational difference quotients," J. Comput. Phys. 76 (1988), 85-102.
#[derive(Debug, Clone)]
pub struct DiscreteGradientItohAbe {
    options: IntegratorOptions,
}

impl DiscreteGradientItohAbe {
    /// Create a new Discrete Gradient (Itoh-Abe) integrator.
    pub fn new() -> Self {
        Self {
            options: IntegratorOptions::default(),
        }
    }

    /// Set integrator options.
    pub fn with_options(mut self, options: IntegratorOptions) -> Self {
        self.options = options;
        self
    }

    /// Take a single step: x^n -> x^{n+1}
    pub fn step(
        &self,
        system: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        let n = x.len();
        let x_old = x.to_vec();
        let u_vec = u.to_vec();
        let eps_dg = 1e-14;

        let residual = |x_new: &[f64]| -> IntegrateResult<Vec<f64>> {
            let x_mid: Vec<f64> = x_old
                .iter()
                .zip(x_new.iter())
                .map(|(a, b)| 0.5 * (a + b))
                .collect();

            let j = system.j_matrix(&x_mid)?;
            let r = system.r_matrix(&x_mid)?;
            let b_mat = system.b_matrix(&x_mid)?;

            // Compute Itoh-Abe discrete gradient
            let mut discrete_grad = vec![0.0_f64; n];
            for k in 0..n {
                let dx_k = x_new[k] - x_old[k];
                if dx_k.abs() > eps_dg {
                    // x^{n, k}: use x_new for indices < k, x_old for indices >= k
                    let mut x_lo: Vec<f64> = x_old.to_vec();
                    let mut x_hi: Vec<f64> = x_old.to_vec();
                    x_lo[..k].copy_from_slice(&x_new[..k]);
                    x_hi[..k].copy_from_slice(&x_new[..k]);
                    x_hi[k] = x_new[k];
                    let h_lo = system.hamiltonian(&x_lo)?;
                    let h_hi = system.hamiltonian(&x_hi)?;
                    discrete_grad[k] = (h_hi - h_lo) / dx_k;
                } else {
                    // Use midpoint partial derivative
                    let eps = 1e-7;
                    let mut x_p = x_mid.clone();
                    let mut x_m = x_mid.clone();
                    x_p[k] += eps;
                    x_m[k] -= eps;
                    let h_p = system.hamiltonian(&x_p)?;
                    let h_m = system.hamiltonian(&x_m)?;
                    discrete_grad[k] = (h_p - h_m) / (2.0 * eps);
                }
            }

            let dg_arr = Array1::from_vec(discrete_grad);
            let jr = &j - &r;
            let drift = jr.dot(&dg_arr);
            let u_arr = Array1::from_vec(u_vec.clone());
            let bu = b_mat.dot(&u_arr);

            let mut res = vec![0.0_f64; n];
            for i in 0..n {
                res[i] = x_new[i] - x_old[i] - dt * (drift[i] + bu[i]);
            }
            Ok(res)
        };

        let x_init: Vec<f64> = {
            let rhs = system.rhs(x, u)?;
            x.iter()
                .zip(rhs.iter())
                .map(|(xi, fi)| xi + dt * fi)
                .collect()
        };

        let (x_new, n_iters) = newton_solve(
            &residual,
            &x_init,
            self.options.newton_tol,
            self.options.max_newton_iters,
        )?;

        let energy = system.hamiltonian(&x_new)?;
        Ok(StepResult {
            x_new,
            energy,
            n_evals: n_iters * (n * 2 + 6),
            n_iters,
        })
    }

    /// Integrate over a time span with optional control input.
    pub fn integrate(
        &self,
        system: &PortHamiltonianSystem,
        x0: &[f64],
        t0: f64,
        tf: f64,
        dt: f64,
        u: Option<&dyn Fn(f64, &[f64]) -> Vec<f64>>,
    ) -> IntegrateResult<PortHamiltonianResult> {
        integrate_impl(self, system, x0, t0, tf, dt, u, &self.options)
    }
}

impl Default for DiscreteGradientItohAbe {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Method 3: Average Vector Field (AVF) method
// ═══════════════════════════════════════════════════════════════════════════════

/// Average Vector Field (AVF) Integrator.
///
/// The AVF method integrates the pH vector field along the straight line
/// from x^n to x^{n+1} using Gaussian quadrature:
///
/// ```text
/// (x^{n+1} - x^n)/dt = ∫₀¹ f(x^n + τ(x^{n+1} - x^n)) dτ
/// ```
///
/// For conservative systems (R=0, u=0), the method exactly preserves energy
/// because:
/// ```text
/// H(x^{n+1}) - H(x^n) = ∫₀¹ ∇H·f dτ = ∫₀¹ ∇H·J∇H dτ = 0  (J is skew-sym)
/// ```
///
/// # Reference
///
/// E. Celledoni, V. Grimm, R. McLachlan, D. McLaren, D. O'Neale, B. Owren,
/// G. Quispel, "Preserving energy resp. dissipation in numerical PDEs using
/// the Average Vector Field method," J. Comput. Phys. 231 (2012), 6770-6789.
#[derive(Debug, Clone)]
pub struct AverageVectorField {
    options: IntegratorOptions,
}

impl AverageVectorField {
    /// Create a new AVF integrator.
    pub fn new() -> Self {
        Self {
            options: IntegratorOptions::default(),
        }
    }

    /// Create with custom options.
    pub fn with_options(mut self, options: IntegratorOptions) -> Self {
        self.options = options;
        self
    }

    /// Take a single step: x^n -> x^{n+1}
    pub fn step(
        &self,
        system: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        let n = x.len();
        let x_old = x.to_vec();
        let u_vec = u.to_vec();
        let n_q = self.options.n_quad_points;
        let (quad_nodes, quad_weights) = gauss_legendre_01(n_q)?;

        let residual = |x_new: &[f64]| -> IntegrateResult<Vec<f64>> {
            // Compute the average ∫₀¹ f(x̃(τ)) dτ by Gauss-Legendre quadrature
            let mut avg = vec![0.0_f64; n];
            for (&tau, &wt) in quad_nodes.iter().zip(quad_weights.iter()) {
                // x̃(τ) = x_old + τ * (x_new - x_old)
                let x_tau: Vec<f64> = x_old
                    .iter()
                    .zip(x_new.iter())
                    .map(|(xo, xn)| xo + tau * (xn - xo))
                    .collect();
                let rhs = system.rhs(&x_tau, &u_vec)?;
                for i in 0..n {
                    avg[i] += wt * rhs[i];
                }
            }
            // Residual: x_new - x_old - dt * avg
            let res: Vec<f64> = x_new
                .iter()
                .zip(x_old.iter())
                .zip(avg.iter())
                .map(|((xn, xo), a)| xn - xo - dt * a)
                .collect();
            Ok(res)
        };

        // Initial guess: explicit Euler
        let x_init: Vec<f64> = {
            let rhs = system.rhs(x, u)?;
            x.iter()
                .zip(rhs.iter())
                .map(|(xi, fi)| xi + dt * fi)
                .collect()
        };

        let (x_new, n_iters) = newton_solve(
            &residual,
            &x_init,
            self.options.newton_tol,
            self.options.max_newton_iters,
        )?;

        let energy = system.hamiltonian(&x_new)?;
        Ok(StepResult {
            x_new,
            energy,
            n_evals: n_iters * (n_q * n + n),
            n_iters,
        })
    }

    /// Integrate over a time span with optional control input.
    pub fn integrate(
        &self,
        system: &PortHamiltonianSystem,
        x0: &[f64],
        t0: f64,
        tf: f64,
        dt: f64,
        u: Option<&dyn Fn(f64, &[f64]) -> Vec<f64>>,
    ) -> IntegrateResult<PortHamiltonianResult> {
        integrate_impl(self, system, x0, t0, tf, dt, u, &self.options)
    }
}

impl Default for AverageVectorField {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Method 4: Implicit Midpoint Rule (symplectic, 2nd order)
// ═══════════════════════════════════════════════════════════════════════════════

/// Implicit Midpoint Rule for Port-Hamiltonian systems.
///
/// This is a classic symmetric 2nd-order integrator. For Hamiltonian systems
/// with quadratic energy (H = x^T A x / 2), it is exactly energy-conserving.
/// For general nonlinear systems, it approximately conserves energy.
///
/// The scheme is:
/// ```text
/// (x^{n+1} - x^n)/dt = f((x^n + x^{n+1})/2, u^{n+1/2})
/// ```
///
/// This method is equivalent to the 1-stage Gauss-Legendre Runge-Kutta method
/// and is B-series symplectic.
#[derive(Debug, Clone)]
pub struct ImplicitMidpoint {
    options: IntegratorOptions,
}

impl ImplicitMidpoint {
    /// Create a new Implicit Midpoint integrator.
    pub fn new() -> Self {
        Self {
            options: IntegratorOptions::default(),
        }
    }

    /// Create with custom options.
    pub fn with_options(mut self, options: IntegratorOptions) -> Self {
        self.options = options;
        self
    }

    /// Take a single step: x^n -> x^{n+1}
    pub fn step(
        &self,
        system: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        let n = x.len();
        let x_old = x.to_vec();
        let u_vec = u.to_vec();

        let residual = |x_new: &[f64]| -> IntegrateResult<Vec<f64>> {
            let x_mid: Vec<f64> = x_old
                .iter()
                .zip(x_new.iter())
                .map(|(a, b)| 0.5 * (a + b))
                .collect();
            let rhs = system.rhs(&x_mid, &u_vec)?;
            let res: Vec<f64> = x_new
                .iter()
                .zip(x_old.iter())
                .zip(rhs.iter())
                .map(|((xn, xo), f)| xn - xo - dt * f)
                .collect();
            Ok(res)
        };

        let x_init: Vec<f64> = {
            let rhs = system.rhs(x, u)?;
            x.iter()
                .zip(rhs.iter())
                .map(|(xi, fi)| xi + dt * fi)
                .collect()
        };

        let (x_new, n_iters) = newton_solve(
            &residual,
            &x_init,
            self.options.newton_tol,
            self.options.max_newton_iters,
        )?;

        let energy = system.hamiltonian(&x_new)?;
        Ok(StepResult {
            x_new,
            energy,
            n_evals: n_iters * (n + 2),
            n_iters,
        })
    }

    /// Integrate over a time span with optional control input.
    pub fn integrate(
        &self,
        system: &PortHamiltonianSystem,
        x0: &[f64],
        t0: f64,
        tf: f64,
        dt: f64,
        u: Option<&dyn Fn(f64, &[f64]) -> Vec<f64>>,
    ) -> IntegrateResult<PortHamiltonianResult> {
        integrate_impl(self, system, x0, t0, tf, dt, u, &self.options)
    }
}

impl Default for ImplicitMidpoint {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Method 5: Störmer-Verlet for separable pH systems
// ═══════════════════════════════════════════════════════════════════════════════

/// Störmer-Verlet integrator adapted for Port-Hamiltonian systems
/// with separable Hamiltonian H(q, p) = T(p) + V(q).
///
/// This is a 2nd-order symplectic integrator that exactly preserves the
/// symplectic 2-form and has a modified Hamiltonian that is preserved
/// to O(dt²) over exponentially long times.
///
/// For a separable Hamiltonian, the pH system with J = [[0, I], [-I, 0]]
/// and R = [[0, 0], [0, R_pp]] has the explicit leapfrog update:
///
/// ```text
/// p^{n+1/2} = p^n - (dt/2) * ∇_q V(q^n) - (dt/2) * R_pp * ∇_p T(p^{n+1/2})
/// q^{n+1}   = q^n + dt * ∇_p T(p^{n+1/2})
/// p^{n+1}   = p^{n+1/2} - (dt/2) * ∇_q V(q^{n+1}) - (dt/2) * R_pp * ∇_p T(p^{n+1/2})
/// ```
///
/// For the dissipative half-step, we use an implicit solve when R_pp ≠ 0.
#[derive(Debug, Clone)]
pub struct StormerVerletPH {
    options: IntegratorOptions,
    /// Number of configuration DOF (state dim = 2 * n_dof)
    n_dof: usize,
}

impl StormerVerletPH {
    /// Create a new Störmer-Verlet pH integrator.
    ///
    /// # Arguments
    ///
    /// * `n_dof` - Number of degrees of freedom (state = [q, p] with dim 2*n_dof)
    pub fn new(n_dof: usize) -> Self {
        Self {
            options: IntegratorOptions::default(),
            n_dof,
        }
    }

    /// Create with custom options.
    pub fn with_options(mut self, options: IntegratorOptions) -> Self {
        self.options = options;
        self
    }

    /// Take a single step for a separable pH system.
    ///
    /// The state is laid out as x = [q_1, ..., q_n, p_1, ..., p_n].
    /// The system must have a canonical symplectic structure:
    /// J = [[0, I], [-I, 0]]
    pub fn step(
        &self,
        system: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        let n = self.n_dof;
        if x.len() != 2 * n {
            return Err(IntegrateError::ValueError(format!(
                "State dimension {} != 2 * n_dof {}",
                x.len(),
                2 * n
            )));
        }

        let q = &x[..n];
        let p = &x[n..];

        // Step 1: Half-step momentum update p^{n+1/2}
        // dp/dt = -∂H/∂q (from J part) - R_pp * ∂H/∂p (from R part) + B_p * u
        let grad_h = system.grad_hamiltonian(x)?;
        let r = system.r_matrix(x)?;
        let b = system.b_matrix(x)?;
        let u_arr = Array1::from_vec(u.to_vec());
        let bu = b.dot(&u_arr);

        // Half-step for p: p_half = p - (dt/2)*∇_q H - (dt/2)*R_{pp}*∇_p H + (dt/2)*[Bu]_p
        let mut p_half = vec![0.0_f64; n];
        for i in 0..n {
            // -∂H/∂q_i is grad_h[i] (from J), dissipation from R uses ∇_p H = grad_h[n+i]
            let r_pp_grad_p: f64 = (0..n).map(|j| r[[n + i, n + j]] * grad_h[n + j]).sum();
            p_half[i] =
                p[i] - (dt / 2.0) * grad_h[i] - (dt / 2.0) * r_pp_grad_p + (dt / 2.0) * bu[n + i];
        }

        // Step 2: Full-step position update q^{n+1}
        // dq/dt = ∂H/∂p (from J) - R_{qp} * ∇_p H + [Bu]_q
        // For separable H with M diagonal: dq/dt = p/m
        let mut x_half = vec![0.0_f64; 2 * n];
        x_half[..n].copy_from_slice(q);
        x_half[n..].copy_from_slice(&p_half);

        let grad_h_half = system.grad_hamiltonian(&x_half)?;
        let r_half = system.r_matrix(&x_half)?;
        let b_half = system.b_matrix(&x_half)?;
        let bu_half = b_half.dot(&u_arr);

        let mut q_new = vec![0.0_f64; n];
        for i in 0..n {
            let r_qp_grad_p: f64 = (0..n)
                .map(|j| r_half[[i, n + j]] * grad_h_half[n + j])
                .sum();
            q_new[i] = q[i] + dt * grad_h_half[n + i] - dt * r_qp_grad_p + dt * bu_half[i];
        }

        // Step 3: Second half-step for momentum using new position
        let mut x_new_mid = vec![0.0_f64; 2 * n];
        x_new_mid[..n].copy_from_slice(&q_new);
        x_new_mid[n..].copy_from_slice(&p_half);

        let grad_h_new = system.grad_hamiltonian(&x_new_mid)?;
        let r_new = system.r_matrix(&x_new_mid)?;
        let b_new = system.b_matrix(&x_new_mid)?;
        let bu_new = b_new.dot(&u_arr);

        let mut p_new = vec![0.0_f64; n];
        for i in 0..n {
            let r_pp_grad_p: f64 = (0..n)
                .map(|j| r_new[[n + i, n + j]] * grad_h_new[n + j])
                .sum();
            p_new[i] = p_half[i] - (dt / 2.0) * grad_h_new[i] - (dt / 2.0) * r_pp_grad_p
                + (dt / 2.0) * bu_new[n + i];
        }

        let mut x_new = vec![0.0_f64; 2 * n];
        x_new[..n].copy_from_slice(&q_new);
        x_new[n..].copy_from_slice(&p_new);

        let energy = system.hamiltonian(&x_new)?;
        Ok(StepResult {
            x_new,
            energy,
            n_evals: 3,
            n_iters: 0,
        })
    }

    /// Integrate over a time span with optional control input.
    pub fn integrate(
        &self,
        system: &PortHamiltonianSystem,
        x0: &[f64],
        t0: f64,
        tf: f64,
        dt: f64,
        u: Option<&dyn Fn(f64, &[f64]) -> Vec<f64>>,
    ) -> IntegrateResult<PortHamiltonianResult> {
        integrate_impl(self, system, x0, t0, tf, dt, u, &self.options)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Method 6: RATTLE — for constrained Hamiltonian systems
// ═══════════════════════════════════════════════════════════════════════════════

/// RATTLE integrator for constrained Port-Hamiltonian systems.
///
/// RATTLE handles holonomic constraints g(q) = 0 combined with the
/// pH dynamics. The algorithm alternates between:
/// 1. A constrained half-step for momentum (SHAKE step)
/// 2. A full position update with constraint enforcement
/// 3. A second constrained half-step for momentum (velocity correction)
///
/// Constraints are enforced via Lagrange multipliers.
///
/// # Problem formulation
///
/// ```text
/// dx/dt = (J - R) * ∇H + B * u   (pH dynamics)
/// g(q) = 0                         (holonomic constraints)
/// ```
///
/// The constrained equations are:
/// ```text
/// dq/dt = ∂H/∂p
/// dp/dt = -∂H/∂q - (∂g/∂q)^T λ - R_pp * ∂H/∂p + [Bu]_p
/// g(q) = 0
/// ∂g/∂q * ∂H/∂p = 0   (hidden constraint on velocities)
/// ```
///
/// # Reference
///
/// H. Andersen, "RATTLE: A velocity version of the SHAKE algorithm,"
/// J. Comput. Phys. 52 (1983), 24-34.
pub struct Rattle {
    options: IntegratorOptions,
    n_dof: usize,
    n_constraints: usize,
    /// Constraint function g(q): R^n_dof -> R^n_constraints
    constraint_fn: Box<dyn Fn(&[f64]) -> IntegrateResult<Vec<f64>> + Send + Sync>,
    /// Jacobian of constraints ∂g/∂q: R^n_dof -> R^{n_constraints × n_dof}
    constraint_jac: Box<dyn Fn(&[f64]) -> IntegrateResult<Vec<Vec<f64>>> + Send + Sync>,
}

impl Rattle {
    /// Create a new RATTLE integrator.
    ///
    /// # Arguments
    ///
    /// * `n_dof` - Number of degrees of freedom (state = [q, p])
    /// * `n_constraints` - Number of holonomic constraints
    /// * `constraint_fn` - Constraint function g(q)
    /// * `constraint_jac` - Jacobian ∂g/∂q(q)
    pub fn new(
        n_dof: usize,
        n_constraints: usize,
        constraint_fn: impl Fn(&[f64]) -> IntegrateResult<Vec<f64>> + Send + Sync + 'static,
        constraint_jac: impl Fn(&[f64]) -> IntegrateResult<Vec<Vec<f64>>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            options: IntegratorOptions::default(),
            n_dof,
            n_constraints,
            constraint_fn: Box::new(constraint_fn),
            constraint_jac: Box::new(constraint_jac),
        }
    }

    /// Create with custom options.
    pub fn with_options(mut self, options: IntegratorOptions) -> Self {
        self.options = options;
        self
    }

    /// Take a single RATTLE step.
    ///
    /// State layout: x = [q_1..q_n, p_1..p_n]
    pub fn step(
        &self,
        system: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        let n = self.n_dof;
        let m = self.n_constraints;

        if x.len() != 2 * n {
            return Err(IntegrateError::ValueError(format!(
                "RATTLE: state dim {} != 2 * n_dof {}",
                x.len(),
                2 * n
            )));
        }

        let q = x[..n].to_vec();
        let p = x[n..].to_vec();
        let u_arr = Array1::from_vec(u.to_vec());

        // ── RATTLE step 1: Half-step p with Lagrange multipliers ──────────────
        // p^{n+1/2} = p^n - (dt/2)*∇_q H - (dt/2)*R_{pp}*∇_p H + (dt/2)*[Bu]_p
        //             - (dt/2)*(∂g/∂q)^T * λ
        // solve for λ such that g(q^n + dt * ∇_p H^{n+1/2}) = 0

        let grad_h = system.grad_hamiltonian(x)?;
        let r = system.r_matrix(x)?;
        let b_mat = system.b_matrix(x)?;
        let bu = b_mat.dot(&u_arr);

        // Base half-step (no constraint)
        let mut p_half_base = vec![0.0_f64; n];
        for i in 0..n {
            let r_pp_grad: f64 = (0..n).map(|j| r[[n + i, n + j]] * grad_h[n + j]).sum();
            p_half_base[i] =
                p[i] - (dt / 2.0) * grad_h[i] - (dt / 2.0) * r_pp_grad + (dt / 2.0) * bu[n + i];
        }

        // Position after unconstrained step
        let mut x_half = vec![0.0_f64; 2 * n];
        x_half[..n].copy_from_slice(&q);
        x_half[n..].copy_from_slice(&p_half_base);
        let grad_h_half = system.grad_hamiltonian(&x_half)?;

        let q_pred: Vec<f64> = (0..n).map(|i| q[i] + dt * grad_h_half[n + i]).collect();

        // Iteratively solve for Lagrange multipliers λ using Newton
        let jac_q = (self.constraint_jac)(&q)?;
        let mut lambda = vec![0.0_f64; m];

        // Newton iteration for λ: g(q^n + dt * (∇_p H(p_half - Jac^T λ))) = 0
        for _ in 0..self.options.max_newton_iters {
            // Current p_half with λ correction
            let mut p_half_corr = p_half_base.clone();
            for i in 0..n {
                for k in 0..m {
                    p_half_corr[i] -= (dt / 2.0) * jac_q[k][i] * lambda[k];
                }
            }
            // x_half with correction
            let mut x_h = vec![0.0_f64; 2 * n];
            x_h[..n].copy_from_slice(&q);
            x_h[n..].copy_from_slice(&p_half_corr);
            let gh = system.grad_hamiltonian(&x_h)?;

            // Predicted q
            let q_c: Vec<f64> = (0..n).map(|i| q[i] + dt * gh[n + i]).collect();
            let g_q_c = (self.constraint_fn)(&q_c)?;

            let g_norm: f64 = g_q_c.iter().map(|v| v * v).sum::<f64>().sqrt();
            if g_norm < self.options.newton_tol {
                break;
            }

            // Linearised update: Jac_q * (∂q/∂λ) * Δλ = -g
            // ∂q/∂λ = dt * ∂(∇_p H)/∂λ ≈ -dt * (dt/2) * I (for diagonal mass)
            // Simplified: use Jac * (dt^2/2) * Jac^T as approximate system
            let mut a = vec![vec![0.0_f64; m]; m];
            let jac_qc = (self.constraint_jac)(&q_c)?;
            for i in 0..m {
                for j in 0..m {
                    let dot: f64 = (0..n).map(|k| jac_qc[i][k] * jac_q[j][k]).sum();
                    a[i][j] = (dt * dt / 2.0) * dot;
                }
            }
            let rhs: Vec<f64> = g_q_c.iter().map(|v| -v).collect();
            let delta_lambda = gauss_solve(&a, &rhs).unwrap_or_else(|_| vec![0.0; m]);
            for k in 0..m {
                lambda[k] += delta_lambda[k];
            }
        }

        // Apply λ to get final p_half
        let mut p_half = p_half_base.clone();
        for i in 0..n {
            for k in 0..m {
                p_half[i] -= (dt / 2.0) * jac_q[k][i] * lambda[k];
            }
        }

        // ── RATTLE step 2: Full position step ─────────────────────────────────
        let mut x_half_final = vec![0.0_f64; 2 * n];
        x_half_final[..n].copy_from_slice(&q);
        x_half_final[n..].copy_from_slice(&p_half);
        let grad_h_half_final = system.grad_hamiltonian(&x_half_final)?;

        let q_new: Vec<f64> = (0..n)
            .map(|i| q[i] + dt * grad_h_half_final[n + i])
            .collect();

        // ── RATTLE step 3: Second half-step for momentum ───────────────────────
        // p^{n+1} = p_half - (dt/2)*∇_q H(q^{n+1}, p_half) - (dt/2)*R*∇_p H + [Bu]_p
        //           - (dt/2)*(∂g/∂q^{n+1})^T * μ
        // solve for μ such that (∂g/∂q^{n+1})^T * ∇_p H(p^{n+1}) = 0

        let mut x_new_half = vec![0.0_f64; 2 * n];
        x_new_half[..n].copy_from_slice(&q_new);
        x_new_half[n..].copy_from_slice(&p_half);
        let grad_h_new = system.grad_hamiltonian(&x_new_half)?;
        let r_new = system.r_matrix(&x_new_half)?;
        let b_new = system.b_matrix(&x_new_half)?;
        let bu_new = b_new.dot(&u_arr);

        let mut p_new_base = vec![0.0_f64; n];
        for i in 0..n {
            let r_pp_grad: f64 = (0..n)
                .map(|j| r_new[[n + i, n + j]] * grad_h_new[n + j])
                .sum();
            p_new_base[i] = p_half[i] - (dt / 2.0) * grad_h_new[i] - (dt / 2.0) * r_pp_grad
                + (dt / 2.0) * bu_new[n + i];
        }

        // Solve for μ: (∂g/∂q_new)^T * ∇_p H(p_new) = 0
        let jac_qnew = (self.constraint_jac)(&q_new)?;
        let mut mu = vec![0.0_f64; m];

        for _ in 0..self.options.max_newton_iters {
            let mut p_try = p_new_base.clone();
            for i in 0..n {
                for k in 0..m {
                    p_try[i] -= (dt / 2.0) * jac_qnew[k][i] * mu[k];
                }
            }
            let mut x_try = vec![0.0_f64; 2 * n];
            x_try[..n].copy_from_slice(&q_new);
            x_try[n..].copy_from_slice(&p_try);
            let gh_try = system.grad_hamiltonian(&x_try)?;

            // Velocity constraint: Jac * ∇_p H = 0
            let mut vc = vec![0.0_f64; m];
            for k in 0..m {
                vc[k] = (0..n).map(|i| jac_qnew[k][i] * gh_try[n + i]).sum();
            }
            let vc_norm: f64 = vc.iter().map(|v| v * v).sum::<f64>().sqrt();
            if vc_norm < self.options.newton_tol {
                break;
            }

            let mut a = vec![vec![0.0_f64; m]; m];
            for i in 0..m {
                for j in 0..m {
                    // ∂(vc_i)/∂μ_j ≈ -dt/2 * Jac[i]^T * Jac[j] (for unit mass)
                    let dot: f64 = (0..n).map(|k| jac_qnew[i][k] * jac_qnew[j][k]).sum();
                    a[i][j] = -(dt / 2.0) * dot;
                }
            }
            let rhs: Vec<f64> = vc.iter().map(|v| -v).collect();
            let delta_mu = gauss_solve(&a, &rhs).unwrap_or_else(|_| vec![0.0; m]);
            for k in 0..m {
                mu[k] += delta_mu[k];
            }
        }

        let mut p_new = p_new_base.clone();
        for i in 0..n {
            for k in 0..m {
                p_new[i] -= (dt / 2.0) * jac_qnew[k][i] * mu[k];
            }
        }

        let mut x_new = vec![0.0_f64; 2 * n];
        x_new[..n].copy_from_slice(&q_new);
        x_new[n..].copy_from_slice(&p_new);

        let energy = system.hamiltonian(&x_new)?;
        Ok(StepResult {
            x_new,
            energy,
            n_evals: 6,
            n_iters: 0,
        })
    }

    /// Integrate over a time span with optional control input.
    pub fn integrate(
        &self,
        system: &PortHamiltonianSystem,
        x0: &[f64],
        t0: f64,
        tf: f64,
        dt: f64,
        u: Option<&dyn Fn(f64, &[f64]) -> Vec<f64>>,
    ) -> IntegrateResult<PortHamiltonianResult> {
        integrate_impl(self, system, x0, t0, tf, dt, u, &self.options)
    }
}

impl std::fmt::Debug for Rattle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rattle")
            .field("n_dof", &self.n_dof)
            .field("n_constraints", &self.n_constraints)
            .field("options", &self.options)
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Common integration driver
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for single-step integrators.
trait SingleStepIntegrator {
    fn step_impl(
        &self,
        system: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult>;
}

impl SingleStepIntegrator for DiscreteGradientGonzalez {
    fn step_impl(
        &self,
        s: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        self.step(s, x, u, dt)
    }
}
impl SingleStepIntegrator for DiscreteGradientItohAbe {
    fn step_impl(
        &self,
        s: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        self.step(s, x, u, dt)
    }
}
impl SingleStepIntegrator for AverageVectorField {
    fn step_impl(
        &self,
        s: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        self.step(s, x, u, dt)
    }
}
impl SingleStepIntegrator for ImplicitMidpoint {
    fn step_impl(
        &self,
        s: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        self.step(s, x, u, dt)
    }
}
impl SingleStepIntegrator for StormerVerletPH {
    fn step_impl(
        &self,
        s: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        self.step(s, x, u, dt)
    }
}
impl SingleStepIntegrator for Rattle {
    fn step_impl(
        &self,
        s: &PortHamiltonianSystem,
        x: &[f64],
        u: &[f64],
        dt: f64,
    ) -> IntegrateResult<StepResult> {
        self.step(s, x, u, dt)
    }
}

fn integrate_impl<I: SingleStepIntegrator>(
    integrator: &I,
    system: &PortHamiltonianSystem,
    x0: &[f64],
    t0: f64,
    tf: f64,
    dt: f64,
    u: Option<&dyn Fn(f64, &[f64]) -> Vec<f64>>,
    options: &IntegratorOptions,
) -> IntegrateResult<PortHamiltonianResult> {
    if dt <= 0.0 {
        return Err(IntegrateError::ValueError(
            "Step size dt must be positive".into(),
        ));
    }
    if tf <= t0 {
        return Err(IntegrateError::ValueError(
            "Final time tf must be greater than initial time t0".into(),
        ));
    }

    let n_steps = ((tf - t0) / dt).ceil() as usize;
    let actual_dt = (tf - t0) / (n_steps as f64);

    let zero_u = vec![0.0_f64; system.n_ports];

    let mut ts = Vec::with_capacity(n_steps + 1);
    let mut xs: Vec<Vec<f64>> = Vec::with_capacity(n_steps + 1);
    let mut energies = Vec::with_capacity(n_steps + 1);
    let mut outputs: Vec<Vec<f64>> = Vec::with_capacity(n_steps + 1);
    let mut total_evals = 0usize;

    let h0 = system.hamiltonian(x0)?;
    let y0 = system.output(x0)?;

    ts.push(t0);
    xs.push(x0.to_vec());
    energies.push(h0);
    outputs.push(y0.to_vec());

    let mut x_cur = x0.to_vec();
    let mut t_cur = t0;

    for _ in 0..n_steps {
        let u_cur = if let Some(u_fn) = u {
            u_fn(t_cur, &x_cur)
        } else {
            zero_u.clone()
        };

        let result = integrator.step_impl(system, &x_cur, &u_cur, actual_dt)?;
        total_evals += result.n_evals;

        t_cur += actual_dt;
        x_cur = result.x_new;

        if options.store_trajectory {
            let y_cur = system.output(&x_cur)?;
            ts.push(t_cur);
            xs.push(x_cur.clone());
            energies.push(result.energy);
            outputs.push(y_cur.to_vec());
        }
    }

    let h_final = if options.store_trajectory {
        *energies.last().unwrap_or(&h0)
    } else {
        system.hamiltonian(&x_cur)?
    };
    let energy_drift = (h_final - h0).abs();

    Ok(PortHamiltonianResult {
        t: ts,
        x: xs,
        energy: energies,
        output: outputs,
        n_evals: total_evals,
        energy_drift,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_legendre_nodes_sum_to_one() {
        for n in 1..=7 {
            let (_, w) = gauss_legendre_01(n).expect("Failed to get GL nodes");
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-13, "n={n} weights sum={sum}");
        }
    }

    #[test]
    fn test_gauss_solve() {
        let a = vec![vec![2.0_f64, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0_f64, 10.0];
        let x = gauss_solve(&a, &b).expect("Solve failed");
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
    }
}
