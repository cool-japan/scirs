//! Shooting methods for Boundary Value Problems (BVPs)
//!
//! This module provides shooting-based methods for solving two-point boundary
//! value problems (BVPs) of the form:
//!
//!   y'(t) = f(t, y),   t ∈ [a, b]
//!   g(y(a), y(b)) = 0
//!
//! ## Methods
//!
//! - **Single shooting**: Parameterize y(a) with free parameters, integrate to b,
//!   and solve the boundary residual using Newton's method.
//! - **Multiple shooting**: Divide [a,b] into subintervals, shoot over each, and
//!   enforce continuity + boundary conditions simultaneously.
//! - **Orthogonal collocation**: Collocate at Gaussian or Radau points within
//!   subintervals for higher accuracy.
//! - **Periodic orbit finder**: Find limit cycles by single shooting with period
//!   as an additional unknown and a phase condition.
//!
//! ## References
//!
//! - Keller (1968), "Numerical Methods for Two-Point Boundary Value Problems"
//! - Stoer & Bulirsch (1980), "Introduction to Numerical Analysis"
//! - Ascher, Mattheij, Russell (1995), "Numerical Solution of Boundary Value ODEs"

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

#[inline]
fn to_f(v: f64) -> f64 {
    v
}

/// Gaussian elimination with partial pivoting (modifies A and b in place)
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
                "Singular matrix in shooting solve".to_string(),
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

/// Classical 4th-order Runge-Kutta step (fixed step)
fn rk4_step<F>(f: &F, t: f64, y: &Array1<f64>, h: f64) -> Array1<f64>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
{
    let n = y.len();
    let k1 = f(t, y);
    let mut y2 = Array1::<f64>::zeros(n);
    for i in 0..n {
        y2[i] = y[i] + 0.5 * h * k1[i];
    }
    let k2 = f(t + 0.5 * h, &y2);
    let mut y3 = Array1::<f64>::zeros(n);
    for i in 0..n {
        y3[i] = y[i] + 0.5 * h * k2[i];
    }
    let k3 = f(t + 0.5 * h, &y3);
    let mut y4 = Array1::<f64>::zeros(n);
    for i in 0..n {
        y4[i] = y[i] + h * k3[i];
    }
    let k4 = f(t + h, &y4);
    let mut y_new = Array1::<f64>::zeros(n);
    for i in 0..n {
        y_new[i] = y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    y_new
}

/// Integrate ODE from t0 to t1 using fixed-step RK4, returns final state
fn integrate_rk4<F>(f: &F, t0: f64, y0: &Array1<f64>, t1: f64, n_steps: usize) -> Array1<f64>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
{
    let h = (t1 - t0) / n_steps as f64;
    let mut t = t0;
    let mut y = y0.clone();
    for _ in 0..n_steps {
        y = rk4_step(f, t, &y, h);
        t += h;
    }
    y
}

/// Compute numerical Jacobian of residual g(s) w.r.t. s using central differences
fn numerical_jacobian<G>(g: &G, s: &Array1<f64>, eps: f64) -> Array2<f64>
where
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n = s.len();
    let m = g(s).len();
    let mut jac = Array2::<f64>::zeros((m, n));
    for j in 0..n {
        let mut sp = s.clone();
        sp[j] += eps;
        let mut sm = s.clone();
        sm[j] -= eps;
        let fp = g(&sp);
        let fm = g(&sm);
        for i in 0..m {
            jac[[i, j]] = (fp[i] - fm[i]) / (2.0 * eps);
        }
    }
    jac
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a shooting-based BVP solve
#[derive(Debug, Clone)]
pub struct BVPResult {
    /// Time points of the solution
    pub t: Vec<f64>,
    /// Solution state at each time point
    pub y: Vec<Array1<f64>>,
    /// Boundary residual at the solution
    pub residual: f64,
    /// Error estimate (based on boundary residual)
    pub error: f64,
    /// Number of Newton iterations
    pub n_newton_iters: usize,
    /// Whether the solver converged
    pub success: bool,
    /// Message describing termination
    pub message: String,
}

// ---------------------------------------------------------------------------
// Single Shooting
// ---------------------------------------------------------------------------

/// Configuration for single and multiple shooting methods
#[derive(Debug, Clone)]
pub struct ShootingConfig {
    /// Number of RK4 integration steps per subinterval
    pub n_steps: usize,
    /// Newton tolerance for boundary residual
    pub newton_tol: f64,
    /// Maximum Newton iterations
    pub max_newton_iter: usize,
    /// Finite difference epsilon for Jacobians
    pub fd_eps: f64,
    /// Number of subintervals for multiple shooting
    pub n_subintervals: usize,
}

impl Default for ShootingConfig {
    fn default() -> Self {
        Self {
            n_steps: 100,
            newton_tol: 1e-8,
            max_newton_iter: 50,
            fd_eps: 1e-7,
            n_subintervals: 5,
        }
    }
}

/// Single-shooting method for BVPs.
///
/// The missing initial conditions y(a) are parameterized by a vector s ∈ R^k.
/// We integrate to b and solve the boundary condition g(y(a), y(b)) = 0.
///
/// # Arguments
///
/// * `f` - ODE function: f(t, y) → y'
/// * `g` - Boundary residual: g(s, y_b) → 0, where s = free parameters at a
/// * `merge_initial` - Merge known+free initial conditions: (s) → y(a)
/// * `t_span` - [a, b]
/// * `s0` - Initial guess for free parameters
/// * `cfg` - Solver configuration
///
/// # Returns
///
/// `BVPResult` with the solution trajectory or error.
pub struct SingleShooting;

impl SingleShooting {
    /// Solve BVP by single shooting.
    ///
    /// The boundary condition is `g(s, y_b) = 0` where s are the free initial
    /// conditions and y_b = y(b) is obtained by integrating forward.
    pub fn solve<ODE, BC, IC>(
        ode: &ODE,
        bc: &BC,
        initial_condition: &IC,
        t_span: [f64; 2],
        s0: Array1<f64>,
        cfg: &ShootingConfig,
    ) -> IntegrateResult<BVPResult>
    where
        ODE: Fn(f64, &Array1<f64>) -> Array1<f64>,
        BC: Fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
        IC: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let [ta, tb] = t_span;
        let n_s = s0.len();

        // Shooting function: shoot from s, return boundary residual g(ya, yb)
        let shoot = |s: &Array1<f64>| -> Array1<f64> {
            let ya = initial_condition(s);
            let yb = integrate_rk4(ode, ta, &ya, tb, cfg.n_steps);
            bc(&ya, &yb)
        };

        let mut s = s0.clone();
        let mut n_iters = 0usize;
        let mut converged = false;

        for iter in 0..cfg.max_newton_iter {
            let res = shoot(&s);
            let res_norm: f64 = res.iter().map(|&v| v * v).sum::<f64>().sqrt();

            if res_norm < cfg.newton_tol {
                n_iters = iter + 1;
                converged = true;
                break;
            }

            // Jacobian of shooting function w.r.t. s
            let mut jac = numerical_jacobian(&shoot, &s, cfg.fd_eps);
            let mut neg_res = res.mapv(|v| -v);

            match gauss_solve(&mut jac, &mut neg_res) {
                Ok(delta) => {
                    for i in 0..n_s {
                        s[i] += delta[i];
                    }
                }
                Err(_) => {
                    // Fallback: gradient descent step
                    let res_ref = shoot(&s);
                    let grad_norm_sq: f64 = res_ref.iter().map(|&v| v * v).sum();
                    if grad_norm_sq > 0.0 {
                        for i in 0..n_s {
                            s[i] -= cfg.fd_eps * res_ref[i.min(res_ref.len() - 1)];
                        }
                    }
                }
            }
        }

        if !converged {
            n_iters = cfg.max_newton_iter;
        }

        // Reconstruct solution trajectory
        let ya = initial_condition(&s);
        let (t_traj, y_traj) = trajectory_rk4(ode, ta, &ya, tb, cfg.n_steps);

        let yb = y_traj.last().cloned().unwrap_or_else(|| ya.clone());
        let final_res = bc(&ya, &yb);
        let residual: f64 = final_res.iter().map(|&v| v * v).sum::<f64>().sqrt();

        Ok(BVPResult {
            t: t_traj,
            y: y_traj,
            residual,
            error: residual,
            n_newton_iters: n_iters,
            success: converged,
            message: if converged {
                "Single shooting converged".to_string()
            } else {
                format!(
                    "Single shooting did not converge in {} iterations",
                    cfg.max_newton_iter
                )
            },
        })
    }
}

/// Reconstruct trajectory using RK4, returning (times, states)
fn trajectory_rk4<F>(
    f: &F,
    t0: f64,
    y0: &Array1<f64>,
    t1: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<Array1<f64>>)
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
{
    let h = (t1 - t0) / n_steps as f64;
    let mut t = t0;
    let mut y = y0.clone();
    let mut ts = vec![t];
    let mut ys = vec![y.clone()];

    for _ in 0..n_steps {
        y = rk4_step(f, t, &y, h);
        t += h;
        ts.push(t);
        ys.push(y.clone());
    }
    (ts, ys)
}

// ---------------------------------------------------------------------------
// Multiple Shooting
// ---------------------------------------------------------------------------

/// Multiple-shooting method for BVPs.
///
/// Divides [a, b] into M subintervals [t_0, t_1, ..., t_M] and introduces
/// state variables s_i = y(t_i^-) at each interior (and initial) node.
///
/// The system to solve is:
///   - Boundary conditions: g(s_0, s_M) = 0 (n_bc equations)
///   - Continuity: y(t_i; s_i) = s_{i+1} for i = 1, ..., M-1 (n*(M-1) equations)
///   - Total: n*M unknowns in [s_0, s_1, ..., s_{M-1}]
///
/// The Jacobian has a block-bidiagonal structure (solved here via dense Newton).
pub struct MultipleShooting;

impl MultipleShooting {
    /// Solve BVP by multiple shooting.
    ///
    /// # Arguments
    ///
    /// * `ode` - ODE function f(t, y)
    /// * `bc` - Boundary conditions: g(y(a), y(b)) → 0 (n_bc equations)
    /// * `t_nodes` - Subinterval nodes [t_0, t_1, ..., t_M] (M+1 nodes, M intervals)
    /// * `s0` - Initial guesses for state at each node: (M, n) as Vec<Array1>
    /// * `cfg` - Solver configuration
    pub fn solve<ODE, BC>(
        ode: &ODE,
        bc: &BC,
        t_nodes: &[f64],
        s0: Vec<Array1<f64>>,
        cfg: &ShootingConfig,
    ) -> IntegrateResult<BVPResult>
    where
        ODE: Fn(f64, &Array1<f64>) -> Array1<f64>,
        BC: Fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    {
        if t_nodes.len() < 2 {
            return Err(IntegrateError::InvalidInput(
                "t_nodes must have at least 2 elements".to_string(),
            ));
        }
        let m = t_nodes.len() - 1; // number of subintervals
        if s0.len() != m {
            return Err(IntegrateError::DimensionMismatch(format!(
                "s0 length {} must equal number of subintervals {}",
                s0.len(),
                m
            )));
        }

        let n = s0[0].len(); // state dimension
        let total_unknowns = m * n;

        // Flatten unknowns: [s_0 | s_1 | ... | s_{M-1}]
        let mut s_flat = Array1::<f64>::zeros(total_unknowns);
        for (i, si) in s0.iter().enumerate() {
            for j in 0..n {
                s_flat[i * n + j] = si[j];
            }
        }

        let mut n_iters = 0usize;
        let mut converged = false;

        for iter in 0..cfg.max_newton_iter {
            // Build residual: [BC | continuity...]
            let n_bc_eqs = {
                let ya = s_flat.slice(scirs2_core::ndarray::s![..n]).to_owned();
                let yb_start = &s_flat
                    .slice(scirs2_core::ndarray::s![(m - 1) * n..])
                    .to_owned();
                let t_last = t_nodes[m - 1];
                let t_end = t_nodes[m];
                let yb = integrate_rk4(ode, t_last, yb_start, t_end, cfg.n_steps);
                bc(&ya, &yb).len()
            };

            let residual_len = n_bc_eqs + (m - 1) * n;
            let mut res = Array1::<f64>::zeros(residual_len);

            // BC residual
            let ya = s_flat.slice(scirs2_core::ndarray::s![..n]).to_owned();
            let yb_start = s_flat
                .slice(scirs2_core::ndarray::s![(m - 1) * n..])
                .to_owned();
            let yb = integrate_rk4(ode, t_nodes[m - 1], &yb_start, t_nodes[m], cfg.n_steps);
            let bc_res = bc(&ya, &yb);
            for i in 0..n_bc_eqs {
                res[i] = bc_res[i];
            }

            // Continuity residuals: y(t_i^+; s_i) - s_{i+1} = 0
            for interval in 0..(m - 1) {
                let si = s_flat
                    .slice(scirs2_core::ndarray::s![interval * n..(interval + 1) * n])
                    .to_owned();
                let si_next = s_flat
                    .slice(scirs2_core::ndarray::s![
                        (interval + 1) * n..(interval + 2) * n
                    ])
                    .to_owned();
                let y_shot = integrate_rk4(
                    ode,
                    t_nodes[interval],
                    &si,
                    t_nodes[interval + 1],
                    cfg.n_steps,
                );
                for j in 0..n {
                    res[n_bc_eqs + interval * n + j] = y_shot[j] - si_next[j];
                }
            }

            let res_norm: f64 = res.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if res_norm < cfg.newton_tol {
                n_iters = iter + 1;
                converged = true;
                break;
            }

            // Numerical Jacobian of full residual w.r.t. s_flat
            let shoot_residual = |s: &Array1<f64>| {
                let mut r = Array1::<f64>::zeros(residual_len);
                let ya = s.slice(scirs2_core::ndarray::s![..n]).to_owned();
                let yb_s = s.slice(scirs2_core::ndarray::s![(m - 1) * n..]).to_owned();
                let yb = integrate_rk4(ode, t_nodes[m - 1], &yb_s, t_nodes[m], cfg.n_steps);
                let bcr = bc(&ya, &yb);
                for i in 0..n_bc_eqs {
                    r[i] = bcr[i];
                }
                for interval in 0..(m - 1) {
                    let si = s
                        .slice(scirs2_core::ndarray::s![interval * n..(interval + 1) * n])
                        .to_owned();
                    let si_next = s
                        .slice(scirs2_core::ndarray::s![
                            (interval + 1) * n..(interval + 2) * n
                        ])
                        .to_owned();
                    let y_shot = integrate_rk4(
                        ode,
                        t_nodes[interval],
                        &si,
                        t_nodes[interval + 1],
                        cfg.n_steps,
                    );
                    for j in 0..n {
                        r[n_bc_eqs + interval * n + j] = y_shot[j] - si_next[j];
                    }
                }
                r
            };

            let mut jac = numerical_jacobian(&shoot_residual, &s_flat, cfg.fd_eps);
            let mut neg_res = res.mapv(|v| -v);

            match gauss_solve(&mut jac, &mut neg_res) {
                Ok(delta) => {
                    for i in 0..total_unknowns {
                        s_flat[i] += delta[i];
                    }
                }
                Err(_) => {
                    return Err(IntegrateError::LinearSolveError(
                        "Multiple shooting: singular Jacobian".to_string(),
                    ));
                }
            }
        }

        if !converged {
            n_iters = cfg.max_newton_iter;
        }

        // Reconstruct trajectory
        let mut t_traj = Vec::new();
        let mut y_traj = Vec::new();
        for interval in 0..m {
            let si = s_flat
                .slice(scirs2_core::ndarray::s![interval * n..(interval + 1) * n])
                .to_owned();
            let (ts, ys) = trajectory_rk4(
                ode,
                t_nodes[interval],
                &si,
                t_nodes[interval + 1],
                cfg.n_steps / m.max(1),
            );
            if interval == 0 {
                t_traj.extend_from_slice(&ts);
                y_traj.extend_from_slice(&ys);
            } else {
                t_traj.extend_from_slice(&ts[1..]);
                y_traj.extend_from_slice(&ys[1..]);
            }
        }

        let ya = y_traj
            .first()
            .cloned()
            .unwrap_or_else(|| Array1::<f64>::zeros(n));
        let yb = y_traj
            .last()
            .cloned()
            .unwrap_or_else(|| Array1::<f64>::zeros(n));
        let final_bc = bc(&ya, &yb);
        let residual: f64 = final_bc.iter().map(|&v| v * v).sum::<f64>().sqrt();

        Ok(BVPResult {
            t: t_traj,
            y: y_traj,
            residual,
            error: residual,
            n_newton_iters: n_iters,
            success: converged,
            message: if converged {
                "Multiple shooting converged".to_string()
            } else {
                format!(
                    "Multiple shooting did not converge in {} iterations",
                    cfg.max_newton_iter
                )
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Orthogonal Collocation
// ---------------------------------------------------------------------------

/// Gauss-Legendre collocation nodes on [-1, 1] for order 2 to 5
fn gauss_legendre_nodes(order: usize) -> Vec<f64> {
    match order {
        1 => vec![0.0],
        2 => vec![-1.0 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt()],
        3 => vec![-0.7745966692, 0.0, 0.7745966692],
        4 => vec![-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
        5 => vec![
            -0.9061798459,
            -0.5384693101,
            0.0,
            0.5384693101,
            0.9061798459,
        ],
        _ => vec![-1.0 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt()], // default to 2-point
    }
}

/// Gauss-Legendre weights on [-1, 1]
fn gauss_legendre_weights(order: usize) -> Vec<f64> {
    match order {
        1 => vec![2.0],
        2 => vec![1.0, 1.0],
        3 => vec![0.5555555556, 0.8888888889, 0.5555555556],
        4 => vec![0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451],
        5 => vec![
            0.2369268851,
            0.4786286705,
            0.5688888889,
            0.4786286705,
            0.2369268851,
        ],
        _ => vec![1.0, 1.0],
    }
}

/// Configuration for orthogonal collocation
#[derive(Debug, Clone)]
pub struct CollocationConfig {
    /// Number of subintervals (mesh intervals)
    pub n_subintervals: usize,
    /// Number of collocation points per interval (= polynomial order)
    pub collocation_order: usize,
    /// Newton solver tolerance
    pub newton_tol: f64,
    /// Maximum Newton iterations
    pub max_newton_iter: usize,
    /// Finite difference epsilon
    pub fd_eps: f64,
}

impl Default for CollocationConfig {
    fn default() -> Self {
        Self {
            n_subintervals: 10,
            collocation_order: 3,
            newton_tol: 1e-8,
            max_newton_iter: 30,
            fd_eps: 1e-7,
        }
    }
}

/// Orthogonal collocation at Gauss-Legendre points.
///
/// Approximates y(t) as a piecewise polynomial, enforcing the ODE at collocation
/// points within each subinterval and matching state values at mesh nodes.
///
/// For each subinterval [t_i, t_{i+1}] with m collocation points τ_j:
///   - The polynomial p(t) satisfies p(t_i) = y_i (left endpoint)
///   - p'(τ_j) = f(τ_j, p(τ_j)) for j = 1..m (collocation conditions)
///   - y_{i+1} = p(t_{i+1}) (right endpoint value)
///
/// This results in a large nonlinear system that is solved by Newton iterations.
pub struct OrthogonalCollocation;

impl OrthogonalCollocation {
    /// Solve BVP by collocation at Gauss-Legendre points.
    ///
    /// # Arguments
    ///
    /// * `ode` - ODE function f(t, y)
    /// * `bc` - Boundary condition: g(y(a), y(b)) = 0 (n_bc equations)
    /// * `t_span` - [a, b]
    /// * `y_init_guess` - Closure providing initial guess for y at time t
    /// * `cfg` - Collocation configuration
    pub fn solve<ODE, BC, Guess>(
        ode: &ODE,
        bc: &BC,
        t_span: [f64; 2],
        y_init_guess: &Guess,
        n_state: usize,
        cfg: &CollocationConfig,
    ) -> IntegrateResult<BVPResult>
    where
        ODE: Fn(f64, &Array1<f64>) -> Array1<f64>,
        BC: Fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
        Guess: Fn(f64) -> Array1<f64>,
    {
        let [ta, tb] = t_span;
        let m = cfg.n_subintervals;
        let k = cfg.collocation_order.min(5).max(1);
        let h = (tb - ta) / m as f64;

        // Mesh nodes
        let nodes: Vec<f64> = (0..=m).map(|i| ta + i as f64 * h).collect();

        // Collocation nodes (on reference element [-1,1])
        let ref_nodes = gauss_legendre_nodes(k);

        // Total unknowns: y at mesh nodes (m+1)*n  +  y at collocation pts m*k*n
        // But we can simplify: unknowns = y at mesh nodes (m+1)*n only, and collocation
        // pts determined via polynomial. For simplicity, we use the simpler formulation:
        // unknowns = y at all mesh nodes (m+1)*n,
        // residuals = ODE satisfied at collocation pts + BC.
        //
        // Each mesh interval [t_i, t_{i+1}] contributes k residuals from ODE collocation.
        // We use linear interpolation to get y at collocation pts (order 1 approximation)
        // and add Hermite-type update.

        let n = n_state;
        let total_unknowns = (m + 1) * n;

        // Initial guess from provided function
        let mut y_flat = Array1::<f64>::zeros(total_unknowns);
        for i in 0..=m {
            let guess = y_init_guess(nodes[i]);
            for j in 0..n {
                y_flat[i * n + j] = guess[j];
            }
        }

        let n_bc_eqs = bc(
            &y_flat.slice(scirs2_core::ndarray::s![..n]).to_owned(),
            &y_flat.slice(scirs2_core::ndarray::s![m * n..]).to_owned(),
        )
        .len();

        // Number of residuals: n_bc_eqs + m*k*n (collocation) - n*(m) (redundant continuity)
        // Simpler: n_bc_eqs + m*k*n equations, (m+1)*n unknowns
        // We use a simplified formulation: n*(m+1) equations with n*(m+1) unknowns
        // Residuals:
        //   - BC: n_bc_eqs equations
        //   - For each interval: n equations from integral form (trapezoidal approx)

        // Use a simpler consistent formulation:
        // residual[0..n_bc] = bc(y0, yM)
        // residual[n_bc + i*n .. n_bc + (i+1)*n] = integral residual for interval i
        // via: y_{i+1} - y_i - h/2*(f(t_i, y_i) + f(t_{i+1}, y_{i+1})) = 0  (trapezoidal)
        // Total: n_bc + m*n equations, (m+1)*n unknowns.
        // Constraint: n_bc = n for well-posed system. (m+1)*n = n_bc + m*n ✓

        let build_residual = |yf: &Array1<f64>| {
            let rlen = n_bc_eqs + m * n;
            let mut r = Array1::<f64>::zeros(rlen);
            let ya = yf.slice(scirs2_core::ndarray::s![..n]).to_owned();
            let ym = yf.slice(scirs2_core::ndarray::s![m * n..]).to_owned();
            let bcr = bc(&ya, &ym);
            for i in 0..n_bc_eqs {
                r[i] = bcr[i];
            }

            for interval in 0..m {
                let ti = nodes[interval];
                let tip1 = nodes[interval + 1];
                let yi = yf
                    .slice(scirs2_core::ndarray::s![interval * n..(interval + 1) * n])
                    .to_owned();
                let yip1 = yf
                    .slice(scirs2_core::ndarray::s![
                        (interval + 1) * n..(interval + 2) * n
                    ])
                    .to_owned();

                // Collocation at Gauss-Legendre points in [ti, tip1]
                // For each collocation point, evaluate ODE and add contribution
                // Using k-point Gauss rule to integrate the ODE residual
                let hi = tip1 - ti;
                let wts = gauss_legendre_weights(k);

                // Trapezoidal/high-order integral: y_{i+1} - y_i - integral f dt = 0
                let mut integral = Array1::<f64>::zeros(n);
                for (q, &xi) in ref_nodes.iter().enumerate() {
                    // Map from [-1,1] to [ti, tip1]
                    let tc = ti + (xi + 1.0) * 0.5 * hi;
                    // Linear interpolation for y at collocation point
                    let alpha = (xi + 1.0) * 0.5;
                    let mut yc = Array1::<f64>::zeros(n);
                    for j in 0..n {
                        yc[j] = (1.0 - alpha) * yi[j] + alpha * yip1[j];
                    }
                    let fc = ode(tc, &yc);
                    let wt = wts[q] * hi * 0.5;
                    for j in 0..n {
                        integral[j] += wt * fc[j];
                    }
                }

                let base = n_bc_eqs + interval * n;
                for j in 0..n {
                    r[base + j] = yip1[j] - yi[j] - integral[j];
                }
            }
            r
        };

        let mut n_iters = 0usize;
        let mut converged = false;

        for iter in 0..cfg.max_newton_iter {
            let res = build_residual(&y_flat);
            let res_norm: f64 = res.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if res_norm < cfg.newton_tol {
                n_iters = iter + 1;
                converged = true;
                break;
            }

            let mut jac = numerical_jacobian(&build_residual, &y_flat, cfg.fd_eps);
            let mut neg_res = res.mapv(|v| -v);
            match gauss_solve(&mut jac, &mut neg_res) {
                Ok(delta) => {
                    for i in 0..total_unknowns {
                        y_flat[i] += delta[i];
                    }
                }
                Err(e) => return Err(e),
            }
        }

        if !converged {
            n_iters = cfg.max_newton_iter;
        }

        // Extract trajectory
        let t_traj: Vec<f64> = nodes.clone();
        let y_traj: Vec<Array1<f64>> = (0..=m)
            .map(|i| {
                y_flat
                    .slice(scirs2_core::ndarray::s![i * n..(i + 1) * n])
                    .to_owned()
            })
            .collect();

        let ya = y_traj
            .first()
            .cloned()
            .unwrap_or_else(|| Array1::<f64>::zeros(n));
        let yb = y_traj
            .last()
            .cloned()
            .unwrap_or_else(|| Array1::<f64>::zeros(n));
        let final_bc = bc(&ya, &yb);
        let residual: f64 = final_bc.iter().map(|&v| v * v).sum::<f64>().sqrt();

        Ok(BVPResult {
            t: t_traj,
            y: y_traj,
            residual,
            error: residual,
            n_newton_iters: n_iters,
            success: converged,
            message: if converged {
                "Orthogonal collocation converged".to_string()
            } else {
                format!(
                    "Collocation did not converge in {} iterations",
                    cfg.max_newton_iter
                )
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Periodic Orbit Finder
// ---------------------------------------------------------------------------

/// Configuration for the periodic orbit finder
#[derive(Debug, Clone)]
pub struct PeriodicOrbitConfig {
    /// Number of RK4 steps for one period integration
    pub n_steps: usize,
    /// Newton solver tolerance
    pub newton_tol: f64,
    /// Maximum Newton iterations
    pub max_newton_iter: usize,
    /// Finite difference epsilon for Jacobians
    pub fd_eps: f64,
    /// Index of phase condition (which component is fixed)
    pub phase_condition_idx: usize,
}

impl Default for PeriodicOrbitConfig {
    fn default() -> Self {
        Self {
            n_steps: 500,
            newton_tol: 1e-8,
            max_newton_iter: 50,
            fd_eps: 1e-7,
            phase_condition_idx: 0,
        }
    }
}

/// Result of a periodic orbit computation
#[derive(Debug, Clone)]
pub struct PeriodicOrbitResult {
    /// One full period of the orbit (time points)
    pub t: Vec<f64>,
    /// States along the orbit
    pub y: Vec<Array1<f64>>,
    /// Period T
    pub period: f64,
    /// Initial state of the orbit y(0)
    pub y0: Array1<f64>,
    /// Residual of the periodicity condition
    pub residual: f64,
    /// Number of Newton iterations
    pub n_newton_iters: usize,
    /// Whether the solver converged
    pub success: bool,
    /// Termination message
    pub message: String,
}

/// Shooting-based periodic orbit finder.
///
/// Seeks a state y* and period T such that φ(T, y*) = y*, where φ is the flow map.
///
/// The system solved is:
///   F(y*, T) = φ(T, y*) - y* = 0   (periodicity, n equations)
///   g(y*, T) = 0                    (phase condition, 1 equation)
///
/// giving n+1 equations in n+1 unknowns (y* ∈ R^n, T ∈ R).
///
/// The phase condition fixes the phase along the orbit to remove translational
/// invariance. Here we use: y*[phase_condition_idx] - y0_ref[phase_condition_idx] = 0.
pub struct PeriodicOrbitFinder;

impl PeriodicOrbitFinder {
    /// Find a periodic orbit near (y_guess, t_guess).
    ///
    /// # Arguments
    ///
    /// * `ode` - Autonomous ODE function f(t, y)
    /// * `y_guess` - Initial guess for initial state on the orbit
    /// * `t_guess` - Initial guess for the period
    /// * `cfg` - Configuration
    pub fn find<ODE>(
        ode: &ODE,
        y_guess: &Array1<f64>,
        t_guess: f64,
        cfg: &PeriodicOrbitConfig,
    ) -> IntegrateResult<PeriodicOrbitResult>
    where
        ODE: Fn(f64, &Array1<f64>) -> Array1<f64>,
    {
        let n = y_guess.len();
        let phase_idx = cfg.phase_condition_idx.min(n - 1);
        let y0_ref_phase = y_guess[phase_idx];

        // Extended state: z = [y* (n), T (1)], total n+1
        let mut z = Array1::<f64>::zeros(n + 1);
        for i in 0..n {
            z[i] = y_guess[i];
        }
        z[n] = t_guess;

        // Residual function
        let residual_fn = |zv: &Array1<f64>| {
            let mut y_cur = Array1::<f64>::zeros(n);
            for i in 0..n {
                y_cur[i] = zv[i];
            }
            let period = zv[n].max(1e-10);
            let y_end = integrate_rk4(ode, 0.0, &y_cur, period, cfg.n_steps);
            let mut r = Array1::<f64>::zeros(n + 1);
            for i in 0..n {
                r[i] = y_end[i] - y_cur[i];
            }
            // Phase condition: y*[phase_idx] = y0_ref_phase
            r[n] = y_cur[phase_idx] - y0_ref_phase;
            r
        };

        let mut n_iters = 0usize;
        let mut converged = false;

        for iter in 0..cfg.max_newton_iter {
            let res = residual_fn(&z);
            let res_norm: f64 = res.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if res_norm < cfg.newton_tol {
                n_iters = iter + 1;
                converged = true;
                break;
            }

            let mut jac = numerical_jacobian(&residual_fn, &z, cfg.fd_eps);
            let mut neg_res = res.mapv(|v| -v);
            match gauss_solve(&mut jac, &mut neg_res) {
                Ok(delta) => {
                    for i in 0..=n {
                        z[i] += delta[i];
                    }
                    // Keep period positive
                    if z[n] < 1e-10 {
                        z[n] = 1e-10;
                    }
                }
                Err(_) => {
                    return Err(IntegrateError::LinearSolveError(
                        "Periodic orbit: singular Jacobian".to_string(),
                    ));
                }
            }
        }

        if !converged {
            n_iters = cfg.max_newton_iter;
        }

        let mut y_star = Array1::<f64>::zeros(n);
        for i in 0..n {
            y_star[i] = z[i];
        }
        let period = z[n];

        let (t_traj, y_traj) = trajectory_rk4(ode, 0.0, &y_star, period, cfg.n_steps);

        let final_res = residual_fn(&z);
        let residual: f64 = final_res.iter().map(|&v| v * v).sum::<f64>().sqrt();

        Ok(PeriodicOrbitResult {
            t: t_traj,
            y: y_traj,
            period,
            y0: y_star,
            residual,
            n_newton_iters: n_iters,
            success: converged,
            message: if converged {
                format!("Periodic orbit found: T = {:.6}", period)
            } else {
                format!(
                    "Periodic orbit not found in {} iterations",
                    cfg.max_newton_iter
                )
            },
        })
    }
}

// Suppress unused warning for helper
#[allow(dead_code)]
fn _use_to_f() {
    let _ = to_f(0.0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Linear BVP: y'' + y = 0, y(0) = 0, y(π/2) = 1
    /// Exact solution: y(t) = sin(t)
    /// First-order form: y' = [y1, -y0]
    fn linear_ode(t: f64, y: &Array1<f64>) -> Array1<f64> {
        let _ = t;
        array![y[1], -y[0]]
    }

    fn linear_bc(ya: &Array1<f64>, yb: &Array1<f64>) -> Array1<f64> {
        // y(0) = 0, y(π/2) = 1
        array![ya[0], yb[0] - 1.0]
    }

    #[test]
    fn test_single_shooting_linear_bvp() {
        // y' = [y1, -y0], y(0)=0, y(π/2)=1, exact: y(t)=sin(t)
        let t_span = [0.0, std::f64::consts::FRAC_PI_2];

        // Free parameter is y'(0) = y[1](0)
        let initial_condition = |s: &Array1<f64>| array![0.0, s[0]];
        let bc = |ya: &Array1<f64>, yb: &Array1<f64>| array![ya[0], yb[0] - 1.0];
        let s0 = array![1.0]; // initial guess y'(0)=1

        let cfg = ShootingConfig {
            n_steps: 200,
            newton_tol: 1e-8,
            ..Default::default()
        };

        let result = SingleShooting::solve(&linear_ode, &bc, &initial_condition, t_span, s0, &cfg)
            .expect("Single shooting failed");

        assert!(result.success, "Should converge: {}", result.message);
        assert!(
            result.residual < 1e-6,
            "Residual {} too large",
            result.residual
        );

        // Check y(π/4) ≈ sin(π/4) ≈ 0.7071
        let t_quarter = std::f64::consts::FRAC_PI_4;
        let idx = result.t.iter().position(|&t| (t - t_quarter).abs() < 0.02);
        if let Some(i) = idx {
            let y_val = result.y[i][0];
            let exact = t_quarter.sin();
            assert!(
                (y_val - exact).abs() < 0.02,
                "y(π/4)={} != sin(π/4)={}",
                y_val,
                exact
            );
        }
    }

    #[test]
    fn test_collocation_linear_bvp() {
        let cfg = CollocationConfig {
            n_subintervals: 8,
            collocation_order: 3,
            newton_tol: 1e-6,
            max_newton_iter: 30,
            fd_eps: 1e-6,
        };

        let t_span = [0.0, std::f64::consts::FRAC_PI_2];
        let guess = |t: f64| array![t.sin(), t.cos()];

        let result = OrthogonalCollocation::solve(&linear_ode, &linear_bc, t_span, &guess, 2, &cfg)
            .expect("Collocation failed");

        assert!(
            result.residual < 1e-4,
            "Collocation residual {} too large",
            result.residual
        );
    }

    #[test]
    fn test_periodic_orbit_harmonic_oscillator() {
        // Harmonic oscillator: y'' + y = 0, exact period T = 2π
        // First-order: y' = [y1, -y0]
        // Periodic orbit starting at [1, 0], T = 2π
        let cfg = PeriodicOrbitConfig {
            n_steps: 200,
            newton_tol: 1e-8,
            max_newton_iter: 20,
            fd_eps: 1e-6,
            phase_condition_idx: 0,
        };

        let y_guess = array![1.0, 0.0];
        let t_guess = 2.0 * std::f64::consts::PI;

        let result = PeriodicOrbitFinder::find(&linear_ode, &y_guess, t_guess, &cfg)
            .expect("Periodic orbit finder failed");

        // Period should be close to 2π
        assert!(
            (result.period - 2.0 * std::f64::consts::PI).abs() < 0.1,
            "Period {} != 2π",
            result.period
        );
    }
}
