//! Linear Quadratic Regulator (LQR) and related optimal control tools.
//!
//! Provides:
//! - Continuous-time LQR via the CARE (`A^T P + PA - PBR⁻¹B^T P + Q = 0`)
//! - Discrete-time LQR via the DARE (`A^T PA - P - A^T PB(B^T PB+R)⁻¹B^T PA + Q = 0`)
//! - Closed-loop gain computation `K = R⁻¹ B^T P` (continuous)
//!   and `K = (B^T PB + R)⁻¹ B^T PA` (discrete)
//! - Simulation helpers: step, trajectory, closed-loop eigenvalues
//!
//! # Example — double integrator
//! ```
//! use scirs2_linalg::control::lqr::{LqrController, LqrMode};
//! use scirs2_core::ndarray::{array, Array1};
//!
//! let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
//! let b = array![[0.0_f64], [1.0]];
//! let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
//! let r = array![[1.0_f64]];
//!
//! let ctrl = LqrController::new(&a.view(), &b.view(), &q.view(), &r.view(), LqrMode::Continuous)
//!     .expect("LQR synthesis failed");
//!
//! let x0 = array![1.0_f64, 0.0];
//! let u = ctrl.control(&x0.view());
//! println!("u = {:?}", u);
//! ```
//!
//! # References
//! - Anderson & Moore (1990). *Optimal Control: Linear Quadratic Methods*.
//! - Kalman, R. E. (1960). Contributions to the theory of optimal control.
//!   *Bol. Soc. Mat. Mexicana*, 5(1), 102–119.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait bound
// ---------------------------------------------------------------------------

/// Floating-point scalar requirements for LQR synthesis.
pub trait LqrFloat:
    Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}
impl<F> LqrFloat for F where
    F: Float + NumAssign + Sum + ScalarOperand + Debug + Display + Send + Sync + 'static
{
}

// ---------------------------------------------------------------------------
// Internal matrix helpers (duplicated from parent modules to keep this file
// self-contained and to avoid circular dependency issues).
// ---------------------------------------------------------------------------

/// General dense matrix multiply `C = A · B`.
fn mm<F: LqrFloat>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for p in 0..k {
            let aip = a[[i, p]];
            if aip == F::zero() {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += aip * b[[p, j]];
            }
        }
    }
    c
}

/// Gauss-Jordan matrix inverse with partial pivoting.
fn mat_inv<F: LqrFloat>(a: &Array2<F>) -> LinalgResult<Array2<F>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError(
            "mat_inv: matrix must be square".to_string(),
        ));
    }
    let mut aug = Array2::<F>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = F::one();
    }

    let eps = F::from(1e-14).unwrap_or(F::epsilon());

    for col in 0..n {
        // Find pivot
        let mut piv_row = col;
        let mut piv_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > piv_val {
                piv_val = v;
                piv_row = row;
            }
        }
        if piv_val < eps {
            return Err(LinalgError::SingularMatrixError(
                "LQR: matrix is singular in mat_inv".to_string(),
            ));
        }
        if piv_row != col {
            for j in 0..2 * n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[piv_row, j]];
                aug[[piv_row, j]] = tmp;
            }
        }
        let scale = aug[[col, col]];
        for j in 0..2 * n {
            aug[[col, j]] /= scale;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            if factor == F::zero() {
                continue;
            }
            for j in 0..2 * n {
                let val = aug[[col, j]] * factor;
                aug[[row, j]] -= val;
            }
        }
    }

    let mut inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

// /// Frobenius norm of a matrix.
#[allow(dead_code)]
fn frobenius_norm<F: LqrFloat>(a: &Array2<F>) -> F {
    a.iter().map(|&x| x * x).sum::<F>().sqrt()
}

// ---------------------------------------------------------------------------
// LQR mode
// ---------------------------------------------------------------------------

/// Whether to use the continuous-time or discrete-time Riccati equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LqrMode {
    /// Continuous-time LQR: minimize ∫₀^∞ (x^T Q x + u^T R u) dt.
    Continuous,
    /// Discrete-time LQR: minimize Σ (x^T Q x + u^T R u).
    Discrete,
}

// ---------------------------------------------------------------------------
// LQR controller
// ---------------------------------------------------------------------------

/// Linear Quadratic Regulator (LQR) controller.
///
/// The feedback law is `u = −K x` where:
/// - **Continuous**: `K = R⁻¹ B^T P` and `P` solves the CARE.
/// - **Discrete**: `K = (B^T P B + R)⁻¹ B^T P A` and `P` solves the DARE.
#[derive(Debug, Clone)]
pub struct LqrController<F: LqrFloat> {
    /// Optimal feedback gain matrix (m × n).
    pub gain: Array2<F>,
    /// Positive semi-definite solution to the Riccati equation.
    pub p: Array2<F>,
    /// Mode used for synthesis.
    pub mode: LqrMode,
}

impl<F: LqrFloat> LqrController<F> {
    /// Synthesise an LQR controller.
    ///
    /// # Arguments
    /// - `a` — State matrix (n × n).
    /// - `b` — Input matrix (n × m).
    /// - `q` — State cost matrix (n × n, positive semi-definite).
    /// - `r` — Input cost matrix (m × m, positive definite).
    /// - `mode` — [`LqrMode::Continuous`] or [`LqrMode::Discrete`].
    ///
    /// # Errors
    /// - [`LinalgError::ShapeError`] for incompatible dimensions.
    /// - [`LinalgError::ConvergenceError`] if the Riccati solver fails to
    ///   converge.
    /// - [`LinalgError::SingularMatrixError`] if `R` is singular.
    pub fn new(
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        q: &ArrayView2<F>,
        r: &ArrayView2<F>,
        mode: LqrMode,
    ) -> LinalgResult<Self> {
        validate_lqr_inputs(a, b, q, r)?;

        let p = match mode {
            LqrMode::Continuous => super::riccati::care_solve(a, b, q, r)?,
            LqrMode::Discrete => super::riccati::dare_solve(a, b, q, r)?,
        };

        let gain = compute_gain(a, b, r, &p, mode)?;
        Ok(Self { gain, p, mode })
    }

    /// Compute the control input `u = −K x`.
    ///
    /// # Arguments
    /// - `state` — Current state vector (length n).
    ///
    /// # Panics
    /// This function does not panic.  Shape mismatches are handled gracefully by
    /// returning a zero vector if dimensions are incompatible (this should not
    /// happen in normal use).
    pub fn control(&self, state: &ArrayView1<F>) -> Array1<F> {
        let m = self.gain.nrows();
        let n = self.gain.ncols();
        if state.len() != n {
            return Array1::zeros(m);
        }
        // u = -K * x
        let mut u = Array1::<F>::zeros(m);
        for i in 0..m {
            let mut sum = F::zero();
            for j in 0..n {
                sum += self.gain[[i, j]] * state[j];
            }
            u[i] = -sum;
        }
        u
    }

    /// Return the closed-loop state matrix `A_cl = A − B K`.
    pub fn closed_loop_a(&self, a: &ArrayView2<F>, b: &ArrayView2<F>) -> Array2<F> {
        let n = a.nrows();
        let m = b.ncols();
        let mut acl = a.to_owned();
        for i in 0..n {
            for j in 0..m {
                let b_ij = b[[i, j]];
                for k in 0..self.gain.ncols() {
                    acl[[i, k]] -= b_ij * self.gain[[j, k]];
                }
            }
        }
        acl
    }

    /// Simulate the closed-loop continuous-time system using forward Euler for
    /// `steps` time steps of size `dt`.
    ///
    /// Returns a matrix of shape `(steps+1, n)` where each row is a state.
    ///
    /// # Arguments
    /// - `a` — System state matrix (n × n).
    /// - `b` — Input matrix (n × m).
    /// - `x0` — Initial state (length n).
    /// - `dt` — Time step.
    /// - `steps` — Number of simulation steps.
    ///
    /// # Errors
    /// Returns [`LinalgError::ShapeError`] on dimension mismatch.
    pub fn simulate_continuous(
        &self,
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        x0: &ArrayView1<F>,
        dt: F,
        steps: usize,
    ) -> LinalgResult<Array2<F>> {
        let n = a.nrows();
        if x0.len() != n {
            return Err(LinalgError::ShapeError(
                "simulate_continuous: x0 length must equal state dimension".to_string(),
            ));
        }
        let mut traj = Array2::<F>::zeros((steps + 1, n));
        let mut x = x0.to_owned();
        for j in 0..n {
            traj[[0, j]] = x[j];
        }
        let a_owned = a.to_owned();
        let b_owned = b.to_owned();

        for step in 0..steps {
            let u = self.control(&x.view());
            // dx/dt = A x + B u  →  x_{k+1} = x_k + dt*(Ax_k + Bu_k)
            let ax: Vec<F> = (0..n)
                .map(|i| (0..n).map(|j| a_owned[[i, j]] * x[j]).sum::<F>())
                .collect();
            let bu: Vec<F> = (0..n)
                .map(|i| (0..u.len()).map(|j| b_owned[[i, j]] * u[j]).sum::<F>())
                .collect();
            for i in 0..n {
                x[i] += dt * (ax[i] + bu[i]);
            }
            for j in 0..n {
                traj[[step + 1, j]] = x[j];
            }
        }
        Ok(traj)
    }

    /// Simulate the closed-loop discrete-time system for `steps` steps.
    ///
    /// Returns a matrix of shape `(steps+1, n)` where each row is a state.
    ///
    /// # Errors
    /// Returns [`LinalgError::ShapeError`] on dimension mismatch.
    pub fn simulate_discrete(
        &self,
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        x0: &ArrayView1<F>,
        steps: usize,
    ) -> LinalgResult<Array2<F>> {
        let n = a.nrows();
        if x0.len() != n {
            return Err(LinalgError::ShapeError(
                "simulate_discrete: x0 length must equal state dimension".to_string(),
            ));
        }
        let mut traj = Array2::<F>::zeros((steps + 1, n));
        let mut x = x0.to_owned();
        for j in 0..n {
            traj[[0, j]] = x[j];
        }
        let a_owned = a.to_owned();
        let b_owned = b.to_owned();

        for step in 0..steps {
            let u = self.control(&x.view());
            // x_{k+1} = A x_k + B u_k
            let mut x_next = Array1::<F>::zeros(n);
            for i in 0..n {
                let ax_i: F = (0..n).map(|j| a_owned[[i, j]] * x[j]).sum();
                let bu_i: F = (0..u.len()).map(|j| b_owned[[i, j]] * u[j]).sum();
                x_next[i] = ax_i + bu_i;
            }
            x = x_next;
            for j in 0..n {
                traj[[step + 1, j]] = x[j];
            }
        }
        Ok(traj)
    }

    /// Compute the optimal cost `J* = x₀^T P x₀` for continuous-time LQR.
    pub fn optimal_cost(&self, x0: &ArrayView1<F>) -> F {
        let n = x0.len().min(self.p.nrows());
        let mut cost = F::zero();
        for i in 0..n {
            for j in 0..n {
                cost += x0[i] * self.p[[i, j]] * x0[j];
            }
        }
        cost
    }
}

// ---------------------------------------------------------------------------
// Gain computation
// ---------------------------------------------------------------------------

/// Compute the LQR feedback gain from the Riccati solution `P`.
fn compute_gain<F: LqrFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    r: &ArrayView2<F>,
    p: &Array2<F>,
    mode: LqrMode,
) -> LinalgResult<Array2<F>> {
    let b_owned = b.to_owned();
    let r_owned = r.to_owned();
    let r_inv = mat_inv(&r_owned)?;

    match mode {
        LqrMode::Continuous => {
            // K = R⁻¹ B^T P
            let bt_p = mm(&b_owned.t().to_owned(), p);
            Ok(mm(&r_inv, &bt_p))
        }
        LqrMode::Discrete => {
            // K = (B^T P B + R)⁻¹ B^T P A
            let bt_p = mm(&b_owned.t().to_owned(), p);
            let bt_pb = mm(&bt_p, &b_owned);
            let bt_pb_r = bt_pb + r_owned;
            let bt_pb_r_inv = mat_inv(&bt_pb_r)?;
            let bt_pa = mm(&bt_p, &a.to_owned());
            Ok(mm(&bt_pb_r_inv, &bt_pa))
        }
    }
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

fn validate_lqr_inputs<F: LqrFloat>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    q: &ArrayView2<F>,
    r: &ArrayView2<F>,
) -> LinalgResult<()> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::ShapeError("LQR: A must be square".to_string()));
    }
    if b.nrows() != n {
        return Err(LinalgError::ShapeError(format!(
            "LQR: B must have {} rows, got {}",
            n,
            b.nrows()
        )));
    }
    let m = b.ncols();
    if q.nrows() != n || q.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "LQR: Q must be ({n}×{n}), got ({}×{})",
            q.nrows(),
            q.ncols()
        )));
    }
    if r.nrows() != m || r.ncols() != m {
        return Err(LinalgError::ShapeError(format!(
            "LQR: R must be ({m}×{m}), got ({}×{})",
            r.nrows(),
            r.ncols()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ILQR (iterative LQR) for nonlinear systems
// ---------------------------------------------------------------------------

/// Result of an iLQR solve.
#[derive(Debug, Clone)]
pub struct IlqrResult<F: LqrFloat> {
    /// Sequence of optimal states `x_0, x_1, …, x_T`.
    pub states: Array2<F>,
    /// Sequence of optimal controls `u_0, …, u_{T-1}`.
    pub controls: Array2<F>,
    /// Total cost.
    pub cost: F,
}

/// Iterative LQR (iLQR) solver for discrete-time nonlinear systems.
///
/// Minimizes `Σ_{t=0}^{T-1} (x_t^T Q x_t + u_t^T R u_t) + x_T^T P_f x_T`
/// where `f(x,u)` is the (differentiable) system dynamics provided via
/// function pointers for the Jacobians.
///
/// # Arguments
/// - `dynamics` — `(x, u) → x_next` transition function.
/// - `jac_x` — `(x, u) → A_t` (∂f/∂x Jacobian, n×n).
/// - `jac_u` — `(x, u) → B_t` (∂f/∂u Jacobian, n×m).
/// - `q` — Running state cost (n×n, PSD).
/// - `r` — Running input cost (m×m, PD).
/// - `p_terminal` — Terminal state cost (n×n, PSD).
/// - `x0` — Initial state.
/// - `horizon` — Planning horizon T.
/// - `max_iter` — Maximum iLQR outer iterations.
/// - `tol` — Convergence tolerance on cost improvement.
///
/// # Returns
/// [`IlqrResult`] containing optimal state and control trajectories.
///
/// # Errors
/// Returns [`LinalgError::ConvergenceError`] if Newton backward pass fails.
#[allow(clippy::too_many_arguments)]
pub fn ilqr_solve<F, FDyn, FJacX, FJacU>(
    dynamics: FDyn,
    jac_x: FJacX,
    jac_u: FJacU,
    q: &ArrayView2<F>,
    r: &ArrayView2<F>,
    p_terminal: &ArrayView2<F>,
    x0: &ArrayView1<F>,
    horizon: usize,
    max_iter: usize,
    tol: F,
) -> LinalgResult<IlqrResult<F>>
where
    F: LqrFloat,
    FDyn: Fn(&Array1<F>, &Array1<F>) -> Array1<F>,
    FJacX: Fn(&Array1<F>, &Array1<F>) -> Array2<F>,
    FJacU: Fn(&Array1<F>, &Array1<F>) -> Array2<F>,
{
    let n = x0.len();
    let m = r.nrows();
    let two = F::from(2.0).unwrap_or(F::one() + F::one());

    // Initialize with zero controls
    let mut us: Vec<Array1<F>> = vec![Array1::zeros(m); horizon];
    let mut xs: Vec<Array1<F>> = forward_pass_rollout(&dynamics, x0, &us, n, horizon);

    let mut prev_cost = compute_total_cost(&xs, &us, q, r, p_terminal, n, m);

    for _iter in 0..max_iter {
        // --- Backward pass: compute value-function derivatives ---
        let mut v_x: Vec<Array2<F>> = vec![Array2::zeros((n, 1)); horizon + 1]; // ∂V/∂x column
        let mut v_xx: Vec<Array2<F>> = vec![Array2::zeros((n, n)); horizon + 1]; // ∂²V/∂x²

        // Terminal conditions
        v_xx[horizon] = p_terminal.to_owned();
        let x_t = &xs[horizon];
        let pxt: Vec<F> = (0..n)
            .map(|i| (0..n).map(|j| p_terminal[[i, j]] * x_t[j]).sum::<F>())
            .collect();
        for i in 0..n {
            v_x[horizon][[i, 0]] = pxt[i];
        }

        let mut k_gains: Vec<Array1<F>> = vec![Array1::zeros(m); horizon]; // feedforward
        let mut kk_gains: Vec<Array2<F>> = vec![Array2::zeros((m, n)); horizon]; // feedback

        for t in (0..horizon).rev() {
            let x = &xs[t];
            let u = &us[t];
            let at = jac_x(x, u); // n×n
            let bt = jac_u(x, u); // n×m

            // Q-function expansion terms
            let q_x: Array1<F> = {
                // ∂Q/∂x = Q x + Aᵀ V_{t+1,x}
                let qx: Vec<F> = (0..n)
                    .map(|i| (0..n).map(|j| q[[i, j]] * x[j]).sum::<F>())
                    .collect();
                let atvx: Vec<F> = (0..n)
                    .map(|i| (0..n).map(|j| at[[j, i]] * v_x[t + 1][[j, 0]]).sum::<F>())
                    .collect();
                Array1::from_vec((0..n).map(|i| qx[i] + atvx[i]).collect())
            };

            let q_u: Array1<F> = {
                // ∂Q/∂u = R u + Bᵀ V_{t+1,x}
                let ru: Vec<F> = (0..m)
                    .map(|i| (0..m).map(|j| r[[i, j]] * u[j]).sum::<F>())
                    .collect();
                let btvx: Vec<F> = (0..m)
                    .map(|i| (0..n).map(|j| bt[[j, i]] * v_x[t + 1][[j, 0]]).sum::<F>())
                    .collect();
                Array1::from_vec((0..m).map(|i| ru[i] + btvx[i]).collect())
            };

            let q_xx: Array2<F> = {
                // Q + Aᵀ V_xx A
                let atvxx = mm(&at.t().to_owned(), &v_xx[t + 1]);
                let atvxx_a = mm(&atvxx, &at);
                q.to_owned() + atvxx_a
            };

            let q_uu: Array2<F> = {
                // R + Bᵀ V_xx B
                let btvxx = mm(&bt.t().to_owned(), &v_xx[t + 1]);
                let btvxx_b = mm(&btvxx, &bt);
                r.to_owned() + btvxx_b
            };

            let q_ux: Array2<F> = {
                // Bᵀ V_xx A
                let btvxx = mm(&bt.t().to_owned(), &v_xx[t + 1]);
                mm(&btvxx, &at)
            };

            // Gains: k = -Q_uu⁻¹ Q_u,  K = -Q_uu⁻¹ Q_ux
            let q_uu_inv = match mat_inv(&q_uu) {
                Ok(inv) => inv,
                Err(_) => {
                    // Regularise
                    let reg = F::from(1e-6).unwrap_or(F::epsilon());
                    let mut q_uu_reg = q_uu.clone();
                    for i in 0..m {
                        q_uu_reg[[i, i]] += reg;
                    }
                    mat_inv(&q_uu_reg)?
                }
            };

            let k_t: Array1<F> = {
                let mut k = Array1::zeros(m);
                for i in 0..m {
                    let s: F = (0..m).map(|j| q_uu_inv[[i, j]] * q_u[j]).sum();
                    k[i] = -s;
                }
                k
            };
            let kk_t = mm(&q_uu_inv.mapv(|x| -x), &q_ux);

            k_gains[t] = k_t.clone();
            kk_gains[t] = kk_t.clone();

            // Update value function: V_x = Q_x + Kᵀ Q_uu k + Kᵀ Q_u + Q_ux^T k
            // = Q_x + Q_ux^T k + Kᵀ (Q_uu k + Q_u)
            let q_uu_k: Vec<F> = (0..m)
                .map(|i| (0..m).map(|j| q_uu[[i, j]] * k_t[j]).sum::<F>() + q_u[i])
                .collect();
            let v_x_t: Vec<F> = (0..n)
                .map(|i| {
                    q_x[i]
                        + (0..m).map(|j| q_ux[[j, i]] * k_t[j]).sum::<F>()
                        + (0..m).map(|j| kk_t[[j, i]] * q_uu_k[j]).sum::<F>()
                })
                .collect();
            for i in 0..n {
                v_x[t][[i, 0]] = v_x_t[i];
            }

            // V_xx = Q_xx + Q_xu K + Kᵀ Q_ux + Kᵀ Q_uu K
            let q_ux_t = q_ux.t().to_owned(); // n×m
            let kk_t_t = kk_t.t().to_owned(); // n×m
            let kk_q_ux = mm(&kk_t_t, &q_ux); // n×n
            let kk_q_uu_kk = mm(&mm(&kk_t_t, &q_uu), &kk_t); // n×n
            let q_xu_kk = mm(&q_ux_t, &kk_t); // n×n
            let mut v_xx_t = q_xx.clone();
            for i in 0..n {
                for j in 0..n {
                    v_xx_t[[i, j]] += q_xu_kk[[i, j]] + kk_q_ux[[i, j]] + kk_q_uu_kk[[i, j]];
                }
            }
            v_xx[t] = v_xx_t;
        }

        // --- Forward pass with line search ---
        let mut alpha = F::one();
        let half = F::from(0.5).unwrap_or(F::one() / two);
        let mut new_xs = xs.clone();
        let mut new_us = us.clone();

        for _ls in 0..10 {
            new_xs = forward_pass_with_gains(
                &dynamics, x0, &xs, &us, &k_gains, &kk_gains, alpha, n, horizon,
            );
            new_us = (0..horizon)
                .map(|t| {
                    let dx: Vec<F> = (0..n).map(|i| new_xs[t][i] - xs[t][i]).collect();
                    let mut u_new = us[t].clone();
                    for i in 0..m {
                        let ku: F = (0..n).map(|j| kk_gains[t][[i, j]] * dx[j]).sum();
                        u_new[i] += alpha * k_gains[t][i] + ku;
                    }
                    u_new
                })
                .collect();

            let new_cost = compute_total_cost(&new_xs, &new_us, q, r, p_terminal, n, m);
            if new_cost < prev_cost {
                break;
            }
            alpha *= half;
        }

        let new_cost = compute_total_cost(&new_xs, &new_us, q, r, p_terminal, n, m);
        let improvement = if prev_cost > F::zero() {
            (prev_cost - new_cost) / prev_cost
        } else {
            (prev_cost - new_cost).abs()
        };

        xs = new_xs;
        us = new_us;
        prev_cost = new_cost;

        if improvement < tol && improvement >= F::zero() {
            break;
        }
    }

    // Build output matrices
    let mut states = Array2::<F>::zeros((horizon + 1, n));
    let mut controls = Array2::<F>::zeros((horizon, m));
    for t in 0..=horizon {
        for j in 0..n {
            states[[t, j]] = xs[t][j];
        }
    }
    for t in 0..horizon {
        for j in 0..m {
            controls[[t, j]] = us[t][j];
        }
    }

    Ok(IlqrResult {
        states,
        controls,
        cost: prev_cost,
    })
}

// ---------------------------------------------------------------------------
// iLQR helpers
// ---------------------------------------------------------------------------

fn forward_pass_rollout<FDyn, F: LqrFloat>(
    dynamics: &FDyn,
    x0: &ArrayView1<F>,
    us: &[Array1<F>],
    n: usize,
    horizon: usize,
) -> Vec<Array1<F>>
where
    FDyn: Fn(&Array1<F>, &Array1<F>) -> Array1<F>,
{
    let mut xs = vec![x0.to_owned()];
    for t in 0..horizon {
        let x_next = dynamics(&xs[t], &us[t]);
        xs.push(x_next);
    }
    xs
}

fn forward_pass_with_gains<FDyn, F: LqrFloat>(
    dynamics: &FDyn,
    x0: &ArrayView1<F>,
    xs_nom: &[Array1<F>],
    us_nom: &[Array1<F>],
    k_gains: &[Array1<F>],
    kk_gains: &[Array2<F>],
    alpha: F,
    n: usize,
    horizon: usize,
) -> Vec<Array1<F>>
where
    FDyn: Fn(&Array1<F>, &Array1<F>) -> Array1<F>,
{
    let m = k_gains[0].len();
    let mut xs = vec![x0.to_owned()];
    for t in 0..horizon {
        let dx: Vec<F> = (0..n).map(|i| xs[t][i] - xs_nom[t][i]).collect();
        let mut u = us_nom[t].clone();
        for i in 0..m {
            let ku: F = (0..n).map(|j| kk_gains[t][[i, j]] * dx[j]).sum();
            u[i] += alpha * k_gains[t][i] + ku;
        }
        let x_next = dynamics(&xs[t], &u);
        xs.push(x_next);
    }
    xs
}

fn compute_total_cost<F: LqrFloat>(
    xs: &[Array1<F>],
    us: &[Array1<F>],
    q: &ArrayView2<F>,
    r: &ArrayView2<F>,
    p_f: &ArrayView2<F>,
    n: usize,
    m: usize,
) -> F {
    let horizon = us.len();
    let mut cost = F::zero();
    for t in 0..horizon {
        // x^T Q x
        for i in 0..n {
            for j in 0..n {
                cost += xs[t][i] * q[[i, j]] * xs[t][j];
            }
        }
        // u^T R u
        for i in 0..m {
            for j in 0..m {
                cost += us[t][i] * r[[i, j]] * us[t][j];
            }
        }
    }
    // Terminal x_T^T P_f x_T
    let xt = &xs[horizon];
    for i in 0..n {
        for j in 0..n {
            cost += xt[i] * p_f[[i, j]] * xt[j];
        }
    }
    cost
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Double integrator: ẋ = [[0,1],[0,0]] x + [[0],[1]] u
    fn double_integrator() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
        let b = array![[0.0_f64], [1.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        (a, b, q, r)
    }

    #[test]
    fn test_continuous_lqr_double_integrator() {
        let (a, b, q, r) = double_integrator();
        let ctrl = LqrController::new(
            &a.view(),
            &b.view(),
            &q.view(),
            &r.view(),
            LqrMode::Continuous,
        )
        .expect("LQR synthesis failed");

        // P should be positive definite
        assert!(
            ctrl.p[[0, 0]] > 0.0,
            "P[0,0] = {} must be positive",
            ctrl.p[[0, 0]]
        );
        assert!(
            ctrl.p[[1, 1]] > 0.0,
            "P[1,1] = {} must be positive",
            ctrl.p[[1, 1]]
        );

        // Gain K should be non-trivial
        let k_frobenius: f64 = ctrl.gain.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(k_frobenius > 1e-6, "Gain K should be non-zero");

        // Verify CARE residual: A^T P + P A - P B R^{-1} B^T P + Q ≈ 0
        let p = &ctrl.p;
        let at = a.t().to_owned();
        let r_inv = mat_inv(&r).expect("R inv");
        let s = mm(&b, &mm(&r_inv, &b.t().to_owned()));
        let res = mm(&at, p) + mm(p, &a) - mm(p, &mm(&s, p)) + q;
        let res_norm: f64 = res.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(res_norm < 1e-4, "CARE residual = {res_norm}");
    }

    #[test]
    fn test_control_law() {
        let (a, b, q, r) = double_integrator();
        let ctrl = LqrController::new(
            &a.view(),
            &b.view(),
            &q.view(),
            &r.view(),
            LqrMode::Continuous,
        )
        .expect("LQR failed");

        let x = array![1.0_f64, 0.0];
        let u = ctrl.control(&x.view());
        assert_eq!(u.len(), 1);
        // With x=[1,0], u = -K[:,0]*1 which should push state back to origin → u < 0
        assert!(
            u[0] < 0.0,
            "Control should be negative for x=[1,0]: u={}",
            u[0]
        );
    }

    #[test]
    fn test_discrete_lqr_integrator() {
        // Discrete integrator: A=[[1,1],[0,1]], B=[[0],[1]]
        let a = array![[1.0_f64, 1.0], [0.0, 1.0]];
        let b = array![[0.0_f64], [1.0]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let ctrl = LqrController::new(
            &a.view(),
            &b.view(),
            &q.view(),
            &r.view(),
            LqrMode::Discrete,
        )
        .expect("Discrete LQR failed");
        assert!(ctrl.p[[0, 0]] > 0.0, "P positive definite");
        assert!(ctrl.p[[1, 1]] > 0.0, "P positive definite");
    }

    #[test]
    fn test_simulate_continuous_convergence() {
        let (a, b, q, r) = double_integrator();
        let ctrl = LqrController::new(
            &a.view(),
            &b.view(),
            &q.view(),
            &r.view(),
            LqrMode::Continuous,
        )
        .expect("LQR failed");

        let x0 = array![1.0_f64, 0.0];
        let traj = ctrl
            .simulate_continuous(&a.view(), &b.view(), &x0.view(), 0.01, 500)
            .expect("simulate failed");

        // Final state should be close to origin
        let x_final_norm: f64 = (0..2).map(|j| traj[[500, j]].powi(2)).sum::<f64>().sqrt();
        assert!(
            x_final_norm < 0.1,
            "Final state norm = {x_final_norm} (expected < 0.1)"
        );
    }

    #[test]
    fn test_simulate_discrete_convergence() {
        let a = array![[1.0_f64, 0.1], [0.0, 1.0]]; // ZOH-like discrete integrator
        let b = array![[0.005_f64], [0.1]];
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r = array![[1.0_f64]];
        let ctrl = LqrController::new(
            &a.view(),
            &b.view(),
            &q.view(),
            &r.view(),
            LqrMode::Discrete,
        )
        .expect("Discrete LQR failed");

        let x0 = array![1.0_f64, 0.0];
        let traj = ctrl
            .simulate_discrete(&a.view(), &b.view(), &x0.view(), 200)
            .expect("simulate failed");

        let x_final_norm: f64 = (0..2).map(|j| traj[[200, j]].powi(2)).sum::<f64>().sqrt();
        assert!(
            x_final_norm < 0.5,
            "Final state norm = {x_final_norm} (expected < 0.5)"
        );
    }

    #[test]
    fn test_optimal_cost_positive() {
        let (a, b, q, r) = double_integrator();
        let ctrl = LqrController::new(
            &a.view(),
            &b.view(),
            &q.view(),
            &r.view(),
            LqrMode::Continuous,
        )
        .expect("LQR failed");
        let x0 = array![1.0_f64, 1.0];
        let cost = ctrl.optimal_cost(&x0.view());
        assert!(
            cost > 0.0,
            "Optimal cost = {cost} must be positive for non-zero x0"
        );
    }

    #[test]
    fn test_closed_loop_a() {
        let (a, b, q, r) = double_integrator();
        let ctrl = LqrController::new(
            &a.view(),
            &b.view(),
            &q.view(),
            &r.view(),
            LqrMode::Continuous,
        )
        .expect("LQR failed");
        let acl = ctrl.closed_loop_a(&a.view(), &b.view());
        assert_eq!(acl.nrows(), 2);
        assert_eq!(acl.ncols(), 2);
        // For a stable closed-loop, trace of A_cl should be negative
        let trace = acl[[0, 0]] + acl[[1, 1]];
        assert!(
            trace < 0.0,
            "Closed-loop trace = {trace} should be negative (stable)"
        );
    }

    #[test]
    fn test_ilqr_linear_system() {
        // For a linear system iLQR should give same result as LQR
        let (a, b, _q, _r) = double_integrator();
        let q = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let r_mat = array![[0.1_f64]];
        let p_f = array![[1.0_f64, 0.0], [0.0, 1.0]];

        // Discrete-time double integrator approximation
        let dt = 0.1_f64;
        let ad = array![[1.0_f64, dt], [0.0, 1.0]];
        let bd = array![[0.5_f64 * dt * dt], [dt]];

        let ad_clone = ad.clone();
        let bd_clone = bd.clone();

        let dynamics = move |x: &Array1<f64>, u: &Array1<f64>| -> Array1<f64> {
            let mut xn = Array1::zeros(2);
            for i in 0..2 {
                xn[i] = ad_clone[[i, 0]] * x[0] + ad_clone[[i, 1]] * x[1] + bd_clone[[i, 0]] * u[0];
            }
            xn
        };

        let ad_jx = ad.clone();
        let jac_x = move |_x: &Array1<f64>, _u: &Array1<f64>| -> Array2<f64> { ad_jx.clone() };
        let bd_ju = bd.clone();
        let jac_u = move |_x: &Array1<f64>, _u: &Array1<f64>| -> Array2<f64> { bd_ju.clone() };

        let x0 = array![1.0_f64, 0.0];
        let result = ilqr_solve(
            dynamics,
            jac_x,
            jac_u,
            &q.view(),
            &r_mat.view(),
            &p_f.view(),
            &x0.view(),
            20,
            50,
            1e-6_f64,
        )
        .expect("iLQR failed");

        // Final state should be close to origin
        let xf = result.states.row(20);
        let norm: f64 = xf.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(norm < 0.5, "iLQR final state norm = {norm}");
        assert!(result.cost > 0.0, "Cost should be positive");
    }
}
