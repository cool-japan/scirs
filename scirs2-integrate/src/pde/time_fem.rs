//! Time-dependent Finite Element Methods
//!
//! This module provides finite element time integration for parabolic
//! (heat equation) and hyperbolic (wave equation) problems in 1-D.
//!
//! ## Heat Equation (θ-method)
//!
//! The semi-discrete heat equation on a uniform mesh is:
//!
//! ```text
//! M ü + K u = f(t)
//! ```
//!
//! where M is the consistent (or lumped) mass matrix and K is the stiffness
//! matrix.  The **θ-method** advances from time level n to n+1:
//!
//! ```text
//! (M + θ·Δt·K) u^{n+1} = (M − (1−θ)·Δt·K) u^n + Δt·[θ f^{n+1} + (1−θ) f^n]
//! ```
//!
//! * θ = 0   → explicit forward Euler (conditionally stable)
//! * θ = 1/2 → Crank-Nicolson (2nd-order, unconditionally stable)
//! * θ = 1   → implicit backward Euler (1st-order, unconditionally stable)
//!
//! ## Wave Equation (Newmark β-method)
//!
//! The second-order system  M ü + K u = f  is integrated by the Newmark
//! family:
//!
//! ```text
//! u^{n+1}  = u^n + Δt v^n + Δt²[(1/2−β) a^n + β a^{n+1}]
//! v^{n+1}  = v^n + Δt[(1−γ) a^n + γ a^{n+1}]
//! M a^{n+1} = f^{n+1} − K u^{n+1}
//! ```
//!
//! The average acceleration method (β = 1/4, γ = 1/2) is unconditionally
//! stable and second-order accurate.
//!
//! ## Adaptive Time-Stepping
//!
//! A simple embedded-method step-size controller estimates the local
//! truncation error by comparing the θ = 1/2 (Crank-Nicolson) and θ = 1
//! (backward Euler) solutions.

use scirs2_core::ndarray::{Array1, Array2};

use crate::pde::{PDEError, PDEResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the θ-method time integration of the FEM heat equation.
#[derive(Debug, Clone)]
pub struct TimeFemConfig {
    /// Theta parameter controlling the time integration scheme:
    /// * 0.0 → explicit forward Euler
    /// * 0.5 → Crank-Nicolson (recommended default)
    /// * 1.0 → implicit backward Euler
    pub theta: f64,

    /// Whether to use the lumped (diagonal) mass matrix instead of the
    /// consistent (tridiagonal) one.  Lumped mass is less accurate but
    /// permits a diagonal mass inversion without a linear solve in the
    /// explicit (θ = 0) case.
    pub lumped_mass: bool,

    /// Tolerance for adaptive time-stepping (`None` = fixed step).
    pub adaptive_tol: Option<f64>,
}

impl Default for TimeFemConfig {
    fn default() -> Self {
        TimeFemConfig {
            theta: 0.5,
            lumped_mass: false,
            adaptive_tol: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Mass matrix assembly
// ---------------------------------------------------------------------------

/// Assemble the **consistent** FEM mass matrix for a uniform 1-D mesh with
/// `n_elements` linear (P1) elements of width `dx`.
///
/// The result is the `(n_elements+1) × (n_elements+1)` tridiagonal matrix
/// with entries  M_{ii} = 2h/3,  M_{i,i±1} = h/6  (for an interior row),
/// where h = dx.  This matrix is symmetric positive definite.
///
/// If `lumped = true` the rows are summed into the diagonal, giving a
/// diagonal matrix with M_{ii} = h (interior) or h/2 (boundary nodes).
pub fn assemble_mass_matrix(n_elements: usize, dx: f64, lumped: bool) -> PDEResult<Array2<f64>> {
    if n_elements < 1 {
        return Err(PDEError::InvalidParameter(
            "Need at least 1 element".to_string(),
        ));
    }
    if dx <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "dx must be positive".to_string(),
        ));
    }

    let n = n_elements + 1;
    let mut m = Array2::<f64>::zeros((n, n));

    if lumped {
        // Row-sum lumping
        m[[0, 0]] = dx * 0.5;
        for i in 1..(n - 1) {
            m[[i, i]] = dx;
        }
        m[[n - 1, n - 1]] = dx * 0.5;
    } else {
        // Consistent mass: assemble element contributions
        // Element mass: M_e = (dx/6) * [[2, 1], [1, 2]]
        for e in 0..n_elements {
            m[[e, e]] += dx / 3.0;
            m[[e, e + 1]] += dx / 6.0;
            m[[e + 1, e]] += dx / 6.0;
            m[[e + 1, e + 1]] += dx / 3.0;
        }
    }

    Ok(m)
}

/// Assemble the FEM **stiffness matrix** for the 1-D diffusion operator
/// −d²u/dx² on a uniform mesh with `n_elements` linear elements of width `dx`.
///
/// The element stiffness is  K_e = (1/dx) * [[1, -1], [-1, 1]].
/// The global matrix has diagonal `2/dx` (interior) and `1/dx` (boundary).
pub fn assemble_stiffness_matrix(n_elements: usize, dx: f64) -> PDEResult<Array2<f64>> {
    if n_elements < 1 {
        return Err(PDEError::InvalidParameter(
            "Need at least 1 element".to_string(),
        ));
    }
    if dx <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "dx must be positive".to_string(),
        ));
    }

    let n = n_elements + 1;
    let mut k = Array2::<f64>::zeros((n, n));

    for e in 0..n_elements {
        let inv_dx = 1.0 / dx;
        k[[e, e]] += inv_dx;
        k[[e, e + 1]] -= inv_dx;
        k[[e + 1, e]] -= inv_dx;
        k[[e + 1, e + 1]] += inv_dx;
    }

    Ok(k)
}

// ---------------------------------------------------------------------------
// θ-method: single step
// ---------------------------------------------------------------------------

/// Advance the heat equation one time step using the θ-method.
///
/// Solves:
/// ```text
/// (M + θ·Δt·K) u^{n+1} = (M − (1−θ)·Δt·K) u^n
///                        + Δt · [θ f^{n+1} + (1−θ) f^n]
/// ```
///
/// Dirichlet boundary conditions are enforced by the penalty method:
/// constrained degrees of freedom are fixed to `bc_values[0]` (left) and
/// `bc_values[1]` (right).
///
/// # Arguments
/// * `u_n`   – solution at time level n, length `n_nodes`
/// * `k_mat` – stiffness matrix, `n_nodes × n_nodes`
/// * `m_mat` – mass matrix, `n_nodes × n_nodes`
/// * `f_n`   – load vector at time n
/// * `f_np1` – load vector at time n+1
/// * `dt`    – time step
/// * `theta` – θ ∈ [0, 1]
/// * `bc`    – optional `(left_value, right_value)` Dirichlet conditions
pub fn time_step_theta(
    u_n: &Array1<f64>,
    k_mat: &Array2<f64>,
    m_mat: &Array2<f64>,
    f_n: &Array1<f64>,
    f_np1: &Array1<f64>,
    dt: f64,
    theta: f64,
    bc: Option<(f64, f64)>,
) -> PDEResult<Array1<f64>> {
    let n = u_n.len();
    if k_mat.nrows() != n
        || k_mat.ncols() != n
        || m_mat.nrows() != n
        || m_mat.ncols() != n
        || f_n.len() != n
        || f_np1.len() != n
    {
        return Err(PDEError::ComputationError(
            "time_step_theta: inconsistent dimensions".to_string(),
        ));
    }
    if dt <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "dt must be positive".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&theta) {
        return Err(PDEError::InvalidParameter(
            "theta must be in [0, 1]".to_string(),
        ));
    }

    // Build LHS: A = M + theta * dt * K
    let mut a_mat = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_mat[[i, j]] = m_mat[[i, j]] + theta * dt * k_mat[[i, j]];
        }
    }

    // Build RHS: b = (M - (1-theta)*dt*K) u_n + dt*(theta*f_{n+1} + (1-theta)*f_n)
    let mut rhs = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut row_val = 0.0_f64;
        for j in 0..n {
            row_val += (m_mat[[i, j]] - (1.0 - theta) * dt * k_mat[[i, j]]) * u_n[j];
        }
        rhs[i] = row_val + dt * (theta * f_np1[i] + (1.0 - theta) * f_n[i]);
    }

    // Apply Dirichlet BCs via penalty method
    if let Some((left_val, right_val)) = bc {
        apply_dirichlet_penalty(&mut a_mat, &mut rhs, 0, left_val);
        apply_dirichlet_penalty(&mut a_mat, &mut rhs, n - 1, right_val);
    }

    // Solve the linear system
    let u_np1 = gauss_solve(&a_mat, &rhs)?;
    Ok(u_np1)
}

// ---------------------------------------------------------------------------
// Heat equation solver
// ---------------------------------------------------------------------------

/// Solve the 1-D heat equation:
/// ```text
///   ∂u/∂t = α ∂²u/∂x²,   x ∈ [0, L],   t ∈ [0, T]
/// ```
/// using the FEM θ-method on a uniform mesh with `n_elements` elements.
///
/// # Arguments
/// * `initial_cond` – function u(x) at t = 0
/// * `alpha`        – thermal diffusivity coefficient
/// * `x_left`, `x_right` – domain endpoints
/// * `bc`           – Dirichlet `(left_value, right_value)` at x_left and x_right
/// * `dt`           – time step size
/// * `n_steps`      – number of time steps
/// * `theta`        – θ-method parameter (0.5 = Crank-Nicolson recommended)
///
/// Returns a `Vec` of snapshots (one `Array1<f64>` per time step, including
/// t = 0).
pub fn solve_heat_equation_fem(
    initial_cond: &dyn Fn(f64) -> f64,
    alpha: f64,
    x_left: f64,
    x_right: f64,
    n_elements: usize,
    bc: (f64, f64),
    dt: f64,
    n_steps: usize,
    theta: f64,
) -> PDEResult<Vec<Array1<f64>>> {
    if n_elements < 1 {
        return Err(PDEError::InvalidParameter(
            "n_elements must be >= 1".to_string(),
        ));
    }
    if x_right <= x_left {
        return Err(PDEError::DomainError(
            "x_right must be > x_left".to_string(),
        ));
    }
    if alpha <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "alpha must be positive".to_string(),
        ));
    }
    if dt <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "dt must be positive".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&theta) {
        return Err(PDEError::InvalidParameter(
            "theta must be in [0, 1]".to_string(),
        ));
    }

    let n = n_elements + 1;
    let dx = (x_right - x_left) / n_elements as f64;

    // Build initial condition
    let mut u = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = x_left + i as f64 * dx;
        u[i] = initial_cond(xi);
    }
    // Enforce BCs on initial condition
    u[0] = bc.0;
    u[n - 1] = bc.1;

    // Assemble (time-independent) matrices
    let m_mat = assemble_mass_matrix(n_elements, dx, false)?;
    // Stiffness includes diffusivity α
    let k_raw = assemble_stiffness_matrix(n_elements, dx)?;
    let mut k_mat = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            k_mat[[i, j]] = alpha * k_raw[[i, j]];
        }
    }

    // Zero source term
    let f_zero = Array1::<f64>::zeros(n);

    let mut history = Vec::with_capacity(n_steps + 1);
    history.push(u.clone());

    for _ in 0..n_steps {
        let u_new = time_step_theta(&u, &k_mat, &m_mat, &f_zero, &f_zero, dt, theta, Some(bc))?;
        u = u_new;
        history.push(u.clone());
    }

    Ok(history)
}

// ---------------------------------------------------------------------------
// Wave equation (Newmark method)
// ---------------------------------------------------------------------------

/// Solve the 1-D wave equation:
/// ```text
///   ∂²u/∂t² = c² ∂²u/∂x²,   x ∈ [0, L],   t ∈ [0, T]
/// ```
/// using the Newmark average-acceleration method (β = 1/4, γ = 1/2).
///
/// # Arguments
/// * `u0`    – initial displacement u(x, 0)
/// * `v0`    – initial velocity ∂u/∂t(x, 0)
/// * `c_wave`– wave speed
/// * `x_left`, `x_right` – domain endpoints
/// * `n_elements` – number of finite elements
/// * `bc`    – Dirichlet `(left_value, right_value)` for displacement
/// * `dt`    – time step
/// * `n_steps` – number of time steps
///
/// Returns displacement snapshots as `Vec<Array1<f64>>`.
pub fn solve_wave_equation_fem(
    u0: &dyn Fn(f64) -> f64,
    v0: &dyn Fn(f64) -> f64,
    c_wave: f64,
    x_left: f64,
    x_right: f64,
    n_elements: usize,
    bc: (f64, f64),
    dt: f64,
    n_steps: usize,
) -> PDEResult<Vec<Array1<f64>>> {
    if n_elements < 1 {
        return Err(PDEError::InvalidParameter(
            "n_elements must be >= 1".to_string(),
        ));
    }
    if x_right <= x_left {
        return Err(PDEError::DomainError(
            "x_right must be > x_left".to_string(),
        ));
    }
    if c_wave <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "c_wave must be positive".to_string(),
        ));
    }
    if dt <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "dt must be positive".to_string(),
        ));
    }

    let n = n_elements + 1;
    let dx = (x_right - x_left) / n_elements as f64;

    // Initial conditions
    let mut u = Array1::<f64>::zeros(n);
    let mut v = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = x_left + i as f64 * dx;
        u[i] = u0(xi);
        v[i] = v0(xi);
    }
    u[0] = bc.0;
    u[n - 1] = bc.1;
    v[0] = 0.0;
    v[n - 1] = 0.0;

    // Assemble matrices
    let m_mat = assemble_mass_matrix(n_elements, dx, false)?;
    let k_raw = assemble_stiffness_matrix(n_elements, dx)?;
    let mut k_mat = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            k_mat[[i, j]] = c_wave * c_wave * k_raw[[i, j]];
        }
    }

    // Compute initial acceleration from  M a^0 = f^0 - K u^0
    let f_zero = Array1::<f64>::zeros(n);
    let rhs0 = compute_newmark_rhs(&f_zero, &k_mat, &u);
    let mut a = newmark_accel_solve(&m_mat, &rhs0, bc)?;

    // Newmark parameters: average acceleration (unconditionally stable)
    let beta = 0.25_f64;
    let gamma = 0.5_f64;

    // Effective stiffness: K_eff = M + beta * dt^2 * K
    let k_eff = build_effective_stiffness(&m_mat, &k_mat, beta, dt);

    let mut history = Vec::with_capacity(n_steps + 1);
    history.push(u.clone());

    for _ in 0..n_steps {
        // Predictors
        let u_pred = newmark_predict_u(&u, &v, &a, beta, dt);
        let v_pred = newmark_predict_v(&v, &a, gamma, dt);

        // Effective RHS: M a^{n+1} = f^{n+1} - K u_pred
        // Using f = 0
        let rhs = compute_newmark_rhs(&f_zero, &k_mat, &u_pred);
        let a_new = newmark_accel_solve_eff(&k_eff, &rhs, bc)?;

        // Correctors
        let mut u_new = Array1::<f64>::zeros(n);
        let mut v_new = Array1::<f64>::zeros(n);
        for i in 0..n {
            u_new[i] = u_pred[i] + beta * dt * dt * a_new[i];
            v_new[i] = v_pred[i] + gamma * dt * a_new[i];
        }

        // Enforce BCs
        u_new[0] = bc.0;
        u_new[n - 1] = bc.1;
        v_new[0] = 0.0;
        v_new[n - 1] = 0.0;

        u = u_new;
        v = v_new;
        a = a_new;

        history.push(u.clone());
    }

    Ok(history)
}

// ---------------------------------------------------------------------------
// Adaptive time-stepping
// ---------------------------------------------------------------------------

/// Estimate an appropriate time-step size for the heat equation using a
/// simple embedded error estimator.
///
/// The idea: advance one step with θ = 0.5 (Crank-Nicolson, 2nd-order) and
/// one step with θ = 1.0 (backward Euler, 1st-order), then compare.  The
/// ratio of the estimated error to the tolerance gives a scaled step-size
/// adjustment via the step-doubling / PI controller:
///
/// ```text
/// Δt_new = Δt_old × min(4, max(0.1, 0.9 × (tol / err)^(1/2)))
/// ```
///
/// # Arguments
/// * `u`     – current solution
/// * `k_mat` – stiffness matrix
/// * `m_mat` – mass matrix
/// * `dt`    – current time step
/// * `tol`   – desired local truncation error tolerance (per-unit step)
///
/// Returns the recommended next time step.
pub fn adaptive_time_stepping(
    u: &Array1<f64>,
    k_mat: &Array2<f64>,
    m_mat: &Array2<f64>,
    dt: f64,
    tol: f64,
) -> PDEResult<f64> {
    if dt <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "dt must be positive".to_string(),
        ));
    }
    if tol <= 0.0 {
        return Err(PDEError::InvalidParameter(
            "tol must be positive".to_string(),
        ));
    }

    let n = u.len();
    let f_zero = Array1::<f64>::zeros(n);

    // High-order step: Crank-Nicolson (θ = 0.5)
    let u_cn = time_step_theta(u, k_mat, m_mat, &f_zero, &f_zero, dt, 0.5, None)?;
    // Low-order step: backward Euler (θ = 1.0)
    let u_be = time_step_theta(u, k_mat, m_mat, &f_zero, &f_zero, dt, 1.0, None)?;

    // Error estimate: ||u_CN - u_BE||_inf
    let mut err = 0.0_f64;
    let mut norm_ref = 0.0_f64;
    for i in 0..n {
        err = err.max((u_cn[i] - u_be[i]).abs());
        norm_ref = norm_ref.max(u_cn[i].abs().max(1.0));
    }
    let rel_err = err / norm_ref;

    if rel_err < f64::EPSILON {
        // Essentially zero error: double the step but cap growth
        return Ok(dt * 4.0);
    }

    // PI controller
    let factor = 0.9 * (tol / rel_err).sqrt();
    let factor_clamped = factor.clamp(0.1, 4.0);
    Ok(dt * factor_clamped)
}

// ---------------------------------------------------------------------------
// Newmark helpers (internal)
// ---------------------------------------------------------------------------

fn compute_newmark_rhs(f: &Array1<f64>, k: &Array2<f64>, u: &Array1<f64>) -> Array1<f64> {
    let n = u.len();
    let mut rhs = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut ku = 0.0_f64;
        for j in 0..n {
            ku += k[[i, j]] * u[j];
        }
        rhs[i] = f[i] - ku;
    }
    rhs
}

fn newmark_predict_u(
    u: &Array1<f64>,
    v: &Array1<f64>,
    a: &Array1<f64>,
    beta: f64,
    dt: f64,
) -> Array1<f64> {
    let n = u.len();
    let mut u_pred = Array1::<f64>::zeros(n);
    for i in 0..n {
        u_pred[i] = u[i] + dt * v[i] + dt * dt * (0.5 - beta) * a[i];
    }
    u_pred
}

fn newmark_predict_v(v: &Array1<f64>, a: &Array1<f64>, gamma: f64, dt: f64) -> Array1<f64> {
    let n = v.len();
    let mut v_pred = Array1::<f64>::zeros(n);
    for i in 0..n {
        v_pred[i] = v[i] + dt * (1.0 - gamma) * a[i];
    }
    v_pred
}

fn build_effective_stiffness(m: &Array2<f64>, k: &Array2<f64>, beta: f64, dt: f64) -> Array2<f64> {
    let n = m.nrows();
    let mut k_eff = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            k_eff[[i, j]] = m[[i, j]] + beta * dt * dt * k[[i, j]];
        }
    }
    k_eff
}

/// Solve M a = rhs for initial acceleration with Dirichlet BC zeroing.
fn newmark_accel_solve(
    m: &Array2<f64>,
    rhs: &Array1<f64>,
    bc: (f64, f64),
) -> PDEResult<Array1<f64>> {
    let n = rhs.len();
    let mut m_bc = m.clone();
    let mut r_bc = rhs.clone();
    apply_dirichlet_penalty(&mut m_bc, &mut r_bc, 0, 0.0);
    apply_dirichlet_penalty(&mut m_bc, &mut r_bc, n - 1, 0.0);
    let _ = bc; // bc values for acceleration are 0 (zero acceleration at fixed nodes)
    gauss_solve(&m_bc, &r_bc)
}

/// Solve K_eff a = rhs for Newmark acceleration corrector.
fn newmark_accel_solve_eff(
    k_eff: &Array2<f64>,
    rhs: &Array1<f64>,
    bc: (f64, f64),
) -> PDEResult<Array1<f64>> {
    let n = rhs.len();
    let mut k_bc = k_eff.clone();
    let mut r_bc = rhs.clone();
    apply_dirichlet_penalty(&mut k_bc, &mut r_bc, 0, 0.0);
    apply_dirichlet_penalty(&mut k_bc, &mut r_bc, n - 1, 0.0);
    let _ = bc;
    gauss_solve(&k_bc, &r_bc)
}

// ---------------------------------------------------------------------------
// Dirichlet boundary condition via penalty method
// ---------------------------------------------------------------------------

/// Enforce a Dirichlet condition at degree of freedom `dof` with the given
/// value by the "penalty replacement" technique: set the entire row/column to
/// zero, place a large penalty P on the diagonal, and set rhs[dof] = P * value.
fn apply_dirichlet_penalty(a: &mut Array2<f64>, rhs: &mut Array1<f64>, dof: usize, value: f64) {
    let n = a.nrows();

    // "Lift" approach: move the known boundary value to the RHS for all
    // rows that couple to the boundary DOF, then zero the row and column.
    // This preserves the symmetry and accuracy of the interior equations.
    for i in 0..n {
        if i != dof {
            rhs[i] -= a[[i, dof]] * value;
            a[[i, dof]] = 0.0;
        }
    }

    // Zero the boundary row and set the equation u[dof] = value.
    for j in 0..n {
        a[[dof, j]] = 0.0;
    }
    a[[dof, dof]] = 1.0;
    rhs[dof] = value;
}

// ---------------------------------------------------------------------------
// Gaussian elimination (dense, partial pivoting)
// ---------------------------------------------------------------------------

/// Solve the `n × n` system A x = b by Gaussian elimination with partial
/// pivoting.  Returns an error if the matrix appears singular.
fn gauss_solve(a: &Array2<f64>, b: &Array1<f64>) -> PDEResult<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(PDEError::ComputationError(
            "gauss_solve: dimension mismatch".to_string(),
        ));
    }

    // Build augmented matrix [A | b]
    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        // Partial pivot
        let mut pivot_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                pivot_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(PDEError::ComputationError(
                "gauss_solve: matrix is singular".to_string(),
            ));
        }
        if pivot_row != col {
            for k in 0..=n {
                let tmp = aug[[col, k]];
                aug[[col, k]] = aug[[pivot_row, k]];
                aug[[pivot_row, k]] = tmp;
            }
        }
        // Eliminate below
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let fac = aug[[row, col]] / pivot;
            for k in col..=n {
                let delta = fac * aug[[col, k]];
                aug[[row, k]] -= delta;
            }
        }
    }

    // Back-substitution
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = aug[[i, n]];
        for j in (i + 1)..n {
            s -= aug[[i, j]] * x[j];
        }
        x[i] = s / aug[[i, i]];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_mass_matrix_consistent_shape() {
        let m = assemble_mass_matrix(4, 0.25, false).expect("mass matrix");
        assert_eq!(m.nrows(), 5);
        assert_eq!(m.ncols(), 5);
    }

    #[test]
    fn test_mass_matrix_row_sum_uniform_mesh() {
        // For a uniform mesh [0,1] with n elements, each row of M sums to h
        // (for interior rows) except boundary nodes (h/2 each).
        // Total sum = L = 1.0 (domain length).
        let n_elem = 10;
        let dx = 1.0 / n_elem as f64;
        let m = assemble_mass_matrix(n_elem, dx, false).expect("mass");
        let total: f64 = m.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "total mass = {total}");
    }

    #[test]
    fn test_mass_matrix_lumped() {
        let n_elem = 4;
        let dx = 0.25;
        let m = assemble_mass_matrix(n_elem, dx, true).expect("lumped mass");
        // All off-diagonal should be zero
        let n = n_elem + 1;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        m[[i, j]].abs() < 1e-15,
                        "off-diagonal [{i},{j}] = {}",
                        m[[i, j]]
                    );
                }
            }
        }
        // Total should also be 1.0
        let total: f64 = m.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "lumped total = {total}");
    }

    #[test]
    fn test_stiffness_matrix_shape() {
        let k = assemble_stiffness_matrix(5, 0.2).expect("stiffness");
        assert_eq!(k.nrows(), 6);
        assert_eq!(k.ncols(), 6);
    }

    #[test]
    fn test_stiffness_matrix_null_space() {
        // K applied to constant vector should give zero (constants are in null space)
        let n_elem = 5;
        let dx = 1.0 / n_elem as f64;
        let k = assemble_stiffness_matrix(n_elem, dx).expect("stiffness");
        let n = n_elem + 1;
        let ones = Array1::<f64>::ones(n);
        let mut ku = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                ku[i] += k[[i, j]] * ones[j];
            }
        }
        for i in 0..n {
            assert!(
                ku[i].abs() < 1e-12,
                "K * 1 has non-zero entry [{i}] = {}",
                ku[i]
            );
        }
    }

    #[test]
    fn test_time_step_theta_steady_state() {
        // If u^n is already a steady state of the heat equation with matching BCs,
        // then u^{n+1} should equal u^n.
        // K u = 0 for interior nodes when u is linear; boundary nodes are pinned by
        // Dirichlet BCs via lift (row/column elimination).
        // u(x) = x on [0,1] with u(0) = 0, u(1) = 1.
        let n_elem = 4;
        let dx = 0.25;
        let n = n_elem + 1;
        let m = assemble_mass_matrix(n_elem, dx, false).expect("M");
        let k = assemble_stiffness_matrix(n_elem, dx).expect("K");

        // Linear solution u(x) = x; boundary values u(0) = 0, u(1) = 1
        let u_n = Array1::from_shape_fn(n, |i| i as f64 * dx);
        let f_zero = Array1::<f64>::zeros(n);
        let u_new = time_step_theta(&u_n, &k, &m, &f_zero, &f_zero, 0.01, 0.5, Some((0.0, 1.0)))
            .expect("theta step");

        for i in 0..n {
            assert!(
                (u_new[i] - u_n[i]).abs() < 1e-10,
                "u[{i}] changed: {} -> {}",
                u_n[i],
                u_new[i]
            );
        }
    }

    #[test]
    fn test_heat_equation_decay() {
        // u(x,0) = sin(πx) with zero Dirichlet BCs decays as e^{-π²t}
        let n_elem = 20;
        let alpha = 1.0;
        let dt = 0.001;
        let n_steps = 100;
        let t_final = dt * n_steps as f64;

        let history = solve_heat_equation_fem(
            &|x| (PI * x).sin(),
            alpha,
            0.0,
            1.0,
            n_elem,
            (0.0, 0.0),
            dt,
            n_steps,
            0.5,
        )
        .expect("heat solve");

        assert_eq!(history.len(), n_steps + 1);

        // Check the amplitude at x ~ 0.5 (midpoint index)
        let mid = n_elem / 2;
        let u_final = history.last().expect("history non-empty")[mid];
        let expected =
            (PI * (mid as f64) / n_elem as f64).sin() * (-PI * PI * alpha * t_final).exp();
        let rel_err = (u_final - expected).abs() / (expected.abs() + 1e-12);
        assert!(
            rel_err < 0.02,
            "heat decay: got {u_final}, expected {expected}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_heat_equation_constant_bc() {
        // With constant BCs (u=0 at both ends) and zero IC the solution stays zero
        let history =
            solve_heat_equation_fem(&|_| 0.0, 1.0, 0.0, 1.0, 10, (0.0, 0.0), 0.01, 50, 0.5)
                .expect("heat solve zero");
        for snap in &history {
            for &v in snap.iter() {
                assert!(v.abs() < 1e-12, "non-zero value {v} in zero solution");
            }
        }
    }

    #[test]
    fn test_wave_equation_returns_snapshots() {
        let n_steps = 20;
        let history = solve_wave_equation_fem(
            &|x| (PI * x).sin(),
            &|_| 0.0,
            1.0,
            0.0,
            1.0,
            10,
            (0.0, 0.0),
            0.01,
            n_steps,
        )
        .expect("wave solve");
        assert_eq!(history.len(), n_steps + 1);
    }

    #[test]
    fn test_wave_equation_zero_ic() {
        // Zero displacement and velocity → stays zero
        let n_steps = 10;
        let history = solve_wave_equation_fem(
            &|_| 0.0,
            &|_| 0.0,
            1.0,
            0.0,
            1.0,
            8,
            (0.0, 0.0),
            0.01,
            n_steps,
        )
        .expect("wave solve zero");
        for snap in &history {
            for &v in snap.iter() {
                assert!(v.abs() < 1e-10, "non-zero value {v} in zero wave solution");
            }
        }
    }

    #[test]
    fn test_adaptive_time_stepping_increases_step() {
        // For a very smooth, nearly steady solution the error should be tiny
        // and the suggested step should be larger than the current one.
        let n_elem = 10;
        let dx = 0.1;
        let m = assemble_mass_matrix(n_elem, dx, false).expect("M");
        let k = assemble_stiffness_matrix(n_elem, dx).expect("K");

        // Near-zero solution → tiny error
        let u = Array1::<f64>::zeros(n_elem + 1);
        let dt = 0.01;
        let tol = 1e-3;

        let new_dt = adaptive_time_stepping(&u, &k, &m, dt, tol).expect("adaptive");
        // With zero u the error is zero, so step should grow
        assert!(new_dt >= dt, "new_dt {new_dt} < old dt {dt}");
    }

    #[test]
    fn test_adaptive_time_stepping_reduces_step() {
        // For a rapidly varying solution with a strict tolerance, the step
        // should decrease.
        let n_elem = 10;
        let dx = 0.1;
        let m = assemble_mass_matrix(n_elem, dx, false).expect("M");
        // Make K large (α = 1000) to force a large step error
        let k_raw = assemble_stiffness_matrix(n_elem, dx).expect("K");
        let n = n_elem + 1;
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                k[[i, j]] = 1000.0 * k_raw[[i, j]];
            }
        }

        let u = Array1::from_shape_fn(n, |i| (PI * i as f64 * dx).sin());
        let dt = 0.1;
        let tol = 1e-6;

        let new_dt = adaptive_time_stepping(&u, &k, &m, dt, tol).expect("adaptive");
        assert!(
            new_dt < dt,
            "step should decrease for stiff problem: new_dt={new_dt}, dt={dt}"
        );
    }

    #[test]
    fn test_time_fem_config_default() {
        let cfg = TimeFemConfig::default();
        assert!((cfg.theta - 0.5).abs() < 1e-12);
        assert!(!cfg.lumped_mass);
        assert!(cfg.adaptive_tol.is_none());
    }

    #[test]
    fn test_gauss_solve() {
        // Simple 3×3 system
        let mut a = Array2::<f64>::zeros((3, 3));
        a[[0, 0]] = 1.0;
        a[[0, 1]] = 2.0;
        a[[0, 2]] = 0.0;
        a[[1, 0]] = 3.0;
        a[[1, 1]] = 4.0;
        a[[1, 2]] = 5.0;
        a[[2, 0]] = 0.0;
        a[[2, 1]] = 6.0;
        a[[2, 2]] = 7.0;

        let b = Array1::from_vec(vec![5.0, 26.0, 45.0]);
        let x = gauss_solve(&a, &b).expect("solve");

        // Verify A x = b
        for i in 0..3 {
            let mut ax_i = 0.0_f64;
            for j in 0..3 {
                ax_i += a[[i, j]] * x[j];
            }
            assert!(
                (ax_i - b[i]).abs() < 1e-9,
                "residual[{i}] = {}",
                ax_i - b[i]
            );
        }
    }
}
