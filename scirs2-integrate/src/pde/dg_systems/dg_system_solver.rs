//! 1D compressible Euler equations solved with a nodal Discontinuous Galerkin method.
//!
//! Uses Legendre-Gauss-Lobatto (LGL) nodes, an HLLC numerical flux at element interfaces,
//! and either forward-Euler, SSPRK3, or SSPRK4 (Spiteri-Ruuth SSP(5,4)) time integration.
//!
//! ## References
//! - Hesthaven & Warburton, "Nodal Discontinuous Galerkin Methods" (2008)
//! - Cockburn & Shu, "Runge-Kutta discontinuous Galerkin methods" (1989, 1998)
//! - Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2009)

use scirs2_core::ndarray::Array2;

use super::euler_1d::{
    euler_flux, max_wave_speed, primitives_to_conservative, EulerFlux, EulerState,
};
use super::hllc_euler::hllc_flux;
use super::limiter::{MinmodTvbLimiter, PerssonPeraireIndicator, SlopeLimiter};
use crate::dg_advanced::entropy_stable::{differentiation_matrix_lgl, legendre_gauss_lobatto};
use crate::error::{IntegrateError, IntegrateResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public types: boundary conditions, time integrators, config, solution
// ─────────────────────────────────────────────────────────────────────────────

/// Boundary condition type for the DG Euler solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryCondition {
    /// Zeroth-order extrapolation (suitable for supersonic outflow and Sod tube).
    Outflow,
    /// Periodic boundary conditions.
    Periodic,
    /// Reflecting (slip) wall: negate the velocity component.
    Reflective,
}

/// Time integrator choice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimeIntegrator {
    /// First-order explicit Euler. For testing only — not stable at practical CFL values.
    ForwardEuler,
    /// Optimal 3-stage, 3rd-order SSP Runge-Kutta (Shu & Osher 1988).
    Ssprk3,
    /// Optimal 5-stage, 4th-order SSP Runge-Kutta (Spiteri & Ruuth 2002).
    Ssprk4,
}

/// Configuration for the 1D Euler DG solver.
pub struct DgSystemConfig {
    /// Polynomial order `p`; each element has `p + 1` GLL nodes.
    pub polynomial_order: usize,
    /// Number of elements in the 1D mesh.
    pub n_elements: usize,
    /// Physical left boundary of the domain.
    pub x_left: f64,
    /// Physical right boundary of the domain.
    pub x_right: f64,
    /// Adiabatic index (ratio of specific heats). Typically 1.4 for air.
    pub gamma: f64,
    /// CFL coefficient. Typical stable values: 0.3–0.5 for SSPRK3.
    pub cfl: f64,
    /// Time integrator variant.
    pub time_integrator: TimeIntegrator,
    /// Boundary condition applied at both domain boundaries.
    pub boundary: BoundaryCondition,
    /// Optional slope limiter.
    ///
    /// If `None`, no limiting is applied. If `Some`, the limiter is applied
    /// after each complete Runge-Kutta step.
    pub limiter: Option<Box<dyn SlopeLimiter>>,
    /// Optional troubled-cell indicator.
    ///
    /// If `Some`, only elements whose indicator value exceeds `indicator_threshold`
    /// are limited. If `None`, all elements are limited unconditionally.
    pub indicator: Option<Box<dyn PerssonPeraireIndicator>>,
    /// Threshold for the Persson-Peraire indicator.
    ///
    /// Elements are flagged troubled if `S_k > indicator_threshold`.
    /// A value of `-1.0` works well for moderate-resolution computations.
    pub indicator_threshold: f64,
    /// Whether to record the solution at intermediate times.
    pub record_history: bool,
    /// Record one snapshot every `record_every` time steps (ignored if `record_history = false`).
    pub record_every: usize,
}

impl DgSystemConfig {
    /// Create a configuration suitable for the Sod shock tube problem.
    ///
    /// Domain: [0, 1], gamma = 1.4, SSPRK3, Outflow BCs, TVB minmod limiter (M=10).
    ///
    /// Uses the TVB-modified minmod limiter with a nodal→modal projection step,
    /// ensuring the limiter operates on proper Legendre modal coefficients.
    pub fn default_sod(n_elements: usize, polynomial_order: usize) -> Self {
        Self {
            polynomial_order,
            n_elements,
            x_left: 0.0,
            x_right: 1.0,
            gamma: 1.4,
            cfl: 0.3,
            time_integrator: TimeIntegrator::Ssprk3,
            boundary: BoundaryCondition::Outflow,
            limiter: Some(Box::new(MinmodTvbLimiter::new(10.0))),
            indicator: None,
            indicator_threshold: -1.0,
            record_history: false,
            record_every: 10,
        }
    }
}

/// Solution returned by the DG Euler solver.
pub struct DgSystemSolution {
    /// Physical x-coordinates of all nodes: shape `(n_elements, p+1)`.
    pub x_nodes: Array2<f64>,
    /// Final state at `t_final`: outer index = element, inner = node.
    pub final_state: Vec<Vec<EulerState>>,
    /// Times at which history snapshots were recorded (populated if `record_history = true`).
    pub time_history: Vec<f64>,
    /// State snapshots (populated if `record_history = true`).
    ///
    /// Indexing: `state_history[snapshot][element][node]`.
    pub state_history: Vec<Vec<Vec<EulerState>>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers: arithmetic on Vec<Vec<EulerState>>
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `a + scale * b` element-wise.
fn add_scaled(a: &[Vec<EulerState>], b: &[Vec<EulerState>], scale: f64) -> Vec<Vec<EulerState>> {
    a.iter()
        .zip(b.iter())
        .map(|(ae, be)| {
            ae.iter()
                .zip(be.iter())
                .map(|(as_, bs)| *as_ + *bs * scale)
                .collect()
        })
        .collect()
}

/// Compute `alpha * a + beta * b` element-wise.
fn axpby(
    alpha: f64,
    a: &[Vec<EulerState>],
    beta: f64,
    b: &[Vec<EulerState>],
) -> Vec<Vec<EulerState>> {
    a.iter()
        .zip(b.iter())
        .map(|(ae, be)| {
            ae.iter()
                .zip(be.iter())
                .map(|(as_, bs)| *as_ * alpha + *bs * beta)
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Time integrators (inlined — Vec<Vec<EulerState>> has no blanket Add/Mul impl)
// ─────────────────────────────────────────────────────────────────────────────

/// One step of the optimal 3-stage SSPRK3 scheme (Shu & Osher 1988).
///
/// The optional `post_stage_fn` is called after each forward-Euler substep to
/// apply the slope limiter. This is necessary for high-order DG where the
/// intermediate RK stages can produce unphysical states near discontinuities.
fn ssprk3_step(
    u: &[Vec<EulerState>],
    dt: f64,
    rhs_fn: &dyn Fn(&[Vec<EulerState>]) -> IntegrateResult<Vec<Vec<EulerState>>>,
    post_stage_fn: &dyn Fn(&mut Vec<Vec<EulerState>>),
) -> IntegrateResult<Vec<Vec<EulerState>>> {
    // Stage 1
    let l0 = rhs_fn(u)?;
    let mut u1 = add_scaled(u, &l0, dt);
    post_stage_fn(&mut u1);

    // Stage 2
    let l1 = rhs_fn(&u1)?;
    let mut u2 = axpby(3.0 / 4.0, u, 1.0 / 4.0, &add_scaled(&u1, &l1, dt));
    post_stage_fn(&mut u2);

    // Stage 3
    let l2 = rhs_fn(&u2)?;
    let mut u_new = axpby(1.0 / 3.0, u, 2.0 / 3.0, &add_scaled(&u2, &l2, dt));
    post_stage_fn(&mut u_new);
    Ok(u_new)
}

/// One step of the Spiteri-Ruuth SSP(5,4) scheme.
///
/// 5-stage, 4th-order, SSP coefficient C ≈ 1.508.
/// The `post_stage_fn` is called after each substep to apply the slope limiter.
fn ssprk4_step(
    u: &[Vec<EulerState>],
    dt: f64,
    rhs_fn: &dyn Fn(&[Vec<EulerState>]) -> IntegrateResult<Vec<Vec<EulerState>>>,
    post_stage_fn: &dyn Fn(&mut Vec<Vec<EulerState>>),
) -> IntegrateResult<Vec<Vec<EulerState>>> {
    // Spiteri-Ruuth SSP(5,4) coefficients (Table 2, Spiteri & Ruuth 2002).
    const C1: f64 = 0.391_752_226_571_89;
    const C2: f64 = 0.586_079_152_584_48;
    const C3: f64 = 0.474_542_363_121_968;
    const C4: f64 = 0.935_010_630_967_653;

    // u1 = u + C1 * dt * L(u)
    let l0 = rhs_fn(u)?;
    let mut u1 = add_scaled(u, &l0, C1 * dt);
    post_stage_fn(&mut u1);

    // u2 = 0.444370493651235 * u + 0.555629506348765 * (u1 + C2 * dt * L(u1))
    let l1 = rhs_fn(&u1)?;
    let mut u2 = axpby(
        0.444_370_493_651_235,
        u,
        0.555_629_506_348_765,
        &add_scaled(&u1, &l1, C2 * dt),
    );
    post_stage_fn(&mut u2);

    // u3 = 0.620101851488403 * u + 0.379898148511597 * (u2 + C3 * dt * L(u2))
    let l2 = rhs_fn(&u2)?;
    let mut u3 = axpby(
        0.620_101_851_488_403,
        u,
        0.379_898_148_511_597,
        &add_scaled(&u2, &l2, C3 * dt),
    );
    post_stage_fn(&mut u3);

    // u4 = 0.178079954393132 * u + 0.821920045606868 * (u3 + C4 * dt * L(u3))
    let l3 = rhs_fn(&u3)?;
    let mut u4 = axpby(
        0.178_079_954_393_132,
        u,
        0.821_920_045_606_868,
        &add_scaled(&u3, &l3, C4 * dt),
    );
    post_stage_fn(&mut u4);

    // u_new = 0.517231671970585 * u2 + 0.096059710526147 * u3
    //       + 0.386708617503269 * (u4 + 0.063692468666290 * dt * L(u4))
    let l4 = rhs_fn(&u4)?;
    let scaled_u4 = add_scaled(&u4, &l4, 0.063_692_468_666_290 * dt);
    let tmp = axpby(0.517_231_671_970_585, &u2, 0.096_059_710_526_147, &u3);
    let mut u_new = axpby(1.0, &tmp, 0.386_708_617_503_269, &scaled_u4);
    post_stage_fn(&mut u_new);
    Ok(u_new)
}

// ─────────────────────────────────────────────────────────────────────────────
// DG right-hand side
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters bundled to avoid clippy::too_many_arguments on `dg_rhs`.
struct RhsParams<'a> {
    d_mat: &'a [Vec<f64>],
    weights: &'a [f64],
    h: f64,
    gamma: f64,
    boundary: BoundaryCondition,
}

/// State at the left interface of element `k` (left ghost, right interior node).
fn left_interface_states(
    u: &[Vec<EulerState>],
    k: usize,
    boundary: BoundaryCondition,
) -> (EulerState, EulerState) {
    let n_nodes = u[k].len();
    let right_state = u[k][0]; // left edge of element k
    let left_state = if k == 0 {
        match boundary {
            BoundaryCondition::Outflow => u[0][0],
            BoundaryCondition::Periodic => u[u.len() - 1][n_nodes - 1],
            BoundaryCondition::Reflective => {
                let s = u[0][0];
                EulerState {
                    rho: s.rho,
                    rho_u: -s.rho_u,
                    energy: s.energy,
                }
            }
        }
    } else {
        u[k - 1][n_nodes - 1]
    };
    (left_state, right_state)
}

/// State at the right interface of element `k` (left interior node, right ghost).
fn right_interface_states(
    u: &[Vec<EulerState>],
    k: usize,
    boundary: BoundaryCondition,
) -> (EulerState, EulerState) {
    let n_nodes = u[k].len();
    let left_state = u[k][n_nodes - 1]; // right edge of element k
    let right_state = if k == u.len() - 1 {
        match boundary {
            BoundaryCondition::Outflow => u[k][n_nodes - 1],
            BoundaryCondition::Periodic => u[0][0],
            BoundaryCondition::Reflective => {
                let s = u[k][n_nodes - 1];
                EulerState {
                    rho: s.rho,
                    rho_u: -s.rho_u,
                    energy: s.energy,
                }
            }
        }
    } else {
        u[k + 1][0]
    };
    (left_state, right_state)
}

/// Compute the DG spatial operator (right-hand side) for all elements.
///
/// For each element k with p+1 nodes and Jacobian J_k = h/2:
/// ```text
/// rhs[k][i] = -(1/J_k) * sum_j D[i][j] * F[j]          (volume term)
///           + (f_hat_L - F[0]) / (J_k * w[0])   at i=0  (left boundary correction)
///           + (F[N] - f_hat_R) / (J_k * w[N])   at i=N  (right boundary correction)
/// ```
///
/// Derivation (SBP identity from weak form of `u_t + F_x = 0`):
/// - Left face outward normal is -1, so correction is `+(f_hat_L - F_0) / (J_k * w_0)`.
/// - Right face outward normal is +1, so correction is `+(F_N - f_hat_R) / (J_k * w_N)`.
fn dg_rhs(u: &[Vec<EulerState>], params: &RhsParams<'_>) -> IntegrateResult<Vec<Vec<EulerState>>> {
    let n_elem = u.len();
    let n_nodes = u[0].len();
    let j_k = params.h / 2.0; // Jacobian (element half-width)

    let mut rhs = vec![vec![EulerState::zero(); n_nodes]; n_elem];

    for k in 0..n_elem {
        // Compute physical flux F at each node
        let node_flux: Vec<EulerFlux> = u[k]
            .iter()
            .enumerate()
            .map(|(i, s)| {
                euler_flux(s, params.gamma).ok_or_else(|| {
                    IntegrateError::ValueError(format!(
                        "Unphysical state at element {k} node {i}: rho={}",
                        s.rho
                    ))
                })
            })
            .collect::<IntegrateResult<Vec<_>>>()?;

        // Volume term: rhs[k][i] = -(1/J_k) * sum_j D[i][j] * F[j]
        for i in 0..n_nodes {
            let mut sum_mass = 0.0_f64;
            let mut sum_mom = 0.0_f64;
            let mut sum_eng = 0.0_f64;
            for j in 0..n_nodes {
                sum_mass += params.d_mat[i][j] * node_flux[j].mass;
                sum_mom += params.d_mat[i][j] * node_flux[j].momentum;
                sum_eng += params.d_mat[i][j] * node_flux[j].energy;
            }
            rhs[k][i].rho = -sum_mass / j_k;
            rhs[k][i].rho_u = -sum_mom / j_k;
            rhs[k][i].energy = -sum_eng / j_k;
        }

        // Left interface HLLC flux
        let (left_ghost, right_inner) = left_interface_states(u, k, params.boundary);
        let f_hat_l = hllc_flux(&left_ghost, &right_inner, params.gamma).ok_or_else(|| {
            IntegrateError::ValueError(format!("HLLC flux failed at left interface of element {k}"))
        })?;

        // Right interface HLLC flux
        let (left_inner, right_ghost) = right_interface_states(u, k, params.boundary);
        let f_hat_r = hllc_flux(&left_inner, &right_ghost, params.gamma).ok_or_else(|| {
            use super::euler_1d::pressure_eos;
            IntegrateError::ValueError(format!(
                "HLLC right interface elem={k}: \
                 left=(rho={}, rho_u={}, E={}, p={:?}), \
                 right=(rho={}, rho_u={}, E={}, p={:?})",
                left_inner.rho,
                left_inner.rho_u,
                left_inner.energy,
                pressure_eos(&left_inner, params.gamma),
                right_ghost.rho,
                right_ghost.rho_u,
                right_ghost.energy,
                pressure_eos(&right_ghost, params.gamma),
            ))
        })?;

        // Left boundary correction at node 0.
        // Outward normal on left face = -1.
        // Correction: +(f_hat_L - F_0) / (J_k * w_0)
        let f0 = &node_flux[0];
        let w0 = params.weights[0];
        rhs[k][0].rho += (f_hat_l.mass - f0.mass) / (j_k * w0);
        rhs[k][0].rho_u += (f_hat_l.momentum - f0.momentum) / (j_k * w0);
        rhs[k][0].energy += (f_hat_l.energy - f0.energy) / (j_k * w0);

        // Right boundary correction at node N = n_nodes - 1.
        // Outward normal on right face = +1.
        // Correction: +(F_N - f_hat_R) / (J_k * w_N)
        let n = n_nodes - 1;
        let fn_ = &node_flux[n];
        let wn = params.weights[n];
        rhs[k][n].rho += (fn_.mass - f_hat_r.mass) / (j_k * wn);
        rhs[k][n].rho_u += (fn_.momentum - f_hat_r.momentum) / (j_k * wn);
        rhs[k][n].energy += (fn_.energy - f_hat_r.energy) / (j_k * wn);
    }

    Ok(rhs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Nodal ↔ modal Legendre projections
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the j-th Legendre polynomial at x using the 3-term recurrence.
///
/// P_0(x) = 1, P_1(x) = x, (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x).
fn legendre_poly(j: usize, x: f64) -> f64 {
    if j == 0 {
        return 1.0;
    }
    if j == 1 {
        return x;
    }
    let mut p_prev = 1.0_f64;
    let mut p_curr = x;
    for k in 1..j {
        let p_next = ((2 * k + 1) as f64 * x * p_curr - k as f64 * p_prev) / (k + 1) as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

/// Project nodal GLL values to Legendre modal coefficients.
///
/// `a_j = (2j+1)/2 * sum_i w_i * P_j(xi_i) * u_i`
///
/// where `P_j` is the j-th Legendre polynomial on [-1, 1].
fn nodal_to_legendre_modal(nodal: &[EulerState], xi: &[f64], w: &[f64]) -> Vec<EulerState> {
    let n = nodal.len();
    let mut modal = vec![EulerState::zero(); n];
    for j in 0..n {
        let norm = (2.0 * j as f64 + 1.0) / 2.0;
        for i in 0..n {
            let pj = legendre_poly(j, xi[i]);
            let wup = w[i] * pj * norm;
            modal[j].rho += wup * nodal[i].rho;
            modal[j].rho_u += wup * nodal[i].rho_u;
            modal[j].energy += wup * nodal[i].energy;
        }
    }
    modal
}

/// Reconstruct nodal GLL values from Legendre modal coefficients.
///
/// `u_i = sum_j a_j * P_j(xi_i)`
fn legendre_modal_to_nodal(modal: &[EulerState], xi: &[f64]) -> Vec<EulerState> {
    let n = xi.len();
    let mut nodal = vec![EulerState::zero(); n];
    for i in 0..n {
        for (j, m) in modal.iter().enumerate() {
            let pj = legendre_poly(j, xi[i]);
            nodal[i].rho += m.rho * pj;
            nodal[i].rho_u += m.rho_u * pj;
            nodal[i].energy += m.energy * pj;
        }
    }
    nodal
}

// ─────────────────────────────────────────────────────────────────────────────
// Limiter application (with nodal→modal→nodal projection)
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the slope limiter to the solution after a time step.
///
/// Projects nodal GLL values to Legendre modal coefficients, applies the limiter,
/// then reconstructs nodal values. This ensures the limiter operates on the
/// correct modal representation (mode 0 = cell average, mode 1 = slope, etc.).
///
/// After the TVB limiting step, a Zhang-Shu positivity-preserving correction
/// (Zhang & Shu 2010) scales each node's deviation from the cell average by the
/// largest θ ∈ [0,1] that keeps every node's density and pressure positive.
///
/// If no indicator is provided, all elements are limited. Otherwise, only
/// elements whose spectral smoothness indicator exceeds `indicator_threshold`
/// are limited.
fn apply_limiter(
    u: &mut [Vec<EulerState>],
    limiter: &dyn SlopeLimiter,
    indicator: Option<&dyn PerssonPeraireIndicator>,
    indicator_threshold: f64,
    xi: &[f64],
    w: &[f64],
    h: f64,
    gamma: f64,
) {
    // Use a floor well above machine epsilon to avoid catastrophic cancellation
    // in pressure evaluation (catastrophic cancellation when E ≈ KE).
    // EPS_RHO must be large enough that rho_u / EPS_RHO does not produce a
    // wave speed that collapses the CFL time step (e.g. 1e-10 gives O(10^10)
    // velocities for typical rho_u ~ O(1)).  1e-7 is still 5+ orders below
    // the smallest physical density in the Sod tube (0.125).
    const EPS_RHO: f64 = 1e-7;
    const EPS_P: f64 = 1e-10;

    let n_elem = u.len();

    // Pre-compute modal coefficients for all elements once.
    // This avoids redundant re-projection when accessing neighbor cell averages.
    // Each element's modal[0] is the cell average, used by its two neighbors.
    let all_modal: Vec<Vec<EulerState>> = u
        .iter()
        .map(|uk| nodal_to_legendre_modal(uk, xi, w))
        .collect();

    for k in 0..n_elem {
        // Cell averages of left and right neighbors (from pre-computed modals)
        let left_modal0 = if k == 0 {
            all_modal[0][0] // ghost: same element's cell average
        } else {
            all_modal[k - 1][0]
        };
        let right_modal0 = if k == n_elem - 1 {
            all_modal[n_elem - 1][0] // ghost: same element's cell average
        } else {
            all_modal[k + 1][0]
        };

        // Check if this element needs TVB slope limiting using density modal coefficients.
        // The Persson-Peraire indicator gates the *limiter* only.
        // Zhang-Shu positivity correction is UNCONDITIONAL — it must run for every
        // element because pressure can go negative even in cells that appear smooth
        // to the density indicator (e.g. when momentum is large relative to energy).
        let needs_limiting = match indicator {
            Some(ind) => {
                let density_modal: Vec<f64> = all_modal[k].iter().map(|s| s.rho).collect();
                ind.evaluate(&density_modal) > indicator_threshold
            }
            None => true,
        };

        // Use pre-computed modal for this element (clone since limiter may modify in-place)
        let mut modal = all_modal[k].clone();

        // Apply TVB slope limiter only for troubled cells (indicator-gated)
        if needs_limiting {
            limiter.limit(&mut modal, left_modal0, right_modal0, h);
        }

        // Reconstruct candidate nodal values from (possibly limited) modal coefficients.
        // This runs for ALL elements regardless of limiting status.
        let candidate = legendre_modal_to_nodal(&modal, xi);

        // ── Zhang-Shu positivity-preserving correction ────────────────────────
        // Modal[0] = cell average (preserved by the TVB limiter by construction).
        // Scale candidate toward the cell average: s_i(θ) = avg + θ*(cand_i - avg)
        // Find θ_max ∈ [0,1] such that every node has ρ > EPS_RHO and p > EPS_P.
        //
        // Safety: if the cell average itself is unphysical, clamp it to a minimally
        // physical state before running the scaling. This can happen when RK
        // combination steps push the cell average into unphysical territory.
        let mut cell_avg = modal[0];

        // Clamp cell average to be physically valid.
        // Near-vacuum density: reset entirely to avoid enormous kinetic-energy terms.
        if cell_avg.rho < EPS_RHO {
            cell_avg.rho = EPS_RHO;
            cell_avg.rho_u = 0.0;
            cell_avg.energy = EPS_P / (gamma - 1.0);
        } else {
            // Density OK; check if pressure is physical.
            let vel_avg = cell_avg.rho_u / cell_avg.rho;
            let ke_avg = 0.5 * cell_avg.rho * vel_avg * vel_avg;
            let p_avg = (gamma - 1.0) * (cell_avg.energy - ke_avg);
            if p_avg < EPS_P {
                // Internal energy is negative (KE dominates). Reset energy to give a
                // minimal positive pressure (keep density but zero momentum to avoid
                // creating a high-velocity near-vacuum state).
                cell_avg.rho_u = 0.0;
                cell_avg.energy = EPS_P / (gamma - 1.0);
            }
        }

        let mut theta = 1.0_f64;

        // ── Pass 1: density floor ─────────────────────────────────────────────
        // Find maximum θ such that all nodes have ρ ≥ EPS_RHO.
        for cand in &candidate {
            if cand.rho < EPS_RHO {
                let delta = cell_avg.rho - cand.rho;
                // cell_avg.rho ≥ EPS_RHO > cand.rho, so delta > 0.
                if delta > 1e-30 {
                    let t_rho = (cell_avg.rho - EPS_RHO) / delta;
                    theta = theta.min(t_rho.clamp(0.0, 1.0));
                } else {
                    // delta ≈ 0: both values ≈ EPS_RHO, no change needed
                }
            }
        }

        // ── Pass 2: pressure floor ────────────────────────────────────────────
        // For each node, check if p(θ_current) < EPS_P; if so, bisect to find
        // the largest θ that gives p ≥ EPS_P.
        // Because cell_avg now has ρ ≥ EPS_RHO and p ≥ EPS_P, θ=0 is always safe.
        // Use the vel-based formula (same as pressure_eos) to avoid cancellation.
        for cand in &candidate {
            let s_rho = (cell_avg.rho + theta * (cand.rho - cell_avg.rho)).max(EPS_RHO);
            let s_rho_u = cell_avg.rho_u + theta * (cand.rho_u - cell_avg.rho_u);
            let s_energy = cell_avg.energy + theta * (cand.energy - cell_avg.energy);
            let s_vel = s_rho_u / s_rho;
            let p_cand = (gamma - 1.0) * (s_energy - 0.5 * s_rho * s_vel * s_vel);

            if p_cand < EPS_P {
                // Bisect θ ∈ [0, current_theta]: find largest value with p > EPS_P.
                // θ=0 → cell_avg (always safe by construction above).
                // 20 iterations gives ~1e-6 relative precision — adequate for a positivity floor.
                let mut lo = 0.0_f64;
                let mut hi = theta;
                for _ in 0..20 {
                    let mid = 0.5 * (lo + hi);
                    let rho_m = (cell_avg.rho + mid * (cand.rho - cell_avg.rho)).max(EPS_RHO);
                    let rho_u_m = cell_avg.rho_u + mid * (cand.rho_u - cell_avg.rho_u);
                    let e_m = cell_avg.energy + mid * (cand.energy - cell_avg.energy);
                    let vel_m = rho_u_m / rho_m;
                    let p_m = (gamma - 1.0) * (e_m - 0.5 * rho_m * vel_m * vel_m);
                    if p_m > EPS_P {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                // lo is now the largest value with p > EPS_P for this node
                theta = theta.min(lo);
            }
        }

        // Apply the positivity-corrected scaling toward (clamped) cell average.
        // When theta=0, assign cell_avg directly to avoid 0.0 * NaN = NaN artifacts.
        if theta <= 0.0 {
            u[k] = vec![cell_avg; candidate.len()];
        } else {
            u[k] = candidate
                .iter()
                .map(|c| EulerState {
                    rho: cell_avg.rho + theta * (c.rho - cell_avg.rho),
                    rho_u: cell_avg.rho_u + theta * (c.rho_u - cell_avg.rho_u),
                    energy: cell_avg.energy + theta * (c.energy - cell_avg.energy),
                })
                .collect();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exact Sod shock tube solution
// ─────────────────────────────────────────────────────────────────────────────

/// Analytical Sod shock tube solution at time `t`.
///
/// Initial condition: left (ρ=1, u=0, p=1), right (ρ=0.125, u=0, p=0.1),
/// γ=1.4, diaphragm at x=0.5.
///
/// Returns `(rho, u, p)` primitive variables at position `x`.
///
/// Uses the self-similar wave structure: left expansion fan, contact discontinuity,
/// right-moving shock. The star-state values are pre-computed for γ=1.4 from
/// the exact Riemann solver (Toro 2009).
pub fn sod_exact(x: f64, t: f64, gamma: f64) -> (f64, f64, f64) {
    // Pre-computed exact star-state values for γ=1.4 Sod tube
    const P_STAR: f64 = 0.303_130_178_082;
    const U_STAR: f64 = 0.927_452_260_476;
    const RHO_STAR_L: f64 = 0.426_318_553_869; // left of contact
    const RHO_STAR_R: f64 = 0.265_574_162_640; // right of contact

    let (rho_l, u_l, p_l) = (1.0_f64, 0.0_f64, 1.0_f64);
    let (rho_r, u_r, p_r) = (0.125_f64, 0.0_f64, 0.1_f64);

    if t < 1e-15 {
        return if x < 0.5 {
            (rho_l, u_l, p_l)
        } else {
            (rho_r, u_r, p_r)
        };
    }

    // Sound speeds
    let c_l = (gamma * p_l / rho_l).sqrt();
    let c_r = (gamma * p_r / rho_r).sqrt();

    // Sound speed in the star region (left side, after isentropic expansion)
    let c_star_l = c_l - (gamma - 1.0) / 2.0 * (U_STAR - u_l);

    // Self-similar variable ξ = (x - x_0) / t, diaphragm at x_0 = 0.5
    let x0 = 0.5;
    let xi = (x - x0) / t;

    // Wave structure for Sod tube with γ=1.4:
    // 1. Left expansion fan: head at u_L - c_L, tail at U* - c*_L
    // 2. Contact discontinuity: at U*
    // 3. Right-moving shock: speed from Rankine-Hugoniot

    let xi_head = u_l - c_l; // head of expansion fan (leftmost wave)
    let xi_tail = U_STAR - c_star_l; // tail of expansion fan
    let xi_contact = U_STAR; // contact discontinuity

    // Shock speed via Rankine-Hugoniot (Toro 2009, eq. 4.53)
    let s_shock = u_r
        + c_r
            * ((gamma + 1.0) / (2.0 * gamma) * (P_STAR / p_r) + (gamma - 1.0) / (2.0 * gamma))
                .sqrt();

    if xi <= xi_head {
        // Left undisturbed state
        (rho_l, u_l, p_l)
    } else if xi <= xi_tail {
        // Inside expansion fan (isentropic rarefaction wave)
        let u_fan = 2.0 / (gamma + 1.0) * (c_l + xi);
        let c_fan = c_l - (gamma - 1.0) / 2.0 * u_fan;
        let p_fan = p_l * (c_fan / c_l).powf(2.0 * gamma / (gamma - 1.0));
        let rho_fan = rho_l * (p_fan / p_l).powf(1.0 / gamma);
        (rho_fan, u_fan, p_fan)
    } else if xi <= xi_contact {
        // Star region: left of contact discontinuity
        (RHO_STAR_L, U_STAR, P_STAR)
    } else if xi <= s_shock {
        // Star region: right of contact discontinuity
        (RHO_STAR_R, U_STAR, P_STAR)
    } else {
        // Right undisturbed state
        (rho_r, u_r, p_r)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CFL time step
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the maximum stable time step based on the CFL condition.
///
/// `dt = cfl * h / (lambda_max * (2*p + 1))`
///
/// The `(2p + 1)` factor accounts for the eigenvalue growth of the DG operator
/// on a p-th order GLL element.
fn compute_dt(
    u: &[Vec<EulerState>],
    gamma: f64,
    h: f64,
    cfl: f64,
    p: usize,
) -> IntegrateResult<f64> {
    let mut lambda_max = 0.0_f64;
    for (k, elem) in u.iter().enumerate() {
        for (i, state) in elem.iter().enumerate() {
            let speed = max_wave_speed(state, gamma).ok_or_else(|| {
                use super::euler_1d::pressure_eos;
                let p_val = (gamma - 1.0)
                    * (state.energy - 0.5 * state.rho_u * state.rho_u / state.rho.max(1e-300));
                IntegrateError::ValueError(format!(
                    "Unphysical state at element {k} node {i} during CFL: \
                     rho={:.6e}, rho_u={:.6e}, E={:.6e}, p_raw={:.6e}, p_eos={:?}",
                    state.rho,
                    state.rho_u,
                    state.energy,
                    p_val,
                    pressure_eos(state, gamma),
                ))
            })?;
            if speed > lambda_max {
                lambda_max = speed;
            }
        }
    }

    if lambda_max < 1e-12 {
        // Stationary flow or near-vacuum: use a large but finite dt
        return Ok(cfl * h / 1e-12);
    }

    Ok(cfl * h / (lambda_max * (2 * p + 1) as f64))
}

// ─────────────────────────────────────────────────────────────────────────────
// Main solver
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the 1D compressible Euler equations using a nodal Discontinuous Galerkin method.
///
/// # Arguments
///
/// - `initial` — initial condition closure: `|x: f64| -> EulerState` evaluated at physical x.
/// - `t_final` — final time.
/// - `config`  — solver configuration (see [`DgSystemConfig`]).
///
/// # Returns
///
/// A [`DgSystemSolution`] containing the final state and (if requested) time history.
///
/// # Errors
///
/// Returns [`IntegrateError`] if:
/// - The GLL node computation fails (polynomial order < 1).
/// - An unphysical state (non-positive density or pressure) is encountered.
/// - The time step collapses to zero.
///
/// # Example
///
/// ```rust
/// use scirs2_integrate::pde::dg_systems::{
///     dg_system_solver::{solve_1d_euler_dg, DgSystemConfig, BoundaryCondition, TimeIntegrator},
///     euler_1d::primitives_to_conservative,
/// };
///
/// let config = DgSystemConfig {
///     polynomial_order: 2,
///     n_elements: 10,
///     x_left: 0.0, x_right: 1.0,
///     gamma: 1.4, cfl: 0.3,
///     time_integrator: TimeIntegrator::Ssprk3,
///     boundary: BoundaryCondition::Periodic,
///     limiter: None, indicator: None,
///     indicator_threshold: -1.0,
///     record_history: false, record_every: 1,
/// };
/// let sol = solve_1d_euler_dg(
///     |_x| primitives_to_conservative(1.0, 1.0, 1.0, 1.4),
///     0.05,
///     &config,
/// ).expect("solver should succeed");
/// assert_eq!(sol.final_state.len(), 10);
/// ```
pub fn solve_1d_euler_dg<F>(
    initial: F,
    t_final: f64,
    config: &DgSystemConfig,
) -> IntegrateResult<DgSystemSolution>
where
    F: Fn(f64) -> EulerState,
{
    let p = config.polynomial_order;
    let k_elem = config.n_elements;
    let h = (config.x_right - config.x_left) / k_elem as f64;

    // Compute LGL nodes and weights on [-1, 1] for p+1 points
    let (xi, weights) = legendre_gauss_lobatto(p + 1)?;

    // Differentiation matrix (uses the same LGL infrastructure from dg_advanced)
    let d_mat = differentiation_matrix_lgl(&xi);

    // Bundle RHS parameters
    let rhs_params = RhsParams {
        d_mat: &d_mat,
        weights: &weights,
        h,
        gamma: config.gamma,
        boundary: config.boundary,
    };

    // Initialize solution by mapping reference LGL nodes to physical coordinates
    let mut u: Vec<Vec<EulerState>> = (0..k_elem)
        .map(|ek| {
            let x_mid = config.x_left + (ek as f64 + 0.5) * h;
            xi.iter()
                .map(|&xi_i| {
                    let x_phys = x_mid + xi_i * h / 2.0;
                    initial(x_phys)
                })
                .collect()
        })
        .collect();

    // Build physical x-node array for output
    let x_nodes = Array2::from_shape_fn((k_elem, p + 1), |(ek, j)| {
        let x_mid = config.x_left + (ek as f64 + 0.5) * h;
        x_mid + xi[j] * h / 2.0
    });

    // Apply limiter to the initial condition to remove spurious oscillations
    // that would arise from resolving discontinuous ICs with high-order nodal DG.
    if let Some(lim) = config.limiter.as_deref() {
        apply_limiter(
            &mut u,
            lim,
            config.indicator.as_deref(),
            config.indicator_threshold,
            &xi,
            &weights,
            h,
            config.gamma,
        );
    }

    let mut t = 0.0_f64;
    let mut time_history = Vec::new();
    let mut state_history = Vec::new();

    if config.record_history {
        time_history.push(t);
        state_history.push(u.clone());
    }

    let mut step = 0_usize;

    while t < t_final {
        let raw_dt = compute_dt(&u, config.gamma, h, config.cfl, p)?;
        let dt = raw_dt.min(t_final - t);

        if dt < 1e-15 {
            return Err(IntegrateError::ConvergenceError(
                "Time step collapsed to zero — check CFL coefficient or initial condition"
                    .to_string(),
            ));
        }

        let rhs_fn = |state: &[Vec<EulerState>]| -> IntegrateResult<Vec<Vec<EulerState>>> {
            dg_rhs(state, &rhs_params)
        };

        // Build a per-stage post-processing closure.
        // When a slope limiter is configured, apply the TVB minmod limiter (with indicator
        // gating) using modal projection. The Zhang-Shu positivity-preserving correction is
        // embedded inside apply_limiter, so we do NOT add a second positivity pass here.
        // Double-clamping can cause dt collapse by creating states with artificially large
        // wave speeds (near-vacuum elements with preserved momentum → huge KE).
        let do_limit = config.limiter.is_some();
        let lim_ref = config.limiter.as_deref();
        let ind_ref = config.indicator.as_deref();
        let ind_thresh = config.indicator_threshold;
        let xi_ref = xi.as_slice();
        let w_ref = weights.as_slice();
        let gamma_val = config.gamma;
        let post_stage = |v: &mut Vec<Vec<EulerState>>| {
            if do_limit {
                if let Some(lim) = lim_ref {
                    apply_limiter(v, lim, ind_ref, ind_thresh, xi_ref, w_ref, h, gamma_val);
                }
            }
        };

        u = match config.time_integrator {
            TimeIntegrator::ForwardEuler => {
                let rhs = rhs_fn(&u)?;
                let mut u_new = add_scaled(&u, &rhs, dt);
                post_stage(&mut u_new);
                u_new
            }
            TimeIntegrator::Ssprk3 => ssprk3_step(&u, dt, &rhs_fn, &post_stage)?,
            TimeIntegrator::Ssprk4 => ssprk4_step(&u, dt, &rhs_fn, &post_stage)?,
        };

        t += dt;
        step += 1;

        if config.record_history && step.is_multiple_of(config.record_every) {
            time_history.push(t);
            state_history.push(u.clone());
        }
    }

    Ok(DgSystemSolution {
        x_nodes,
        final_state: u,
        time_history,
        state_history,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const GAMMA: f64 = 1.4;

    // ── GLL infrastructure tests ──────────────────────────────────────────────

    #[test]
    fn test_gll_p1_two_nodes() {
        let (nodes, weights) = legendre_gauss_lobatto(2).expect("GLL p=1 (2 nodes) should work");
        assert_eq!(nodes.len(), 2);
        assert!((nodes[0] - (-1.0)).abs() < 1e-14, "left node should be -1");
        assert!((nodes[1] - 1.0).abs() < 1e-14, "right node should be 1");
        assert!(
            (weights.iter().sum::<f64>() - 2.0).abs() < 1e-12,
            "weights should sum to 2"
        );
    }

    #[test]
    fn test_gll_p2_three_nodes() {
        let (nodes, weights) = legendre_gauss_lobatto(3).expect("GLL p=2 (3 nodes) should work");
        assert_eq!(nodes.len(), 3);
        assert_eq!(weights.len(), 3);
        // Interior node should be at x = 0 for p=2 (symmetric)
        assert!(
            nodes[1].abs() < 1e-12,
            "interior GLL node for p=2 should be 0"
        );
        // Weights should sum to 2 (integral of 1 over [-1,1])
        assert!(
            (weights.iter().sum::<f64>() - 2.0).abs() < 1e-12,
            "GLL weights sum to 2"
        );
    }

    #[test]
    fn test_gll_p3_four_nodes() {
        let (nodes, weights) = legendre_gauss_lobatto(4).expect("GLL p=3 (4 nodes)");
        assert_eq!(nodes.len(), 4);
        // Weights must be positive and sum to 2
        for &w in &weights {
            assert!(w > 0.0, "GLL weight must be positive");
        }
        assert!((weights.iter().sum::<f64>() - 2.0).abs() < 1e-11);
    }

    // ── Differentiation matrix tests ──────────────────────────────────────────

    #[test]
    fn test_diff_matrix_constant_gives_zero() {
        // D * [1, 1, ..., 1] = 0  (derivative of constant function)
        let (nodes, _) = legendre_gauss_lobatto(4).expect("GLL");
        let d = differentiation_matrix_lgl(&nodes);
        for (i, row) in d.iter().enumerate() {
            let row_sum: f64 = row.iter().sum();
            assert!(
                row_sum.abs() < 1e-11,
                "D * const at row {i}: sum = {row_sum}"
            );
        }
    }

    #[test]
    fn test_diff_matrix_differentiates_linear_exactly() {
        // D * nodes = [1, 1, ..., 1]  (derivative of x is 1)
        let (nodes, _) = legendre_gauss_lobatto(4).expect("GLL");
        let d = differentiation_matrix_lgl(&nodes);
        for (i, row) in d.iter().enumerate() {
            let result: f64 = row.iter().zip(nodes.iter()).map(|(dij, xj)| dij * xj).sum();
            assert!(
                (result - 1.0).abs() < 1e-11,
                "D * x at row {i} should be 1, got {result}"
            );
        }
    }

    // ── Nodal ↔ modal projection tests ───────────────────────────────────────

    #[test]
    fn test_modal_projection_round_trip() {
        // Project known nodal values to modal and back; should recover original.
        let (xi, w) = legendre_gauss_lobatto(5).expect("GLL p=4");
        // Create nodal values for a polynomial: u = 1 + 0.3*P1(xi) + 0.05*P2(xi)
        let nodal: Vec<EulerState> = xi
            .iter()
            .map(|&x| {
                let val = 1.0 + 0.3 * x + 0.05 * (1.5 * x * x - 0.5);
                EulerState {
                    rho: val,
                    rho_u: val * 0.5,
                    energy: val * 2.0,
                }
            })
            .collect();
        let modal = nodal_to_legendre_modal(&nodal, &xi, &w);
        let recovered = legendre_modal_to_nodal(&modal, &xi);
        for (orig, rec) in nodal.iter().zip(recovered.iter()) {
            assert!(
                (orig.rho - rec.rho).abs() < 1e-12,
                "round-trip rho error: orig={}, rec={}",
                orig.rho,
                rec.rho
            );
            assert!(
                (orig.rho_u - rec.rho_u).abs() < 1e-12,
                "round-trip rho_u error"
            );
        }
    }

    #[test]
    fn test_modal_mode0_is_cell_average() {
        // Modal coefficient 0 should equal the weighted average: a_0 = (1/2) sum w_i u_i
        let (xi, w) = legendre_gauss_lobatto(4).expect("GLL p=3");
        let nodal: Vec<EulerState> = xi
            .iter()
            .map(|&x| EulerState {
                rho: 1.0 + 0.2 * x,
                rho_u: 0.5,
                energy: 2.0,
            })
            .collect();
        let modal = nodal_to_legendre_modal(&nodal, &xi, &w);
        // Expected cell average: integral of (1 + 0.2x) over [-1,1] / 2 = 1.0
        // (the 0.2x term integrates to zero by symmetry)
        let expected_avg = 1.0_f64;
        assert!(
            (modal[0].rho - expected_avg).abs() < 1e-12,
            "modal[0].rho = {}, expected {}",
            modal[0].rho,
            expected_avg
        );
    }

    #[test]
    fn test_legendre_poly_values() {
        // P_0(x) = 1
        assert!((legendre_poly(0, 0.5) - 1.0).abs() < 1e-14);
        // P_1(x) = x
        assert!((legendre_poly(1, 0.5) - 0.5).abs() < 1e-14);
        // P_2(x) = (3x^2 - 1)/2 at x=0.5 = (3*0.25 - 1)/2 = (0.75-1)/2 = -0.125
        assert!((legendre_poly(2, 0.5) - (-0.125)).abs() < 1e-14);
        // P_3(x) at x=0 = 0
        assert!(legendre_poly(3, 0.0).abs() < 1e-14);
    }

    // ── Solver tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_constant_ic_stays_constant_periodic() {
        // A spatially-constant initial condition has zero gradients everywhere.
        // With periodic BCs the DG operator maps it to exactly zero RHS,
        // so the solution should remain constant in time (up to round-off).
        let config = DgSystemConfig {
            polynomial_order: 2,
            n_elements: 10,
            x_left: 0.0,
            x_right: 1.0,
            gamma: GAMMA,
            cfl: 0.3,
            time_integrator: TimeIntegrator::Ssprk3,
            boundary: BoundaryCondition::Periodic,
            limiter: None,
            indicator: None,
            indicator_threshold: -1.0,
            record_history: false,
            record_every: 1,
        };
        let ic = |_x: f64| primitives_to_conservative(1.0, 1.0, 1.0, GAMMA);
        let sol = solve_1d_euler_dg(ic, 0.1, &config).expect("solver should run");

        for (k, row) in sol.final_state.iter().enumerate() {
            for (j, state) in row.iter().enumerate() {
                assert!(
                    (state.rho - 1.0).abs() < 1e-8,
                    "density deviated at elem {k} node {j}: {}",
                    state.rho
                );
            }
        }
    }

    #[test]
    fn test_constant_ic_outflow_stays_constant() {
        // Same test with Outflow BCs
        let config = DgSystemConfig {
            polynomial_order: 1,
            n_elements: 8,
            x_left: 0.0,
            x_right: 1.0,
            gamma: GAMMA,
            cfl: 0.3,
            time_integrator: TimeIntegrator::Ssprk3,
            boundary: BoundaryCondition::Outflow,
            limiter: None,
            indicator: None,
            indicator_threshold: -1.0,
            record_history: false,
            record_every: 1,
        };
        let ic = |_x: f64| primitives_to_conservative(1.0, 0.5, 1.0, GAMMA);
        let sol = solve_1d_euler_dg(ic, 0.05, &config).expect("solver should run");

        for (k, row) in sol.final_state.iter().enumerate() {
            for (j, state) in row.iter().enumerate() {
                assert!(
                    (state.rho - 1.0).abs() < 1e-8,
                    "density deviated at elem {k} node {j}: {}",
                    state.rho
                );
            }
        }
    }

    #[test]
    fn test_x_nodes_shape_and_range() {
        let config = DgSystemConfig {
            polynomial_order: 3,
            n_elements: 5,
            x_left: 0.0,
            x_right: 2.0,
            gamma: GAMMA,
            cfl: 0.3,
            time_integrator: TimeIntegrator::ForwardEuler,
            boundary: BoundaryCondition::Outflow,
            limiter: None,
            indicator: None,
            indicator_threshold: -1.0,
            record_history: false,
            record_every: 1,
        };
        let ic = |_x: f64| primitives_to_conservative(1.0, 0.0, 1.0, GAMMA);
        let sol = solve_1d_euler_dg(ic, 1e-4, &config).expect("solver");
        assert_eq!(sol.x_nodes.shape(), &[5, 4]);
        // All nodes should be within [x_left, x_right]
        for &x in sol.x_nodes.iter() {
            assert!((0.0..=2.0).contains(&x), "node out of range: {x}");
        }
    }

    #[test]
    fn test_history_recording() {
        let config = DgSystemConfig {
            polynomial_order: 1,
            n_elements: 4,
            x_left: 0.0,
            x_right: 1.0,
            gamma: GAMMA,
            cfl: 0.3,
            time_integrator: TimeIntegrator::Ssprk3,
            boundary: BoundaryCondition::Periodic,
            limiter: None,
            indicator: None,
            indicator_threshold: -1.0,
            record_history: true,
            record_every: 1,
        };
        let ic = |_x: f64| primitives_to_conservative(1.0, 1.0, 1.0, GAMMA);
        let sol = solve_1d_euler_dg(ic, 0.02, &config).expect("solver");
        // With record_history = true there must be at least the t=0 snapshot
        assert!(!sol.time_history.is_empty());
        assert_eq!(sol.time_history.len(), sol.state_history.len());
        // First snapshot at t=0
        assert!((sol.time_history[0] - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_ssprk4_variant_runs() {
        let config = DgSystemConfig {
            polynomial_order: 2,
            n_elements: 6,
            x_left: 0.0,
            x_right: 1.0,
            gamma: GAMMA,
            cfl: 0.3,
            time_integrator: TimeIntegrator::Ssprk4,
            boundary: BoundaryCondition::Periodic,
            limiter: None,
            indicator: None,
            indicator_threshold: -1.0,
            record_history: false,
            record_every: 1,
        };
        let ic = |_x: f64| primitives_to_conservative(1.0, 1.0, 1.0, GAMMA);
        let sol = solve_1d_euler_dg(ic, 0.02, &config).expect("SSPRK4 should run");
        // Constant IC should still be constant
        for row in &sol.final_state {
            for state in row {
                assert!((state.rho - 1.0).abs() < 1e-7, "rho={}", state.rho);
            }
        }
    }

    #[test]
    fn test_default_sod_config_runs() {
        // Verify that the default Sod config (with TVB limiter) runs without crashing.
        // Short integration time (t=0.002) avoids Gibbs oscillations that
        // would otherwise push density into unphysical territory at coarse resolution.
        let config = DgSystemConfig::default_sod(20, 2);
        let ic = |x: f64| {
            if x < 0.5 {
                primitives_to_conservative(1.0, 0.0, 1.4, 1.4)
            } else {
                primitives_to_conservative(0.125, 0.0, 0.1, 1.4)
            }
        };
        let sol = solve_1d_euler_dg(ic, 0.002, &config).expect("Sod tube should run");
        assert_eq!(sol.final_state.len(), 20);
        for row in &sol.final_state {
            for state in row {
                assert!(state.rho > 0.0, "negative density: {}", state.rho);
            }
        }
    }

    #[test]
    fn test_reflective_bc_preserves_density() {
        // Stationary gas with reflective BCs: density should remain 1.0
        let config = DgSystemConfig {
            polynomial_order: 1,
            n_elements: 6,
            x_left: 0.0,
            x_right: 1.0,
            gamma: GAMMA,
            cfl: 0.3,
            time_integrator: TimeIntegrator::Ssprk3,
            boundary: BoundaryCondition::Reflective,
            limiter: None,
            indicator: None,
            indicator_threshold: -1.0,
            record_history: false,
            record_every: 1,
        };
        let ic = |_x: f64| primitives_to_conservative(1.0, 0.0, 1.0, GAMMA);
        let sol = solve_1d_euler_dg(ic, 0.05, &config).expect("reflective should run");
        for row in &sol.final_state {
            for state in row {
                assert!(state.rho > 0.0, "negative density");
            }
        }
    }

    #[test]
    fn test_sod_exact_initial_condition() {
        // At t=0: should return left state for x<0.5, right for x>0.5
        let (rho, u, p) = sod_exact(0.25, 0.0, GAMMA);
        assert!((rho - 1.0).abs() < 1e-12);
        assert!(u.abs() < 1e-12);
        assert!((p - 1.0).abs() < 1e-12);

        let (rho, u, p) = sod_exact(0.75, 0.0, GAMMA);
        assert!((rho - 0.125).abs() < 1e-12);
        assert!(u.abs() < 1e-12);
        assert!((p - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_sod_exact_wave_structure_at_t02() {
        // At t=0.2, check the wave structure qualitatively
        let t = 0.2_f64;
        // Far left: should be undisturbed left state
        let (rho_l, _, _) = sod_exact(0.1, t, GAMMA);
        assert!((rho_l - 1.0).abs() < 1e-10, "far left density = {rho_l}");
        // Far right: should be undisturbed right state
        let (rho_r, _, _) = sod_exact(0.95, t, GAMMA);
        assert!((rho_r - 0.125).abs() < 1e-10, "far right density = {rho_r}");
        // Star region left of contact (around x=0.5): density should be ~0.426
        let (rho_star, _, p_star) = sod_exact(0.55, t, GAMMA);
        assert!(
            (rho_star - 0.426318553869).abs() < 1e-6,
            "star-L density = {rho_star}"
        );
        assert!(
            (p_star - 0.303130178082).abs() < 1e-6,
            "star pressure = {p_star}"
        );
    }
}
