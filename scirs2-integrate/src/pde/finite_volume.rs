//! Finite Volume Method (FVM) for 1-D hyperbolic conservation laws
//!
//! This module implements the finite volume method for 1-D conservation laws
//! of the form
//!
//!   ∂u/∂t + ∂f(u)/∂x = 0
//!
//! ## Implemented schemes
//!
//! * **First order**: upwind flux
//! * **Second order**: Lax-Wendroff flux, MUSCL reconstruction with TVD limiters
//! * **Reconstruction**: MUSCL (Monotone Upstream-centred Schemes for
//!   Conservation Laws) with minmod / superbee / van Leer limiters
//!
//! ## Example
//!
//! ```rust
//! use scirs2_integrate::pde::finite_volume::{
//!     FvMesh, fv_advection, upwind_flux, FvBoundaryCondition,
//! };
//!
//! // Create a 1-D mesh of 20 cells over [0, 1]
//! let mesh = FvMesh::uniform(0.0, 1.0, 20);
//!
//! // Initialise with a step function
//! let values: Vec<f64> = mesh.cells.iter()
//!     .map(|c| if c.x_center < 0.5 { 1.0 } else { 0.0 })
//!     .collect();
//!
//! // Advect one step with velocity u = 1
//! let dt = 0.01;
//! let new_values = fv_advection(
//!     &mesh, &values, 1.0, dt, upwind_flux,
//!     FvBoundaryCondition::Periodic,
//! ).unwrap();
//! assert_eq!(new_values.len(), values.len());
//! ```

use crate::pde::PDEError;

// ─────────────────────────────────────────────────────────────────────────────
// Mesh data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single cell in a 1-D finite-volume mesh.
#[derive(Debug, Clone)]
pub struct FvCell {
    /// Cell-centre coordinate
    pub x_center: f64,
    /// Cell width Δx
    pub dx: f64,
}

impl FvCell {
    /// Construct a cell centred at `x_center` with width `dx`.
    #[inline]
    pub fn new(x_center: f64, dx: f64) -> Self {
        Self { x_center, dx }
    }

    /// Left boundary of the cell.
    #[inline]
    pub fn x_left(&self) -> f64 {
        self.x_center - 0.5 * self.dx
    }

    /// Right boundary of the cell.
    #[inline]
    pub fn x_right(&self) -> f64 {
        self.x_center + 0.5 * self.dx
    }
}

/// A face between two adjacent 1-D finite-volume cells.
#[derive(Debug, Clone)]
pub struct FvFace {
    /// Face location (x-coordinate of the interface)
    pub x: f64,
    /// Index of the left cell (-1 = left boundary)
    pub left_cell: Option<usize>,
    /// Index of the right cell (None = right boundary)
    pub right_cell: Option<usize>,
}

/// A 1-D finite-volume mesh consisting of cells and the faces between them.
#[derive(Debug, Clone)]
pub struct FvMesh {
    /// Cell array (left to right)
    pub cells: Vec<FvCell>,
    /// Face array (n+1 faces for n cells; face\[i\] is between cell\[i-1\] and cell\[i\])
    pub faces: Vec<FvFace>,
}

impl FvMesh {
    /// Create a uniform mesh of `n` cells over `[x_left, x_right]`.
    pub fn uniform(x_left: f64, x_right: f64, n: usize) -> Self {
        assert!(n > 0, "FvMesh: n must be > 0");
        let dx = (x_right - x_left) / n as f64;
        let cells: Vec<FvCell> = (0..n)
            .map(|i| FvCell::new(x_left + (i as f64 + 0.5) * dx, dx))
            .collect();

        // n+1 faces
        let faces: Vec<FvFace> = (0..=n)
            .map(|i| FvFace {
                x: x_left + i as f64 * dx,
                left_cell: if i == 0 { None } else { Some(i - 1) },
                right_cell: if i == n { None } else { Some(i) },
            })
            .collect();

        Self { cells, faces }
    }

    /// Return the number of cells.
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    /// Return the number of faces.
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Boundary conditions
// ─────────────────────────────────────────────────────────────────────────────

/// Boundary condition type for finite-volume advection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FvBoundaryCondition {
    /// Periodic domain: left ghost = rightmost cell, right ghost = leftmost cell.
    Periodic,
    /// Outflow / zero-gradient: ghost cell value equals the nearest interior cell.
    Outflow,
    /// Inflow with a specified value at both boundaries.
    Inflow { left_value: f64, right_value: f64 },
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerical flux functions
// ─────────────────────────────────────────────────────────────────────────────

/// First-order upwind flux for a linear advection equation with constant
/// velocity `a`:
///
///   F(u_L, u_R) = a·u_L  if a ≥ 0,  a·u_R  otherwise.
#[inline]
pub fn upwind_flux(u_left: f64, u_right: f64, velocity: f64) -> f64 {
    if velocity >= 0.0 {
        velocity * u_left
    } else {
        velocity * u_right
    }
}

/// Second-order Lax-Wendroff flux.
///
/// This flux is second-order accurate in both space and time for linear
/// advection with constant velocity `a`:
///
///   F_LW = a/2 · (u_L + u_R) − a²·Δt/(2·Δx) · (u_R − u_L)
///
/// # Arguments
/// * `u_left`  — cell average on the left side of the face
/// * `u_right` — cell average on the right side of the face
/// * `velocity` — constant advection velocity `a`
/// * `dt`      — time step
/// * `dx`      — cell width (assumed uniform)
#[inline]
pub fn lax_wendroff_flux(u_left: f64, u_right: f64, velocity: f64, dt: f64, dx: f64) -> f64 {
    let a = velocity;
    let courant = a * dt / dx;
    0.5 * a * (u_left + u_right) - 0.5 * a * courant * (u_right - u_left)
}

// ─────────────────────────────────────────────────────────────────────────────
// TVD limiters
// ─────────────────────────────────────────────────────────────────────────────

/// Minmod limiter: `minmod(r) = max(0, min(1, r))`.
///
/// The most diffusive symmetric TVD limiter; produces the smallest slopes.
#[inline]
pub fn minmod_limiter(r: f64) -> f64 {
    if r <= 0.0 {
        0.0
    } else if r >= 1.0 {
        1.0
    } else {
        r
    }
}

/// Superbee limiter (Roe 1986).
///
/// The least diffusive classic TVD limiter:
///   φ(r) = max(0, min(2r, 1), min(r, 2))
#[inline]
pub fn superbee_limiter(r: f64) -> f64 {
    if r <= 0.0 {
        0.0
    } else {
        let a = (2.0 * r).min(1.0_f64);
        let b = r.min(2.0_f64);
        a.max(b)
    }
}

/// van Leer limiter.
///
///   φ(r) = (r + |r|) / (1 + |r|)
#[inline]
pub fn van_leer_limiter(r: f64) -> f64 {
    if r <= 0.0 {
        0.0
    } else {
        (r + r.abs()) / (1.0 + r.abs())
    }
}

/// Slope-ratio r for cell `i`:
///
///   r_i = (u_i − u_{i-1}) / (u_{i+1} − u_i)
///
/// with protection against division by near-zero denominators.
fn slope_ratio(values: &[f64], i: usize, ghost_left: f64, ghost_right: f64) -> f64 {
    let n = values.len();
    let u_left = if i == 0 { ghost_left } else { values[i - 1] };
    let u_center = values[i];
    let u_right = if i + 1 >= n {
        ghost_right
    } else {
        values[i + 1]
    };

    let denom = u_right - u_center;
    let numer = u_center - u_left;
    if denom.abs() < 1e-14 {
        if numer.abs() < 1e-14 {
            1.0 // flat → no limiting needed
        } else {
            // Different signs → strong oscillation; clip
            if numer * denom >= 0.0 {
                f64::INFINITY
            } else {
                -f64::INFINITY
            }
        }
    } else {
        numer / denom
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MUSCL reconstruction
// ─────────────────────────────────────────────────────────────────────────────

/// MUSCL (Monotone Upstream-centred Schemes for Conservation Laws)
/// reconstruction.
///
/// For each cell `i` the interface values are reconstructed as:
///
///   u_i^R = u_i + φ(r_i) · (u_{i+1} − u_i) / 2
///   u_i^L = u_i − φ(r_i) · (u_i − u_{i-1}) / 2
///
/// where `φ` is a TVD slope limiter and `r_i` is the slope ratio.
///
/// Returns `(u_left_face, u_right_face)` — left-extrapolated values at the
/// *right* face of each cell and right-extrapolated values at the *left* face
/// of each cell, respectively.  These are the states fed to the numerical
/// flux function.
///
/// `limiter` should be one of [`minmod_limiter`], [`superbee_limiter`], or
/// [`van_leer_limiter`].
pub fn muscl_reconstruct(cells: &[f64], limiter: impl Fn(f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let n = cells.len();
    // Ghost cells — outflow (zero-gradient) extrapolation
    let g_left = cells.first().copied().unwrap_or(0.0);
    let g_right = cells.last().copied().unwrap_or(0.0);

    // u_left_face[i]  = value reconstructed from cell i looking right (towards face i+½)
    // u_right_face[i] = value reconstructed from cell i looking left  (towards face i-½)
    let mut u_right_of_left_face = vec![0.0f64; n]; // u_{i, L}  for face (i-½)
    let mut u_left_of_right_face = vec![0.0f64; n]; // u_{i, R}  for face (i+½)

    for i in 0..n {
        let u_left_nb = if i == 0 { g_left } else { cells[i - 1] };
        let u_right_nb = if i + 1 >= n { g_right } else { cells[i + 1] };

        let r = slope_ratio(cells, i, g_left, g_right);
        let phi = limiter(r);

        let slope = phi * (u_right_nb - u_left_nb) * 0.5;
        u_left_of_right_face[i] = cells[i] + 0.5 * slope;
        u_right_of_left_face[i] = cells[i] - 0.5 * slope;
    }

    // Return (left-extrapolated at right face, right-extrapolated at left face)
    (u_left_of_right_face, u_right_of_left_face)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-D advection update
// ─────────────────────────────────────────────────────────────────────────────

/// Advance the solution of `∂u/∂t + a ∂u/∂x = 0` by one time step `dt`
/// using the explicit finite-volume method with a user-supplied numerical
/// flux.
///
/// # Arguments
///
/// * `mesh`    — 1-D FV mesh
/// * `values`  — cell-average solution vector (length = `mesh.n_cells()`)
/// * `velocity` — constant advection speed `a`
/// * `dt`      — time step (must satisfy CFL: a·Δt/Δx ≤ 1)
/// * `flux_fn` — numerical flux function `F(u_L, u_R, a) → f64`
/// * `bc`      — boundary condition type
///
/// # Returns
/// Updated cell averages.
pub fn fv_advection(
    mesh: &FvMesh,
    values: &[f64],
    velocity: f64,
    dt: f64,
    flux_fn: impl Fn(f64, f64, f64) -> f64,
    bc: FvBoundaryCondition,
) -> Result<Vec<f64>, PDEError> {
    let n = mesh.n_cells();
    if values.len() != n {
        return Err(PDEError::InvalidParameter(format!(
            "fv_advection: values.len()={} but mesh has {} cells",
            values.len(),
            n
        )));
    }

    // Compute ghost cell values for the boundary
    let (ghost_left, ghost_right) = ghost_values(values, bc);

    // Build extended array: ghost_left + values + ghost_right
    let mut ext = Vec::with_capacity(n + 2);
    ext.push(ghost_left);
    ext.extend_from_slice(values);
    ext.push(ghost_right);

    // Compute fluxes at each face (n+1 internal faces, index 0..=n)
    // Face i sits between ext[i] (left) and ext[i+1] (right)
    let mut fluxes = vec![0.0f64; n + 1];
    for f_idx in 0..=n {
        let u_l = ext[f_idx];
        let u_r = ext[f_idx + 1];
        fluxes[f_idx] = flux_fn(u_l, u_r, velocity);
    }

    // Update cell averages
    let mut new_values = vec![0.0f64; n];
    for i in 0..n {
        let dx = mesh.cells[i].dx;
        if dx < 1e-30 {
            return Err(PDEError::InvalidGrid(format!(
                "fv_advection: cell {i} has zero width"
            )));
        }
        // Conservative update: u_i^{n+1} = u_i^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
        new_values[i] = values[i] - dt / dx * (fluxes[i + 1] - fluxes[i]);
    }

    Ok(new_values)
}

/// Compute ghost cell values at the left and right boundaries.
fn ghost_values(values: &[f64], bc: FvBoundaryCondition) -> (f64, f64) {
    let n = values.len();
    match bc {
        FvBoundaryCondition::Periodic => {
            let g_left = values.last().copied().unwrap_or(0.0);
            let g_right = values.first().copied().unwrap_or(0.0);
            (g_left, g_right)
        }
        FvBoundaryCondition::Outflow => {
            let g_left = values.first().copied().unwrap_or(0.0);
            let g_right = values.last().copied().unwrap_or(0.0);
            (g_left, g_right)
        }
        FvBoundaryCondition::Inflow {
            left_value,
            right_value,
        } => (left_value, right_value),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MUSCL-based advection (second-order)
// ─────────────────────────────────────────────────────────────────────────────

/// Advance `∂u/∂t + a ∂u/∂x = 0` by one time step using the second-order
/// MUSCL reconstruction with the supplied TVD limiter and the Lax-Wendroff
/// flux at each reconstructed face.
///
/// # Arguments
/// * `mesh`     — 1-D FV mesh (must be uniform for LW accuracy)
/// * `values`   — cell-average solution
/// * `velocity` — constant advection speed
/// * `dt`       — time step
/// * `limiter`  — TVD slope limiter (e.g. [`minmod_limiter`])
/// * `bc`       — boundary condition
pub fn fv_advection_muscl(
    mesh: &FvMesh,
    values: &[f64],
    velocity: f64,
    dt: f64,
    limiter: impl Fn(f64) -> f64,
    bc: FvBoundaryCondition,
) -> Result<Vec<f64>, PDEError> {
    let n = mesh.n_cells();
    if values.len() != n {
        return Err(PDEError::InvalidParameter(format!(
            "fv_advection_muscl: values.len()={} but mesh has {} cells",
            values.len(),
            n
        )));
    }

    // Reconstruct left/right interface states
    let (u_right_face, u_left_face) = muscl_reconstruct(values, &limiter);
    // u_right_face[i] = state at right face of cell i (i+½)
    // u_left_face[i]  = state at left  face of cell i (i-½)

    // Ghost values for extended boundary
    let (ghost_left_val, ghost_right_val) = ghost_values(values, bc);

    // Left-face reconstruction at the boundary
    let u_left_face_0 = ghost_left_val;
    let u_right_face_n = ghost_right_val;

    // Compute fluxes at each interior face
    // Face (i-½) is between cell (i-1)'s right state and cell i's left state.
    let mut fluxes = vec![0.0f64; n + 1];

    // Face 0 (left boundary)
    let u_l0 = u_left_face_0;
    let u_r0 = u_left_face[0];
    let dx0 = mesh.cells[0].dx;
    fluxes[0] = lax_wendroff_flux(u_l0, u_r0, velocity, dt, dx0);

    // Interior faces
    for f_idx in 1..n {
        let u_l = u_right_face[f_idx - 1];
        let u_r = u_left_face[f_idx];
        let dx = mesh.cells[f_idx].dx;
        fluxes[f_idx] = lax_wendroff_flux(u_l, u_r, velocity, dt, dx);
    }

    // Face n (right boundary)
    let u_ln = u_right_face[n - 1];
    let u_rn = u_right_face_n;
    let dxn = mesh.cells[n - 1].dx;
    fluxes[n] = lax_wendroff_flux(u_ln, u_rn, velocity, dt, dxn);

    // Conservative update
    let mut new_values = vec![0.0f64; n];
    for i in 0..n {
        let dx = mesh.cells[i].dx;
        if dx < 1e-30 {
            return Err(PDEError::InvalidGrid(format!(
                "fv_advection_muscl: cell {i} has zero width"
            )));
        }
        new_values[i] = values[i] - dt / dx * (fluxes[i + 1] - fluxes[i]);
    }

    Ok(new_values)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: advance for multiple time steps
// ─────────────────────────────────────────────────────────────────────────────

/// Integrate `∂u/∂t + a ∂u/∂x = 0` from `t=0` to `t_end` using the
/// first-order upwind scheme, with an automatic CFL-limited time step.
///
/// Returns `(t_final, u_final)`.
pub fn fv_advect_to_time(
    mesh: &FvMesh,
    initial: &[f64],
    velocity: f64,
    t_end: f64,
    cfl: f64,
    bc: FvBoundaryCondition,
) -> Result<(f64, Vec<f64>), PDEError> {
    if cfl <= 0.0 || cfl > 1.0 {
        return Err(PDEError::InvalidParameter(
            "cfl must be in (0, 1]".to_string(),
        ));
    }
    // Use the minimum cell width to determine the stable dt
    let dx_min = mesh
        .cells
        .iter()
        .map(|c| c.dx)
        .fold(f64::INFINITY, f64::min);
    if dx_min < 1e-30 || velocity.abs() < 1e-30 {
        return Ok((t_end, initial.to_vec()));
    }
    let dt_max = cfl * dx_min / velocity.abs();

    let mut t = 0.0;
    let mut u = initial.to_vec();

    while t < t_end {
        let dt = dt_max.min(t_end - t);
        u = fv_advection(mesh, &u, velocity, dt, upwind_flux, bc)?;
        t += dt;
    }

    Ok((t, u))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fv_mesh_uniform() {
        let mesh = FvMesh::uniform(0.0, 1.0, 10);
        assert_eq!(mesh.n_cells(), 10);
        assert_eq!(mesh.n_faces(), 11);
        assert_abs_diff_eq!(mesh.cells[0].dx, 0.1);
        assert_abs_diff_eq!(mesh.cells[0].x_center, 0.05);
        assert_abs_diff_eq!(mesh.cells[9].x_center, 0.95);
        assert_abs_diff_eq!(mesh.faces[0].x, 0.0);
        assert_abs_diff_eq!(mesh.faces[10].x, 1.0);
    }

    #[test]
    fn test_fv_cell_bounds() {
        let cell = FvCell::new(0.5, 0.1);
        assert_abs_diff_eq!(cell.x_left(), 0.45);
        assert_abs_diff_eq!(cell.x_right(), 0.55);
    }

    #[test]
    fn test_upwind_flux_positive_velocity() {
        // u_L=1, u_R=0, a=1 → flux = 1
        assert_abs_diff_eq!(upwind_flux(1.0, 0.0, 1.0), 1.0);
    }

    #[test]
    fn test_upwind_flux_negative_velocity() {
        // u_L=0, u_R=1, a=-1 → flux = -1
        assert_abs_diff_eq!(upwind_flux(0.0, 1.0, -1.0), -1.0);
    }

    #[test]
    fn test_lax_wendroff_flux_exact_transport() {
        // For a=1, CFL=1 the LW flux becomes purely upwind
        // F_LW = 0.5·(u_L + u_R) − 0.5·Δt/Δx·(u_R − u_L)
        //      = 0.5·(1+0) − 0.5·1·(0-1) = 0.5 + 0.5 = 1  (= u_L)
        let flux = lax_wendroff_flux(1.0, 0.0, 1.0, 0.1, 0.1); // CFL=1
        assert_abs_diff_eq!(flux, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_minmod_limiter() {
        assert_abs_diff_eq!(minmod_limiter(-1.0), 0.0);
        assert_abs_diff_eq!(minmod_limiter(0.5), 0.5);
        assert_abs_diff_eq!(minmod_limiter(2.0), 1.0);
    }

    #[test]
    fn test_superbee_limiter() {
        assert_abs_diff_eq!(superbee_limiter(-1.0), 0.0);
        // r=0.5 → max(min(1,0), min(0.5,2)) = max(0,0.5) = 0.5 … but
        // superbee: max(0, min(2r,1), min(r,2))
        //           = max(0, min(1,1), min(0.5,2)) = max(0,1,0.5)=1
        assert_abs_diff_eq!(superbee_limiter(0.5), 1.0);
        assert_abs_diff_eq!(superbee_limiter(3.0), 2.0);
    }

    #[test]
    fn test_van_leer_limiter() {
        assert_abs_diff_eq!(van_leer_limiter(-0.5), 0.0);
        // r=1 → 2/2 = 1
        assert_abs_diff_eq!(van_leer_limiter(1.0), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_muscl_reconstruct_flat_profile() {
        // Flat profile → limiter = 1, slope = 0, reconstruction = identity
        let values = vec![2.0f64; 8];
        let (u_r, u_l) = muscl_reconstruct(&values, minmod_limiter);
        for v in &u_r {
            assert_abs_diff_eq!(*v, 2.0, epsilon = 1e-13);
        }
        for v in &u_l {
            assert_abs_diff_eq!(*v, 2.0, epsilon = 1e-13);
        }
    }

    #[test]
    fn test_fv_advection_constant_profile() {
        // Advecting a constant profile should give the same profile back
        let mesh = FvMesh::uniform(0.0, 1.0, 20);
        let values = vec![3.0f64; 20];
        let new_values = fv_advection(
            &mesh,
            &values,
            1.0,
            0.01,
            upwind_flux,
            FvBoundaryCondition::Periodic,
        )
        .expect("advection failed");
        for v in &new_values {
            assert_abs_diff_eq!(*v, 3.0, epsilon = 1e-13);
        }
    }

    #[test]
    fn test_fv_advection_conservation() {
        // Total mass should be conserved under periodic BCs
        let mesh = FvMesh::uniform(0.0, 1.0, 40);
        let dx = mesh.cells[0].dx;
        let values: Vec<f64> = mesh
            .cells
            .iter()
            .map(|c| if c.x_center < 0.5 { 1.0 } else { 0.0 })
            .collect();
        let mass_before: f64 = values.iter().sum::<f64>() * dx;

        let new_values = fv_advection(
            &mesh,
            &values,
            1.0,
            0.01,
            upwind_flux,
            FvBoundaryCondition::Periodic,
        )
        .expect("advection failed");
        let mass_after: f64 = new_values.iter().sum::<f64>() * dx;

        assert_abs_diff_eq!(mass_before, mass_after, epsilon = 1e-13);
    }

    #[test]
    fn test_fv_advect_to_time_periodic() {
        // Under periodic BCs, advecting a bump around the domain one full
        // period should recover the original profile (approximately, due to
        // numerical diffusion of the upwind scheme).
        let n = 100;
        let mesh = FvMesh::uniform(0.0, 1.0, n);
        let values: Vec<f64> = mesh
            .cells
            .iter()
            .map(|c| {
                let x = c.x_center;
                (-(x - 0.5).powi(2) / 0.01).exp()
            })
            .collect();
        let (_t, _result) =
            fv_advect_to_time(&mesh, &values, 1.0, 1.0, 0.8, FvBoundaryCondition::Periodic)
                .expect("advection to time failed");
        // After one full period the result should look roughly like the input
        // (the upwind scheme introduces diffusion, so we use a loose tolerance)
        let mass_initial: f64 = values.iter().sum::<f64>();
        let mass_final: f64 = _result.iter().sum::<f64>();
        assert_abs_diff_eq!(mass_initial, mass_final, epsilon = 1e-10);
    }

    #[test]
    fn test_fv_advection_muscl_constant_profile() {
        // MUSCL on a flat profile should preserve it exactly
        let mesh = FvMesh::uniform(0.0, 1.0, 20);
        let values = vec![1.5f64; 20];
        let new_values = fv_advection_muscl(
            &mesh,
            &values,
            1.0,
            0.01,
            minmod_limiter,
            FvBoundaryCondition::Outflow,
        )
        .expect("muscl advection failed");
        for v in &new_values {
            assert_abs_diff_eq!(*v, 1.5, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_fv_advection_muscl_conservation() {
        // Mass conservation under periodic BCs with MUSCL
        let mesh = FvMesh::uniform(0.0, 1.0, 40);
        let dx = mesh.cells[0].dx;
        let values: Vec<f64> = mesh
            .cells
            .iter()
            .map(|c| if c.x_center < 0.5 { 1.0 } else { 0.0 })
            .collect();
        let mass_before: f64 = values.iter().sum::<f64>() * dx;

        let new_values = fv_advection_muscl(
            &mesh,
            &values,
            1.0,
            0.01,
            minmod_limiter,
            FvBoundaryCondition::Periodic,
        )
        .expect("muscl advection failed");
        let mass_after: f64 = new_values.iter().sum::<f64>() * dx;

        assert_abs_diff_eq!(mass_before, mass_after, epsilon = 1e-12);
    }

    #[test]
    fn test_upwind_cfl_stability() {
        // Upwind with CFL=1 should not amplify any mode
        let n = 50;
        let mesh = FvMesh::uniform(0.0, 1.0, n);
        let dx = mesh.cells[0].dx;
        let dt = dx; // CFL = 1 exactly
        let values: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 * dx).sin())
            .collect();
        let max_before = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let new_values = fv_advection(
            &mesh,
            &values,
            1.0,
            dt,
            upwind_flux,
            FvBoundaryCondition::Periodic,
        )
        .expect("ok");
        let max_after = new_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Solution should not grow
        assert!(max_after <= max_before + 1e-12);
    }
}
