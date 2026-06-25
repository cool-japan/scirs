//! Lattice Boltzmann Method (LBM) — D2Q9 simulation.
//!
//! This module provides a D2Q9 LBM solver with a BGK (Bhatnagar-Gross-Krook)
//! single-relaxation-time collision operator.
//!
//! **Execution:** the solver currently runs entirely on the CPU. The type is
//! named `GpuLbm2D` and exposes a [`GpuLbmDispatch`] selector to keep the API
//! stable for a future GPU back-end, but no GPU code path exists yet — every
//! dispatch variant executes the same CPU numerics. See [`GpuLbmDispatch`] for
//! details.
//!
//! ## Lattice geometry (D2Q9)
//!
//! Nine velocity directions on a 2-D square lattice:
//!
//! ```text
//! 6  2  5
//! 3  0  1
//! 7  4  8
//! ```
//!
//! Speed of sound squared in lattice units: `cs² = 1/3`.
//!
//! ## Memory layout
//!
//! Distribution functions are stored in a flat 1-D array using row-major order:
//!
//! ```text
//! f[y * nx * 9 + x * 9 + q]
//! ```
//!
//! ## BGK step
//!
//! Each timestep:
//! 1. Compute macroscopic fields (`ρ`, `u`) from `f`.
//! 2. Collision: `f_q ← f_q − (f_q − f_q^{eq}) / τ`.
//! 3. Streaming: shift distributions along their velocity directions.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_integrate::gpu_lbm::{GpuLbm2D, LbmConfig, BoundaryCondition, GpuLbmDispatch};
//!
//! let config = LbmConfig {
//!     nx: 32,
//!     ny: 32,
//!     tau: 0.8,
//!     boundary: BoundaryCondition::Periodic,
//!     dispatch: GpuLbmDispatch::Cpu,
//! };
//! let mut sim = GpuLbm2D::new(config);
//! sim.step(10);
//! let mass = sim.total_mass();
//! assert!(mass > 0.0);
//! ```

/// D2Q9 lattice velocity vectors: `D2Q9_VELOCITIES[q] = [cx, cy]`.
pub const D2Q9_VELOCITIES: [[i32; 2]; 9] = [
    [0, 0],
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1],
];

/// D2Q9 equilibrium weights.
pub const D2Q9_WEIGHTS: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// Opposite-direction index used for bounce-back boundary conditions.
///
/// `OPPOSITE[q]` is the direction with velocity `−c_q`.
const OPPOSITE: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Speed of sound squared in lattice units `(cs² = 1/3)`.
const CS2: f64 = 1.0 / 3.0;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Boundary condition applied at every timestep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Wrap around all four edges (no solid walls).
    Periodic,
    /// Bounce-back at all four edges — no-slip walls.
    NoSlip,
    /// Mirror (free-slip) reflection at all four edges.
    FreeSlip,
}

/// Selects the execution path for the LBM solver.
///
/// No GPU back-end is currently wired in: both variants run identical CPU
/// numerics.  `CpuReference` exists as an explicit, separately-named reference
/// path so that a future GPU implementation can be validated against it (see
/// `test_cpu_reference_matches_cpu`).  It does **not** execute on a GPU and
/// makes no such claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLbmDispatch {
    /// Standard sequential CPU execution.
    Cpu,
    /// CPU reference path used to validate (future) accelerated back-ends.
    ///
    /// Produces bit-for-bit the same results as [`GpuLbmDispatch::Cpu`]; it is
    /// not a GPU execution path.
    CpuReference,
}

/// Configuration for a `GpuLbm2D` simulation.
#[derive(Debug, Clone)]
pub struct LbmConfig {
    /// Grid width (number of cells in x-direction).
    pub nx: usize,
    /// Grid height (number of cells in y-direction).
    pub ny: usize,
    /// BGK relaxation time τ.  Kinematic viscosity `ν = (τ − 0.5) / 3`.
    /// Must satisfy `τ > 0.5` for stability.
    pub tau: f64,
    /// Boundary condition applied to the domain edges.
    pub boundary: BoundaryCondition,
    /// Execution dispatch strategy.
    pub dispatch: GpuLbmDispatch,
}

/// Snapshot of the LBM state.
#[derive(Debug, Clone)]
pub struct LbmState {
    /// Distribution functions, layout: `f[y * nx * 9 + x * 9 + q]`.
    pub f: Vec<f64>,
    /// Density field, layout: `rho[y * nx + x]`.
    pub rho: Vec<f64>,
    /// Velocity field, layout: `u[y * nx + x] = [ux, uy]`.
    pub u: Vec<[f64; 2]>,
    /// Number of timesteps executed so far.
    pub step: usize,
}

/// D2Q9 LBM solver (CPU execution; GPU-ready API surface).
pub struct GpuLbm2D {
    config: LbmConfig,
    state: LbmState,
    /// Streaming buffer (same size as `state.f`).
    f_buf: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core physics — free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the equilibrium distribution for all 9 directions given macroscopic
/// density `rho` and velocity `(ux, uy)`.
///
/// Formula:
/// `f_q^{eq} = w_q * ρ * [1 + (c_q·u)/cs² + (c_q·u)²/(2 cs⁴) − |u|²/(2 cs²)]`
pub fn equilibrium_f(rho: f64, ux: f64, uy: f64) -> [f64; 9] {
    let u2 = ux * ux + uy * uy;
    let mut feq = [0.0_f64; 9];
    for q in 0..9 {
        let cx = D2Q9_VELOCITIES[q][0] as f64;
        let cy = D2Q9_VELOCITIES[q][1] as f64;
        let cu = cx * ux + cy * uy;
        feq[q] = D2Q9_WEIGHTS[q]
            * rho
            * (1.0 + cu / CS2 + cu * cu / (2.0 * CS2 * CS2) - u2 / (2.0 * CS2));
    }
    feq
}

/// BGK collision: relax each distribution towards equilibrium.
///
/// `f_q ← f_q − (f_q − f_q^{eq}) / τ`
pub fn collide(f: &mut [f64], rho: &[f64], u: &[[f64; 2]], tau: f64, nx: usize, ny: usize) {
    let omega = 1.0 / tau;
    for y in 0..ny {
        for x in 0..nx {
            let cell = y * nx + x;
            let r = rho[cell];
            let [ux, uy] = u[cell];
            let feq = equilibrium_f(r, ux, uy);
            let base = cell * 9;
            for q in 0..9 {
                f[base + q] += omega * (feq[q] - f[base + q]);
            }
        }
    }
}

/// Streaming step with periodic boundary conditions.
///
/// Each distribution `f_q(x, y)` is propagated to `(x + cx_q, y + cy_q)` with
/// wrap-around at all four domain edges.
pub fn stream(f: &mut [f64], nx: usize, ny: usize) {
    let n = nx * ny * 9;
    let mut f_new = vec![0.0_f64; n];

    for y in 0..ny {
        for x in 0..nx {
            for q in 0..9 {
                let cx = D2Q9_VELOCITIES[q][0];
                let cy = D2Q9_VELOCITIES[q][1];
                // Destination cell (with periodic wrap)
                let xd = ((x as i64 + cx as i64).rem_euclid(nx as i64)) as usize;
                let yd = ((y as i64 + cy as i64).rem_euclid(ny as i64)) as usize;
                let src = (y * nx + x) * 9 + q;
                let dst = (yd * nx + xd) * 9 + q;
                f_new[dst] = f[src];
            }
        }
    }
    f.copy_from_slice(&f_new);
}

/// Compute macroscopic density and velocity from the distribution functions.
///
/// `ρ = Σ_q f_q`,  `u = (Σ_q c_q f_q) / ρ`
pub fn compute_macroscopic(f: &[f64], rho: &mut [f64], u: &mut [[f64; 2]], nx: usize, ny: usize) {
    for y in 0..ny {
        for x in 0..nx {
            let cell = y * nx + x;
            let base = cell * 9;
            let mut r = 0.0_f64;
            let mut mx = 0.0_f64;
            let mut my = 0.0_f64;
            for q in 0..9 {
                let fq = f[base + q];
                r += fq;
                mx += D2Q9_VELOCITIES[q][0] as f64 * fq;
                my += D2Q9_VELOCITIES[q][1] as f64 * fq;
            }
            rho[cell] = r;
            if r > 1e-30 {
                u[cell] = [mx / r, my / r];
            } else {
                u[cell] = [0.0, 0.0];
            }
        }
    }
}

/// Apply no-slip (bounce-back) boundary at all four walls.
fn apply_noslip_walls(f: &mut [f64], nx: usize, ny: usize) {
    // For wall nodes we reverse all populations in-place (full-way bounce-back).
    let mut is_wall = vec![false; nx * ny];
    // Bottom and top rows
    for x in 0..nx {
        is_wall[x] = true;
        is_wall[(ny - 1) * nx + x] = true;
    }
    // Left and right columns
    for y in 0..ny {
        is_wall[y * nx] = true;
        is_wall[y * nx + (nx - 1)] = true;
    }
    for cell in 0..nx * ny {
        if !is_wall[cell] {
            continue;
        }
        let base = cell * 9;
        let mut tmp = [0.0_f64; 9];
        for q in 0..9 {
            tmp[OPPOSITE[q]] = f[base + q];
        }
        f[base..base + 9].copy_from_slice(&tmp);
    }
}

/// Apply free-slip (mirror) boundary at all four walls.
fn apply_freeslip_walls(f: &mut [f64], nx: usize, ny: usize) {
    // Mirror reflections along y for top/bottom, along x for left/right.
    // Free-slip: tangential velocity unchanged, normal velocity reversed.
    // For D2Q9 we mirror the incoming populations across the wall normal.
    //
    // Velocity indices:
    //   q=0  (0, 0)  — rest
    //   q=1  (1, 0)  — right       q=3 (-1, 0) — left
    //   q=2  (0, 1)  — up          q=4  (0,-1) — down
    //   q=5  (1, 1)  — up-right    q=7 (-1,-1) — down-left   (opposite of 5)
    //   q=6 (-1, 1)  — up-left     q=8  (1,-1) — down-right  (opposite of 6)
    //
    // Bottom wall (y=0): incoming from below (q=4,7,8) → mirror cy  (q=4↔2, 7↔6, 8↔5)
    // Top wall (y=ny-1): incoming from above (q=2,5,6) → mirror cy  (q=2↔4, 5↔8, 6↔7)
    // Left wall (x=0):   incoming from left  (q=3,6,7) → mirror cx  (q=3↔1, 6↔5, 7↔8)
    // Right wall (x=nx-1): incoming from right (q=1,5,8) → mirror cx (q=1↔3, 5↔6, 8↔7)

    // Bottom / top rows
    for x in 0..nx {
        // Bottom (y=0)
        {
            let cell = x;
            let b = cell * 9;
            let f4 = f[b + 4];
            let f7 = f[b + 7];
            let f8 = f[b + 8];
            f[b + 2] = f4;
            f[b + 6] = f7;
            f[b + 5] = f8;
            f[b + 4] = 0.0;
            f[b + 7] = 0.0;
            f[b + 8] = 0.0;
        }
        // Top (y=ny-1)
        {
            let cell = (ny - 1) * nx + x;
            let b = cell * 9;
            let f2 = f[b + 2];
            let f5 = f[b + 5];
            let f6 = f[b + 6];
            f[b + 4] = f2;
            f[b + 8] = f5;
            f[b + 7] = f6;
            f[b + 2] = 0.0;
            f[b + 5] = 0.0;
            f[b + 6] = 0.0;
        }
    }
    // Left / right columns
    for y in 0..ny {
        // Left (x=0)
        {
            let cell = y * nx;
            let b = cell * 9;
            let f3 = f[b + 3];
            let f6 = f[b + 6];
            let f7 = f[b + 7];
            f[b + 1] = f3;
            f[b + 5] = f6;
            f[b + 8] = f7;
            f[b + 3] = 0.0;
            f[b + 6] = 0.0;
            f[b + 7] = 0.0;
        }
        // Right (x=nx-1)
        {
            let cell = y * nx + (nx - 1);
            let b = cell * 9;
            let f1 = f[b + 1];
            let f5 = f[b + 5];
            let f8 = f[b + 8];
            f[b + 3] = f1;
            f[b + 6] = f5;
            f[b + 7] = f8;
            f[b + 1] = 0.0;
            f[b + 5] = 0.0;
            f[b + 8] = 0.0;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuLbm2D implementation
// ─────────────────────────────────────────────────────────────────────────────

impl GpuLbm2D {
    /// Create a new solver, initialised to the equilibrium distribution for
    /// uniform density `ρ = 1.0` and zero velocity.
    pub fn new(config: LbmConfig) -> Self {
        let nx = config.nx;
        let ny = config.ny;
        let n_cells = nx * ny;

        let mut f = vec![0.0_f64; n_cells * 9];
        let rho = vec![1.0_f64; n_cells];
        let u = vec![[0.0_f64; 2]; n_cells];

        // Initialise to equilibrium: f_q = w_q * rho_0
        for cell in 0..n_cells {
            let feq = equilibrium_f(1.0, 0.0, 0.0);
            let base = cell * 9;
            f[base..base + 9].copy_from_slice(&feq);
        }

        let f_buf = f.clone();
        let state = LbmState { f, rho, u, step: 0 };
        Self {
            config,
            state,
            f_buf,
        }
    }

    /// Create a solver initialised to the Poiseuille flow equilibrium.
    ///
    /// The x-velocity follows a parabolic profile:
    /// `ux(y) = max_velocity * 4 * y_norm * (1 − y_norm)`
    /// where `y_norm = y / (ny − 1)`.  Density is uniform, `uy = 0`.
    pub fn poiseuille_init(config: LbmConfig, max_velocity: f64) -> Self {
        let nx = config.nx;
        let ny = config.ny;
        let n_cells = nx * ny;

        let mut f = vec![0.0_f64; n_cells * 9];
        let mut rho = vec![1.0_f64; n_cells];
        let mut u = vec![[0.0_f64; 2]; n_cells];

        let ny_f = (ny.saturating_sub(1)).max(1) as f64;
        for y in 0..ny {
            let y_norm = y as f64 / ny_f;
            let ux = max_velocity * 4.0 * y_norm * (1.0 - y_norm);
            for x in 0..nx {
                let cell = y * nx + x;
                rho[cell] = 1.0;
                u[cell] = [ux, 0.0];
                let feq = equilibrium_f(1.0, ux, 0.0);
                let base = cell * 9;
                f[base..base + 9].copy_from_slice(&feq);
            }
        }

        let f_buf = f.clone();
        let state = LbmState { f, rho, u, step: 0 };
        Self {
            config,
            state,
            f_buf,
        }
    }

    /// Execute `n_steps` full LBM timesteps (collision + streaming).
    pub fn step(&mut self, n_steps: usize) {
        match self.config.dispatch {
            GpuLbmDispatch::Cpu => {
                for _ in 0..n_steps {
                    self.step_cpu();
                }
            }
            GpuLbmDispatch::CpuReference => {
                // CPU reference path: identical numerics to `Cpu`. Kept as a
                // distinct branch so a future GPU back-end can be validated
                // against it without changing call sites.
                for _ in 0..n_steps {
                    self.step_cpu();
                }
            }
        }
    }

    /// One LBM timestep on CPU.
    fn step_cpu(&mut self) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let tau = self.config.tau;

        // 1. Compute macroscopic fields
        compute_macroscopic(
            &self.state.f,
            &mut self.state.rho,
            &mut self.state.u,
            nx,
            ny,
        );

        // 2. BGK collision
        collide(
            &mut self.state.f,
            &self.state.rho,
            &self.state.u,
            tau,
            nx,
            ny,
        );

        // 3. Streaming (always periodic internally)
        std::mem::swap(&mut self.state.f, &mut self.f_buf);
        self.state.f.copy_from_slice(&self.f_buf);
        stream(&mut self.state.f, nx, ny);

        // 4. Apply boundary conditions
        match self.config.boundary {
            BoundaryCondition::Periodic => {
                // Streaming already periodic — nothing more to do
            }
            BoundaryCondition::NoSlip => {
                apply_noslip_walls(&mut self.state.f, nx, ny);
            }
            BoundaryCondition::FreeSlip => {
                apply_freeslip_walls(&mut self.state.f, nx, ny);
            }
        }

        self.state.step += 1;
    }

    /// Density field: `rho[y * nx + x]`.
    pub fn density(&self) -> &[f64] {
        &self.state.rho
    }

    /// Velocity field: `u[y * nx + x] = [ux, uy]`.
    pub fn velocity(&self) -> &[[f64; 2]] {
        &self.state.u
    }

    /// Total kinetic energy `Σ ρ |u|² / 2`.
    pub fn kinetic_energy(&self) -> f64 {
        self.state
            .rho
            .iter()
            .zip(self.state.u.iter())
            .map(|(&r, &[ux, uy])| 0.5 * r * (ux * ux + uy * uy))
            .sum()
    }

    /// Total mass `Σ ρ`.
    pub fn total_mass(&self) -> f64 {
        self.state.rho.iter().sum()
    }

    /// Relative deviations in mass (and x-momentum) from a reference mass.
    ///
    /// Returns `(|Δmass| / initial_mass, |Δmx| / initial_mass)`.
    pub fn conservation_check(&self, initial_mass: f64) -> (f64, f64) {
        let current_mass = self.total_mass();
        let mx: f64 = self
            .state
            .rho
            .iter()
            .zip(self.state.u.iter())
            .map(|(&r, &[ux, _])| r * ux)
            .sum();
        let mass_dev = (current_mass - initial_mass).abs() / initial_mass.abs().max(f64::EPSILON);
        let momentum_dev = mx.abs() / initial_mass.abs().max(f64::EPSILON);
        (mass_dev, momentum_dev)
    }

    /// Grid width.
    pub fn nx(&self) -> usize {
        self.config.nx
    }

    /// Grid height.
    pub fn ny(&self) -> usize {
        self.config.ny
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(nx: usize, ny: usize, tau: f64) -> LbmConfig {
        LbmConfig {
            nx,
            ny,
            tau,
            boundary: BoundaryCondition::Periodic,
            dispatch: GpuLbmDispatch::Cpu,
        }
    }

    /// Mass must be conserved to floating-point precision for periodic BC.
    #[test]
    fn test_mass_conservation_periodic() {
        let mut sim = GpuLbm2D::new(make_config(16, 16, 0.8));
        let m0 = sim.total_mass();
        sim.step(100);
        let m1 = sim.total_mass();
        assert!(
            (m1 - m0).abs() < 1e-10,
            "mass not conserved: Δm = {:.3e}",
            m1 - m0
        );
    }

    /// Poiseuille initialisation should give a parabolic x-velocity profile.
    #[test]
    fn test_poiseuille_profile() {
        let config = make_config(4, 20, 0.8);
        let max_v = 0.05;
        let sim = GpuLbm2D::poiseuille_init(config, max_v);
        // The midpoint (y = ny/2) should have the highest speed
        let ny = sim.ny();
        let nx = sim.nx();
        // At y = ny/2, ux ≈ max_v (parabola peak)
        let mid_y = ny / 2;
        let mid_cell = mid_y * nx;
        let ux_mid = sim.velocity()[mid_cell][0];
        // At y=0 and y=ny-1, ux = 0 (endpoints of parabola)
        let ux_bottom = sim.velocity()[0][0];
        let ux_top = sim.velocity()[(ny - 1) * nx][0];
        assert!(
            ux_mid > ux_bottom,
            "mid velocity {ux_mid} should exceed bottom {ux_bottom}"
        );
        assert!(
            ux_mid > ux_top,
            "mid velocity {ux_mid} should exceed top {ux_top}"
        );
        assert!(
            ux_mid > 0.0,
            "midpoint velocity should be positive, got {ux_mid}"
        );
    }

    /// Kinetic energy must be positive when the flow has non-zero velocity.
    #[test]
    fn test_kinetic_energy_positive() {
        let config = make_config(8, 8, 0.8);
        let max_v = 0.05;
        let sim = GpuLbm2D::poiseuille_init(config, max_v);
        let ke = sim.kinetic_energy();
        assert!(ke > 0.0, "kinetic energy should be positive, got {ke}");
    }

    /// After stepping, no cell should contain NaN or infinity.
    #[test]
    fn test_no_nan_after_step() {
        let mut sim = GpuLbm2D::new(make_config(8, 8, 0.8));
        sim.step(50);
        for &r in sim.density() {
            assert!(r.is_finite(), "density contains non-finite value: {r}");
        }
        for &[ux, uy] in sim.velocity() {
            assert!(
                ux.is_finite() && uy.is_finite(),
                "velocity contains non-finite values: [{ux}, {uy}]"
            );
        }
    }

    /// Equilibrium initialisation: all f[q] must be strictly positive.
    #[test]
    fn test_equilibrium_f_all_positive() {
        let sim = GpuLbm2D::new(make_config(4, 4, 0.8));
        let f = &sim.state.f;
        for (idx, &fq) in f.iter().enumerate() {
            assert!(
                fq > 0.0,
                "f[{idx}] = {fq} should be strictly positive at equilibrium"
            );
        }
    }

    /// The CPU reference dispatch produces the same result as the standard CPU
    /// dispatch (they must never diverge while both run CPU numerics).
    #[test]
    fn test_cpu_reference_matches_cpu() {
        let config_cpu = LbmConfig {
            nx: 8,
            ny: 8,
            tau: 0.8,
            boundary: BoundaryCondition::Periodic,
            dispatch: GpuLbmDispatch::Cpu,
        };
        let config_ref = LbmConfig {
            dispatch: GpuLbmDispatch::CpuReference,
            ..config_cpu.clone()
        };
        let mut sim_cpu = GpuLbm2D::new(config_cpu);
        let mut sim_ref = GpuLbm2D::new(config_ref);
        sim_cpu.step(20);
        sim_ref.step(20);
        let mass_cpu = sim_cpu.total_mass();
        let mass_ref = sim_ref.total_mass();
        assert!(
            (mass_cpu - mass_ref).abs() < 1e-12,
            "CPU and CPU-reference paths diverge: Δm = {:.3e}",
            mass_cpu - mass_ref
        );
    }

    /// NoSlip boundary: total mass is still conserved (bounce-back conserves mass).
    #[test]
    fn test_noslip_mass_conservation() {
        let config = LbmConfig {
            nx: 10,
            ny: 10,
            tau: 0.8,
            boundary: BoundaryCondition::NoSlip,
            dispatch: GpuLbmDispatch::Cpu,
        };
        let mut sim = GpuLbm2D::new(config);
        let m0 = sim.total_mass();
        sim.step(50);
        let m1 = sim.total_mass();
        assert!(
            (m1 - m0).abs() < 1e-8,
            "NoSlip mass not conserved: Δm = {:.3e}",
            m1 - m0
        );
    }

    /// Conservation check returns (near-zero, near-zero) for equilibrium init.
    #[test]
    fn test_conservation_check_equilibrium() {
        let mut sim = GpuLbm2D::new(make_config(8, 8, 0.8));
        let m0 = sim.total_mass();
        sim.step(30);
        let (dm, _dpx) = sim.conservation_check(m0);
        assert!(dm < 1e-8, "mass deviation too large: {dm:.3e}");
    }

    /// Accessor methods return sensible sizes.
    #[test]
    fn test_accessors() {
        let sim = GpuLbm2D::new(make_config(12, 10, 0.7));
        assert_eq!(sim.nx(), 12);
        assert_eq!(sim.ny(), 10);
        assert_eq!(sim.density().len(), 120);
        assert_eq!(sim.velocity().len(), 120);
    }
}
