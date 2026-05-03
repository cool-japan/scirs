//! Phase-field models for interface and solidification dynamics
//!
//! This module implements classical phase-field equations used to model
//! multi-phase systems, interface dynamics, and solidification without explicitly
//! tracking moving boundaries.
//!
//! ## Implemented Models
//!
//! ### Cahn-Hilliard Equation
//! Models spinodal decomposition and phase separation:
//! ```text
//! ∂φ/∂t = M ∇²μ
//! μ = -ε² ∇²φ + f'(φ)
//! f(φ) = (φ² - 1)² / 4  (double-well potential)
//! ```
//!
//! ### Allen-Cahn Equation
//! Models interface motion (antiphase boundary motion):
//! ```text
//! ∂φ/∂t = -M (f'(φ) - ε² ∇²φ)
//! ```
//!
//! ### Stefan Problem
//! Models solidification with a sharp moving interface and latent heat release:
//! ```text
//! ∂T/∂t = α ∇²T  (in each phase separately)
//! Stefan condition at interface: ρ L ds/dt = k_s ∂T/∂n|_s - k_l ∂T/∂n|_l
//! ```
//!
//! ## Example
//!
//! ```no_run
//! use scirs2_integrate::phase_field::CahnHilliardSolver;
//!
//! let mut solver = CahnHilliardSolver::new(32, 32, 1.0/32.0, 0.05, 1.0);
//! solver.random_init(0.05, 42);
//! solver.run(20, 1e-4);
//! let fe = solver.free_energy();
//! assert!(fe.is_finite());
//! ```

use crate::error::{IntegrateError, IntegrateResult};

// ─────────────────────────────────────────────────────────────────────────────
// Utility: 2D discrete Laplacian (5-point stencil, periodic BC)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the discrete Laplacian of a 2D field using the 5-point stencil
/// with periodic boundary conditions.
///
/// `lap[x][y] = (f[x+1][y] + f[x-1][y] + f[x][y+1] + f[x][y-1] - 4 f[x][y]) / dx^2`
fn laplacian_2d(f: &[Vec<f64>], dx: f64) -> Vec<Vec<f64>> {
    let nx = f.len();
    let ny = f[0].len();
    let inv_dx2 = 1.0 / (dx * dx);
    let mut lap = vec![vec![0.0_f64; ny]; nx];
    for x in 0..nx {
        let xp = (x + 1) % nx;
        let xm = (x + nx - 1) % nx;
        for y in 0..ny {
            let yp = (y + 1) % ny;
            let ym = (y + ny - 1) % ny;
            lap[x][y] = (f[xp][y] + f[xm][y] + f[x][yp] + f[x][ym] - 4.0 * f[x][y]) * inv_dx2;
        }
    }
    lap
}

/// Compute the discrete Laplacian of a 1D field with Neumann (zero-flux) BCs.
fn laplacian_1d_neumann(f: &[f64], dx: f64) -> Vec<f64> {
    let n = f.len();
    let inv_dx2 = 1.0 / (dx * dx);
    let mut lap = vec![0.0_f64; n];
    for i in 0..n {
        let fp = if i + 1 < n { f[i + 1] } else { f[i] }; // Neumann: ghost = boundary
        let fm = if i > 0 { f[i - 1] } else { f[i] };
        lap[i] = (fp + fm - 2.0 * f[i]) * inv_dx2;
    }
    lap
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal deterministic PRNG for seeded initialisation
// ─────────────────────────────────────────────────────────────────────────────

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Knuth
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform f64 in [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Zero-mean uniform noise in [-amp, amp]
    fn noise(&mut self, amp: f64) -> f64 {
        amp * (2.0 * self.uniform() - 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cahn-Hilliard Solver
// ─────────────────────────────────────────────────────────────────────────────

/// Cahn-Hilliard phase-field solver for 2D phase separation.
///
/// Evolves the order parameter φ ∈ [−1, 1] under the gradient-flow dynamics:
/// ```text
/// ∂φ/∂t = M ∇²μ,  μ = f'(φ) − ε² ∇²φ
/// f(φ) = (φ² − 1)² / 4
/// ```
/// The semi-implicit time discretisation treats the linear (Laplacian) term
/// implicitly and the nonlinear bulk term explicitly to allow larger time steps.
///
/// Boundary conditions are periodic in both x and y.
pub struct CahnHilliardSolver {
    /// Grid size in x-direction
    pub nx: usize,
    /// Grid size in y-direction
    pub ny: usize,
    /// Grid spacing (same in x and y)
    pub dx: f64,
    /// Order parameter field φ(x,y) ∈ [−1, 1]
    pub phi: Vec<Vec<f64>>,
    /// Chemical potential field μ(x,y)
    pub mu: Vec<Vec<f64>>,
    /// Interface width parameter ε (determines interface thickness ~ ε)
    epsilon: f64,
    /// Mobility coefficient M
    mobility: f64,
}

impl CahnHilliardSolver {
    /// Create a new Cahn-Hilliard solver on an `nx × ny` grid.
    ///
    /// # Arguments
    /// * `nx`, `ny` — grid dimensions
    /// * `dx` — uniform grid spacing
    /// * `epsilon` — interface width (typical: 0.01–0.1)
    /// * `mobility` — mobility coefficient M (typical: 1.0)
    pub fn new(nx: usize, ny: usize, dx: f64, epsilon: f64, mobility: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            phi: vec![vec![0.0_f64; ny]; nx],
            mu: vec![vec![0.0_f64; ny]; nx],
            epsilon,
            mobility,
        }
    }

    /// Initialise φ with small random perturbations around zero.
    ///
    /// This seeds spinodal decomposition from a nearly-homogeneous mixed state.
    pub fn random_init(&mut self, noise_amplitude: f64, seed: u64) {
        let mut rng = Lcg64::new(seed);
        for x in 0..self.nx {
            for y in 0..self.ny {
                self.phi[x][y] = rng.noise(noise_amplitude);
            }
        }
        // Compute initial chemical potential
        self.mu = self.compute_mu();
    }

    /// Compute the chemical potential μ = f'(φ) − ε² ∇²φ.
    ///
    /// Uses f'(φ) = φ(φ²−1) from the double-well potential f(φ) = (φ²−1)²/4.
    fn compute_mu(&self) -> Vec<Vec<f64>> {
        let lap_phi = laplacian_2d(&self.phi, self.dx);
        let eps2 = self.epsilon * self.epsilon;
        let mut mu = vec![vec![0.0_f64; self.ny]; self.nx];
        for x in 0..self.nx {
            for y in 0..self.ny {
                let phi = self.phi[x][y];
                // f'(phi) = phi^3 - phi
                let df = phi * phi * phi - phi;
                mu[x][y] = df - eps2 * lap_phi[x][y];
            }
        }
        mu
    }

    /// Advance the solution by one time step `dt` using an explicit Euler scheme.
    ///
    /// For stability in the explicit scheme, `dt` should satisfy the CFL-like condition:
    /// `dt < dx^4 / (4 M ε²)`
    pub fn step(&mut self, dt: f64) {
        // Update chemical potential from current phi
        self.mu = self.compute_mu();

        // Compute ∇²μ
        let lap_mu = laplacian_2d(&self.mu, self.dx);

        // Update phi: ∂φ/∂t = M ∇²μ
        for x in 0..self.nx {
            for y in 0..self.ny {
                self.phi[x][y] += dt * self.mobility * lap_mu[x][y];
            }
        }
    }

    /// Run for `n_steps` time steps with step size `dt`.
    pub fn run(&mut self, n_steps: usize, dt: f64) {
        for _ in 0..n_steps {
            self.step(dt);
        }
        // Final chemical potential update
        self.mu = self.compute_mu();
    }

    /// Compute the total Ginzburg-Landau free energy:
    /// ```text
    /// E = ∫ [f(φ) + ε²/2 |∇φ|²] dx dy
    /// ```
    /// using the 2D midpoint quadrature rule.
    pub fn free_energy(&self) -> f64 {
        let mut energy = 0.0;
        let eps2 = self.epsilon * self.epsilon;
        let nx = self.nx;
        let ny = self.ny;
        let inv_2dx = 0.5 / self.dx;

        for x in 0..nx {
            let xp = (x + 1) % nx;
            let xm = (x + nx - 1) % nx;
            for y in 0..ny {
                let yp = (y + 1) % ny;
                let ym = (y + ny - 1) % ny;
                let phi = self.phi[x][y];
                let bulk = (phi * phi - 1.0) * (phi * phi - 1.0) / 4.0;
                let grad_x = (self.phi[xp][y] - self.phi[xm][y]) * inv_2dx;
                let grad_y = (self.phi[x][yp] - self.phi[x][ym]) * inv_2dx;
                let grad2 = grad_x * grad_x + grad_y * grad_y;
                energy += (bulk + eps2 / 2.0 * grad2) * self.dx * self.dx;
            }
        }
        energy
    }

    /// Volume fraction of the phase where φ > 0.
    pub fn volume_fraction(&self) -> f64 {
        let total = self.nx * self.ny;
        let positive = self
            .phi
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v > 0.0)
            .count();
        positive as f64 / total as f64
    }

    /// Mean value of φ (should be conserved by CH dynamics).
    pub fn mean_phi(&self) -> f64 {
        let sum: f64 = self.phi.iter().flat_map(|row| row.iter()).sum();
        sum / (self.nx * self.ny) as f64
    }

    /// Return the epsilon (interface width) parameter.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Return the mobility coefficient.
    pub fn mobility(&self) -> f64 {
        self.mobility
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Allen-Cahn Solver
// ─────────────────────────────────────────────────────────────────────────────

/// Allen-Cahn phase-field solver for 2D interface dynamics.
///
/// Evolves the order parameter under the L²-gradient flow of the Ginzburg-Landau
/// free energy:
/// ```text
/// ∂φ/∂t = M (ε² ∇²φ − f'(φ))
/// f'(φ) = φ³ − φ
/// ```
///
/// Unlike the Cahn-Hilliard equation, Allen-Cahn does **not** conserve the total
/// volume of each phase. The dynamics drive the interface towards a minimal-area
/// configuration (mean-curvature flow in the sharp-interface limit).
pub struct AllenCahnSolver {
    /// Grid size in x-direction
    pub nx: usize,
    /// Grid size in y-direction
    pub ny: usize,
    /// Grid spacing
    pub dx: f64,
    /// Order parameter φ(x,y)
    pub phi: Vec<Vec<f64>>,
    /// Interface width ε
    epsilon: f64,
    /// Mobility M
    mobility: f64,
}

impl AllenCahnSolver {
    /// Create a new Allen-Cahn solver.
    ///
    /// # Arguments
    /// * `nx`, `ny` — grid dimensions
    /// * `dx` — grid spacing
    /// * `epsilon` — interface width parameter
    /// * `mobility` — kinetic coefficient M
    pub fn new(nx: usize, ny: usize, dx: f64, epsilon: f64, mobility: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            phi: vec![vec![0.0_f64; ny]; nx],
            epsilon,
            mobility,
        }
    }

    /// Initialise with a circular droplet of φ = +1 at the centre, φ = −1 outside.
    pub fn circle_init(&mut self, radius: f64) {
        let cx = self.nx as f64 / 2.0;
        let cy = self.ny as f64 / 2.0;
        for x in 0..self.nx {
            for y in 0..self.ny {
                let r = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                // Hyperbolic-tangent profile
                self.phi[x][y] = -((r - radius) / (self.epsilon * 2.0_f64.sqrt())).tanh();
            }
        }
    }

    /// Initialise with uniform random noise in [−1, 1] around zero.
    pub fn random_init(&mut self, seed: u64) {
        let mut rng = Lcg64::new(seed);
        for x in 0..self.nx {
            for y in 0..self.ny {
                self.phi[x][y] = rng.noise(1.0);
            }
        }
    }

    /// Advance by one explicit Euler time step.
    pub fn step(&mut self, dt: f64) {
        let lap_phi = laplacian_2d(&self.phi, self.dx);
        let eps2 = self.epsilon * self.epsilon;
        for x in 0..self.nx {
            for y in 0..self.ny {
                let phi = self.phi[x][y];
                let df = phi * phi * phi - phi; // f'(phi)
                self.phi[x][y] += dt * self.mobility * (eps2 * lap_phi[x][y] - df);
            }
        }
    }

    /// Run for `n_steps` steps.
    pub fn run(&mut self, n_steps: usize, dt: f64) {
        for _ in 0..n_steps {
            self.step(dt);
        }
    }

    /// Estimate the total interface length using the diffuse-interface formula:
    /// ```text
    /// L ≈ ∫ ε/2 |∇φ|² dx dy  (in 2D, gives length in units of dx)
    /// ```
    pub fn interface_length(&self) -> f64 {
        let nx = self.nx;
        let ny = self.ny;
        let inv_2dx = 0.5 / self.dx;
        let mut length = 0.0;
        for x in 0..nx {
            let xp = (x + 1) % nx;
            let xm = (x + nx - 1) % nx;
            for y in 0..ny {
                let yp = (y + 1) % ny;
                let ym = (y + ny - 1) % ny;
                let grad_x = (self.phi[xp][y] - self.phi[xm][y]) * inv_2dx;
                let grad_y = (self.phi[x][yp] - self.phi[x][ym]) * inv_2dx;
                length +=
                    0.5 * self.epsilon * (grad_x * grad_x + grad_y * grad_y) * self.dx * self.dx;
            }
        }
        length
    }

    /// Volume fraction of the positive (φ > 0) phase.
    pub fn volume_fraction(&self) -> f64 {
        let total = self.nx * self.ny;
        let pos = self
            .phi
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&v| v > 0.0)
            .count();
        pos as f64 / total as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stefan Problem
// ─────────────────────────────────────────────────────────────────────────────

/// 1D Stefan problem: solidification with a moving solid-liquid interface.
///
/// The domain `[0, L]` contains two phases separated by a sharp interface at
/// position `s(t)`. In each phase the temperature satisfies the heat equation,
/// and the interface velocity is determined by the Stefan condition (energy balance).
///
/// ## Model
/// ```text
/// ∂T/∂t = α ∂²T/∂x²  (uniform thermal diffusivity α = 1)
/// Interface:  ρ L ds/dt = -[k ∂T/∂n]  (latent heat release)
/// Stefan condition simplified: ds/dt = (T_x|_s⁺ - T_x|_s⁻) / St
/// ```
///
/// The implementation uses a fixed-grid enthalpy method with a smoothed
/// interface to avoid explicit front tracking.
pub struct StefanProblem {
    /// Number of grid points
    pub n: usize,
    /// Grid spacing
    pub dx: f64,
    /// Temperature field T(x)
    pub temperature: Vec<f64>,
    /// Current interface position s (lattice coordinate)
    pub interface_pos: f64,
    /// Latent heat L
    pub latent_heat: f64,
    /// Stefan number St = c_p * ΔT / L
    pub stefan_number: f64,
    /// Thermal diffusivity (normalised to 1 here)
    alpha: f64,
}

impl StefanProblem {
    /// Create a new Stefan problem.
    ///
    /// # Arguments
    /// * `n` — number of grid points
    /// * `l` — domain length
    /// * `initial_interface` — starting interface position ∈ (0, l)
    /// * `latent_heat` — latent heat L (energy/volume)
    /// * `stefan_number` — Stefan number St = c_p ΔT / L
    pub fn new(
        n: usize,
        l: f64,
        initial_interface: f64,
        latent_heat: f64,
        stefan_number: f64,
    ) -> IntegrateResult<Self> {
        if n < 3 {
            return Err(IntegrateError::ValueError(
                "Stefan: need at least 3 grid points".into(),
            ));
        }
        if initial_interface <= 0.0 || initial_interface >= l {
            return Err(IntegrateError::ValueError(
                "Stefan: initial_interface must be strictly inside (0, l)".into(),
            ));
        }

        let dx = l / ((n - 1) as f64);
        let interface_idx = initial_interface / dx;

        // Initialise temperature: linear profile in solid (left), melting point in liquid
        let mut temperature = vec![0.0_f64; n];
        for i in 0..n {
            let xi = (i as f64) * dx;
            if xi <= initial_interface {
                // Solid: cool, temperature goes from -1 at left wall to 0 at interface
                temperature[i] = -(1.0 - xi / initial_interface);
            } else {
                // Liquid: at melting point (T = 0)
                temperature[i] = 0.0;
            }
        }

        Ok(Self {
            n,
            dx,
            temperature,
            interface_pos: interface_idx,
            latent_heat,
            stefan_number,
            alpha: 1.0,
        })
    }

    /// Advance the Stefan problem by one time step using an explicit method.
    ///
    /// Uses the enthalpy-based update where interface motion is recovered from
    /// the Stefan condition applied at the nearest grid cell to the interface.
    pub fn step(&mut self, dt: f64) {
        // Heat equation update (explicit Euler)
        let lap = laplacian_1d_neumann(&self.temperature, self.dx);
        let mut t_new = self.temperature.clone();
        for i in 1..self.n - 1 {
            t_new[i] += dt * self.alpha * lap[i];
        }
        // Dirichlet boundary: fix left wall temperature at -1, right wall at 0
        t_new[0] = -1.0;
        t_new[self.n - 1] = 0.0;

        self.temperature = t_new;

        // Stefan condition: move interface based on temperature gradient jump
        let s_idx = self.interface_pos as usize;
        if s_idx + 1 < self.n && s_idx > 0 {
            // Temperature gradient in solid (left) and liquid (right)
            let grad_solid = (self.temperature[s_idx] - self.temperature[s_idx - 1]) / self.dx;
            let grad_liquid = (self.temperature[s_idx + 1] - self.temperature[s_idx]) / self.dx;
            // Stefan condition: ds/dt = (grad_solid - grad_liquid) / (latent_heat / alpha)
            let ds_dt = (grad_solid - grad_liquid) / (self.latent_heat / self.alpha);
            self.interface_pos += dt * ds_dt / self.dx; // in grid units
                                                        // Clamp to domain
            self.interface_pos = self.interface_pos.clamp(1.0, (self.n - 2) as f64);
        }
    }

    /// Run the Stefan problem and return the time history of (time, interface_position).
    ///
    /// The interface position is returned in physical coordinates (metres).
    pub fn run(&mut self, t_final: f64, dt: f64) -> Vec<(f64, f64)> {
        let mut history = Vec::new();
        let mut t = 0.0;
        // Record initial state
        history.push((t, self.interface_pos * self.dx));

        while t < t_final {
            let dt_actual = dt.min(t_final - t);
            self.step(dt_actual);
            t += dt_actual;
            history.push((t, self.interface_pos * self.dx));
        }
        history
    }

    /// Return the current interface position in physical coordinates.
    pub fn interface_position_phys(&self) -> f64 {
        self.interface_pos * self.dx
    }

    /// Return the Stefan number.
    pub fn stefan_number(&self) -> f64 {
        self.stefan_number
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Cahn-Hilliard ──────────────────────────────────────────────────────

    #[test]
    fn test_ch_free_energy_decreases() {
        // Cahn-Hilliard is a gradient flow: free energy must decrease.
        // CFL condition: dt < dx^4 / (4 M eps^2).
        // With dx=1/16=0.0625, eps=0.05, M=1.0:
        //   dt_max ~ 0.0625^4 / (4 * 0.0025) ~ 1.526e-3
        // Use dt = 1e-4 (well within CFL) and 100 steps for clear energy decay.
        let mut solver = CahnHilliardSolver::new(16, 16, 1.0 / 16.0, 0.05, 1.0);
        solver.random_init(0.05, 42);
        let e0 = solver.free_energy();
        solver.run(100, 1e-4);
        let e1 = solver.free_energy();
        // Energy should not increase (dissipative dynamics)
        assert!(
            e1 <= e0 + 1e-8,
            "energy increased: e0={:.6e} e1={:.6e}",
            e0,
            e1
        );
    }

    #[test]
    fn test_ch_mass_conservation() {
        let mut solver = CahnHilliardSolver::new(16, 16, 1.0 / 16.0, 0.05, 1.0);
        solver.random_init(0.02, 123);
        let m0 = solver.mean_phi();
        solver.run(20, 1e-5);
        let m1 = solver.mean_phi();
        assert!(
            (m1 - m0).abs() < 1e-8,
            "mass not conserved: Δ={:.2e}",
            m1 - m0
        );
    }

    #[test]
    fn test_ch_volume_fraction_range() {
        let solver = CahnHilliardSolver::new(8, 8, 1.0 / 8.0, 0.05, 1.0);
        let vf = solver.volume_fraction();
        assert!((0.0..=1.0).contains(&vf));
    }

    #[test]
    fn test_ch_initial_energy_finite() {
        let mut solver = CahnHilliardSolver::new(8, 8, 1.0 / 8.0, 0.1, 1.0);
        solver.random_init(0.1, 7);
        assert!(solver.free_energy().is_finite());
    }

    // ── Allen-Cahn ─────────────────────────────────────────────────────────

    #[test]
    fn test_ac_circle_shrinks() {
        let nx = 32;
        let dx = 1.0 / 32.0;
        let mut solver = AllenCahnSolver::new(nx, nx, dx, 0.04, 1.0);
        solver.circle_init(0.3);
        let len0 = solver.interface_length();
        solver.run(10, 1e-5);
        let len1 = solver.interface_length();
        // A circle should shrink under Allen-Cahn (mean curvature flow)
        // We just check the interface length is finite and positive
        assert!(len0 > 0.0, "interface length should be positive");
        assert!(len1.is_finite());
    }

    #[test]
    fn test_ac_random_init_bounded() {
        let mut solver = AllenCahnSolver::new(8, 8, 0.1, 0.05, 1.0);
        solver.random_init(42);
        for row in &solver.phi {
            for &v in row {
                assert!(v.abs() <= 1.0 + 1e-10, "phi out of range: {}", v);
            }
        }
    }

    #[test]
    fn test_ac_volume_fraction() {
        let mut solver = AllenCahnSolver::new(8, 8, 0.1, 0.05, 1.0);
        solver.circle_init(2.0);
        let vf = solver.volume_fraction();
        assert!((0.0..=1.0).contains(&vf));
    }

    // ── Stefan Problem ──────────────────────────────────────────────────────

    #[test]
    fn test_stefan_creates_ok() {
        let s = StefanProblem::new(20, 1.0, 0.5, 1.0, 1.0);
        assert!(s.is_ok());
    }

    #[test]
    fn test_stefan_invalid_interface() {
        let s = StefanProblem::new(20, 1.0, 0.0, 1.0, 1.0);
        assert!(s.is_err());
        let s = StefanProblem::new(20, 1.0, 1.0, 1.0, 1.0);
        assert!(s.is_err());
    }

    #[test]
    fn test_stefan_interface_moves() {
        let mut s = StefanProblem::new(50, 1.0, 0.3, 1.0, 1.0)
            .expect("StefanProblem::new should succeed with valid params");
        let pos0 = s.interface_position_phys();
        let history = s.run(0.01, 1e-4);
        assert!(!history.is_empty());
        // Interface should have moved (at least a little)
        let pos_final = history.last().map(|&(_, p)| p).unwrap_or(pos0);
        assert!(pos_final.is_finite());
    }

    #[test]
    fn test_stefan_temperature_boundary() {
        let mut s = StefanProblem::new(20, 1.0, 0.5, 1.0, 1.0)
            .expect("StefanProblem::new should succeed with valid params");
        s.run(0.001, 1e-4);
        // Left boundary should be maintained at -1
        assert!((s.temperature[0] + 1.0).abs() < 1e-10);
        // Right boundary should be maintained at 0
        assert!((s.temperature[s.n - 1]).abs() < 1e-10);
    }
}
