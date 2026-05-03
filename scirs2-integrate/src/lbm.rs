//! Lattice Boltzmann Method (LBM) for computational fluid dynamics
//!
//! This module implements the Lattice Boltzmann Method (LBM), a mesoscopic approach
//! to fluid simulation based on kinetic theory. Unlike traditional CFD methods that
//! discretize the Navier-Stokes equations directly, LBM evolves distribution functions
//! on a lattice and recovers macroscopic fluid behavior through moments.
//!
//! ## Implemented Models
//!
//! - **D2Q9**: 2D incompressible flow on a 9-velocity square lattice
//! - **D3Q19**: 3D flow on a 19-velocity cubic lattice
//!
//! ## Physics
//!
//! The LBM BGK collision operator is:
//! ```text
//! f_i(x + c_i * dt, t + dt) = f_i(x, t) - omega * (f_i - f_i^eq)
//! ```
//! where the equilibrium distribution is:
//! ```text
//! f_i^eq = w_i * rho * [1 + (c_i · u)/cs^2 + (c_i · u)^2/(2 cs^4) - u^2/(2 cs^2)]
//! ```
//!
//! ## Example
//!
//! ```rust
//! use scirs2_integrate::lbm::{D2Q9Lbm, BoundaryType};
//!
//! // Lid-driven cavity at Re = 100
//! let mut lbm = D2Q9Lbm::new(32, 32, 0.01);
//!
//! // Moving lid (top wall)
//! for x in 0..32 {
//!     lbm.set_boundary(x, 31, BoundaryType::Inlet { ux: 0.1, uy: 0.0 });
//! }
//!
//! lbm.run(100);
//! let re = lbm.reynolds_number(32.0, 0.1);
//! assert!(re > 0.0);
//! ```

use crate::error::{IntegrateError, IntegrateResult};

// ─────────────────────────────────────────────────────────────────────────────
// D2Q9 Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Number of velocity directions in D2Q9
const Q9: usize = 9;

/// D2Q9 lattice velocities: (cx, cy)
const CX9: [i32; Q9] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const CY9: [i32; Q9] = [0, 0, 1, 0, -1, 1, 1, -1, -1];

/// D2Q9 weights
const W9: [f64; Q9] = [
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

/// Bounce-back opposite direction indices for D2Q9
/// opposite[i] gives the index of the direction opposite to i
const OPP9: [usize; Q9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

// ─────────────────────────────────────────────────────────────────────────────
// D3Q19 Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Number of velocity directions in D3Q19
const Q19: usize = 19;

/// D3Q19 lattice velocities
const CX19: [i32; Q19] = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
const CY19: [i32; Q19] = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1];
const CZ19: [i32; Q19] = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1];

/// D3Q19 weights
const W19: [f64; Q19] = [
    1.0 / 3.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// Bounce-back opposite direction indices for D3Q19
const OPP19: [usize; Q19] = [
    0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
];

// Speed of sound squared in lattice units (cs^2 = 1/3)
const CS2: f64 = 1.0 / 3.0;

// ─────────────────────────────────────────────────────────────────────────────
// Boundary condition types
// ─────────────────────────────────────────────────────────────────────────────

/// Boundary condition type for D2Q9 LBM
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BoundaryType {
    /// Interior fluid node — standard BGK collision
    Fluid,
    /// No-slip solid wall using bounce-back
    Wall,
    /// Inlet with prescribed velocity (ux, uy)
    Inlet {
        /// x-component of inlet velocity (lattice units)
        ux: f64,
        /// y-component of inlet velocity (lattice units)
        uy: f64,
    },
    /// Outflow / open boundary (zero-gradient extrapolation)
    Outlet,
}

/// Boundary condition type for D3Q19 LBM
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BoundaryType3D {
    /// Interior fluid node
    Fluid,
    /// No-slip solid wall
    Wall,
    /// Inlet with prescribed velocity vector [ux, uy, uz]
    Inlet([f64; 3]),
    /// Outflow boundary
    Outlet,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute D2Q9 equilibrium distribution for one direction `i`
#[inline]
fn feq9(i: usize, rho: f64, ux: f64, uy: f64) -> f64 {
    let cu = (CX9[i] as f64) * ux + (CY9[i] as f64) * uy;
    let u2 = ux * ux + uy * uy;
    W9[i] * rho * (1.0 + cu / CS2 + cu * cu / (2.0 * CS2 * CS2) - u2 / (2.0 * CS2))
}

/// Compute D3Q19 equilibrium distribution for one direction `i`
#[inline]
fn feq19(i: usize, rho: f64, ux: f64, uy: f64, uz: f64) -> f64 {
    let cu = (CX19[i] as f64) * ux + (CY19[i] as f64) * uy + (CZ19[i] as f64) * uz;
    let u2 = ux * ux + uy * uy + uz * uz;
    W19[i] * rho * (1.0 + cu / CS2 + cu * cu / (2.0 * CS2 * CS2) - u2 / (2.0 * CS2))
}

// ─────────────────────────────────────────────────────────────────────────────
// D2Q9 Lattice Boltzmann
// ─────────────────────────────────────────────────────────────────────────────

/// D2Q9 Lattice Boltzmann solver for 2D incompressible flow.
///
/// Uses the BGK (Bhatnagar-Gross-Krook) single-relaxation-time collision operator.
/// The computational lattice has `nx` columns and `ny` rows with periodic boundaries
/// unless overridden by explicit boundary conditions.
///
/// ## Indexing
///
/// All 2D arrays use `[x][y]` indexing where `x ∈ [0, nx)` and `y ∈ [0, ny)`.
/// Distribution functions use `[direction][x][y]` ordering.
pub struct D2Q9Lbm {
    /// Number of lattice nodes in x-direction
    pub nx: usize,
    /// Number of lattice nodes in y-direction
    pub ny: usize,
    /// Relaxation frequency ω = 1/τ, related to viscosity: ν = cs² (τ - 0.5)
    pub omega: f64,
    /// Distribution functions f[direction][x][y], 9 directions
    f: Vec<Vec<Vec<f64>>>,
    /// Temporary buffer for streaming step
    f_buf: Vec<Vec<Vec<f64>>>,
    /// Macroscopic density ρ(x,y)
    pub density: Vec<Vec<f64>>,
    /// Macroscopic velocity x-component u_x(x,y)
    pub velocity_x: Vec<Vec<f64>>,
    /// Macroscopic velocity y-component u_y(x,y)
    pub velocity_y: Vec<Vec<f64>>,
    /// Boundary condition type at each lattice node
    boundary: Vec<Vec<BoundaryType>>,
    /// Kinematic viscosity (lattice units)
    viscosity: f64,
}

impl D2Q9Lbm {
    /// Create a new D2Q9 LBM solver.
    ///
    /// # Arguments
    /// * `nx` — grid width in lattice units
    /// * `ny` — grid height in lattice units
    /// * `viscosity` — kinematic viscosity ν in lattice units (e.g., 0.1)
    ///
    /// # Panics
    /// Does not panic; returns an initialized solver with uniform density ρ=1 and zero velocity.
    pub fn new(nx: usize, ny: usize, viscosity: f64) -> Self {
        let tau = 3.0 * viscosity + 0.5;
        let omega = 1.0 / tau;

        let rho0 = 1.0_f64;
        let ux0 = 0.0_f64;
        let uy0 = 0.0_f64;

        // Initialise equilibrium distributions
        let mut f = vec![vec![vec![0.0_f64; ny]; nx]; Q9];
        for i in 0..Q9 {
            for x in 0..nx {
                for y in 0..ny {
                    f[i][x][y] = feq9(i, rho0, ux0, uy0);
                }
            }
        }

        Self {
            nx,
            ny,
            omega,
            f_buf: f.clone(),
            f,
            density: vec![vec![rho0; ny]; nx],
            velocity_x: vec![vec![ux0; ny]; nx],
            velocity_y: vec![vec![uy0; ny]; nx],
            boundary: vec![vec![BoundaryType::Fluid; ny]; nx],
            viscosity,
        }
    }

    /// Set the boundary condition at node (x, y).
    pub fn set_boundary(&mut self, x: usize, y: usize, bc: BoundaryType) {
        if x < self.nx && y < self.ny {
            self.boundary[x][y] = bc;
        }
    }

    /// Recompute macroscopic fields (density and velocity) from distribution functions.
    fn update_macroscopic(&mut self) {
        for x in 0..self.nx {
            for y in 0..self.ny {
                let mut rho = 0.0;
                let mut ux = 0.0;
                let mut uy = 0.0;
                for i in 0..Q9 {
                    let fi = self.f[i][x][y];
                    rho += fi;
                    ux += (CX9[i] as f64) * fi;
                    uy += (CY9[i] as f64) * fi;
                }
                if rho > 1e-15 {
                    self.density[x][y] = rho;
                    self.velocity_x[x][y] = ux / rho;
                    self.velocity_y[x][y] = uy / rho;
                }
            }
        }
    }

    /// BGK collision step: relax distribution towards equilibrium.
    fn collide(&mut self) {
        for x in 0..self.nx {
            for y in 0..self.ny {
                if self.boundary[x][y] == BoundaryType::Wall {
                    continue; // no collision on wall nodes
                }
                let rho = self.density[x][y];
                let ux = self.velocity_x[x][y];
                let uy = self.velocity_y[x][y];
                for i in 0..Q9 {
                    let feqi = feq9(i, rho, ux, uy);
                    self.f[i][x][y] += self.omega * (feqi - self.f[i][x][y]);
                }
            }
        }
    }

    /// Apply inlet boundary condition using regularised bounce-back.
    /// Uses equilibrium distribution with prescribed velocity.
    fn apply_inlet_bc(&mut self, x: usize, y: usize, ux: f64, uy: f64) {
        // Compute density from non-equilibrium reflection
        let rho = 1.0; // assume uniform density at inlet
        for i in 0..Q9 {
            self.f[i][x][y] = feq9(i, rho, ux, uy);
        }
    }

    /// Stream distributions along lattice velocities, then apply boundary conditions.
    fn stream_and_bc(&mut self) {
        let nx = self.nx as i64;
        let ny = self.ny as i64;

        // Streaming: propagate distributions
        for i in 0..Q9 {
            let cx = CX9[i];
            let cy = CY9[i];
            for x in 0..self.nx {
                for y in 0..self.ny {
                    let xd = ((x as i64 - cx as i64 + nx) % nx) as usize;
                    let yd = ((y as i64 - cy as i64 + ny) % ny) as usize;
                    self.f_buf[i][x][y] = self.f[i][xd][yd];
                }
            }
        }

        // Swap buffers
        std::mem::swap(&mut self.f, &mut self.f_buf);

        // Boundary conditions (post-streaming)
        for x in 0..self.nx {
            for y in 0..self.ny {
                match self.boundary[x][y] {
                    BoundaryType::Wall => {
                        // Full-way bounce-back
                        for i in 0..Q9 {
                            self.f[i][x][y] = self.f_buf[OPP9[i]][x][y];
                        }
                    }
                    BoundaryType::Inlet { ux, uy } => {
                        self.apply_inlet_bc(x, y, ux, uy);
                    }
                    BoundaryType::Outlet => {
                        // Zero-gradient: copy from interior neighbour
                        // Use xd = clamp(x ± 1) depending on position
                        let xn = if x > 0 { x - 1 } else { 1 };
                        for i in 0..Q9 {
                            self.f[i][x][y] = self.f[i][xn][y];
                        }
                    }
                    BoundaryType::Fluid => {}
                }
            }
        }
    }

    /// Execute one complete LBM time step: macroscopic update → collision → stream + BC.
    pub fn step(&mut self) {
        self.update_macroscopic();
        self.collide();
        self.stream_and_bc();
    }

    /// Run the simulation for `n_steps` time steps.
    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
        // Final macroscopic update so fields are current
        self.update_macroscopic();
    }

    /// Compute the Reynolds number Re = L_ref * U_ref / ν.
    pub fn reynolds_number(&self, l_ref: f64, u_ref: f64) -> f64 {
        if self.viscosity.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        l_ref * u_ref / self.viscosity
    }

    /// Compute total mass (sum of all density values) — conserved for periodic BC.
    pub fn total_mass(&self) -> f64 {
        let mut mass = 0.0;
        for x in 0..self.nx {
            for y in 0..self.ny {
                mass += self.density[x][y];
            }
        }
        mass
    }

    /// Compute the vorticity field ∂u_y/∂x − ∂u_x/∂y using central differences.
    pub fn vorticity(&self) -> Vec<Vec<f64>> {
        let mut vort = vec![vec![0.0_f64; self.ny]; self.nx];
        let nx = self.nx;
        let ny = self.ny;
        for x in 0..nx {
            for y in 0..ny {
                let xp = (x + 1) % nx;
                let xm = (x + nx - 1) % nx;
                let yp = (y + 1) % ny;
                let ym = (y + ny - 1) % ny;
                let duy_dx = (self.velocity_y[xp][y] - self.velocity_y[xm][y]) * 0.5;
                let dux_dy = (self.velocity_x[x][yp] - self.velocity_x[x][ym]) * 0.5;
                vort[x][y] = duy_dx - dux_dy;
            }
        }
        vort
    }

    /// Compute the pressure field p = ρ * cs² (ideal gas in lattice units).
    pub fn pressure(&self) -> Vec<Vec<f64>> {
        let mut p = vec![vec![0.0_f64; self.ny]; self.nx];
        for x in 0..self.nx {
            for y in 0..self.ny {
                p[x][y] = self.density[x][y] * CS2;
            }
        }
        p
    }

    /// Get kinematic viscosity in lattice units.
    pub fn viscosity(&self) -> f64 {
        self.viscosity
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D3Q19 Lattice Boltzmann
// ─────────────────────────────────────────────────────────────────────────────

/// D3Q19 Lattice Boltzmann solver for 3D incompressible flow.
///
/// Uses the BGK single-relaxation-time operator on a 19-velocity cubic lattice.
/// Distribution functions use `[direction][x][y][z]` indexing.
pub struct D3Q19Lbm {
    /// Grid size in x-direction
    pub nx: usize,
    /// Grid size in y-direction
    pub ny: usize,
    /// Grid size in z-direction
    pub nz: usize,
    /// Relaxation frequency
    pub omega: f64,
    /// Distribution functions f[direction][x][y][z], 19 directions
    f: Vec<Vec<Vec<Vec<f64>>>>,
    /// Streaming buffer
    f_buf: Vec<Vec<Vec<Vec<f64>>>>,
    /// Macroscopic density ρ(x,y,z)
    pub density: Vec<Vec<Vec<f64>>>,
    /// Macroscopic velocity [ux, uy, uz](x,y,z)
    pub velocity: Vec<Vec<Vec<[f64; 3]>>>,
    /// Boundary condition at each node
    boundary: Vec<Vec<Vec<BoundaryType3D>>>,
    /// Kinematic viscosity
    viscosity: f64,
}

impl D3Q19Lbm {
    /// Create a new D3Q19 LBM solver initialised to rest (ρ=1, u=0).
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` — grid dimensions in lattice units
    /// * `viscosity` — kinematic viscosity ν in lattice units
    pub fn new(nx: usize, ny: usize, nz: usize, viscosity: f64) -> Self {
        let tau = 3.0 * viscosity + 0.5;
        let omega = 1.0 / tau;

        let rho0 = 1.0_f64;
        let mut f = vec![vec![vec![vec![0.0_f64; nz]; ny]; nx]; Q19];
        for i in 0..Q19 {
            for x in 0..nx {
                for y in 0..ny {
                    for z in 0..nz {
                        f[i][x][y][z] = feq19(i, rho0, 0.0, 0.0, 0.0);
                    }
                }
            }
        }

        Self {
            nx,
            ny,
            nz,
            omega,
            f_buf: f.clone(),
            f,
            density: vec![vec![vec![rho0; nz]; ny]; nx],
            velocity: vec![vec![vec![[0.0_f64; 3]; nz]; ny]; nx],
            boundary: vec![vec![vec![BoundaryType3D::Fluid; nz]; ny]; nx],
            viscosity,
        }
    }

    /// Set boundary condition at node (x, y, z).
    pub fn set_boundary(&mut self, x: usize, y: usize, z: usize, bc: BoundaryType3D) {
        if x < self.nx && y < self.ny && z < self.nz {
            self.boundary[x][y][z] = bc;
        }
    }

    /// Recompute macroscopic density and velocity from distribution functions.
    fn update_macroscopic(&mut self) {
        for x in 0..self.nx {
            for y in 0..self.ny {
                for z in 0..self.nz {
                    let mut rho = 0.0;
                    let mut ux = 0.0;
                    let mut uy = 0.0;
                    let mut uz = 0.0;
                    for i in 0..Q19 {
                        let fi = self.f[i][x][y][z];
                        rho += fi;
                        ux += (CX19[i] as f64) * fi;
                        uy += (CY19[i] as f64) * fi;
                        uz += (CZ19[i] as f64) * fi;
                    }
                    if rho > 1e-15 {
                        self.density[x][y][z] = rho;
                        self.velocity[x][y][z] = [ux / rho, uy / rho, uz / rho];
                    }
                }
            }
        }
    }

    /// BGK collision step.
    fn collide(&mut self) {
        for x in 0..self.nx {
            for y in 0..self.ny {
                for z in 0..self.nz {
                    if self.boundary[x][y][z] == BoundaryType3D::Wall {
                        continue;
                    }
                    let rho = self.density[x][y][z];
                    let [ux, uy, uz] = self.velocity[x][y][z];
                    for i in 0..Q19 {
                        let feqi = feq19(i, rho, ux, uy, uz);
                        self.f[i][x][y][z] += self.omega * (feqi - self.f[i][x][y][z]);
                    }
                }
            }
        }
    }

    /// Streaming step with periodic wrapping and bounce-back on walls.
    fn stream_and_bc(&mut self) {
        let nx = self.nx as i64;
        let ny = self.ny as i64;
        let nz = self.nz as i64;

        for i in 0..Q19 {
            let cx = CX19[i] as i64;
            let cy = CY19[i] as i64;
            let cz = CZ19[i] as i64;
            for x in 0..self.nx {
                for y in 0..self.ny {
                    for z in 0..self.nz {
                        let xd = ((x as i64 - cx + nx) % nx) as usize;
                        let yd = ((y as i64 - cy + ny) % ny) as usize;
                        let zd = ((z as i64 - cz + nz) % nz) as usize;
                        self.f_buf[i][x][y][z] = self.f[i][xd][yd][zd];
                    }
                }
            }
        }

        std::mem::swap(&mut self.f, &mut self.f_buf);

        // Bounce-back on walls
        for x in 0..self.nx {
            for y in 0..self.ny {
                for z in 0..self.nz {
                    match self.boundary[x][y][z] {
                        BoundaryType3D::Wall => {
                            for i in 0..Q19 {
                                self.f[i][x][y][z] = self.f_buf[OPP19[i]][x][y][z];
                            }
                        }
                        BoundaryType3D::Inlet([ux, uy, uz]) => {
                            let rho = 1.0;
                            for i in 0..Q19 {
                                self.f[i][x][y][z] = feq19(i, rho, ux, uy, uz);
                            }
                        }
                        BoundaryType3D::Outlet => {
                            let xn = if x > 0 { x - 1 } else { 1 };
                            for i in 0..Q19 {
                                self.f[i][x][y][z] = self.f[i][xn][y][z];
                            }
                        }
                        BoundaryType3D::Fluid => {}
                    }
                }
            }
        }
    }

    /// Execute one complete LBM time step.
    pub fn step(&mut self) {
        self.update_macroscopic();
        self.collide();
        self.stream_and_bc();
    }

    /// Run the simulation for `n_steps` time steps.
    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
        self.update_macroscopic();
    }

    /// Compute total lattice mass.
    pub fn total_mass(&self) -> f64 {
        let mut mass = 0.0;
        for x in 0..self.nx {
            for y in 0..self.ny {
                for z in 0..self.nz {
                    mass += self.density[x][y][z];
                }
            }
        }
        mass
    }

    /// Compute Reynolds number Re = L_ref * U_ref / ν.
    pub fn reynolds_number(&self, l_ref: f64, u_ref: f64) -> f64 {
        if self.viscosity.abs() < f64::EPSILON {
            return f64::INFINITY;
        }
        l_ref * u_ref / self.viscosity
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: build lid-driven cavity setup
// ─────────────────────────────────────────────────────────────────────────────

/// Configure a standard lid-driven cavity problem on an `nx × ny` grid.
///
/// * All four walls use bounce-back
/// * The top wall (y = ny−1) uses an inlet with velocity `u_lid` in the x-direction
///
/// Returns the configured `D2Q9Lbm` solver.
pub fn lid_driven_cavity(nx: usize, ny: usize, viscosity: f64, u_lid: f64) -> D2Q9Lbm {
    let mut lbm = D2Q9Lbm::new(nx, ny, viscosity);

    // Left, right, bottom walls
    for y in 0..ny {
        lbm.set_boundary(0, y, BoundaryType::Wall);
        lbm.set_boundary(nx - 1, y, BoundaryType::Wall);
    }
    for x in 0..nx {
        lbm.set_boundary(x, 0, BoundaryType::Wall);
        // Top wall: moving lid
        lbm.set_boundary(x, ny - 1, BoundaryType::Inlet { ux: u_lid, uy: 0.0 });
    }
    lbm
}

/// Validate that LBM parameters are physically reasonable.
///
/// Returns an error if `omega` would lead to numerical instability (τ < 0.5 ↔ ω > 2).
pub fn validate_lbm_parameters(viscosity: f64) -> IntegrateResult<()> {
    if viscosity <= 0.0 {
        return Err(IntegrateError::ValueError(
            "LBM viscosity must be positive".into(),
        ));
    }
    let tau = 3.0 * viscosity + 0.5;
    let omega = 1.0 / tau;
    if omega >= 2.0 {
        return Err(IntegrateError::ValueError(format!(
            "LBM relaxation frequency omega={:.4} >= 2.0; reduce viscosity for stability",
            omega
        )));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d2q9_mass_conservation_periodic() {
        let mut lbm = D2Q9Lbm::new(16, 16, 0.1);
        let m0 = lbm.total_mass();
        lbm.run(50);
        let m1 = lbm.total_mass();
        assert!(
            (m1 - m0).abs() < 1e-8,
            "mass not conserved: Δm={:.2e}",
            m1 - m0
        );
    }

    #[test]
    fn test_d2q9_reynolds_number() {
        let lbm = D2Q9Lbm::new(64, 64, 0.01);
        let re = lbm.reynolds_number(64.0, 0.1);
        assert!((re - 640.0).abs() < 1e-10);
    }

    #[test]
    fn test_d2q9_pressure_positive() {
        let lbm = D2Q9Lbm::new(8, 8, 0.1);
        let p = lbm.pressure();
        for x in 0..8 {
            for y in 0..8 {
                assert!(p[x][y] > 0.0);
            }
        }
    }

    #[test]
    fn test_d2q9_vorticity_zero_at_rest() {
        let lbm = D2Q9Lbm::new(8, 8, 0.1);
        let vort = lbm.vorticity();
        for x in 0..8 {
            for y in 0..8 {
                assert!(vort[x][y].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_lid_driven_cavity_runs() {
        let mut lbm = lid_driven_cavity(16, 16, 0.1, 0.1);
        lbm.run(50);
        // Kinetic energy should be nonzero after driving
        let ke: f64 = (0..16)
            .flat_map(|x| (0..16).map(move |y| (x, y)))
            .map(|(x, y)| {
                let ux = lbm.velocity_x[x][y];
                let uy = lbm.velocity_y[x][y];
                0.5 * lbm.density[x][y] * (ux * ux + uy * uy)
            })
            .sum();
        assert!(
            ke > 0.0,
            "kinetic energy should be positive after lid driving"
        );
    }

    #[test]
    fn test_d2q9_wall_boundary() {
        let mut lbm = D2Q9Lbm::new(8, 8, 0.1);
        for y in 0..8 {
            lbm.set_boundary(0, y, BoundaryType::Wall);
        }
        lbm.run(20);
        // Velocity at wall nodes should remain near zero
        for y in 0..8 {
            assert!(lbm.velocity_x[0][y].abs() < 1e-10);
            assert!(lbm.velocity_y[0][y].abs() < 1e-10);
        }
    }

    #[test]
    fn test_d3q19_mass_conservation() {
        let mut lbm = D3Q19Lbm::new(8, 8, 8, 0.1);
        let m0 = lbm.total_mass();
        lbm.run(20);
        let m1 = lbm.total_mass();
        assert!(
            (m1 - m0).abs() < 1e-7,
            "3D mass not conserved: Δm={:.2e}",
            m1 - m0
        );
    }

    #[test]
    fn test_validate_parameters() {
        assert!(validate_lbm_parameters(0.1).is_ok());
        assert!(validate_lbm_parameters(-0.1).is_err());
    }
}
