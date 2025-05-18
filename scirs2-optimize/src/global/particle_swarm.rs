//! Particle Swarm Optimization (PSO) algorithm for global optimization
//!
//! PSO is a population-based stochastic optimization algorithm inspired by
//! the social behavior of bird flocking or fish schooling. Each particle
//! moves through the search space with a velocity influenced by its own
//! best position and the global best position found by the swarm.

use crate::error::OptimizeError;
use crate::unconstrained::OptimizeResult;
use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use rand::rngs::StdRng;

/// Options for Particle Swarm Optimization
#[derive(Debug, Clone)]
pub struct ParticleSwarmOptions {
    /// Number of particles in the swarm
    pub swarm_size: usize,
    /// Maximum number of iterations
    pub maxiter: usize,
    /// Cognitive parameter (attraction to personal best)
    pub c1: f64,
    /// Social parameter (attraction to global best)
    pub c2: f64,
    /// Inertia weight
    pub w: f64,
    /// Minimum velocity
    pub vmin: f64,
    /// Maximum velocity
    pub vmax: f64,
    /// Tolerance for convergence
    pub tol: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to use adaptive parameters
    pub adaptive: bool,
}

impl Default for ParticleSwarmOptions {
    fn default() -> Self {
        Self {
            swarm_size: 50,
            maxiter: 500,
            c1: 2.0,
            c2: 2.0,
            w: 0.9,
            vmin: -0.5,
            vmax: 0.5,
            tol: 1e-8,
            seed: None,
            adaptive: false,
        }
    }
}

/// Bounds for variables
pub type Bounds = Vec<(f64, f64)>;

/// Particle in the swarm
#[derive(Debug, Clone)]
struct Particle {
    /// Current position
    position: Array1<f64>,
    /// Current velocity
    velocity: Array1<f64>,
    /// Personal best position
    best_position: Array1<f64>,
    /// Personal best value
    best_value: f64,
}

/// Particle Swarm Optimization solver
pub struct ParticleSwarm<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    func: F,
    bounds: Bounds,
    options: ParticleSwarmOptions,
    ndim: usize,
    particles: Vec<Particle>,
    global_best_position: Array1<f64>,
    global_best_value: f64,
    rng: StdRng,
    nfev: usize,
    iteration: usize,
    inertia_weight: f64,
}

impl<F> ParticleSwarm<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    /// Create new Particle Swarm Optimization solver
    pub fn new(func: F, bounds: Bounds, options: ParticleSwarmOptions) -> Self {
        let ndim = bounds.len();
        let seed = options.seed.unwrap_or_else(rand::random);
        let mut rng = StdRng::seed_from_u64(seed);

        // Initialize particles
        let mut particles = Vec::with_capacity(options.swarm_size);
        let mut global_best_position = Array1::zeros(ndim);
        let mut global_best_value = f64::INFINITY;
        let mut nfev = 0;

        for _ in 0..options.swarm_size {
            // Random initial position within bounds
            let mut position = Array1::zeros(ndim);
            let mut velocity = Array1::zeros(ndim);

            for j in 0..ndim {
                let (lb, ub) = bounds[j];
                position[j] = rng.random_range(lb..ub);
                velocity[j] = rng.random_range(options.vmin..options.vmax) * (ub - lb);
            }

            // Evaluate initial position
            let value = func(&position.view());
            nfev += 1;

            // Update global best if necessary
            if value < global_best_value {
                global_best_value = value;
                global_best_position = position.clone();
            }

            particles.push(Particle {
                position: position.clone(),
                velocity,
                best_position: position,
                best_value: value,
            });
        }

        Self {
            func,
            bounds,
            options: options.clone(),
            ndim,
            particles,
            global_best_position,
            global_best_value,
            rng,
            nfev,
            iteration: 0,
            inertia_weight: options.w,
        }
    }

    /// Update the inertia weight adaptively
    fn update_inertia_weight(&mut self) {
        if self.options.adaptive {
            // Linear decrease from w to 0.4
            let w_max = self.options.w;
            let w_min = 0.4;
            self.inertia_weight =
                w_max - (w_max - w_min) * (self.iteration as f64 / self.options.maxiter as f64);
        }
    }

    /// Update particle velocity and position
    fn update_particle(&mut self, idx: usize) {
        let particle = &mut self.particles[idx];

        // Update velocity
        for j in 0..self.ndim {
            let r1 = self.rng.random::<f64>();
            let r2 = self.rng.random::<f64>();

            // Velocity update formula
            let cognitive =
                self.options.c1 * r1 * (particle.best_position[j] - particle.position[j]);
            let social =
                self.options.c2 * r2 * (self.global_best_position[j] - particle.position[j]);

            particle.velocity[j] = self.inertia_weight * particle.velocity[j] + cognitive + social;

            // Clamp velocity
            let (lb, ub) = self.bounds[j];
            let vmax = self.options.vmax * (ub - lb);
            let vmin = self.options.vmin * (ub - lb);
            particle.velocity[j] = particle.velocity[j].max(vmin).min(vmax);
        }

        // Update position
        for j in 0..self.ndim {
            particle.position[j] += particle.velocity[j];

            // Enforce bounds
            let (lb, ub) = self.bounds[j];
            if particle.position[j] < lb {
                particle.position[j] = lb;
                particle.velocity[j] = 0.0; // Reset velocity at boundary
            } else if particle.position[j] > ub {
                particle.position[j] = ub;
                particle.velocity[j] = 0.0; // Reset velocity at boundary
            }
        }

        // Evaluate new position
        let value = (self.func)(&particle.position.view());
        self.nfev += 1;

        // Update personal best
        if value < particle.best_value {
            particle.best_value = value;
            particle.best_position = particle.position.clone();

            // Update global best
            if value < self.global_best_value {
                self.global_best_value = value;
                self.global_best_position = particle.position.clone();
            }
        }
    }

    /// Check convergence criterion
    fn check_convergence(&self) -> bool {
        // Check if all particles have converged to the same region
        let mut max_distance: f64 = 0.0;

        for particle in &self.particles {
            let distance = (&particle.position - &self.global_best_position)
                .mapv(|x| x.abs())
                .sum();
            max_distance = max_distance.max(distance);
        }

        max_distance < self.options.tol
    }

    /// Run one iteration of the algorithm
    fn step(&mut self) -> bool {
        self.iteration += 1;
        self.update_inertia_weight();

        // Update all particles
        for i in 0..self.options.swarm_size {
            self.update_particle(i);
        }

        self.check_convergence()
    }

    /// Run the particle swarm optimization algorithm
    pub fn run(&mut self) -> OptimizeResult<f64> {
        let mut converged = false;

        for _ in 0..self.options.maxiter {
            converged = self.step();

            if converged {
                break;
            }
        }

        OptimizeResult {
            x: self.global_best_position.clone(),
            fun: self.global_best_value,
            nfev: self.nfev,
            func_evals: self.nfev,
            nit: self.iteration,
            iterations: self.iteration,
            success: converged,
            message: if converged {
                "Optimization converged successfully"
            } else {
                "Maximum number of iterations reached"
            }
            .to_string(),
            ..Default::default()
        }
    }
}

/// Perform global optimization using particle swarm optimization
pub fn particle_swarm<F>(
    func: F,
    bounds: Bounds,
    options: Option<ParticleSwarmOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    let options = options.unwrap_or_default();
    let mut solver = ParticleSwarm::new(func, bounds, options);
    Ok(solver.run())
}
