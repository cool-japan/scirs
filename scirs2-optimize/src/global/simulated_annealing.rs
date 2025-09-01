//! Simulated Annealing algorithm for global optimization
//!
//! Simulated annealing is a probabilistic optimization algorithm inspired by
//! the physical process of annealing in metallurgy. It accepts worse solutions
//! with a probability that decreases over time as the "temperature" cools.

use crate::error::OptimizeError;
use crate::unconstrained::OptimizeResult;
use ndarray::{Array1, ArrayView1};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{rng, Rng, SeedableRng};

/// Options for Simulated Annealing
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingOptions {
    /// Maximum number of iterations
    pub maxiter: usize,
    /// Initial temperature
    pub initial_temp: f64,
    /// Final temperature
    pub final_temp: f64,
    /// Temperature reduction rate (0 < alpha < 1)
    pub alpha: f64,
    /// Maximum steps at each temperature
    pub max_steps_per_temp: usize,
    /// Step size for neighbor generation
    pub step_size: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Cooling schedule: "exponential", "linear", or "adaptive"
    pub schedule: String,
    /// Whether to print progress
    pub verbose: bool,
}

impl Default for SimulatedAnnealingOptions {
    fn default() -> Self {
        Self {
            maxiter: 10000,
            initial_temp: 100.0,
            final_temp: 1e-8,
            alpha: 0.95,
            max_steps_per_temp: 100,
            step_size: 0.5,
            seed: None,
            schedule: "exponential".to_string(),
            verbose: false,
        }
    }
}

/// Bounds for variables
pub type Bounds = Vec<(f64, f64)>;

/// Simulated Annealing solver
pub struct SimulatedAnnealing<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    func: F,
    x0: Array1<f64>,
    bounds: Option<Bounds>,
    options: SimulatedAnnealingOptions,
    ndim: usize,
    current_x: Array1<f64>,
    current_value: f64,
    best_x: Array1<f64>,
    best_value: f64,
    temperature: f64,
    rng: StdRng,
    nfev: usize,
    nit: usize,
}

impl<F> SimulatedAnnealing<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    /// Create new Simulated Annealing solver
    pub fn new(
        func: F,
        x0: Array1<f64>,
        bounds: Option<Bounds>,
        options: SimulatedAnnealingOptions,
    ) -> Self {
        let ndim = x0.len();
        let seed = options.seed.unwrap_or_else(|| rng().random());
        let rng = StdRng::seed_from_u64(seed);

        // Evaluate initial point
        let initial_value = func(&x0.view());

        Self {
            func,
            x0: x0.clone(),
            bounds,
            options: options.clone(),
            ndim,
            current_x: x0.clone(),
            current_value: initial_value,
            best_x: x0,
            best_value: initial_value,
            temperature: options.initial_temp,
            rng,
            nfev: 1,
            nit: 0,
        }
    }

    /// Generate a random neighbor
    fn generate_neighbor(&mut self) -> Array1<f64> {
        let mut neighbor = self.current_x.clone();

        // Randomly perturb one or more dimensions
        let num_dims_to_perturb = self.rng.gen_range(1..=self.ndim);
        let mut dims: Vec<usize> = (0..self.ndim).collect();
        dims.shuffle(&mut self.rng);

        for &i in dims.iter().take(num_dims_to_perturb) {
            let perturbation = self
                .rng
                .random_range(-self.options.step_size..self.options.step_size);
            neighbor[i] += perturbation;

            // Apply bounds if specified
            if let Some(ref bounds) = self.bounds {
                let (lb, ub) = bounds[i];
                neighbor[i] = neighbor[i].max(lb).min(ub);
            }
        }

        neighbor
    }

    /// Calculate acceptance probability
    fn acceptance_probability(&self, new_value: f64) -> f64 {
        if new_value < self.current_value {
            1.0
        } else {
            let delta = new_value - self.current_value;
            (-delta / self.temperature).exp()
        }
    }

    /// Update temperature according to cooling schedule
    fn update_temperature(&mut self) {
        match self.options.schedule.as_str() {
            "exponential" => {
                self.temperature *= self.options.alpha;
            }
            "linear" => {
                let temp_range = self.options.initial_temp - self.options.final_temp;
                let temp_decrement = temp_range / self.options.maxiter as f64;
                self.temperature = (self.temperature - temp_decrement).max(self.options.final_temp);
            }
            "adaptive" => {
                // Adaptive cooling based on acceptance rate
                let acceptance_rate = self.calculate_acceptance_rate();
                if acceptance_rate > 0.8 {
                    self.temperature *= 0.9; // Cool faster if accepting too many
                } else if acceptance_rate < 0.2 {
                    self.temperature *= 0.99; // Cool slower if accepting too few
                } else {
                    self.temperature *= self.options.alpha;
                }
            }
            _ => {
                self.temperature *= self.options.alpha; // Default to exponential
            }
        }
    }

    /// Calculate recent acceptance rate (simplified)
    fn calculate_acceptance_rate(&self) -> f64 {
        // In a real implementation, this would track actual acceptance rate
        // For now, return a default value
        0.5
    }

    /// Run one step of the algorithm
    fn step(&mut self) -> bool {
        self.nit += 1;

        for _ in 0..self.options.max_steps_per_temp {
            // Generate neighbor
            let neighbor = self.generate_neighbor();
            let neighbor_value = (self.func)(&neighbor.view());
            self.nfev += 1;

            // Accept or reject
            let acceptance_prob = self.acceptance_probability(neighbor_value);
            if self.rng.gen_range(0.0..1.0) < acceptance_prob {
                self.current_x = neighbor;
                self.current_value = neighbor_value;

                // Update best if improved
                if neighbor_value < self.best_value {
                    self.best_x = self.current_x.clone();
                    self.best_value = neighbor_value;
                }
            }
        }

        // Update temperature
        self.update_temperature();

        // Check convergence
        self.temperature < self.options.final_temp
    }

    /// Run the simulated annealing algorithm
    pub fn run(&mut self) -> OptimizeResult<f64> {
        let mut converged = false;

        while self.nit < self.options.maxiter {
            converged = self.step();

            if converged {
                break;
            }

            if self.options.verbose && self.nit % 100 == 0 {
                println!(
                    "Iteration {}: T = {:.6}..best = {:.6}",
                    self.nit, self.temperature, self.best_value
                );
            }
        }

        OptimizeResult {
            x: self.best_x.clone(),
            fun: self.best_value,
            nfev: self.nfev,
            func_evals: self.nfev,
            nit: self.nit,
            success: converged,
            message: if converged {
                "Temperature reached minimum"
            } else {
                "Maximum iterations reached"
            }
            .to_string(),
            ..Default::default()
        }
    }
}

/// Perform global optimization using simulated annealing
#[allow(dead_code)]
pub fn simulated_annealing<F>(
    func: F,
    x0: Array1<f64>,
    bounds: Option<Bounds>,
    options: Option<SimulatedAnnealingOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    let options = options.unwrap_or_default();
    let mut solver = SimulatedAnnealing::new(func, x0, bounds, options);
    Ok(solver.run())
}
