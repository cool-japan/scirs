//! Dual Annealing algorithm for global optimization
//!
//! A global optimization algorithm combining classical simulated annealing
//! with a fast simulated annealing (FSA) algorithm for finding the global
//! minimum of multivariate functions.

use crate::error::OptimizeError;
use crate::unconstrained::{minimize, Bounds, Method, OptimizeResult, Options};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[allow(unused_imports)]
use rand_distr::{Cauchy, Distribution as RandDistribution};

/// Options for Dual Annealing algorithm
#[derive(Debug, Clone)]
pub struct DualAnnealingOptions {
    /// Maximum number of global search iterations
    pub maxiter: usize,
    /// Minimum temperature for annealing
    pub initial_temp: f64,
    /// Visiting parameter (between 1 and 3)
    pub visit: f64,
    /// Acceptance parameter (between 0 and 1)
    pub accept: f64,
    /// Maximum number of function evaluations with no improvement
    pub maxfun: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Number of function evaluations without improvement before restarting
    pub restart_temp_ratio: f64,
    /// Bounds for variables
    pub bounds: Vec<(f64, f64)>,
}

impl Default for DualAnnealingOptions {
    fn default() -> Self {
        Self {
            maxiter: 1000,
            initial_temp: 5230.0,
            visit: 2.62,
            accept: -5.0,
            maxfun: 10000000,
            seed: None,
            restart_temp_ratio: 2e-5,
            bounds: vec![],
        }
    }
}

/// Dual Annealing solver
pub struct DualAnnealing<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    func: F,
    x0: Array1<f64>,
    options: DualAnnealingOptions,
    ndim: usize,
    rng: StdRng,
    temperature: f64,
    markov_chain_length: usize,
    current_x: Array1<f64>,
    current_energy: f64,
    best_x: Array1<f64>,
    best_energy: f64,
    nfev: usize,
    not_improved_counter: usize,
}

impl<F> DualAnnealing<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    /// Create new Dual Annealing solver
    pub fn new(func: F, x0: Array1<f64>, options: DualAnnealingOptions) -> Self {
        let ndim = x0.len();
        let seed = options
            .seed
            .unwrap_or_else(|| rand::rng().random_range(0..u64::MAX));
        let rng = StdRng::seed_from_u64(seed);

        let initial_energy = func(&x0.view());
        let temperature = options.initial_temp;

        Self {
            func,
            x0: x0.clone(),
            options,
            ndim,
            rng,
            temperature,
            markov_chain_length: 100 * ndim,
            current_x: x0.clone(),
            current_energy: initial_energy,
            best_x: x0.clone(),
            best_energy: initial_energy,
            nfev: 1,
            not_improved_counter: 0,
        }
    }

    /// Generate new point using visiting distribution
    fn generate_new_point(&mut self) -> Array1<f64> {
        let mut x_new = self.current_x.clone();

        // Using generalized visiting distribution
        for i in 0..self.ndim {
            let (lb, ub) = self.options.bounds[i];
            let y = self.current_x[i];

            // Generate random value using visiting distribution
            let q = self.options.visit;
            let mut v;

            // Generate from Power distribution
            loop {
                let u: f64 = self.rng.gen_range(0.0..1.0);
                let u1: f64 = self.rng.gen_range(0.0..1.0);
                let sign = if u1 < 0.5 { -1.0 } else { 1.0 };

                v = sign * self.temperature * ((1.0 + 1.0 / q).powf(u.abs()) - 1.0);

                // Apply bounds
                let new_val = y + v;
                if new_val >= lb && new_val <= ub {
                    x_new[i] = new_val;
                    break;
                }
            }
        }

        x_new
    }

    /// Calculate acceptance probability
    fn accept_probability(&self, energy_new: f64) -> f64 {
        if energy_new <= self.current_energy {
            1.0
        } else {
            let delta = energy_new - self.current_energy;
            (-delta / self.temperature).exp()
        }
    }

    /// Perform local search using gradient-based method
    fn local_search(&self) -> (Array1<f64>, f64, usize) {
        let result = minimize(
            |x| (self.func)(x),
            &self.current_x.to_vec(),
            Method::LBFGS,
            Some(Options {
                bounds: Some(
                    Bounds::from_vecs(
                        self.options
                            .bounds
                            .iter()
                            .map(|&(lb, _)| Some(lb))
                            .collect(),
                        self.options
                            .bounds
                            .iter()
                            .map(|&(_, ub)| Some(ub))
                            .collect(),
                    )
                    .unwrap(),
                ),
                ..Default::default()
            }),
        )
        .unwrap();

        (result.x, result.fun, result.nfev)
    }

    /// Update temperature using annealing schedule
    fn update_temperature(&mut self, k: usize) {
        // Classical annealing schedule
        self.temperature = self.options.initial_temp / (k as f64).ln_1p();
    }

    /// Check if restart is needed
    fn check_restart(&mut self) -> bool {
        if self.not_improved_counter >= self.markov_chain_length {
            self.not_improved_counter = 0;
            self.temperature = self.options.initial_temp;
            true
        } else {
            false
        }
    }

    /// Run one iteration of the algorithm
    fn step(&mut self, iteration: usize) -> bool {
        let mut improved = false;

        // Global search phase
        for _ in 0..self.markov_chain_length {
            let x_new = self.generate_new_point();
            let energy_new = (self.func)(&x_new.view());
            self.nfev += 1;

            // Acceptance test
            let accept_prob = self.accept_probability(energy_new);
            if self.rng.gen_range(0.0..1.0) < accept_prob {
                self.current_x = x_new;
                self.current_energy = energy_new;

                if energy_new < self.best_energy {
                    self.best_x = self.current_x.clone();
                    self.best_energy = energy_new;
                    improved = true;
                    self.not_improved_counter = 0;
                }
            }
        }

        // Local search phase
        if iteration % 10 == 0 {
            // Perform local search periodically
            let (x_local, energy_local, nfev_local) = self.local_search();
            self.nfev += nfev_local;

            if energy_local < self.best_energy {
                self.best_x = x_local;
                self.best_energy = energy_local;
                self.current_x = self.best_x.clone();
                self.current_energy = self.best_energy;
                improved = true;
                self.not_improved_counter = 0;
            }
        }

        if !improved {
            self.not_improved_counter += 1;
        }

        // Update temperature
        self.update_temperature(iteration + 1);

        // Check for restart
        self.check_restart();

        improved
    }

    /// Run the dual annealing algorithm
    pub fn run(&mut self) -> OptimizeResult<f64> {
        let mut nit = 0;
        let mut success = false;
        let mut message = "Maximum number of iterations reached".to_string();

        for i in 0..self.options.maxiter {
            let _improved = self.step(i);
            nit += 1;

            // Check convergence
            if self.temperature < self.options.restart_temp_ratio * self.options.initial_temp {
                success = true;
                message = "Temperature converged".to_string();
                break;
            }

            if self.nfev >= self.options.maxfun {
                message = "Maximum number of function evaluations reached".to_string();
                break;
            }
        }

        // Final local search for polish
        let (x_final, energy_final, nfev_final) = self.local_search();
        self.nfev += nfev_final;

        if energy_final < self.best_energy {
            self.best_x = x_final;
            self.best_energy = energy_final;
        }

        OptimizeResult {
            x: self.best_x.clone(),
            fun: self.best_energy,
            nfev: self.nfev,
            func_evals: self.nfev,
            nit,
            success,
            message,
            ..Default::default()
        }
    }
}

/// Perform global optimization using dual annealing
#[allow(dead_code)]
pub fn dual_annealing<F>(
    func: F,
    x0: Array1<f64>,
    bounds: Vec<(f64, f64)>,
    options: Option<DualAnnealingOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    let mut options = options.unwrap_or_default();

    // Ensure bounds are set
    if options.bounds.is_empty() {
        options.bounds = bounds;
    }

    let mut solver = DualAnnealing::new(func, x0, options);
    Ok(solver.run())
}
