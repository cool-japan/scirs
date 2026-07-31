//! Differential Evolution algorithm for global optimization
//!
//! The differential evolution method is a stochastic global optimization
//! algorithm that does not use gradient methods to find the minimum and
//! can search large areas of candidate space.

use crate::error::OptimizeError;
use crate::global::qmc::SobolGenerator;
use crate::parallel::{parallel_evaluate_batch, ParallelOptions};
use crate::unconstrained::{
    minimize, Bounds as UnconstrainedBounds, Method, OptimizeResult, Options,
};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::prelude::SliceRandom;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Random;
use scirs2_core::random::Uniform;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

/// Options for Differential Evolution algorithm
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionOptions {
    /// Maximum number of generations
    pub maxiter: usize,
    /// Population size (as a multiple of the number of parameters or absolute size)
    pub popsize: usize,
    /// The tolerance for convergence
    pub tol: f64,
    /// The mutation coefficient (F) or tuple of (lower_bound, upper_bound)
    pub mutation: (f64, f64),
    /// The recombination coefficient (CR)
    pub recombination: f64,
    /// Whether to polish the best solution with local optimization
    pub polish: bool,
    /// Initial population method: "latinhypercube", "halton", "sobol", or "random"
    pub init: String,
    /// Absolute tolerance for convergence
    pub atol: f64,
    /// Strategy for updating the population: "immediate" or "deferred"
    pub updating: String,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Initial guess
    pub x0: Option<Array1<f64>>,
    /// Parallel computation options
    pub parallel: Option<ParallelOptions>,
}

impl Default for DifferentialEvolutionOptions {
    fn default() -> Self {
        Self {
            maxiter: 1000,
            popsize: 15,
            tol: 0.01,
            mutation: (0.5, 1.0),
            recombination: 0.7,
            polish: true,
            init: "latinhypercube".to_string(),
            atol: 0.0,
            updating: "immediate".to_string(),
            seed: None,
            x0: None,
            parallel: None,
        }
    }
}

/// Strategy names for mutation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Strategy {
    Best1Bin,
    Best1Exp,
    Rand1Bin,
    Rand1Exp,
    Best2Bin,
    Best2Exp,
    Rand2Bin,
    Rand2Exp,
    CurrentToBest1Bin,
    CurrentToBest1Exp,
}

impl Strategy {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "best1bin" => Some(Strategy::Best1Bin),
            "best1exp" => Some(Strategy::Best1Exp),
            "rand1bin" => Some(Strategy::Rand1Bin),
            "rand1exp" => Some(Strategy::Rand1Exp),
            "best2bin" => Some(Strategy::Best2Bin),
            "best2exp" => Some(Strategy::Best2Exp),
            "rand2bin" => Some(Strategy::Rand2Bin),
            "rand2exp" => Some(Strategy::Rand2Exp),
            "currenttobest1bin" => Some(Strategy::CurrentToBest1Bin),
            "currenttobest1exp" => Some(Strategy::CurrentToBest1Exp),
            _ => None,
        }
    }
}

/// Bounds for variables in differential evolution
pub type Bounds = Vec<(f64, f64)>;

/// Differential Evolution solver
pub struct DifferentialEvolution<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone + Sync,
{
    func: F,
    bounds: Bounds,
    options: DifferentialEvolutionOptions,
    strategy: Strategy,
    ndim: usize,
    population: Array2<f64>,
    energies: Array1<f64>,
    best_energy: f64,
    best_idx: usize,
    rng: Random<StdRng>,
    nfev: usize,
}

use scirs2_core::ndarray::Array2;

impl<F> DifferentialEvolution<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone + Sync,
{
    /// Create new Differential Evolution solver
    pub fn new(
        func: F,
        bounds: Bounds,
        options: DifferentialEvolutionOptions,
        strategy: &str,
    ) -> Self {
        let ndim = bounds.len();
        let popsize = if options.popsize < ndim {
            options.popsize * ndim
        } else {
            options.popsize
        };

        let seed = options
            .seed
            .unwrap_or_else(|| scirs2_core::random::rng().random());
        let rng = Random::seed(seed);

        let strategy_enum = Strategy::from_str(strategy).unwrap_or(Strategy::Best1Bin);

        let mut solver = Self {
            func,
            bounds,
            options,
            strategy: strategy_enum,
            ndim,
            population: Array2::zeros((popsize, ndim)),
            energies: Array1::zeros(popsize),
            best_energy: f64::INFINITY,
            best_idx: 0,
            rng,
            nfev: 0,
        };

        // Validate bounds
        solver.validate_bounds();
        solver.init_population();
        solver
    }

    /// Validate that bounds are properly specified
    fn validate_bounds(&self) {
        for (i, &(lb, ub)) in self.bounds.iter().enumerate() {
            if !lb.is_finite() || !ub.is_finite() {
                panic!(
                    "Bounds must be finite values. Variable {}: bounds = ({}, {})",
                    i, lb, ub
                );
            }
            if lb >= ub {
                panic!(
                    "Lower bound must be less than upper bound. Variable {}: lb = {}, ub = {}",
                    i, lb, ub
                );
            }
            if (ub - lb) < 1e-12 {
                panic!(
                    "Bounds range is too small. Variable {}: range = {}",
                    i,
                    ub - lb
                );
            }
        }
    }

    /// Initialize the population
    fn init_population(&mut self) {
        let popsize = self.population.nrows();

        // Initialize population based on initialization method
        match self.options.init.as_str() {
            "latinhypercube" => self.init_latinhypercube(),
            "halton" => self.init_halton(),
            "sobol" => self.init_sobol(),
            _ => self.init_random(),
        }

        // If x0 is provided, replace one member with it (bounds-checked)
        if let Some(x0) = self.options.x0.clone() {
            for (i, &val) in x0.iter().enumerate() {
                self.population[[0, i]] = self.ensure_bounds(i, val);
            }
        }

        // Evaluate initial population
        if self.options.parallel.is_some() {
            // Parallel evaluation
            let candidates: Vec<Array1<f64>> = (0..popsize)
                .map(|i| self.population.row(i).to_owned())
                .collect();

            // Extract parallel options for evaluation
            let parallel_opts = self.options.parallel.as_ref().expect("Operation failed");

            let energies = parallel_evaluate_batch(&self.func, &candidates, parallel_opts);
            self.energies = Array1::from_vec(energies);
            self.nfev += popsize;

            // Find best
            for i in 0..popsize {
                if self.energies[i] < self.best_energy {
                    self.best_energy = self.energies[i];
                    self.best_idx = i;
                }
            }
        } else {
            // Sequential evaluation
            for i in 0..popsize {
                let candidate = self.population.row(i);
                self.energies[i] = (self.func)(&candidate);
                self.nfev += 1;

                if self.energies[i] < self.best_energy {
                    self.best_energy = self.energies[i];
                    self.best_idx = i;
                }
            }
        }
    }

    /// Initialize population with random values
    fn init_random(&mut self) {
        let popsize = self.population.nrows();

        for i in 0..popsize {
            for j in 0..self.ndim {
                let (lb, ub) = self.bounds[j];
                let uniform = Uniform::new(lb, ub).expect("Operation failed");
                self.population[[i, j]] = self.rng.sample(uniform);
            }
        }
    }

    /// Initialize population using Latin hypercube sampling
    fn init_latinhypercube(&mut self) {
        let popsize = self.population.nrows();

        // Create segments for each dimension
        for j in 0..self.ndim {
            let (lb, ub) = self.bounds[j];
            let segment_size = (ub - lb) / popsize as f64;

            let mut segments: Vec<usize> = (0..popsize).collect();
            segments.shuffle(&mut self.rng);

            for (i, &seg) in segments.iter().enumerate() {
                let segment_lb = lb + seg as f64 * segment_size;
                let segment_ub = segment_lb + segment_size;
                let uniform = Uniform::new(segment_lb, segment_ub).expect("Operation failed");
                self.population[[i, j]] = self.rng.sample(uniform);
            }
        }
    }

    /// Initialize population using Halton sequence
    fn init_halton(&mut self) {
        let primes: [u64; 25] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97,
        ];

        let popsize = self.population.nrows();
        for i in 0..popsize {
            for j in 0..self.ndim {
                // Use the (i+1)th term of the Halton sequence for base primes[j % primes.len()]
                let base = primes[j % primes.len()];
                let halton_value = crate::global::qmc::halton_radical_inverse(i + 1, base);

                // Scale to bounds
                let (lb, ub) = self.bounds[j];
                self.population[[i, j]] = lb + halton_value * (ub - lb);
            }
        }
    }

    /// Initialize population using a Sobol sequence (validated direction
    /// numbers for the first [`crate::global::qmc::MAX_SOBOL_DIM`]
    /// dimensions, falling back to a genuine Halton sequence -- not plain
    /// randomness -- beyond that; see [`crate::global::qmc`]).
    fn init_sobol(&mut self) {
        let mut sobol_gen = SobolGenerator::new(self.ndim);

        let popsize = self.population.nrows();
        for i in 0..popsize {
            let sobol_point = sobol_gen.next_point();
            for j in 0..self.ndim {
                let (lb, ub) = self.bounds[j];
                self.population[[i, j]] = lb + sobol_point[j] * (ub - lb);
            }
        }
    }

    /// Ensure bounds for a parameter using reflection method
    fn ensure_bounds(&mut self, idx: usize, val: f64) -> f64 {
        let (lb, ub) = self.bounds[idx];

        if val >= lb && val <= ub {
            // Value is within bounds
            val
        } else if val < lb {
            // Reflect around lower bound
            let excess = lb - val;
            let range = ub - lb;
            if excess <= range {
                lb + excess
            } else {
                // If reflection goes beyond upper bound, use random value in range
                self.rng.random_range(lb..ub)
            }
        } else {
            // val > ub..reflect around upper bound
            let excess = val - ub;
            let range = ub - lb;
            if excess <= range {
                ub - excess
            } else {
                // If reflection goes beyond lower bound, use random value in range
                self.rng.random_range(lb..ub)
            }
        }
    }

    /// Create mutant vector using differential evolution
    fn create_mutant(&mut self, candidate_idx: usize) -> Array1<f64> {
        let popsize = self.population.nrows();
        let mut mutant = Array1::zeros(self.ndim);

        // Select indices for mutation
        let mut indices: Vec<usize> = Vec::with_capacity(5);
        while indices.len() < 5 {
            let idx = self.rng.random_range(0..popsize);
            if idx != candidate_idx && !indices.contains(&idx) {
                indices.push(idx);
            }
        }

        let mutation_factor = if self.options.mutation.0 == self.options.mutation.1 {
            self.options.mutation.0
        } else {
            self.rng
                .random_range(self.options.mutation.0..self.options.mutation.1)
        };

        match self.strategy {
            Strategy::Best1Bin | Strategy::Best1Exp => {
                // mutant = best + F * (r1 - r2)
                let best = self.population.row(self.best_idx);
                let r1 = self.population.row(indices[0]);
                let r2 = self.population.row(indices[1]);
                for i in 0..self.ndim {
                    mutant[i] = best[i] + mutation_factor * (r1[i] - r2[i]);
                }
            }
            Strategy::Rand1Bin | Strategy::Rand1Exp => {
                // mutant = r0 + F * (r1 - r2)
                let r0 = self.population.row(indices[0]);
                let r1 = self.population.row(indices[1]);
                let r2 = self.population.row(indices[2]);
                for i in 0..self.ndim {
                    mutant[i] = r0[i] + mutation_factor * (r1[i] - r2[i]);
                }
            }
            Strategy::Best2Bin | Strategy::Best2Exp => {
                // mutant = best + F * (r1 - r2) + F * (r3 - r4)
                let best = self.population.row(self.best_idx);
                let r1 = self.population.row(indices[0]);
                let r2 = self.population.row(indices[1]);
                let r3 = self.population.row(indices[2]);
                let r4 = self.population.row(indices[3]);
                for i in 0..self.ndim {
                    mutant[i] = best[i]
                        + mutation_factor * (r1[i] - r2[i])
                        + mutation_factor * (r3[i] - r4[i]);
                }
            }
            Strategy::Rand2Bin | Strategy::Rand2Exp => {
                // mutant = r0 + F * (r1 - r2) + F * (r3 - r4)
                let r0 = self.population.row(indices[0]);
                let r1 = self.population.row(indices[1]);
                let r2 = self.population.row(indices[2]);
                let r3 = self.population.row(indices[3]);
                let r4 = self.population.row(indices[4]);
                for i in 0..self.ndim {
                    mutant[i] = r0[i]
                        + mutation_factor * (r1[i] - r2[i])
                        + mutation_factor * (r3[i] - r4[i]);
                }
            }
            Strategy::CurrentToBest1Bin | Strategy::CurrentToBest1Exp => {
                // mutant = current + F * (best - current) + F * (r1 - r2)
                let current = self.population.row(candidate_idx);
                let best = self.population.row(self.best_idx);
                let r1 = self.population.row(indices[0]);
                let r2 = self.population.row(indices[1]);
                for i in 0..self.ndim {
                    mutant[i] = current[i]
                        + mutation_factor * (best[i] - current[i])
                        + mutation_factor * (r1[i] - r2[i]);
                }
            }
        }

        // Apply bounds checking after all calculations are done
        for i in 0..self.ndim {
            mutant[i] = self.ensure_bounds(i, mutant[i]);
        }

        mutant
    }

    /// Create trial vector using crossover
    fn create_trial(&mut self, candidate_idx: usize, mutant: &Array1<f64>) -> Array1<f64> {
        let candidate = self.population.row(candidate_idx).to_owned();
        let mut trial = candidate.clone();

        match self.strategy {
            Strategy::Best1Bin
            | Strategy::Rand1Bin
            | Strategy::Best2Bin
            | Strategy::Rand2Bin
            | Strategy::CurrentToBest1Bin => {
                // Binomial crossover
                let randn = self.rng.random_range(0..self.ndim);
                for i in 0..self.ndim {
                    if i == randn || self.rng.random_range(0.0..1.0) < self.options.recombination {
                        trial[i] = mutant[i];
                    }
                }
            }
            Strategy::Best1Exp
            | Strategy::Rand1Exp
            | Strategy::Best2Exp
            | Strategy::Rand2Exp
            | Strategy::CurrentToBest1Exp => {
                // Exponential crossover
                let randn = self.rng.random_range(0..self.ndim);
                let mut i = randn;
                loop {
                    trial[i] = mutant[i];
                    i = (i + 1) % self.ndim;
                    if i == randn || self.rng.random_range(0.0..1.0) >= self.options.recombination {
                        break;
                    }
                }
            }
        }

        // Ensure all trial elements are within bounds after crossover
        for i in 0..self.ndim {
            trial[i] = self.ensure_bounds(i, trial[i]);
        }

        trial
    }

    /// Run one generation of the algorithm
    fn evolve(&mut self) -> bool {
        let popsize = self.population.nrows();
        let mut converged = true;

        if self.options.parallel.is_some() {
            // First, generate all mutants and trials
            let mut trials_and_indices: Vec<(Array1<f64>, usize)> = Vec::with_capacity(popsize);
            for idx in 0..popsize {
                let mutant = self.create_mutant(idx);
                let trial = self.create_trial(idx, &mutant);
                trials_and_indices.push((trial, idx));
            }

            // Extract just the trials for batch evaluation
            let trials: Vec<Array1<f64>> = trials_and_indices
                .iter()
                .map(|(trial_, _)| trial_.clone())
                .collect();

            // Extract the parallel options for evaluation
            let parallel_opts = self.options.parallel.as_ref().expect("Operation failed");

            // Evaluate all trials in parallel
            let trial_energies = parallel_evaluate_batch(&self.func, &trials, parallel_opts);
            self.nfev += popsize;

            // Process results
            for ((trial, idx), trial_energy) in
                trials_and_indices.into_iter().zip(trial_energies.iter())
            {
                if *trial_energy < self.energies[idx] && self.options.updating == "immediate" {
                    for i in 0..self.ndim {
                        self.population[[idx, i]] = trial[i];
                    }
                    self.energies[idx] = *trial_energy;

                    if *trial_energy < self.best_energy {
                        self.best_energy = *trial_energy;
                        self.best_idx = idx;
                    }
                }

                // Check convergence
                let diff = (self.energies[idx] - self.best_energy).abs();
                if diff > self.options.tol + self.options.atol {
                    converged = false;
                }
            }
        } else {
            // Sequential evolution
            for idx in 0..popsize {
                let mutant = self.create_mutant(idx);
                let trial = self.create_trial(idx, &mutant);

                let trial_energy = (self.func)(&trial.view());
                self.nfev += 1;

                if trial_energy < self.energies[idx] && self.options.updating == "immediate" {
                    for i in 0..self.ndim {
                        self.population[[idx, i]] = trial[i];
                    }
                    self.energies[idx] = trial_energy;

                    if trial_energy < self.best_energy {
                        self.best_energy = trial_energy;
                        self.best_idx = idx;
                    }
                }

                // Check convergence
                let diff = (self.energies[idx] - self.best_energy).abs();
                if diff > self.options.tol + self.options.atol {
                    converged = false;
                }
            }
        }

        converged
    }

    /// Run the differential evolution algorithm
    pub fn run(&mut self) -> OptimizeResult<f64> {
        let mut converged = false;
        let mut nit = 0;

        for _ in 0..self.options.maxiter {
            converged = self.evolve();
            nit += 1;

            if converged {
                break;
            }
        }

        let mut result = OptimizeResult {
            x: self.population.row(self.best_idx).to_owned(),
            fun: self.best_energy,
            nfev: self.nfev,
            func_evals: self.nfev,
            nit,
            success: converged,
            message: if converged {
                "Optimization converged successfully"
            } else {
                "Maximum number of iterations reached"
            }
            .to_string(),
            ..Default::default()
        };

        // Polish the result with local optimization if requested
        if self.options.polish {
            let bounds_vec: Vec<(f64, f64)> = self.bounds.clone();
            let local_result = minimize(
                |x| (self.func)(x),
                &result.x.to_vec(),
                Method::LBFGS,
                Some(Options {
                    bounds: Some(
                        UnconstrainedBounds::from_vecs(
                            bounds_vec.iter().map(|b| Some(b.0)).collect(),
                            bounds_vec.iter().map(|b| Some(b.1)).collect(),
                        )
                        .expect("Operation failed"),
                    ),
                    ..Default::default()
                }),
            )
            .expect("Operation failed");
            if local_result.success && local_result.fun < result.fun {
                // Ensure polished result respects bounds
                let mut polished_x = local_result.x;
                for (i, &(lb, ub)) in self.bounds.iter().enumerate() {
                    polished_x[i] = polished_x[i].max(lb).min(ub);
                }

                // Only accept if still better after bounds enforcement
                let polished_fun = (self.func)(&polished_x.view());
                if polished_fun < result.fun {
                    result.x = polished_x;
                    result.fun = polished_fun;
                    result.nfev += local_result.nfev + 1; // +1 for our re-evaluation
                    result.func_evals = result.nfev;
                }
            }
        }

        result
    }
}

/// Perform global optimization using differential evolution
///
/// # Arguments
///
/// * `func` - Objective function to minimize
/// * `bounds` - Variable bounds as Vec<(lower, upper)>
/// * `options` - DE configuration options
/// * `strategy` - Mutation strategy name (e.g., "best1bin", "rand1bin")
///
/// # Returns
///
/// `OptimizeResult<f64>` with the optimization result
#[allow(dead_code)]
pub fn differential_evolution<F>(
    func: F,
    bounds: Bounds,
    options: Option<DifferentialEvolutionOptions>,
    strategy: Option<&str>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone + Sync,
{
    if bounds.is_empty() {
        return Err(OptimizeError::InvalidInput(
            "Bounds must not be empty".to_string(),
        ));
    }
    for (i, &(lb, ub)) in bounds.iter().enumerate() {
        if !lb.is_finite() || !ub.is_finite() {
            return Err(OptimizeError::InvalidInput(format!(
                "Bounds must be finite for dimension {} ({}, {})",
                i, lb, ub
            )));
        }
        if lb >= ub {
            return Err(OptimizeError::InvalidInput(format!(
                "Lower bound must be less than upper bound for dimension {} ({} >= {})",
                i, lb, ub
            )));
        }
    }

    let options = options.unwrap_or_default();
    let strategy = strategy.unwrap_or("best1bin");

    let mut solver = DifferentialEvolution::new(func, bounds, options, strategy);
    Ok(solver.run())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sphere(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn rastrigin(x: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    #[test]
    fn test_de_sphere_best1bin() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = DifferentialEvolutionOptions::default();
        opts.seed = Some(42);
        opts.maxiter = 200;
        opts.polish = false;

        let result = differential_evolution(sphere, bounds, Some(opts), Some("best1bin"));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.fun < 0.01, "Sphere min should be ~0, got {}", res.fun);
    }

    #[test]
    fn test_de_sphere_rand1bin() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = DifferentialEvolutionOptions::default();
        opts.seed = Some(42);
        opts.maxiter = 300;
        opts.polish = false;

        let result = differential_evolution(sphere, bounds, Some(opts), Some("rand1bin"));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            res.fun < 0.1,
            "Sphere with rand1bin should be near 0, got {}",
            res.fun
        );
    }

    #[test]
    fn test_de_currenttobest1bin() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = DifferentialEvolutionOptions::default();
        opts.seed = Some(42);
        opts.maxiter = 300;
        opts.polish = false;

        let result = differential_evolution(sphere, bounds, Some(opts), Some("currenttobest1bin"));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            res.fun < 0.1,
            "currenttobest1bin should be near 0, got {}",
            res.fun
        );
    }

    #[test]
    fn test_de_rastrigin() {
        let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
        let mut opts = DifferentialEvolutionOptions::default();
        opts.seed = Some(42);
        opts.popsize = 30;
        opts.maxiter = 500;
        opts.polish = false;

        let result = differential_evolution(rastrigin, bounds, Some(opts), None);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            res.fun < 10.0,
            "DE should find reasonable Rastrigin value, got {}",
            res.fun
        );
    }

    #[test]
    fn test_de_rosenbrock() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = DifferentialEvolutionOptions::default();
        opts.seed = Some(42);
        opts.popsize = 30;
        opts.maxiter = 500;
        opts.polish = false;

        let result = differential_evolution(rosenbrock, bounds, Some(opts), None);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            res.fun < 5.0,
            "DE should find reasonable Rosenbrock value, got {}",
            res.fun
        );
    }

    #[test]
    fn test_de_with_initial_guess() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = DifferentialEvolutionOptions::default();
        opts.seed = Some(42);
        opts.maxiter = 100;
        opts.polish = false;
        opts.x0 = Some(Array1::from_vec(vec![0.1, 0.1]));

        let result = differential_evolution(sphere, bounds, Some(opts), None);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.fun < 0.1);
    }

    #[test]
    fn test_de_sobol_init_matches_validated_sobol_sequence() {
        // Regression test for the ad-hoc "polynomial recurrence" SobolState
        // this crate used to build its own direction numbers with (not a
        // real Sobol sequence). `init_sobol` now delegates to
        // `crate::global::qmc::SobolGenerator`, whose first two points are
        // always exactly (0, 0, ...) and (0.5, 0.5, ...) -- verified
        // bit-for-bit against `scipy.stats.qmc.Sobol` in the `qmc` module's
        // own tests. The old ad-hoc generator's first two points did NOT
        // land on these exact values (it used `1u32 << 31` as its first
        // direction number, not the same recurrence), so this is a real,
        // non-trivial check on the actual initial-population contents, not
        // merely "did it fail to panic".
        let bounds = vec![(0.0, 10.0), (0.0, 20.0), (-4.0, 4.0)];
        let mut opts = DifferentialEvolutionOptions::default();
        opts.seed = Some(1);
        opts.init = "sobol".to_string();

        let solver = DifferentialEvolution::new(sphere, bounds.clone(), opts, "best1bin");

        // Sobol point 0 is exactly the origin of the unit cube.
        for j in 0..bounds.len() {
            assert_abs_diff_eq!(solver.population[[0, j]], bounds[j].0, epsilon = 1e-9);
        }
        // Sobol point 1 is exactly the center of the unit cube.
        for j in 0..bounds.len() {
            let mid = (bounds[j].0 + bounds[j].1) / 2.0;
            assert_abs_diff_eq!(solver.population[[1, j]], mid, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_de_init_methods() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        for init_method in &["latinhypercube", "halton", "sobol", "random"] {
            let mut opts = DifferentialEvolutionOptions::default();
            opts.seed = Some(42);
            opts.maxiter = 50;
            opts.polish = false;
            opts.init = init_method.to_string();

            let result = differential_evolution(sphere, bounds.clone(), Some(opts), None);
            assert!(
                result.is_ok(),
                "Init method {} should not fail",
                init_method
            );
        }
    }

    #[test]
    fn test_de_invalid_bounds() {
        // Empty bounds
        let result = differential_evolution(sphere, vec![], None, None);
        assert!(result.is_err());

        // Lower >= upper
        let bounds = vec![(5.0, 2.0)];
        let result = differential_evolution(sphere, bounds, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_de_convergence() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = DifferentialEvolutionOptions::default();
        opts.seed = Some(42);
        opts.maxiter = 500;
        opts.tol = 1.0; // Very relaxed tolerance so it converges quickly
        opts.polish = false;

        let result = differential_evolution(sphere, bounds, Some(opts), None);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.nit > 0);
        assert!(res.nfev > 0);
    }

    #[test]
    fn test_de_options_default() {
        let opts = DifferentialEvolutionOptions::default();
        assert_eq!(opts.maxiter, 1000);
        assert_eq!(opts.popsize, 15);
        assert!((opts.tol - 0.01).abs() < 1e-12);
        assert!((opts.recombination - 0.7).abs() < 1e-12);
        assert!(opts.polish);
        assert_eq!(opts.init, "latinhypercube");
    }

    #[test]
    fn test_de_strategy_from_str() {
        assert_eq!(Strategy::from_str("best1bin"), Some(Strategy::Best1Bin));
        assert_eq!(Strategy::from_str("rand1bin"), Some(Strategy::Rand1Bin));
        assert_eq!(
            Strategy::from_str("currenttobest1bin"),
            Some(Strategy::CurrentToBest1Bin)
        );
        assert_eq!(Strategy::from_str("best2bin"), Some(Strategy::Best2Bin));
        assert_eq!(Strategy::from_str("rand2bin"), Some(Strategy::Rand2Bin));
        assert_eq!(Strategy::from_str("best1exp"), Some(Strategy::Best1Exp));
        assert_eq!(Strategy::from_str("rand1exp"), Some(Strategy::Rand1Exp));
        assert_eq!(Strategy::from_str("unknown"), None);
    }

    #[test]
    fn test_de_high_dimensional() {
        let n = 5;
        let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); n];
        let mut opts = DifferentialEvolutionOptions::default();
        opts.seed = Some(42);
        opts.popsize = 30;
        opts.maxiter = 500;
        opts.polish = false;

        let result = differential_evolution(sphere, bounds, Some(opts), None);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            res.fun < 1.0,
            "High-dim DE should converge reasonably, got {}",
            res.fun
        );
    }
}
