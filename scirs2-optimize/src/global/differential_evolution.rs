//! Differential Evolution algorithm for global optimization
//!
//! The differential evolution method is a stochastic global optimization
//! algorithm that does not use gradient methods to find the minimum and
//! can search large areas of candidate space.

use crate::error::OptimizeError;
use crate::parallel::{parallel_evaluate_batch, ParallelOptions};
use crate::unconstrained::{
    minimize, Bounds as UnconstrainedBounds, Method, OptimizeResult, Options,
};
use ndarray::{Array1, ArrayView1};
use rand::distr::Uniform;
use rand::prelude::*;
use rand::rngs::StdRng;

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
    rng: StdRng,
    nfev: usize,
}

use ndarray::Array2;

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

        let seed = options.seed.unwrap_or_else(rand::random);
        let rng = StdRng::seed_from_u64(seed);

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

        solver.init_population();
        solver
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

        // If x0 is provided, replace one member with it
        if let Some(ref x0) = self.options.x0 {
            for (i, &val) in x0.iter().enumerate() {
                self.population[[0, i]] = val;
            }
        }

        // Evaluate initial population
        if self.options.parallel.is_some() {
            // Parallel evaluation
            let candidates: Vec<Array1<f64>> = (0..popsize)
                .map(|i| self.population.row(i).to_owned())
                .collect();

            // Extract parallel options for evaluation
            let parallel_opts = self.options.parallel.as_ref().unwrap();

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
                let uniform = Uniform::new(lb, ub).unwrap();
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
                let uniform = Uniform::new(segment_lb, segment_ub).unwrap();
                self.population[[i, j]] = self.rng.sample(uniform);
            }
        }
    }

    /// Initialize population using Halton sequence
    fn init_halton(&mut self) {
        // Simplified Halton sequence implementation
        self.init_random(); // Fallback to random for now
    }

    /// Initialize population using Sobol sequence
    fn init_sobol(&mut self) {
        // Simplified Sobol sequence implementation
        self.init_random(); // Fallback to random for now
    }

    /// Ensure bounds for a parameter
    fn ensure_bounds(&self, idx: usize, val: f64) -> f64 {
        let (lb, ub) = self.bounds[idx];
        val.max(lb).min(ub)
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
                    mutant[i] = self.ensure_bounds(i, mutant[i]);
                }
            }
            Strategy::Rand1Bin | Strategy::Rand1Exp => {
                // mutant = r0 + F * (r1 - r2)
                let r0 = self.population.row(indices[0]);
                let r1 = self.population.row(indices[1]);
                let r2 = self.population.row(indices[2]);
                for i in 0..self.ndim {
                    mutant[i] = r0[i] + mutation_factor * (r1[i] - r2[i]);
                    mutant[i] = self.ensure_bounds(i, mutant[i]);
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
                    mutant[i] = self.ensure_bounds(i, mutant[i]);
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
                    mutant[i] = self.ensure_bounds(i, mutant[i]);
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
                    mutant[i] = self.ensure_bounds(i, mutant[i]);
                }
            }
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
                    if i == randn || self.rng.random::<f64>() < self.options.recombination {
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
                    if i == randn || self.rng.random::<f64>() >= self.options.recombination {
                        break;
                    }
                }
            }
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
                .map(|(trial, _)| trial.clone())
                .collect();

            // Extract the parallel options for evaluation
            let parallel_opts = self.options.parallel.as_ref().unwrap();

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
            iterations: nit,
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
                        .unwrap(),
                    ),
                    ..Default::default()
                }),
            )
            .unwrap();
            if local_result.success && local_result.fun < result.fun {
                result.x = local_result.x;
                result.fun = local_result.fun;
                result.nfev += local_result.nfev;
                result.func_evals = result.nfev;
            }
        }

        result
    }
}

/// Perform global optimization using differential evolution
pub fn differential_evolution<F>(
    func: F,
    bounds: Bounds,
    options: Option<DifferentialEvolutionOptions>,
    strategy: Option<&str>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone + Sync,
{
    let options = options.unwrap_or_default();
    let strategy = strategy.unwrap_or("best1bin");

    let mut solver = DifferentialEvolution::new(func, bounds, options, strategy);
    Ok(solver.run())
}
