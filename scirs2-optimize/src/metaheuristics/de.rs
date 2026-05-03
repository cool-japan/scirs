//! # Differential Evolution (DE) Metaheuristic
//!
//! A comprehensive implementation of Differential Evolution for continuous optimization:
//! - **DE/rand/1**: Classic random-base mutation
//! - **DE/best/1**: Best-member mutation for fast convergence
//! - **DE/rand-to-best/1**: Hybrid using both random and best members
//! - **Binomial and exponential crossover**
//! - **Self-adaptive parameter control (jDE)**
//! - **Opposition-based learning** for population initialization
//! - **Constraint handling** via penalty and feasibility rules

use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{rng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ---------------------------------------------------------------------------
// Enums & Configuration
// ---------------------------------------------------------------------------

/// DE mutation strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeStrategy {
    /// DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)
    Rand1,
    /// DE/best/1: v = x_best + F * (x_r1 - x_r2)
    Best1,
    /// DE/rand-to-best/1: v = x_ri + F * (x_best - x_ri) + F * (x_r1 - x_r2)
    RandToBest1,
}

impl Default for DeStrategy {
    fn default() -> Self {
        DeStrategy::Rand1
    }
}

/// Crossover type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossoverType {
    /// Binomial (uniform) crossover
    Binomial,
    /// Exponential crossover
    Exponential,
}

impl Default for CrossoverType {
    fn default() -> Self {
        CrossoverType::Binomial
    }
}

/// Constraint handling method for DE
#[derive(Debug, Clone)]
pub struct DeConstraintHandler {
    /// Penalty coefficient for constraint violations
    pub penalty_coeff: f64,
    /// Use feasibility rules (feasible solutions always preferred over infeasible)
    pub use_feasibility_rules: bool,
}

impl Default for DeConstraintHandler {
    fn default() -> Self {
        Self {
            penalty_coeff: 1e6,
            use_feasibility_rules: true,
        }
    }
}

/// Opposition-based learning configuration
#[derive(Debug, Clone)]
pub struct OppositionBasedInit {
    /// Enable opposition-based learning for initialization
    pub enabled: bool,
    /// Jumping rate: probability of applying opposition in each generation
    pub jumping_rate: f64,
}

impl Default for OppositionBasedInit {
    fn default() -> Self {
        Self {
            enabled: true,
            jumping_rate: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// DE Options
// ---------------------------------------------------------------------------

/// Options for Differential Evolution optimizer
#[derive(Debug, Clone)]
pub struct DeOptions {
    /// Population size (typically 5-10 times the dimension)
    pub pop_size: usize,
    /// Maximum number of generations
    pub max_generations: usize,
    /// Mutation factor F in [0, 2]
    pub mutation_factor: f64,
    /// Crossover probability CR in [0, 1]
    pub crossover_prob: f64,
    /// Mutation strategy
    pub strategy: DeStrategy,
    /// Crossover type
    pub crossover: CrossoverType,
    /// Search bounds per dimension: (lower, upper)
    pub bounds: Vec<(f64, f64)>,
    /// Random seed
    pub seed: Option<u64>,
    /// Convergence tolerance on function value spread
    pub tol: f64,
    /// Patience: generations without improvement
    pub patience: usize,
    /// Opposition-based learning
    pub opposition: OppositionBasedInit,
    /// Constraint handler
    pub constraint_handler: Option<DeConstraintHandler>,
}

impl Default for DeOptions {
    fn default() -> Self {
        Self {
            pop_size: 50,
            max_generations: 1000,
            mutation_factor: 0.8,
            crossover_prob: 0.9,
            strategy: DeStrategy::Rand1,
            crossover: CrossoverType::Binomial,
            bounds: Vec::new(),
            seed: None,
            tol: 1e-12,
            patience: 100,
            opposition: OppositionBasedInit::default(),
            constraint_handler: None,
        }
    }
}

// ---------------------------------------------------------------------------
// jDE (self-adaptive) Options
// ---------------------------------------------------------------------------

/// Options for self-adaptive jDE variant
#[derive(Debug, Clone)]
pub struct JdeOptions {
    /// Base DE options
    pub base: DeOptions,
    /// Probability of adapting F
    pub tau_f: f64,
    /// Probability of adapting CR
    pub tau_cr: f64,
    /// Lower bound for F
    pub f_lower: f64,
    /// Upper bound for F
    pub f_upper: f64,
}

impl Default for JdeOptions {
    fn default() -> Self {
        Self {
            base: DeOptions::default(),
            tau_f: 0.1,
            tau_cr: 0.1,
            f_lower: 0.1,
            f_upper: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// DE Result
// ---------------------------------------------------------------------------

/// Result from DE optimization
#[derive(Debug, Clone)]
pub struct DeResult {
    /// Best solution found
    pub x: Array1<f64>,
    /// Objective value at best solution
    pub fun: f64,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of generations
    pub generations: usize,
    /// Whether the optimization converged
    pub converged: bool,
    /// Termination message
    pub message: String,
    /// Final population fitness spread (std dev)
    pub population_spread: f64,
}

impl DeResult {
    /// Convert to standard OptimizeResults
    pub fn to_optimize_results(&self) -> OptimizeResults<f64> {
        OptimizeResults {
            x: self.x.clone(),
            fun: self.fun,
            jac: None,
            hess: None,
            constr: None,
            nit: self.generations,
            nfev: self.nfev,
            njev: 0,
            nhev: 0,
            maxcv: 0,
            message: self.message.clone(),
            success: self.converged,
            status: if self.converged { 0 } else { 1 },
        }
    }
}

// ---------------------------------------------------------------------------
// Core DE Optimizer
// ---------------------------------------------------------------------------

/// Differential Evolution optimizer
pub struct DifferentialEvolutionOptimizer {
    options: DeOptions,
    rng: StdRng,
}

impl DifferentialEvolutionOptimizer {
    /// Create a new DE optimizer
    pub fn new(options: DeOptions) -> OptimizeResult<Self> {
        if options.bounds.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "Bounds must be provided for DE".to_string(),
            ));
        }
        if options.pop_size < 4 {
            return Err(OptimizeError::InvalidParameter(
                "Population size must be >= 4 for DE".to_string(),
            ));
        }
        if options.mutation_factor < 0.0 || options.mutation_factor > 2.0 {
            return Err(OptimizeError::InvalidParameter(
                "Mutation factor F must be in [0, 2]".to_string(),
            ));
        }
        if options.crossover_prob < 0.0 || options.crossover_prob > 1.0 {
            return Err(OptimizeError::InvalidParameter(
                "Crossover probability CR must be in [0, 1]".to_string(),
            ));
        }

        let seed = options.seed.unwrap_or_else(|| rng().random());
        Ok(Self {
            options,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Optimize an unconstrained objective function
    pub fn optimize<F>(&mut self, func: F) -> OptimizeResult<DeResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        self.optimize_constrained(func, None::<fn(&ArrayView1<f64>) -> Vec<f64>>)
    }

    /// Optimize with optional constraint functions
    ///
    /// `constraints_fn` returns a vector of g_i(x) where g_i(x) > 0 means violation.
    pub fn optimize_constrained<F, G>(
        &mut self,
        func: F,
        constraints_fn: Option<G>,
    ) -> OptimizeResult<DeResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Vec<f64>,
    {
        let ndim = self.options.bounds.len();
        let np = self.options.pop_size;

        // Initialize population
        let mut population = self.initialize_population(ndim, np);
        let mut fitness: Vec<f64> = Vec::with_capacity(np);
        let mut violations: Vec<f64> = vec![0.0; np]; // total violation per member
        let mut nfev: usize = 0;

        // Evaluate initial population
        for i in 0..np {
            let row = population.row(i);
            let f_val = func(&row);
            nfev += 1;
            let viol = if let Some(ref cf) = constraints_fn {
                let v = cf(&row);
                v.iter().map(|vi| vi.max(0.0)).sum::<f64>()
            } else {
                0.0
            };
            fitness.push(self.penalized_fitness(f_val, viol));
            violations[i] = viol;
        }

        // Track best
        let mut best_idx = self.find_best(&fitness, &violations);
        let mut best_x = population.row(best_idx).to_owned();
        let mut best_fun = func(&best_x.view());
        let mut no_improve_count: usize = 0;

        for gen in 0..self.options.max_generations {
            let mut new_population = population.clone();
            let mut new_fitness = fitness.clone();
            let mut new_violations = violations.clone();

            for i in 0..np {
                // Generate mutant vector
                let mutant = self.mutate(&population, i, best_idx, ndim);

                // Crossover
                let trial = self.crossover(&population.row(i).to_owned(), &mutant, ndim);

                // Clip to bounds
                let trial_clipped = self.clip_to_bounds(&trial);

                // Evaluate trial
                let trial_view = trial_clipped.view();
                let trial_f = func(&trial_view);
                nfev += 1;
                let trial_viol = if let Some(ref cf) = constraints_fn {
                    let v = cf(&trial_view);
                    v.iter().map(|vi| vi.max(0.0)).sum::<f64>()
                } else {
                    0.0
                };
                let trial_penalized = self.penalized_fitness(trial_f, trial_viol);

                // Selection
                let replace = if self
                    .options
                    .constraint_handler
                    .as_ref()
                    .map_or(false, |ch| ch.use_feasibility_rules)
                {
                    self.feasibility_selection(
                        trial_penalized,
                        trial_viol,
                        fitness[i],
                        violations[i],
                    )
                } else {
                    trial_penalized <= fitness[i]
                };

                if replace {
                    for d in 0..ndim {
                        new_population[[i, d]] = trial_clipped[d];
                    }
                    new_fitness[i] = trial_penalized;
                    new_violations[i] = trial_viol;
                }
            }

            population = new_population;
            fitness = new_fitness;
            violations = new_violations;

            // Update best
            let new_best_idx = self.find_best(&fitness, &violations);
            let candidate_fun = func(&population.row(new_best_idx));

            if candidate_fun < best_fun {
                best_idx = new_best_idx;
                best_x = population.row(best_idx).to_owned();
                best_fun = candidate_fun;
                no_improve_count = 0;
            } else {
                best_idx = new_best_idx;
                no_improve_count += 1;
            }

            // Check convergence: population spread
            let spread = self.population_spread(&fitness);
            if spread < self.options.tol {
                return Ok(DeResult {
                    x: best_x,
                    fun: best_fun,
                    nfev,
                    generations: gen + 1,
                    converged: true,
                    message: format!(
                        "DE converged: spread {:.2e} < tol {:.2e} at generation {}",
                        spread,
                        self.options.tol,
                        gen + 1
                    ),
                    population_spread: spread,
                });
            }

            if no_improve_count >= self.options.patience {
                return Ok(DeResult {
                    x: best_x,
                    fun: best_fun,
                    nfev,
                    generations: gen + 1,
                    converged: true,
                    message: format!(
                        "DE converged: no improvement for {} generations",
                        self.options.patience
                    ),
                    population_spread: spread,
                });
            }
        }

        let spread = self.population_spread(&fitness);
        Ok(DeResult {
            x: best_x,
            fun: best_fun,
            nfev,
            generations: self.options.max_generations,
            converged: false,
            message: format!(
                "DE completed {} generations without full convergence",
                self.options.max_generations
            ),
            population_spread: spread,
        })
    }

    // --- Population Initialization ---

    fn initialize_population(&mut self, ndim: usize, np: usize) -> Array2<f64> {
        let mut pop = Array2::zeros((np, ndim));

        // Random initialization within bounds
        for i in 0..np {
            for d in 0..ndim {
                let (lo, hi) = self.options.bounds[d];
                pop[[i, d]] = lo + self.rng.random::<f64>() * (hi - lo);
            }
        }

        // Opposition-based learning: double the candidates, keep the best NP
        if self.options.opposition.enabled {
            let mut all_candidates = Vec::with_capacity(2 * np);
            for i in 0..np {
                let mut member = Vec::with_capacity(ndim);
                let mut opposite = Vec::with_capacity(ndim);
                for d in 0..ndim {
                    let val = pop[[i, d]];
                    member.push(val);
                    let (lo, hi) = self.options.bounds[d];
                    opposite.push(lo + hi - val);
                }
                all_candidates.push(member);
                all_candidates.push(opposite);
            }

            // We just return the first NP as-is plus opposition awareness
            // For full OBL, one would evaluate and select top NP, but
            // that requires the objective function here. So we interleave.
            for i in 0..np {
                if i < all_candidates.len() / 2 {
                    // Use opposition for every other member
                    let opp_idx = 2 * i + 1;
                    if opp_idx < all_candidates.len() && i % 2 == 1 {
                        for d in 0..ndim {
                            pop[[i, d]] = all_candidates[opp_idx][d];
                        }
                    }
                }
            }
        }

        pop
    }

    // --- Mutation ---

    fn mutate(
        &mut self,
        population: &Array2<f64>,
        target_idx: usize,
        best_idx: usize,
        ndim: usize,
    ) -> Array1<f64> {
        let np = population.nrows();
        let f = self.options.mutation_factor;

        match self.options.strategy {
            DeStrategy::Rand1 => {
                let (r1, r2, r3) = self.pick_three_distinct(np, target_idx);
                let mut mutant = Array1::zeros(ndim);
                for d in 0..ndim {
                    mutant[d] =
                        population[[r1, d]] + f * (population[[r2, d]] - population[[r3, d]]);
                }
                mutant
            }
            DeStrategy::Best1 => {
                let (r1, r2) = self.pick_two_distinct(np, target_idx);
                let mut mutant = Array1::zeros(ndim);
                for d in 0..ndim {
                    mutant[d] =
                        population[[best_idx, d]] + f * (population[[r1, d]] - population[[r2, d]]);
                }
                mutant
            }
            DeStrategy::RandToBest1 => {
                let (r1, r2) = self.pick_two_distinct(np, target_idx);
                let mut mutant = Array1::zeros(ndim);
                for d in 0..ndim {
                    mutant[d] = population[[target_idx, d]]
                        + f * (population[[best_idx, d]] - population[[target_idx, d]])
                        + f * (population[[r1, d]] - population[[r2, d]]);
                }
                mutant
            }
        }
    }

    // --- Crossover ---

    fn crossover(
        &mut self,
        target: &Array1<f64>,
        mutant: &Array1<f64>,
        ndim: usize,
    ) -> Array1<f64> {
        match self.options.crossover {
            CrossoverType::Binomial => self.binomial_crossover(target, mutant, ndim),
            CrossoverType::Exponential => self.exponential_crossover(target, mutant, ndim),
        }
    }

    fn binomial_crossover(
        &mut self,
        target: &Array1<f64>,
        mutant: &Array1<f64>,
        ndim: usize,
    ) -> Array1<f64> {
        let cr = self.options.crossover_prob;
        let j_rand = self.rng.random_range(0..ndim);
        let mut trial = target.clone();
        for d in 0..ndim {
            if self.rng.random::<f64>() < cr || d == j_rand {
                trial[d] = mutant[d];
            }
        }
        trial
    }

    fn exponential_crossover(
        &mut self,
        target: &Array1<f64>,
        mutant: &Array1<f64>,
        ndim: usize,
    ) -> Array1<f64> {
        let cr = self.options.crossover_prob;
        let mut trial = target.clone();
        let start = self.rng.random_range(0..ndim);
        let mut d = start;
        loop {
            trial[d] = mutant[d];
            d = (d + 1) % ndim;
            if d == start || self.rng.random::<f64>() >= cr {
                break;
            }
        }
        trial
    }

    // --- Helpers ---

    fn clip_to_bounds(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut clipped = x.clone();
        for (d, (lo, hi)) in self.options.bounds.iter().enumerate() {
            if d < clipped.len() {
                clipped[d] = clipped[d].clamp(*lo, *hi);
            }
        }
        clipped
    }

    fn penalized_fitness(&self, obj: f64, violation: f64) -> f64 {
        if let Some(ref ch) = self.options.constraint_handler {
            obj + ch.penalty_coeff * violation
        } else {
            obj
        }
    }

    fn feasibility_selection(
        &self,
        trial_fit: f64,
        trial_viol: f64,
        current_fit: f64,
        current_viol: f64,
    ) -> bool {
        let trial_feasible = trial_viol <= 1e-15;
        let current_feasible = current_viol <= 1e-15;

        match (trial_feasible, current_feasible) {
            (true, true) => trial_fit <= current_fit,
            (true, false) => true,  // feasible beats infeasible
            (false, true) => false, // infeasible loses to feasible
            (false, false) => trial_viol < current_viol, // less violation wins
        }
    }

    fn find_best(&self, fitness: &[f64], violations: &[f64]) -> usize {
        let mut best_idx = 0;
        for i in 1..fitness.len() {
            let is_better = if self
                .options
                .constraint_handler
                .as_ref()
                .map_or(false, |ch| ch.use_feasibility_rules)
            {
                self.feasibility_selection(
                    fitness[i],
                    violations[i],
                    fitness[best_idx],
                    violations[best_idx],
                )
            } else {
                fitness[i] < fitness[best_idx]
            };
            if is_better {
                best_idx = i;
            }
        }
        best_idx
    }

    fn population_spread(&self, fitness: &[f64]) -> f64 {
        if fitness.is_empty() {
            return 0.0;
        }
        let mean = fitness.iter().sum::<f64>() / fitness.len() as f64;
        let variance =
            fitness.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / fitness.len() as f64;
        variance.sqrt()
    }

    fn pick_three_distinct(&mut self, np: usize, exclude: usize) -> (usize, usize, usize) {
        let mut r1 = self.rng.random_range(0..np);
        while r1 == exclude {
            r1 = self.rng.random_range(0..np);
        }
        let mut r2 = self.rng.random_range(0..np);
        while r2 == exclude || r2 == r1 {
            r2 = self.rng.random_range(0..np);
        }
        let mut r3 = self.rng.random_range(0..np);
        while r3 == exclude || r3 == r1 || r3 == r2 {
            r3 = self.rng.random_range(0..np);
        }
        (r1, r2, r3)
    }

    fn pick_two_distinct(&mut self, np: usize, exclude: usize) -> (usize, usize) {
        let mut r1 = self.rng.random_range(0..np);
        while r1 == exclude {
            r1 = self.rng.random_range(0..np);
        }
        let mut r2 = self.rng.random_range(0..np);
        while r2 == exclude || r2 == r1 {
            r2 = self.rng.random_range(0..np);
        }
        (r1, r2)
    }
}

// ---------------------------------------------------------------------------
// Self-Adaptive jDE
// ---------------------------------------------------------------------------

/// Self-adaptive Differential Evolution (jDE) optimizer
///
/// Automatically adapts F and CR during the search (Brest et al. 2006).
pub fn jde_optimize<F>(func: F, options: JdeOptions) -> OptimizeResult<DeResult>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    if options.base.bounds.is_empty() {
        return Err(OptimizeError::InvalidInput(
            "Bounds must be provided for jDE".to_string(),
        ));
    }
    let ndim = options.base.bounds.len();
    let np = options.base.pop_size;
    if np < 4 {
        return Err(OptimizeError::InvalidParameter(
            "Population size must be >= 4".to_string(),
        ));
    }

    let seed = options.base.seed.unwrap_or_else(|| rng().random());
    let mut local_rng = StdRng::seed_from_u64(seed);
    let bounds = &options.base.bounds;

    // Initialize population
    let mut population = Array2::zeros((np, ndim));
    for i in 0..np {
        for d in 0..ndim {
            let (lo, hi) = bounds[d];
            population[[i, d]] = lo + local_rng.random::<f64>() * (hi - lo);
        }
    }

    // Per-member F and CR
    let mut f_vec = vec![options.base.mutation_factor; np];
    let mut cr_vec = vec![options.base.crossover_prob; np];

    // Evaluate
    let mut fitness: Vec<f64> = (0..np).map(|i| func(&population.row(i))).collect();
    let mut nfev = np;

    let mut best_idx = fitness
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut best_x = population.row(best_idx).to_owned();
    let mut best_fun = fitness[best_idx];
    let mut no_improve: usize = 0;

    for gen in 0..options.base.max_generations {
        let mut new_pop = population.clone();
        let mut new_fitness = fitness.clone();
        let mut new_f_vec = f_vec.clone();
        let mut new_cr_vec = cr_vec.clone();

        for i in 0..np {
            // Self-adapt F
            let fi = if local_rng.random::<f64>() < options.tau_f {
                let new_f = options.f_lower + local_rng.random::<f64>() * options.f_upper;
                new_f_vec[i] = new_f;
                new_f
            } else {
                f_vec[i]
            };

            // Self-adapt CR
            let cri = if local_rng.random::<f64>() < options.tau_cr {
                let new_cr = local_rng.random::<f64>();
                new_cr_vec[i] = new_cr;
                new_cr
            } else {
                cr_vec[i]
            };

            // DE/rand/1 mutation with adapted F
            let (r1, r2, r3) = pick_three_distinct_rng(&mut local_rng, np, i);
            let mut mutant = Array1::zeros(ndim);
            for d in 0..ndim {
                mutant[d] = population[[r1, d]] + fi * (population[[r2, d]] - population[[r3, d]]);
            }

            // Binomial crossover with adapted CR
            let j_rand = local_rng.random_range(0..ndim);
            let mut trial = Array1::zeros(ndim);
            for d in 0..ndim {
                if local_rng.random::<f64>() < cri || d == j_rand {
                    trial[d] = mutant[d];
                } else {
                    trial[d] = population[[i, d]];
                }
            }

            // Clip
            for d in 0..ndim {
                let (lo, hi) = bounds[d];
                trial[d] = trial[d].clamp(lo, hi);
            }

            let trial_f = func(&trial.view());
            nfev += 1;

            if trial_f <= fitness[i] {
                for d in 0..ndim {
                    new_pop[[i, d]] = trial[d];
                }
                new_fitness[i] = trial_f;
            } else {
                // Revert F and CR adaptation
                new_f_vec[i] = f_vec[i];
                new_cr_vec[i] = cr_vec[i];
            }
        }

        population = new_pop;
        fitness = new_fitness;
        f_vec = new_f_vec;
        cr_vec = new_cr_vec;

        // Update best
        let new_best_idx = fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if fitness[new_best_idx] < best_fun {
            best_idx = new_best_idx;
            best_x = population.row(best_idx).to_owned();
            best_fun = fitness[best_idx];
            no_improve = 0;
        } else {
            no_improve += 1;
        }

        // Check convergence
        let spread = {
            let mean = fitness.iter().sum::<f64>() / np as f64;
            let var = fitness.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / np as f64;
            var.sqrt()
        };

        if spread < options.base.tol {
            return Ok(DeResult {
                x: best_x,
                fun: best_fun,
                nfev,
                generations: gen + 1,
                converged: true,
                message: format!("jDE converged at generation {}", gen + 1),
                population_spread: spread,
            });
        }

        if no_improve >= options.base.patience {
            return Ok(DeResult {
                x: best_x,
                fun: best_fun,
                nfev,
                generations: gen + 1,
                converged: true,
                message: format!(
                    "jDE: no improvement for {} generations",
                    options.base.patience
                ),
                population_spread: spread,
            });
        }
    }

    let spread = {
        let mean = fitness.iter().sum::<f64>() / np as f64;
        let var = fitness.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / np as f64;
        var.sqrt()
    };

    Ok(DeResult {
        x: best_x,
        fun: best_fun,
        nfev,
        generations: options.base.max_generations,
        converged: false,
        message: "jDE completed max generations".to_string(),
        population_spread: spread,
    })
}

fn pick_three_distinct_rng(
    rng_ref: &mut StdRng,
    np: usize,
    exclude: usize,
) -> (usize, usize, usize) {
    let mut r1 = rng_ref.random_range(0..np);
    while r1 == exclude {
        r1 = rng_ref.random_range(0..np);
    }
    let mut r2 = rng_ref.random_range(0..np);
    while r2 == exclude || r2 == r1 {
        r2 = rng_ref.random_range(0..np);
    }
    let mut r3 = rng_ref.random_range(0..np);
    while r3 == exclude || r3 == r1 || r3 == r2 {
        r3 = rng_ref.random_range(0..np);
    }
    (r1, r2, r3)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn sphere(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    fn rastrigin(x: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    // --- DE/rand/1 tests ---

    #[test]
    fn test_de_rand1_sphere() {
        let opts = DeOptions {
            pop_size: 30,
            max_generations: 500,
            mutation_factor: 0.8,
            crossover_prob: 0.9,
            strategy: DeStrategy::Rand1,
            crossover: CrossoverType::Binomial,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            patience: 200,
            ..Default::default()
        };

        let mut de = DifferentialEvolutionOptimizer::new(opts).expect("valid options");
        let result = de.optimize(sphere).expect("DE should optimize sphere");

        assert!(result.fun < 1e-4, "DE/rand/1 sphere: got {}", result.fun);
        assert!(result.nfev > 0);
    }

    #[test]
    fn test_de_best1_sphere() {
        let opts = DeOptions {
            pop_size: 30,
            max_generations: 500,
            mutation_factor: 0.5,
            crossover_prob: 0.9,
            strategy: DeStrategy::Best1,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            patience: 200,
            ..Default::default()
        };

        let mut de = DifferentialEvolutionOptimizer::new(opts).expect("valid options");
        let result = de.optimize(sphere).expect("DE/best/1 should work");

        assert!(result.fun < 1e-4, "DE/best/1 sphere: got {}", result.fun);
    }

    #[test]
    fn test_de_rand_to_best1_sphere() {
        let opts = DeOptions {
            pop_size: 30,
            max_generations: 500,
            mutation_factor: 0.7,
            crossover_prob: 0.9,
            strategy: DeStrategy::RandToBest1,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            patience: 200,
            ..Default::default()
        };

        let mut de = DifferentialEvolutionOptimizer::new(opts).expect("valid options");
        let result = de.optimize(sphere).expect("DE/rand-to-best/1 should work");

        assert!(
            result.fun < 1e-3,
            "DE/rand-to-best/1 sphere: got {}",
            result.fun
        );
    }

    // --- Crossover type tests ---

    #[test]
    fn test_de_exponential_crossover() {
        let opts = DeOptions {
            pop_size: 30,
            max_generations: 500,
            mutation_factor: 0.8,
            crossover_prob: 0.9,
            strategy: DeStrategy::Rand1,
            crossover: CrossoverType::Exponential,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            patience: 200,
            ..Default::default()
        };

        let mut de = DifferentialEvolutionOptimizer::new(opts).expect("valid options");
        let result = de.optimize(sphere).expect("exp crossover should work");

        assert!(
            result.fun < 0.1,
            "Exponential crossover sphere: got {}",
            result.fun
        );
    }

    // --- Rastrigin (multimodal) ---

    #[test]
    fn test_de_rastrigin() {
        let opts = DeOptions {
            pop_size: 50,
            max_generations: 1000,
            mutation_factor: 0.8,
            crossover_prob: 0.9,
            strategy: DeStrategy::Rand1,
            bounds: vec![(-5.12, 5.12); 3],
            seed: Some(42),
            patience: 300,
            ..Default::default()
        };

        let mut de = DifferentialEvolutionOptimizer::new(opts).expect("valid options");
        let result = de.optimize(rastrigin).expect("DE on rastrigin");

        assert!(result.fun < 10.0, "DE rastrigin: got {}", result.fun);
    }

    // --- jDE tests ---

    #[test]
    fn test_jde_sphere() {
        let opts = JdeOptions {
            base: DeOptions {
                pop_size: 30,
                max_generations: 500,
                mutation_factor: 0.5,
                crossover_prob: 0.9,
                bounds: vec![(-5.0, 5.0); 2],
                seed: Some(42),
                patience: 200,
                ..Default::default()
            },
            tau_f: 0.1,
            tau_cr: 0.1,
            f_lower: 0.1,
            f_upper: 0.9,
        };

        let result = jde_optimize(sphere, opts).expect("jDE should work");

        assert!(result.fun < 1e-4, "jDE sphere: got {}", result.fun);
    }

    #[test]
    fn test_jde_rosenbrock() {
        let opts = JdeOptions {
            base: DeOptions {
                pop_size: 40,
                max_generations: 2000,
                bounds: vec![(-5.0, 5.0); 2],
                seed: Some(42),
                patience: 500,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = jde_optimize(rosenbrock, opts).expect("jDE on rosenbrock");

        assert!(result.fun < 1.0, "jDE rosenbrock: got {}", result.fun);
    }

    // --- Constraint handling tests ---

    #[test]
    fn test_de_with_constraints() {
        // Minimize x^2 + y^2 subject to x + y >= 2
        let constraints = |x: &ArrayView1<f64>| -> Vec<f64> {
            vec![2.0 - (x[0] + x[1])] // violation when x+y < 2
        };

        let opts = DeOptions {
            pop_size: 40,
            max_generations: 500,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            patience: 200,
            constraint_handler: Some(DeConstraintHandler {
                penalty_coeff: 1e4,
                use_feasibility_rules: true,
            }),
            ..Default::default()
        };

        let mut de = DifferentialEvolutionOptimizer::new(opts).expect("valid options");
        let result = de
            .optimize_constrained(sphere, Some(constraints))
            .expect("constrained DE should work");

        // Optimal: x = y = 1, f = 2
        let sum = result.x[0] + result.x[1];
        assert!(sum >= 1.5, "Constraint should be ~satisfied: sum = {}", sum);
        assert!(result.fun < 5.0, "Constrained DE fun: {}", result.fun);
    }

    // --- Opposition-based learning ---

    #[test]
    fn test_de_opposition_based_init() {
        let opts = DeOptions {
            pop_size: 30,
            max_generations: 300,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            opposition: OppositionBasedInit {
                enabled: true,
                jumping_rate: 0.3,
            },
            patience: 150,
            ..Default::default()
        };

        let mut de = DifferentialEvolutionOptimizer::new(opts).expect("valid options");
        let result = de.optimize(sphere).expect("OBL DE should work");

        assert!(result.fun < 1.0, "OBL DE sphere: got {}", result.fun);
    }

    #[test]
    fn test_de_no_opposition() {
        let opts = DeOptions {
            pop_size: 30,
            max_generations: 300,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            opposition: OppositionBasedInit {
                enabled: false,
                jumping_rate: 0.0,
            },
            patience: 150,
            ..Default::default()
        };

        let mut de = DifferentialEvolutionOptimizer::new(opts).expect("valid options");
        let result = de.optimize(sphere).expect("DE without OBL should work");

        assert!(result.fun < 1.0, "DE no-OBL sphere: got {}", result.fun);
    }

    // --- Edge cases ---

    #[test]
    fn test_de_empty_bounds_error() {
        let opts = DeOptions {
            bounds: vec![],
            ..Default::default()
        };
        let result = DifferentialEvolutionOptimizer::new(opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_de_small_popsize_error() {
        let opts = DeOptions {
            pop_size: 2,
            bounds: vec![(-1.0, 1.0)],
            ..Default::default()
        };
        let result = DifferentialEvolutionOptimizer::new(opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_de_invalid_mutation_error() {
        let opts = DeOptions {
            mutation_factor: 3.0,
            bounds: vec![(-1.0, 1.0)],
            ..Default::default()
        };
        let result = DifferentialEvolutionOptimizer::new(opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_de_invalid_crossover_error() {
        let opts = DeOptions {
            crossover_prob: 1.5,
            bounds: vec![(-1.0, 1.0)],
            ..Default::default()
        };
        let result = DifferentialEvolutionOptimizer::new(opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_de_to_optimize_results() {
        let de_result = DeResult {
            x: array![1.0, 2.0],
            fun: 5.0,
            nfev: 1000,
            generations: 50,
            converged: true,
            message: "test".to_string(),
            population_spread: 0.01,
        };
        let opt = de_result.to_optimize_results();
        assert_eq!(opt.nfev, 1000);
        assert_eq!(opt.nit, 50);
        assert!(opt.success);
    }

    #[test]
    fn test_jde_empty_bounds_error() {
        let opts = JdeOptions {
            base: DeOptions {
                bounds: vec![],
                ..Default::default()
            },
            ..Default::default()
        };
        let result = jde_optimize(sphere, opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_de_1d() {
        let opts = DeOptions {
            pop_size: 20,
            max_generations: 200,
            bounds: vec![(-10.0, 10.0)],
            seed: Some(42),
            patience: 100,
            ..Default::default()
        };

        let mut de = DifferentialEvolutionOptimizer::new(opts).expect("valid");
        let result = de
            .optimize(|x: &ArrayView1<f64>| (x[0] - 3.0).powi(2))
            .expect("1D DE");

        assert!(
            (result.x[0] - 3.0).abs() < 0.5,
            "1D DE: x = {}",
            result.x[0]
        );
    }
}
