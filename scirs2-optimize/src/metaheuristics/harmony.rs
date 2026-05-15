//! # Harmony Search (HS) Metaheuristic
//!
//! Music-inspired optimization algorithm with:
//! - **Basic Harmony Search**: Classic HS with HMCR, PAR, and bandwidth
//! - **Improved Harmony Search (IHS)**: Dynamic HMCR and PAR adaptation
//! - **Global-best Harmony Search**: Uses best harmony to guide search
//! - **Multi-objective variant**: Pareto-based HS for multi-objective problems

use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{rng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Options for basic Harmony Search
#[derive(Debug, Clone)]
pub struct HarmonySearchOptions {
    /// Harmony memory size (number of stored solutions)
    pub hms: usize,
    /// Maximum improvisations (iterations)
    pub max_improvisations: usize,
    /// Harmony Memory Considering Rate (probability of picking from memory)
    pub hmcr: f64,
    /// Pitch Adjustment Rate (probability of adjusting a memory-picked value)
    pub par: f64,
    /// Bandwidth for pitch adjustment (absolute perturbation range)
    pub bandwidth: f64,
    /// Search bounds per dimension
    pub bounds: Vec<(f64, f64)>,
    /// Random seed
    pub seed: Option<u64>,
    /// Convergence tolerance
    pub tol: f64,
    /// Patience for early stopping
    pub patience: usize,
}

impl Default for HarmonySearchOptions {
    fn default() -> Self {
        Self {
            hms: 20,
            max_improvisations: 10_000,
            hmcr: 0.9,
            par: 0.3,
            bandwidth: 0.01,
            bounds: Vec::new(),
            seed: None,
            tol: 1e-12,
            patience: 500,
        }
    }
}

/// Options for Improved Harmony Search (IHS)
///
/// Dynamically adjusts HMCR and PAR over the course of the search.
#[derive(Debug, Clone)]
pub struct ImprovedHarmonySearchOptions {
    /// Base options (hms, max_improvisations, bounds, etc.)
    pub base: HarmonySearchOptions,
    /// Minimum HMCR (at start)
    pub hmcr_min: f64,
    /// Maximum HMCR (at end)
    pub hmcr_max: f64,
    /// Minimum PAR (at end)
    pub par_min: f64,
    /// Maximum PAR (at start)
    pub par_max: f64,
    /// Minimum bandwidth (at end)
    pub bw_min: f64,
    /// Maximum bandwidth (at start)
    pub bw_max: f64,
}

impl Default for ImprovedHarmonySearchOptions {
    fn default() -> Self {
        Self {
            base: HarmonySearchOptions::default(),
            hmcr_min: 0.7,
            hmcr_max: 0.99,
            par_min: 0.01,
            par_max: 0.99,
            bw_min: 1e-5,
            bw_max: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result from Harmony Search
#[derive(Debug, Clone)]
pub struct HarmonySearchResult {
    /// Best solution found
    pub x: Array1<f64>,
    /// Objective value at best solution
    pub fun: f64,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of improvisations performed
    pub improvisations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Termination message
    pub message: String,
}

impl HarmonySearchResult {
    /// Convert to standard OptimizeResults
    pub fn to_optimize_results(&self) -> OptimizeResults<f64> {
        OptimizeResults {
            x: self.x.clone(),
            fun: self.fun,
            jac: None,
            hess: None,
            constr: None,
            nit: self.improvisations,
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
// Basic Harmony Search
// ---------------------------------------------------------------------------

/// Basic Harmony Search optimizer
pub struct HarmonySearchOptimizer {
    options: HarmonySearchOptions,
    rng: StdRng,
}

impl HarmonySearchOptimizer {
    /// Create a new HS optimizer
    pub fn new(options: HarmonySearchOptions) -> OptimizeResult<Self> {
        Self::validate_options(&options)?;
        let seed = options.seed.unwrap_or_else(|| rng().random());
        Ok(Self {
            options,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    pub(crate) fn validate_options(opts: &HarmonySearchOptions) -> OptimizeResult<()> {
        if opts.bounds.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "Bounds must be provided for Harmony Search".to_string(),
            ));
        }
        if opts.hms < 2 {
            return Err(OptimizeError::InvalidParameter(
                "Harmony memory size must be >= 2".to_string(),
            ));
        }
        if opts.hmcr < 0.0 || opts.hmcr > 1.0 {
            return Err(OptimizeError::InvalidParameter(
                "HMCR must be in [0, 1]".to_string(),
            ));
        }
        if opts.par < 0.0 || opts.par > 1.0 {
            return Err(OptimizeError::InvalidParameter(
                "PAR must be in [0, 1]".to_string(),
            ));
        }
        Ok(())
    }

    /// Run the harmony search
    pub fn optimize<F>(&mut self, func: F) -> OptimizeResult<HarmonySearchResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let ndim = self.options.bounds.len();
        let hms = self.options.hms;

        // Initialize harmony memory with random solutions
        let mut memory: Vec<Array1<f64>> = Vec::with_capacity(hms);
        let mut memory_fitness: Vec<f64> = Vec::with_capacity(hms);
        let mut nfev: usize = 0;

        for _ in 0..hms {
            let harmony = self.random_harmony(ndim);
            let f_val = func(&harmony.view());
            nfev += 1;
            memory_fitness.push(f_val);
            memory.push(harmony);
        }

        let mut best_idx = self.best_index(&memory_fitness);
        let mut best_fun = memory_fitness[best_idx];
        let mut no_improve: usize = 0;

        for imp in 0..self.options.max_improvisations {
            // Improvise a new harmony
            let new_harmony = self.improvise(&memory, ndim);
            let new_f = func(&new_harmony.view());
            nfev += 1;

            // Find the worst in memory
            let worst_idx = self.worst_index(&memory_fitness);

            // Replace worst if new is better
            if new_f < memory_fitness[worst_idx] {
                memory[worst_idx] = new_harmony;
                memory_fitness[worst_idx] = new_f;

                if new_f < best_fun {
                    best_fun = new_f;
                    best_idx = worst_idx;
                    no_improve = 0;
                } else {
                    no_improve += 1;
                }
            } else {
                no_improve += 1;
            }

            // Check convergence
            let spread = self.fitness_spread(&memory_fitness);
            if spread < self.options.tol {
                let bi = self.best_index(&memory_fitness);
                return Ok(HarmonySearchResult {
                    x: memory[bi].clone(),
                    fun: memory_fitness[bi],
                    nfev,
                    improvisations: imp + 1,
                    converged: true,
                    message: format!(
                        "HS converged: spread {:.2e} at improvisation {}",
                        spread,
                        imp + 1
                    ),
                });
            }

            if no_improve >= self.options.patience {
                let bi = self.best_index(&memory_fitness);
                return Ok(HarmonySearchResult {
                    x: memory[bi].clone(),
                    fun: memory_fitness[bi],
                    nfev,
                    improvisations: imp + 1,
                    converged: true,
                    message: format!(
                        "HS: no improvement for {} improvisations",
                        self.options.patience
                    ),
                });
            }
        }

        let bi = self.best_index(&memory_fitness);
        Ok(HarmonySearchResult {
            x: memory[bi].clone(),
            fun: memory_fitness[bi],
            nfev,
            improvisations: self.options.max_improvisations,
            converged: false,
            message: format!(
                "HS completed {} improvisations",
                self.options.max_improvisations
            ),
        })
    }

    /// Improvise a new harmony vector
    fn improvise(&mut self, memory: &[Array1<f64>], ndim: usize) -> Array1<f64> {
        let mut harmony = Array1::zeros(ndim);
        let hms = memory.len();

        for d in 0..ndim {
            let (lo, hi) = self.options.bounds[d];
            if self.rng.random::<f64>() < self.options.hmcr {
                // Memory consideration: pick from harmony memory
                let mem_idx = self.rng.random_range(0..hms);
                harmony[d] = memory[mem_idx][d];

                // Pitch adjustment
                if self.rng.random::<f64>() < self.options.par {
                    let adjustment =
                        (self.rng.random::<f64>() * 2.0 - 1.0) * self.options.bandwidth;
                    harmony[d] += adjustment;
                }
            } else {
                // Random selection
                harmony[d] = lo + self.rng.random::<f64>() * (hi - lo);
            }
            // Enforce bounds
            harmony[d] = harmony[d].clamp(lo, hi);
        }
        harmony
    }

    fn random_harmony(&mut self, ndim: usize) -> Array1<f64> {
        Array1::from_vec(
            (0..ndim)
                .map(|d| {
                    let (lo, hi) = self.options.bounds[d];
                    lo + self.rng.random::<f64>() * (hi - lo)
                })
                .collect(),
        )
    }

    fn best_index(&self, fitness: &[f64]) -> usize {
        fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn worst_index(&self, fitness: &[f64]) -> usize {
        fitness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn fitness_spread(&self, fitness: &[f64]) -> f64 {
        if fitness.is_empty() {
            return 0.0;
        }
        let mean = fitness.iter().sum::<f64>() / fitness.len() as f64;
        let var = fitness.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / fitness.len() as f64;
        var.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Improved Harmony Search (IHS)
// ---------------------------------------------------------------------------

/// Improved Harmony Search with dynamic parameter adaptation
///
/// Linearly interpolates HMCR, PAR, and bandwidth over the search.
pub fn improved_harmony_search<F>(
    func: F,
    options: ImprovedHarmonySearchOptions,
) -> OptimizeResult<HarmonySearchResult>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let base = &options.base;
    if base.bounds.is_empty() {
        return Err(OptimizeError::InvalidInput(
            "Bounds must be provided".to_string(),
        ));
    }
    if base.hms < 2 {
        return Err(OptimizeError::InvalidParameter(
            "HMS must be >= 2".to_string(),
        ));
    }

    let ndim = base.bounds.len();
    let hms = base.hms;
    let max_imp = base.max_improvisations;
    let seed = base.seed.unwrap_or_else(|| rng().random());
    let mut local_rng = StdRng::seed_from_u64(seed);

    // Initialize memory
    let mut memory: Vec<Array1<f64>> = Vec::with_capacity(hms);
    let mut memory_fitness: Vec<f64> = Vec::with_capacity(hms);
    let mut nfev: usize = 0;

    for _ in 0..hms {
        let h = Array1::from_vec(
            (0..ndim)
                .map(|d| {
                    let (lo, hi) = base.bounds[d];
                    lo + local_rng.random::<f64>() * (hi - lo)
                })
                .collect(),
        );
        let fv = func(&h.view());
        nfev += 1;
        memory_fitness.push(fv);
        memory.push(h);
    }

    let mut best_fun = *memory_fitness
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(&f64::INFINITY);
    let mut no_improve: usize = 0;

    for imp in 0..max_imp {
        let t = imp as f64 / max_imp.max(1) as f64;

        // Dynamic parameters (linearly interpolated)
        let hmcr = options.hmcr_min + (options.hmcr_max - options.hmcr_min) * t;
        let par = options.par_max - (options.par_max - options.par_min) * t;
        let bw = options.bw_max * ((options.bw_min / options.bw_max.max(1e-15)).powf(t));

        // Improvise
        let mut harmony = Array1::zeros(ndim);
        for d in 0..ndim {
            let (lo, hi) = base.bounds[d];
            if local_rng.random::<f64>() < hmcr {
                let mem_idx = local_rng.random_range(0..hms);
                harmony[d] = memory[mem_idx][d];
                if local_rng.random::<f64>() < par {
                    harmony[d] += (local_rng.random::<f64>() * 2.0 - 1.0) * bw;
                }
            } else {
                harmony[d] = lo + local_rng.random::<f64>() * (hi - lo);
            }
            harmony[d] = harmony[d].clamp(lo, hi);
        }

        let new_f = func(&harmony.view());
        nfev += 1;

        // Replace worst
        let worst_idx = memory_fitness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if new_f < memory_fitness[worst_idx] {
            memory[worst_idx] = harmony;
            memory_fitness[worst_idx] = new_f;

            if new_f < best_fun {
                best_fun = new_f;
                no_improve = 0;
            } else {
                no_improve += 1;
            }
        } else {
            no_improve += 1;
        }

        // Early stopping
        if no_improve >= base.patience {
            let bi = memory_fitness
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            return Ok(HarmonySearchResult {
                x: memory[bi].clone(),
                fun: memory_fitness[bi],
                nfev,
                improvisations: imp + 1,
                converged: true,
                message: format!("IHS: no improvement for {} improvisations", base.patience),
            });
        }
    }

    let bi = memory_fitness
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(HarmonySearchResult {
        x: memory[bi].clone(),
        fun: memory_fitness[bi],
        nfev,
        improvisations: max_imp,
        converged: false,
        message: format!("IHS completed {} improvisations", max_imp),
    })
}

// ---------------------------------------------------------------------------
// Global-best Harmony Search
// ---------------------------------------------------------------------------

/// Global-best Harmony Search optimizer
///
/// Uses the global best harmony to guide pitch adjustments.
pub struct GlobalBestHarmonySearch {
    options: HarmonySearchOptions,
    rng: StdRng,
}

impl GlobalBestHarmonySearch {
    /// Create a new GHS optimizer
    pub fn new(options: HarmonySearchOptions) -> OptimizeResult<Self> {
        HarmonySearchOptimizer::validate_options(&options)?;
        let seed = options.seed.unwrap_or_else(|| rng().random());
        Ok(Self {
            options,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Run the global-best HS
    pub fn optimize<F>(&mut self, func: F) -> OptimizeResult<HarmonySearchResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let ndim = self.options.bounds.len();
        let hms = self.options.hms;

        // Initialize memory
        let mut memory: Vec<Array1<f64>> = Vec::with_capacity(hms);
        let mut memory_fitness: Vec<f64> = Vec::with_capacity(hms);
        let mut nfev: usize = 0;

        for _ in 0..hms {
            let h = Array1::from_vec(
                (0..ndim)
                    .map(|d| {
                        let (lo, hi) = self.options.bounds[d];
                        lo + self.rng.random::<f64>() * (hi - lo)
                    })
                    .collect(),
            );
            let fv = func(&h.view());
            nfev += 1;
            memory_fitness.push(fv);
            memory.push(h);
        }

        let mut best_idx = self.best_index(&memory_fitness);
        let mut best_fun = memory_fitness[best_idx];
        let mut no_improve: usize = 0;

        for imp in 0..self.options.max_improvisations {
            // Improvise with global-best guidance
            let mut harmony = Array1::zeros(ndim);

            for d in 0..ndim {
                let (lo, hi) = self.options.bounds[d];
                if self.rng.random::<f64>() < self.options.hmcr {
                    // Memory consideration
                    let mem_idx = self.rng.random_range(0..hms);
                    harmony[d] = memory[mem_idx][d];

                    // Pitch adjustment using global best
                    if self.rng.random::<f64>() < self.options.par {
                        // Move towards global best with random factor
                        let r = self.rng.random::<f64>();
                        harmony[d] = harmony[d] + r * (memory[best_idx][d] - harmony[d]);
                        // Add small perturbation
                        harmony[d] +=
                            (self.rng.random::<f64>() * 2.0 - 1.0) * self.options.bandwidth * 0.1;
                    }
                } else {
                    harmony[d] = lo + self.rng.random::<f64>() * (hi - lo);
                }
                harmony[d] = harmony[d].clamp(lo, hi);
            }

            let new_f = func(&harmony.view());
            nfev += 1;

            let worst_idx = self.worst_index(&memory_fitness);
            if new_f < memory_fitness[worst_idx] {
                memory[worst_idx] = harmony;
                memory_fitness[worst_idx] = new_f;

                if new_f < best_fun {
                    best_fun = new_f;
                    best_idx = worst_idx;
                    no_improve = 0;
                } else {
                    no_improve += 1;
                }
            } else {
                no_improve += 1;
            }

            // Convergence
            let spread = self.fitness_spread(&memory_fitness);
            if spread < self.options.tol {
                let bi = self.best_index(&memory_fitness);
                return Ok(HarmonySearchResult {
                    x: memory[bi].clone(),
                    fun: memory_fitness[bi],
                    nfev,
                    improvisations: imp + 1,
                    converged: true,
                    message: format!("GHS converged at improvisation {}", imp + 1),
                });
            }

            if no_improve >= self.options.patience {
                let bi = self.best_index(&memory_fitness);
                return Ok(HarmonySearchResult {
                    x: memory[bi].clone(),
                    fun: memory_fitness[bi],
                    nfev,
                    improvisations: imp + 1,
                    converged: true,
                    message: format!(
                        "GHS: no improvement for {} improvisations",
                        self.options.patience
                    ),
                });
            }
        }

        let bi = self.best_index(&memory_fitness);
        Ok(HarmonySearchResult {
            x: memory[bi].clone(),
            fun: memory_fitness[bi],
            nfev,
            improvisations: self.options.max_improvisations,
            converged: false,
            message: format!(
                "GHS completed {} improvisations",
                self.options.max_improvisations
            ),
        })
    }

    fn best_index(&self, fitness: &[f64]) -> usize {
        fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn worst_index(&self, fitness: &[f64]) -> usize {
        fitness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn fitness_spread(&self, fitness: &[f64]) -> f64 {
        if fitness.is_empty() {
            return 0.0;
        }
        let mean = fitness.iter().sum::<f64>() / fitness.len() as f64;
        let var = fitness.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / fitness.len() as f64;
        var.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Multi-objective Harmony Search
// ---------------------------------------------------------------------------

/// Result for multi-objective HS
#[derive(Debug, Clone)]
pub struct MultiObjectiveHsResult {
    /// Pareto front solutions
    pub pareto_front: Vec<Array1<f64>>,
    /// Objective values for each Pareto solution (vec of vec: \[solution_idx\]\[obj_idx\])
    pub pareto_objectives: Vec<Vec<f64>>,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of improvisations
    pub improvisations: usize,
    /// Termination message
    pub message: String,
}

/// Multi-objective Harmony Search
pub struct MultiObjectiveHarmonySearch {
    options: HarmonySearchOptions,
    rng: StdRng,
}

impl MultiObjectiveHarmonySearch {
    /// Create a new MOHS optimizer
    pub fn new(options: HarmonySearchOptions) -> OptimizeResult<Self> {
        HarmonySearchOptimizer::validate_options(&options)?;
        let seed = options.seed.unwrap_or_else(|| rng().random());
        Ok(Self {
            options,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Run multi-objective HS
    ///
    /// `objectives` takes a vector and returns a vector of objective values (all to be minimized).
    pub fn optimize<F>(
        &mut self,
        objectives: F,
        num_objectives: usize,
    ) -> OptimizeResult<MultiObjectiveHsResult>
    where
        F: Fn(&ArrayView1<f64>) -> Vec<f64>,
    {
        if num_objectives == 0 {
            return Err(OptimizeError::InvalidParameter(
                "Number of objectives must be > 0".to_string(),
            ));
        }

        let ndim = self.options.bounds.len();
        let hms = self.options.hms;

        // Initialize memory
        let mut memory: Vec<Array1<f64>> = Vec::with_capacity(hms);
        let mut memory_objectives: Vec<Vec<f64>> = Vec::with_capacity(hms);
        let mut nfev: usize = 0;

        for _ in 0..hms {
            let h = Array1::from_vec(
                (0..ndim)
                    .map(|d| {
                        let (lo, hi) = self.options.bounds[d];
                        lo + self.rng.random::<f64>() * (hi - lo)
                    })
                    .collect(),
            );
            let objs = objectives(&h.view());
            nfev += 1;
            memory_objectives.push(objs);
            memory.push(h);
        }

        for imp in 0..self.options.max_improvisations {
            // Improvise
            let mut harmony = Array1::zeros(ndim);
            for d in 0..ndim {
                let (lo, hi) = self.options.bounds[d];
                if self.rng.random::<f64>() < self.options.hmcr {
                    let mem_idx = self.rng.random_range(0..hms);
                    harmony[d] = memory[mem_idx][d];
                    if self.rng.random::<f64>() < self.options.par {
                        harmony[d] +=
                            (self.rng.random::<f64>() * 2.0 - 1.0) * self.options.bandwidth;
                    }
                } else {
                    harmony[d] = lo + self.rng.random::<f64>() * (hi - lo);
                }
                harmony[d] = harmony[d].clamp(lo, hi);
            }

            let new_objs = objectives(&harmony.view());
            nfev += 1;

            // Find a dominated member to replace (if any)
            let mut replaced = false;
            for i in 0..hms {
                if self.dominates(&new_objs, &memory_objectives[i]) {
                    memory[i] = harmony.clone();
                    memory_objectives[i] = new_objs.clone();
                    replaced = true;
                    break;
                }
            }

            // If no member is dominated, replace a random non-dominating member
            // if the new harmony is non-dominated
            if !replaced {
                let is_dominated =
                    (0..hms).any(|i| self.dominates(&memory_objectives[i], &new_objs));

                if !is_dominated {
                    // Replace the member with the highest crowding distance (most redundant)
                    let replace_idx = self.rng.random_range(0..hms);
                    memory[replace_idx] = harmony;
                    memory_objectives[replace_idx] = new_objs;
                }
            }
        }

        // Extract Pareto front from final memory
        let (pareto_solutions, pareto_objs) =
            self.extract_pareto_front(&memory, &memory_objectives);

        Ok(MultiObjectiveHsResult {
            pareto_front: pareto_solutions,
            pareto_objectives: pareto_objs,
            nfev,
            improvisations: self.options.max_improvisations,
            message: format!(
                "MOHS completed {} improvisations",
                self.options.max_improvisations
            ),
        })
    }

    /// Check if solution a dominates solution b (all objectives minimized)
    fn dominates(&self, a: &[f64], b: &[f64]) -> bool {
        let mut at_least_one_better = false;
        for (ai, bi) in a.iter().zip(b.iter()) {
            if *ai > *bi {
                return false; // a is worse in at least one objective
            }
            if *ai < *bi {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }

    /// Extract non-dominated solutions
    fn extract_pareto_front(
        &self,
        memory: &[Array1<f64>],
        objectives: &[Vec<f64>],
    ) -> (Vec<Array1<f64>>, Vec<Vec<f64>>) {
        let n = memory.len();
        let mut is_dominated = vec![false; n];

        for i in 0..n {
            if is_dominated[i] {
                continue;
            }
            for j in 0..n {
                if i == j || is_dominated[j] {
                    continue;
                }
                if self.dominates(&objectives[j], &objectives[i]) {
                    is_dominated[i] = true;
                    break;
                }
            }
        }

        let mut pareto_solutions = Vec::new();
        let mut pareto_objs = Vec::new();
        for i in 0..n {
            if !is_dominated[i] {
                pareto_solutions.push(memory[i].clone());
                pareto_objs.push(objectives[i].clone());
            }
        }
        (pareto_solutions, pareto_objs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    // --- Basic HS tests ---

    #[test]
    fn test_hs_sphere() {
        let opts = HarmonySearchOptions {
            hms: 20,
            max_improvisations: 10_000,
            hmcr: 0.95,
            par: 0.3,
            bandwidth: 0.5,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            patience: 3000,
            ..Default::default()
        };

        let mut hs = HarmonySearchOptimizer::new(opts).expect("valid options");
        let result = hs.optimize(sphere).expect("HS should work");

        assert!(result.fun < 1.0, "HS sphere: got {}", result.fun);
        assert!(result.nfev > 0);
    }

    #[test]
    fn test_hs_rosenbrock() {
        let opts = HarmonySearchOptions {
            hms: 30,
            max_improvisations: 20_000,
            hmcr: 0.95,
            par: 0.3,
            bandwidth: 0.3,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            patience: 5000,
            ..Default::default()
        };

        let mut hs = HarmonySearchOptimizer::new(opts).expect("valid options");
        let result = hs.optimize(rosenbrock).expect("HS on rosenbrock");

        assert!(result.fun < 50.0, "HS rosenbrock: got {}", result.fun);
    }

    #[test]
    fn test_hs_1d() {
        let opts = HarmonySearchOptions {
            hms: 10,
            max_improvisations: 5000,
            hmcr: 0.9,
            par: 0.3,
            bandwidth: 0.2,
            bounds: vec![(-10.0, 10.0)],
            seed: Some(42),
            patience: 2000,
            ..Default::default()
        };

        let mut hs = HarmonySearchOptimizer::new(opts).expect("valid");
        let result = hs
            .optimize(|x: &ArrayView1<f64>| (x[0] - 3.0).powi(2))
            .expect("1D HS");

        assert!(
            (result.x[0] - 3.0).abs() < 2.0,
            "1D HS: x = {}",
            result.x[0]
        );
    }

    // --- IHS tests ---

    #[test]
    fn test_ihs_sphere() {
        let opts = ImprovedHarmonySearchOptions {
            base: HarmonySearchOptions {
                hms: 20,
                max_improvisations: 10_000,
                bounds: vec![(-5.0, 5.0); 2],
                seed: Some(42),
                patience: 3000,
                ..Default::default()
            },
            hmcr_min: 0.7,
            hmcr_max: 0.99,
            par_min: 0.01,
            par_max: 0.99,
            bw_min: 1e-4,
            bw_max: 1.0,
        };

        let result = improved_harmony_search(sphere, opts).expect("IHS should work");

        assert!(result.fun < 1.0, "IHS sphere: got {}", result.fun);
    }

    #[test]
    fn test_ihs_rastrigin() {
        let opts = ImprovedHarmonySearchOptions {
            base: HarmonySearchOptions {
                hms: 30,
                max_improvisations: 20_000,
                bounds: vec![(-5.12, 5.12); 3],
                seed: Some(42),
                patience: 5000,
                ..Default::default()
            },
            bw_max: 2.0,
            ..Default::default()
        };

        let result = improved_harmony_search(rastrigin, opts).expect("IHS on rastrigin");

        assert!(result.fun < 30.0, "IHS rastrigin: got {}", result.fun);
    }

    // --- GHS tests ---

    #[test]
    fn test_ghs_sphere() {
        let opts = HarmonySearchOptions {
            hms: 20,
            max_improvisations: 10_000,
            hmcr: 0.95,
            par: 0.5,
            bandwidth: 0.5,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            patience: 3000,
            ..Default::default()
        };

        let mut ghs = GlobalBestHarmonySearch::new(opts).expect("valid options");
        let result = ghs.optimize(sphere).expect("GHS should work");

        assert!(result.fun < 1.0, "GHS sphere: got {}", result.fun);
    }

    #[test]
    fn test_ghs_rosenbrock() {
        let opts = HarmonySearchOptions {
            hms: 30,
            max_improvisations: 20_000,
            hmcr: 0.95,
            par: 0.5,
            bandwidth: 0.3,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            patience: 5000,
            ..Default::default()
        };

        let mut ghs = GlobalBestHarmonySearch::new(opts).expect("valid options");
        let result = ghs.optimize(rosenbrock).expect("GHS on rosenbrock");

        assert!(result.fun < 50.0, "GHS rosenbrock: got {}", result.fun);
    }

    // --- MOHS tests ---

    #[test]
    fn test_mohs_two_objectives() {
        let objectives = |x: &ArrayView1<f64>| -> Vec<f64> {
            let f1 = x[0] * x[0];
            let f2 = (x[0] - 2.0) * (x[0] - 2.0);
            vec![f1, f2]
        };

        let opts = HarmonySearchOptions {
            hms: 30,
            max_improvisations: 5000,
            hmcr: 0.9,
            par: 0.3,
            bandwidth: 0.2,
            bounds: vec![(-5.0, 5.0)],
            seed: Some(42),
            ..Default::default()
        };

        let mut mohs = MultiObjectiveHarmonySearch::new(opts).expect("valid");
        let result = mohs.optimize(objectives, 2).expect("MOHS should work");

        assert!(
            !result.pareto_front.is_empty(),
            "Should have Pareto solutions"
        );
        assert_eq!(result.pareto_objectives[0].len(), 2);
        assert!(result.nfev > 0);

        // Check that Pareto solutions are non-dominated
        for i in 0..result.pareto_objectives.len() {
            for j in 0..result.pareto_objectives.len() {
                if i != j {
                    assert!(
                        !dominates_helper(
                            &result.pareto_objectives[j],
                            &result.pareto_objectives[i]
                        ),
                        "Pareto front should have no dominated solutions"
                    );
                }
            }
        }
    }

    fn dominates_helper(a: &[f64], b: &[f64]) -> bool {
        let mut better = false;
        for (ai, bi) in a.iter().zip(b.iter()) {
            if *ai > *bi {
                return false;
            }
            if *ai < *bi {
                better = true;
            }
        }
        better
    }

    #[test]
    fn test_mohs_three_objectives() {
        let objectives = |x: &ArrayView1<f64>| -> Vec<f64> {
            vec![
                x[0] * x[0],
                (x[0] - 1.0).powi(2) + x[1] * x[1],
                (x[0] + x[1] - 2.0).powi(2),
            ]
        };

        let opts = HarmonySearchOptions {
            hms: 30,
            max_improvisations: 3000,
            hmcr: 0.9,
            par: 0.3,
            bandwidth: 0.3,
            bounds: vec![(-5.0, 5.0); 2],
            seed: Some(42),
            ..Default::default()
        };

        let mut mohs = MultiObjectiveHarmonySearch::new(opts).expect("valid");
        let result = mohs.optimize(objectives, 3).expect("3-objective MOHS");

        assert!(!result.pareto_front.is_empty());
    }

    // --- Edge case tests ---

    #[test]
    fn test_hs_empty_bounds_error() {
        let opts = HarmonySearchOptions {
            bounds: vec![],
            ..Default::default()
        };
        let result = HarmonySearchOptimizer::new(opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_hs_small_memory_error() {
        let opts = HarmonySearchOptions {
            hms: 1,
            bounds: vec![(-1.0, 1.0)],
            ..Default::default()
        };
        let result = HarmonySearchOptimizer::new(opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_hs_invalid_hmcr_error() {
        let opts = HarmonySearchOptions {
            hmcr: 1.5,
            bounds: vec![(-1.0, 1.0)],
            ..Default::default()
        };
        let result = HarmonySearchOptimizer::new(opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_hs_invalid_par_error() {
        let opts = HarmonySearchOptions {
            par: -0.1,
            bounds: vec![(-1.0, 1.0)],
            ..Default::default()
        };
        let result = HarmonySearchOptimizer::new(opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_hs_to_optimize_results() {
        let result = HarmonySearchResult {
            x: scirs2_core::ndarray::array![1.0, 2.0],
            fun: 5.0,
            nfev: 1000,
            improvisations: 500,
            converged: true,
            message: "test".to_string(),
        };
        let opt = result.to_optimize_results();
        assert_eq!(opt.nfev, 1000);
        assert!(opt.success);
    }

    #[test]
    fn test_mohs_zero_objectives_error() {
        let opts = HarmonySearchOptions {
            bounds: vec![(-1.0, 1.0)],
            ..Default::default()
        };
        let mut mohs = MultiObjectiveHarmonySearch::new(opts).expect("valid");
        let result = mohs.optimize(|_x: &ArrayView1<f64>| vec![], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ihs_empty_bounds_error() {
        let opts = ImprovedHarmonySearchOptions {
            base: HarmonySearchOptions {
                bounds: vec![],
                ..Default::default()
            },
            ..Default::default()
        };
        let result = improved_harmony_search(sphere, opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_ghs_empty_bounds_error() {
        let opts = HarmonySearchOptions {
            bounds: vec![],
            ..Default::default()
        };
        let result = GlobalBestHarmonySearch::new(opts);
        assert!(result.is_err());
    }
}
