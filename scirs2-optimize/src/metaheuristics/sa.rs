//! # Simulated Annealing (SA) Metaheuristic
//!
//! A comprehensive implementation of Simulated Annealing with:
//! - Configurable cooling schedules (linear, exponential, logarithmic, adaptive)
//! - Metropolis acceptance criterion
//! - Reheating strategy for escaping local optima
//! - Multi-start SA for improved global search
//! - Constraint handling via penalty functions

use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{rng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ---------------------------------------------------------------------------
// Cooling Schedules
// ---------------------------------------------------------------------------

/// Cooling schedule for temperature reduction
#[derive(Debug, Clone)]
pub enum CoolingSchedule {
    /// Linear: T(k) = T0 - k * (T0 - Tf) / max_iter
    Linear,
    /// Exponential: T(k) = T0 * alpha^k
    Exponential {
        /// Cooling rate (0 < alpha < 1)
        alpha: f64,
    },
    /// Logarithmic (Cauchy): T(k) = T0 / (1 + c * ln(1 + k))
    Logarithmic {
        /// Scaling constant c > 0
        c: f64,
    },
    /// Adaptive: adjusts cooling rate based on acceptance ratio
    Adaptive {
        /// Target acceptance ratio (e.g. 0.44 for Metropolis)
        target_acceptance: f64,
        /// Adjustment factor (> 1 speeds up, < 1 slows down cooling)
        adjustment_factor: f64,
    },
}

impl Default for CoolingSchedule {
    fn default() -> Self {
        CoolingSchedule::Exponential { alpha: 0.95 }
    }
}

/// State for adaptive cooling schedule
#[derive(Debug, Clone)]
pub struct AdaptiveCoolingState {
    /// Number of accepted moves in current window
    pub accepted: usize,
    /// Number of total moves in current window
    pub total: usize,
    /// Current effective alpha
    pub effective_alpha: f64,
    /// Window size for acceptance ratio computation
    pub window_size: usize,
}

impl Default for AdaptiveCoolingState {
    fn default() -> Self {
        Self {
            accepted: 0,
            total: 0,
            effective_alpha: 0.95,
            window_size: 100,
        }
    }
}

impl AdaptiveCoolingState {
    /// Record an acceptance/rejection and return updated alpha
    fn update(&mut self, accepted: bool, target: f64, adjustment: f64) -> f64 {
        if accepted {
            self.accepted += 1;
        }
        self.total += 1;

        if self.total >= self.window_size {
            let ratio = self.accepted as f64 / self.total as f64;
            // If acceptance too high => cool faster; if too low => cool slower
            if ratio > target {
                self.effective_alpha *= adjustment; // speed up cooling
            } else {
                self.effective_alpha /= adjustment; // slow down cooling
            }
            // Clamp alpha to [0.5, 0.999]
            self.effective_alpha = self.effective_alpha.clamp(0.5, 0.999);
            self.accepted = 0;
            self.total = 0;
        }
        self.effective_alpha
    }
}

// ---------------------------------------------------------------------------
// Reheating Strategy
// ---------------------------------------------------------------------------

/// Reheating strategy for escaping local optima
#[derive(Debug, Clone)]
pub enum ReheatingStrategy {
    /// No reheating
    None,
    /// Reheat periodically every N iterations to a fraction of initial temperature
    Periodic {
        /// Interval (in iterations) between reheats
        interval: usize,
        /// Fraction of initial temperature to reheat to (0, 1]
        fraction: f64,
    },
    /// Reheat when stagnation detected (no improvement for N iterations)
    Stagnation {
        /// Number of iterations without improvement to trigger reheat
        patience: usize,
        /// Fraction of initial temperature to reheat to
        fraction: f64,
    },
}

impl Default for ReheatingStrategy {
    fn default() -> Self {
        ReheatingStrategy::None
    }
}

// ---------------------------------------------------------------------------
// Constraint Handling
// ---------------------------------------------------------------------------

/// Penalty-based constraint for SA
#[derive(Debug, Clone)]
pub struct PenaltyConstraint {
    /// Penalty coefficient (multiplier for constraint violation)
    pub penalty_coeff: f64,
    /// Whether to increase penalty over time
    pub adaptive: bool,
    /// Maximum penalty coefficient (for adaptive mode)
    pub max_penalty: f64,
}

impl Default for PenaltyConstraint {
    fn default() -> Self {
        Self {
            penalty_coeff: 1000.0,
            adaptive: false,
            max_penalty: 1e8,
        }
    }
}

/// Constraint handler for SA
#[derive(Debug, Clone)]
pub struct ConstraintHandler {
    /// Inequality constraints: g_i(x) <= 0
    /// Each function returns a vector of constraint violations
    /// Positive values indicate violation
    pub penalty: PenaltyConstraint,
}

impl Default for ConstraintHandler {
    fn default() -> Self {
        Self {
            penalty: PenaltyConstraint::default(),
        }
    }
}

impl ConstraintHandler {
    /// Compute penalty given constraint violation values
    pub fn compute_penalty(&self, violations: &[f64], iteration: usize, max_iter: usize) -> f64 {
        let coeff = if self.penalty.adaptive {
            let progress = iteration as f64 / max_iter.max(1) as f64;
            let scaled = self.penalty.penalty_coeff * (1.0 + progress * 10.0);
            scaled.min(self.penalty.max_penalty)
        } else {
            self.penalty.penalty_coeff
        };

        violations
            .iter()
            .map(|v| {
                if *v > 0.0 {
                    coeff * v * v // quadratic penalty
                } else {
                    0.0
                }
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// SA Options and Result
// ---------------------------------------------------------------------------

/// Options for metaheuristic SA optimizer
#[derive(Debug, Clone)]
pub struct MetaheuristicSaOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Initial temperature
    pub initial_temp: f64,
    /// Final (minimum) temperature
    pub final_temp: f64,
    /// Cooling schedule
    pub cooling: CoolingSchedule,
    /// Number of candidate solutions tried at each temperature level
    pub steps_per_temp: usize,
    /// Perturbation step size (scaled by dimension)
    pub step_size: f64,
    /// Reheating strategy
    pub reheating: ReheatingStrategy,
    /// Random seed (None for non-deterministic)
    pub seed: Option<u64>,
    /// Search bounds per dimension: (lower, upper)
    pub bounds: Option<Vec<(f64, f64)>>,
    /// Tolerance for convergence (stop if best value changes less than this)
    pub tol: f64,
}

impl Default for MetaheuristicSaOptions {
    fn default() -> Self {
        Self {
            max_iter: 10_000,
            initial_temp: 100.0,
            final_temp: 1e-10,
            cooling: CoolingSchedule::default(),
            steps_per_temp: 50,
            step_size: 1.0,
            reheating: ReheatingStrategy::None,
            seed: None,
            bounds: None,
            tol: 1e-12,
        }
    }
}

/// Result from SA optimization
#[derive(Debug, Clone)]
pub struct MetaheuristicSaResult {
    /// Best solution found
    pub x: Array1<f64>,
    /// Objective value at best solution
    pub fun: f64,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of iterations performed
    pub nit: usize,
    /// Whether the optimization was successful
    pub success: bool,
    /// Termination message
    pub message: String,
    /// Final temperature
    pub final_temperature: f64,
    /// Acceptance ratio over the run
    pub acceptance_ratio: f64,
}

impl MetaheuristicSaResult {
    /// Convert to standard OptimizeResults
    pub fn to_optimize_results(&self) -> OptimizeResults<f64> {
        OptimizeResults {
            x: self.x.clone(),
            fun: self.fun,
            jac: None,
            hess: None,
            constr: None,
            nit: self.nit,
            nfev: self.nfev,
            njev: 0,
            nhev: 0,
            maxcv: 0,
            message: self.message.clone(),
            success: self.success,
            status: if self.success { 0 } else { 1 },
        }
    }
}

// ---------------------------------------------------------------------------
// Core SA Optimizer
// ---------------------------------------------------------------------------

/// Simulated Annealing optimizer
pub struct SimulatedAnnealingOptimizer {
    options: MetaheuristicSaOptions,
    rng: StdRng,
}

impl SimulatedAnnealingOptimizer {
    /// Create a new SA optimizer with the given options
    pub fn new(options: MetaheuristicSaOptions) -> Self {
        let seed = options.seed.unwrap_or_else(|| rng().random());
        Self {
            options,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Compute the next temperature given the schedule
    fn next_temperature(
        &self,
        current_temp: f64,
        iteration: usize,
        adaptive_state: &mut AdaptiveCoolingState,
        accepted: bool,
    ) -> f64 {
        match &self.options.cooling {
            CoolingSchedule::Linear => {
                let t0 = self.options.initial_temp;
                let tf = self.options.final_temp;
                let max_iter = self.options.max_iter;
                let step = (t0 - tf) / max_iter.max(1) as f64;
                (current_temp - step).max(tf)
            }
            CoolingSchedule::Exponential { alpha } => {
                (current_temp * alpha).max(self.options.final_temp)
            }
            CoolingSchedule::Logarithmic { c } => {
                let t0 = self.options.initial_temp;
                let denom = 1.0 + c * ((1.0 + iteration as f64).ln());
                (t0 / denom).max(self.options.final_temp)
            }
            CoolingSchedule::Adaptive {
                target_acceptance,
                adjustment_factor,
            } => {
                let alpha = adaptive_state.update(accepted, *target_acceptance, *adjustment_factor);
                (current_temp * alpha).max(self.options.final_temp)
            }
        }
    }

    /// Generate a neighbor solution by perturbing x
    fn generate_neighbor(&mut self, x: &Array1<f64>) -> Array1<f64> {
        let ndim = x.len();
        let mut neighbor = x.clone();
        // Perturb a random dimension
        let dim_idx = self.rng.random_range(0..ndim);
        let perturbation = (self.rng.random::<f64>() * 2.0 - 1.0) * self.options.step_size;
        neighbor[dim_idx] += perturbation;

        // Enforce bounds if present
        if let Some(ref bounds) = self.options.bounds {
            for (i, (lo, hi)) in bounds.iter().enumerate() {
                neighbor[i] = neighbor[i].clamp(*lo, *hi);
            }
        }
        neighbor
    }

    /// Metropolis acceptance criterion
    fn accept(&mut self, delta_e: f64, temperature: f64) -> bool {
        if delta_e < 0.0 {
            true // always accept improvements
        } else if temperature <= 0.0 {
            false
        } else {
            let prob = (-delta_e / temperature).exp();
            self.rng.random::<f64>() < prob
        }
    }

    /// Apply reheating if conditions are met
    fn maybe_reheat(
        &self,
        current_temp: f64,
        iteration: usize,
        stagnation_count: usize,
    ) -> Option<f64> {
        match &self.options.reheating {
            ReheatingStrategy::None => None,
            ReheatingStrategy::Periodic { interval, fraction } => {
                if *interval > 0 && iteration > 0 && iteration % interval == 0 {
                    Some(self.options.initial_temp * fraction)
                } else {
                    None
                }
            }
            ReheatingStrategy::Stagnation { patience, fraction } => {
                if stagnation_count >= *patience {
                    // Only reheat if we've really stagnated and temperature is low
                    if current_temp < self.options.initial_temp * 0.1 {
                        Some(self.options.initial_temp * fraction)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    /// Run SA optimization on an unconstrained problem
    pub fn optimize<F>(
        &mut self,
        func: F,
        x0: &Array1<f64>,
    ) -> OptimizeResult<MetaheuristicSaResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        self.optimize_constrained(func, x0, None::<fn(&ArrayView1<f64>) -> Vec<f64>>)
    }

    /// Run SA optimization with optional constraint functions
    ///
    /// `constraints_fn` returns a vector of g_i(x) values where g_i(x) > 0 means violation.
    pub fn optimize_constrained<F, G>(
        &mut self,
        func: F,
        x0: &Array1<f64>,
        constraints_fn: Option<G>,
    ) -> OptimizeResult<MetaheuristicSaResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Vec<f64>,
    {
        let constraint_handler = ConstraintHandler::default();
        self.optimize_with_handler(func, x0, constraints_fn, &constraint_handler)
    }

    /// Run SA optimization with full constraint handler configuration
    pub fn optimize_with_handler<F, G>(
        &mut self,
        func: F,
        x0: &Array1<f64>,
        constraints_fn: Option<G>,
        constraint_handler: &ConstraintHandler,
    ) -> OptimizeResult<MetaheuristicSaResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Vec<f64>,
    {
        if x0.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "Initial point must not be empty".to_string(),
            ));
        }
        if self.options.initial_temp <= 0.0 {
            return Err(OptimizeError::InvalidParameter(
                "Initial temperature must be positive".to_string(),
            ));
        }

        let max_iter = self.options.max_iter;

        let evaluate = |x: &Array1<f64>, iter: usize| -> f64 {
            let obj = func(&x.view());
            if let Some(ref cf) = constraints_fn {
                let violations = cf(&x.view());
                let penalty = constraint_handler.compute_penalty(&violations, iter, max_iter);
                obj + penalty
            } else {
                obj
            }
        };

        let mut current_x = x0.clone();
        // Enforce initial bounds
        if let Some(ref bounds) = self.options.bounds {
            for (i, (lo, hi)) in bounds.iter().enumerate() {
                if i < current_x.len() {
                    current_x[i] = current_x[i].clamp(*lo, *hi);
                }
            }
        }

        let mut current_val = evaluate(&current_x, 0);
        let mut best_x = current_x.clone();
        let mut best_val = current_val;
        let mut temperature = self.options.initial_temp;
        let mut nfev: usize = 1;
        let mut nit: usize = 0;
        let mut total_accepted: usize = 0;
        let mut total_tried: usize = 0;
        let mut stagnation_count: usize = 0;
        let mut adaptive_state = AdaptiveCoolingState::default();

        for iteration in 0..self.options.max_iter {
            nit = iteration + 1;
            let mut step_accepted = false;

            for _step in 0..self.options.steps_per_temp {
                let neighbor = self.generate_neighbor(&current_x);
                let neighbor_val = evaluate(&neighbor, iteration);
                nfev += 1;

                let delta_e = neighbor_val - current_val;
                total_tried += 1;

                if self.accept(delta_e, temperature) {
                    current_x = neighbor;
                    current_val = neighbor_val;
                    total_accepted += 1;
                    step_accepted = true;

                    if current_val < best_val {
                        best_x = current_x.clone();
                        best_val = current_val;
                        stagnation_count = 0;
                    }
                }
            }

            if !step_accepted {
                stagnation_count += 1;
            }

            // Update temperature
            temperature =
                self.next_temperature(temperature, iteration, &mut adaptive_state, step_accepted);

            // Check for reheating
            if let Some(new_temp) = self.maybe_reheat(temperature, iteration, stagnation_count) {
                temperature = new_temp;
                stagnation_count = 0;
            }

            // Check convergence
            if temperature <= self.options.final_temp {
                break;
            }
        }

        let acceptance_ratio = if total_tried > 0 {
            total_accepted as f64 / total_tried as f64
        } else {
            0.0
        };

        Ok(MetaheuristicSaResult {
            x: best_x,
            fun: best_val,
            nfev,
            nit,
            success: true,
            message: format!(
                "SA completed: {} iterations, {} evaluations, acceptance ratio {:.4}",
                nit, nfev, acceptance_ratio
            ),
            final_temperature: temperature,
            acceptance_ratio,
        })
    }
}

// ---------------------------------------------------------------------------
// Multi-Start SA
// ---------------------------------------------------------------------------

/// Options for multi-start SA
#[derive(Debug, Clone)]
pub struct MultiStartSaOptions {
    /// Number of independent SA runs
    pub n_starts: usize,
    /// Per-run SA options
    pub sa_options: MetaheuristicSaOptions,
    /// Random seed for generating start points (None for non-deterministic)
    pub seed: Option<u64>,
}

impl Default for MultiStartSaOptions {
    fn default() -> Self {
        Self {
            n_starts: 5,
            sa_options: MetaheuristicSaOptions::default(),
            seed: None,
        }
    }
}

/// Run multi-start SA: perform multiple independent SA runs from different starting points
/// and return the best result.
///
/// Requires bounds to generate random starting points.
pub fn multi_start_sa<F>(
    func: F,
    bounds: &[(f64, f64)],
    options: MultiStartSaOptions,
) -> OptimizeResult<MetaheuristicSaResult>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    if bounds.is_empty() {
        return Err(OptimizeError::InvalidInput(
            "Bounds must not be empty for multi-start SA".to_string(),
        ));
    }
    if options.n_starts == 0 {
        return Err(OptimizeError::InvalidParameter(
            "n_starts must be at least 1".to_string(),
        ));
    }

    let ndim = bounds.len();
    let outer_seed = options.seed.unwrap_or_else(|| rng().random());
    let mut outer_rng = StdRng::seed_from_u64(outer_seed);

    let mut best_result: Option<MetaheuristicSaResult> = None;
    let mut total_nfev: usize = 0;

    for start_idx in 0..options.n_starts {
        // Generate random starting point within bounds
        let x0 = Array1::from_vec(
            bounds
                .iter()
                .map(|(lo, hi)| lo + outer_rng.random::<f64>() * (hi - lo))
                .collect::<Vec<_>>(),
        );

        let mut sa_opts = options.sa_options.clone();
        sa_opts.bounds = Some(bounds.to_vec());
        // Give each run a unique seed derived from the outer seed
        sa_opts.seed = Some(outer_rng.random::<u64>());

        let mut optimizer = SimulatedAnnealingOptimizer::new(sa_opts);
        match optimizer.optimize(&func, &x0) {
            Ok(result) => {
                total_nfev += result.nfev;
                let is_better = best_result
                    .as_ref()
                    .map_or(true, |best| result.fun < best.fun);
                if is_better {
                    best_result = Some(result);
                }
            }
            Err(_) => {
                // Skip failed runs
                continue;
            }
        }
    }

    match best_result {
        Some(mut result) => {
            result.nfev = total_nfev;
            result.message = format!(
                "Multi-start SA: best of {} starts, {} total evaluations",
                options.n_starts, total_nfev
            );
            Ok(result)
        }
        None => Err(OptimizeError::ComputationError(
            "All multi-start SA runs failed".to_string(),
        )),
    }
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

    // --- Basic SA tests ---

    #[test]
    fn test_sa_sphere_exponential() {
        let opts = MetaheuristicSaOptions {
            max_iter: 5000,
            initial_temp: 100.0,
            final_temp: 1e-10,
            cooling: CoolingSchedule::Exponential { alpha: 0.99 },
            steps_per_temp: 20,
            step_size: 0.5,
            seed: Some(42),
            bounds: Some(vec![(-5.0, 5.0); 2]),
            ..Default::default()
        };

        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = array![3.0, -2.0];
        let result = optimizer.optimize(sphere, &x0).expect("SA should succeed");

        assert!(
            result.fun < 1.0,
            "Sphere minimum should be near 0, got {}",
            result.fun
        );
        assert!(result.success);
        assert!(result.nfev > 0);
    }

    #[test]
    fn test_sa_sphere_linear_cooling() {
        let opts = MetaheuristicSaOptions {
            max_iter: 5000,
            initial_temp: 50.0,
            final_temp: 1e-8,
            cooling: CoolingSchedule::Linear,
            steps_per_temp: 20,
            step_size: 0.3,
            seed: Some(123),
            bounds: Some(vec![(-5.0, 5.0); 2]),
            ..Default::default()
        };

        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = array![2.0, 2.0];
        let result = optimizer.optimize(sphere, &x0).expect("SA should succeed");

        assert!(
            result.fun < 2.0,
            "Sphere should converge near 0, got {}",
            result.fun
        );
    }

    #[test]
    fn test_sa_sphere_logarithmic_cooling() {
        let opts = MetaheuristicSaOptions {
            max_iter: 5000,
            initial_temp: 100.0,
            final_temp: 1e-10,
            cooling: CoolingSchedule::Logarithmic { c: 1.0 },
            steps_per_temp: 20,
            step_size: 0.5,
            seed: Some(7),
            bounds: Some(vec![(-5.0, 5.0); 2]),
            ..Default::default()
        };

        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = array![4.0, -3.0];
        let result = optimizer.optimize(sphere, &x0).expect("SA should succeed");

        // Logarithmic is slow cooling so still should get a decent result
        assert!(result.fun < 5.0, "Log cooling on sphere got {}", result.fun);
    }

    #[test]
    fn test_sa_adaptive_cooling() {
        let opts = MetaheuristicSaOptions {
            max_iter: 5000,
            initial_temp: 100.0,
            final_temp: 1e-10,
            cooling: CoolingSchedule::Adaptive {
                target_acceptance: 0.44,
                adjustment_factor: 1.02,
            },
            steps_per_temp: 20,
            step_size: 0.5,
            seed: Some(99),
            bounds: Some(vec![(-5.0, 5.0); 2]),
            ..Default::default()
        };

        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = array![3.0, 3.0];
        let result = optimizer.optimize(sphere, &x0).expect("SA should succeed");

        assert!(result.fun < 5.0, "Adaptive SA on sphere got {}", result.fun);
    }

    // --- Reheating tests ---

    #[test]
    fn test_sa_periodic_reheating() {
        let opts = MetaheuristicSaOptions {
            max_iter: 3000,
            initial_temp: 50.0,
            final_temp: 1e-8,
            cooling: CoolingSchedule::Exponential { alpha: 0.99 },
            steps_per_temp: 10,
            step_size: 0.3,
            reheating: ReheatingStrategy::Periodic {
                interval: 500,
                fraction: 0.3,
            },
            seed: Some(55),
            bounds: Some(vec![(-5.0, 5.0); 2]),
            ..Default::default()
        };

        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = array![4.0, -4.0];
        let result = optimizer
            .optimize(sphere, &x0)
            .expect("SA with reheating should succeed");

        assert!(result.success);
        assert!(result.fun < 5.0, "Periodic reheating got {}", result.fun);
    }

    #[test]
    fn test_sa_stagnation_reheating() {
        let opts = MetaheuristicSaOptions {
            max_iter: 3000,
            initial_temp: 50.0,
            final_temp: 1e-8,
            cooling: CoolingSchedule::Exponential { alpha: 0.99 },
            steps_per_temp: 10,
            step_size: 0.3,
            reheating: ReheatingStrategy::Stagnation {
                patience: 200,
                fraction: 0.5,
            },
            seed: Some(77),
            bounds: Some(vec![(-5.0, 5.0); 2]),
            ..Default::default()
        };

        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = array![3.0, 3.0];
        let result = optimizer
            .optimize(sphere, &x0)
            .expect("SA with stagnation reheating should succeed");
        assert!(result.success);
    }

    // --- Constraint handling tests ---

    #[test]
    fn test_sa_with_constraints() {
        // Minimize x^2 + y^2 subject to x + y >= 2  (i.e. -(x+y-2) <= 0)
        let constraints = |x: &ArrayView1<f64>| -> Vec<f64> {
            let sum = x[0] + x[1];
            vec![2.0 - sum] // violation when sum < 2
        };

        let opts = MetaheuristicSaOptions {
            max_iter: 5000,
            initial_temp: 50.0,
            final_temp: 1e-10,
            cooling: CoolingSchedule::Exponential { alpha: 0.995 },
            steps_per_temp: 20,
            step_size: 0.2,
            seed: Some(42),
            bounds: Some(vec![(-5.0, 5.0); 2]),
            ..Default::default()
        };

        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = array![1.5, 1.5];
        let result = optimizer
            .optimize_constrained(sphere, &x0, Some(constraints))
            .expect("Constrained SA should succeed");

        // Optimal: x = y = 1, f = 2
        let sum = result.x[0] + result.x[1];
        assert!(
            sum >= 1.5,
            "Constraint should be approximately satisfied: sum = {}",
            sum
        );
    }

    #[test]
    fn test_sa_adaptive_penalty() {
        let constraints = |x: &ArrayView1<f64>| -> Vec<f64> {
            vec![1.0 - x[0], 1.0 - x[1]] // x >= 1, y >= 1
        };

        let handler = ConstraintHandler {
            penalty: PenaltyConstraint {
                penalty_coeff: 100.0,
                adaptive: true,
                max_penalty: 1e6,
            },
        };

        let opts = MetaheuristicSaOptions {
            max_iter: 3000,
            initial_temp: 50.0,
            final_temp: 1e-8,
            cooling: CoolingSchedule::Exponential { alpha: 0.995 },
            steps_per_temp: 20,
            step_size: 0.3,
            seed: Some(42),
            bounds: Some(vec![(-5.0, 5.0); 2]),
            ..Default::default()
        };

        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = array![2.0, 2.0];
        let result = optimizer
            .optimize_with_handler(sphere, &x0, Some(constraints), &handler)
            .expect("Adaptive penalty SA should succeed");

        // Both x and y should be >= ~1
        assert!(
            result.x[0] > 0.5 && result.x[1] > 0.5,
            "Constrained result: x={:.4}, y={:.4}",
            result.x[0],
            result.x[1]
        );
    }

    // --- Multi-start SA tests ---

    #[test]
    fn test_multi_start_sa() {
        let bounds = vec![(-5.0, 5.0); 2];
        let ms_opts = MultiStartSaOptions {
            n_starts: 5,
            sa_options: MetaheuristicSaOptions {
                max_iter: 2000,
                initial_temp: 50.0,
                final_temp: 1e-8,
                cooling: CoolingSchedule::Exponential { alpha: 0.99 },
                steps_per_temp: 10,
                step_size: 0.3,
                seed: None,
                ..Default::default()
            },
            seed: Some(42),
        };

        let result =
            multi_start_sa(sphere, &bounds, ms_opts).expect("Multi-start SA should succeed");

        assert!(result.fun < 2.0, "Multi-start on sphere got {}", result.fun);
        assert!(result.success);
    }

    #[test]
    fn test_multi_start_sa_rastrigin() {
        let bounds = vec![(-5.12, 5.12); 3];
        let ms_opts = MultiStartSaOptions {
            n_starts: 10,
            sa_options: MetaheuristicSaOptions {
                max_iter: 3000,
                initial_temp: 100.0,
                final_temp: 1e-8,
                cooling: CoolingSchedule::Exponential { alpha: 0.995 },
                steps_per_temp: 20,
                step_size: 0.5,
                seed: None,
                ..Default::default()
            },
            seed: Some(123),
        };

        let result =
            multi_start_sa(rastrigin, &bounds, ms_opts).expect("Multi-start SA on rastrigin");

        // Rastrigin global min is 0 at origin; multi-start should get close
        assert!(
            result.fun < 20.0,
            "Multi-start rastrigin got {}",
            result.fun
        );
    }

    // --- Edge case tests ---

    #[test]
    fn test_sa_empty_input_error() {
        let opts = MetaheuristicSaOptions {
            seed: Some(1),
            ..Default::default()
        };
        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = Array1::<f64>::zeros(0);
        let result = optimizer.optimize(sphere, &x0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sa_invalid_temp_error() {
        let opts = MetaheuristicSaOptions {
            initial_temp: -1.0,
            seed: Some(1),
            ..Default::default()
        };
        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = array![1.0, 1.0];
        let result = optimizer.optimize(sphere, &x0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sa_single_dimension() {
        let opts = MetaheuristicSaOptions {
            max_iter: 2000,
            initial_temp: 50.0,
            final_temp: 1e-8,
            cooling: CoolingSchedule::Exponential { alpha: 0.99 },
            steps_per_temp: 10,
            step_size: 0.5,
            seed: Some(42),
            bounds: Some(vec![(-10.0, 10.0)]),
            ..Default::default()
        };

        let mut optimizer = SimulatedAnnealingOptimizer::new(opts);
        let x0 = array![5.0];
        let result = optimizer
            .optimize(|x: &ArrayView1<f64>| x[0] * x[0], &x0)
            .expect("1D SA");
        assert!(result.fun < 1.0, "1D SA got {}", result.fun);
    }

    #[test]
    fn test_sa_to_optimize_results() {
        let sa_result = MetaheuristicSaResult {
            x: array![1.0, 2.0],
            fun: 5.0,
            nfev: 1000,
            nit: 500,
            success: true,
            message: "test".to_string(),
            final_temperature: 0.001,
            acceptance_ratio: 0.4,
        };
        let opt_results = sa_result.to_optimize_results();
        assert_eq!(opt_results.nfev, 1000);
        assert_eq!(opt_results.nit, 500);
        assert!(opt_results.success);
    }

    #[test]
    fn test_multi_start_sa_empty_bounds_error() {
        let ms_opts = MultiStartSaOptions {
            n_starts: 3,
            seed: Some(1),
            ..Default::default()
        };
        let result = multi_start_sa(sphere, &[], ms_opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_start_sa_zero_starts_error() {
        let ms_opts = MultiStartSaOptions {
            n_starts: 0,
            seed: Some(1),
            ..Default::default()
        };
        let result = multi_start_sa(sphere, &[(-1.0, 1.0)], ms_opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_constraint_handler_no_violation() {
        let handler = ConstraintHandler::default();
        let violations = vec![-1.0, -0.5]; // no violations
        let penalty = handler.compute_penalty(&violations, 0, 100);
        assert!((penalty - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_constraint_handler_with_violation() {
        let handler = ConstraintHandler {
            penalty: PenaltyConstraint {
                penalty_coeff: 100.0,
                adaptive: false,
                max_penalty: 1e8,
            },
        };
        let violations = vec![2.0]; // violation of 2
        let penalty = handler.compute_penalty(&violations, 0, 100);
        assert!((penalty - 400.0).abs() < 1e-10); // 100 * 2^2 = 400
    }

    #[test]
    fn test_adaptive_cooling_state_update() {
        let mut state = AdaptiveCoolingState {
            window_size: 5,
            ..Default::default()
        };
        // All accepted -> should speed up cooling
        for _ in 0..5 {
            state.update(true, 0.44, 1.05);
        }
        assert!(state.effective_alpha > 0.95); // alpha should increase
    }
}
