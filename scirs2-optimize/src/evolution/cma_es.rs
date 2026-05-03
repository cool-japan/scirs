//! Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
//!
//! CMA-ES is a stochastic, derivative-free method for numerical optimization
//! of non-linear or non-convex continuous optimization problems. It belongs
//! to the class of evolutionary algorithms and evolution strategies.
//!
//! ## Key features
//!
//! - Invariant under order-preserving transformations of the fitness function
//! - Robust to scaling, rotation, and translation of the search space
//! - Adaptive step-size control (Cumulative Step-size Adaptation / CSA)
//! - Covariance matrix adaptation via rank-1 and rank-mu updates
//! - IPOP restart strategy for escaping local optima
//! - Multiple boundary handling strategies
//!
//! ## References
//!
//! - Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772
//! - Auger, A. & Hansen, N. (2005). A Restart CMA Evolution Strategy with
//!   Increasing Population Size. CEC 2005.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::RngExt;

/// Boundary handling strategy for CMA-ES
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryHandling {
    /// No boundary handling (unconstrained)
    None,
    /// Project infeasible solutions onto the boundary
    Projection,
    /// Reflect infeasible solutions at the boundary
    Reflection,
    /// Penalize infeasible solutions with a quadratic penalty
    Penalty {
        /// Penalty weight factor
        weight: f64,
    },
    /// Resample infeasible solutions until feasible
    Resampling {
        /// Maximum number of resampling attempts
        max_attempts: usize,
    },
}

impl Default for BoundaryHandling {
    fn default() -> Self {
        BoundaryHandling::Reflection
    }
}

/// Restart strategy for CMA-ES
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RestartStrategy {
    /// No restart (single run)
    NoRestart,
    /// IPOP: Increasing Population Size restart
    /// Population doubles on each restart
    Ipop {
        /// Maximum number of restarts
        max_restarts: usize,
    },
    /// BIPOP: Bi-Population restart
    /// Alternates between large and small populations
    Bipop {
        /// Maximum number of restarts
        max_restarts: usize,
    },
}

impl Default for RestartStrategy {
    fn default() -> Self {
        RestartStrategy::Ipop { max_restarts: 9 }
    }
}

/// Options for CMA-ES optimization
#[derive(Debug, Clone)]
pub struct CmaEsOptions {
    /// Initial step size (sigma). Controls the spread of the initial search distribution.
    pub sigma0: f64,
    /// Population size (lambda). If None, uses default 4 + floor(3 * ln(n)).
    pub population_size: Option<usize>,
    /// Maximum number of function evaluations
    pub max_fevals: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Function value tolerance for convergence
    pub ftol: f64,
    /// Solution tolerance for convergence
    pub xtol: f64,
    /// Boundary handling strategy
    pub boundary_handling: BoundaryHandling,
    /// Restart strategy
    pub restart_strategy: RestartStrategy,
    /// Lower bounds for each dimension (None = unbounded)
    pub lower_bounds: Option<Vec<f64>>,
    /// Upper bounds for each dimension (None = unbounded)
    pub upper_bounds: Option<Vec<f64>>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Verbosity level (0 = silent, 1 = per-restart, 2 = per-iteration)
    pub verbosity: usize,
}

impl Default for CmaEsOptions {
    fn default() -> Self {
        Self {
            sigma0: 0.3,
            population_size: None,
            max_fevals: 100_000,
            max_iterations: 100_000,
            ftol: 1e-12,
            xtol: 1e-12,
            boundary_handling: BoundaryHandling::default(),
            restart_strategy: RestartStrategy::default(),
            lower_bounds: None,
            upper_bounds: None,
            seed: None,
            verbosity: 0,
        }
    }
}

/// Result of CMA-ES optimization
#[derive(Debug, Clone)]
pub struct CmaEsResult {
    /// Best solution found
    pub x: Array1<f64>,
    /// Best function value found
    pub fun: f64,
    /// Total number of function evaluations
    pub nfev: usize,
    /// Total number of iterations (across all restarts)
    pub nit: usize,
    /// Number of restarts performed
    pub n_restarts: usize,
    /// Whether optimization converged successfully
    pub success: bool,
    /// Termination message
    pub message: String,
    /// Final step size (sigma)
    pub sigma_final: f64,
    /// Final condition number of the covariance matrix
    pub cond_final: f64,
}

/// Internal state of the CMA-ES algorithm
#[derive(Debug)]
pub struct CmaEsState {
    /// Dimension of the problem
    n: usize,
    /// Current mean of the distribution
    mean: Array1<f64>,
    /// Current step size
    sigma: f64,
    /// Population size (lambda)
    lambda: usize,
    /// Number of selected parents (mu)
    mu: usize,
    /// Recombination weights
    weights: Vec<f64>,
    /// Variance effective selection mass
    mu_eff: f64,
    /// Evolution path for sigma (cumulation)
    p_sigma: Array1<f64>,
    /// Evolution path for covariance matrix (cumulation)
    p_c: Array1<f64>,
    /// Covariance matrix
    cov: Array2<f64>,
    /// Eigenvalues of the covariance matrix (squared, i.e. D^2)
    eigenvalues: Array1<f64>,
    /// Eigenvectors (columns of B)
    eigenvectors: Array2<f64>,
    /// Inverse square root of covariance matrix: C^(-1/2)
    inv_sqrt_cov: Array2<f64>,
    /// Learning rate for cumulation of sigma
    c_sigma: f64,
    /// Damping for sigma
    d_sigma: f64,
    /// Learning rate for cumulation of C
    c_c: f64,
    /// Learning rate for rank-1 update
    c_1: f64,
    /// Learning rate for rank-mu update
    c_mu: f64,
    /// Expected norm of N(0,I) distributed random vector
    chi_n: f64,
    /// Iteration counter
    generation: usize,
    /// Total function evaluations
    fevals: usize,
    /// Best solution found so far
    best_x: Array1<f64>,
    /// Best function value found so far
    best_f: f64,
    /// Eigendecomposition update counter
    eigen_update_counter: usize,
    /// RNG
    rng: StdRng,
}

impl CmaEsState {
    /// Create a new CMA-ES state
    pub fn new(x0: &[f64], sigma0: f64, lambda: Option<usize>, seed: u64) -> OptimizeResult<Self> {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Dimension must be at least 1".to_string(),
            ));
        }
        if sigma0 <= 0.0 || !sigma0.is_finite() {
            return Err(OptimizeError::InvalidInput(format!(
                "sigma0 must be positive and finite, got {}",
                sigma0
            )));
        }

        // Default population size: 4 + floor(3 * ln(n))
        let lambda = lambda.unwrap_or_else(|| 4 + (3.0 * (n as f64).ln()).floor() as usize);
        let lambda = lambda.max(4); // minimum population size
        let mu = lambda / 2;

        // Recombination weights (log-linear)
        let raw_weights: Vec<f64> = (0..mu)
            .map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0))
            .collect();
        let w_sum: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / w_sum).collect();

        // Variance effective selection mass
        let w_sq_sum: f64 = weights.iter().map(|w| w * w).sum();
        let mu_eff = 1.0 / w_sq_sum;

        // Strategy parameter setting: adaptation
        let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);
        let c_1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let c_mu_candidate = (1.0 - c_1)
            .min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n as f64 + 2.0).powi(2) + mu_eff));
        let c_mu = c_mu_candidate.max(0.0);

        // Expected norm of N(0,I)
        let chi_n =
            (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));

        let mean = Array1::from_vec(x0.to_vec());
        let cov = Array2::eye(n);
        let eigenvalues = Array1::ones(n);
        let eigenvectors = Array2::eye(n);
        let inv_sqrt_cov = Array2::eye(n);
        let p_sigma = Array1::zeros(n);
        let p_c = Array1::zeros(n);

        let rng = StdRng::seed_from_u64(seed);

        Ok(Self {
            n,
            mean: mean.clone(),
            sigma: sigma0,
            lambda,
            mu,
            weights,
            mu_eff,
            p_sigma,
            p_c,
            cov,
            eigenvalues,
            eigenvectors,
            inv_sqrt_cov,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
            generation: 0,
            fevals: 0,
            best_x: mean,
            best_f: f64::INFINITY,
            eigen_update_counter: 0,
            rng,
        })
    }

    /// Sample a population of candidate solutions
    fn sample_population(&mut self) -> Vec<Array1<f64>> {
        let mut population = Vec::with_capacity(self.lambda);
        for _ in 0..self.lambda {
            // z ~ N(0, I)
            let z: Array1<f64> = Array1::from_vec(
                (0..self.n)
                    .map(|_| sample_standard_normal(&mut self.rng))
                    .collect(),
            );
            // y = B * D * z  (where C = B * D^2 * B^T)
            let d_z = &z * &self.eigenvalues.mapv(f64::sqrt);
            let y = self.eigenvectors.dot(&d_z);
            // x = mean + sigma * y
            let x = &self.mean + &(y * self.sigma);
            population.push(x);
        }
        population
    }

    /// Apply boundary handling to a candidate solution
    fn apply_boundary_handling(
        &mut self,
        x: &Array1<f64>,
        lower: &Option<Vec<f64>>,
        upper: &Option<Vec<f64>>,
        handling: BoundaryHandling,
    ) -> (Array1<f64>, f64) {
        let mut x_fixed = x.clone();
        let mut penalty = 0.0;

        let lb = lower.as_deref();
        let ub = upper.as_deref();

        if lb.is_none() && ub.is_none() {
            return (x_fixed, 0.0);
        }

        match handling {
            BoundaryHandling::None => {}
            BoundaryHandling::Projection => {
                for i in 0..self.n {
                    if let Some(lb_vals) = lb {
                        if x_fixed[i] < lb_vals[i] {
                            x_fixed[i] = lb_vals[i];
                        }
                    }
                    if let Some(ub_vals) = ub {
                        if x_fixed[i] > ub_vals[i] {
                            x_fixed[i] = ub_vals[i];
                        }
                    }
                }
            }
            BoundaryHandling::Reflection => {
                for i in 0..self.n {
                    let lo = lb.map_or(f64::NEG_INFINITY, |v| v[i]);
                    let hi = ub.map_or(f64::INFINITY, |v| v[i]);
                    if lo.is_finite() && hi.is_finite() {
                        let range = hi - lo;
                        if range > 0.0 {
                            // Reflect until within bounds
                            let mut val = x_fixed[i];
                            for _ in 0..10 {
                                if val < lo {
                                    val = lo + (lo - val);
                                } else if val > hi {
                                    val = hi - (val - hi);
                                } else {
                                    break;
                                }
                            }
                            // Final clamp if reflection didn't converge
                            x_fixed[i] = val.clamp(lo, hi);
                        }
                    } else {
                        if lo.is_finite() && x_fixed[i] < lo {
                            x_fixed[i] = lo;
                        }
                        if hi.is_finite() && x_fixed[i] > hi {
                            x_fixed[i] = hi;
                        }
                    }
                }
            }
            BoundaryHandling::Penalty { weight } => {
                for i in 0..self.n {
                    if let Some(lb_vals) = lb {
                        if x_fixed[i] < lb_vals[i] {
                            let diff = lb_vals[i] - x_fixed[i];
                            penalty += weight * diff * diff;
                            x_fixed[i] = lb_vals[i];
                        }
                    }
                    if let Some(ub_vals) = ub {
                        if x_fixed[i] > ub_vals[i] {
                            let diff = x_fixed[i] - ub_vals[i];
                            penalty += weight * diff * diff;
                            x_fixed[i] = ub_vals[i];
                        }
                    }
                }
            }
            BoundaryHandling::Resampling { max_attempts } => {
                let mut feasible = true;
                for i in 0..self.n {
                    let lo = lb.map_or(f64::NEG_INFINITY, |v| v[i]);
                    let hi = ub.map_or(f64::INFINITY, |v| v[i]);
                    if x_fixed[i] < lo || x_fixed[i] > hi {
                        feasible = false;
                        break;
                    }
                }
                if !feasible {
                    // Try resampling
                    for _ in 0..max_attempts {
                        let z: Array1<f64> = Array1::from_vec(
                            (0..self.n)
                                .map(|_| sample_standard_normal(&mut self.rng))
                                .collect(),
                        );
                        let d_z = &z * &self.eigenvalues.mapv(f64::sqrt);
                        let y = self.eigenvectors.dot(&d_z);
                        x_fixed = &self.mean + &(y * self.sigma);

                        let mut all_feasible = true;
                        for i in 0..self.n {
                            let lo = lb.map_or(f64::NEG_INFINITY, |v| v[i]);
                            let hi = ub.map_or(f64::INFINITY, |v| v[i]);
                            if x_fixed[i] < lo || x_fixed[i] > hi {
                                all_feasible = false;
                                break;
                            }
                        }
                        if all_feasible {
                            return (x_fixed, 0.0);
                        }
                    }
                    // If all resampling failed, project
                    for i in 0..self.n {
                        if let Some(lb_vals) = lb {
                            if x_fixed[i] < lb_vals[i] {
                                x_fixed[i] = lb_vals[i];
                            }
                        }
                        if let Some(ub_vals) = ub {
                            if x_fixed[i] > ub_vals[i] {
                                x_fixed[i] = ub_vals[i];
                            }
                        }
                    }
                }
            }
        }

        (x_fixed, penalty)
    }

    /// Perform eigendecomposition of the covariance matrix
    fn update_eigen(&mut self) {
        // Symmetrize C (numerical safety)
        let n = self.n;
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (self.cov[[i, j]] + self.cov[[j, i]]);
                self.cov[[i, j]] = avg;
                self.cov[[j, i]] = avg;
            }
        }

        // Simple Jacobi eigendecomposition for small-medium problems
        let (eigenvalues, eigenvectors) = jacobi_eigen(&self.cov, n);

        // Ensure eigenvalues are positive (numerical stability)
        let min_eigenval = 1e-20;
        self.eigenvalues = eigenvalues.mapv(|v| v.max(min_eigenval));
        self.eigenvectors = eigenvectors;

        // Compute C^(-1/2) = B * D^(-1) * B^T
        let d_inv = self.eigenvalues.mapv(|v| 1.0 / v.sqrt());
        let mut inv_sqrt = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self.eigenvectors[[i, k]] * d_inv[k] * self.eigenvectors[[j, k]];
                }
                inv_sqrt[[i, j]] = sum;
            }
        }
        self.inv_sqrt_cov = inv_sqrt;
    }

    /// Update the CMA-ES state after evaluating a generation
    fn update(&mut self, population: &[Array1<f64>], fitness: &[f64]) {
        // Sort by fitness (ascending)
        let mut indices: Vec<usize> = (0..self.lambda).collect();
        indices.sort_by(|&a, &b| {
            fitness[a]
                .partial_cmp(&fitness[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update best
        if fitness[indices[0]] < self.best_f {
            self.best_f = fitness[indices[0]];
            self.best_x = population[indices[0]].clone();
        }

        // Compute weighted mean of selected points
        let old_mean = self.mean.clone();
        self.mean = Array1::zeros(self.n);
        for i in 0..self.mu {
            let idx = indices[i];
            self.mean = &self.mean + &(&population[idx] * self.weights[i]);
        }

        // Compute displacement from old mean
        let mean_diff = &self.mean - &old_mean;

        // Update evolution path for sigma (p_sigma)
        let inv_sqrt_cov_mean_diff = self.inv_sqrt_cov.dot(&mean_diff);
        let c_sigma_factor = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        self.p_sigma = &self.p_sigma * (1.0 - self.c_sigma)
            + &(&inv_sqrt_cov_mean_diff * (c_sigma_factor / self.sigma));

        // Determine if we should use heavy-side function (h_sigma)
        let p_sigma_norm = self.p_sigma.dot(&self.p_sigma).sqrt();
        let threshold = (1.0 - (1.0 - self.c_sigma).powi(2 * (self.generation as i32 + 1))).sqrt()
            * (1.4 + 2.0 / (self.n as f64 + 1.0))
            * self.chi_n;
        let h_sigma: f64 = if p_sigma_norm < threshold { 1.0 } else { 0.0 };

        // Update evolution path for C (p_c)
        let c_c_factor = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        self.p_c =
            &self.p_c * (1.0 - self.c_c) + &(&mean_diff * (h_sigma * c_c_factor / self.sigma));

        // Rank-1 update component
        let pc_outer = outer_product(&self.p_c, &self.p_c);

        // Correction factor for h_sigma
        let delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);

        // Rank-mu update component
        let mut rank_mu_update: Array2<f64> = Array2::zeros((self.n, self.n));
        for i in 0..self.mu {
            let idx = indices[i];
            let y_i = (&population[idx] - &old_mean) / self.sigma;
            let y_outer = outer_product(&y_i, &y_i);
            rank_mu_update = rank_mu_update + &(&y_outer * self.weights[i]);
        }

        // Update covariance matrix
        // C = (1 - c_1 - c_mu + delta_h * c_1) * C + c_1 * pc_outer + c_mu * rank_mu_update
        let scale = 1.0 - self.c_1 - self.c_mu + delta_h * self.c_1;
        self.cov = &self.cov * scale + &(&pc_outer * self.c_1) + &(&rank_mu_update * self.c_mu);

        // Update step size (sigma) via CSA
        let sigma_exp = (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0);
        self.sigma *= sigma_exp.exp();

        // Clamp sigma to prevent explosion/implosion
        self.sigma = self.sigma.clamp(1e-20, 1e20);

        // Update eigendecomposition periodically
        self.eigen_update_counter += 1;
        let update_freq = (self.lambda as f64 / ((self.c_1 + self.c_mu) * self.n as f64 * 10.0))
            .max(1.0) as usize;
        if self.eigen_update_counter >= update_freq {
            self.update_eigen();
            self.eigen_update_counter = 0;
        }

        self.generation += 1;
    }

    /// Check stopping criteria
    fn check_termination(&self, recent_fitness: &[f64], options: &CmaEsOptions) -> Option<String> {
        // Max function evaluations
        if self.fevals >= options.max_fevals {
            return Some(format!(
                "Maximum function evaluations ({}) reached",
                options.max_fevals
            ));
        }

        // Max iterations
        if self.generation >= options.max_iterations {
            return Some(format!(
                "Maximum iterations ({}) reached",
                options.max_iterations
            ));
        }

        // Function value tolerance (flat fitness landscape)
        if recent_fitness.len() >= self.lambda {
            let f_best = recent_fitness.iter().copied().fold(f64::INFINITY, f64::min);
            let f_worst = recent_fitness
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if (f_worst - f_best).abs() < options.ftol {
                return Some("Function tolerance reached (flat fitness)".to_string());
            }
        }

        // Solution tolerance (sigma * max(eigenvalue) is very small)
        let max_eigenval = self.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
        if self.sigma * max_eigenval.sqrt() < options.xtol {
            return Some("Solution tolerance reached".to_string());
        }

        // Condition number check
        let min_eigenval = self
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if min_eigenval > 0.0 {
            let cond = max_eigenval / min_eigenval;
            if cond > 1e14 {
                return Some(format!("Condition number too large: {:.2e}", cond));
            }
        }

        // Sigma too small
        if self.sigma < 1e-20 {
            return Some("Step size sigma below minimum threshold".to_string());
        }

        None
    }

    /// Get the condition number of the covariance matrix
    pub fn condition_number(&self) -> f64 {
        let max_ev = self.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
        let min_ev = self
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if min_ev > 0.0 {
            max_ev / min_ev
        } else {
            f64::INFINITY
        }
    }
}

/// IPOP-CMA-ES: CMA-ES with Increasing Population Restarts
pub struct IpopCmaEs<F>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    func: F,
    x0: Vec<f64>,
    options: CmaEsOptions,
}

impl<F> IpopCmaEs<F>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    /// Create a new IPOP-CMA-ES optimizer
    pub fn new(func: F, x0: &[f64], options: CmaEsOptions) -> Self {
        Self {
            func,
            x0: x0.to_vec(),
            options,
        }
    }

    /// Run the IPOP-CMA-ES algorithm
    pub fn run(&self) -> OptimizeResult<CmaEsResult> {
        let max_restarts = match self.options.restart_strategy {
            RestartStrategy::NoRestart => 0,
            RestartStrategy::Ipop { max_restarts } => max_restarts,
            RestartStrategy::Bipop { max_restarts } => max_restarts,
        };

        let seed = self
            .options
            .seed
            .unwrap_or_else(|| scirs2_core::random::rng().random_range(0..u64::MAX));

        let mut overall_best_x = Array1::from_vec(self.x0.clone());
        let mut overall_best_f = f64::INFINITY;
        let mut total_fevals = 0_usize;
        let mut total_iterations = 0_usize;
        let mut final_message = String::new();
        let mut final_sigma = self.options.sigma0;
        let mut final_cond = 1.0;
        let mut any_success = false;

        let base_lambda = self
            .options
            .population_size
            .unwrap_or_else(|| 4 + (3.0 * (self.x0.len() as f64).ln()).floor() as usize);

        let mut rng = StdRng::seed_from_u64(seed);

        for restart_idx in 0..=max_restarts {
            // Increase population size for IPOP
            let current_lambda = if restart_idx == 0 {
                base_lambda
            } else {
                match self.options.restart_strategy {
                    RestartStrategy::Ipop { .. } => base_lambda * (1 << restart_idx.min(10)),
                    RestartStrategy::Bipop { .. } => {
                        // Alternate between large and small
                        if restart_idx % 2 == 1 {
                            base_lambda * (1 << ((restart_idx / 2 + 1).min(10)))
                        } else {
                            (base_lambda / 2).max(4)
                        }
                    }
                    RestartStrategy::NoRestart => base_lambda,
                }
            };

            // For restarts > 0, randomize starting point within bounds
            let x0_restart = if restart_idx == 0 {
                self.x0.clone()
            } else {
                let mut x0_new = self.x0.clone();
                for i in 0..x0_new.len() {
                    let lo = self.options.lower_bounds.as_ref().map_or(-10.0, |lb| lb[i]);
                    let hi = self.options.upper_bounds.as_ref().map_or(10.0, |ub| ub[i]);
                    x0_new[i] = rng.random_range(lo..hi);
                }
                x0_new
            };

            let run_seed = seed.wrapping_add(restart_idx as u64 * 1_000_000);
            let result = self.run_single(
                &x0_restart,
                current_lambda,
                run_seed,
                self.options.max_fevals.saturating_sub(total_fevals),
            )?;

            total_fevals += result.nfev;
            total_iterations += result.nit;

            if result.fun < overall_best_f {
                overall_best_f = result.fun;
                overall_best_x = result.x.clone();
                final_sigma = result.sigma_final;
                final_cond = result.cond_final;
            }
            if result.success {
                any_success = true;
            }
            final_message = result.message.clone();

            // Check if we've exhausted the budget
            if total_fevals >= self.options.max_fevals {
                final_message = "Budget exhausted across restarts".to_string();
                break;
            }

            // If converged with good tolerance, no need for more restarts
            if result.success && result.fun < self.options.ftol {
                break;
            }
        }

        let n_restarts = if any_success { 0 } else { max_restarts };

        Ok(CmaEsResult {
            x: overall_best_x,
            fun: overall_best_f,
            nfev: total_fevals,
            nit: total_iterations,
            n_restarts,
            success: any_success || overall_best_f < f64::INFINITY,
            message: final_message,
            sigma_final: final_sigma,
            cond_final: final_cond,
        })
    }

    /// Run a single CMA-ES optimization
    fn run_single(
        &self,
        x0: &[f64],
        lambda: usize,
        seed: u64,
        max_fevals: usize,
    ) -> OptimizeResult<CmaEsResult> {
        let mut state = CmaEsState::new(x0, self.options.sigma0, Some(lambda), seed)?;

        // Evaluate initial point
        let x0_arr = Array1::from_vec(x0.to_vec());
        let f0 = (self.func)(&x0_arr.view());
        state.fevals += 1;
        if f0 < state.best_f {
            state.best_f = f0;
            state.best_x = x0_arr;
        }

        let mut termination_msg = String::new();
        let mut converged = false;

        loop {
            // Sample population
            let mut population = state.sample_population();

            // Apply boundary handling
            let mut penalties = vec![0.0; state.lambda];
            for i in 0..state.lambda {
                let (x_fixed, pen) = state.apply_boundary_handling(
                    &population[i],
                    &self.options.lower_bounds,
                    &self.options.upper_bounds,
                    self.options.boundary_handling,
                );
                population[i] = x_fixed;
                penalties[i] = pen;
            }

            // Evaluate fitness
            let mut fitness: Vec<f64> = population.iter().map(|x| (self.func)(&x.view())).collect();
            state.fevals += state.lambda;

            // Add penalties
            for (f, p) in fitness.iter_mut().zip(penalties.iter()) {
                *f += *p;
            }

            // Check termination
            if let Some(msg) = state.check_termination(&fitness, &self.options) {
                termination_msg = msg;
                // If solution tolerance was reached, it's a success
                converged = termination_msg.contains("tolerance reached");
                break;
            }

            // Check local budget
            if state.fevals >= max_fevals {
                termination_msg = "Local budget exhausted".to_string();
                break;
            }

            // Update state
            state.update(&population, &fitness);
        }

        let cond_final = state.condition_number();
        Ok(CmaEsResult {
            x: state.best_x,
            fun: state.best_f,
            nfev: state.fevals,
            nit: state.generation,
            n_restarts: 0,
            success: converged,
            message: termination_msg,
            sigma_final: state.sigma,
            cond_final,
        })
    }
}

/// Convenience function to minimize using CMA-ES
///
/// # Arguments
///
/// * `func` - Objective function to minimize
/// * `x0` - Initial guess
/// * `options` - CMA-ES options (uses defaults if None)
///
/// # Returns
///
/// * `CmaEsResult` with the best solution found
pub fn cma_es_minimize<F>(
    func: F,
    x0: &[f64],
    options: Option<CmaEsOptions>,
) -> OptimizeResult<CmaEsResult>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let options = options.unwrap_or_default();
    let optimizer = IpopCmaEs::new(func, x0, options);
    optimizer.run()
}

// ---- Utility functions ----

/// Sample from standard normal using Box-Muller transform
fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(1e-10..1.0);
    let u2: f64 = rng.random_range(0.0..std::f64::consts::TAU);
    (-2.0 * u1.ln()).sqrt() * u2.cos()
}

/// Compute outer product of two vectors
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

/// Jacobi eigendecomposition for symmetric matrices
/// Returns (eigenvalues, eigenvectors) where eigenvectors are column-wise
fn jacobi_eigen(mat: &Array2<f64>, n: usize) -> (Array1<f64>, Array2<f64>) {
    let mut a = mat.clone();
    let mut v = Array2::eye(n);
    let max_iter = 100 * n * n;
    let tol = 1e-15;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[[i, j]].abs() > max_val {
                    max_val = a[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute rotation
        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];

        let theta = if (app - aqq).abs() < tol {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply Givens rotation
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[[i, p]] = cos_t * a[[i, p]] + sin_t * a[[i, q]];
                new_a[[p, i]] = new_a[[i, p]];
                new_a[[i, q]] = -sin_t * a[[i, p]] + cos_t * a[[i, q]];
                new_a[[q, i]] = new_a[[i, q]];
            }
        }
        new_a[[p, p]] = cos_t * cos_t * app + 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq;
        new_a[[q, q]] = sin_t * sin_t * app - 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq;
        new_a[[p, q]] = 0.0;
        new_a[[q, p]] = 0.0;
        a = new_a;

        // Update eigenvectors
        let mut new_v = v.clone();
        for i in 0..n {
            new_v[[i, p]] = cos_t * v[[i, p]] + sin_t * v[[i, q]];
            new_v[[i, q]] = -sin_t * v[[i, p]] + cos_t * v[[i, q]];
        }
        v = new_v;
    }

    let eigenvalues = Array1::from_vec((0..n).map(|i| a[[i, i]]).collect());
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sphere function: sum(x_i^2)
    fn sphere(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    /// Rosenbrock function
    fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    /// Rastrigin function (multimodal)
    fn rastrigin(x: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let mut sum = 10.0 * n;
        for &xi in x.iter() {
            sum += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
        }
        sum
    }

    #[test]
    fn test_cma_es_sphere_2d() {
        let options = CmaEsOptions {
            sigma0: 0.5,
            max_fevals: 10_000,
            restart_strategy: RestartStrategy::NoRestart,
            seed: Some(42),
            ..Default::default()
        };
        let result = cma_es_minimize(sphere, &[3.0, -2.0], Some(options));
        assert!(result.is_ok());
        let res = result.expect("CMA-ES sphere 2D failed");
        assert!(res.fun < 1e-6, "Sphere function value: {}", res.fun);
        for &xi in res.x.iter() {
            assert!(xi.abs() < 1e-3, "Solution component: {}", xi);
        }
    }

    #[test]
    fn test_cma_es_sphere_5d() {
        let options = CmaEsOptions {
            sigma0: 1.0,
            max_fevals: 50_000,
            restart_strategy: RestartStrategy::NoRestart,
            seed: Some(123),
            ..Default::default()
        };
        let result = cma_es_minimize(sphere, &[5.0, -3.0, 2.0, -1.0, 4.0], Some(options));
        assert!(result.is_ok());
        let res = result.expect("CMA-ES sphere 5D failed");
        assert!(res.fun < 1e-4, "Sphere 5D value: {}", res.fun);
    }

    #[test]
    fn test_cma_es_rosenbrock_2d() {
        let options = CmaEsOptions {
            sigma0: 0.5,
            max_fevals: 50_000,
            restart_strategy: RestartStrategy::NoRestart,
            seed: Some(99),
            ..Default::default()
        };
        let result = cma_es_minimize(rosenbrock, &[0.0, 0.0], Some(options));
        assert!(result.is_ok());
        let res = result.expect("CMA-ES Rosenbrock 2D failed");
        // Rosenbrock is harder, allow more tolerance
        assert!(res.fun < 1e-2, "Rosenbrock value: {}", res.fun);
    }

    #[test]
    fn test_cma_es_with_bounds() {
        let options = CmaEsOptions {
            sigma0: 0.3,
            max_fevals: 10_000,
            restart_strategy: RestartStrategy::NoRestart,
            lower_bounds: Some(vec![0.0, 0.0]),
            upper_bounds: Some(vec![5.0, 5.0]),
            boundary_handling: BoundaryHandling::Reflection,
            seed: Some(77),
            ..Default::default()
        };

        // Minimum of sphere is at (0,0), which is on the boundary
        let result = cma_es_minimize(sphere, &[2.5, 2.5], Some(options));
        assert!(result.is_ok());
        let res = result.expect("CMA-ES bounded failed");
        assert!(res.fun < 0.1, "Bounded sphere value: {}", res.fun);
        // Check bounds are respected
        for &xi in res.x.iter() {
            assert!(xi >= -0.01, "Lower bound violated: {}", xi);
            assert!(xi <= 5.01, "Upper bound violated: {}", xi);
        }
    }

    #[test]
    fn test_cma_es_ipop_restart() {
        let options = CmaEsOptions {
            sigma0: 2.0,
            max_fevals: 50_000,
            restart_strategy: RestartStrategy::Ipop { max_restarts: 3 },
            lower_bounds: Some(vec![-5.12, -5.12]),
            upper_bounds: Some(vec![5.12, 5.12]),
            seed: Some(55),
            ..Default::default()
        };

        let result = cma_es_minimize(rastrigin, &[3.0, -2.0], Some(options));
        assert!(result.is_ok());
        let res = result.expect("IPOP CMA-ES Rastrigin failed");
        // Rastrigin global minimum is 0 at origin
        assert!(res.fun < 5.0, "Rastrigin IPOP value: {}", res.fun);
    }

    #[test]
    fn test_cma_es_penalty_boundary() {
        let options = CmaEsOptions {
            sigma0: 0.5,
            max_fevals: 10_000,
            restart_strategy: RestartStrategy::NoRestart,
            lower_bounds: Some(vec![-1.0, -1.0]),
            upper_bounds: Some(vec![1.0, 1.0]),
            boundary_handling: BoundaryHandling::Penalty { weight: 100.0 },
            seed: Some(42),
            ..Default::default()
        };

        let result = cma_es_minimize(sphere, &[0.5, 0.5], Some(options));
        assert!(result.is_ok());
    }

    #[test]
    fn test_cma_es_projection_boundary() {
        let options = CmaEsOptions {
            sigma0: 0.5,
            max_fevals: 10_000,
            restart_strategy: RestartStrategy::NoRestart,
            lower_bounds: Some(vec![-2.0, -2.0]),
            upper_bounds: Some(vec![2.0, 2.0]),
            boundary_handling: BoundaryHandling::Projection,
            seed: Some(42),
            ..Default::default()
        };

        let result = cma_es_minimize(sphere, &[1.0, 1.0], Some(options));
        assert!(result.is_ok());
        let res = result.expect("CMA-ES projection failed");
        assert!(res.fun < 0.01, "Projection sphere value: {}", res.fun);
    }

    #[test]
    fn test_cma_es_resampling_boundary() {
        let options = CmaEsOptions {
            sigma0: 0.3,
            max_fevals: 10_000,
            restart_strategy: RestartStrategy::NoRestart,
            lower_bounds: Some(vec![-2.0, -2.0]),
            upper_bounds: Some(vec![2.0, 2.0]),
            boundary_handling: BoundaryHandling::Resampling { max_attempts: 100 },
            seed: Some(42),
            ..Default::default()
        };

        let result = cma_es_minimize(sphere, &[1.0, 1.0], Some(options));
        assert!(result.is_ok());
    }

    #[test]
    fn test_state_creation() {
        let state = CmaEsState::new(&[0.0, 0.0, 0.0], 0.5, None, 42);
        assert!(state.is_ok());
        let s = state.expect("State creation failed");
        assert_eq!(s.n, 3);
        assert_eq!(s.sigma, 0.5);
        assert!(s.lambda >= 4);
        assert!(s.mu >= 2);
    }

    #[test]
    fn test_state_creation_invalid() {
        let state = CmaEsState::new(&[], 0.5, None, 42);
        assert!(state.is_err());

        let state = CmaEsState::new(&[0.0], -1.0, None, 42);
        assert!(state.is_err());

        let state = CmaEsState::new(&[0.0], f64::NAN, None, 42);
        assert!(state.is_err());
    }

    #[test]
    fn test_jacobi_eigen_identity() {
        let mat = Array2::eye(3);
        let (eigenvalues, eigenvectors) = jacobi_eigen(&mat, 3);
        for &ev in eigenvalues.iter() {
            assert!((ev - 1.0).abs() < 1e-10);
        }
        // Eigenvectors should form orthonormal basis
        let prod = eigenvectors.t().dot(&eigenvectors);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[[i, j]] - expected).abs() < 1e-10,
                    "Orthogonality check failed at ({}, {}): {}",
                    i,
                    j,
                    prod[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_bipop_restart() {
        let options = CmaEsOptions {
            sigma0: 1.0,
            max_fevals: 20_000,
            restart_strategy: RestartStrategy::Bipop { max_restarts: 2 },
            seed: Some(42),
            ..Default::default()
        };

        let result = cma_es_minimize(sphere, &[3.0, -2.0], Some(options));
        assert!(result.is_ok());
    }
}
