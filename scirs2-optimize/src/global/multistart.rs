//! Advanced Multi-Start Methods for Global Optimization
//!
//! This module provides advanced multi-start strategies that go beyond simple
//! random restarts, including basin-hopping variants, stochastic tunneling,
//! and deflation methods for finding multiple optima.
//!
//! ## Algorithms
//!
//! - **Multi-Start Local Search**: Parallel local optimization from diverse starting points
//! - **Monotonic Basin-Hopping**: Basin-hopping with monotonic acceptance (always move downhill)
//! - **Stochastic Tunneling**: Transforms the objective to flatten barriers between basins
//! - **Deflation Methods**: Systematically finds multiple distinct optima
//!
//! ## References
//!
//! - Wales, D.J. & Doye, J.P.K. (1997). Global Optimization by Basin-Hopping
//! - Wenzel, W. & Hamacher, K. (1999). Stochastic Tunneling Approach for Global Minimization
//! - Brown, C.T. & Liebovitch, L.S. (2010). Fractal Analysis

use crate::error::{OptimizeError, OptimizeResult};
use crate::unconstrained::{
    minimize, Bounds as UnconstrainedBounds, Method as UnconstrainedMethod,
    OptimizeResult as LocalOptResult, Options,
};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

/// Bounds type for multi-start methods
pub type Bounds = Vec<(f64, f64)>;

// =====================================================================
// Multi-Start Local Search
// =====================================================================

/// Options for advanced multi-start local search
#[derive(Debug, Clone)]
pub struct AdvancedMultiStartOptions {
    /// Number of starting points
    pub n_starts: usize,
    /// Local optimization method
    pub local_method: UnconstrainedMethod,
    /// Maximum function evaluations per local optimization
    pub max_local_fevals: usize,
    /// Merge tolerance: distinct optima within this distance are merged
    pub merge_tol: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for AdvancedMultiStartOptions {
    fn default() -> Self {
        Self {
            n_starts: 20,
            local_method: UnconstrainedMethod::BFGS,
            max_local_fevals: 5_000,
            merge_tol: 1e-4,
            seed: None,
        }
    }
}

/// Result of advanced multi-start optimization
#[derive(Debug, Clone)]
pub struct AdvancedMultiStartResult {
    /// Best solution found
    pub x: Array1<f64>,
    /// Best function value
    pub fun: f64,
    /// All distinct local optima found (sorted by function value)
    pub local_optima: Vec<(Array1<f64>, f64)>,
    /// Total number of function evaluations
    pub nfev: usize,
    /// Number of successful local optimizations
    pub n_successful: usize,
    /// Whether optimization was successful
    pub success: bool,
    /// Termination message
    pub message: String,
}

/// Run advanced multi-start local search
pub fn advanced_multi_start<F>(
    func: F,
    bounds: &Bounds,
    options: Option<AdvancedMultiStartOptions>,
) -> OptimizeResult<AdvancedMultiStartResult>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone + Send + Sync,
{
    let options = options.unwrap_or_default();
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptimizeError::InvalidInput(
            "Bounds must have at least one dimension".to_string(),
        ));
    }

    let seed = options
        .seed
        .unwrap_or_else(|| scirs2_core::random::rng().random_range(0..u64::MAX));
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate Latin hypercube starting points
    let starting_points = generate_lhs_points(ndim, options.n_starts, bounds, &mut rng);

    let unconstrained_bounds = UnconstrainedBounds::from_vecs(
        bounds.iter().map(|&(lb, _)| Some(lb)).collect(),
        bounds.iter().map(|&(_, ub)| Some(ub)).collect(),
    )
    .ok();

    let mut all_results: Vec<(Array1<f64>, f64)> = Vec::new();
    let mut total_fevals = 0_usize;
    let mut n_successful = 0_usize;

    for x0 in starting_points {
        let f = func.clone();
        let opts = Options {
            bounds: unconstrained_bounds.clone(),
            max_iter: options.max_local_fevals,
            ..Default::default()
        };

        let result = minimize(
            move |x: &ArrayView1<f64>| f(x),
            &x0.to_vec(),
            options.local_method,
            Some(opts),
        );

        match result {
            Ok(res) => {
                total_fevals += res.nfev;
                if res.success {
                    n_successful += 1;
                    all_results.push((res.x, res.fun));
                }
            }
            Err(_) => {
                // Skip failed optimizations
            }
        }
    }

    // Merge nearby optima
    let merged = merge_optima(&all_results, options.merge_tol);

    // Sort by function value
    let mut sorted = merged;
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.is_empty() {
        return Ok(AdvancedMultiStartResult {
            x: Array1::zeros(ndim),
            fun: f64::INFINITY,
            local_optima: Vec::new(),
            nfev: total_fevals,
            n_successful: 0,
            success: false,
            message: "All local optimizations failed".to_string(),
        });
    }

    let best = sorted[0].clone();

    Ok(AdvancedMultiStartResult {
        x: best.0,
        fun: best.1,
        local_optima: sorted,
        nfev: total_fevals,
        n_successful,
        success: true,
        message: format!(
            "Found {} distinct optima from {} starts",
            n_successful, options.n_starts
        ),
    })
}

// =====================================================================
// Monotonic Basin-Hopping
// =====================================================================

/// Options for monotonic basin-hopping
#[derive(Debug, Clone)]
pub struct MonotonicBasinHoppingOptions {
    /// Number of basin-hopping steps
    pub n_iterations: usize,
    /// Step size for perturbation
    pub step_size: f64,
    /// Local optimization method
    pub local_method: UnconstrainedMethod,
    /// Random seed
    pub seed: Option<u64>,
    /// Step size adaptation: increase factor when accepted
    pub step_increase: f64,
    /// Step size adaptation: decrease factor when rejected
    pub step_decrease: f64,
    /// Minimum step size
    pub step_min: f64,
    /// Maximum step size
    pub step_max: f64,
    /// Target acceptance ratio for adaptive step size
    pub target_accept_ratio: f64,
}

impl Default for MonotonicBasinHoppingOptions {
    fn default() -> Self {
        Self {
            n_iterations: 100,
            step_size: 0.5,
            local_method: UnconstrainedMethod::BFGS,
            seed: None,
            step_increase: 1.1,
            step_decrease: 0.9,
            step_min: 1e-6,
            step_max: 10.0,
            target_accept_ratio: 0.5,
        }
    }
}

/// Result of monotonic basin-hopping
#[derive(Debug, Clone)]
pub struct MonotonicBasinHoppingResult {
    /// Best solution found
    pub x: Array1<f64>,
    /// Best function value
    pub fun: f64,
    /// Total function evaluations
    pub nfev: usize,
    /// Number of accepted steps
    pub n_accepted: usize,
    /// Total iterations
    pub nit: usize,
    /// Final step size
    pub final_step_size: f64,
    /// Whether optimization was successful
    pub success: bool,
    /// Message
    pub message: String,
}

/// Run monotonic basin-hopping
///
/// Unlike standard basin-hopping, monotonic basin-hopping only accepts
/// moves that strictly decrease the function value. This converges faster
/// to the nearest deep basin.
pub fn monotonic_basin_hopping<F>(
    func: F,
    x0: &[f64],
    bounds: &Bounds,
    options: Option<MonotonicBasinHoppingOptions>,
) -> OptimizeResult<MonotonicBasinHoppingResult>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    let options = options.unwrap_or_default();
    let ndim = x0.len();

    let seed = options
        .seed
        .unwrap_or_else(|| scirs2_core::random::rng().random_range(0..u64::MAX));
    let mut rng = StdRng::seed_from_u64(seed);

    // Initial local minimization
    let unconstrained_bounds = UnconstrainedBounds::from_vecs(
        bounds.iter().map(|&(lb, _)| Some(lb)).collect(),
        bounds.iter().map(|&(_, ub)| Some(ub)).collect(),
    )
    .ok();

    let initial_opts = Options {
        bounds: unconstrained_bounds.clone(),
        ..Default::default()
    };

    let initial_result = minimize(func.clone(), x0, options.local_method, Some(initial_opts))
        .map_err(|e| {
            OptimizeError::ComputationError(format!("Initial minimization failed: {}", e))
        })?;

    let mut current_x = initial_result.x;
    let mut current_f = initial_result.fun;
    let mut best_x = current_x.clone();
    let mut best_f = current_f;
    let mut total_fevals = initial_result.nfev;
    let mut step_size = options.step_size;
    let mut n_accepted = 0_usize;

    for iteration in 0..options.n_iterations {
        // Perturb current point
        let mut x_new = current_x.clone();
        for i in 0..ndim {
            x_new[i] += rng.random_range(-step_size..step_size);
            // Enforce bounds
            if i < bounds.len() {
                x_new[i] = x_new[i].clamp(bounds[i].0, bounds[i].1);
            }
        }

        // Local minimization from perturbed point
        let local_opts = Options {
            bounds: unconstrained_bounds.clone(),
            ..Default::default()
        };

        let result = minimize(
            func.clone(),
            &x_new.to_vec(),
            options.local_method,
            Some(local_opts),
        );

        if let Ok(res) = result {
            total_fevals += res.nfev;

            // Monotonic acceptance: only accept if strictly better
            if res.fun < current_f {
                current_x = res.x;
                current_f = res.fun;
                n_accepted += 1;

                if current_f < best_f {
                    best_f = current_f;
                    best_x = current_x.clone();
                }

                // Increase step size (encouraging exploration)
                step_size = (step_size * options.step_increase).min(options.step_max);
            } else {
                // Decrease step size (focus on local area)
                step_size = (step_size * options.step_decrease).max(options.step_min);
            }
        }

        // Adaptive step size based on acceptance ratio
        let accept_ratio = if iteration > 0 {
            n_accepted as f64 / (iteration + 1) as f64
        } else {
            0.5
        };

        if accept_ratio < options.target_accept_ratio * 0.5 {
            step_size = (step_size * 0.8).max(options.step_min);
        } else if accept_ratio > options.target_accept_ratio * 1.5 {
            step_size = (step_size * 1.2).min(options.step_max);
        }
    }

    Ok(MonotonicBasinHoppingResult {
        x: best_x,
        fun: best_f,
        nfev: total_fevals,
        n_accepted,
        nit: options.n_iterations,
        final_step_size: step_size,
        success: true,
        message: format!(
            "Monotonic basin-hopping: {} accepted of {} iterations",
            n_accepted, options.n_iterations
        ),
    })
}

// =====================================================================
// Stochastic Tunneling
// =====================================================================

/// Options for stochastic tunneling
#[derive(Debug, Clone)]
pub struct StochasticTunnelingOptions {
    /// Number of iterations
    pub n_iterations: usize,
    /// Temperature parameter (controls tunneling probability)
    pub gamma: f64,
    /// Step size for random walk
    pub step_size: f64,
    /// Local optimization method (used periodically)
    pub local_method: UnconstrainedMethod,
    /// How often to run local optimization (every N iterations)
    pub local_every: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for StochasticTunnelingOptions {
    fn default() -> Self {
        Self {
            n_iterations: 1_000,
            gamma: 1.0,
            step_size: 0.1,
            local_method: UnconstrainedMethod::BFGS,
            local_every: 50,
            seed: None,
        }
    }
}

/// Result of stochastic tunneling
#[derive(Debug, Clone)]
pub struct StochasticTunnelingResult {
    /// Best solution found
    pub x: Array1<f64>,
    /// Best function value
    pub fun: f64,
    /// Total function evaluations
    pub nfev: usize,
    /// Number of iterations
    pub nit: usize,
    /// Number of local optimizations performed
    pub n_local_opts: usize,
    /// Whether successful
    pub success: bool,
    /// Message
    pub message: String,
}

/// Run stochastic tunneling optimization
///
/// Stochastic tunneling transforms the objective function to:
///   STUN(x) = 1 - exp(-gamma * (f(x) - f_ref))
///
/// This flattens barriers between basins while preserving the global minimum,
/// allowing the random walk to "tunnel" through barriers.
pub fn stochastic_tunneling<F>(
    func: F,
    x0: &[f64],
    bounds: &Bounds,
    options: Option<StochasticTunnelingOptions>,
) -> OptimizeResult<StochasticTunnelingResult>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    let options = options.unwrap_or_default();
    let ndim = x0.len();

    let seed = options
        .seed
        .unwrap_or_else(|| scirs2_core::random::rng().random_range(0..u64::MAX));
    let mut rng = StdRng::seed_from_u64(seed);

    let mut current_x = Array1::from_vec(x0.to_vec());
    // Enforce bounds on initial point
    for i in 0..ndim.min(bounds.len()) {
        current_x[i] = current_x[i].clamp(bounds[i].0, bounds[i].1);
    }

    let mut current_f = func(&current_x.view());
    let mut best_x = current_x.clone();
    let mut best_f = current_f;
    let mut f_ref = current_f; // Reference value for STUN transformation
    let mut fevals = 1_usize;
    let mut n_local_opts = 0_usize;

    let unconstrained_bounds = UnconstrainedBounds::from_vecs(
        bounds.iter().map(|&(lb, _)| Some(lb)).collect(),
        bounds.iter().map(|&(_, ub)| Some(ub)).collect(),
    )
    .ok();

    for iteration in 0..options.n_iterations {
        // STUN transformation
        let stun_current = 1.0 - (-options.gamma * (current_f - f_ref)).exp();

        // Propose new point
        let mut x_new = current_x.clone();
        for i in 0..ndim {
            x_new[i] += rng.random_range(-options.step_size..options.step_size);
            if i < bounds.len() {
                x_new[i] = x_new[i].clamp(bounds[i].0, bounds[i].1);
            }
        }

        let f_new = func(&x_new.view());
        fevals += 1;

        let stun_new = 1.0 - (-options.gamma * (f_new - f_ref)).exp();

        // Metropolis acceptance on the STUN-transformed landscape
        let delta_stun = stun_new - stun_current;
        let accept = if delta_stun <= 0.0 {
            true
        } else {
            let prob = (-delta_stun).exp();
            rng.random_range(0.0..1.0) < prob
        };

        if accept {
            current_x = x_new;
            current_f = f_new;

            if current_f < best_f {
                best_f = current_f;
                best_x = current_x.clone();
                f_ref = best_f; // Update reference to best found
            }
        }

        // Periodically run local optimization
        if (iteration + 1) % options.local_every == 0 {
            let local_opts = Options {
                bounds: unconstrained_bounds.clone(),
                ..Default::default()
            };

            if let Ok(res) = minimize(
                func.clone(),
                &current_x.to_vec(),
                options.local_method,
                Some(local_opts),
            ) {
                fevals += res.nfev;
                n_local_opts += 1;

                if res.fun < best_f {
                    best_f = res.fun;
                    best_x = res.x.clone();
                    f_ref = best_f;
                }
                if res.success {
                    current_x = res.x;
                    current_f = res.fun;
                }
            }
        }
    }

    Ok(StochasticTunnelingResult {
        x: best_x,
        fun: best_f,
        nfev: fevals,
        nit: options.n_iterations,
        n_local_opts,
        success: true,
        message: format!(
            "Stochastic tunneling: {} iterations, {} local optimizations",
            options.n_iterations, n_local_opts
        ),
    })
}

// =====================================================================
// Deflation Methods
// =====================================================================

/// Options for deflation-based multi-optima search
#[derive(Debug, Clone)]
pub struct DeflationOptions {
    /// Maximum number of optima to find
    pub max_optima: usize,
    /// Deflation radius: penalty is applied within this radius of known optima
    pub deflation_radius: f64,
    /// Deflation exponent (higher = sharper penalty)
    pub deflation_power: f64,
    /// Number of random starts for each deflated search
    pub n_starts: usize,
    /// Local optimization method
    pub local_method: UnconstrainedMethod,
    /// Function value threshold: stop if all found optima have f > threshold
    pub f_threshold: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for DeflationOptions {
    fn default() -> Self {
        Self {
            max_optima: 10,
            deflation_radius: 0.1,
            deflation_power: 2.0,
            n_starts: 10,
            local_method: UnconstrainedMethod::BFGS,
            f_threshold: f64::INFINITY,
            seed: None,
        }
    }
}

/// Result of deflation-based search
#[derive(Debug, Clone)]
pub struct DeflationResult {
    /// All distinct optima found, sorted by function value
    pub optima: Vec<(Array1<f64>, f64)>,
    /// Total function evaluations
    pub nfev: usize,
    /// Whether successful (found at least one optimum)
    pub success: bool,
    /// Message
    pub message: String,
}

/// Find multiple optima using deflation
///
/// The deflation method works by:
/// 1. Finding a local/global optimum
/// 2. Applying a "deflation" transformation that repels solutions away from known optima
/// 3. Searching the deflated landscape to find the next optimum
/// 4. Repeating until the desired number of optima are found
pub fn deflation_search<F>(
    func: F,
    bounds: &Bounds,
    options: Option<DeflationOptions>,
) -> OptimizeResult<DeflationResult>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    let options = options.unwrap_or_default();
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptimizeError::InvalidInput(
            "Bounds must have at least one dimension".to_string(),
        ));
    }

    let seed = options
        .seed
        .unwrap_or_else(|| scirs2_core::random::rng().random_range(0..u64::MAX));
    let mut rng = StdRng::seed_from_u64(seed);

    let mut found_optima: Vec<(Array1<f64>, f64)> = Vec::new();
    let mut total_fevals = 0_usize;

    let unconstrained_bounds = UnconstrainedBounds::from_vecs(
        bounds.iter().map(|&(lb, _)| Some(lb)).collect(),
        bounds.iter().map(|&(_, ub)| Some(ub)).collect(),
    )
    .ok();

    for search_round in 0..options.max_optima {
        let mut best_result: Option<LocalOptResult<f64>> = None;
        let known = found_optima.clone();
        let deflation_radius = options.deflation_radius;
        let deflation_power = options.deflation_power;

        // Create deflated objective
        let deflated_func = {
            let f = func.clone();
            let known_optima = known.clone();
            move |x: &ArrayView1<f64>| -> f64 {
                let f_val = f(x);

                // Compute deflation factor
                let mut deflation = 1.0;
                for (opt_x, _) in &known_optima {
                    let mut sq_dist = 0.0;
                    for i in 0..x.len() {
                        let diff = x[i] - opt_x[i];
                        sq_dist += diff * diff;
                    }
                    let dist = sq_dist.sqrt();
                    if dist < deflation_radius {
                        // Polynomial deflation: multiply by (dist/radius)^power
                        let ratio = dist / deflation_radius;
                        deflation *= ratio.powf(deflation_power);
                    }
                }

                if deflation < 1e-30 {
                    f64::MAX / 2.0 // Very high value near known optima
                } else {
                    f_val / deflation // Inflate function value near known optima
                }
            }
        };

        // Multi-start search on deflated landscape
        for _ in 0..options.n_starts {
            let mut x0 = vec![0.0; ndim];
            for i in 0..ndim {
                x0[i] = rng.random_range(bounds[i].0..bounds[i].1);
            }

            let local_opts = Options {
                bounds: unconstrained_bounds.clone(),
                ..Default::default()
            };

            let df = deflated_func.clone();
            if let Ok(res) = minimize(
                move |x: &ArrayView1<f64>| df(x),
                &x0,
                options.local_method,
                Some(local_opts),
            ) {
                total_fevals += res.nfev;

                // Evaluate original function at result
                let f_original = func(&res.x.view());
                total_fevals += 1;

                let is_new = !found_optima.iter().any(|(opt_x, _)| {
                    let mut sq_dist = 0.0;
                    for i in 0..ndim {
                        let diff = res.x[i] - opt_x[i];
                        sq_dist += diff * diff;
                    }
                    sq_dist.sqrt() < options.deflation_radius
                });

                if is_new && f_original < options.f_threshold {
                    let update = match &best_result {
                        None => true,
                        Some(prev) => f_original < prev.fun,
                    };
                    if update {
                        best_result = Some(LocalOptResult {
                            x: res.x,
                            fun: f_original,
                            success: true,
                            message: format!("Deflation round {}", search_round),
                            ..Default::default()
                        });
                    }
                }
            }
        }

        match best_result {
            Some(res) => {
                found_optima.push((res.x, res.fun));
            }
            None => {
                // No more optima found
                break;
            }
        }
    }

    // Sort by function value
    found_optima.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let success = !found_optima.is_empty();
    let message = format!("Found {} distinct optima", found_optima.len());

    Ok(DeflationResult {
        optima: found_optima,
        nfev: total_fevals,
        success,
        message,
    })
}

// =====================================================================
// Utility functions
// =====================================================================

/// Generate Latin Hypercube Sampling points
fn generate_lhs_points(
    ndim: usize,
    n: usize,
    bounds: &Bounds,
    rng: &mut StdRng,
) -> Vec<Array1<f64>> {
    let mut points = Vec::with_capacity(n);

    // Create permutation for each dimension
    let mut perms: Vec<Vec<usize>> = (0..ndim)
        .map(|_| {
            let mut perm: Vec<usize> = (0..n).collect();
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                perm.swap(i, j);
            }
            perm
        })
        .collect();

    for i in 0..n {
        let mut point = Array1::zeros(ndim);
        for j in 0..ndim {
            let cell = perms[j][i];
            let u = rng.random_range(0.0..1.0);
            let t = (cell as f64 + u) / n as f64;
            let (lb, ub) = bounds[j];
            point[j] = lb + t * (ub - lb);
        }
        points.push(point);
    }

    let _ = perms; // suppress unused warning
    points
}

/// Merge nearby optima
fn merge_optima(optima: &[(Array1<f64>, f64)], tol: f64) -> Vec<(Array1<f64>, f64)> {
    let mut merged: Vec<(Array1<f64>, f64)> = Vec::new();

    for (x, f) in optima {
        let mut is_duplicate = false;
        for (mx, mf) in &mut merged {
            let mut sq_dist = 0.0;
            for i in 0..x.len() {
                let diff = x[i] - mx[i];
                sq_dist += diff * diff;
            }
            if sq_dist.sqrt() < tol {
                // Keep the better one
                if *f < *mf {
                    *mx = x.clone();
                    *mf = *f;
                }
                is_duplicate = true;
                break;
            }
        }
        if !is_duplicate {
            merged.push((x.clone(), *f));
        }
    }

    merged
}

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
        let mut sum = 10.0 * n;
        for &xi in x.iter() {
            sum += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
        }
        sum
    }

    /// Function with two minima
    fn two_minima(x: &ArrayView1<f64>) -> f64 {
        let x0 = x[0];
        // Two wells at x=-2 (deeper) and x=2 (shallower)
        let well1 = (x0 + 2.0).powi(2);
        let well2 = 0.5 * (x0 - 2.0).powi(2) + 0.5;
        well1.min(well2)
    }

    #[test]
    fn test_advanced_multi_start_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let result = advanced_multi_start(
            sphere,
            &bounds,
            Some(AdvancedMultiStartOptions {
                n_starts: 10,
                seed: Some(42),
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("Multi-start sphere failed");
        assert!(res.fun < 0.1, "Multi-start sphere value: {}", res.fun);
        assert!(res.n_successful > 0);
    }

    #[test]
    fn test_advanced_multi_start_rosenbrock() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let result = advanced_multi_start(
            rosenbrock,
            &bounds,
            Some(AdvancedMultiStartOptions {
                n_starts: 15,
                seed: Some(123),
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("Multi-start Rosenbrock failed");
        assert!(res.fun < 1.0, "Multi-start Rosenbrock: {}", res.fun);
    }

    #[test]
    fn test_monotonic_basin_hopping_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let result = monotonic_basin_hopping(
            sphere,
            &[3.0, -2.0],
            &bounds,
            Some(MonotonicBasinHoppingOptions {
                n_iterations: 30,
                seed: Some(42),
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("Monotonic BH sphere failed");
        assert!(res.fun < 0.1, "Monotonic BH sphere: {}", res.fun);
    }

    #[test]
    fn test_monotonic_basin_hopping_rastrigin() {
        let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
        let result = monotonic_basin_hopping(
            rastrigin,
            &[2.0, -3.0],
            &bounds,
            Some(MonotonicBasinHoppingOptions {
                n_iterations: 50,
                step_size: 1.0,
                seed: Some(99),
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("Monotonic BH Rastrigin failed");
        assert!(res.fun < 20.0, "Monotonic BH Rastrigin: {}", res.fun);
    }

    #[test]
    fn test_stochastic_tunneling_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let result = stochastic_tunneling(
            sphere,
            &[3.0, -2.0],
            &bounds,
            Some(StochasticTunnelingOptions {
                n_iterations: 200,
                gamma: 1.0,
                step_size: 0.5,
                local_every: 50,
                seed: Some(42),
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("Stochastic tunneling sphere failed");
        assert!(res.fun < 1.0, "Stochastic tunneling sphere: {}", res.fun);
    }

    #[test]
    fn test_stochastic_tunneling_rastrigin() {
        let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
        let result = stochastic_tunneling(
            rastrigin,
            &[2.0, -3.0],
            &bounds,
            Some(StochasticTunnelingOptions {
                n_iterations: 500,
                gamma: 0.5,
                step_size: 0.5,
                local_every: 50,
                seed: Some(42),
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("Stochastic tunneling Rastrigin failed");
        assert!(
            res.fun < 20.0,
            "Stochastic tunneling Rastrigin: {}",
            res.fun
        );
    }

    #[test]
    fn test_deflation_two_minima() {
        let bounds = vec![(-5.0, 5.0)];
        let result = deflation_search(
            two_minima,
            &bounds,
            Some(DeflationOptions {
                max_optima: 3,
                deflation_radius: 0.5,
                n_starts: 10,
                seed: Some(42),
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("Deflation two_minima failed");
        assert!(!res.optima.is_empty(), "Should find at least one optimum");
    }

    #[test]
    fn test_deflation_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let result = deflation_search(
            sphere,
            &bounds,
            Some(DeflationOptions {
                max_optima: 3,
                deflation_radius: 1.0,
                n_starts: 5,
                seed: Some(42),
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("Deflation sphere failed");
        assert!(res.success);
        // Sphere has only one global minimum, so we should find it
        assert!(
            res.optima[0].1 < 1.0,
            "Best optimum value: {}",
            res.optima[0].1
        );
    }

    #[test]
    fn test_lhs_generation() {
        let bounds = vec![(-1.0, 1.0), (0.0, 10.0)];
        let mut rng = StdRng::seed_from_u64(42);
        let points = generate_lhs_points(2, 10, &bounds, &mut rng);
        assert_eq!(points.len(), 10);
        for p in &points {
            assert!(p[0] >= -1.0 && p[0] <= 1.0);
            assert!(p[1] >= 0.0 && p[1] <= 10.0);
        }
    }

    #[test]
    fn test_merge_optima() {
        let optima = vec![
            (Array1::from_vec(vec![1.0, 1.0]), 0.5),
            (Array1::from_vec(vec![1.001, 1.001]), 0.4), // close to first
            (Array1::from_vec(vec![5.0, 5.0]), 1.0),     // far from first
        ];
        let merged = merge_optima(&optima, 0.01);
        assert_eq!(merged.len(), 2);
        // The better of the two close ones should be kept
        assert!((merged[0].1 - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_advanced_multi_start_empty_bounds() {
        let result = advanced_multi_start(sphere, &vec![], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_monotonic_bh_adaptive_step() {
        let bounds = vec![(-5.0, 5.0)];
        let result = monotonic_basin_hopping(
            |x: &ArrayView1<f64>| x[0] * x[0],
            &[4.0],
            &bounds,
            Some(MonotonicBasinHoppingOptions {
                n_iterations: 20,
                step_size: 0.5,
                step_increase: 1.2,
                step_decrease: 0.8,
                seed: Some(42),
                ..Default::default()
            }),
        );
        assert!(result.is_ok());
        let res = result.expect("Adaptive step MBH failed");
        assert!(res.fun < 1.0, "Adaptive step MBH value: {}", res.fun);
    }
}
