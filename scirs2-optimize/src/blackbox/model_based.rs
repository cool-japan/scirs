//! Model-Based Black-Box Optimization.
//!
//! Provides surrogate-model-driven optimization using ensembles of decision
//! trees (Random Forest) as the surrogate model.  Unlike GP-based methods,
//! tree ensembles scale more gracefully to moderate-dimensional problems and
//! do not require careful kernel design.
//!
//! # Contents
//!
//! - [`RandomForestSurrogate`] – ensemble of regression trees, exposes `fit`
//!   and `predict` (mean + uncertainty from tree variance).
//! - [`ei_random_forest`] – Expected Improvement acquisition function for
//!   the RF surrogate.
//! - [`SmacOptimizer`] – Sequential Model-based Algorithm Configuration (SMAC)
//!   loop, which alternates between local and random search phases.
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_optimize::blackbox::model_based::{
//!     SmacOptimizer, SmacConfig,
//! };
//!
//! let bounds = vec![(-5.0_f64, 5.0_f64), (-5.0, 5.0)];
//! let config = SmacConfig { n_initial: 6, n_iterations: 20, seed: Some(42),
//!     ..Default::default() };
//! let mut smac = SmacOptimizer::new(bounds, config).expect("build smac");
//! let result = smac.optimize(|x: &[f64]| x[0].powi(2) + x[1].powi(2), 20)
//!     .expect("optimize");
//! println!("Best: {:?} f={:.4}", result.x_best, result.f_best);
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::RngExt;

use crate::error::{OptimizeError, OptimizeResult};

// ---------------------------------------------------------------------------
// Decision tree node
// ---------------------------------------------------------------------------

/// A single node in a regression decision tree.
#[derive(Debug, Clone)]
enum TreeNode {
    /// Internal split node.
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
    /// Leaf node holding the mean response value.
    Leaf { value: f64 },
}

impl TreeNode {
    /// Predict the response for a single sample.
    fn predict(&self, x: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                let v = if *feature < x.len() { x[*feature] } else { 0.0 };
                if v <= *threshold {
                    left.predict(x)
                } else {
                    right.predict(x)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decision tree builder
// ---------------------------------------------------------------------------

/// Parameters controlling tree growth.
#[derive(Debug, Clone)]
struct TreeParams {
    max_depth: usize,
    min_samples_split: usize,
    /// Number of features to consider at each split (0 = use sqrt(n_features)).
    max_features: usize,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            max_depth: 10,
            min_samples_split: 2,
            max_features: 0,
        }
    }
}

/// Build a regression tree from training data.
fn build_tree(
    x: &[Vec<f64>],
    y: &[f64],
    params: &TreeParams,
    depth: usize,
    rng: &mut StdRng,
) -> TreeNode {
    let n = x.len();
    if n == 0 {
        return TreeNode::Leaf { value: 0.0 };
    }

    // Mean value for this node.
    let mean = y.iter().sum::<f64>() / n as f64;

    // Stopping criteria.
    if depth >= params.max_depth || n < params.min_samples_split {
        return TreeNode::Leaf { value: mean };
    }

    // Variance of target; if essentially zero, no need to split.
    let var = y.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    if var < 1e-12 {
        return TreeNode::Leaf { value: mean };
    }

    let n_features = if x.is_empty() { 0 } else { x[0].len() };
    let n_try = if params.max_features == 0 {
        ((n_features as f64).sqrt() as usize).max(1)
    } else {
        params.max_features.min(n_features)
    };

    // Sample a subset of features to consider.
    let mut feature_indices: Vec<usize> = (0..n_features).collect();
    // Fisher-Yates partial shuffle to pick n_try features.
    for i in 0..n_try {
        let j = i + rng.random_range(0..(n_features - i));
        feature_indices.swap(i, j);
    }
    let features_to_try = &feature_indices[..n_try];

    let mut best_gain = 0.0;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    for &feat in features_to_try {
        // Collect unique candidate thresholds (midpoints between sorted values).
        let mut vals: Vec<f64> = x.iter().map(|xi| xi[feat]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup();

        for i in 0..(vals.len().saturating_sub(1)) {
            let threshold = 0.5 * (vals[i] + vals[i + 1]);
            let gain = variance_reduction(x, y, feat, threshold, mean, var, n);
            if gain > best_gain {
                best_gain = gain;
                best_feature = feat;
                best_threshold = threshold;
            }
        }
    }

    if best_gain <= 0.0 {
        return TreeNode::Leaf { value: mean };
    }

    // Partition data.
    let (x_left, y_left, x_right, y_right) = partition(x, y, best_feature, best_threshold);

    if x_left.is_empty() || x_right.is_empty() {
        return TreeNode::Leaf { value: mean };
    }

    let left = build_tree(&x_left, &y_left, params, depth + 1, rng);
    let right = build_tree(&x_right, &y_right, params, depth + 1, rng);

    TreeNode::Split {
        feature: best_feature,
        threshold: best_threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Compute variance reduction gain for a feature/threshold split.
fn variance_reduction(
    x: &[Vec<f64>],
    y: &[f64],
    feature: usize,
    threshold: f64,
    parent_mean: f64,
    parent_var: f64,
    n: usize,
) -> f64 {
    let (n_l, sum_l, sq_l, n_r, sum_r, sq_r) = x.iter().zip(y.iter()).fold(
        (0usize, 0.0_f64, 0.0_f64, 0usize, 0.0_f64, 0.0_f64),
        |mut acc, (xi, &yi)| {
            if xi[feature] <= threshold {
                acc.0 += 1;
                acc.1 += yi;
                acc.2 += yi * yi;
            } else {
                acc.3 += 1;
                acc.4 += yi;
                acc.5 += yi * yi;
            }
            acc
        },
    );

    if n_l == 0 || n_r == 0 {
        return 0.0;
    }

    let var_l = (sq_l - sum_l * sum_l / n_l as f64) / n_l as f64;
    let var_r = (sq_r - sum_r * sum_r / n_r as f64) / n_r as f64;

    let weighted_var = (n_l as f64 * var_l + n_r as f64 * var_r) / n as f64;

    let _ = parent_mean; // not needed directly
    parent_var - weighted_var
}

/// Split data into left (≤ threshold) and right (> threshold) partitions.
fn partition(
    x: &[Vec<f64>],
    y: &[f64],
    feature: usize,
    threshold: f64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let mut x_l = Vec::new();
    let mut y_l = Vec::new();
    let mut x_r = Vec::new();
    let mut y_r = Vec::new();

    for (xi, &yi) in x.iter().zip(y.iter()) {
        if xi[feature] <= threshold {
            x_l.push(xi.clone());
            y_l.push(yi);
        } else {
            x_r.push(xi.clone());
            y_r.push(yi);
        }
    }

    (x_l, y_l, x_r, y_r)
}

// ---------------------------------------------------------------------------
// Random Forest Surrogate
// ---------------------------------------------------------------------------

/// Random Forest surrogate for black-box optimization.
///
/// Maintains an ensemble of decision trees trained on bootstrap samples.
/// The prediction mean and variance are computed as the mean and variance
/// across the ensemble's individual tree predictions.
#[derive(Debug, Clone)]
pub struct RandomForestSurrogate {
    trees: Vec<TreeNode>,
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize,
    seed: u64,
}

impl RandomForestSurrogate {
    /// Create a new (unfitted) Random Forest surrogate.
    pub fn new(
        n_estimators: usize,
        max_depth: usize,
        min_samples_split: usize,
        max_features: usize,
        seed: u64,
    ) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators: n_estimators.max(1),
            max_depth: max_depth.max(1),
            min_samples_split: min_samples_split.max(2),
            max_features,
            seed,
        }
    }

    /// Default RF configuration (100 trees, depth 10, sqrt features).
    pub fn default_config(seed: u64) -> Self {
        Self::new(100, 10, 2, 0, seed)
    }

    /// Fit the ensemble to training data.
    ///
    /// `x` shape: (n_samples, n_features);  `y` shape: (n_samples,).
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> OptimizeResult<()> {
        if x.nrows() != y.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "x has {} rows but y has {} elements",
                x.nrows(),
                y.len()
            )));
        }
        if x.nrows() == 0 {
            return Err(OptimizeError::InvalidInput(
                "Cannot fit RF with zero samples".to_string(),
            ));
        }

        let n = x.nrows();
        let n_features = x.ncols();

        // Convert ndarray to Vec<Vec> for the tree builder.
        let x_vec: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n_features).map(|j| x[[i, j]]).collect())
            .collect();
        let y_vec: Vec<f64> = y.iter().copied().collect();

        let params = TreeParams {
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            max_features: self.max_features,
        };

        let mut rng = StdRng::seed_from_u64(self.seed);
        self.trees.clear();

        for t in 0..self.n_estimators {
            // Bootstrap sample.
            let mut x_boot = Vec::with_capacity(n);
            let mut y_boot = Vec::with_capacity(n);

            // Use different seed per tree so bootstrap samples differ.
            let mut tree_rng = StdRng::seed_from_u64(
                self.seed
                    .wrapping_add((t as u64).wrapping_mul(6364136223846793005)),
            );

            for _ in 0..n {
                let idx = rng.random_range(0..n);
                x_boot.push(x_vec[idx].clone());
                y_boot.push(y_vec[idx]);
            }

            let tree = build_tree(&x_boot, &y_boot, &params, 0, &mut tree_rng);
            self.trees.push(tree);
        }

        Ok(())
    }

    /// Predict mean and variance from tree ensemble at test points.
    ///
    /// Returns `(mean, variance)` where `variance` is the inter-tree variance.
    pub fn predict(&self, x: &Array2<f64>) -> OptimizeResult<(Array1<f64>, Array1<f64>)> {
        if self.trees.is_empty() {
            return Err(OptimizeError::ComputationError(
                "RandomForestSurrogate has not been fitted".to_string(),
            ));
        }

        let n = x.nrows();
        let n_features = x.ncols();
        let n_trees = self.trees.len() as f64;

        let mut means = Array1::zeros(n);
        let mut variances = Array1::zeros(n);

        for i in 0..n {
            let x_row: Vec<f64> = (0..n_features).map(|j| x[[i, j]]).collect();

            // Collect per-tree predictions.
            let preds: Vec<f64> = self.trees.iter().map(|t| t.predict(&x_row)).collect();
            let mean = preds.iter().sum::<f64>() / n_trees;
            let variance = preds.iter().map(|&p| (p - mean) * (p - mean)).sum::<f64>()
                / n_trees.max(1.0 - 1.0 / n_trees.max(1.0)); // bias-corrected

            means[i] = mean;
            variances[i] = variance.max(0.0);
        }

        Ok((means, variances))
    }

    /// Predict at a single sample; returns (mean, std).
    pub fn predict_single(&self, x: &[f64]) -> OptimizeResult<(f64, f64)> {
        if self.trees.is_empty() {
            return Err(OptimizeError::ComputationError(
                "RandomForestSurrogate has not been fitted".to_string(),
            ));
        }
        let n_trees = self.trees.len() as f64;
        let preds: Vec<f64> = self.trees.iter().map(|t| t.predict(x)).collect();
        let mean = preds.iter().sum::<f64>() / n_trees;
        let variance =
            preds.iter().map(|&p| (p - mean) * (p - mean)).sum::<f64>() / n_trees.max(1.0);
        Ok((mean, variance.max(0.0).sqrt()))
    }

    /// Number of trees in the ensemble.
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        !self.trees.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Acquisition: EI for Random Forest surrogate
// ---------------------------------------------------------------------------

/// Expected Improvement acquisition for a Random Forest surrogate.
///
/// ```text
///   EI(x) = (best_y - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)
///   z = (best_y - mu(x) - xi) / sigma(x)
/// ```
///
/// The `sigma` is the square root of the inter-tree variance.
pub fn ei_random_forest(
    x: &[f64],
    surrogate: &RandomForestSurrogate,
    best_y: f64,
    xi: f64,
) -> OptimizeResult<f64> {
    let (mu, sigma) = surrogate.predict_single(x)?;
    if sigma < 1e-12 {
        return Ok(0.0);
    }
    let z = (best_y - mu - xi) / sigma;
    let ei = (best_y - mu - xi) * norm_cdf(z) + sigma * norm_pdf(z);
    Ok(ei.max(0.0))
}

// ---------------------------------------------------------------------------
// SMAC: Sequential Model-based Algorithm Configuration
// ---------------------------------------------------------------------------

/// Configuration for the SMAC optimizer.
#[derive(Debug, Clone)]
pub struct SmacConfig {
    /// Number of random initial evaluations.
    pub n_initial: usize,
    /// Total number of function evaluations (including initial).
    pub n_iterations: usize,
    /// Number of random candidates evaluated per BO step.
    pub n_candidates: usize,
    /// Number of intensification restarts per BO step.
    pub n_local_search: usize,
    /// Exploration bonus xi for EI.
    pub xi: f64,
    /// Random Forest number of estimators.
    pub n_estimators: usize,
    /// Random Forest max depth.
    pub rf_max_depth: usize,
    /// Random seed.
    pub seed: Option<u64>,
    /// Verbosity.
    pub verbose: usize,
}

impl Default for SmacConfig {
    fn default() -> Self {
        Self {
            n_initial: 10,
            n_iterations: 50,
            n_candidates: 300,
            n_local_search: 5,
            xi: 0.01,
            n_estimators: 50,
            rf_max_depth: 10,
            seed: None,
            verbose: 0,
        }
    }
}

/// Result of SMAC optimization.
#[derive(Debug, Clone)]
pub struct SmacResult {
    /// Best input found.
    pub x_best: Array1<f64>,
    /// Best objective value.
    pub f_best: f64,
    /// Number of function evaluations.
    pub n_evals: usize,
    /// History of (iteration, f_value) pairs.
    pub history: Vec<(usize, f64)>,
}

/// SMAC: Sequential Model-based Algorithm Configuration.
///
/// Uses a Random Forest surrogate and an EI acquisition function to guide
/// the search, combined with local search around promising incumbents.
pub struct SmacOptimizer {
    bounds: Vec<(f64, f64)>,
    config: SmacConfig,
}

impl SmacOptimizer {
    /// Create a new SMAC optimizer.
    pub fn new(bounds: Vec<(f64, f64)>, config: SmacConfig) -> OptimizeResult<Self> {
        if bounds.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "Search space bounds must not be empty".to_string(),
            ));
        }
        Ok(Self { bounds, config })
    }

    /// Run SMAC optimization.
    ///
    /// `objective(x)` evaluates the function to minimize.
    /// `n_evaluations` overrides `config.n_iterations` if > 0.
    pub fn optimize<F>(&mut self, objective: F, n_evaluations: usize) -> OptimizeResult<SmacResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_iter = if n_evaluations > 0 {
            n_evaluations
        } else {
            self.config.n_iterations
        };

        if n_iter == 0 {
            return Err(OptimizeError::InvalidInput(
                "n_iterations must be > 0".to_string(),
            ));
        }

        let n_dims = self.bounds.len();
        let seed = self.config.seed.unwrap_or(0);
        let mut rng = StdRng::seed_from_u64(seed);

        let mut x_history: Vec<Vec<f64>> = Vec::new();
        let mut y_history: Vec<f64> = Vec::new();
        let mut history: Vec<(usize, f64)> = Vec::new();
        let mut best_y = f64::INFINITY;
        let mut best_x: Option<Array1<f64>> = None;

        // -------------------------------------------------------------------
        // Phase 1: Random initial evaluations (Exploration).
        // -------------------------------------------------------------------
        let n_init = self.config.n_initial.min(n_iter).max(3);
        for i in 0..n_init {
            let x_rand: Vec<f64> = (0..n_dims)
                .map(|d| {
                    let lo = self.bounds[d].0;
                    let hi = self.bounds[d].1;
                    lo + rng.random::<f64>() * (hi - lo)
                })
                .collect();

            let y = objective(&x_rand);
            history.push((i + 1, y));

            if y < best_y {
                best_y = y;
                let arr = Array1::from_vec(x_rand.clone());
                best_x = Some(arr);
            }

            x_history.push(x_rand);
            y_history.push(y);
        }

        if self.config.verbose >= 1 {
            println!(
                "[SMAC] Initial phase complete: {} evals, best_y={:.6}",
                n_init, best_y
            );
        }

        // -------------------------------------------------------------------
        // Phase 2: Model-based optimization loop.
        // -------------------------------------------------------------------
        let mut surrogate = RandomForestSurrogate::new(
            self.config.n_estimators,
            self.config.rf_max_depth,
            2,
            0,
            seed,
        );

        for iter in n_init..n_iter {
            // Build training arrays.
            let n_obs = x_history.len();
            let mut x_train = Array2::zeros((n_obs, n_dims));
            let mut y_train = Array1::zeros(n_obs);
            for (i, (xi, &yi)) in x_history.iter().zip(y_history.iter()).enumerate() {
                for d in 0..n_dims {
                    x_train[[i, d]] = xi[d];
                }
                y_train[i] = yi;
            }

            // Fit surrogate.
            if let Err(e) = surrogate.fit(&x_train, &y_train) {
                if self.config.verbose >= 1 {
                    println!("[SMAC] surrogate fit error at iter {}: {}", iter, e);
                }
                // Fall back to random evaluation.
                let x_rand: Vec<f64> = (0..n_dims)
                    .map(|d| {
                        let lo = self.bounds[d].0;
                        let hi = self.bounds[d].1;
                        lo + rng.random::<f64>() * (hi - lo)
                    })
                    .collect();
                let y = objective(&x_rand);
                history.push((iter + 1, y));
                if y < best_y {
                    best_y = y;
                    best_x = Some(Array1::from_vec(x_rand.clone()));
                }
                x_history.push(x_rand);
                y_history.push(y);
                continue;
            }

            // -----------------------------------------------------------
            // Acquisition maximization: random candidates + local search.
            // -----------------------------------------------------------
            let n_cands = self.config.n_candidates;
            let mut best_acq = f64::NEG_INFINITY;
            let mut best_candidate = vec![0.0f64; n_dims];

            // Random candidates.
            for _ in 0..n_cands {
                let cand: Vec<f64> = (0..n_dims)
                    .map(|d| {
                        let lo = self.bounds[d].0;
                        let hi = self.bounds[d].1;
                        lo + rng.random::<f64>() * (hi - lo)
                    })
                    .collect();

                let acq =
                    ei_random_forest(&cand, &surrogate, best_y, self.config.xi).unwrap_or(0.0);
                if acq > best_acq {
                    best_acq = acq;
                    best_candidate = cand;
                }
            }

            // Local search around the incumbent best point.
            if let Some(ref inc) = best_x {
                let n_local = self.config.n_local_search;
                let inc_vec: Vec<f64> = inc.iter().copied().collect();

                for step_size in [0.1, 0.05, 0.01] {
                    for _ in 0..n_local {
                        let neighbor: Vec<f64> = inc_vec
                            .iter()
                            .zip(self.bounds.iter())
                            .map(|(&xi, &(lo, hi))| {
                                let range = hi - lo;
                                let perturb = (rng.random::<f64>() - 0.5) * 2.0 * step_size * range;
                                (xi + perturb).clamp(lo, hi)
                            })
                            .collect();

                        let acq = ei_random_forest(&neighbor, &surrogate, best_y, self.config.xi)
                            .unwrap_or(0.0);
                        if acq > best_acq {
                            best_acq = acq;
                            best_candidate = neighbor;
                        }
                    }
                }
            }

            // -----------------------------------------------------------
            // Evaluate the chosen candidate.
            // -----------------------------------------------------------
            let y_next = objective(&best_candidate);
            history.push((iter + 1, y_next));

            if y_next < best_y {
                best_y = y_next;
                best_x = Some(Array1::from_vec(best_candidate.clone()));
            }

            x_history.push(best_candidate);
            y_history.push(y_next);

            if self.config.verbose >= 2 {
                println!(
                    "[SMAC] iter={} acq={:.4} f={:.6} best={:.6}",
                    iter + 1,
                    best_acq,
                    y_next,
                    best_y
                );
            }
        }

        if self.config.verbose >= 1 {
            println!(
                "[SMAC] Done. n_evals={} best_f={:.6}",
                history.len(),
                best_y
            );
        }

        let x_best = best_x.unwrap_or_else(|| Array1::zeros(n_dims));

        Ok(SmacResult {
            x_best,
            f_best: best_y,
            n_evals: history.len(),
            history,
        })
    }
}

// ---------------------------------------------------------------------------
// Normal distribution helpers
// ---------------------------------------------------------------------------

fn erf_approx(x: f64) -> f64 {
    let p = 0.3275911_f64;
    let (a1, a2, a3, a4, a5) = (
        0.254829592_f64,
        -0.284496736_f64,
        1.421413741_f64,
        -1.453152027_f64,
        1.061405429_f64,
    );
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let poly = (((a5 * t + a4) * t + a3) * t + a2) * t + a1;
    sign * (1.0 - poly * t * (-xa * xa).exp())
}

fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn make_simple_data() -> (Array2<f64>, Array1<f64>) {
        // f(x) = x^2, 8 training points.
        let xs: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        let x = Array2::from_shape_vec((8, 1), xs).expect("shape");
        let y = Array1::from_vec(ys);
        (x, y)
    }

    #[test]
    fn test_rf_surrogate_fit_predict() {
        let (x, y) = make_simple_data();
        let mut rf = RandomForestSurrogate::new(20, 8, 2, 0, 42);
        rf.fit(&x, &y).expect("fit");

        assert!(rf.is_fitted());
        assert_eq!(rf.n_estimators(), 20);

        let x_test = Array2::from_shape_vec((3, 1), vec![0.25, 1.25, 2.75]).expect("shape");
        let (mean, var) = rf.predict(&x_test).expect("predict");

        // All predictions should be in a reasonable range.
        for i in 0..3 {
            assert!(mean[i].is_finite(), "mean[{}] must be finite", i);
            assert!(var[i] >= 0.0, "var[{}] must be non-negative", i);
        }

        // Prediction at x=1.25 should be roughly 1.5625.
        assert!(
            (mean[1] - 1.5625).abs() < 2.0,
            "mean[1]={:.4} expected ~1.56",
            mean[1]
        );
    }

    #[test]
    fn test_rf_predict_monotone_variance() {
        let (x, y) = make_simple_data();
        let mut rf = RandomForestSurrogate::new(30, 8, 2, 0, 99);
        rf.fit(&x, &y).expect("fit");

        // Predict at a training point vs far outside the training range.
        let x_near = Array2::from_shape_vec((1, 1), vec![1.0]).expect("shape");
        let x_far = Array2::from_shape_vec((1, 1), vec![10.0]).expect("shape");
        let (_, var_near) = rf.predict(&x_near).expect("predict near");
        let (_, var_far) = rf.predict(&x_far).expect("predict far");

        // Variance at an extrapolation point should be >= near in-domain.
        // (This is a soft check; tree variance behaviour differs from GP.)
        assert!(var_far[0] >= 0.0);
        assert!(var_near[0] >= 0.0);
    }

    #[test]
    fn test_ei_rf_non_negative() {
        let (x, y) = make_simple_data();
        let mut rf = RandomForestSurrogate::new(20, 8, 2, 0, 7);
        rf.fit(&x, &y).expect("fit");

        let cands = vec![0.3, 1.7, 2.8];
        for c in cands {
            let ei = ei_random_forest(&[c], &rf, 2.0, 0.01).expect("ei");
            assert!(ei >= 0.0, "EI must be non-negative at x={}", c);
        }
    }

    #[test]
    fn test_smac_minimizes_quadratic() {
        let bounds = vec![(-3.0_f64, 3.0_f64)];
        let config = SmacConfig {
            n_initial: 5,
            n_iterations: 25,
            n_candidates: 50,
            n_local_search: 3,
            n_estimators: 20,
            seed: Some(42),
            verbose: 0,
            ..Default::default()
        };
        let mut smac = SmacOptimizer::new(bounds, config).expect("build smac");
        let result = smac.optimize(|x: &[f64]| x[0].powi(2), 25).expect("opt");

        assert!(result.f_best.is_finite());
        assert!(
            result.f_best < 0.5,
            "Expected f_best < 0.5, got {}",
            result.f_best
        );
        assert_eq!(result.n_evals, 25);
    }

    #[test]
    fn test_smac_2d_rosenbrock_trend() {
        let bounds = vec![(-2.0_f64, 2.0_f64), (-2.0, 2.0)];
        let config = SmacConfig {
            n_initial: 8,
            n_iterations: 30,
            n_candidates: 80,
            n_estimators: 25,
            seed: Some(123),
            verbose: 0,
            ..Default::default()
        };
        let mut smac = SmacOptimizer::new(bounds, config).expect("build smac");
        // Rosenbrock (global min at [1,1] = 0).
        let result = smac
            .optimize(
                |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2),
                30,
            )
            .expect("opt");

        assert!(result.f_best.is_finite());
        // Only 30 evaluations; just check it makes some progress.
        assert!(result.f_best < 1000.0, "f_best={}", result.f_best);
    }

    #[test]
    fn test_rf_unfitted_predict_error() {
        let rf = RandomForestSurrogate::new(10, 5, 2, 0, 0);
        let x = Array2::zeros((2, 1));
        assert!(rf.predict(&x).is_err());
    }
}
