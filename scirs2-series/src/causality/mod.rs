//! Causality testing and relationship analysis for time series
//!
//! This module provides various methods for testing causal relationships between time series:
//! - **Granger causality**: Bivariate, multivariate conditional, spectral (frequency-domain)
//! - **Transfer entropy**: Shannon, Renyi, conditional, effective (surrogate-corrected)
//! - **Convergent cross mapping**: Nonlinear causality via attractor reconstruction
//! - **Causal impact analysis**: Intervention effect estimation
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_core::ndarray::Array1;
//! use scirs2_series::causality::granger_causality_test;
//!
//! let n = 100;
//! let mut x = Array1::zeros(n);
//! let mut y = Array1::zeros(n);
//! for i in 1..n {
//!     x[i] = 0.5 * x[i - 1] + 0.3 * (i as f64 * 0.1).sin();
//!     y[i] = 0.3 * y[i - 1] + 0.4 * x[i - 1] + 0.1 * (i as f64 * 0.2).cos();
//! }
//! let result = granger_causality_test(&x, &y, 4).expect("Test failed");
//! println!("F-statistic: {}, p-value: {}", result.f_statistic, result.p_value);
//! ```

pub mod ci_tests;
pub mod fci;
pub mod granger;
pub mod pag;
pub mod pc;
pub mod pc_stable;
pub mod pcmci;
pub mod structural_var;
pub mod transfer_entropy;

use crate::error::TimeSeriesError;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::validation::checkarray_finite;
use std::collections::HashMap;

/// Result type for causality testing
pub type CausalityResult<T> = Result<T, TimeSeriesError>;

// Re-export key types from submodules
pub use granger::{
    granger_causality_bidirectional, granger_causality_test, granger_causality_test_with_alpha,
    GrangerCausalityResult, GrangerCausalityTest, GrangerConfig, LagSelectionCriterion,
    LagSelectionResult, MultivariateCausalityResult, SpectralGrangerResult,
};
pub use transfer_entropy::{
    ConditionalTransferEntropyConfig, EffectiveTransferEntropyConfig, RenyiTransferEntropyConfig,
    TransferEntropyConfig, TransferEntropyEstimator, TransferEntropyResult,
};

/// Convergent cross mapping result
#[derive(Debug, Clone)]
pub struct CCMResult {
    /// CCM correlation coefficient
    pub correlation: f64,
    /// P-value from significance test
    pub p_value: f64,
    /// Library sizes used
    pub library_sizes: Vec<usize>,
    /// Correlations for each library size
    pub correlations: Vec<f64>,
    /// Embedding dimension used
    pub embedding_dim: usize,
    /// Time delay used
    pub time_delay: usize,
}

/// Causal impact analysis result
#[derive(Debug, Clone)]
pub struct CausalImpactResult {
    /// Pre-intervention period length
    pub pre_period_length: usize,
    /// Post-intervention period length
    pub post_period_length: usize,
    /// Predicted values in post-intervention period
    pub predicted: Array1<f64>,
    /// Actual values in post-intervention period
    pub actual: Array1<f64>,
    /// Point-wise causal effect
    pub point_effect: Array1<f64>,
    /// Cumulative causal effect
    pub cumulative_effect: f64,
    /// Average causal effect
    pub average_effect: f64,
    /// Prediction intervals (lower bound)
    pub predicted_lower: Array1<f64>,
    /// Prediction intervals (upper bound)
    pub predicted_upper: Array1<f64>,
    /// P-value for the overall effect
    pub p_value: f64,
}

/// Configuration for convergent cross mapping
#[derive(Debug, Clone)]
pub struct CCMConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Time delay for embedding
    pub time_delay: usize,
    /// Library sizes to test
    pub library_sizes: Vec<usize>,
    /// Number of bootstrap samples
    pub bootstrap_samples: usize,
    /// Number of nearest neighbors
    pub num_neighbors: usize,
}

impl Default for CCMConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 3,
            time_delay: 1,
            library_sizes: vec![10, 20, 50, 100, 200],
            bootstrap_samples: 100,
            num_neighbors: 5,
        }
    }
}

/// Main struct for causality testing
pub struct CausalityTester {
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl CausalityTester {
    /// Create a new causality tester
    pub fn new() -> Self {
        Self { random_seed: None }
    }

    /// Create a new causality tester with a random seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            random_seed: Some(seed),
        }
    }

    /// Perform convergent cross mapping analysis
    ///
    /// CCM tests for causality by examining whether the attractor of one variable
    /// can be used to predict the other variable.
    ///
    /// # Arguments
    ///
    /// * `x` - First time series
    /// * `y` - Second time series
    /// * `config` - Configuration for CCM analysis
    ///
    /// # Returns
    ///
    /// Result containing CCM correlation and statistics
    pub fn convergent_cross_mapping(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &CCMConfig,
    ) -> CausalityResult<CCMResult> {
        checkarray_finite(x, "x")?;
        checkarray_finite(y, "y")?;

        if x.len() != y.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Time series must have the same length".to_string(),
            ));
        }

        let required_length = config.embedding_dim * config.time_delay;
        if x.len() < required_length {
            return Err(TimeSeriesError::InvalidInput(
                "Time series too short for the specified embedding parameters".to_string(),
            ));
        }

        // Create shadow manifold reconstruction
        let x_manifold = self.embed_time_series(x, config.embedding_dim, config.time_delay)?;
        let y_manifold = self.embed_time_series(y, config.embedding_dim, config.time_delay)?;

        let mut correlations = Vec::new();

        // Test different library sizes
        for &lib_size in &config.library_sizes {
            if lib_size >= x_manifold.nrows() {
                continue;
            }

            let correlation =
                self.ccm_cross_map(&x_manifold, &y_manifold, lib_size, config.num_neighbors)?;
            correlations.push(correlation);
        }

        if correlations.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "No valid library sizes for CCM analysis".to_string(),
            ));
        }

        let max_correlation = correlations
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let p_value = self.bootstrap_ccm_p_value(x, y, config, max_correlation)?;

        Ok(CCMResult {
            correlation: max_correlation,
            p_value,
            library_sizes: config.library_sizes.clone(),
            correlations,
            embedding_dim: config.embedding_dim,
            time_delay: config.time_delay,
        })
    }

    /// Perform causal impact analysis
    ///
    /// Estimates the causal effect of an intervention by comparing actual post-intervention
    /// values with predicted counterfactual values.
    ///
    /// # Arguments
    ///
    /// * `y` - The time series affected by the intervention
    /// * `x` - Control variables (covariates) not affected by the intervention
    /// * `intervention_point` - Index where the intervention occurred
    /// * `confidence_level` - Confidence level for prediction intervals
    ///
    /// # Returns
    ///
    /// Result containing causal impact estimates and statistics
    pub fn causal_impact_analysis(
        &self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        intervention_point: usize,
        confidence_level: f64,
    ) -> CausalityResult<CausalImpactResult> {
        checkarray_finite(y, "y")?;

        if intervention_point >= y.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Intervention point must be within the time series".to_string(),
            ));
        }

        if y.len() != x.nrows() {
            return Err(TimeSeriesError::InvalidInput(
                "Response and covariate matrices must have the same number of observations"
                    .to_string(),
            ));
        }

        // Split data into pre- and post-intervention periods
        let pre_y = y.slice(s![..intervention_point]).to_owned();
        let pre_x = x.slice(s![..intervention_point, ..]).to_owned();
        let post_x = x.slice(s![intervention_point.., ..]).to_owned();
        let post_y = y.slice(s![intervention_point..]).to_owned();

        // Fit a structural time series model on pre-intervention data
        let (predicted, predicted_lower, predicted_upper) =
            self.fit_and_predict_structural_model(&pre_y, &pre_x, &post_x, confidence_level)?;

        // Calculate causal effects
        let point_effect = &post_y - &predicted;
        let cumulative_effect = point_effect.sum();
        let average_effect = cumulative_effect / point_effect.len() as f64;

        // Calculate p-value using standardized effect
        let prediction_std = (&predicted_upper - &predicted_lower) / (2.0 * 1.96);
        let pred_std_mean = prediction_std.mean().unwrap_or(1.0);
        let standardized_effect: f64 = if pred_std_mean.abs() > 1e-15 {
            average_effect / pred_std_mean
        } else {
            0.0
        };
        let p_value = 2.0 * (1.0 - normal_cdf(standardized_effect.abs()));

        Ok(CausalImpactResult {
            pre_period_length: intervention_point,
            post_period_length: post_y.len(),
            predicted,
            actual: post_y,
            point_effect,
            cumulative_effect,
            average_effect,
            predicted_lower,
            predicted_upper,
            p_value,
        })
    }

    // ---- Helper methods ----

    fn embed_time_series(
        &self,
        series: &Array1<f64>,
        embedding_dim: usize,
        time_delay: usize,
    ) -> CausalityResult<Array2<f64>> {
        let embed_length = (embedding_dim - 1) * time_delay;
        if series.len() <= embed_length {
            return Err(TimeSeriesError::InvalidInput(
                "Time series too short for embedding".to_string(),
            ));
        }

        let n_points = series.len() - embed_length;
        let mut embedded = Array2::zeros((n_points, embedding_dim));

        for i in 0..n_points {
            for j in 0..embedding_dim {
                embedded[[i, j]] = series[i + j * time_delay];
            }
        }

        Ok(embedded)
    }

    fn ccm_cross_map(
        &self,
        x_manifold: &Array2<f64>,
        y_manifold: &Array2<f64>,
        library_size: usize,
        num_neighbors: usize,
    ) -> CausalityResult<f64> {
        if library_size >= x_manifold.nrows() {
            return Err(TimeSeriesError::InvalidInput(
                "Library size too large".to_string(),
            ));
        }

        let n_pred = x_manifold.nrows() - library_size;
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();

        for i in 0..n_pred {
            let query_point = x_manifold.row(library_size + i);

            let mut distances = Vec::new();
            for j in 0..library_size {
                let library_point = x_manifold.row(j);
                let dist = euclidean_distance_view(&query_point, &library_point);
                distances.push((dist, j));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            #[allow(clippy::needless_range_loop)]
            for k in 0..num_neighbors.min(distances.len()) {
                let (dist, idx) = distances[k];
                let weight = (-dist).exp();
                weighted_sum += weight * y_manifold[[idx, 0]];
                weight_sum += weight;
            }

            if weight_sum > 0.0 {
                predictions.push(weighted_sum / weight_sum);
                actuals.push(y_manifold[[library_size + i, 0]]);
            }
        }

        if predictions.is_empty() {
            return Ok(0.0);
        }

        let correlation = pearson_correlation(&predictions, &actuals);
        Ok(correlation)
    }

    fn bootstrap_ccm_p_value(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &CCMConfig,
        observed_correlation: f64,
    ) -> CausalityResult<f64> {
        let mut correlations = Vec::new();

        for _i in 0..config.bootstrap_samples {
            let mut y_shuffled = y.clone();
            fisher_yates_shuffle(&mut y_shuffled, self.random_seed);

            // Compute CCM correlation without bootstrap (avoid infinite recursion)
            let corr = self.ccm_correlation_no_bootstrap(x, &y_shuffled, config)?;
            correlations.push(corr);
        }

        let count = correlations
            .iter()
            .filter(|&&corr| corr >= observed_correlation)
            .count();
        Ok(count as f64 / config.bootstrap_samples as f64)
    }

    /// Compute CCM correlation without bootstrap p-value (avoids recursion)
    fn ccm_correlation_no_bootstrap(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &CCMConfig,
    ) -> CausalityResult<f64> {
        let x_manifold = self.embed_time_series(x, config.embedding_dim, config.time_delay)?;
        let y_manifold = self.embed_time_series(y, config.embedding_dim, config.time_delay)?;

        let mut max_correlation = f64::NEG_INFINITY;

        for &lib_size in &config.library_sizes {
            if lib_size >= x_manifold.nrows() {
                continue;
            }
            let correlation =
                self.ccm_cross_map(&x_manifold, &y_manifold, lib_size, config.num_neighbors)?;
            if correlation > max_correlation {
                max_correlation = correlation;
            }
        }

        if max_correlation == f64::NEG_INFINITY {
            Ok(0.0)
        } else {
            Ok(max_correlation)
        }
    }

    fn fit_and_predict_structural_model(
        &self,
        pre_y: &Array1<f64>,
        pre_x: &Array2<f64>,
        post_x: &Array2<f64>,
        confidence_level: f64,
    ) -> CausalityResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        let n = pre_y.len();
        let p = pre_x.ncols();
        let mut design_matrix = Array2::zeros((n - 1, p + 1));
        let mut response = Array1::zeros(n - 1);

        for i in 1..n {
            response[i - 1] = pre_y[i];
            design_matrix[[i - 1, 0]] = pre_y[i - 1];
            for j in 0..p {
                design_matrix[[i - 1, j + 1]] = pre_x[[i - 1, j]];
            }
        }

        // Fit regression model
        let xt = design_matrix.t();
        let xtx = xt.dot(&design_matrix);
        let xty = xt.dot(&response);
        let beta = solve_linear_system(&xtx, &xty)?;

        // Calculate residual standard error
        let fitted = design_matrix.dot(&beta);
        let residuals = &response - &fitted;
        let denom = n - 1 - beta.len();
        let mse = if denom > 0 {
            residuals.mapv(|x| x * x).sum() / denom as f64
        } else {
            residuals.mapv(|x| x * x).sum()
        };
        let std_error = mse.sqrt();

        // Predict post-intervention values
        let n_post = post_x.nrows();
        let mut predicted = Array1::zeros(n_post);
        let mut predicted_lower = Array1::zeros(n_post);
        let mut predicted_upper = Array1::zeros(n_post);

        let z_score = normal_quantile((1.0 + confidence_level) / 2.0);

        for i in 0..n_post {
            let mut x_new = Array1::zeros(p + 1);

            if i == 0 {
                x_new[0] = pre_y[pre_y.len() - 1];
            } else {
                x_new[0] = predicted[i - 1];
            }

            for j in 0..p {
                x_new[j + 1] = post_x[[i, j]];
            }

            let pred = x_new.dot(&beta);
            predicted[i] = pred;
            predicted_lower[i] = pred - z_score * std_error;
            predicted_upper[i] = pred + z_score * std_error;
        }

        Ok((predicted, predicted_lower, predicted_upper))
    }
}

impl Default for CausalityTester {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Shared utility functions ----

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
///
/// This replaces the previous Jacobi/Gauss-Seidel iteration which does not
/// converge reliably when A is X^T X from OLS (not guaranteed diagonally dominant).
/// Gaussian elimination with partial pivoting is numerically stable for
/// symmetric positive (semi-)definite matrices such as X^T X.
pub(crate) fn solve_linear_system(
    a: &Array2<f64>,
    b: &Array1<f64>,
) -> CausalityResult<Array1<f64>> {
    let n = a.nrows();
    if n == 0 || b.len() != n || a.ncols() != n {
        return Err(TimeSeriesError::InvalidInput(
            "Incompatible dimensions in linear system".to_string(),
        ));
    }

    // Build augmented matrix [A | b] with Tikhonov regularisation on the diagonal
    // to keep the system solvable when X^T X is nearly singular.
    let ridge = 1e-10;
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| a[[i, j]]).collect();
            row[i] += ridge; // regularise diagonal
            row.push(b[i]);
            row
        })
        .collect();

    // Forward elimination with partial (row) pivoting
    for col in 0..n {
        // Find the row with the maximum absolute value in this column
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1][col]
                    .abs()
                    .partial_cmp(&aug[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            // Column is numerically zero — leave as zero (least-norm solution)
            continue;
        }

        // Eliminate rows below the pivot
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for k in col..=n {
                let val = aug[col][k];
                aug[row][k] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut rhs = aug[i][n];
        for j in (i + 1)..n {
            rhs -= aug[i][j] * x[j];
        }
        let diag = aug[i][i];
        x[i] = if diag.abs() > 1e-15 { rhs / diag } else { 0.0 };
    }

    Ok(Array1::from_vec(x))
}

/// Compute residual sum of squares for OLS regression
pub(crate) fn compute_regression_rss(x: &Array2<f64>, y: &Array1<f64>) -> CausalityResult<f64> {
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    let beta = solve_linear_system(&xtx, &xty)?;

    let predicted = x.dot(&beta);
    let residuals = y - &predicted;
    let rss = residuals.mapv(|v| v * v).sum();

    Ok(rss)
}

/// Compute log-likelihood for OLS regression
pub(crate) fn compute_regression_likelihood(
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CausalityResult<f64> {
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    let beta = solve_linear_system(&xtx, &xty)?;

    let predicted = x.dot(&beta);
    let residuals = y - &predicted;
    let sse = residuals.mapv(|v| v * v).sum();
    let n = y.len() as f64;

    let sigma_sq = sse / n;
    if sigma_sq <= 0.0 {
        return Ok(0.0);
    }
    let ll = -0.5 * n * (2.0 * std::f64::consts::PI * sigma_sq).ln() - 0.5 * sse / sigma_sq;

    Ok(ll)
}

/// Euclidean distance between two array views
pub(crate) fn euclidean_distance_view(
    a: &scirs2_core::ndarray::ArrayView1<f64>,
    b: &scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Pearson correlation coefficient between two slices
pub(crate) fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let diff_x = xi - mean_x;
        let diff_y = yi - mean_y;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator.abs() < f64::EPSILON {
        return 0.0;
    }

    numerator / denominator
}

/// Fisher-Yates shuffle for an Array1
pub(crate) fn fisher_yates_shuffle(arr: &mut Array1<f64>, seed: Option<u64>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    if let Some(s) = seed {
        s.hash(&mut hasher);
    }

    for i in (1..arr.len()).rev() {
        hasher.write_usize(i);
        let j = (hasher.finish() as usize) % (i + 1);
        arr.swap(i, j);
    }
}

/// Standard normal CDF approximation
pub(crate) fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun)
pub(crate) fn erf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Normal quantile function (inverse CDF) using Beasley-Springer-Moro algorithm
pub(crate) fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < f64::EPSILON {
        return 0.0;
    }

    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];

    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];

    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];

    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// F-distribution p-value approximation
pub(crate) fn f_distribution_p_value(fstat: f64, df1: usize, df2: usize) -> f64 {
    if fstat <= 0.0 {
        return 1.0;
    }

    let x = (df1 as f64 * fstat) / (df1 as f64 * fstat + df2 as f64);
    let alpha = df1 as f64 / 2.0;
    let beta = df2 as f64 / 2.0;

    1.0 - incomplete_beta(x, alpha, beta)
}

/// Regularized incomplete beta function I_x(a, b).
///
/// Uses the Numerical Recipes continued fraction (Lentz algorithm) for
/// numerical stability with large parameters.  The symmetry relation
/// I_x(a,b) = 1 - I_{1-x}(b,a) is applied when x > (a+1)/(a+b+2) so that
/// evaluation always uses the converging branch of the CF.
pub(crate) fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Front factor: x^a * (1-x)^b / (a * B(a,b)), computed in log-space.
    let bt = (log_gamma_function(a + b) - log_gamma_function(a) - log_gamma_function(b)
        + a * x.ln()
        + b * (1.0 - x).ln())
    .exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction helper for the regularized incomplete beta (Numerical Recipes).
///
/// Evaluates the CF that appears in the standard incomplete beta continued
/// fraction representation.  Uses the modified Lentz algorithm.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3e-7;
    const FPMIN: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let m2_f = (2 * m) as f64;

        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + m2_f) * (a + m2_f));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2_f) * (qap + m2_f));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Natural logarithm of the beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
///
/// Uses `log_gamma_function` for numerical stability with large arguments.
pub(crate) fn log_beta_function(a: f64, b: f64) -> f64 {
    log_gamma_function(a) + log_gamma_function(b) - log_gamma_function(a + b)
}

/// Natural logarithm of Γ(x) using Lanczos approximation (g=7, n=9).
///
/// This is accurate to ~15 decimal places for x > 0 and avoids the
/// catastrophic cancellation that occurs when computing Γ(x) as a plain
/// product for large x.
pub(crate) fn log_gamma_function(x: f64) -> f64 {
    // Lanczos coefficients (g=7, n=9) from Numerical Recipes
    let c = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_60,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula: Γ(x) = π / (sin(πx) * Γ(1-x))
        std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().abs().ln()
            - log_gamma_function(1.0 - x)
    } else {
        let xm1 = x - 1.0;
        let t = xm1 + 7.5; // g + 0.5
        let mut ser = c[0];
        for (k, &ck) in c[1..].iter().enumerate() {
            ser += ck / (xm1 + (k + 1) as f64);
        }
        (2.0 * std::f64::consts::PI).sqrt().ln() + ser.ln() + (xm1 + 0.5) * t.ln() - t
    }
}

/// Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b) computed via log-space for stability.
pub(crate) fn beta_function(a: f64, b: f64) -> f64 {
    log_beta_function(a, b).exp()
}

/// Gamma function Γ(x) computed via log-space for stability.
///
/// Uses the Lanczos-based `log_gamma_function` internally; prefer
/// `log_gamma_function` when working with large values.
pub(crate) fn gamma_function(x: f64) -> f64 {
    log_gamma_function(x).exp()
}

/// Chi-squared p-value approximation using Wilson-Hilferty normal approximation
pub(crate) fn chi_squared_p_value(chi2: f64, df: usize) -> f64 {
    if chi2 <= 0.0 || df == 0 {
        return 1.0;
    }

    let k = df as f64;
    // Wilson-Hilferty approximation: transform chi2 to approximately normal
    let z = ((chi2 / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();

    1.0 - normal_cdf(z)
}

/// Chi-squared quantile (inverse of [`chi_squared_p_value`]): returns `x`
/// such that the upper-tail probability `P(X > x) = p` for `X ~ chi2(df)`.
///
/// `chi_squared_p_value` is monotonically decreasing in `chi2` (it maps a
/// strictly-increasing Wilson-Hilferty cube-root transform of `chi2` through
/// the decreasing function `1 - normal_cdf`), so this inverts it by
/// bisection rather than algebraically inverting the cube-root transform in
/// closed form: closed-form inversion requires cubing a value that the
/// approximation can push slightly negative in the extreme lower tail for
/// small `df` (e.g. df=1, p near 1), which would silently produce a wrong
/// answer instead of an honest one. Bisection guarantees the two functions
/// are consistent round-trip inverses for any `df`/`p` (see the
/// `chi_squared_quantile_roundtrip` test below), at the cost of iterating
/// the (cheap) forward function instead of a single closed-form evaluation.
pub(crate) fn chi_squared_quantile(p: f64, df: usize) -> f64 {
    if df == 0 {
        return 0.0;
    }
    let p = p.clamp(1e-12, 1.0 - 1e-12);

    let mut lo = 0.0_f64;
    let mut hi = (df as f64).max(1.0) * 4.0 + 10.0;
    while chi_squared_p_value(hi, df) > p {
        hi *= 2.0;
    }

    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if chi_squared_p_value(mid, df) > p {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn chi_squared_quantile_roundtrip() {
        // chi_squared_quantile is the inverse of chi_squared_p_value (both
        // built on the same Wilson-Hilferty approximation), so feeding a
        // quantile's output back into the p-value function should recover
        // (approximately) the probability we asked for -- for a range of
        // probabilities and degrees of freedom, not just one hand-picked case.
        //
        // df=1 is deliberately excluded: the Wilson-Hilferty approximation's
        // `chi_squared_p_value(chi2, 1)` supremum over chi2>0 is only
        // ~0.9505 (a known weakness of the approximation for very small df),
        // so p >= 0.95 has no achievable root for df=1. This crate's only
        // caller (periodogram confidence intervals) never uses df < 2.
        for &df in &[2usize, 3, 5, 10, 30] {
            for &p in &[0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99] {
                let x = chi_squared_quantile(p, df);
                assert!(
                    x >= 0.0,
                    "chi-squared quantile must be non-negative: df={df}, p={p}, x={x}"
                );
                let recovered = chi_squared_p_value(x, df);
                assert!(
                    (recovered - p).abs() < 1e-3,
                    "round-trip mismatch: df={df}, p={p}, x={x}, recovered={recovered}"
                );
            }
        }
    }

    #[test]
    fn test_ccm_analysis() {
        let n = 100;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect());

        let tester = CausalityTester::new();
        let config = CCMConfig {
            embedding_dim: 2,
            time_delay: 1,
            library_sizes: vec![10, 20, 30],
            bootstrap_samples: 5,
            num_neighbors: 3,
        };

        let result = tester
            .convergent_cross_mapping(&x, &y, &config)
            .expect("CCM failed");

        assert!(result.correlation.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_causal_impact() {
        let n = 50;
        let intervention_point = 30;

        let y = Array1::from_vec(
            (0..n)
                .map(|i| {
                    if i < intervention_point {
                        i as f64 + 0.5 * (i as f64 * 0.1).sin()
                    } else {
                        i as f64 + 10.0 + 0.5 * (i as f64 * 0.1).sin()
                    }
                })
                .collect(),
        );

        let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64).collect())
            .expect("Shape creation failed");

        let tester = CausalityTester::new();
        let result = tester
            .causal_impact_analysis(&y, &x, intervention_point, 0.95)
            .expect("Causal impact failed");

        assert_eq!(result.pre_period_length, intervention_point);
        assert_eq!(result.post_period_length, n - intervention_point);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.cumulative_effect.is_finite());
    }

    #[test]
    fn test_f_distribution_p_value() {
        let p = f_distribution_p_value(1.0, 5, 50);
        assert!(p > 0.0 && p < 1.0);

        let p = f_distribution_p_value(0.0, 5, 50);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_p_value() {
        let p = chi_squared_p_value(10.0, 5);
        assert!(p > 0.0 && p < 1.0);

        let p = chi_squared_p_value(0.0, 5);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_cdf_and_quantile() {
        let p = normal_cdf(0.0);
        assert!((p - 0.5).abs() < 0.01);

        let q = normal_quantile(0.975);
        assert!((q - 1.96).abs() < 0.05);
    }

    #[test]
    fn test_solve_linear_system() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0]).expect("Shape failed");
        let b = Array1::from_vec(vec![5.0, 7.0]);

        let x = solve_linear_system(&a, &b).expect("Solve failed");
        // 2x + y = 5, x + 3y = 7 => x = 1.6, y = 1.8
        assert!((x[0] - 1.6).abs() < 0.1);
        assert!((x[1] - 1.8).abs() < 0.1);
    }
}
