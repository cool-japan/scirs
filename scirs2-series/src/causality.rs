//! Causality testing and relationship analysis for time series
//!
//! This module provides various methods for testing causal relationships between time series:
//! - Granger causality testing
//! - Transfer entropy measures
//! - Convergent cross mapping
//! - Causal impact analysis

use crate::error::TimeSeriesError;
use ndarray::{s, Array1, Array2};
use scirs2_core::validation::check_array_finite;
use std::collections::HashMap;

/// Result type for causality testing
pub type CausalityResult<T> = Result<T, TimeSeriesError>;

/// Granger causality test result
#[derive(Debug, Clone)]
pub struct GrangerCausalityResult {
    /// F-statistic for the causality test
    pub f_statistic: f64,
    /// P-value of the test
    pub p_value: f64,
    /// Whether causality is rejected at the significance level
    pub is_causal: bool,
    /// Significance level used
    pub significance_level: f64,
    /// Degrees of freedom for the F-test
    pub degrees_of_freedom: (usize, usize),
    /// Log-likelihood of the restricted model
    pub ll_restricted: f64,
    /// Log-likelihood of the unrestricted model
    pub ll_unrestricted: f64,
}

/// Transfer entropy result
#[derive(Debug, Clone)]
pub struct TransferEntropyResult {
    /// Transfer entropy value
    pub transfer_entropy: f64,
    /// P-value from significance test (if computed)
    pub p_value: Option<f64>,
    /// Number of bins used for entropy calculation
    pub bins: usize,
    /// Embedding dimension used
    pub embedding_dim: usize,
    /// Time delay used
    pub time_delay: usize,
}

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

/// Configuration for Granger causality test
#[derive(Debug, Clone)]
pub struct GrangerConfig {
    /// Maximum lag to test
    pub max_lag: usize,
    /// Significance level for the test
    pub significance_level: f64,
    /// Whether to include trend in the model
    pub include_trend: bool,
    /// Whether to include constant in the model
    pub include_constant: bool,
}

impl Default for GrangerConfig {
    fn default() -> Self {
        Self {
            max_lag: 4,
            significance_level: 0.05,
            include_trend: false,
            include_constant: true,
        }
    }
}

/// Configuration for transfer entropy calculation
#[derive(Debug, Clone)]
pub struct TransferEntropyConfig {
    /// Number of bins for entropy calculation
    pub bins: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Time delay for embedding
    pub time_delay: usize,
    /// Number of bootstrap samples for significance testing
    pub bootstrap_samples: Option<usize>,
}

impl Default for TransferEntropyConfig {
    fn default() -> Self {
        Self {
            bins: 10,
            embedding_dim: 3,
            time_delay: 1,
            bootstrap_samples: Some(1000),
        }
    }
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

    /// Test Granger causality between two time series
    ///
    /// Tests whether `x` Granger-causes `y` using vector autoregression.
    /// Returns the F-statistic and p-value for the null hypothesis of no causality.
    ///
    /// # Arguments
    ///
    /// * `x` - The potential causal series
    /// * `y` - The potentially caused series
    /// * `config` - Configuration for the test
    ///
    /// # Returns
    ///
    /// Result containing Granger causality test statistics
    pub fn granger_causality(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &GrangerConfig,
    ) -> CausalityResult<GrangerCausalityResult> {
        check_array_finite(x, "x")?;
        check_array_finite(y, "y")?;

        if x.len() != y.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Time series must have the same length".to_string(),
            ));
        }

        if x.len() <= config.max_lag + 2 {
            return Err(TimeSeriesError::InvalidInput(
                "Time series too short for the specified lag".to_string(),
            ));
        }

        // Prepare data matrix with lags
        let n = x.len() - config.max_lag;
        let mut data = Array2::zeros((
            n,
            2 * config.max_lag + if config.include_constant { 1 } else { 0 },
        ));
        let mut y_vec = Array1::zeros(n);

        // Fill the data matrix
        for i in 0..n {
            let row_idx = config.max_lag + i;
            y_vec[i] = y[row_idx];

            // Add lagged y values
            for lag in 1..=config.max_lag {
                data[[i, lag - 1]] = y[row_idx - lag];
            }

            // Add lagged x values
            for lag in 1..=config.max_lag {
                data[[i, config.max_lag + lag - 1]] = x[row_idx - lag];
            }

            // Add constant if requested
            if config.include_constant {
                data[[i, 2 * config.max_lag]] = 1.0;
            }
        }

        // Fit unrestricted model (with x lags)
        let unrestricted_rss = self.compute_regression_rss(&data, &y_vec)?;
        let unrestricted_ll = self.compute_regression_likelihood(&data, &y_vec)?;

        // Fit restricted model (without x lags)
        let restricted_data = if config.include_constant {
            data.slice(s![.., ..config.max_lag + 1]).to_owned()
        } else {
            data.slice(s![.., ..config.max_lag]).to_owned()
        };
        let restricted_rss = self.compute_regression_rss(&restricted_data, &y_vec)?;
        let restricted_ll = self.compute_regression_likelihood(&restricted_data, &y_vec)?;

        // Compute F-statistic
        let df_num = config.max_lag;
        let df_den = n - data.ncols();
        let f_statistic = if df_den > 0 && unrestricted_rss > 0.0 {
            ((restricted_rss - unrestricted_rss) / df_num as f64)
                / (unrestricted_rss / df_den as f64)
        } else {
            0.0
        };

        // Compute p-value using F-distribution approximation
        let p_value = self.f_distribution_p_value(f_statistic, df_num, df_den);

        Ok(GrangerCausalityResult {
            f_statistic,
            p_value,
            is_causal: p_value < config.significance_level,
            significance_level: config.significance_level,
            degrees_of_freedom: (df_num, df_den),
            ll_restricted: restricted_ll,
            ll_unrestricted: unrestricted_ll,
        })
    }

    /// Calculate transfer entropy from x to y
    ///
    /// Transfer entropy measures the amount of uncertainty reduced in future values of y
    /// by knowing past values of x, given past values of y.
    ///
    /// # Arguments
    ///
    /// * `x` - Source time series
    /// * `y` - Target time series
    /// * `config` - Configuration for transfer entropy calculation
    ///
    /// # Returns
    ///
    /// Result containing transfer entropy value and statistics
    pub fn transfer_entropy(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &TransferEntropyConfig,
    ) -> CausalityResult<TransferEntropyResult> {
        check_array_finite(x, "x")?;
        check_array_finite(y, "y")?;

        if x.len() != y.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Time series must have the same length".to_string(),
            ));
        }

        let required_length = config.embedding_dim * config.time_delay + 1;
        if x.len() < required_length {
            return Err(TimeSeriesError::InvalidInput(
                "Time series too short for the specified embedding parameters".to_string(),
            ));
        }

        // Create embeddings
        let (x_embed, y_embed, y_future) = self.create_embeddings(x, y, config)?;

        // Discretize the data
        let x_discrete = self.discretize_data(&x_embed, config.bins)?;
        let y_discrete = self.discretize_data(&y_embed, config.bins)?;
        let y_future_discrete = self.discretize_array(&y_future, config.bins)?;

        // Calculate transfer entropy
        let te = self.calculate_transfer_entropy(&x_discrete, &y_discrete, &y_future_discrete)?;

        // Calculate p-value if bootstrap samples are specified
        let p_value = if let Some(n_bootstrap) = config.bootstrap_samples {
            Some(self.bootstrap_transfer_entropy_p_value(x, y, config, te, n_bootstrap)?)
        } else {
            None
        };

        Ok(TransferEntropyResult {
            transfer_entropy: te,
            p_value,
            bins: config.bins,
            embedding_dim: config.embedding_dim,
            time_delay: config.time_delay,
        })
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
        check_array_finite(x, "x")?;
        check_array_finite(y, "y")?;

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

        // Use the maximum correlation as the primary result
        let max_correlation = correlations
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Calculate p-value using bootstrap
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
        check_array_finite(y, "y")?;

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
        let prediction_std = (&predicted_upper - &predicted_lower) / (2.0 * 1.96); // Approximate standard error
        let standardized_effect = average_effect / (prediction_std.mean().unwrap_or(1.0));
        let p_value = 2.0 * (1.0 - self.normal_cdf(standardized_effect.abs()));

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

    // Helper methods

    fn compute_regression_likelihood(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> CausalityResult<f64> {
        // Simple OLS regression for likelihood calculation
        let xt = x.t();
        let xtx = xt.dot(x);

        // Add small regularization to handle near-singular matrices
        let mut xtx_reg = xtx;
        for i in 0..xtx_reg.nrows() {
            xtx_reg[[i, i]] += 1e-10;
        }

        // Solve using Cholesky decomposition approximation
        let xty = xt.dot(y);
        let beta = self.solve_linear_system(&xtx_reg, &xty)?;

        let predicted = x.dot(&beta);
        let residuals = y - &predicted;
        let sse = residuals.mapv(|x| x * x).sum();
        let n = y.len() as f64;

        // Log-likelihood for normal distribution
        let sigma_sq = sse / n;
        let ll = -0.5 * n * (2.0 * std::f64::consts::PI * sigma_sq).ln() - 0.5 * sse / sigma_sq;

        Ok(ll)
    }

    fn solve_linear_system(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> CausalityResult<Array1<f64>> {
        // Simple iterative solver for Ax = b
        let n = a.nrows();
        let mut x = Array1::zeros(n);
        let max_iter = 1000;
        let tolerance = 1e-10;

        for _iter in 0..max_iter {
            let mut x_new = x.clone();

            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    if i != j {
                        sum += a[[i, j]] * x[j];
                    }
                }
                x_new[i] = (b[i] - sum) / a[[i, i]];
            }

            let diff = (&x_new - &x).mapv(|x| x.abs()).sum();
            x = x_new;

            if diff < tolerance {
                break;
            }
        }

        Ok(x)
    }

    fn f_distribution_p_value(&self, f_stat: f64, df1: usize, df2: usize) -> f64 {
        // Approximation for F-distribution p-value
        // This is a simplified implementation
        if f_stat <= 0.0 {
            return 1.0;
        }

        // Transform to beta distribution
        let x = (df1 as f64 * f_stat) / (df1 as f64 * f_stat + df2 as f64);
        let alpha = df1 as f64 / 2.0;
        let beta = df2 as f64 / 2.0;

        // Approximate beta CDF complement
        1.0 - self.incomplete_beta(x, alpha, beta)
    }

    fn incomplete_beta(&self, x: f64, a: f64, b: f64) -> f64 {
        // Simplified incomplete beta function approximation
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }

        // Use continued fraction approximation
        let mut result = x.powf(a) * (1.0 - x).powf(b) / a;
        let mut term = result;

        for n in 1..100 {
            let n_f = n as f64;
            term *= (a + n_f - 1.0) * x / n_f;
            result += term;
            if term.abs() < 1e-10 {
                break;
            }
        }

        result / self.beta_function(a, b)
    }

    fn beta_function(&self, a: f64, b: f64) -> f64 {
        // Approximation using gamma function relation
        self.gamma_function(a) * self.gamma_function(b) / self.gamma_function(a + b)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn gamma_function(&self, x: f64) -> f64 {
        // Stirling's approximation for gamma function
        if x < 1.0 {
            return self.gamma_function(x + 1.0) / x;
        }

        (2.0 * std::f64::consts::PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
    }

    fn create_embeddings(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &TransferEntropyConfig,
    ) -> CausalityResult<(Array2<f64>, Array2<f64>, Array1<f64>)> {
        let embed_length = config.embedding_dim * config.time_delay;
        let n_points = x.len() - embed_length;

        let mut x_embed = Array2::zeros((n_points, config.embedding_dim));
        let mut y_embed = Array2::zeros((n_points, config.embedding_dim));
        let mut y_future = Array1::zeros(n_points);

        for i in 0..n_points {
            // Future value of y
            y_future[i] = y[i + embed_length];

            // Embedded values
            for j in 0..config.embedding_dim {
                let idx = i + j * config.time_delay;
                x_embed[[i, j]] = x[idx];
                y_embed[[i, j]] = y[idx];
            }
        }

        Ok((x_embed, y_embed, y_future))
    }

    fn discretize_data(&self, data: &Array2<f64>, bins: usize) -> CausalityResult<Array2<usize>> {
        let mut discrete = Array2::zeros((data.nrows(), data.ncols()));

        for col in 0..data.ncols() {
            let column = data.column(col);
            let min_val = column.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            if (max_val - min_val).abs() < f64::EPSILON {
                // All values are the same
                continue;
            }

            let bin_width = (max_val - min_val) / bins as f64;

            for row in 0..data.nrows() {
                let val = data[[row, col]];
                let bin = ((val - min_val) / bin_width).floor() as usize;
                discrete[[row, col]] = bin.min(bins - 1);
            }
        }

        Ok(discrete)
    }

    fn discretize_array(&self, data: &Array1<f64>, bins: usize) -> CausalityResult<Array1<usize>> {
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < f64::EPSILON {
            return Ok(Array1::zeros(data.len()));
        }

        let bin_width = (max_val - min_val) / bins as f64;
        let discrete = data.mapv(|val| {
            let bin = ((val - min_val) / bin_width).floor() as usize;
            bin.min(bins - 1)
        });

        Ok(discrete)
    }

    fn calculate_transfer_entropy(
        &self,
        x_discrete: &Array2<usize>,
        y_discrete: &Array2<usize>,
        y_future_discrete: &Array1<usize>,
    ) -> CausalityResult<f64> {
        // Calculate joint and marginal probabilities using separate maps
        let mut joint_counts = HashMap::new();
        let mut marginal_y_counts = HashMap::new();
        let mut conditional_counts = HashMap::new();
        let mut y_only_counts = HashMap::new();
        let n_samples = x_discrete.nrows();

        // Count occurrences
        for i in 0..n_samples {
            let x_state = x_discrete.row(i).to_vec();
            let y_state = y_discrete.row(i).to_vec();
            let y_fut = y_future_discrete[i];

            // Joint distribution P(y_n+1, y_n, x_n)
            let joint_key = (y_fut, y_state.clone(), x_state.clone());
            *joint_counts.entry(joint_key).or_insert(0) += 1;

            // Marginal P(y_n+1, y_n)
            let marginal_key = (y_fut, y_state.clone());
            *marginal_y_counts.entry(marginal_key).or_insert(0) += 1;

            // Conditional P(y_n, x_n)
            let cond_key = (y_state.clone(), x_state);
            *conditional_counts.entry(cond_key).or_insert(0) += 1;

            // Marginal P(y_n)
            *y_only_counts.entry(y_state).or_insert(0) += 1;
        }

        // Calculate transfer entropy
        let mut te = 0.0;

        for (joint_key, &joint_count) in &joint_counts {
            let prob_joint = joint_count as f64 / n_samples as f64;

            if prob_joint > 0.0 {
                // Extract components for conditional probabilities
                let (y_fut, y_state, x_state) = joint_key;

                let marginal_count = marginal_y_counts
                    .get(&(*y_fut, y_state.clone()))
                    .unwrap_or(&0);
                let cond_count = conditional_counts
                    .get(&(y_state.clone(), x_state.clone()))
                    .unwrap_or(&0);
                let y_only_count = y_only_counts.get(y_state).unwrap_or(&0);

                if *marginal_count > 0 && *cond_count > 0 && *y_only_count > 0 {
                    let prob_marginal = *marginal_count as f64 / n_samples as f64;
                    let prob_cond = *cond_count as f64 / n_samples as f64;
                    let prob_y_only = *y_only_count as f64 / n_samples as f64;

                    let ratio = (prob_joint * prob_y_only) / (prob_marginal * prob_cond);
                    if ratio > 0.0 {
                        te += prob_joint * ratio.ln();
                    }
                }
            }
        }

        Ok(te)
    }

    fn bootstrap_transfer_entropy_p_value(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &TransferEntropyConfig,
        observed_te: f64,
        n_bootstrap: usize,
    ) -> CausalityResult<f64> {
        let mut te_values = Vec::new();

        for _i in 0..n_bootstrap {
            // Create shuffled version of x to break temporal dependence
            let mut x_shuffled = x.clone();
            self.fisher_yates_shuffle(&mut x_shuffled);

            let te_result = self.transfer_entropy(&x_shuffled, y, config)?;
            te_values.push(te_result.transfer_entropy);
        }

        // Calculate p-value as proportion of bootstrap samples >= observed value
        let count = te_values.iter().filter(|&&te| te >= observed_te).count();
        Ok(count as f64 / n_bootstrap as f64)
    }

    fn fisher_yates_shuffle(&self, arr: &mut Array1<f64>) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        if let Some(seed) = self.random_seed {
            seed.hash(&mut hasher);
        }

        for i in (1..arr.len()).rev() {
            hasher.write_usize(i);
            let j = (hasher.finish() as usize) % (i + 1);
            arr.swap(i, j);
        }
    }

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

            // Find nearest neighbors in the library
            let mut distances = Vec::new();
            for j in 0..library_size {
                let library_point = x_manifold.row(j);
                let dist = self.euclidean_distance(&query_point, &library_point);
                distances.push((dist, j));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Use nearest neighbors to predict y value
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            #[allow(clippy::needless_range_loop)]
            for k in 0..num_neighbors.min(distances.len()) {
                let (dist, idx) = distances[k];
                let weight = (-dist).exp(); // Exponential weighting
                weighted_sum += weight * y_manifold[[idx, 0]]; // Use first component of y
                weight_sum += weight;
            }

            if weight_sum > 0.0 {
                predictions.push(weighted_sum / weight_sum);
                actuals.push(y_manifold[[library_size + i, 0]]);
            }
        }

        // Calculate correlation coefficient
        if predictions.is_empty() {
            return Ok(0.0);
        }

        let correlation = self.pearson_correlation(&predictions, &actuals);
        Ok(correlation)
    }

    fn euclidean_distance(
        &self,
        a: &ndarray::ArrayView1<f64>,
        b: &ndarray::ArrayView1<f64>,
    ) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
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

    fn bootstrap_ccm_p_value(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        config: &CCMConfig,
        observed_correlation: f64,
    ) -> CausalityResult<f64> {
        let mut correlations = Vec::new();

        for _i in 0..config.bootstrap_samples {
            // Create surrogate data by shuffling one of the series
            let mut y_shuffled = y.clone();
            self.fisher_yates_shuffle(&mut y_shuffled);

            let ccm_result = self.convergent_cross_mapping(x, &y_shuffled, config)?;
            correlations.push(ccm_result.correlation);
        }

        // Calculate p-value
        let count = correlations
            .iter()
            .filter(|&&corr| corr >= observed_correlation)
            .count();
        Ok(count as f64 / config.bootstrap_samples as f64)
    }

    fn fit_and_predict_structural_model(
        &self,
        pre_y: &Array1<f64>,
        pre_x: &Array2<f64>,
        post_x: &Array2<f64>,
        confidence_level: f64,
    ) -> CausalityResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        // Simple linear regression model for structural prediction
        // In practice, this would use more sophisticated state-space models

        // Prepare design matrix with lag terms
        let n = pre_y.len();
        let p = pre_x.ncols();
        let mut design_matrix = Array2::zeros((n - 1, p + 1)); // +1 for lagged y
        let mut response = Array1::zeros(n - 1);

        for i in 1..n {
            response[i - 1] = pre_y[i];
            design_matrix[[i - 1, 0]] = pre_y[i - 1]; // Lagged y
            for j in 0..p {
                design_matrix[[i - 1, j + 1]] = pre_x[[i - 1, j]];
            }
        }

        // Fit regression model
        let xt = design_matrix.t();
        let xtx = xt.dot(&design_matrix);
        let xty = xt.dot(&response);
        let beta = self.solve_linear_system(&xtx, &xty)?;

        // Calculate residual standard error
        let fitted = design_matrix.dot(&beta);
        let residuals = &response - &fitted;
        let mse = residuals.mapv(|x| x * x).sum() / (n - 1 - beta.len()) as f64;
        let std_error = mse.sqrt();

        // Predict post-intervention values
        let n_post = post_x.nrows();
        let mut predicted = Array1::zeros(n_post);
        let mut predicted_lower = Array1::zeros(n_post);
        let mut predicted_upper = Array1::zeros(n_post);

        let z_score = self.normal_quantile((1.0 + confidence_level) / 2.0);

        for i in 0..n_post {
            let mut x_new = Array1::zeros(p + 1);

            // Use last observed value or previous prediction as lagged y
            if i == 0 {
                x_new[0] = pre_y[pre_y.len() - 1];
            } else {
                x_new[0] = predicted[i - 1];
            }

            // Add covariates
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

    fn normal_cdf(&self, x: f64) -> f64 {
        // Approximation for standard normal CDF
        0.5 * (1.0 + self.erf(x / (2.0_f64).sqrt()))
    }

    fn normal_quantile(&self, p: f64) -> f64 {
        // Approximation for normal quantile function
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if (p - 0.5).abs() < f64::EPSILON {
            return 0.0;
        }

        // Beasley-Springer-Moro algorithm approximation
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
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
    }

    fn erf(&self, x: f64) -> f64 {
        // Approximation for error function
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    fn compute_regression_rss(&self, x: &Array2<f64>, y: &Array1<f64>) -> CausalityResult<f64> {
        // Simple OLS regression for RSS calculation
        let xt = x.t();
        let xtx = xt.dot(x);

        // Add small regularization to handle near-singular matrices
        let mut xtx_reg = xtx;
        for i in 0..xtx_reg.nrows() {
            xtx_reg[[i, i]] += 1e-10;
        }

        // Solve using linear system
        let xty = xt.dot(y);
        let beta = self.solve_linear_system(&xtx_reg, &xty)?;

        let predicted = x.dot(&beta);
        let residuals = y - &predicted;
        let rss = residuals.mapv(|x| x * x).sum();

        Ok(rss)
    }
}

impl Default for CausalityTester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_granger_causality() {
        let n = 100;
        let mut x = Array1::zeros(n);
        let mut y = Array1::zeros(n);

        // Generate test data where x causes y
        for i in 1..n {
            x[i] = 0.5 * x[i - 1] + rand::random::<f64>();
            y[i] = 0.3 * y[i - 1] + 0.4 * x[i - 1] + rand::random::<f64>();
        }

        let tester = CausalityTester::new();
        let config = GrangerConfig::default();
        let result = tester.granger_causality(&x, &y, &config).unwrap();

        assert!(result.f_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_transfer_entropy() {
        let n = 50;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect());

        let tester = CausalityTester::new();
        let config = TransferEntropyConfig {
            bins: 5,
            embedding_dim: 2,
            time_delay: 1,
            bootstrap_samples: None,
        };

        let result = tester.transfer_entropy(&x, &y, &config).unwrap();

        assert!(result.transfer_entropy >= 0.0);
        assert_eq!(result.bins, 5);
        assert_eq!(result.embedding_dim, 2);
    }

    #[test]
    fn test_causal_impact() {
        let n = 50;
        let intervention_point = 30;

        // Generate synthetic data
        let y = Array1::from_vec(
            (0..n)
                .map(|i| {
                    if i < intervention_point {
                        i as f64 + rand::random::<f64>()
                    } else {
                        i as f64 + 10.0 + rand::random::<f64>() // Effect after intervention
                    }
                })
                .collect(),
        );

        let x = Array2::from_shape_vec((n, 1), (0..n).map(|i| i as f64).collect()).unwrap();

        let tester = CausalityTester::new();
        let result = tester
            .causal_impact_analysis(&y, &x, intervention_point, 0.95)
            .unwrap();

        assert_eq!(result.pre_period_length, intervention_point);
        assert_eq!(result.post_period_length, n - intervention_point);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.cumulative_effect.is_finite());
    }
}
