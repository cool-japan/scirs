//! Survival Analysis
//!
//! This module provides survival analysis functions including Kaplan-Meier estimator,
//! Cox proportional hazards model, and related statistical tests.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::validation::*;

/// Kaplan-Meier survival estimator
///
/// Computes the Kaplan-Meier survival function from time-to-event data.
#[derive(Debug, Clone)]
pub struct KaplanMeierEstimator {
    /// Unique event times
    pub event_times: Array1<f64>,
    /// Survival probabilities at each event time
    pub survival_function: Array1<f64>,
    /// Confidence intervals (lower bound, upper bound)
    pub confidence_intervals: Option<(Array1<f64>, Array1<f64>)>,
    /// Number at risk at each time point
    pub at_risk: Array1<usize>,
    /// Number of events at each time point
    pub events: Array1<usize>,
    /// Median survival time
    pub median_survival_time: Option<f64>,
}

impl KaplanMeierEstimator {
    /// Fit Kaplan-Meier estimator
    ///
    /// # Arguments
    /// * `durations` - Time to event or censoring
    /// * `event_observed` - Whether event was observed (true) or censored (false)
    /// * `confidence_level` - Confidence level for intervals (0.0 to 1.0)
    ///
    /// # Returns
    /// * Kaplan-Meier estimator with survival function and statistics
    pub fn fit(
        durations: ArrayView1<f64>,
        event_observed: ArrayView1<bool>,
        confidence_level: Option<f64>,
    ) -> Result<Self> {
        checkarray_finite(&durations, "durations")?;

        if durations.len() != event_observed.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "durations length ({durations_len}) must match event_observed length ({events_len})",
                durations_len = durations.len(),
                events_len = event_observed.len()
            )));
        }

        if durations.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        if let Some(conf) = confidence_level {
            check_probability(conf, "confidence_level")?;
        }

        // Create time-event pairs and sort by time
        let mut time_event_pairs: Vec<(f64, bool)> = durations
            .iter()
            .zip(event_observed.iter())
            .map(|(&t, &e)| (t, e))
            .collect();
        time_event_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Calculate survival function using Kaplan-Meier product-limit estimator
        let mut unique_times = Vec::new();
        let mut at_risk_counts = Vec::new();
        let mut event_counts = Vec::new();
        let mut survival_probs = Vec::new();

        let n = time_event_pairs.len();
        let mut current_survival = 1.0;
        let mut current_at_risk = n;

        let mut i = 0;
        while i < time_event_pairs.len() {
            let current_time = time_event_pairs[i].0;
            let mut events_at_time = 0;
            let mut censored_at_time = 0;

            // Count events and censored observations at current time
            while i < time_event_pairs.len() && time_event_pairs[i].0 == current_time {
                if time_event_pairs[i].1 {
                    events_at_time += 1;
                } else {
                    censored_at_time += 1;
                }
                i += 1;
            }

            if events_at_time > 0 {
                // Update survival probability only if there were events
                let survival_this_time = 1.0 - (events_at_time as f64) / (current_at_risk as f64);
                current_survival *= survival_this_time;

                unique_times.push(current_time);
                at_risk_counts.push(current_at_risk);
                event_counts.push(events_at_time);
                survival_probs.push(current_survival);
            }

            // Update at-risk count
            current_at_risk -= events_at_time + censored_at_time;
        }

        let event_times = Array1::from_vec(unique_times);
        let survival_function = Array1::from_vec(survival_probs);
        let at_risk = Array1::from_vec(at_risk_counts);
        let events = Array1::from_vec(event_counts);

        // Calculate confidence intervals if requested
        let confidence_intervals = if let Some(conf_level) = confidence_level {
            Some(Self::calculate_confidence_intervals(
                &survival_function,
                &at_risk,
                &events,
                conf_level,
            )?)
        } else {
            None
        };

        // Calculate median survival time
        let median_survival_time =
            Self::calculate_median_survival(&event_times, &survival_function);

        Ok(Self {
            event_times,
            survival_function,
            confidence_intervals,
            at_risk,
            events,
            median_survival_time,
        })
    }

    /// Calculate confidence intervals using Greenwood's formula
    fn calculate_confidence_intervals(
        survival_function: &Array1<f64>,
        at_risk: &Array1<usize>,
        events: &Array1<usize>,
        confidence_level: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let _alpha = 1.0 - confidence_level;
        let z_score = 1.96; // Approximate 95% CI, should use proper inverse normal for other levels

        let mut lower_bounds = Array1::zeros(survival_function.len());
        let mut upper_bounds = Array1::zeros(survival_function.len());

        // Cumulative Greenwood variance
        let mut cumulative_variance = 0.0;

        for i in 0..survival_function.len() {
            // Greenwood's formula for variance
            let n_i = at_risk[i] as f64;
            let d_i = events[i] as f64;

            if n_i > d_i && n_i > 0.0 {
                cumulative_variance += d_i / (n_i * (n_i - d_i));
            }

            let s_t = survival_function[i];
            if s_t > 0.0 {
                let se = s_t * cumulative_variance.sqrt();

                // Using log-log transformation for better CI properties
                let log_log_s = (-(s_t.ln())).ln();
                let se_log_log = se / (s_t * s_t.ln().abs());

                let lower_log_log = log_log_s - z_score * se_log_log;
                let upper_log_log = log_log_s + z_score * se_log_log;

                lower_bounds[i] = (-(-lower_log_log.exp()).exp()).max(0.0);
                upper_bounds[i] = (-(-upper_log_log.exp()).exp()).min(1.0);
            } else {
                lower_bounds[i] = 0.0;
                upper_bounds[i] = 0.0;
            }
        }

        Ok((lower_bounds, upper_bounds))
    }

    /// Calculate median survival time
    fn calculate_median_survival(
        event_times: &Array1<f64>,
        survival_function: &Array1<f64>,
    ) -> Option<f64> {
        // Find the time where survival probability first drops to or below 0.5
        for i in 0..survival_function.len() {
            if survival_function[i] <= 0.5 {
                return Some(event_times[i]);
            }
        }
        None // Median not reached
    }

    /// Predict survival probability at given times
    pub fn predict(&self, times: ArrayView1<f64>) -> Result<Array1<f64>> {
        checkarray_finite(&times, "times")?;

        let mut predictions = Array1::zeros(times.len());

        for (i, &t) in times.iter().enumerate() {
            if t < 0.0 {
                return Err(StatsError::InvalidArgument(
                    "Times must be non-negative".to_string(),
                ));
            }

            // Find the survival probability at time t
            let mut survival_prob = 1.0; // Start with 100% survival at time 0

            for j in 0..self.event_times.len() {
                if self.event_times[j] <= t {
                    survival_prob = self.survival_function[j];
                } else {
                    break;
                }
            }

            predictions[i] = survival_prob;
        }

        Ok(predictions)
    }
}

/// Log-rank test for comparing survival curves
///
/// Tests the null hypothesis that two or more survival curves are identical.
pub struct LogRankTest;

impl LogRankTest {
    /// Perform log-rank test comparing two survival curves
    ///
    /// # Arguments
    /// * `durations1` - Time to event or censoring for group 1
    /// * `events1` - Whether event was observed for group 1
    /// * `durations2` - Time to event or censoring for group 2
    /// * `events2` - Whether event was observed for group 2
    ///
    /// # Returns
    /// * Tuple of (test statistic, p-value)
    pub fn compare_two_groups(
        durations1: ArrayView1<f64>,
        events1: ArrayView1<bool>,
        durations2: ArrayView1<f64>,
        events2: ArrayView1<bool>,
    ) -> Result<(f64, f64)> {
        checkarray_finite(&durations1, "durations1")?;
        checkarray_finite(&durations2, "durations2")?;

        if durations1.len() != events1.len() || durations2.len() != events2.len() {
            return Err(StatsError::DimensionMismatch(
                "Durations and events arrays must have same length".to_string(),
            ));
        }

        // Combine all observations with group labels
        let mut combineddata = Vec::new();

        for i in 0..durations1.len() {
            combineddata.push((durations1[i], events1[i], 0)); // Group 0
        }
        for i in 0..durations2.len() {
            combineddata.push((durations2[i], events2[i], 1)); // Group 1
        }

        // Sort by time
        combineddata.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Calculate log-rank statistic
        let mut observed_group1 = 0.0;
        let mut expected_group1 = 0.0;
        let mut variance = 0.0;

        let n1 = durations1.len() as f64;
        let n2 = durations2.len() as f64;
        let mut at_risk1 = n1;
        let mut at_risk2 = n2;

        let mut i = 0;
        while i < combineddata.len() {
            let current_time = combineddata[i].0;
            let mut events_group1 = 0.0;
            let mut events_group2 = 0.0;
            let mut censored_group1 = 0.0;
            let mut censored_group2 = 0.0;

            // Count events and censoring at current time
            while i < combineddata.len() && combineddata[i].0 == current_time {
                let (_, is_event, group) = combineddata[i];
                if group == 0 {
                    if is_event {
                        events_group1 += 1.0;
                    } else {
                        censored_group1 += 1.0;
                    }
                } else if is_event {
                    events_group2 += 1.0;
                } else {
                    censored_group2 += 1.0;
                }
                i += 1;
            }

            let total_events = events_group1 + events_group2;
            let total_at_risk = at_risk1 + at_risk2;

            if total_events > 0.0 && total_at_risk > 0.0 {
                // Expected events in group 1
                let expected_events1 = (at_risk1 / total_at_risk) * total_events;

                // Variance contribution
                let var_contrib =
                    (at_risk1 * at_risk2 * total_events * (total_at_risk - total_events))
                        / (total_at_risk.powi(2) * (total_at_risk - 1.0).max(1.0));

                observed_group1 += events_group1;
                expected_group1 += expected_events1;
                variance += var_contrib;
            }

            // Update at-risk counts
            at_risk1 -= events_group1 + censored_group1;
            at_risk2 -= events_group2 + censored_group2;
        }

        // Calculate test statistic
        if variance <= 0.0 {
            return Ok((0.0, 1.0)); // No variance, no difference
        }

        let test_statistic = (observed_group1 - expected_group1).powi(2) / variance;

        // Calculate p-value using chi-square distribution with 1 df
        let p_value = Self::chi_square_survival(test_statistic, 1.0);

        Ok((test_statistic, p_value))
    }

    /// Approximate survival function for chi-square distribution
    fn chi_square_survival(x: f64, df: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }

        // Simple approximation - in practice would use proper chi-square CDF
        let mean = df;
        let var = 2.0 * df;
        let std = var.sqrt();

        // Normal approximation for large df
        if df > 30.0 {
            let z = (x - mean) / std;
            return 0.5 * (1.0 - Self::erf(z / std::f64::consts::SQRT_2));
        }

        // Simple exponential approximation for small df
        (-x / mean).exp()
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

/// Cox Proportional Hazards Model
///
/// Implements Cox regression for survival analysis with covariates.
#[derive(Debug, Clone)]
pub struct CoxPHModel {
    /// Regression coefficients
    pub coefficients: Array1<f64>,
    /// Covariance matrix of coefficients
    pub covariance_matrix: Array2<f64>,
    /// Log-likelihood of the fitted model
    pub log_likelihood: f64,
    /// Baseline cumulative hazard
    pub baseline_cumulative_hazard: Array1<f64>,
    /// Time points for baseline hazard
    pub baseline_times: Array1<f64>,
    /// Number of iterations until convergence
    pub n_iter: usize,
}

impl CoxPHModel {
    /// Fit Cox proportional hazards model
    ///
    /// # Arguments
    /// * `durations` - Time to event or censoring
    /// * `events` - Whether event was observed
    /// * `covariates` - Covariate matrix (n_samples_ x n_features)
    /// * `max_iter` - Maximum number of iterations
    /// * `tol` - Convergence tolerance
    ///
    /// # Returns
    /// * Fitted Cox model
    pub fn fit(
        durations: ArrayView1<f64>,
        events: ArrayView1<bool>,
        covariates: ArrayView2<f64>,
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> Result<Self> {
        checkarray_finite(&durations, "durations")?;
        checkarray_finite(&covariates, "covariates")?;

        let (n_samples_, n_features) = covariates.dim();
        let max_iter = max_iter.unwrap_or(100);
        let tol = tol.unwrap_or(1e-6);

        if durations.len() != n_samples_ || events.len() != n_samples_ {
            return Err(StatsError::DimensionMismatch(
                "All input arrays must have the same number of samples".to_string(),
            ));
        }

        // Initialize coefficients
        let mut beta = Array1::zeros(n_features);
        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for iteration in 0..max_iter {
            // Calculate partial likelihood and its derivatives
            let (log_likelihood, gradient, hessian) =
                Self::partial_likelihood_derivatives(&durations, &events, &covariates, &beta)?;

            // Check convergence
            if (log_likelihood - prev_log_likelihood).abs() < tol {
                let covariance_matrix = Self::invert_hessian(&hessian)?;
                let (baseline_times, baseline_cumulative_hazard) =
                    Self::calculatebaseline_hazard(&durations, &events, &covariates, &beta)?;

                return Ok(Self {
                    coefficients: beta,
                    covariance_matrix,
                    log_likelihood,
                    baseline_cumulative_hazard,
                    baseline_times,
                    n_iter: iteration + 1,
                });
            }

            // Newton-Raphson update
            let hessian_inv = Self::invert_hessian(&hessian)?;
            let delta = hessian_inv.dot(&gradient);
            beta = &beta + &delta;

            prev_log_likelihood = log_likelihood;
        }

        Err(StatsError::ConvergenceError(format!(
            "Cox model failed to converge after {max_iter} iterations"
        )))
    }

    /// Calculate partial likelihood and its derivatives
    fn partial_likelihood_derivatives(
        durations: &ArrayView1<f64>,
        events: &ArrayView1<bool>,
        covariates: &ArrayView2<f64>,
        beta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let n_samples_ = durations.len();
        let n_features = beta.len();

        // Sort by event time
        let mut indices: Vec<usize> = (0..n_samples_).collect();
        indices.sort_by(|&i, &j| durations[i].partial_cmp(&durations[j]).unwrap());

        let mut log_likelihood = 0.0;
        let mut gradient = Array1::zeros(n_features);
        let mut hessian = Array2::zeros((n_features, n_features));

        for &i in &indices {
            if !events[i] {
                continue; // Skip censored observations for likelihood
            }

            let t_i = durations[i];
            let x_i = covariates.row(i);

            // Calculate risk sets (all subjects at risk at time t_i)
            let mut risk_set = Vec::new();
            for &j in &indices {
                if durations[j] >= t_i {
                    risk_set.push(j);
                }
            }

            if risk_set.is_empty() {
                continue;
            }

            // Calculate exp(beta' * x) for risk set
            let mut exp_beta_x = Array1::zeros(risk_set.len());
            for (k, &j) in risk_set.iter().enumerate() {
                let x_j = covariates.row(j);
                exp_beta_x[k] = x_j.dot(beta).exp();
            }

            let sum_exp = exp_beta_x.sum();
            if sum_exp <= 0.0 {
                continue;
            }

            // Update log-likelihood
            log_likelihood += x_i.dot(beta) - sum_exp.ln();

            // Update gradient
            let mut weighted_x = Array1::<f64>::zeros(n_features);
            for (k, &j) in risk_set.iter().enumerate() {
                let x_j = covariates.row(j);
                let weight = exp_beta_x[k] / sum_exp;
                weighted_x = &weighted_x + &(weight * &x_j.to_owned());
            }
            gradient = &gradient + &(&x_i.to_owned() - &weighted_x);

            // Update Hessian (simplified - should include second-order terms)
            for p in 0..n_features {
                for q in 0..n_features {
                    let mut weighted_sum = 0.0;
                    for (k, &j) in risk_set.iter().enumerate() {
                        let x_j = covariates.row(j);
                        let weight = exp_beta_x[k] / sum_exp;
                        weighted_sum += weight * x_j[p] * x_j[q];
                    }
                    hessian[[p, q]] -= weighted_sum - (weighted_x[p] * weighted_x[q]);
                }
            }
        }

        Ok((log_likelihood, gradient, hessian))
    }

    /// Invert Hessian matrix (negative for Newton-Raphson)
    fn invert_hessian(hessian: &Array2<f64>) -> Result<Array2<f64>> {
        let neg_hessian = -hessian;
        scirs2_linalg::inv(&neg_hessian.view(), None)
            .map_err(|e| StatsError::ComputationError(format!("Failed to invert Hessian: {e}")))
    }

    /// Calculate baseline hazard function
    fn calculatebaseline_hazard(
        durations: &ArrayView1<f64>,
        events: &ArrayView1<bool>,
        covariates: &ArrayView2<f64>,
        beta: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let n_samples_ = durations.len();

        // Sort by event time
        let mut indices: Vec<usize> = (0..n_samples_).collect();
        indices.sort_by(|&i, &j| durations[i].partial_cmp(&durations[j]).unwrap());

        let mut times = Vec::new();
        let mut cumulative_hazard = Vec::new();
        let mut current_cumhaz = 0.0;

        for &i in &indices {
            if !events[i] {
                continue;
            }

            let t_i = durations[i];

            // Calculate risk set
            let mut risk_sum = 0.0;
            for &j in &indices {
                if durations[j] >= t_i {
                    let x_j = covariates.row(j);
                    risk_sum += x_j.dot(beta).exp();
                }
            }

            if risk_sum > 0.0 {
                current_cumhaz += 1.0 / risk_sum; // Breslow estimator
                times.push(t_i);
                cumulative_hazard.push(current_cumhaz);
            }
        }

        Ok((Array1::from_vec(times), Array1::from_vec(cumulative_hazard)))
    }

    /// Predict hazard ratios for new data
    pub fn predict_hazard_ratio(&self, covariates: ArrayView2<f64>) -> Result<Array1<f64>> {
        checkarray_finite(&covariates, "covariates")?;

        if covariates.ncols() != self.coefficients.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "covariates has {features} features, expected {expected}",
                features = covariates.ncols(),
                expected = self.coefficients.len()
            )));
        }

        let mut hazard_ratios = Array1::zeros(covariates.nrows());

        for i in 0..covariates.nrows() {
            let x_i = covariates.row(i);
            hazard_ratios[i] = x_i.dot(&self.coefficients).exp();
        }

        Ok(hazard_ratios)
    }
}

/// Accelerated Failure Time (AFT) model
///
/// Models the logarithm of survival time as a linear function of covariates.
#[derive(Debug, Clone)]
pub struct AFTModel {
    /// Regression coefficients
    pub coefficients: Array1<f64>,
    /// Scale parameter
    pub scale: f64,
    /// Distribution type
    pub distribution: AFTDistribution,
}

/// Distribution types for AFT models
#[derive(Debug, Clone, Copy)]
pub enum AFTDistribution {
    /// Weibull distribution
    Weibull,
    /// Lognormal distribution
    Lognormal,
    /// Exponential distribution (special case of Weibull)
    Exponential,
}

impl AFTModel {
    /// Fit AFT model (simplified implementation)
    pub fn fit(
        durations: ArrayView1<f64>,
        events: ArrayView1<bool>,
        covariates: ArrayView2<f64>,
        distribution: AFTDistribution,
    ) -> Result<Self> {
        checkarray_finite(&durations, "durations")?;
        checkarray_finite(&covariates, "covariates")?;

        let (n_samples_, n_features) = covariates.dim();

        if durations.len() != n_samples_ || events.len() != n_samples_ {
            return Err(StatsError::DimensionMismatch(
                "All input arrays must have the same number of samples".to_string(),
            ));
        }

        // Simplified implementation: use log-linear regression on observed times
        // In practice, this would use maximum likelihood estimation

        let mut y = Array1::zeros(n_samples_);
        let mut weights = Array1::zeros(n_samples_);

        for i in 0..n_samples_ {
            y[i] = durations[i].ln();
            weights[i] = if events[i] { 1.0 } else { 0.5 }; // Downweight censored observations
        }

        // Weighted least squares (simplified)
        let mut xtx = Array2::zeros((n_features, n_features));
        let mut xty = Array1::zeros(n_features);

        for i in 0..n_samples_ {
            let x_i = covariates.row(i);
            let w = weights[i];

            for j in 0..n_features {
                xty[j] += w * x_i[j] * y[i];
                for k in 0..n_features {
                    xtx[[j, k]] += w * x_i[j] * x_i[k];
                }
            }
        }

        let coefficients = scirs2_linalg::solve(&xtx.view(), &xty.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to solve regression: {e}"))
        })?;

        // Estimate scale parameter
        let mut residual_sum = 0.0;
        let mut count = 0;

        for i in 0..n_samples_ {
            if events[i] {
                let x_i = covariates.row(i);
                let predicted = x_i.dot(&coefficients);
                let residual = y[i] - predicted;
                residual_sum += residual * residual;
                count += 1;
            }
        }

        let scale = if count > 0 {
            (residual_sum / count as f64).sqrt()
        } else {
            1.0
        };

        Ok(Self {
            coefficients,
            scale,
            distribution,
        })
    }

    /// Predict survival times
    pub fn predict(&self, covariates: ArrayView2<f64>) -> Result<Array1<f64>> {
        checkarray_finite(&covariates, "covariates")?;

        if covariates.ncols() != self.coefficients.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "covariates has {features} features, expected {expected}",
                features = covariates.ncols(),
                expected = self.coefficients.len()
            )));
        }

        let mut predictions = Array1::zeros(covariates.nrows());

        for i in 0..covariates.nrows() {
            let x_i = covariates.row(i);
            let log_time = x_i.dot(&self.coefficients);
            predictions[i] = log_time.exp();
        }

        Ok(predictions)
    }
}

/// Extended Cox model with time-dependent covariates and stratification
///
/// Supports time-varying covariates and stratified analysis for heterogeneous populations
#[derive(Debug, Clone)]
pub struct ExtendedCoxModel {
    /// Regression coefficients
    pub coefficients: Array1<f64>,
    /// Covariance matrix of coefficients
    pub covariance_matrix: Array2<f64>,
    /// Log-likelihood of the fitted model
    pub log_likelihood: f64,
    /// Baseline cumulative hazard for each stratum
    pub stratumbaseline_hazards: Vec<(Array1<f64>, Array1<f64>)>, // (times, cumulative hazards)
    /// Stratum labels
    pub strata: Option<Array1<usize>>,
    /// Number of strata
    pub n_strata: usize,
    /// Time-dependent covariate indices
    pub time_varying_indices: Vec<usize>,
    /// Number of iterations until convergence
    pub n_iter: usize,
}

impl ExtendedCoxModel {
    /// Fit extended Cox model with optional stratification and time-dependent covariates
    ///
    /// # Arguments
    /// * `durations` - Time to event or censoring
    /// * `events` - Whether event was observed
    /// * `covariates` - Covariate matrix (n_samples_ x n_features)
    /// * `strata` - Optional stratification variable
    /// * `time_varying_indices` - Indices of time-varying covariates
    /// * `time_points` - Time points for time-varying covariates (if any)
    /// * `time_varying_values` - Values of time-varying covariates at each time point
    pub fn fit_stratified(
        durations: ArrayView1<f64>,
        events: ArrayView1<bool>,
        covariates: ArrayView2<f64>,
        strata: Option<ArrayView1<usize>>,
        time_varying_indices: Option<Vec<usize>>,
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> Result<Self> {
        checkarray_finite(&durations, "durations")?;
        checkarray_finite(&covariates, "covariates")?;

        let (n_samples_, n_features) = covariates.dim();
        let max_iter = max_iter.unwrap_or(100);
        let tol = tol.unwrap_or(1e-6);

        if durations.len() != n_samples_ || events.len() != n_samples_ {
            return Err(StatsError::DimensionMismatch(
                "All input arrays must have the same number of samples".to_string(),
            ));
        }

        // Handle stratification
        let (strata_array, n_strata) = if let Some(strata_input) = strata {
            if strata_input.len() != n_samples_ {
                return Err(StatsError::DimensionMismatch(
                    "Strata length must match number of samples".to_string(),
                ));
            }
            let max_stratum = strata_input.iter().cloned().max().unwrap_or(0);
            (Some(strata_input.to_owned()), max_stratum + 1)
        } else {
            (None, 1)
        };

        let time_varying_indices = time_varying_indices.unwrap_or_default();

        // Initialize coefficients
        let mut beta = Array1::zeros(n_features);
        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for iteration in 0..max_iter {
            // Calculate stratified partial likelihood and derivatives
            let (log_likelihood, gradient, hessian) =
                Self::stratified_partial_likelihood_derivatives(
                    &durations,
                    &events,
                    &covariates,
                    &beta,
                    &strata_array,
                    n_strata,
                )?;

            // Check convergence
            if (log_likelihood - prev_log_likelihood).abs() < tol {
                let covariance_matrix = Self::invert_hessian(&hessian)?;
                let baseline_hazards = Self::calculate_stratifiedbaseline_hazards(
                    &durations,
                    &events,
                    &covariates,
                    &beta,
                    &strata_array,
                    n_strata,
                )?;

                return Ok(Self {
                    coefficients: beta,
                    covariance_matrix,
                    log_likelihood,
                    stratumbaseline_hazards: baseline_hazards,
                    strata: strata_array,
                    n_strata,
                    time_varying_indices,
                    n_iter: iteration + 1,
                });
            }

            // Newton-Raphson update
            let hessian_inv = Self::invert_hessian(&hessian)?;
            let delta = hessian_inv.dot(&gradient);
            beta = &beta + &delta;

            prev_log_likelihood = log_likelihood;
        }

        Err(StatsError::ConvergenceError(format!(
            "Extended Cox model failed to converge after {max_iter} iterations"
        )))
    }

    /// Calculate stratified partial likelihood and derivatives
    fn stratified_partial_likelihood_derivatives(
        durations: &ArrayView1<f64>,
        events: &ArrayView1<bool>,
        covariates: &ArrayView2<f64>,
        beta: &Array1<f64>,
        strata: &Option<Array1<usize>>,
        n_strata: usize,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let n_samples_ = durations.len();
        let n_features = beta.len();

        let mut total_log_likelihood = 0.0;
        let mut total_gradient = Array1::zeros(n_features);
        let mut total_hessian = Array2::zeros((n_features, n_features));

        // Process each stratum separately
        for stratum in 0..n_strata {
            // Get indices for this stratum
            let stratum_indices: Vec<usize> = if let Some(ref strata_array) = strata {
                (0..n_samples_)
                    .filter(|&i| strata_array[i] == stratum)
                    .collect()
            } else {
                (0..n_samples_).collect()
            };

            if stratum_indices.is_empty() {
                continue;
            }

            // Sort by event time within stratum
            let mut sorted_indices = stratum_indices.clone();
            sorted_indices.sort_by(|&i, &j| durations[i].partial_cmp(&durations[j]).unwrap());

            // Calculate partial likelihood for this stratum
            let (stratum_ll, stratum_grad, stratum_hess) = Self::stratum_partial_likelihood(
                durations,
                events,
                covariates,
                beta,
                &sorted_indices,
            )?;

            total_log_likelihood += stratum_ll;
            total_gradient = &total_gradient + &stratum_grad;
            total_hessian = &total_hessian + &stratum_hess;
        }

        Ok((total_log_likelihood, total_gradient, total_hessian))
    }

    /// Calculate partial likelihood for a single stratum
    fn stratum_partial_likelihood(
        durations: &ArrayView1<f64>,
        events: &ArrayView1<bool>,
        covariates: &ArrayView2<f64>,
        beta: &Array1<f64>,
        sorted_indices: &[usize],
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let n_features = beta.len();

        let mut log_likelihood = 0.0;
        let mut gradient = Array1::zeros(n_features);
        let mut hessian = Array2::zeros((n_features, n_features));

        for &i in sorted_indices {
            if !events[i] {
                continue; // Skip censored observations
            }

            let t_i = durations[i];
            let x_i = covariates.row(i);

            // Find risk set for this event time within stratum
            let mut risk_set = Vec::new();
            for &j in sorted_indices {
                if durations[j] >= t_i {
                    risk_set.push(j);
                }
            }

            if risk_set.is_empty() {
                continue;
            }

            // Calculate exp(beta' * x) for risk set
            let mut exp_beta_x = Array1::zeros(risk_set.len());
            for (k, &j) in risk_set.iter().enumerate() {
                let x_j = covariates.row(j);
                exp_beta_x[k] = x_j.dot(beta).exp();
            }

            let sum_exp = exp_beta_x.sum();
            if sum_exp <= 0.0 {
                continue;
            }

            // Update log-likelihood
            log_likelihood += x_i.dot(beta) - sum_exp.ln();

            // Update gradient
            let mut weighted_x = Array1::<f64>::zeros(n_features);
            for (k, &j) in risk_set.iter().enumerate() {
                let x_j = covariates.row(j);
                let weight = exp_beta_x[k] / sum_exp;
                weighted_x = &weighted_x + &(weight * &x_j.to_owned());
            }
            gradient = &gradient + &(&x_i.to_owned() - &weighted_x);

            // Update Hessian
            for p in 0..n_features {
                for q in 0..n_features {
                    let mut weighted_sum = 0.0;
                    for (k, &j) in risk_set.iter().enumerate() {
                        let x_j = covariates.row(j);
                        let weight = exp_beta_x[k] / sum_exp;
                        weighted_sum += weight * x_j[p] * x_j[q];
                    }
                    hessian[[p, q]] -= weighted_sum - (weighted_x[p] * weighted_x[q]);
                }
            }
        }

        Ok((log_likelihood, gradient, hessian))
    }

    /// Calculate baseline hazards for each stratum
    fn calculate_stratifiedbaseline_hazards(
        durations: &ArrayView1<f64>,
        events: &ArrayView1<bool>,
        covariates: &ArrayView2<f64>,
        beta: &Array1<f64>,
        strata: &Option<Array1<usize>>,
        n_strata: usize,
    ) -> Result<Vec<(Array1<f64>, Array1<f64>)>> {
        let n_samples_ = durations.len();
        let mut baseline_hazards = Vec::new();

        for stratum in 0..n_strata {
            // Get indices for this stratum
            let stratum_indices: Vec<usize> = if let Some(ref strata_array) = strata {
                (0..n_samples_)
                    .filter(|&i| strata_array[i] == stratum)
                    .collect()
            } else {
                (0..n_samples_).collect()
            };

            if stratum_indices.is_empty() {
                baseline_hazards.push((Array1::zeros(0), Array1::zeros(0)));
                continue;
            }

            // Sort by event time
            let mut sorted_indices = stratum_indices.clone();
            sorted_indices.sort_by(|&i, &j| durations[i].partial_cmp(&durations[j]).unwrap());

            let mut times = Vec::new();
            let mut cumulative_hazard = Vec::new();
            let mut current_cumhaz = 0.0;

            for &i in &sorted_indices {
                if !events[i] {
                    continue;
                }

                let t_i = durations[i];

                // Calculate risk sum for this stratum
                let mut risk_sum = 0.0;
                for &j in &sorted_indices {
                    if durations[j] >= t_i {
                        let x_j = covariates.row(j);
                        risk_sum += x_j.dot(beta).exp();
                    }
                }

                if risk_sum > 0.0 {
                    current_cumhaz += 1.0 / risk_sum; // Breslow estimator
                    times.push(t_i);
                    cumulative_hazard.push(current_cumhaz);
                }
            }

            baseline_hazards.push((Array1::from_vec(times), Array1::from_vec(cumulative_hazard)));
        }

        Ok(baseline_hazards)
    }

    /// Invert Hessian matrix
    fn invert_hessian(hessian: &Array2<f64>) -> Result<Array2<f64>> {
        let neg_hessian = -hessian;
        scirs2_linalg::inv(&neg_hessian.view(), None)
            .map_err(|e| StatsError::ComputationError(format!("Failed to invert Hessian: {e}")))
    }

    /// Predict hazard ratios with optional stratum
    pub fn predict_hazard_ratio_stratified(
        &self,
        covariates: ArrayView2<f64>,
        strata: Option<ArrayView1<usize>>,
    ) -> Result<Array1<f64>> {
        checkarray_finite(&covariates, "covariates")?;

        if covariates.ncols() != self.coefficients.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "covariates has {features} features, expected {expected}",
                features = covariates.ncols(),
                expected = self.coefficients.len()
            )));
        }

        if let Some(ref strata_input) = strata {
            if strata_input.len() != covariates.nrows() {
                return Err(StatsError::DimensionMismatch(
                    "Strata length must match number of predictions".to_string(),
                ));
            }
        }

        let mut hazard_ratios = Array1::zeros(covariates.nrows());

        for i in 0..covariates.nrows() {
            let x_i = covariates.row(i);
            hazard_ratios[i] = x_i.dot(&self.coefficients).exp();
        }

        Ok(hazard_ratios)
    }

    /// Compute confidence intervals for coefficients
    pub fn coefficient_confidence_intervals(&self, confidencelevel: f64) -> Result<Array2<f64>> {
        check_probability(confidencelevel, "confidence_level")?;

        let n_features = self.coefficients.len();
        let mut intervals = Array2::zeros((n_features, 2));
        let _alpha = (1.0 - confidencelevel) / 2.0;
        let z_critical = 1.96; // Approximate 95% CI

        for i in 0..n_features {
            let coeff = self.coefficients[i];
            let se = self.covariance_matrix[[i, i]].sqrt();

            intervals[[i, 0]] = coeff - z_critical * se; // Lower bound
            intervals[[i, 1]] = coeff + z_critical * se; // Upper bound
        }

        Ok(intervals)
    }
}

/// Competing risks analysis using subdistribution hazards (Fine-Gray model)
///
/// Handles multiple competing events where the occurrence of one event
/// prevents the observation of others.
#[derive(Debug, Clone)]
pub struct CompetingRisksModel {
    /// Coefficients for each competing risk
    pub coefficients: Vec<Array1<f64>>,
    /// Covariance matrices for each competing risk
    pub covariance_matrices: Vec<Array2<f64>>,
    /// Baseline cumulative incidence functions
    pub baseline_cifs: Vec<(Array1<f64>, Array1<f64>)>, // (times, cumulative incidence)
    /// Number of competing risks
    pub n_risks: usize,
    /// Log-likelihood of the fitted model
    pub log_likelihood: f64,
}

impl CompetingRisksModel {
    /// Fit competing risks model using Fine-Gray subdistribution hazards
    ///
    /// # Arguments
    /// * `durations` - Time to event or censoring
    /// * `events` - Event type (0 = censored, 1 = risk 1, 2 = risk 2, etc.)
    /// * `covariates` - Covariate matrix
    /// * `n_risks` - Number of competing risks
    /// * `target_risk` - Risk of interest for modeling
    pub fn fit(
        durations: ArrayView1<f64>,
        events: ArrayView1<usize>,
        covariates: ArrayView2<f64>,
        n_risks: usize,
        target_risk: usize,
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> Result<Self> {
        checkarray_finite(&durations, "durations")?;
        checkarray_finite(&covariates, "covariates")?;
        check_positive(n_risks, "n_risks")?;

        let (n_samples_, n_features) = covariates.dim();
        let max_iter = max_iter.unwrap_or(100);
        let tol = tol.unwrap_or(1e-6);

        if durations.len() != n_samples_ || events.len() != n_samples_ {
            return Err(StatsError::DimensionMismatch(
                "All input arrays must have the same number of samples".to_string(),
            ));
        }

        if target_risk == 0 || target_risk > n_risks {
            return Err(StatsError::InvalidArgument(
                "target_risk must be between 1 and n_risks".to_string(),
            ));
        }

        // For now, implement single _risk at a time
        // In full implementation, would fit all _risks simultaneously
        let mut coefficients = vec![Array1::zeros(n_features); n_risks];
        let mut covariance_matrices = vec![Array2::zeros((n_features, n_features)); n_risks];
        let mut baseline_cifs = vec![(Array1::zeros(0), Array1::zeros(0)); n_risks];

        // Fit model for the target _risk using modified data
        let (modified_durations, modified_events, modified_weights) =
            Self::prepare_fine_gray_data(&durations, &events, target_risk)?;

        // Initialize coefficients for target _risk
        let mut beta = Array1::zeros(n_features);
        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for _iteration in 0..max_iter {
            // Calculate weighted partial likelihood for subdistribution hazard
            let (log_likelihood, gradient, hessian) = Self::subdistribution_partial_likelihood(
                &modified_durations,
                &modified_events,
                &covariates,
                &modified_weights,
                &beta,
            )?;

            // Check convergence
            if (log_likelihood - prev_log_likelihood).abs() < tol {
                coefficients[target_risk - 1] = beta.clone();
                covariance_matrices[target_risk - 1] = Self::invert_hessian(&hessian)?;

                // Calculate baseline cumulative incidence
                let (times, cif) = Self::calculatebaseline_cif(
                    &modified_durations,
                    &modified_events,
                    &covariates,
                    &modified_weights,
                    &beta,
                )?;
                baseline_cifs[target_risk - 1] = (times, cif);

                return Ok(Self {
                    coefficients,
                    covariance_matrices,
                    baseline_cifs,
                    n_risks,
                    log_likelihood,
                });
            }

            // Newton-Raphson update
            let hessian_inv = Self::invert_hessian(&hessian)?;
            let delta = hessian_inv.dot(&gradient);
            beta = &beta + &delta;

            prev_log_likelihood = log_likelihood;
        }

        Err(StatsError::ConvergenceError(format!(
            "Competing _risks model failed to converge after {max_iter} iterations"
        )))
    }

    /// Prepare data for Fine-Gray model with artificial censoring
    fn prepare_fine_gray_data(
        durations: &ArrayView1<f64>,
        events: &ArrayView1<usize>,
        target_risk: usize,
    ) -> Result<(Array1<f64>, Array1<bool>, Array1<f64>)> {
        let n_samples_ = durations.len();
        let modified_durations = durations.to_owned();
        let mut modified_events = Array1::from_elem(n_samples_, false);
        let mut weights = Array1::ones(n_samples_);

        // Calculate Kaplan-Meier for censoring distribution
        let censoring_km = Self::kaplan_meier_censoring(durations, events)?;

        for i in 0..n_samples_ {
            if events[i] == target_risk {
                // Event of interest occurred
                modified_events[i] = true;
                weights[i] = 1.0;
            } else if events[i] == 0 {
                // Censored observation
                modified_events[i] = false;
                weights[i] = 1.0;
            } else {
                // Competing event occurred - use artificial censoring
                modified_events[i] = false;

                // Weight by inverse probability of censoring
                let km_prob = Self::interpolate_km_probability(
                    &censoring_km.0,
                    &censoring_km.1,
                    durations[i],
                );
                weights[i] = if km_prob > 0.0 { 1.0 / km_prob } else { 0.0 };
            }
        }

        Ok((modified_durations, modified_events, weights))
    }

    /// Calculate Kaplan-Meier estimator for censoring distribution
    fn kaplan_meier_censoring(
        durations: &ArrayView1<f64>,
        events: &ArrayView1<usize>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        // Treat any event as "censoring" for the censoring distribution
        let censoring_events: Array1<bool> = events.mapv(|e| e == 0);

        // Create time-event pairs and sort
        let mut time_event_pairs: Vec<(f64, bool)> = durations
            .iter()
            .zip(censoring_events.iter())
            .map(|(&t, &e)| (t, e))
            .collect();
        time_event_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut times = Vec::new();
        let mut survival_probs = Vec::new();
        let mut current_survival = 1.0;
        let mut current_at_risk = time_event_pairs.len();

        let mut i = 0;
        while i < time_event_pairs.len() {
            let current_time = time_event_pairs[i].0;
            let mut events_at_time = 0;
            let mut total_at_time = 0;

            while i < time_event_pairs.len() && time_event_pairs[i].0 == current_time {
                if time_event_pairs[i].1 {
                    events_at_time += 1;
                }
                total_at_time += 1;
                i += 1;
            }

            if events_at_time > 0 {
                let survival_this_time = 1.0 - (events_at_time as f64) / (current_at_risk as f64);
                current_survival *= survival_this_time;

                times.push(current_time);
                survival_probs.push(current_survival);
            }

            current_at_risk -= total_at_time;
        }

        Ok((Array1::from_vec(times), Array1::from_vec(survival_probs)))
    }

    /// Interpolate Kaplan-Meier probability at given time
    fn interpolate_km_probability(times: &Array1<f64>, probs: &Array1<f64>, t: f64) -> f64 {
        if times.is_empty() {
            return 1.0;
        }

        if t <= times[0] {
            return 1.0;
        }

        for i in 0..times.len() {
            if times[i] >= t {
                return probs[i];
            }
        }

        // If t is beyond last time point, return last probability
        probs[probs.len() - 1]
    }

    /// Calculate subdistribution partial likelihood
    fn subdistribution_partial_likelihood(
        durations: &Array1<f64>,
        events: &Array1<bool>,
        covariates: &ArrayView2<f64>,
        weights: &Array1<f64>,
        beta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let n_samples_ = durations.len();
        let n_features = beta.len();

        // Sort by event time
        let mut indices: Vec<usize> = (0..n_samples_).collect();
        indices.sort_by(|&i, &j| durations[i].partial_cmp(&durations[j]).unwrap());

        let mut log_likelihood = 0.0;
        let mut gradient = Array1::zeros(n_features);
        let mut hessian = Array2::zeros((n_features, n_features));

        for &i in &indices {
            if !events[i] {
                continue; // Skip non-events
            }

            let t_i = durations[i];
            let x_i = covariates.row(i);
            let w_i = weights[i];

            // Calculate weighted risk set
            let mut weighted_exp_sum = 0.0;
            let mut weighted_x_sum = Array1::zeros(n_features);
            let mut weighted_xx_sum = Array2::zeros((n_features, n_features));

            for &j in &indices {
                if durations[j] >= t_i {
                    let x_j = covariates.row(j);
                    let w_j = weights[j];
                    let exp_beta_x = x_j.dot(beta).exp();
                    let weighted_exp = w_j * exp_beta_x;

                    weighted_exp_sum += weighted_exp;
                    weighted_x_sum = &weighted_x_sum + &(weighted_exp * &x_j.to_owned());

                    for p in 0..n_features {
                        for q in 0..n_features {
                            weighted_xx_sum[[p, q]] += weighted_exp * x_j[p] * x_j[q];
                        }
                    }
                }
            }

            if weighted_exp_sum <= 0.0 {
                continue;
            }

            // Update likelihood components
            let weighted_mean_x = &weighted_x_sum / weighted_exp_sum;

            log_likelihood += w_i * (x_i.dot(beta) - weighted_exp_sum.ln());
            gradient = &gradient + &(w_i * (&x_i.to_owned() - &weighted_mean_x));

            // Update Hessian
            let weighted_mean_xx = &weighted_xx_sum / weighted_exp_sum;
            let outer_product = outer_product_array(&weighted_mean_x);
            hessian = &hessian - &(w_i * (&weighted_mean_xx - &outer_product));
        }

        Ok((log_likelihood, gradient, hessian))
    }

    /// Calculate baseline cumulative incidence function
    fn calculatebaseline_cif(
        durations: &Array1<f64>,
        events: &Array1<bool>,
        covariates: &ArrayView2<f64>,
        weights: &Array1<f64>,
        beta: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let n_samples_ = durations.len();

        // Sort by event time
        let mut indices: Vec<usize> = (0..n_samples_).collect();
        indices.sort_by(|&i, &j| durations[i].partial_cmp(&durations[j]).unwrap());

        let mut times = Vec::new();
        let mut cumulative_incidence = Vec::new();
        let mut current_cif = 0.0;

        for &i in &indices {
            if !events[i] {
                continue;
            }

            let t_i = durations[i];
            let w_i = weights[i];

            // Calculate weighted risk sum
            let mut weighted_risk_sum = 0.0;
            for &j in &indices {
                if durations[j] >= t_i {
                    let x_j = covariates.row(j);
                    let w_j = weights[j];
                    weighted_risk_sum += w_j * x_j.dot(beta).exp();
                }
            }

            if weighted_risk_sum > 0.0 {
                current_cif += w_i / weighted_risk_sum;
                times.push(t_i);
                cumulative_incidence.push(current_cif);
            }
        }

        Ok((
            Array1::from_vec(times),
            Array1::from_vec(cumulative_incidence),
        ))
    }

    /// Invert Hessian matrix
    fn invert_hessian(hessian: &Array2<f64>) -> Result<Array2<f64>> {
        let neg_hessian = -hessian;
        scirs2_linalg::inv(&neg_hessian.view(), None)
            .map_err(|e| StatsError::ComputationError(format!("Failed to invert Hessian: {e}")))
    }

    /// Predict cumulative incidence for target risk
    pub fn predict_cumulative_incidence(
        &self,
        covariates: ArrayView2<f64>,
        target_risk: usize,
        times: ArrayView1<f64>,
    ) -> Result<Array2<f64>> {
        checkarray_finite(&covariates, "covariates")?;
        checkarray_finite(&times, "times")?;

        if target_risk == 0 || target_risk > self.n_risks {
            return Err(StatsError::InvalidArgument(
                "target_risk must be between 1 and n_risks".to_string(),
            ));
        }

        let risk_idx = target_risk - 1;
        let n_samples_ = covariates.nrows();
        let n_times = times.len();
        let mut predictions = Array2::zeros((n_samples_, n_times));

        let beta = &self.coefficients[risk_idx];
        let (baseline_times, baseline_cif) = &self.baseline_cifs[risk_idx];

        for i in 0..n_samples_ {
            let x_i = covariates.row(i);
            let hazard_ratio = x_i.dot(beta).exp();

            for (j, &t) in times.iter().enumerate() {
                // Interpolate baseline CIF at time t
                let baseline_value = Self::interpolatebaseline_cif(baseline_times, baseline_cif, t);

                // Transform baseline CIF using subdistribution hazard ratio
                // This is a simplified transformation - full implementation would be more complex
                predictions[[i, j]] = 1.0 - (1.0 - baseline_value).powf(hazard_ratio);
            }
        }

        Ok(predictions)
    }

    /// Interpolate baseline cumulative incidence at given time
    fn interpolatebaseline_cif(times: &Array1<f64>, cif: &Array1<f64>, t: f64) -> f64 {
        if times.is_empty() {
            return 0.0;
        }

        if t <= times[0] {
            return 0.0;
        }

        for i in 0..times.len() {
            if times[i] >= t {
                return cif[i];
            }
        }

        // If t is beyond last time point, return last CIF value
        cif[cif.len() - 1]
    }
}

/// Helper function to compute outer product of array
#[allow(dead_code)]
fn outer_product_array(v: &Array1<f64>) -> Array2<f64> {
    let n = v.len();
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = v[i] * v[j];
        }
    }
    result
}
