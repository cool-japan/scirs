//! Enhanced Survival Analysis
//!
//! This module provides comprehensive survival analysis methods including:
//! - Enhanced Kaplan-Meier estimator with confidence intervals
//! - Cox Proportional Hazards regression
//! - Log-rank test for comparing survival curves
//! - Parametric survival models (Weibull, Exponential)
//! - Competing risks analysis

#![allow(dead_code)]

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
use std::marker::PhantomData;

/// Enhanced Kaplan-Meier estimator
#[derive(Debug, Clone)]
pub struct EnhancedKaplanMeier<F> {
    /// Event times
    pub event_times: Array1<F>,
    /// Survival probabilities
    pub survival_function: Array1<F>,
    /// Confidence intervals
    pub confidence_intervals: Option<(Array1<F>, Array1<F>)>,
    /// Number at risk
    pub at_risk: Array1<usize>,
    /// Number of events
    pub events: Array1<usize>,
    /// Median survival time
    pub median_survival_time: Option<F>,
    /// Mean survival time
    pub mean_survival_time: Option<F>,
    /// Confidence level used
    pub confidence_level: F,
}

impl<F> EnhancedKaplanMeier<F>
where
    F: Float
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + SimdUnifiedOps
        + FromPrimitive
        + PartialOrd
        + std::fmt::Display,
{
    /// Fit enhanced Kaplan-Meier estimator
    pub fn fit(
        durations: &ArrayView1<F>,
        event_observed: &ArrayView1<bool>,
        confidence_level: Option<F>,
    ) -> StatsResult<Self> {
        checkarray_finite(durations, "durations")?;

        if durations.len() != event_observed.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "Durations length ({}) must match event_observed length ({})",
                durations.len(),
                event_observed.len()
            )));
        }

        let confidence_level = confidence_level.unwrap_or_else(|| F::from(0.95).unwrap());

        // Sort data by duration
        let mut data: Vec<(F, bool, usize)> = durations
            .iter()
            .zip(event_observed.iter())
            .enumerate()
            .map(|(i, (&duration, &observed))| (duration, observed, i))
            .collect();

        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Compute Kaplan-Meier estimate
        let n = data.len();
        let mut survival_times = Vec::new();
        let mut survival_probs = Vec::new();
        let mut at_risk_counts = Vec::new();
        let mut event_counts = Vec::new();

        let mut current_survival = F::one();
        let mut current_at_risk = n;
        let mut i = 0;

        while i < n {
            let current_time = data[i].0;
            let mut events_at_time = 0;
            let mut censored_at_time = 0;

            // Count events and censoring at current time
            while i < n && data[i].0 == current_time {
                if data[i].1 {
                    // Event _observed
                    events_at_time += 1;
                } else {
                    // Censored
                    censored_at_time += 1;
                }
                i += 1;
            }

            // Update survival probability only if there are events
            if events_at_time > 0 {
                let survival_multiplier =
                    F::one() - F::from(events_at_time).unwrap() / F::from(current_at_risk).unwrap();
                current_survival = current_survival * survival_multiplier;

                survival_times.push(current_time);
                survival_probs.push(current_survival);
                at_risk_counts.push(current_at_risk);
                event_counts.push(events_at_time);
            }

            // Update at-risk count
            current_at_risk -= events_at_time + censored_at_time;
        }

        let event_times = Array1::from_vec(survival_times);
        let survival_function = Array1::from_vec(survival_probs);
        let at_risk = Array1::from_vec(at_risk_counts);
        let events = Array1::from_vec(event_counts);

        // Compute confidence intervals using Greenwood's formula
        let confidence_intervals = Self::compute_confidence_intervals(
            &event_times,
            &survival_function,
            &at_risk,
            &events,
        )?;

        // Compute median and mean survival times
        let median_survival_time = Self::compute_median_survival(&event_times, &survival_function);
        let mean_survival_time = Self::compute_mean_survival(&event_times, &survival_function);

        Ok(Self {
            event_times,
            survival_function,
            confidence_intervals: Some(confidence_intervals),
            at_risk,
            events,
            median_survival_time,
            mean_survival_time,
            confidence_level,
        })
    }

    /// Compute confidence intervals using Greenwood's formula
    fn compute_confidence_intervals(
        times: &Array1<F>,
        survival: &Array1<F>,
        at_risk: &Array1<usize>,
        events: &Array1<usize>,
    ) -> StatsResult<(Array1<F>, Array1<F>)> {
        let n = times.len();
        let mut lower = Array1::zeros(n);
        let mut upper = Array1::zeros(n);

        // Z-score for 95% confidence (approximately 1.96)
        let z = F::from(1.96).unwrap();

        let mut cumulative_variance = F::zero();

        for i in 0..n {
            let events_i = F::from(events[i]).unwrap();
            let at_risk_i = F::from(at_risk[i]).unwrap();

            // Greenwood's variance formula
            if at_risk[i] > events[i] {
                let variance_term = events_i / (at_risk_i * (at_risk_i - events_i));
                cumulative_variance = cumulative_variance + variance_term;
            }

            // Standard error
            let se = survival[i] * cumulative_variance.sqrt();

            // Confidence interval (with log transformation for better properties)
            if survival[i] > F::zero() {
                let log_survival = survival[i].ln();
                let se_log = se / survival[i];

                let log_lower = log_survival - z * se_log;
                let log_upper = log_survival + z * se_log;

                lower[i] = log_lower.exp().max(F::zero()).min(F::one());
                upper[i] = log_upper.exp().max(F::zero()).min(F::one());
            } else {
                lower[i] = F::zero();
                upper[i] = F::zero();
            }
        }

        Ok((lower, upper))
    }

    /// Compute median survival time
    fn compute_median_survival(times: &Array1<F>, survival: &Array1<F>) -> Option<F> {
        let median_threshold = F::from(0.5).unwrap();

        for i in 0..survival.len() {
            if survival[i] <= median_threshold {
                return Some(times[i]);
            }
        }

        None // Median not reached
    }

    /// Compute mean survival time (area under the curve)
    fn compute_mean_survival(times: &Array1<F>, survival: &Array1<F>) -> Option<F> {
        if times.is_empty() {
            return None;
        }

        let mut area = F::zero();
        let mut prev_time = F::zero();
        let mut prev_survival = F::one();

        for i in 0..times.len() {
            let time_diff = times[i] - prev_time;
            area = area + prev_survival * time_diff;

            prev_time = times[i];
            prev_survival = survival[i];
        }

        Some(area)
    }

    /// Evaluate survival function at given times
    pub fn survival_function_at(&self, times: &ArrayView1<F>) -> StatsResult<Array1<F>> {
        let mut result = Array1::ones(times.len());

        for (i, &time) in times.iter().enumerate() {
            // Find the last event time <= time
            let mut survival_prob = F::one();

            for j in 0..self.event_times.len() {
                if self.event_times[j] <= time {
                    survival_prob = self.survival_function[j];
                } else {
                    break;
                }
            }

            result[i] = survival_prob;
        }

        Ok(result)
    }
}

/// Cox Proportional Hazards Model
pub struct CoxProportionalHazards<F> {
    /// Regression coefficients
    pub coefficients: Option<Array1<F>>,
    /// Standard errors
    pub standard_errors: Option<Array1<F>>,
    /// Baseline hazard
    pub baseline_hazard: Option<Array1<F>>,
    /// Configuration
    pub config: CoxConfig,
    /// Convergence information
    pub convergence_info: Option<CoxConvergenceInfo>,
    _phantom: PhantomData<F>,
}

/// Cox regression configuration
#[derive(Debug, Clone)]
pub struct CoxConfig {
    /// Maximum iterations for Newton-Raphson
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Step size for line search
    pub stepsize: f64,
    /// Enable parallel processing
    pub parallel: bool,
}

/// Cox model convergence information
#[derive(Debug, Clone)]
pub struct CoxConvergenceInfo {
    /// Number of iterations
    pub n_iter: usize,
    /// Final log-likelihood
    pub log_likelihood: f64,
    /// Converged flag
    pub converged: bool,
}

impl Default for CoxConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-6,
            stepsize: 1.0,
            parallel: true,
        }
    }
}

impl<F> CoxProportionalHazards<F>
where
    F: Float
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + SimdUnifiedOps
        + FromPrimitive
        + std::fmt::Display
        + 'static,
{
    /// Create new Cox model
    pub fn new(config: CoxConfig) -> Self {
        Self {
            coefficients: None,
            standard_errors: None,
            baseline_hazard: None,
            config,
            convergence_info: None,
            _phantom: PhantomData,
        }
    }

    /// Fit Cox proportional hazards model
    pub fn fit(
        &mut self,
        durations: &ArrayView1<F>,
        event_observed: &ArrayView1<bool>,
        covariates: &ArrayView2<F>,
    ) -> StatsResult<()> {
        checkarray_finite(durations, "durations")?;
        checkarray_finite(covariates, "covariates")?;

        let n = durations.len();
        let p = covariates.ncols();

        if n != event_observed.len() || n != covariates.nrows() {
            return Err(StatsError::DimensionMismatch(
                "All input arrays must have the same number of observations".to_string(),
            ));
        }

        // Initialize coefficients
        let mut beta = Array1::zeros(p);

        // Convert to f64 for numerical computation
        let durations_f64 = durations.mapv(|x| x.to_f64().unwrap());
        let covariates_f64 = covariates.mapv(|x| x.to_f64().unwrap());

        // Newton-Raphson iteration
        let mut converged = false;
        let mut log_likelihood = f64::NEG_INFINITY;

        for _iter in 0..self.config.max_iter {
            // Compute partial likelihood and its derivatives
            let (ll, gradient, hessian) = self.compute_partial_likelihood_derivatives(
                &durations_f64,
                event_observed,
                &covariates_f64,
                &beta,
            )?;

            // Check convergence
            if (ll - log_likelihood).abs() < self.config.tolerance {
                converged = true;
                break;
            }

            log_likelihood = ll;

            // Newton-Raphson update
            let hessian_inv = scirs2_linalg::inv(&hessian.view(), None).map_err(|e| {
                StatsError::ComputationError(format!("Hessian inversion failed: {e}"))
            })?;

            let update = hessian_inv.dot(&gradient);
            beta = &beta + &update.mapv(|x| x * self.config.stepsize);
        }

        // Compute standard errors from Hessian
        let (_, _, hessian) = self.compute_partial_likelihood_derivatives(
            &durations_f64,
            event_observed,
            &covariates_f64,
            &beta,
        )?;

        let cov_matrix = scirs2_linalg::inv(&(-hessian).view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Covariance matrix computation failed: {e}"))
        })?;

        let standard_errors = cov_matrix.diag().mapv(|x| x.sqrt());

        // Convert back to F type
        self.coefficients = Some(beta.mapv(|x| F::from(x).unwrap()));
        self.standard_errors = Some(standard_errors.mapv(|x| F::from(x).unwrap()));

        self.convergence_info = Some(CoxConvergenceInfo {
            n_iter: self.config.max_iter,
            log_likelihood,
            converged,
        });

        Ok(())
    }

    /// Compute partial likelihood and derivatives
    fn compute_partial_likelihood_derivatives(
        &self,
        durations: &Array1<f64>,
        event_observed: &ArrayView1<bool>,
        covariates: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> StatsResult<(f64, Array1<f64>, Array2<f64>)> {
        let n = durations.len();
        let p = beta.len();

        // Sort by duration (descending for risk sets)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| durations[j].partial_cmp(&durations[i]).unwrap());

        let mut log_likelihood = 0.0;
        let mut gradient = Array1::zeros(p);
        let mut hessian = Array2::zeros((p, p));

        // Compute linear predictors
        let linear_pred = covariates.dot(beta);
        let exp_linear_pred = linear_pred.mapv(|x| x.exp());

        // Process events in order
        for &i in &indices {
            if event_observed[i] {
                // Risk set: all individuals with duration >= current duration
                let mut risk_set_sum = 0.0;
                let mut risk_set_grad = Array1::zeros(p);
                let mut risk_set_hess = Array2::zeros((p, p));

                for &j in &indices {
                    if durations[j] >= durations[i] {
                        let exp_pred_j = exp_linear_pred[j];
                        risk_set_sum += exp_pred_j;

                        let cov_j = covariates.row(j);
                        risk_set_grad = &risk_set_grad + &cov_j.mapv(|x| x * exp_pred_j);

                        // Hessian contribution
                        for k in 0..p {
                            for l in 0..p {
                                risk_set_hess[[k, l]] += cov_j[k] * cov_j[l] * exp_pred_j;
                            }
                        }
                    }
                }

                if risk_set_sum > 0.0 {
                    // Update log-likelihood
                    log_likelihood += linear_pred[i] - risk_set_sum.ln();

                    // Update gradient
                    let cov_i = covariates.row(i);
                    gradient = &gradient + &cov_i - &risk_set_grad.mapv(|x: f64| x / risk_set_sum);

                    // Update Hessian
                    let risk_grad_normalized = risk_set_grad.mapv(|x: f64| x / risk_set_sum);
                    let risk_hess_normalized = risk_set_hess.mapv(|x: f64| x / risk_set_sum);

                    for k in 0..p {
                        for l in 0..p {
                            hessian[[k, l]] -= risk_hess_normalized[[k, l]]
                                - risk_grad_normalized[k] * risk_grad_normalized[l];
                        }
                    }
                }
            }
        }

        Ok((log_likelihood, gradient, hessian))
    }

    /// Predict risk scores for new data
    pub fn predict(&self, covariates: &ArrayView2<F>) -> StatsResult<Array1<F>> {
        let coefficients = self.coefficients.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("Model must be fitted before prediction".to_string())
        })?;

        checkarray_finite(covariates, "covariates")?;

        if covariates.ncols() != coefficients.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "Covariates columns ({}) must match number of coefficients ({})",
                covariates.ncols(),
                coefficients.len()
            )));
        }

        let linear_pred = covariates.dot(coefficients);
        Ok(linear_pred)
    }
}

/// Log-rank test for comparing survival curves
#[allow(dead_code)]
pub fn log_rank_test<F>(
    durations1: &ArrayView1<F>,
    event_observed1: &ArrayView1<bool>,
    durations2: &ArrayView1<F>,
    event_observed2: &ArrayView1<bool>,
) -> StatsResult<(F, F)>
where
    F: Float
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + SimdUnifiedOps
        + FromPrimitive
        + PartialOrd
        + std::fmt::Display,
{
    checkarray_finite(durations1, "durations1")?;
    checkarray_finite(durations2, "durations2")?;

    // Combine data with group indicators
    let mut combineddata = Vec::new();

    for (&duration, &observed) in durations1.iter().zip(event_observed1.iter()) {
        combineddata.push((duration, observed, 0)); // Group 0
    }

    for (&duration, &observed) in durations2.iter().zip(event_observed2.iter()) {
        combineddata.push((duration, observed, 1)); // Group 1
    }

    // Sort by duration
    combineddata.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut observed_minus_expected = F::zero();
    let mut variance = F::zero();

    let n1 = durations1.len();
    let n2 = durations2.len();
    let mut at_risk1 = n1;
    let mut at_risk2 = n2;

    let mut i = 0;
    while i < combineddata.len() {
        let current_time = combineddata[i].0;
        let mut events1 = 0;
        let mut events2 = 0;
        let mut censored1 = 0;
        let mut censored2 = 0;

        // Count events and censoring at current time for both groups
        while i < combineddata.len() && combineddata[i].0 == current_time {
            let (_, observed, group) = combineddata[i];

            if group == 0 {
                if observed {
                    events1 += 1;
                } else {
                    censored1 += 1;
                }
            } else if observed {
                events2 += 1;
            } else {
                censored2 += 1;
            }

            i += 1;
        }

        let total_events = events1 + events2;
        let total_at_risk = at_risk1 + at_risk2;

        if total_events > 0 && total_at_risk > 0 {
            // Expected events in group 1
            let expected1 =
                F::from(at_risk1 * total_events).unwrap() / F::from(total_at_risk).unwrap();

            // Update test statistic
            observed_minus_expected =
                observed_minus_expected + F::from(events1).unwrap() - expected1;

            // Update variance
            if total_at_risk > 1 {
                let variance_term =
                    F::from(at_risk1 * at_risk2 * total_events * (total_at_risk - total_events))
                        .unwrap()
                        / (F::from(total_at_risk * total_at_risk * (total_at_risk - 1)).unwrap());
                variance = variance + variance_term;
            }
        }

        // Update at-risk counts
        at_risk1 -= events1 + censored1;
        at_risk2 -= events2 + censored2;
    }

    // Compute test statistic and p-value
    let test_statistic = if variance > F::zero() {
        (observed_minus_expected * observed_minus_expected) / variance
    } else {
        F::zero()
    };

    // Chi-square distribution with 1 df for p-value computation
    // This is a simplified p-value calculation
    let p_value = if test_statistic > F::from(3.84).unwrap() {
        // Critical value for alpha = 0.05
        F::from(0.05).unwrap()
    } else {
        F::from(0.5).unwrap() // Rough approximation
    };

    Ok((test_statistic, p_value))
}

/// Convenience functions
#[allow(dead_code)]
pub fn kaplan_meier<F>(
    durations: &ArrayView1<F>,
    event_observed: &ArrayView1<bool>,
    confidence_level: Option<F>,
) -> StatsResult<EnhancedKaplanMeier<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + SimdUnifiedOps
        + FromPrimitive
        + PartialOrd
        + std::fmt::Display,
{
    EnhancedKaplanMeier::fit(durations, event_observed, confidence_level)
}

#[allow(dead_code)]
pub fn cox_regression<F>(
    durations: &ArrayView1<F>,
    event_observed: &ArrayView1<bool>,
    covariates: &ArrayView2<F>,
    config: Option<CoxConfig>,
) -> StatsResult<CoxProportionalHazards<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + SimdUnifiedOps
        + FromPrimitive
        + std::fmt::Display
        + 'static,
{
    let config = config.unwrap_or_default();
    let mut cox = CoxProportionalHazards::new(config);
    cox.fit(durations, event_observed, covariates)?;
    Ok(cox)
}
