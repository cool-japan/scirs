//! Confidence interval calculations for trend estimation
//!
//! This module provides functions for calculating confidence intervals around
//! estimated trends, using bootstrap, parametric, and prediction interval methods.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use rand::prelude::*;
use std::fmt::Debug;

use super::{ConfidenceIntervalMethod, ConfidenceIntervalOptions, TrendWithConfidenceInterval};
use crate::error::{Result, TimeSeriesError};

/// Computes confidence intervals for an estimated trend
///
/// This function calculates confidence intervals around an estimated trend using
/// the specified method (bootstrap, parametric, or prediction intervals).
///
/// # Arguments
///
/// * `ts` - The original time series data
/// * `trend` - The estimated trend
/// * `options` - Options controlling the confidence interval calculation
/// * `trend_estimator` - A function that estimates the trend given a time series
///
/// # Returns
///
/// A tuple of (lower_bound, upper_bound) arrays
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_series::trends::{compute_trend_confidence_interval, ConfidenceIntervalOptions, ConfidenceIntervalMethod};
///
/// // Create a sample time series with a trend and noise
/// let n = 100;
/// let xs = Array1::from_vec((0..n).map(|t| t as f64).collect());
/// let trend = Array1::from_vec((0..n).map(|t| (t as f64 / 10.0).sin() + t as f64 / 50.0).collect());
/// let noise = Array1::from_vec((0..n).map(|_| 0.1 * rand::random::<f64>()).collect());
/// let ts = &trend + &noise;
///
/// // Configure confidence interval options
/// let options = ConfidenceIntervalOptions {
///     method: ConfidenceIntervalMethod::Parametric,
///     level: 0.95,
///     ..Default::default()
/// };
///
/// // Calculate confidence intervals using a simple trend estimator
/// let (lower, upper) = compute_trend_confidence_interval(
///     &ts,
///     &trend,
///     &options,
///     |data| Ok(data.clone())  // Identity function as a simple trend estimator
/// ).unwrap();
///
/// // Check the shape of the confidence bounds
/// assert_eq!(lower.len(), ts.len());
/// assert_eq!(upper.len(), ts.len());
/// ```
pub fn compute_trend_confidence_interval<F>(
    ts: &Array1<F>,
    trend: &Array1<F>,
    options: &ConfidenceIntervalOptions,
    trend_estimator: impl Fn(&Array1<F>) -> Result<Array1<F>>,
) -> Result<(Array1<F>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = ts.len();

    if trend.len() != n {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Trend length ({}) must match time series length ({})",
            trend.len(),
            n
        )));
    }

    if options.level <= 0.0 || options.level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Confidence level must be between 0 and 1, got {}",
            options.level
        )));
    }

    match options.method {
        ConfidenceIntervalMethod::Bootstrap => {
            bootstrap_confidence_interval(ts, trend, options, &trend_estimator)
        }
        ConfidenceIntervalMethod::Parametric => parametric_confidence_interval(ts, trend, options),
        ConfidenceIntervalMethod::Prediction => prediction_interval(ts, trend, options),
    }
}

/// Calculates bootstrap confidence intervals
fn bootstrap_confidence_interval<F>(
    ts: &Array1<F>,
    trend: &Array1<F>,
    options: &ConfidenceIntervalOptions,
    trend_estimator: &impl Fn(&Array1<F>) -> Result<Array1<F>>,
) -> Result<(Array1<F>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = ts.len();
    let num_bootstrap = options.num_bootstrap;
    let alpha = F::one() - F::from_f64(options.level).unwrap();
    let lower_percentile = (alpha / F::from_f64(2.0).unwrap() * F::from_usize(100).unwrap())
        .to_f64()
        .unwrap();
    let upper_percentile = ((F::one() - alpha / F::from_f64(2.0).unwrap())
        * F::from_usize(100).unwrap())
    .to_f64()
    .unwrap();

    // Calculate residuals
    let residuals = Array1::from_shape_fn(n, |i| ts[i] - trend[i]);

    // Storage for bootstrap trend estimates
    let mut bootstrap_trends = Vec::with_capacity(num_bootstrap);

    // Use fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate bootstrap samples
    for _ in 0..num_bootstrap {
        // Create bootstrap sample based on type
        let bootstrap_sample = if options.block_bootstrap {
            // Block bootstrap (handles autocorrelation)
            let block_length = options.block_length;
            let num_blocks = n.div_ceil(block_length);

            let mut sample = Array1::<F>::zeros(n);

            for b in 0..num_blocks {
                let start_idx = rng.random_range(0..(n - block_length.min(n) + 1));
                let end_idx = start_idx + block_length.min(n - start_idx);

                for i in 0..end_idx.saturating_sub(start_idx) {
                    let dest_idx = b * block_length + i;
                    if dest_idx < n {
                        sample[dest_idx] = trend[dest_idx] + residuals[start_idx + i];
                    }
                }
            }

            sample
        } else {
            // Standard bootstrap (resamples residuals)
            let mut sample = Array1::<F>::zeros(n);

            for i in 0..n {
                let residual_idx = rng.random_range(0..n);
                sample[i] = trend[i] + residuals[residual_idx];
            }

            sample
        };

        // Estimate trend from bootstrap sample
        let bootstrap_trend = trend_estimator(&bootstrap_sample)?;
        bootstrap_trends.push(bootstrap_trend);
    }

    // Calculate percentiles at each time point
    let mut lower = Array1::<F>::zeros(n);
    let mut upper = Array1::<F>::zeros(n);

    for i in 0..n {
        let mut values = Vec::with_capacity(num_bootstrap);

        for trend in bootstrap_trends.iter().take(num_bootstrap) {
            values.push(trend[i]);
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx = (lower_percentile / 100.0 * (num_bootstrap as f64)).round() as usize;
        let upper_idx = (upper_percentile / 100.0 * (num_bootstrap as f64)).round() as usize;

        lower[i] = values[lower_idx.min(num_bootstrap - 1)];
        upper[i] = values[upper_idx.min(num_bootstrap - 1)];
    }

    Ok((lower, upper))
}

/// Calculates parametric confidence intervals
fn parametric_confidence_interval<F>(
    ts: &Array1<F>,
    trend: &Array1<F>,
    options: &ConfidenceIntervalOptions,
) -> Result<(Array1<F>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let _alpha = F::one() - F::from_f64(options.level).unwrap();

    // Calculate residuals and estimate noise variance
    let residuals = Array1::from_shape_fn(n, |i| ts[i] - trend[i]);

    let noise_variance = if options.estimate_noise_variance {
        let sum_squared = residuals.iter().fold(F::zero(), |acc, &r| acc + r * r);
        let degrees_freedom = F::from_usize(n - 2).unwrap(); // Assuming at least linear model
        sum_squared / degrees_freedom
    } else if let Some(var) = options.noise_variance {
        F::from_f64(var).unwrap()
    } else {
        return Err(TimeSeriesError::InvalidInput(
            "No noise variance provided and estimate_noise_variance is false".to_string(),
        ));
    };

    // Get critical value for the desired confidence level
    // Approximate using normal distribution
    let critical_value = match options.level {
        0.90 => 1.645,
        0.95 => 1.96,
        0.99 => 2.576,
        _ => {
            // Approximate normal quantile for arbitrary confidence level
            let p = (F::one() + F::from_f64(options.level).unwrap()) / F::from_f64(2.0).unwrap();
            normal_quantile(p.to_f64().unwrap())
        }
    };

    let critical_value_f = F::from_f64(critical_value).unwrap();
    let std_error = noise_variance.sqrt();

    // Calculate confidence intervals
    let mut lower = Array1::<F>::zeros(n);
    let mut upper = Array1::<F>::zeros(n);

    for i in 0..n {
        lower[i] = trend[i] - critical_value_f * std_error;
        upper[i] = trend[i] + critical_value_f * std_error;
    }

    Ok((lower, upper))
}

/// Calculates prediction intervals
fn prediction_interval<F>(
    ts: &Array1<F>,
    trend: &Array1<F>,
    options: &ConfidenceIntervalOptions,
) -> Result<(Array1<F>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let _alpha = F::one() - F::from_f64(options.level).unwrap();

    // Calculate residuals and estimate noise variance
    let residuals = Array1::from_shape_fn(n, |i| ts[i] - trend[i]);

    let noise_variance = if options.estimate_noise_variance {
        let sum_squared = residuals.iter().fold(F::zero(), |acc, &r| acc + r * r);
        let degrees_freedom = F::from_usize(n - 2).unwrap(); // Assuming at least linear model
        sum_squared / degrees_freedom
    } else if let Some(var) = options.noise_variance {
        F::from_f64(var).unwrap()
    } else {
        return Err(TimeSeriesError::InvalidInput(
            "No noise variance provided and estimate_noise_variance is false".to_string(),
        ));
    };

    // Get critical value for the desired confidence level
    // Approximate using normal distribution
    let critical_value = match options.level {
        0.90 => 1.645,
        0.95 => 1.96,
        0.99 => 2.576,
        _ => {
            // Approximate normal quantile for arbitrary confidence level
            let p = (F::one() + F::from_f64(options.level).unwrap()) / F::from_f64(2.0).unwrap();
            normal_quantile(p.to_f64().unwrap())
        }
    };

    let critical_value_f = F::from_f64(critical_value).unwrap();

    // Additional variance components for prediction interval
    let model_uncertainty = F::from_f64(0.1).unwrap() * noise_variance; // Approximate model uncertainty

    let prediction_error = if options.include_observation_noise {
        (noise_variance + model_uncertainty).sqrt()
    } else {
        model_uncertainty.sqrt()
    };

    // Calculate prediction intervals
    let mut lower = Array1::<F>::zeros(n);
    let mut upper = Array1::<F>::zeros(n);

    for i in 0..n {
        lower[i] = trend[i] - critical_value_f * prediction_error;
        upper[i] = trend[i] + critical_value_f * prediction_error;
    }

    Ok((lower, upper))
}

/// Approximation of the normal quantile function
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        panic!("Probability must be between 0 and 1");
    }

    // Constants for Beasley-Springer-Moro algorithm
    let a = [
        2.50662823884,
        -18.61500062529,
        41.39119773534,
        -25.44106049637,
    ];
    let b = [
        -8.47351093090,
        23.08336743743,
        -21.06224101826,
        3.13082909833,
    ];
    let c = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ];

    // Approximation near the center
    if (0.08..=0.92).contains(&p) {
        let q = p - 0.5;
        let r = q * q;
        let mut result = q * (a[0] + r * (a[1] + r * (a[2] + r * a[3])));
        result /= 1.0 + r * (b[0] + r * (b[1] + r * (b[2] + r * b[3])));
        return result;
    }

    // Approximation in the tails
    let q = if p < 0.08 {
        (-2.0 * (p).ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let result = c[0]
        + q * (c[1]
            + q * (c[2]
                + q * (c[3] + q * (c[4] + q * (c[5] + q * (c[6] + q * (c[7] + q * c[8])))))));

    if p < 0.08 {
        -result
    } else {
        result
    }
}

/// Creates a trend estimate along with confidence intervals
///
/// This is a helper function to wrap the main computation of trend and confidence intervals
/// into a single function call, for convenience.
///
/// # Arguments
///
/// * `ts` - The input time series data
/// * `trend_estimator` - A function that estimates the trend given a time series
/// * `options` - Options controlling trend estimation
/// * `ci_options` - Options controlling confidence interval calculation
///
/// # Returns
///
/// A `TrendWithConfidenceInterval` struct containing the estimated trend and confidence bounds
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_series::trends::{
///     create_trend_with_ci, SplineTrendOptions, ConfidenceIntervalOptions,
///     SplineType, KnotPlacementStrategy, ConfidenceIntervalMethod,
///     estimate_spline_trend
/// };
///
/// // Create a sample time series
/// let n = 100;
/// let ts = Array1::from_vec((0..n).map(|t| (t as f64 / 10.0).sin() + t as f64 / 50.0 + 0.1 * rand::random::<f64>()).collect());
///
/// // Configure options
/// let trend_options = SplineTrendOptions {
///     spline_type: SplineType::Cubic,
///     num_knots: 10,
///     knot_placement: KnotPlacementStrategy::Uniform,
///     ..Default::default()
/// };
///
/// let ci_options = ConfidenceIntervalOptions {
///     method: ConfidenceIntervalMethod::Parametric,
///     level: 0.95,
///     ..Default::default()
/// };
///
/// // Create trend with confidence intervals
/// let result = create_trend_with_ci(
///     &ts,
///     |data| estimate_spline_trend(data, &trend_options),
///     &ci_options
/// ).unwrap();
///
/// assert_eq!(result.trend.len(), ts.len());
/// assert_eq!(result.lower.len(), ts.len());
/// assert_eq!(result.upper.len(), ts.len());
/// ```
pub fn create_trend_with_ci<F, E>(
    ts: &Array1<F>,
    trend_estimator: E,
    ci_options: &ConfidenceIntervalOptions,
) -> Result<TrendWithConfidenceInterval<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
    E: Fn(&Array1<F>) -> Result<Array1<F>>,
{
    // First, compute the main trend estimate
    let trend = trend_estimator(ts)?;

    // Then compute confidence intervals
    let (lower, upper) =
        compute_trend_confidence_interval(ts, &trend, ci_options, &trend_estimator)?;

    Ok(TrendWithConfidenceInterval {
        trend,
        lower,
        upper,
    })
}
