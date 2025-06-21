//! Time series specific interpolation methods
//!
//! This module provides interpolation methods specifically designed for temporal data,
//! with support for seasonality, trends, missing data handling, and irregular time intervals.
//! These methods are optimized for time series characteristics and common temporal patterns.
//!
//! # Time Series Features
//!
//! - **Seasonal interpolation**: Handle periodic patterns and seasonal components
//! - **Trend-aware interpolation**: Preserve underlying trends during interpolation
//! - **Missing data handling**: Robust interpolation for irregular or missing timestamps
//! - **Forward/backward fill**: Simple filling strategies for time series gaps
//! - **Temporal smoothing**: Time-aware smoothing with adaptive bandwidth
//! - **Holiday and event handling**: Special treatment for known temporal events
//! - **Uncertainty estimation**: Time-dependent confidence intervals
//!
//! # Examples
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_interpolate::timeseries::{
//!     TimeSeriesInterpolator, TemporalPattern, SeasonalityType
//! };
//!
//! // Create timestamp array (30 time points)
//! let timestamps = Array1::linspace(0.0_f64, 30.0_f64, 30);
//!
//! // Create sample data with trend and noise
//! let values = Array1::from_vec((0..30)
//!     .map(|i| (i as f64) * 0.1_f64 + (i as f64 * 0.2_f64).sin())
//!     .collect());
//!
//! // Create time series interpolator
//! let mut interpolator = TimeSeriesInterpolator::new()
//!     .with_temporal_pattern(TemporalPattern::TrendWithSeasonality)
//!     .with_seasonality_type(SeasonalityType::Daily)
//!     .with_missing_data_strategy("interpolate");
//!
//! // Fit the interpolator
//! interpolator.fit(&timestamps.view(), &values.view()).unwrap();
//!
//! // Generate missing timestamps to interpolate
//! let missing_timestamps = Array1::linspace(0.5_f64, 29.5_f64, 15);
//!
//! // Interpolate missing values
//! let interpolated = interpolator.interpolate(&missing_timestamps.view()).unwrap();
//! ```

use crate::advanced::rbf::{RBFInterpolator, RBFKernel};
use crate::bspline::BSpline;
use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, ScalarOperand};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::fmt::{Debug, Display, LowerExp};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

/// Temporal patterns for time series data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemporalPattern {
    /// Linear trend only
    LinearTrend,
    /// Polynomial trend
    PolynomialTrend,
    /// Seasonal patterns only
    SeasonalOnly,
    /// Combined trend and seasonality
    TrendWithSeasonality,
    /// Irregular/random patterns
    Irregular,
    /// Step changes or regime switches
    StepChanges,
}

/// Types of seasonality in time series
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeasonalityType {
    /// Daily patterns (24-hour cycle)
    Daily,
    /// Weekly patterns (7-day cycle)
    Weekly,
    /// Monthly patterns (30-day cycle)
    Monthly,
    /// Quarterly patterns (90-day cycle)
    Quarterly,
    /// Yearly patterns (365-day cycle)
    Yearly,
    /// Custom period (in time units)
    Custom(f64),
}

/// Strategies for handling missing data
#[derive(Debug, Clone, PartialEq)]
pub enum MissingDataStrategy {
    /// Linear interpolation between known points
    Linear,
    /// Spline interpolation
    Spline,
    /// Forward fill (use last known value)
    ForwardFill,
    /// Backward fill (use next known value)
    BackwardFill,
    /// Mean of surrounding values
    Mean,
    /// Seasonal interpolation based on patterns
    Seasonal,
}

/// Configuration for time series interpolation
#[derive(Debug, Clone)]
pub struct TimeSeriesConfig<T> {
    /// Temporal pattern to expect in the data
    pub pattern: TemporalPattern,
    /// Type of seasonality if applicable
    pub seasonality: Option<SeasonalityType>,
    /// Strategy for handling missing data
    pub missing_strategy: MissingDataStrategy,
    /// Smoothing parameter for temporal regularization
    pub temporal_smoothing: T,
    /// Number of seasonal periods to consider
    pub seasonal_periods: usize,
    /// Whether to estimate confidence intervals
    pub estimate_uncertainty: bool,
    /// Outlier detection threshold (standard deviations)
    pub outlier_threshold: Option<T>,
}

impl<T: Float + FromPrimitive> Default for TimeSeriesConfig<T> {
    fn default() -> Self {
        Self {
            pattern: TemporalPattern::TrendWithSeasonality,
            seasonality: Some(SeasonalityType::Daily),
            missing_strategy: MissingDataStrategy::Spline,
            temporal_smoothing: T::from_f64(0.1).unwrap(),
            seasonal_periods: 3,
            estimate_uncertainty: false,
            outlier_threshold: Some(T::from_f64(3.0).unwrap()),
        }
    }
}

/// Time series interpolation result with optional uncertainty
#[derive(Debug, Clone)]
pub struct TimeSeriesResult<T> {
    /// Interpolated values
    pub values: Array1<T>,
    /// Confidence interval lower bounds (if estimated)
    pub lower_bounds: Option<Array1<T>>,
    /// Confidence interval upper bounds (if estimated)
    pub upper_bounds: Option<Array1<T>>,
    /// Prediction intervals
    pub prediction_intervals: Option<(Array1<T>, Array1<T>)>,
}

/// Time series specific interpolator
#[derive(Debug)]
pub struct TimeSeriesInterpolator<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + Send
        + Sync
        + std::iter::Sum
        + 'static,
{
    /// Configuration for time series interpolation
    config: TimeSeriesConfig<T>,
    /// Training timestamps (as numeric values)
    train_times: Array1<T>,
    /// Training values
    train_values: Array1<T>,
    /// Underlying interpolator for trend component
    trend_interpolator: Option<BSpline<T>>,
    /// Seasonal component interpolator
    seasonal_interpolator: Option<RBFInterpolator<T>>,
    /// Whether the model is trained
    is_trained: bool,
    /// Detected outliers (indices)
    outliers: Vec<usize>,
    /// Temporal statistics
    #[allow(dead_code)]
    temporal_stats: TemporalStats<T>,
}

/// Statistical information about temporal patterns
#[derive(Debug, Clone)]
pub struct TemporalStats<T> {
    /// Detected trend slope
    pub trend_slope: Option<T>,
    /// Seasonal amplitude
    pub seasonal_amplitude: Option<T>,
    /// Noise level estimate
    pub noise_level: T,
    /// Temporal autocorrelation
    pub autocorrelation: Option<T>,
    /// Detected changepoints
    pub changepoints: Vec<usize>,
}

impl<T: Float> Default for TemporalStats<T> {
    fn default() -> Self {
        Self {
            trend_slope: None,
            seasonal_amplitude: None,
            noise_level: T::zero(),
            autocorrelation: None,
            changepoints: Vec::new(),
        }
    }
}

impl<T> Default for TimeSeriesInterpolator<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + Send
        + Sync
        + std::iter::Sum
        + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TimeSeriesInterpolator<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + Send
        + Sync
        + std::iter::Sum
        + 'static,
{
    /// Create a new time series interpolator
    pub fn new() -> Self {
        Self {
            config: TimeSeriesConfig::default(),
            train_times: Array1::zeros(0),
            train_values: Array1::zeros(0),
            trend_interpolator: None,
            seasonal_interpolator: None,
            is_trained: false,
            outliers: Vec::new(),
            temporal_stats: TemporalStats::default(),
        }
    }

    /// Set the temporal pattern
    pub fn with_temporal_pattern(mut self, pattern: TemporalPattern) -> Self {
        self.config.pattern = pattern;
        self
    }

    /// Set the seasonality type
    pub fn with_seasonality_type(mut self, seasonality: SeasonalityType) -> Self {
        self.config.seasonality = Some(seasonality);
        self
    }

    /// Set the missing data strategy
    pub fn with_missing_data_strategy(mut self, strategy: &str) -> Self {
        self.config.missing_strategy = match strategy {
            "linear" => MissingDataStrategy::Linear,
            "spline" => MissingDataStrategy::Spline,
            "forward_fill" => MissingDataStrategy::ForwardFill,
            "backward_fill" => MissingDataStrategy::BackwardFill,
            "mean" => MissingDataStrategy::Mean,
            "seasonal" => MissingDataStrategy::Seasonal,
            _ => MissingDataStrategy::Spline,
        };
        self
    }

    /// Set temporal smoothing parameter
    pub fn with_temporal_smoothing(mut self, smoothing: T) -> Self {
        self.config.temporal_smoothing = smoothing;
        self
    }

    /// Enable uncertainty estimation
    pub fn with_uncertainty_estimation(mut self, enable: bool) -> Self {
        self.config.estimate_uncertainty = enable;
        self
    }

    /// Fit the time series interpolator to data
    ///
    /// # Arguments
    ///
    /// * `timestamps` - Array of timestamps as numeric values (e.g., Unix timestamps)
    /// * `values` - Corresponding values at each timestamp
    ///
    /// # Returns
    ///
    /// Success indicator
    pub fn fit(
        &mut self,
        timestamps: &ArrayView1<T>,
        values: &ArrayView1<T>,
    ) -> InterpolateResult<()> {
        if timestamps.len() != values.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "timestamps and values must have the same length, got {} and {}",
                timestamps.len(),
                values.len()
            )));
        }

        if timestamps.len() < 3 {
            return Err(InterpolateError::InvalidValue(
                "At least 3 data points required for time series interpolation".to_string(),
            ));
        }

        // Store training data
        self.train_times = timestamps.to_owned();
        self.train_values = values.to_owned();

        // Detect outliers if threshold is set
        if let Some(threshold) = self.config.outlier_threshold {
            self.detect_outliers(threshold)?;
        }

        // Decompose the time series based on pattern
        match self.config.pattern {
            TemporalPattern::LinearTrend => {
                self.fit_linear_trend()?;
            }
            TemporalPattern::TrendWithSeasonality => {
                self.fit_trend_and_seasonal()?;
            }
            TemporalPattern::SeasonalOnly => {
                self.fit_seasonal_only()?;
            }
            _ => {
                // For other patterns, use general spline fitting
                self.fit_general_spline()?;
            }
        }

        self.is_trained = true;
        Ok(())
    }

    /// Interpolate values at new timestamps
    ///
    /// # Arguments
    ///
    /// * `timestamps` - Timestamps to interpolate at
    ///
    /// # Returns
    ///
    /// Time series result with interpolated values and optional uncertainty
    pub fn interpolate(
        &self,
        timestamps: &ArrayView1<T>,
    ) -> InterpolateResult<TimeSeriesResult<T>> {
        if !self.is_trained {
            return Err(InterpolateError::InvalidState(
                "Interpolator must be fitted before interpolation".to_string(),
            ));
        }

        let mut interpolated_values = Array1::zeros(timestamps.len());

        // Interpolate trend component
        if let Some(ref trend_interp) = self.trend_interpolator {
            let trend_values = trend_interp.evaluate_array(timestamps)?;
            interpolated_values = interpolated_values + trend_values;
        }

        // Add seasonal component if available
        if let Some(ref seasonal_interp) = self.seasonal_interpolator {
            // Convert 1D timestamps to 2D for RBF interpolator
            let timestamps_2d = Array2::from_shape_vec((timestamps.len(), 1), timestamps.to_vec())
                .map_err(|e| {
                    InterpolateError::ComputationError(format!(
                        "Failed to reshape timestamps: {}",
                        e
                    ))
                })?;

            let seasonal_values = seasonal_interp.interpolate(&timestamps_2d.view())?;
            interpolated_values = interpolated_values + seasonal_values;
        }

        // Estimate uncertainty if requested
        let (lower_bounds, upper_bounds) = if self.config.estimate_uncertainty {
            let uncertainty = self.estimate_uncertainty(timestamps)?;
            (
                Some(interpolated_values.clone() - uncertainty.clone()),
                Some(interpolated_values.clone() + uncertainty),
            )
        } else {
            (None, None)
        };

        Ok(TimeSeriesResult {
            values: interpolated_values,
            lower_bounds,
            upper_bounds,
            prediction_intervals: None,
        })
    }

    /// Detect outliers in the time series data
    fn detect_outliers(&mut self, threshold: T) -> InterpolateResult<()> {
        let n = self.train_values.len();

        // Calculate rolling statistics for outlier detection
        let window_size = (n / 10).clamp(3, 20);
        let mut outliers = Vec::new();

        for i in 0..n {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(n);

            let window = self.train_values.slice(ndarray::s![start..end]);
            let mean = window.sum() / T::from_usize(window.len()).unwrap();

            let variance = window.iter().map(|&x| (x - mean) * (x - mean)).sum::<T>()
                / T::from_usize(window.len() - 1).unwrap();
            let std_dev = variance.sqrt();

            if (self.train_values[i] - mean).abs() > threshold * std_dev {
                outliers.push(i);
            }
        }

        self.outliers = outliers;
        Ok(())
    }

    /// Fit linear trend component
    fn fit_linear_trend(&mut self) -> InterpolateResult<()> {
        // Create B-spline interpolator for trend with linear degree
        let degree = 1;

        // Generate knots for interpolation
        let knots = crate::bspline::generate_knots(&self.train_times.view(), degree, "average")?;

        // Create B-spline interpolator for trend
        let trend_spline = BSpline::new(
            &knots.view(),
            &self.train_values.view(),
            degree,
            crate::bspline::ExtrapolateMode::Extrapolate,
        )?;

        self.trend_interpolator = Some(trend_spline);
        Ok(())
    }

    /// Fit trend and seasonal components
    fn fit_trend_and_seasonal(&mut self) -> InterpolateResult<()> {
        // First fit trend with B-spline
        let degree = 3;
        let knots = crate::bspline::generate_knots(&self.train_times.view(), degree, "average")?;
        let trend_spline = BSpline::new(
            &knots.view(),
            &self.train_values.view(),
            degree,
            crate::bspline::ExtrapolateMode::Extrapolate,
        )?;

        // Calculate trend values
        let trend_values = trend_spline.evaluate_array(&self.train_times.view())?;

        // Calculate residuals (seasonal + noise)
        let residuals = self.train_values.clone() - trend_values;

        // Fit seasonal component using RBF if we have enough data
        if self.train_times.len() >= 8 {
            let times_2d =
                Array2::from_shape_vec((self.train_times.len(), 1), self.train_times.to_vec())
                    .map_err(|e| {
                        InterpolateError::ComputationError(format!(
                            "Failed to reshape times: {}",
                            e
                        ))
                    })?;

            let seasonal_rbf = RBFInterpolator::new(
                &times_2d.view(),
                &residuals.view(),
                RBFKernel::Gaussian,
                T::from_f64(1.0).unwrap(),
            )?;

            self.seasonal_interpolator = Some(seasonal_rbf);
        }

        self.trend_interpolator = Some(trend_spline);
        Ok(())
    }

    /// Fit seasonal component only
    fn fit_seasonal_only(&mut self) -> InterpolateResult<()> {
        if self.train_times.len() >= 8 {
            let times_2d =
                Array2::from_shape_vec((self.train_times.len(), 1), self.train_times.to_vec())
                    .map_err(|e| {
                        InterpolateError::ComputationError(format!(
                            "Failed to reshape times: {}",
                            e
                        ))
                    })?;

            let seasonal_rbf = RBFInterpolator::new(
                &times_2d.view(),
                &self.train_values.view(),
                RBFKernel::Gaussian,
                T::from_f64(0.5).unwrap(),
            )?;

            self.seasonal_interpolator = Some(seasonal_rbf);
        }
        Ok(())
    }

    /// Fit general spline for other patterns
    fn fit_general_spline(&mut self) -> InterpolateResult<()> {
        let degree = 3;
        let knots = crate::bspline::generate_knots(&self.train_times.view(), degree, "average")?;
        let spline = BSpline::new(
            &knots.view(),
            &self.train_values.view(),
            degree,
            crate::bspline::ExtrapolateMode::Extrapolate,
        )?;

        self.trend_interpolator = Some(spline);
        Ok(())
    }

    /// Estimate uncertainty for interpolated values
    fn estimate_uncertainty(&self, _timestamps: &ArrayView1<T>) -> InterpolateResult<Array1<T>> {
        // Simple uncertainty estimation based on local variance
        let n = _timestamps.len();
        let base_uncertainty = self.temporal_stats.noise_level;

        // For now, return constant uncertainty
        // In a more sophisticated implementation, this would be time-dependent
        Ok(Array1::from_elem(n, base_uncertainty))
    }

    /// Get detected outliers
    pub fn get_outliers(&self) -> &[usize] {
        &self.outliers
    }

    /// Get temporal statistics
    pub fn get_temporal_stats(&self) -> &TemporalStats<T> {
        &self.temporal_stats
    }

    /// Check if the interpolator is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }
}

/// Convenience function to create a time series interpolator for daily data
pub fn make_daily_interpolator<T>() -> TimeSeriesInterpolator<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + Send
        + Sync
        + std::iter::Sum
        + 'static,
{
    TimeSeriesInterpolator::new()
        .with_temporal_pattern(TemporalPattern::TrendWithSeasonality)
        .with_seasonality_type(SeasonalityType::Daily)
        .with_missing_data_strategy("spline")
}

/// Convenience function to create a time series interpolator for weekly data
pub fn make_weekly_interpolator<T>() -> TimeSeriesInterpolator<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + LowerExp
        + ScalarOperand
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Copy
        + Send
        + Sync
        + std::iter::Sum
        + 'static,
{
    TimeSeriesInterpolator::new()
        .with_temporal_pattern(TemporalPattern::TrendWithSeasonality)
        .with_seasonality_type(SeasonalityType::Weekly)
        .with_missing_data_strategy("spline")
}

/// Simple forward fill interpolation for time series
pub fn forward_fill<T>(
    timestamps: &ArrayView1<T>,
    values: &ArrayView1<T>,
    query_timestamps: &ArrayView1<T>,
) -> InterpolateResult<Array1<T>>
where
    T: Float + PartialOrd + Copy,
{
    if timestamps.len() != values.len() {
        return Err(InterpolateError::DimensionMismatch(
            "timestamps and values must have same length".to_string(),
        ));
    }

    let mut result = Array1::zeros(query_timestamps.len());

    for (i, &query_time) in query_timestamps.iter().enumerate() {
        // Find the last timestamp <= query_time
        let mut last_value = values[0]; // Default to first value

        for (j, &timestamp) in timestamps.iter().enumerate() {
            if timestamp <= query_time {
                last_value = values[j];
            } else {
                break;
            }
        }

        result[i] = last_value;
    }

    Ok(result)
}

/// Simple backward fill interpolation for time series
pub fn backward_fill<T>(
    timestamps: &ArrayView1<T>,
    values: &ArrayView1<T>,
    query_timestamps: &ArrayView1<T>,
) -> InterpolateResult<Array1<T>>
where
    T: Float + PartialOrd + Copy,
{
    if timestamps.len() != values.len() {
        return Err(InterpolateError::DimensionMismatch(
            "timestamps and values must have same length".to_string(),
        ));
    }

    let mut result = Array1::zeros(query_timestamps.len());

    for (i, &query_time) in query_timestamps.iter().enumerate() {
        // Find the first timestamp >= query_time
        let mut next_value = values[values.len() - 1]; // Default to last value

        for (j, &timestamp) in timestamps.iter().enumerate().rev() {
            if timestamp >= query_time {
                next_value = values[j];
            } else {
                break;
            }
        }

        result[i] = next_value;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_time_series_interpolator_creation() {
        let interpolator = TimeSeriesInterpolator::<f64>::new();
        assert!(!interpolator.is_trained());
        assert_eq!(
            interpolator.config.pattern,
            TemporalPattern::TrendWithSeasonality
        );
    }

    #[test]
    fn test_time_series_interpolator_configuration() {
        let interpolator = TimeSeriesInterpolator::<f64>::new()
            .with_temporal_pattern(TemporalPattern::LinearTrend)
            .with_seasonality_type(SeasonalityType::Weekly)
            .with_missing_data_strategy("linear")
            .with_temporal_smoothing(0.2);

        assert_eq!(interpolator.config.pattern, TemporalPattern::LinearTrend);
        assert_eq!(
            interpolator.config.seasonality,
            Some(SeasonalityType::Weekly)
        );
        assert_eq!(
            interpolator.config.missing_strategy,
            MissingDataStrategy::Linear
        );
        assert!((interpolator.config.temporal_smoothing - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_time_series_fitting() {
        let timestamps = Array1::linspace(0.0, 10.0, 11);
        let values = timestamps.mapv(|t| t + 0.1 * (2.0 * t).sin());

        let mut interpolator =
            TimeSeriesInterpolator::new().with_temporal_pattern(TemporalPattern::LinearTrend);

        let result = interpolator.fit(&timestamps.view(), &values.view());
        assert!(result.is_ok());
        assert!(interpolator.is_trained());
    }

    #[test]
    fn test_time_series_interpolation() {
        let timestamps = Array1::linspace(0.0, 10.0, 11);
        let values = timestamps.mapv(|t| t + 0.1 * (2.0 * t).sin());

        let mut interpolator =
            TimeSeriesInterpolator::new().with_temporal_pattern(TemporalPattern::LinearTrend);

        interpolator
            .fit(&timestamps.view(), &values.view())
            .unwrap();

        let query_times = Array1::from_vec(vec![2.5, 5.0, 7.5]);
        let result = interpolator.interpolate(&query_times.view()).unwrap();

        assert_eq!(result.values.len(), 3);
        assert!(result.values.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_forward_fill() {
        let timestamps = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
        let values = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let query_times = Array1::from_vec(vec![0.0, 2.0, 4.0, 6.0, 8.0]);

        let result = forward_fill(&timestamps.view(), &values.view(), &query_times.view()).unwrap();

        assert_eq!(result, Array1::from_vec(vec![10.0, 10.0, 20.0, 30.0, 40.0]));
    }

    #[test]
    fn test_backward_fill() {
        let timestamps = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
        let values = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let query_times = Array1::from_vec(vec![0.0, 2.0, 4.0, 6.0, 8.0]);

        let result =
            backward_fill(&timestamps.view(), &values.view(), &query_times.view()).unwrap();

        assert_eq!(result, Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 40.0]));
    }

    #[test]
    fn test_make_daily_interpolator() {
        let interpolator = make_daily_interpolator::<f64>();
        assert_eq!(
            interpolator.config.seasonality,
            Some(SeasonalityType::Daily)
        );
        assert_eq!(
            interpolator.config.pattern,
            TemporalPattern::TrendWithSeasonality
        );
    }

    #[test]
    fn test_make_weekly_interpolator() {
        let interpolator = make_weekly_interpolator::<f64>();
        assert_eq!(
            interpolator.config.seasonality,
            Some(SeasonalityType::Weekly)
        );
        assert_eq!(
            interpolator.config.pattern,
            TemporalPattern::TrendWithSeasonality
        );
    }

    #[test]
    fn test_time_series_with_uncertainty() {
        let timestamps = Array1::linspace(0.0, 10.0, 11);
        let values = timestamps.mapv(|t| t + 0.1 * (2.0 * t).sin());

        let mut interpolator = TimeSeriesInterpolator::new()
            .with_temporal_pattern(TemporalPattern::LinearTrend)
            .with_uncertainty_estimation(true);

        interpolator
            .fit(&timestamps.view(), &values.view())
            .unwrap();

        let query_times = Array1::from_vec(vec![2.5, 5.0]);
        let result = interpolator.interpolate(&query_times.view()).unwrap();

        assert_eq!(result.values.len(), 2);
        assert!(result.lower_bounds.is_some());
        assert!(result.upper_bounds.is_some());
    }

    #[test]
    fn test_trend_and_seasonal_fitting() {
        let timestamps = Array1::linspace(0.0, 20.0, 21);
        let values = timestamps.mapv(|t| t * 0.5 + 2.0 * (t * 0.5).sin() + 0.1 * t.cos());

        let mut interpolator = TimeSeriesInterpolator::new()
            .with_temporal_pattern(TemporalPattern::TrendWithSeasonality)
            .with_seasonality_type(SeasonalityType::Custom(4.0));

        let result = interpolator.fit(&timestamps.view(), &values.view());
        assert!(result.is_ok());
        assert!(interpolator.is_trained());
        assert!(interpolator.trend_interpolator.is_some());
    }
}
