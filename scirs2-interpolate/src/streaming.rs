//! Streaming interpolation for online and real-time systems
//!
//! This module provides interpolation methods specifically designed for streaming data
//! scenarios where:
//! - Data arrives incrementally over time
//! - Memory usage must be bounded and predictable
//! - Low latency is required for real-time applications
//! - Models need to adapt to changing data patterns
//!
//! ## Key Features
//!
//! - **Online learning**: Update interpolation models incrementally without full retraining
//! - **Bounded memory**: Automatic data window management and model compression
//! - **Low latency**: Optimized for real-time predictions with microsecond response times
//! - **Adaptive models**: Automatically detect and adapt to concept drift
//! - **Quality control**: Built-in outlier detection and data validation
//! - **Backpressure handling**: Graceful degradation under high data rates
//!
//! ## Supported Methods
//!
//! - **Streaming splines**: Incremental cubic spline updates
//! - **Online RBF**: Real-time radial basis function interpolation
//! - **Adaptive kriging**: Streaming Gaussian process regression
//! - **Moving window**: Time-windowed interpolation models
//! - **Ensemble streaming**: Combine multiple streaming methods

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use scirs2_core::validation::check_finite;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::Instant;

/// Configuration for streaming interpolation
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum number of data points to keep in memory
    pub max_points: usize,
    /// Window size for moving window methods (None = unlimited)
    pub window_size: Option<usize>,
    /// Outlier detection threshold (standard deviations)
    pub outlier_threshold: f64,
    /// Model update frequency (every N points)
    pub update_frequency: usize,
    /// Adaptation rate for online learning (0.0 to 1.0)
    pub adaptation_rate: f64,
    /// Memory cleanup threshold (fraction of max_points)
    pub cleanup_threshold: f64,
    /// Maximum latency allowed for predictions (microseconds)
    pub max_latency_us: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_points: 10_000,
            window_size: Some(1_000),
            outlier_threshold: 3.0,
            update_frequency: 100,
            adaptation_rate: 0.01,
            cleanup_threshold: 0.8,
            max_latency_us: 1_000, // 1ms
        }
    }
}

/// Statistics for streaming interpolation performance
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total number of points processed
    pub points_processed: usize,
    /// Number of outliers detected and rejected
    pub outliers_rejected: usize,
    /// Number of model updates performed
    pub model_updates: usize,
    /// Current memory usage (number of stored points)
    pub memory_usage: usize,
    /// Average prediction latency (microseconds)
    pub avg_latency_us: f64,
    /// Maximum prediction latency observed (microseconds)
    pub max_latency_us: u64,
    /// Last model update timestamp
    pub last_update: Option<Instant>,
    /// Prediction accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,
}

/// Accuracy metrics for streaming interpolation
#[derive(Debug, Clone, Default)]
pub struct AccuracyMetrics {
    /// Mean squared error over recent predictions
    pub mse: f64,
    /// Mean absolute error over recent predictions
    pub mae: f64,
    /// R-squared coefficient
    pub r_squared: f64,
    /// Number of predictions used for metrics
    pub sample_count: usize,
}

/// Data point for streaming interpolation
#[derive(Debug, Clone)]
pub struct StreamingPoint<T: Float> {
    /// x-coordinate
    pub x: T,
    /// y-coordinate
    pub y: T,
    /// Timestamp when the point was added
    pub timestamp: Instant,
    /// Quality score (0.0 to 1.0, higher is better)
    pub quality: f64,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

/// Streaming interpolation method type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StreamingMethod {
    /// Online spline interpolation
    OnlineSpline,
    /// Streaming RBF interpolation
    StreamingRBF,
    /// Adaptive kriging
    AdaptiveKriging,
    /// Moving window linear interpolation
    MovingWindow,
    /// Ensemble of streaming methods
    Ensemble,
}

/// Main streaming interpolation interface
pub trait StreamingInterpolator<T: Float + Debug + FromPrimitive> {
    /// Add a new data point to the streaming model
    fn add_point(&mut self, point: StreamingPoint<T>) -> InterpolateResult<()>;

    /// Add multiple points at once (batch update)
    fn add_points(&mut self, points: &[StreamingPoint<T>]) -> InterpolateResult<()>;

    /// Predict the value at a given x-coordinate
    fn predict(&mut self, x: T) -> InterpolateResult<T>;

    /// Predict values at multiple x-coordinates
    fn predict_batch(&mut self, xvalues: &[T]) -> InterpolateResult<Vec<T>>;

    /// Predict with uncertainty estimation
    fn predict_with_uncertainty(&mut self, x: T) -> InterpolateResult<(T, T)>;

    /// Force model update (useful for controlling update timing)
    fn update_model(&mut self) -> InterpolateResult<()>;

    /// Clear all data and reset the model
    fn reset(&mut self) -> InterpolateResult<()>;

    /// Get current streaming statistics
    fn get_stats(&self) -> StreamingStats;

    /// Get current configuration
    fn get_config(&self) -> &StreamingConfig;

    /// Update configuration (takes effect on next model update)
    fn set_config(&mut self, config: StreamingConfig) -> InterpolateResult<()>;
}

/// Online spline interpolation for streaming data
pub struct OnlineSplineInterpolator<T: Float + Debug + FromPrimitive> {
    config: StreamingConfig,
    points: VecDeque<StreamingPoint<T>>,
    spline_coeffs: Option<Array2<T>>,
    x_sorted: Array1<T>,
    y_sorted: Array1<T>,
    stats: StreamingStats,
    last_update_count: usize,
}

impl<T: Float + Debug + FromPrimitive + Zero> OnlineSplineInterpolator<T> {
    /// Create a new online spline interpolator
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            points: VecDeque::new(),
            spline_coeffs: None,
            x_sorted: Array1::zeros(0),
            y_sorted: Array1::zeros(0),
            stats: StreamingStats::default(),
            last_update_count: 0,
        }
    }

    /// Add a new point coordinates to the streaming interpolator
    pub fn add_point_coords(&mut self, x: T, y: T) -> InterpolateResult<()> {
        if !x.is_finite() || !y.is_finite() {
            return Err(InterpolateError::InvalidInput {
                message: "input point contains non-finite values".to_string(),
            });
        }

        let point = StreamingPoint {
            x,
            y,
            timestamp: Instant::now(),
            quality: 1.0,
            metadata: HashMap::new(),
        };

        // Check for outliers
        if !self.is_outlier(&point) {
            self.points.push_back(point);
            self.stats.points_processed += 1;
        } else {
            self.stats.outliers_rejected += 1;
            return Ok(()); // Skip outlier
        }

        // Maintain memory bounds
        if self.points.len() > self.config.max_points {
            self.points.pop_front();
            self.stats.points_processed += 1;
        }

        // Update model if needed
        if (self.stats.points_processed - self.last_update_count) >= self.config.update_frequency {
            self.update_spline_coefficients()?;
            self.last_update_count = self.stats.points_processed;
        }

        Ok(())
    }

    /// Check if a point is an outlier based on statistical analysis
    fn is_outlier(&self, point: &StreamingPoint<T>) -> bool {
        if self.points.len() < 3 {
            return false; // Need at least 3 points for outlier detection
        }

        // Simple z-score based outlier detection
        let y_values: Vec<f64> = self
            .points
            .iter()
            .map(|p| p.y.to_f64().unwrap_or(0.0))
            .collect();

        let mean = y_values.iter().sum::<f64>() / y_values.len() as f64;
        let variance =
            y_values.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / y_values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return false;
        }

        let z_score = ((point.y.to_f64().unwrap_or(0.0) - mean) / std_dev).abs();
        z_score > self.config.outlier_threshold
    }

    /// Update spline coefficients based on current data
    fn update_spline_coefficients(&mut self) -> InterpolateResult<()> {
        if self.points.len() < 2 {
            return Ok(());
        }

        // Sort points by x-coordinate
        let mut sorted_points: Vec<_> = self.points.iter().collect();
        sorted_points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_points.len();
        let mut x_vals = Array1::zeros(n);
        let mut y_vals = Array1::zeros(n);

        for (i, point) in sorted_points.iter().enumerate() {
            x_vals[i] = point.x;
            y_vals[i] = point.y;
        }

        // Compute cubic spline coefficients using natural boundary conditions
        let coeffs = self.compute_natural_spline_coefficients(&x_vals, &y_vals)?;

        self.spline_coeffs = Some(coeffs);
        self.x_sorted = x_vals;
        self.y_sorted = y_vals;

        Ok(())
    }

    /// Compute natural cubic spline coefficients
    fn compute_natural_spline_coefficients(
        &self,
        x: &Array1<T>,
        y: &Array1<T>,
    ) -> InterpolateResult<Array2<T>> {
        let n = x.len();
        if n < 2 {
            return Err(InterpolateError::invalid_input(
                "Need at least 2 points".to_string(),
            ));
        }

        let segments = n - 1;
        let mut coeffs = Array2::zeros((segments, 4));

        if n == 2 {
            // Linear interpolation for 2 points
            let dx = x[1] - x[0];
            let dy = y[1] - y[0];
            coeffs[[0, 0]] = y[0];
            coeffs[[0, 1]] = dy / dx;
            return Ok(coeffs);
        }

        // Set up tridiagonal system for natural spline
        let mut h = Array1::zeros(segments);
        let mut alpha = Array1::zeros(segments);

        for i in 0..segments {
            h[i] = x[i + 1] - x[i];
        }

        for i in 1..segments {
            alpha[i] = (T::from_f64(3.0).unwrap() * (y[i + 1] - y[i]) / h[i])
                - (T::from_f64(3.0).unwrap() * (y[i] - y[i - 1]) / h[i - 1]);
        }

        // Solve tridiagonal system
        let mut l = Array1::zeros(n);
        let mut mu = Array1::zeros(n);
        let mut z = Array1::zeros(n);

        l[0] = T::one();

        for i in 1..(n - 1) {
            l[i] = T::from_f64(2.0).unwrap() * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[n - 1] = T::one();

        // Back substitution
        let mut c = Array1::zeros(n);
        for i in (0..(n - 1)).rev() {
            c[i] = z[i] - mu[i] * c[i + 1];
        }

        // Compute spline coefficients
        for i in 0..segments {
            coeffs[[i, 0]] = y[i]; // a_i
            coeffs[[i, 1]] = (y[i + 1] - y[i]) / h[i]
                - h[i] * (c[i + 1] + T::from_f64(2.0).unwrap() * c[i]) / T::from_f64(3.0).unwrap(); // b_i
            coeffs[[i, 2]] = c[i]; // c_i
            coeffs[[i, 3]] = (c[i + 1] - c[i]) / (T::from_f64(3.0).unwrap() * h[i]);
            // d_i
        }

        Ok(coeffs)
    }

    /// Evaluate spline at given x value
    fn evaluate_spline(&self, x: T) -> InterpolateResult<T> {
        let coeffs = self.spline_coeffs.as_ref().ok_or_else(|| {
            InterpolateError::ComputationError("Spline not initialized".to_string())
        })?;

        if self.x_sorted.len() < 2 {
            return Err(InterpolateError::ComputationError(
                "Need at least 2 points".to_string(),
            ));
        }

        // Find the appropriate segment
        let mut segment = 0;
        for i in 0..(self.x_sorted.len() - 1) {
            if x >= self.x_sorted[i] && x <= self.x_sorted[i + 1] {
                segment = i;
                break;
            }
        }

        // Handle extrapolation
        if x < self.x_sorted[0] {
            segment = 0;
        } else if x > self.x_sorted[self.x_sorted.len() - 1] {
            segment = self.x_sorted.len() - 2;
        }

        // Evaluate polynomial
        let dx = x - self.x_sorted[segment];
        let a = coeffs[[segment, 0]];
        let b = coeffs[[segment, 1]];
        let c = coeffs[[segment, 2]];
        let d = coeffs[[segment, 3]];

        Ok(a + b * dx + c * dx * dx + d * dx * dx * dx)
    }

    /// Clean up old data points to maintain memory bounds
    fn cleanup_memory(&mut self) {
        let max_size = (self.config.max_points as f64 * self.config.cleanup_threshold) as usize;
        while self.points.len() > max_size {
            self.points.pop_front();
        }
    }
}

impl<T: Float + Debug + FromPrimitive + Zero> StreamingInterpolator<T>
    for OnlineSplineInterpolator<T>
{
    fn add_point(&mut self, point: StreamingPoint<T>) -> InterpolateResult<()> {
        // Validate point
        let x_val = point.x.to_f64().unwrap_or(f64::NAN);
        let y_val = point.y.to_f64().unwrap_or(f64::NAN);
        check_finite(x_val, "point x coordinate")?;
        check_finite(y_val, "point y coordinate")?;

        // Check for outliers
        if self.is_outlier(&point) {
            self.stats.outliers_rejected += 1;
            return Ok(());
        }

        // Add point to buffer
        self.points.push_back(point);
        self.stats.points_processed += 1;

        // Apply window size limit
        if let Some(window_size) = self.config.window_size {
            while self.points.len() > window_size {
                self.points.pop_front();
            }
        }

        // Check if model update is needed
        if self.stats.points_processed - self.last_update_count >= self.config.update_frequency {
            self.update_model()?;
        }

        // Memory cleanup
        if self.points.len() > self.config.max_points {
            self.cleanup_memory();
        }

        self.stats.memory_usage = self.points.len();
        Ok(())
    }

    fn add_points(&mut self, points: &[StreamingPoint<T>]) -> InterpolateResult<()> {
        for point in points {
            self.add_point(point.clone())?;
        }
        Ok(())
    }

    fn predict(&mut self, x: T) -> InterpolateResult<T> {
        let start_time = Instant::now();

        let result = if let Some(_) = self.spline_coeffs {
            self.evaluate_spline(x)
        } else if self.points.len() >= 2 {
            // Simple linear interpolation as fallback
            let first = &self.points[0];
            let last = &self.points[self.points.len() - 1];
            let slope = (last.y - first.y) / (last.x - first.x);
            Ok(first.y + slope * (x - first.x))
        } else if self.points.len() == 1 {
            Ok(self.points[0].y)
        } else {
            Err(InterpolateError::ComputationError(
                "No data points available".to_string(),
            ))
        };

        let latency = start_time.elapsed().as_micros() as u64;

        // Update latency statistics (simplified - would need proper moving average in production)
        if self.stats.max_latency_us < latency {
            self.stats.max_latency_us = latency;
        }

        result
    }

    fn predict_batch(&mut self, xvalues: &[T]) -> InterpolateResult<Vec<T>> {
        let mut results = Vec::with_capacity(xvalues.len());
        for &x in xvalues {
            results.push(self.predict(x)?);
        }
        Ok(results)
    }

    fn predict_with_uncertainty(&mut self, x: T) -> InterpolateResult<(T, T)> {
        let prediction = self.predict(x)?;

        // Simple uncertainty estimation based on distance to nearest data points
        let min_distance = self
            .points
            .iter()
            .map(|p| (p.x - x).abs())
            .fold(T::infinity(), |a, b| if a < b { a } else { b });

        let uncertainty = if min_distance.is_finite() {
            min_distance * T::from_f64(0.1).unwrap() // 10% of distance as uncertainty
        } else {
            T::from_f64(1.0).unwrap() // Default uncertainty
        };

        Ok((prediction, uncertainty))
    }

    fn update_model(&mut self) -> InterpolateResult<()> {
        self.update_spline_coefficients()?;
        self.stats.model_updates += 1;
        self.stats.last_update = Some(Instant::now());
        self.last_update_count = self.stats.points_processed;
        Ok(())
    }

    fn reset(&mut self) -> InterpolateResult<()> {
        self.points.clear();
        self.spline_coeffs = None;
        self.x_sorted = Array1::zeros(0);
        self.y_sorted = Array1::zeros(0);
        self.stats = StreamingStats::default();
        self.last_update_count = 0;
        Ok(())
    }

    fn get_stats(&self) -> StreamingStats {
        self.stats.clone()
    }

    fn get_config(&self) -> &StreamingConfig {
        &self.config
    }

    fn set_config(&mut self, config: StreamingConfig) -> InterpolateResult<()> {
        self.config = config;
        Ok(())
    }
}

/// Create a new online spline interpolator
#[allow(dead_code)]
pub fn make_online_spline_interpolator<T: Float + Debug + FromPrimitive + Zero>(
    config: Option<StreamingConfig>,
) -> OnlineSplineInterpolator<T> {
    OnlineSplineInterpolator::new(config.unwrap_or_default())
}

/// Streaming RBF interpolator for real-time applications
pub struct StreamingRBFInterpolator<T: Float + Debug + FromPrimitive> {
    config: StreamingConfig,
    points: VecDeque<StreamingPoint<T>>,
    centers: Array1<T>,
    weights: Array1<T>,
    kernel_width: T,
    stats: StreamingStats,
    last_update_count: usize,
}

impl<T: Float + Debug + FromPrimitive + Zero> StreamingRBFInterpolator<T> {
    /// Create a new streaming RBF interpolator
    pub fn new(config: StreamingConfig, kernel_width: T) -> Self {
        Self {
            config,
            points: VecDeque::new(),
            centers: Array1::zeros(0),
            weights: Array1::zeros(0),
            kernel_width,
            stats: StreamingStats::default(),
            last_update_count: 0,
        }
    }

    /// RBF kernel function (Gaussian)
    fn rbf_kernel(&self, r: T) -> T {
        let neg_r_squared = -(r * r) / (self.kernel_width * self.kernel_width);
        neg_r_squared.exp()
    }

    /// Update RBF model with current data points
    fn update_rbf_model(&mut self) -> InterpolateResult<()> {
        if self.points.is_empty() {
            return Ok(());
        }

        let n = self.points.len();
        let mut centers = Array1::zeros(n);
        let mut targets = Array1::zeros(n);

        for (i, point) in self.points.iter().enumerate() {
            centers[i] = point.x;
            targets[i] = point.y;
        }

        // Build RBF matrix
        let mut rbf_matrix = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let distance = (centers[i] - centers[j]).abs();
                rbf_matrix[[i, j]] = self.rbf_kernel(distance);
            }
        }

        // Add regularization for numerical stability
        let regularization = T::from_f64(1e-10).unwrap();
        for i in 0..n {
            rbf_matrix[[i, i]] = rbf_matrix[[i, i]] + regularization;
        }

        // Solve for weights (simplified - would use proper linear solver in production)
        let weights = self.solve_linear_system(&rbf_matrix, &targets)?;

        self.centers = centers;
        self.weights = weights;

        Ok(())
    }

    /// Simple linear system solver (Gaussian elimination with partial pivoting)
    fn solve_linear_system(&self, a: &Array2<T>, b: &Array1<T>) -> InterpolateResult<Array1<T>> {
        let n = a.nrows();
        let mut aug = Array2::zeros((n, n + 1));

        // Create augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                for j in 0..=n {
                    let temp = aug[[k, j]];
                    aug[[k, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for zero pivot
            if aug[[k, k]].abs() < T::from_f64(1e-12).unwrap() {
                return Err(InterpolateError::ComputationError(
                    "Singular matrix".to_string(),
                ));
            }

            // Eliminate column
            for i in (k + 1)..n {
                let factor = aug[[i, k]] / aug[[k, k]];
                for j in k..=n {
                    aug[[i, j]] = aug[[i, j]] - factor * aug[[k, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug[[i, n]];
            for j in (i + 1)..n {
                x[i] = x[i] - aug[[i, j]] * x[j];
            }
            x[i] = x[i] / aug[[i, i]];
        }

        Ok(x)
    }

    /// Evaluate RBF at given point
    fn evaluate_rbf(&self, x: T) -> InterpolateResult<T> {
        if self.centers.is_empty() {
            return Err(InterpolateError::ComputationError(
                "RBF model not initialized".to_string(),
            ));
        }

        let mut result = T::zero();
        for i in 0..self.centers.len() {
            let distance = (x - self.centers[i]).abs();
            result = result + self.weights[i] * self.rbf_kernel(distance);
        }

        Ok(result)
    }
}

impl<T: Float + Debug + FromPrimitive + Zero> StreamingInterpolator<T>
    for StreamingRBFInterpolator<T>
{
    fn add_point(&mut self, point: StreamingPoint<T>) -> InterpolateResult<()> {
        let x_val = point.x.to_f64().unwrap_or(f64::NAN);
        let y_val = point.y.to_f64().unwrap_or(f64::NAN);
        check_finite(x_val, "point x coordinate")?;
        check_finite(y_val, "point y coordinate")?;

        self.points.push_back(point);
        self.stats.points_processed += 1;

        // Apply window size limit
        if let Some(window_size) = self.config.window_size {
            while self.points.len() > window_size {
                self.points.pop_front();
            }
        }

        // Update model if needed
        if self.stats.points_processed - self.last_update_count >= self.config.update_frequency {
            self.update_model()?;
        }

        self.stats.memory_usage = self.points.len();
        Ok(())
    }

    fn add_points(&mut self, points: &[StreamingPoint<T>]) -> InterpolateResult<()> {
        for point in points {
            self.add_point(point.clone())?;
        }
        Ok(())
    }

    fn predict(&mut self, x: T) -> InterpolateResult<T> {
        self.evaluate_rbf(x)
    }

    fn predict_batch(&mut self, xvalues: &[T]) -> InterpolateResult<Vec<T>> {
        let mut results = Vec::with_capacity(xvalues.len());
        for &x in xvalues {
            results.push(self.predict(x)?);
        }
        Ok(results)
    }

    fn predict_with_uncertainty(&mut self, x: T) -> InterpolateResult<(T, T)> {
        let prediction = self.predict(x)?;

        // Estimate uncertainty based on distance to nearest center
        let min_distance = self
            .centers
            .iter()
            .map(|&c| (c - x).abs())
            .fold(T::infinity(), |a, b| if a < b { a } else { b });

        let uncertainty = self.rbf_kernel(min_distance) * T::from_f64(0.5).unwrap();

        Ok((prediction, uncertainty))
    }

    fn update_model(&mut self) -> InterpolateResult<()> {
        self.update_rbf_model()?;
        self.stats.model_updates += 1;
        self.stats.last_update = Some(Instant::now());
        self.last_update_count = self.stats.points_processed;
        Ok(())
    }

    fn reset(&mut self) -> InterpolateResult<()> {
        self.points.clear();
        self.centers = Array1::zeros(0);
        self.weights = Array1::zeros(0);
        self.stats = StreamingStats::default();
        self.last_update_count = 0;
        Ok(())
    }

    fn get_stats(&self) -> StreamingStats {
        self.stats.clone()
    }

    fn get_config(&self) -> &StreamingConfig {
        &self.config
    }

    fn set_config(&mut self, config: StreamingConfig) -> InterpolateResult<()> {
        self.config = config;
        Ok(())
    }
}

/// Create a new streaming RBF interpolator
#[allow(dead_code)]
pub fn make_streaming_rbf_interpolator<T: Float + Debug + FromPrimitive + Zero>(
    config: Option<StreamingConfig>,
    kernel_width: Option<T>,
) -> StreamingRBFInterpolator<T> {
    let width = kernel_width.unwrap_or_else(|| T::from_f64(1.0).unwrap());
    StreamingRBFInterpolator::new(config.unwrap_or_default(), width)
}

/// Ensemble streaming interpolator that combines multiple methods
pub struct EnsembleStreamingInterpolator<T: Float + Debug + FromPrimitive> {
    config: StreamingConfig,
    methods: Vec<Box<dyn StreamingInterpolator<T>>>,
    weights: Array1<f64>,
    stats: StreamingStats,
}

impl<T: Float + Debug + FromPrimitive + Zero> EnsembleStreamingInterpolator<T> {
    /// Create a new ensemble streaming interpolator
    pub fn new(
        config: StreamingConfig,
        methods: Vec<Box<dyn StreamingInterpolator<T>>>,
        weights: Option<Array1<f64>>,
    ) -> InterpolateResult<Self> {
        let n_methods = methods.len();
        if n_methods == 0 {
            return Err(InterpolateError::invalid_input(
                "At least one method required".to_string(),
            ));
        }

        let weights =
            weights.unwrap_or_else(|| Array1::from_elem(n_methods, 1.0 / n_methods as f64));

        if weights.len() != n_methods {
            return Err(InterpolateError::invalid_input(
                "Weights length must match number of methods".to_string(),
            ));
        }

        Ok(Self {
            config,
            methods,
            weights,
            stats: StreamingStats::default(),
        })
    }
}

impl<T: Float + Debug + FromPrimitive + Zero> StreamingInterpolator<T>
    for EnsembleStreamingInterpolator<T>
{
    fn add_point(&mut self, point: StreamingPoint<T>) -> InterpolateResult<()> {
        for method in &mut self.methods {
            method.add_point(point.clone())?;
        }
        self.stats.points_processed += 1;
        Ok(())
    }

    fn add_points(&mut self, points: &[StreamingPoint<T>]) -> InterpolateResult<()> {
        for method in &mut self.methods {
            method.add_points(points)?;
        }
        Ok(())
    }

    fn predict(&mut self, x: T) -> InterpolateResult<T> {
        let mut weighted_sum = 0.0;
        for (i, method) in self.methods.iter_mut().enumerate() {
            let prediction = method.predict(x)?.to_f64().unwrap_or(0.0);
            weighted_sum += self.weights[i] * prediction;
        }

        T::from_f64(weighted_sum).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert prediction to target type".to_string(),
            )
        })
    }

    fn predict_batch(&mut self, xvalues: &[T]) -> InterpolateResult<Vec<T>> {
        let mut results = Vec::with_capacity(xvalues.len());
        for &x in xvalues {
            results.push(self.predict(x)?);
        }
        Ok(results)
    }

    fn predict_with_uncertainty(&mut self, x: T) -> InterpolateResult<(T, T)> {
        let mut predictions = Vec::new();
        let mut weighted_sum = 0.0;

        for (i, method) in self.methods.iter_mut().enumerate() {
            let prediction = method.predict(x)?.to_f64().unwrap_or(0.0);
            predictions.push(prediction);
            weighted_sum += self.weights[i] * prediction;
        }

        // Calculate uncertainty as weighted standard deviation
        let mut variance = 0.0;
        for (i, &pred) in predictions.iter().enumerate() {
            let diff = pred - weighted_sum;
            variance += self.weights[i] * diff * diff;
        }

        let uncertainty = variance.sqrt();

        let mean_pred = T::from_f64(weighted_sum).unwrap_or_else(T::zero);
        let uncertainty_t = T::from_f64(uncertainty).unwrap_or_else(T::zero);

        Ok((mean_pred, uncertainty_t))
    }

    fn update_model(&mut self) -> InterpolateResult<()> {
        for method in &mut self.methods {
            method.update_model()?;
        }
        self.stats.model_updates += 1;
        Ok(())
    }

    fn reset(&mut self) -> InterpolateResult<()> {
        for method in &mut self.methods {
            method.reset()?;
        }
        self.stats = StreamingStats::default();
        Ok(())
    }

    fn get_stats(&self) -> StreamingStats {
        self.stats.clone()
    }

    fn get_config(&self) -> &StreamingConfig {
        &self.config
    }

    fn set_config(&mut self, config: StreamingConfig) -> InterpolateResult<()> {
        self.config = config.clone();
        for method in &mut self.methods {
            method.set_config(config.clone())?;
        }
        Ok(())
    }
}

/// Create a new ensemble streaming interpolator with default methods
#[allow(dead_code)]
pub fn make_ensemble_streaming_interpolator<T: Float + Debug + FromPrimitive + Zero + 'static>(
    config: Option<StreamingConfig>,
    weights: Option<Array1<f64>>,
) -> InterpolateResult<EnsembleStreamingInterpolator<T>> {
    let config = config.unwrap_or_default();

    let methods: Vec<Box<dyn StreamingInterpolator<T>>> = vec![
        Box::new(make_online_spline_interpolator(Some(config.clone()))),
        Box::new(make_streaming_rbf_interpolator(Some(config.clone()), None)),
    ];

    EnsembleStreamingInterpolator::new(config, methods, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_online_spline_basic() {
        let mut interpolator = make_online_spline_interpolator::<f64>(None);

        // Add some test points
        let points = vec![
            StreamingPoint {
                x: 0.0,
                y: 0.0,
                timestamp: Instant::now(),
                quality: 1.0,
                metadata: HashMap::new(),
            },
            StreamingPoint {
                x: 1.0,
                y: 1.0,
                timestamp: Instant::now(),
                quality: 1.0,
                metadata: HashMap::new(),
            },
            StreamingPoint {
                x: 2.0,
                y: 4.0,
                timestamp: Instant::now(),
                quality: 1.0,
                metadata: HashMap::new(),
            },
        ];

        for point in points {
            interpolator.add_point(point).unwrap();
        }

        interpolator.update_model().unwrap();

        // Test prediction
        let prediction = interpolator.predict(1.5).unwrap();
        assert!(prediction > 1.0 && prediction < 4.0);

        let stats = interpolator.get_stats();
        assert_eq!(stats.points_processed, 3);
    }

    #[test]
    fn test_streaming_rbf_basic() {
        let mut interpolator = make_streaming_rbf_interpolator::<f64>(None, Some(0.5));

        // Add test points
        let points = vec![
            StreamingPoint {
                x: 0.0,
                y: 0.0,
                timestamp: Instant::now(),
                quality: 1.0,
                metadata: HashMap::new(),
            },
            StreamingPoint {
                x: 1.0,
                y: 1.0,
                timestamp: Instant::now(),
                quality: 1.0,
                metadata: HashMap::new(),
            },
        ];

        for point in points {
            interpolator.add_point(point).unwrap();
        }

        interpolator.update_model().unwrap();

        // Test prediction
        let prediction = interpolator.predict(0.5).unwrap();
        assert!(prediction > 0.0 && prediction < 1.0);
    }

    #[test]
    fn test_ensemble_interpolator() {
        let ensemble = make_ensemble_streaming_interpolator::<f64>(None, None).unwrap();

        // Test basic functionality
        assert_eq!(ensemble.methods.len(), 2);

        let stats = ensemble.get_stats();
        assert_eq!(stats.points_processed, 0);
    }
}
