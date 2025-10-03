//! Error analysis and convergence visualization tools
//!
//! This module provides tools for visualizing numerical errors, convergence behavior,
//! and algorithm performance analysis for iterative methods and numerical solvers.

use super::types::*;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use std::collections::HashMap;

/// Error visualization engine for numerical analysis
#[derive(Debug, Clone)]
pub struct ErrorVisualizationEngine {
    /// Error analysis options
    pub options: ErrorVisualizationOptions,
    /// Color schemes for different error types
    pub error_color_schemes: HashMap<ErrorType, ColorScheme>,
}

impl ErrorVisualizationEngine {
    /// Create new error visualization engine
    pub fn new() -> Self {
        let mut error_color_schemes = HashMap::new();
        error_color_schemes.insert(ErrorType::Absolute, ColorScheme::Viridis);
        error_color_schemes.insert(ErrorType::Relative, ColorScheme::Plasma);
        error_color_schemes.insert(ErrorType::Truncation, ColorScheme::Inferno);
        error_color_schemes.insert(ErrorType::Roundoff, ColorScheme::Grayscale);
        error_color_schemes.insert(ErrorType::Discretization, ColorScheme::Viridis);

        Self {
            options: ErrorVisualizationOptions::default(),
            error_color_schemes,
        }
    }

    /// Visualize error distribution
    pub fn visualize_error_distribution(
        &self,
        errors: &Array1<f64>,
        error_type: ErrorType,
    ) -> IntegrateResult<ErrorDistributionPlot> {
        let n_bins = 50;
        let min_error = errors.iter().copied().fold(f64::INFINITY, f64::min);
        let max_error = errors.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if min_error >= max_error {
            return Err(IntegrateError::ValueError(
                "Invalid error range for distribution".to_string(),
            ));
        }

        let bin_width = (max_error - min_error) / n_bins as f64;
        let mut histogram = Array1::zeros(n_bins);
        let mut bin_centers = Array1::zeros(n_bins);

        for i in 0..n_bins {
            bin_centers[i] = min_error + (i as f64 + 0.5) * bin_width;
        }

        for &error in errors {
            let bin_idx = ((error - min_error) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(n_bins - 1);
            histogram[bin_idx] += 1.0;
        }

        // Normalize histogram
        let total_count = histogram.sum();
        if total_count > 0.0 {
            histogram /= total_count;
        }

        let statistics = ErrorStatistics {
            mean: errors.iter().sum::<f64>() / errors.len() as f64,
            std_dev: errors.std(0.0),
            min: min_error,
            max: max_error,
            median: self.compute_median(errors),
            percentile_95: self.compute_percentile(errors, 0.95),
        };

        Ok(ErrorDistributionPlot {
            bin_centers,
            histogram,
            error_type,
            statistics,
            color_scheme: self.error_color_schemes[&error_type],
        })
    }

    /// Compute median of array
    fn compute_median(&self, values: &Array1<f64>) -> f64 {
        let mut sorted_values: Vec<f64> = values.iter().copied().collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_values.len();
        if n == 0 {
            return 0.0;
        }

        if n.is_multiple_of(2) {
            (sorted_values[n / 2 - 1] + sorted_values[n / 2]) / 2.0
        } else {
            sorted_values[n / 2]
        }
    }

    /// Compute percentile of array
    fn compute_percentile(&self, values: &Array1<f64>, percentile: f64) -> f64 {
        let mut sorted_values: Vec<f64> = values.iter().copied().collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_values.len();
        if n == 0 {
            return 0.0;
        }

        let index = (percentile * (n - 1) as f64).round() as usize;
        let index = index.min(n - 1);
        sorted_values[index]
    }

    /// Create error evolution plot over time/iterations
    pub fn visualize_error_evolution(
        &self,
        time_points: &Array1<f64>,
        errors: &Array1<f64>,
        error_type: ErrorType,
    ) -> IntegrateResult<PhaseSpacePlot> {
        let mut metadata = PlotMetadata::default();
        metadata.title = format!("{:?} Error Evolution", error_type);
        metadata.xlabel = "Time/Iteration".to_string();
        metadata.ylabel = "Error Magnitude".to_string();

        Ok(PhaseSpacePlot {
            x: time_points.to_vec(),
            y: errors.to_vec(),
            colors: Some(errors.iter().map(|&e| e.log10()).collect()),
            metadata,
        })
    }

    /// Compare multiple error types
    pub fn compare_error_types(
        &self,
        time_points: &Array1<f64>,
        error_data: &[(ErrorType, Array1<f64>)],
    ) -> IntegrateResult<Vec<PhaseSpacePlot>> {
        let mut plots = Vec::new();

        for (error_type, errors) in error_data {
            let plot = self.visualize_error_evolution(time_points, errors, *error_type)?;
            plots.push(plot);
        }

        Ok(plots)
    }
}

impl Default for ErrorVisualizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced convergence visualization for iterative algorithms
#[derive(Debug, Clone)]
pub struct ConvergenceVisualizer {
    /// Maximum number of iterations to track
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Track multiple convergence metrics
    pub track_multiple_metrics: bool,
}

impl ConvergenceVisualizer {
    /// Create new convergence visualizer
    pub fn new(maxiterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations: maxiterations,
            tolerance,
            track_multiple_metrics: true,
        }
    }

    /// Create convergence plot for residuals
    pub fn plot_residual_convergence(
        &self,
        residuals: &Array1<f64>,
        algorithm_name: &str,
    ) -> IntegrateResult<ConvergencePlot> {
        let n_iter = residuals.len().min(self.max_iterations);
        let iterations: Array1<f64> = Array1::range(1.0, n_iter as f64 + 1.0, 1.0);

        // Apply log scale for better visualization
        let log_residuals = residuals.slice(s![..n_iter]).mapv(|r| r.abs().log10());

        // Detect convergence point
        let convergence_iteration = self.detect_convergence_point(residuals);

        // Compute convergence rate
        let convergence_rate = self.estimate_convergence_rate(residuals);

        // Create theoretical convergence line for comparison
        let theoretical_line = if convergence_rate > 0.0 {
            Some(self.create_theoretical_convergence(&iterations, convergence_rate))
        } else {
            None
        };

        Ok(ConvergencePlot {
            iterations: iterations.slice(s![..n_iter]).to_owned(),
            residuals: log_residuals,
            convergence_iteration,
            convergence_rate,
            theoretical_line,
            algorithm_name: algorithm_name.to_string(),
            tolerance_line: self.tolerance.log10(),
        })
    }

    /// Create multi-metric convergence plot
    pub fn plot_multi_metric_convergence(
        &self,
        metrics: &[(String, Array1<f64>)],
    ) -> IntegrateResult<MultiMetricConvergencePlot> {
        let mut convergence_curves = Vec::new();
        let mut convergence_rates = Vec::new();

        let max_len = metrics
            .iter()
            .map(|(_, data)| data.len())
            .max()
            .unwrap_or(0);
        let iterations: Array1<f64> = Array1::range(1.0, max_len as f64 + 1.0, 1.0);

        for (name, data) in metrics {
            let n_points = data.len().min(self.max_iterations);
            let log_data = data.slice(s![..n_points]).mapv(|r| r.abs().log10());
            let rate = self.estimate_convergence_rate(data);

            convergence_curves.push(ConvergenceCurve {
                name: name.clone(),
                data: log_data,
                convergence_rate: rate,
                color: self.assign_curve_color(&convergence_curves),
            });

            convergence_rates.push((name.clone(), rate));
        }

        Ok(MultiMetricConvergencePlot {
            iterations: iterations
                .slice(s![..max_len.min(self.max_iterations)])
                .to_owned(),
            curves: convergence_curves,
            convergence_rates,
            tolerance_line: self.tolerance.log10(),
        })
    }

    /// Visualize error vs. step size for method comparison
    pub fn plot_step_size_analysis(
        &self,
        step_sizes: &Array1<f64>,
        errors: &Array1<f64>,
        method_name: &str,
    ) -> IntegrateResult<StepSizeAnalysisPlot> {
        let log_step_sizes = step_sizes.mapv(|h| h.log10());
        let log_errors = errors.mapv(|e| e.abs().log10());

        // Estimate order of accuracy via linear regression
        let order = self.estimate_order_of_accuracy(&log_step_sizes, &log_errors);

        // Create theoretical line showing expected convergence order
        let theoretical_errors = if order > 0.0 {
            Some(self.create_theoretical_error_line(&log_step_sizes, order, &log_errors))
        } else {
            None
        };

        Ok(StepSizeAnalysisPlot {
            log_step_sizes,
            log_errors,
            theoretical_errors,
            order_of_accuracy: order,
            method_name: method_name.to_string(),
        })
    }

    /// Create phase space density plot for attractor visualization
    pub fn plot_phase_space_density(
        &self,
        x_data: &Array1<f64>,
        y_data: &Array1<f64>,
        grid_size: usize,
    ) -> IntegrateResult<PhaseDensityPlot> {
        // Find data bounds with padding
        let x_min = x_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let padding = 0.1;

        let x_bounds = (x_min - padding * x_range, x_max + padding * x_range);
        let y_bounds = (y_min - padding * y_range, y_max + padding * y_range);

        // Create density grid
        let mut density_grid = Array2::zeros((grid_size, grid_size));
        let dx = (x_bounds.1 - x_bounds.0) / grid_size as f64;
        let dy = (y_bounds.1 - y_bounds.0) / grid_size as f64;

        // Fill density grid
        for (&x, &y) in x_data.iter().zip(y_data.iter()) {
            let i = ((x - x_bounds.0) / dx).floor() as usize;
            let j = ((y - y_bounds.0) / dy).floor() as usize;

            let i = i.min(grid_size - 1);
            let j = j.min(grid_size - 1);

            density_grid[[i, j]] += 1.0;
        }

        // Normalize density
        let max_density = density_grid.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_density > 0.0 {
            density_grid /= max_density;
        }

        // Create coordinate grids
        let x_grid = Array1::range(x_bounds.0, x_bounds.1, dx);
        let y_grid = Array1::range(y_bounds.0, y_bounds.1, dy);

        Ok(PhaseDensityPlot {
            x_grid,
            y_grid,
            density_grid,
            x_bounds,
            y_bounds,
            n_points: x_data.len(),
        })
    }

    /// Detect convergence point based on tolerance
    fn detect_convergence_point(&self, residuals: &Array1<f64>) -> Option<usize> {
        for (i, &residual) in residuals.iter().enumerate() {
            if residual.abs() < self.tolerance {
                return Some(i + 1);
            }
        }
        None
    }

    /// Estimate convergence rate using linear regression on log data
    fn estimate_convergence_rate(&self, residuals: &Array1<f64>) -> f64 {
        if residuals.len() < 10 {
            return 0.0;
        }

        // Use last portion of data for rate estimation
        let start_idx = residuals.len() / 2;
        let end_idx = residuals.len();

        // Linear regression to estimate convergence rate
        let x: Array1<f64> = Array1::range(start_idx as f64, end_idx as f64, 1.0);
        let y: Array1<f64> = residuals
            .slice(s![start_idx..end_idx])
            .mapv(|r| r.abs().log10());

        // Calculate slope using least squares
        let x_mean = x.iter().sum::<f64>() / x.len() as f64;
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();
        let denominator: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();

        if denominator.abs() > 1e-12 {
            -numerator / denominator // Negative because residuals should decrease
        } else {
            0.0
        }
    }

    /// Create theoretical convergence line
    fn create_theoretical_convergence(&self, iterations: &Array1<f64>, rate: f64) -> Array1<f64> {
        let initial_residual = self.tolerance * 10.0; // Start above tolerance
        iterations.mapv(|iter| (initial_residual * (-rate * iter).exp()).log10())
    }

    /// Assign color to convergence curve
    fn assign_curve_color(&self, existingcurves: &[ConvergenceCurve]) -> [f64; 3] {
        let colors = [
            [0.0, 0.4470, 0.7410],    // Blue
            [0.8500, 0.3250, 0.0980], // Orange
            [0.9290, 0.6940, 0.1250], // Yellow
            [0.4940, 0.1840, 0.5560], // Purple
            [0.4660, 0.6740, 0.1880], // Green
            [0.3011, 0.7450, 0.9330], // Cyan
            [0.6350, 0.0780, 0.1840], // Red
        ];

        let index = existingcurves.len() % colors.len();
        colors[index]
    }

    /// Estimate order of accuracy via linear regression
    fn estimate_order_of_accuracy(
        &self,
        log_step_sizes: &Array1<f64>,
        log_errors: &Array1<f64>,
    ) -> f64 {
        if log_step_sizes.len() < 3 {
            return 0.0;
        }

        let x_mean = log_step_sizes.iter().sum::<f64>() / log_step_sizes.len() as f64;
        let y_mean = log_errors.iter().sum::<f64>() / log_errors.len() as f64;

        let numerator: f64 = log_step_sizes
            .iter()
            .zip(log_errors.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();
        let denominator: f64 = log_step_sizes.iter().map(|&xi| (xi - x_mean).powi(2)).sum();

        if denominator.abs() > 1e-12 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Create theoretical error line for step size analysis
    fn create_theoretical_error_line(
        &self,
        log_step_sizes: &Array1<f64>,
        order: f64,
        log_errors: &Array1<f64>,
    ) -> Array1<f64> {
        if log_errors.is_empty() {
            return Array1::zeros(log_step_sizes.len());
        }

        // Use first point as reference
        let ref_log_h = log_step_sizes[0];
        let ref_log_e = log_errors[0];

        log_step_sizes.mapv(|log_h| ref_log_e + order * (log_h - ref_log_h))
    }
}

impl Default for ConvergenceVisualizer {
    fn default() -> Self {
        Self::new(1000, 1e-6)
    }
}

/// Visualization engine that combines error and convergence analysis
#[derive(Debug, Clone)]
pub struct ConvergenceVisualizationEngine {
    /// Convergence visualizer
    pub convergence_visualizer: ConvergenceVisualizer,
    /// Error visualization engine
    pub error_engine: ErrorVisualizationEngine,
    /// Performance tracking
    pub performance_tracker: PerformanceTracker,
}

impl ConvergenceVisualizationEngine {
    /// Create new convergence visualization engine
    pub fn new() -> Self {
        Self {
            convergence_visualizer: ConvergenceVisualizer::default(),
            error_engine: ErrorVisualizationEngine::new(),
            performance_tracker: PerformanceTracker::default(),
        }
    }

    /// Track metric over time
    pub fn track_metric(&mut self, metricname: &str, value: f64, time: f64) {
        self.performance_tracker
            .add_metric_value(metricname, value, time);
    }

    /// Create comprehensive convergence analysis
    pub fn create_convergence_plot(
        &self,
        residuals: &Array1<f64>,
        algorithm_name: &str,
    ) -> IntegrateResult<ConvergencePlot> {
        self.convergence_visualizer
            .plot_residual_convergence(residuals, algorithm_name)
    }

    /// Create multi-metric analysis
    pub fn create_multi_metric_plot(
        &self,
        metrics: &[(String, Array1<f64>)],
    ) -> IntegrateResult<MultiMetricConvergencePlot> {
        self.convergence_visualizer
            .plot_multi_metric_convergence(metrics)
    }

    /// Analyze step size effects
    pub fn create_step_size_analysis(
        &self,
        step_sizes: &Array1<f64>,
        errors: &Array1<f64>,
        method_name: &str,
    ) -> IntegrateResult<StepSizeAnalysisPlot> {
        self.convergence_visualizer
            .plot_step_size_analysis(step_sizes, errors, method_name)
    }
}

impl Default for ConvergenceVisualizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance tracking for algorithm metrics
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    /// Metric data storage
    pub metrics: HashMap<String, Vec<(f64, f64)>>, // (time, value) pairs
    /// Metric statistics
    pub statistics: HashMap<String, MetricStatistics>,
}

impl PerformanceTracker {
    /// Create new performance tracker
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            statistics: HashMap::new(),
        }
    }

    /// Add metric value at specific time
    pub fn add_metric_value(&mut self, metricname: &str, value: f64, time: f64) {
        // This would be implemented to track metrics over time
        // Simplified for now
    }

    /// Get metric statistics
    pub fn get_statistics(metricname: &str) -> Option<&MetricStatistics> {
        // This would return computed statistics for the metric
        None
    }

    /// Get all tracked metric names
    pub fn get_metricnames(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for performance metrics
#[derive(Debug, Clone)]
pub struct MetricStatistics {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Total number of samples
    pub count: usize,
}

/// Information about convergence behavior
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether convergence was achieved
    pub converged: bool,
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Final residual value
    pub final_residual: f64,
    /// Convergence rate estimate
    pub convergence_rate: f64,
}
