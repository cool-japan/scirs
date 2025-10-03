//! Trend estimation and filtering methods for time series
//!
//! This module provides various methods for estimating and filtering trends in time series data,
//! including:
//! - Spline-based trend estimation (cubic splines, B-splines)
//! - Robust trend filtering methods (Hodrick-Prescott, L1 filtering, Whittaker)
//! - Piecewise trend estimation with automatic or manual breakpoint detection
//! - Trend confidence intervals (bootstrap, parametric, and prediction intervals)

mod confidence;
mod piecewise;
mod robust;
mod seasonal;
mod spline;

// Re-export all public functions and types
pub use confidence::*;
pub use piecewise::*;
pub use robust::*;
pub use seasonal::*;
pub use spline::*;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// Type of spline to use for trend estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineType {
    /// Cubic spline
    Cubic,
    /// Natural cubic spline (second derivatives at endpoints are zero)
    NaturalCubic,
    /// B-spline
    BSpline,
    /// P-spline (penalized B-spline)
    PSpline,
}

/// Knot placement strategy for splines
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnotPlacementStrategy {
    /// Evenly spaced knots
    Uniform,
    /// Knots placed at quantiles of the data
    Quantile,
    /// Custom knot positions
    Custom,
}

/// Options for spline-based trend estimation
#[derive(Debug, Clone)]
pub struct SplineTrendOptions {
    /// Type of spline to use
    pub spline_type: SplineType,
    /// Number of knots to use
    pub num_knots: usize,
    /// Knot placement strategy
    pub knot_placement: KnotPlacementStrategy,
    /// Custom knot positions (used only if knot_placement is Custom)
    pub knot_positions: Option<Vec<usize>>,
    /// Whether to extrapolate beyond the available data
    pub extrapolate: bool,
    /// Degree of the B-spline (used only for BSpline and PSpline)
    pub degree: usize,
    /// Regularization parameter for P-splines (used only for PSpline)
    pub lambda: f64,
}

impl Default for SplineTrendOptions {
    fn default() -> Self {
        SplineTrendOptions {
            spline_type: SplineType::Cubic,
            num_knots: 10,
            knot_placement: KnotPlacementStrategy::Uniform,
            knot_positions: None,
            extrapolate: false,
            degree: 3,
            lambda: 1.0,
        }
    }
}

/// Method for robust trend filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustFilterMethod {
    /// Hodrick-Prescott filter
    HodrickPrescott,
    /// L1 trend filter
    L1Filter,
    /// Whittaker smoother
    Whittaker,
}

/// Options for robust trend filtering
#[derive(Debug, Clone)]
pub struct RobustFilterOptions {
    /// Filtering method to use
    pub method: RobustFilterMethod,
    /// Smoothing parameter (lambda)
    pub lambda: f64,
    /// Order of the difference penalty (1=first differences, 2=second differences, etc.)
    pub order: usize,
    /// Maximum number of iterations for iterative reweighting
    pub max_iter: usize,
    /// Convergence tolerance for iterative reweighting
    pub tol: f64,
    /// Weight function for robust estimation
    pub weight_function: WeightFunction,
    /// Tuning parameter for the weight function
    pub tuning_parameter: f64,
}

impl Default for RobustFilterOptions {
    fn default() -> Self {
        RobustFilterOptions {
            method: RobustFilterMethod::HodrickPrescott,
            lambda: 1600.0,
            order: 2,
            max_iter: 100,
            tol: 1e-6,
            weight_function: WeightFunction::Bisquare,
            tuning_parameter: 4.685,
        }
    }
}

/// Robust weight function for outlier handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFunction {
    /// Huber weight function
    Huber,
    /// Bisquare (Tukey's biweight) function
    Bisquare,
    /// Andrews' sine function
    Andrews,
    /// Cauchy weight function
    Cauchy,
}

/// Method for trend breakpoint detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakpointMethod {
    /// Binary segmentation algorithm
    BinarySegmentation,
    /// PELT (Pruned Exact Linear Time) algorithm
    PELT,
    /// Bottom-up segmentation
    BottomUp,
    /// Custom breakpoints provided by the user
    Custom,
}

/// Criterion for evaluating breakpoint candidates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakpointCriterion {
    /// AIC (Akaike Information Criterion)
    AIC,
    /// BIC (Bayesian Information Criterion)
    BIC,
    /// Residual sum of squares
    RSS,
    /// Modified BIC with stronger penalty
    ModifiedBIC,
}

/// Options for piecewise trend estimation
#[derive(Debug, Clone)]
pub struct PiecewiseTrendOptions {
    /// Method for detecting breakpoints
    pub breakpoint_method: BreakpointMethod,
    /// Criterion for evaluating breakpoints
    pub criterion: BreakpointCriterion,
    /// Minimum segment length between breakpoints
    pub min_segment_length: usize,
    /// Maximum number of breakpoints to detect
    pub max_breakpoints: Option<usize>,
    /// Custom breakpoint positions (used only if method is Custom)
    pub custom_breakpoints: Option<Vec<usize>>,
    /// Penalty parameter for breakpoint detection
    pub penalty: Option<f64>,
    /// Type of model to fit to each segment
    pub segment_model: SegmentModelType,
    /// Whether to allow discontinuities at breakpoints
    pub allow_discontinuities: bool,
}

impl Default for PiecewiseTrendOptions {
    fn default() -> Self {
        PiecewiseTrendOptions {
            breakpoint_method: BreakpointMethod::BinarySegmentation,
            criterion: BreakpointCriterion::BIC,
            min_segment_length: 10,
            max_breakpoints: None,
            custom_breakpoints: None,
            penalty: None,
            segment_model: SegmentModelType::Linear,
            allow_discontinuities: true,
        }
    }
}

/// Type of model to fit to each segment in piecewise trend estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentModelType {
    /// Constant (mean) model
    Constant,
    /// Linear model
    Linear,
    /// Quadratic model
    Quadratic,
    /// Cubic model
    Cubic,
    /// Spline model
    Spline,
}

/// Method for calculating confidence intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceIntervalMethod {
    /// Bootstrap confidence intervals
    Bootstrap,
    /// Parametric confidence intervals
    Parametric,
    /// Prediction intervals
    Prediction,
}

/// Options for confidence interval calculation
#[derive(Debug, Clone)]
pub struct ConfidenceIntervalOptions {
    /// Method for calculating confidence intervals
    pub method: ConfidenceIntervalMethod,
    /// Confidence level (0.95 = 95% confidence interval)
    pub level: f64,
    /// Number of bootstrap samples (for Bootstrap method)
    pub num_bootstrap: usize,
    /// Whether to use a block bootstrap (for time series, handles autocorrelation)
    pub block_bootstrap: bool,
    /// Block length for block bootstrap
    pub block_length: usize,
    /// Whether to compute prediction intervals that include observation noise
    pub include_observation_noise: bool,
    /// Whether to estimate noise variance from residuals
    pub estimate_noise_variance: bool,
    /// Custom noise variance (used only if estimate_noise_variance is false)
    pub noise_variance: Option<f64>,
}

impl Default for ConfidenceIntervalOptions {
    fn default() -> Self {
        ConfidenceIntervalOptions {
            method: ConfidenceIntervalMethod::Bootstrap,
            level: 0.95,
            num_bootstrap: 1000,
            block_bootstrap: true,
            block_length: 10,
            include_observation_noise: true,
            estimate_noise_variance: true,
            noise_variance: None,
        }
    }
}

/// Result of a trend estimation with confidence intervals
#[derive(Debug, Clone)]
pub struct TrendWithConfidenceInterval<F: Float> {
    /// Estimated trend
    pub trend: Array1<F>,
    /// Lower confidence bound
    pub lower: Array1<F>,
    /// Upper confidence bound
    pub upper: Array1<F>,
}

/// Creates a difference matrix of a specified order
///
/// This is a helper function used by multiple trend filtering methods
///
/// # Arguments
///
/// * `n` - Size of the matrix (number of time points)
/// * `order` - Order of the difference matrix (1 for first differences, 2 for second differences, etc.)
///
/// # Returns
///
/// A difference matrix of the specified order
pub(crate) fn create_difference_matrix<F>(n: usize, order: usize) -> Array2<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mut d = Array2::<F>::zeros((n - 1, n));

    // First-order difference matrix
    for i in 0..(n - 1) {
        d[[i, i]] = F::one();
        d[[i, i + 1]] = -F::one();
    }

    // For higher orders, recursively apply the difference operator
    if order > 1 {
        let d_lower = create_difference_matrix::<F>(n - 1, order - 1);
        let mut d_higher = Array2::<F>::zeros((n - order, n));

        for i in 0..(n - order) {
            for j in 0..n {
                let mut sum = F::zero();
                for k in 0..(n - 1) {
                    if j < n - 1 {
                        sum = sum + d_lower[[i, k]] * d[[k, j]];
                    }
                }
                d_higher[[i, j]] = sum;
            }
        }

        return d_higher;
    }

    d
}
