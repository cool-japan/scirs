//! Dynamic Time Warping barycenter averaging for time series.
//!
//! This module provides DTW-specific types, configurations, and computation
//! functions for barycenter averaging of irregular time series.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Configuration for Dynamic Time Warping barycenter averaging
#[derive(Debug, Clone)]
pub struct DTWBarycenterConfig {
    /// Maximum number of iterations for barycenter computation
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Initialization method for barycenter
    pub initialization_method: BarycenterInit,
    /// Weights for each time series (None = equal weights)
    pub weights: Option<Array1<f64>>,
    /// Window constraint for DTW (None = no constraint)
    pub window_constraint: Option<usize>,
    /// Distance metric for DTW
    pub distance_metric: DTWDistance,
    /// Whether to use approximation methods for speed
    pub use_approximation: bool,
}

impl Default for DTWBarycenterConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_tolerance: 1e-6,
            initialization_method: BarycenterInit::Random,
            weights: None,
            window_constraint: None,
            distance_metric: DTWDistance::Euclidean,
            use_approximation: false,
        }
    }
}

/// Initialization methods for barycenter computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarycenterInit {
    /// Random initialization
    Random,
    /// Use first time series as initialization
    First,
    /// Use medoid (most central) time series
    Medoid,
    /// Use mean of all time series (ignoring alignment)
    Mean,
}

/// Distance metrics for DTW
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DTWDistance {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Squared Euclidean distance
    SquaredEuclidean,
}

/// Result of DTW barycenter averaging
#[derive(Debug, Clone)]
pub struct DTWBarycenterResult<F> {
    /// Computed barycenter time series
    pub barycenter: Array1<F>,
    /// Distances from each series to barycenter
    pub distances: Array1<F>,
    /// Number of iterations until convergence
    pub iterations: usize,
    /// Final convergence error
    pub convergence_error: F,
    /// Alignment paths for each series to barycenter
    pub alignment_paths: Vec<Vec<(usize, usize)>>,
    /// Warping costs for each series
    pub warping_costs: Array1<F>,
}

/// Compute DTW barycenter of multiple time series
///
/// # Arguments
///
/// * `_timeseries` - Vector of time series to average
/// * `config` - DTW barycenter configuration
///
/// # Returns
///
/// DTW barycenter result including the computed barycenter and alignment information
#[allow(dead_code)]
pub fn compute_dtw_barycenter<F>(
    _timeseries: &[Array1<F>],
    config: &DTWBarycenterConfig,
) -> Result<DTWBarycenterResult<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    if _timeseries.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "Cannot compute barycenter of empty time _series collection".to_string(),
        ));
    }

    let n_series = _timeseries.len();

    // Initialize weights
    let weights = config
        .weights
        .clone()
        .unwrap_or_else(|| Array1::from_elem(n_series, 1.0 / n_series as f64));

    if weights.len() != n_series {
        return Err(TimeSeriesError::InvalidInput(
            "Weights length must match number of time _series".to_string(),
        ));
    }

    // Initialize barycenter
    let mut barycenter = initialize_barycenter(_timeseries, &config.initialization_method)?;
    let mut prev_barycenter = barycenter.clone();

    let mut convergence_error = F::infinity();
    let mut iterations = 0;

    let mut alignment_paths = Vec::new();
    let mut warping_costs = Array1::zeros(n_series);

    // Iterative barycenter computation
    while iterations < config.max_iterations
        && convergence_error
            > F::from(config.convergence_tolerance).expect("Failed to convert to float")
    {
        alignment_paths.clear();

        // Compute alignments for all _series to current barycenter
        for (i, series) in _timeseries.iter().enumerate() {
            let (cost, path) = compute_dtw_alignment(&barycenter, series, config)?;
            alignment_paths.push(path);
            warping_costs[i] = cost;
        }

        // Update barycenter based on alignments
        barycenter = update_barycenter_from_alignments(_timeseries, &alignment_paths, &weights)?;

        // Check convergence
        convergence_error = compute_barycenter_difference(&barycenter, &prev_barycenter);
        prev_barycenter = barycenter.clone();
        iterations += 1;
    }

    // Compute final distances
    let mut distances = Array1::zeros(n_series);
    for (i, series) in _timeseries.iter().enumerate() {
        let (distance, _) = compute_dtw_alignment(&barycenter, series, config)?;
        distances[i] = distance;
    }

    Ok(DTWBarycenterResult {
        barycenter,
        distances,
        iterations,
        convergence_error,
        alignment_paths,
        warping_costs,
    })
}

// ---------------------------------------------------------------------------
// DTW helper functions
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn initialize_barycenter<F>(_timeseries: &[Array1<F>], method: &BarycenterInit) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    match method {
        BarycenterInit::Random => {
            let median_length = _timeseries.len() / 2;
            let length = _timeseries[median_length].len();
            Ok(Array1::zeros(length))
        }
        BarycenterInit::First => Ok(_timeseries[0].clone()),
        BarycenterInit::Medoid => compute_medoid(_timeseries),
        BarycenterInit::Mean => compute_mean_series(_timeseries),
    }
}

#[allow(dead_code)]
fn compute_medoid<F>(_timeseries: &[Array1<F>]) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let n = _timeseries.len();
    let mut min_total_distance = F::infinity();
    let mut medoid_idx = 0;

    for i in 0..n {
        let mut total_distance = F::zero();
        for j in 0..n {
            if i != j {
                let distance = compute_euclidean_distance(&_timeseries[i], &_timeseries[j]);
                total_distance = total_distance + distance;
            }
        }

        if total_distance < min_total_distance {
            min_total_distance = total_distance;
            medoid_idx = i;
        }
    }

    Ok(_timeseries[medoid_idx].clone())
}

#[allow(dead_code)]
fn compute_mean_series<F>(_timeseries: &[Array1<F>]) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Simple mean ignoring alignment issues
    let max_length = _timeseries.iter().map(|ts| ts.len()).max().unwrap_or(0);
    let mut mean_series = Array1::zeros(max_length);
    let mut counts = Array1::zeros(max_length);

    for ts in _timeseries {
        for (i, &val) in ts.iter().enumerate() {
            mean_series[i] = mean_series[i] + val;
            counts[i] = counts[i] + F::one();
        }
    }

    for i in 0..max_length {
        if counts[i] > F::zero() {
            mean_series[i] = mean_series[i] / counts[i];
        }
    }

    Ok(mean_series)
}

#[allow(dead_code)]
fn compute_dtw_alignment<F>(
    series1: &Array1<F>,
    series2: &Array1<F>,
    config: &DTWBarycenterConfig,
) -> Result<(F, Vec<(usize, usize)>)>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let n1 = series1.len();
    let n2 = series2.len();

    // Initialize DTW matrix
    let mut _dtwmatrix = Array2::from_elem((n1 + 1, n2 + 1), F::infinity());
    _dtwmatrix[(0, 0)] = F::zero();

    // Fill DTW matrix
    for i in 1..=n1 {
        for j in 1..=n2 {
            // Check window constraint
            if let Some(window) = config.window_constraint {
                let _window_f = window as f64;
                let ratio = n1 as f64 / n2 as f64;
                let expected_j = (i as f64 / ratio) as usize;
                if j.abs_diff(expected_j) > window {
                    continue;
                }
            }

            let cost =
                compute_point_distance(series1[i - 1], series2[j - 1], &config.distance_metric);
            let min_prev = _dtwmatrix[(i - 1, j)]
                .min(_dtwmatrix[(i, j - 1)])
                .min(_dtwmatrix[(i - 1, j - 1)]);

            _dtwmatrix[(i, j)] = cost + min_prev;
        }
    }

    let total_cost = _dtwmatrix[(n1, n2)];

    // Backtrack to find optimal path
    let path = backtrack_dtw_path(&_dtwmatrix, n1, n2);

    Ok((total_cost, path))
}

#[allow(dead_code)]
fn compute_point_distance<F>(point1: F, point2: F, metric: &DTWDistance) -> F
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let diff = point1 - point2;

    match metric {
        DTWDistance::Euclidean => diff.abs(),
        DTWDistance::Manhattan => diff.abs(),
        DTWDistance::SquaredEuclidean => diff * diff,
    }
}

#[allow(dead_code)]
fn backtrack_dtw_path<F>(_dtwmatrix: &Array2<F>, n1: usize, n2: usize) -> Vec<(usize, usize)>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let mut path = Vec::new();
    let mut i = n1;
    let mut j = n2;

    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));

        // Find minimum of three predecessors
        let diag = _dtwmatrix[(i - 1, j - 1)];
        let up = _dtwmatrix[(i - 1, j)];
        let left = _dtwmatrix[(i, j - 1)];

        if diag <= up && diag <= left {
            i -= 1;
            j -= 1;
        } else if up <= left {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    path.reverse();
    path
}

#[allow(dead_code)]
fn update_barycenter_from_alignments<F>(
    _timeseries: &[Array1<F>],
    alignment_paths: &[Vec<(usize, usize)>],
    weights: &Array1<f64>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    // Find the maximum barycenter length needed
    let max_barycenter_length = alignment_paths
        .iter()
        .map(|path| path.iter().map(|(i_, _)| *i_).max().unwrap_or(0) + 1)
        .max()
        .unwrap_or(0);

    let mut new_barycenter = Array1::zeros(max_barycenter_length);
    let mut counts = Array1::zeros(max_barycenter_length);

    // Accumulate weighted contributions
    for (series_idx, path) in alignment_paths.iter().enumerate() {
        let weight = F::from(weights[series_idx]).expect("Failed to convert to float");
        let series = &_timeseries[series_idx];

        for &(barycenter_idx, series_idx_in_path) in path {
            if barycenter_idx < max_barycenter_length && series_idx_in_path < series.len() {
                new_barycenter[barycenter_idx] =
                    new_barycenter[barycenter_idx] + weight * series[series_idx_in_path];
                counts[barycenter_idx] = counts[barycenter_idx] + weight;
            }
        }
    }

    // Normalize by counts
    for i in 0..max_barycenter_length {
        if counts[i] > F::zero() {
            new_barycenter[i] = new_barycenter[i] / counts[i];
        }
    }

    Ok(new_barycenter)
}

#[allow(dead_code)]
fn compute_barycenter_difference<F>(barycenter1: &Array1<F>, barycenter2: &Array1<F>) -> F
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let min_len = std::cmp::min(barycenter1.len(), barycenter2.len());
    let mut sum_sq_diff = F::zero();

    for i in 0..min_len {
        let diff = barycenter1[i] - barycenter2[i];
        sum_sq_diff = sum_sq_diff + diff * diff;
    }

    sum_sq_diff.sqrt()
}

#[allow(dead_code)]
pub(super) fn compute_euclidean_distance<F>(series1: &Array1<F>, series2: &Array1<F>) -> F
where
    F: Float + FromPrimitive + Debug + Clone + 'static,
{
    let min_len = std::cmp::min(series1.len(), series2.len());
    let mut sum_sq_diff = F::zero();

    for i in 0..min_len {
        let diff = series1[i] - series2[i];
        sum_sq_diff = sum_sq_diff + diff * diff;
    }

    sum_sq_diff.sqrt()
}
