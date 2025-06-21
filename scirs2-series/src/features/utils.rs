//! Utility functions for time series feature extraction
//!
//! This module contains utility functions that are used across multiple
//! feature extraction algorithms, including mathematical operations,
//! data transformations, pattern detection, and statistical computations.

use ndarray::{s, Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::config::TurningPointsConfig;
use crate::error::{Result, TimeSeriesError};

/// Type alias for spectral peak detection results
pub type SpectralPeakResult<F> = (Vec<F>, Vec<F>, Vec<F>, Vec<F>, usize, F, F, Vec<F>);

/// Type alias for phase spectrum analysis results
pub type PhaseSpectrumResult<F> = (
    Vec<F>,
    Vec<F>,
    F,
    PhaseSpectrumFeatures<F>,
    BispectrumFeatures<F>,
);

/// Type alias for complex frequency feature extraction results
pub type FrequencyFeatureResult<F> = (F, F, F, F, F, F, F, F);

/// Type alias for multiscale spectral analysis results  
pub type MultiscaleSpectralResult<F> = (Vec<F>, Vec<ScaleSpectralFeatures<F>>, Vec<F>, F);

/// Type alias for cross frequency coupling analysis results
pub type CrossFrequencyCouplingResult<F> = (F, F, F, F, F, F, F, Vec<F>, Vec<F>, F);

// Forward declarations for types used in type aliases

/// Phase spectrum analysis features
#[derive(Debug, Clone, Default)]
pub struct PhaseSpectrumFeatures<F> {
    /// Mean phase value
    pub phase_mean: F,
    /// Standard deviation of phase
    pub phase_std: F,
    /// Phase entropy measure
    pub phase_entropy: F,
}

/// Bispectrum analysis features
#[derive(Debug, Clone, Default)]
pub struct BispectrumFeatures<F> {
    /// Peak value in bispectrum
    pub bispectrum_peak: F,
    /// Entropy of bispectrum
    pub bispectrum_entropy: F,
}

/// Scale-based spectral analysis features
#[derive(Debug, Clone, Default)]
pub struct ScaleSpectralFeatures<F> {
    /// Scale parameter
    pub scale: F,
    /// Energy at this scale
    pub energy: F,
    /// Entropy at this scale
    pub entropy: F,
}

// ============================================================================
// Basic Statistical Functions
// ============================================================================

/// Find minimum and maximum values in a time series
pub fn find_min_max<F>(ts: &Array1<F>) -> (F, F)
where
    F: Float + FromPrimitive,
{
    let mut min_val = F::infinity();
    let mut max_val = F::neg_infinity();

    for &x in ts.iter() {
        if x < min_val {
            min_val = x;
        }
        if x > max_val {
            max_val = x;
        }
    }

    (min_val, max_val)
}

/// Calculate median of a time series
pub fn calculate_median<F>(ts: &Array1<F>) -> F
where
    F: Float + FromPrimitive + Clone,
{
    let mut sorted: Vec<F> = ts.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / F::from(2.0).unwrap()
    } else {
        sorted[n / 2]
    }
}

/// Calculate standard deviation of a time series
pub fn calculate_std_dev<F>(ts: &Array1<F>) -> F
where
    F: Float + FromPrimitive,
{
    let n = ts.len();
    let mean = ts.sum() / F::from(n).unwrap();
    let variance = ts.mapv(|x| (x - mean) * (x - mean)).sum() / F::from(n).unwrap();
    variance.sqrt()
}

/// Calculate percentile from sorted data
pub fn calculate_percentile<F>(sorted: &[F], percentile: f64) -> F
where
    F: Float + FromPrimitive,
{
    let n = sorted.len();
    if n == 0 {
        return F::zero();
    }

    let index = (percentile / 100.0) * (n - 1) as f64;
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;

    if lower_index == upper_index {
        sorted[lower_index]
    } else {
        let fraction = F::from(index - lower_index as f64).unwrap();
        sorted[lower_index] + fraction * (sorted[upper_index] - sorted[lower_index])
    }
}

// ============================================================================
// Linear Regression and Correlation
// ============================================================================

/// Simple linear fit for two variables
pub fn linear_fit<F>(x: &[F], y: &[F]) -> (F, F)
where
    F: Float + FromPrimitive,
{
    let n = x.len() as f64;
    if n < 2.0 {
        return (F::zero(), F::zero());
    }

    let n_f = F::from(n).unwrap();
    let sum_x = x.iter().fold(F::zero(), |acc, &xi| acc + xi);
    let sum_y = y.iter().fold(F::zero(), |acc, &yi| acc + yi);
    let sum_xy = x
        .iter()
        .zip(y.iter())
        .fold(F::zero(), |acc, (&xi, &yi)| acc + xi * yi);
    let sum_x2 = x.iter().fold(F::zero(), |acc, &xi| acc + xi * xi);

    let denominator = n_f * sum_x2 - sum_x * sum_x;
    if denominator == F::zero() {
        return (F::zero(), sum_y / n_f);
    }

    let slope = (n_f * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n_f;

    (slope, intercept)
}

/// Calculate Pearson correlation coefficient between two arrays
pub fn calculate_pearson_correlation<F>(x: &Array1<F>, y: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    if x.len() != y.len() {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Arrays must have the same length for correlation calculation".to_string(),
        ));
    }

    let n = x.len();
    if n < 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "At least 2 points required for correlation calculation".to_string(),
        ));
    }

    let n_f = F::from_usize(n).unwrap();

    // Calculate means
    let mean_x = x.iter().fold(F::zero(), |acc, &val| acc + val) / n_f;
    let mean_y = y.iter().fold(F::zero(), |acc, &val| acc + val) / n_f;

    // Calculate correlation components
    let mut numerator = F::zero();
    let mut sum_sq_x = F::zero();
    let mut sum_sq_y = F::zero();

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator = numerator + dx * dy;
        sum_sq_x = sum_sq_x + dx * dx;
        sum_sq_y = sum_sq_y + dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == F::zero() {
        Ok(F::zero())
    } else {
        Ok(numerator / denominator)
    }
}

// ============================================================================
// Data Transformation Functions
// ============================================================================

/// Discretize and get probability distribution
pub fn discretize_and_get_probabilities<F>(ts: &Array1<F>, n_bins: usize) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let (min_val, max_val) = find_min_max(ts);
    if min_val == max_val {
        return Ok(vec![F::one(); n_bins]);
    }

    let mut counts = vec![0; n_bins];
    for &value in ts.iter() {
        let bin = discretize_value(value, min_val, max_val, n_bins);
        counts[bin] += 1;
    }

    let n_f = F::from(ts.len()).unwrap();
    let probabilities = counts
        .into_iter()
        .map(|count| F::from(count).unwrap() / n_f)
        .collect();

    Ok(probabilities)
}

/// Discretize a single value into a bin
pub fn discretize_value<F>(value: F, min_val: F, max_val: F, n_bins: usize) -> usize
where
    F: Float + FromPrimitive,
{
    let range = max_val - min_val;
    if range == F::zero() {
        return 0;
    }

    let normalized = (value - min_val) / range;
    let bin = (normalized * F::from(n_bins).unwrap())
        .to_usize()
        .unwrap_or(0);
    bin.min(n_bins - 1)
}

/// Coarse grain time series for multiscale analysis
pub fn coarse_grain_series<F>(ts: &Array1<F>, scale: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if scale == 1 {
        return Ok(ts.clone());
    }

    let n = ts.len() / scale;
    let mut coarse_grained = Vec::with_capacity(n);

    for i in 0..n {
        let start = i * scale;
        let end = (start + scale).min(ts.len());
        let sum = (start..end).fold(F::zero(), |acc, j| acc + ts[j]);
        coarse_grained.push(sum / F::from(end - start).unwrap());
    }

    Ok(Array1::from_vec(coarse_grained))
}

/// Refined coarse grain series with offset
pub fn refined_coarse_grain_series<F>(
    ts: &Array1<F>,
    scale: usize,
    offset: usize,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if scale == 1 {
        return Ok(ts.clone());
    }

    let mut coarse_grained = Vec::new();
    let mut i = offset;

    while i + scale <= ts.len() {
        let sum = (i..i + scale).fold(F::zero(), |acc, j| acc + ts[j]);
        coarse_grained.push(sum / F::from(scale).unwrap());
        i += scale;
    }

    Ok(Array1::from_vec(coarse_grained))
}

// ============================================================================
// Downsampling Functions
// ============================================================================

/// Downsample signal by taking every nth sample
pub fn downsample_signal<F>(ts: &Array1<F>, factor: usize) -> Result<Array1<F>>
where
    F: Float + Clone,
{
    if factor <= 1 {
        return Ok(ts.clone());
    }

    let downsampled: Vec<F> = ts.iter().step_by(factor).cloned().collect();

    Ok(Array1::from_vec(downsampled))
}

/// Downsample time series
pub fn downsample_series<F>(ts: &Array1<F>, factor: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if factor == 1 {
        return Ok(ts.clone());
    }

    let downsampled: Vec<F> = ts.iter().step_by(factor).cloned().collect();
    Ok(Array1::from_vec(downsampled))
}

// ============================================================================
// Pattern Detection Functions
// ============================================================================

/// Get ordinal pattern from a window
pub fn get_ordinal_pattern<F>(window: &ArrayView1<F>) -> Vec<usize>
where
    F: Float + FromPrimitive,
{
    let mut indices: Vec<usize> = (0..window.len()).collect();
    indices.sort_by(|&i, &j| window[i].partial_cmp(&window[j]).unwrap());
    indices
}

/// Find local extrema in a signal
pub fn find_local_extrema<F>(signal: &Array1<F>, find_maxima: bool) -> Result<(Vec<usize>, Vec<F>)>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = signal.len();
    let mut indices = Vec::new();
    let mut values = Vec::new();

    // Add boundary points with appropriate extension
    if n < 3 {
        return Ok((indices, values));
    }

    // Check for extrema in the interior
    for i in 1..(n - 1) {
        let is_extremum = if find_maxima {
            signal[i] > signal[i - 1] && signal[i] > signal[i + 1]
        } else {
            signal[i] < signal[i - 1] && signal[i] < signal[i + 1]
        };

        if is_extremum {
            indices.push(i);
            values.push(signal[i]);
        }
    }

    Ok((indices, values))
}

/// Detect turning points in time series
pub fn detect_turning_points<F>(
    ts: &Array1<F>,
    config: &TurningPointsConfig,
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    let window_size = config.extrema_window_size;
    let threshold = F::from(config.min_turning_point_threshold).unwrap();

    let mut turning_points = Vec::new();
    let mut local_maxima = Vec::new();
    let mut local_minima = Vec::new();

    // Calculate relative threshold based on data range
    let min_val = ts.iter().fold(F::infinity(), |a, &b| a.min(b));
    let max_val = ts.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
    let range = max_val - min_val;
    let abs_threshold = threshold * range;

    // Detect local extrema using sliding window
    for i in window_size..n - window_size {
        let current = ts[i];
        let window_start = i - window_size;
        let window_end = i + window_size + 1;

        // Check if current point is local maximum
        let is_local_max = ts
            .slice(s![window_start..window_end])
            .iter()
            .all(|&x| current >= x);

        // Check if current point is local minimum
        let is_local_min = ts
            .slice(s![window_start..window_end])
            .iter()
            .all(|&x| current <= x);

        if is_local_max && (i == 0 || (current - ts[i - 1]).abs() >= abs_threshold) {
            local_maxima.push(i);
            turning_points.push(i);
        }

        if is_local_min && (i == 0 || (current - ts[i - 1]).abs() >= abs_threshold) {
            local_minima.push(i);
            turning_points.push(i);
        }
    }

    Ok((turning_points, local_maxima, local_minima))
}

// ============================================================================
// Distance and Similarity Functions
// ============================================================================

/// Calculate Euclidean distance between two subsequences
pub fn euclidean_distance_subsequence<F>(
    ts: &Array1<F>,
    start1: usize,
    start2: usize,
    length: usize,
) -> F
where
    F: Float + FromPrimitive,
{
    let mut sum = F::zero();
    for i in 0..length {
        if start1 + i < ts.len() && start2 + i < ts.len() {
            let diff = ts[start1 + i] - ts[start2 + i];
            sum = sum + diff * diff;
        }
    }
    sum.sqrt()
}

// ============================================================================
// Interpolation Functions
// ============================================================================

/// Linear interpolation between points
pub fn linear_interpolate<F>(x: usize, indices: &[usize], values: &[F]) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if indices.is_empty() {
        return Ok(F::zero());
    }

    if indices.len() == 1 {
        return Ok(values[0]);
    }

    // Find the two nearest points
    let mut left_idx = 0;
    let mut right_idx = indices.len() - 1;

    for i in 0..(indices.len() - 1) {
        if indices[i] <= x && x <= indices[i + 1] {
            left_idx = i;
            right_idx = i + 1;
            break;
        }
    }

    let x1 = F::from(indices[left_idx]).unwrap();
    let x2 = F::from(indices[right_idx]).unwrap();
    let y1 = values[left_idx];
    let y2 = values[right_idx];

    if x2 == x1 {
        return Ok(y1);
    }

    let x_f = F::from(x).unwrap();
    let interpolated = y1 + (y2 - y1) * (x_f - x1) / (x2 - x1);

    Ok(interpolated)
}

/// Cubic interpolation (fallback to linear for now)
pub fn cubic_interpolate<F>(x: usize, indices: &[usize], values: &[F]) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // For simplicity, fall back to linear interpolation
    // In practice, implement proper cubic spline interpolation
    linear_interpolate(x, indices, values)
}

// ============================================================================
// Statistical Utility Functions
// ============================================================================

/// Get Gaussian breakpoints for SAX conversion
pub fn gaussian_breakpoints(alphabet_size: usize) -> Vec<f64> {
    match alphabet_size {
        2 => vec![0.0],
        3 => vec![-0.43, 0.43],
        4 => vec![-0.67, 0.0, 0.67],
        5 => vec![-0.84, -0.25, 0.25, 0.84],
        6 => vec![-0.97, -0.43, 0.0, 0.43, 0.97],
        7 => vec![-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
        8 => vec![-1.15, -0.67, -0.32, 0.0, 0.32, 0.67, 1.15],
        _ => {
            // For larger alphabets, use normal distribution inverse CDF
            let mut breakpoints = Vec::new();
            for i in 1..alphabet_size {
                let p = i as f64 / alphabet_size as f64;
                // Calculate the z-score for cumulative probability p
                let z = standard_normal_quantile(p);
                breakpoints.push(z);
            }
            breakpoints
        }
    }
}

/// Standard normal quantile function (inverse CDF)
pub fn standard_normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-10 {
        return 0.0;
    }

    // Use the inverse of the error function (erf^-1)
    // Normal quantile: sqrt(2) * erf^-1(2*p - 1)
    // We'll use a rational approximation

    let x = 2.0 * p - 1.0; // Convert to erf domain [-1, 1]

    if x.abs() > 0.7 {
        // Use tail approximation for extreme values
        let sign = if x > 0.0 { 1.0 } else { -1.0 };
        let w = -((1.0 - x.abs()).ln());

        if w < 5.0 {
            let w = w - 2.5;
            let z = 2.81022636e-08;
            let z = z * w + 3.43273939e-07;
            let z = z * w - 3.5233877e-06;
            let z = z * w - 4.39150654e-06;
            let z = z * w + 0.00021858087;
            let z = z * w - 0.00125372503;
            let z = z * w - 0.00417768164;
            let z = z * w + 0.006531194649;
            let z = z * w + 0.005504751339;
            let z = z * w + 0.00713309612;
            let z = z * w + 0.0021063958;
            let z = z * w + (-0.008198294287);

            sign * (w.sqrt() * z + w.sqrt())
        } else {
            let w = w.sqrt();
            let z = 6.657_904_643_501_103;
            let z = z * w + 5.463_784_911_164_114;
            let z = z * w + 1.784_826_539_917_291_3;
            let z = z * w + 0.296_560_571_828_504_87;
            let z = z * w + 0.026_532_189_526_576_124;
            let z = z * w + 0.001_242_660_947_388_078_4;
            let z = z * w + 0.000_027_115_555_687_434_876;
            let z = z * w + 0.000_002_010_334_399_292_288;

            sign * z
        }
    } else {
        // Use central approximation for |x| <= 0.7
        let x2 = x * x;
        let z = x * (1.0 - x2 / 3.0 + x2 * x2 * 2.0 / 15.0);
        z * std::f64::consts::FRAC_2_SQRT_PI.sqrt()
    }
}

/// Calculate entropy from class counts
pub fn calculate_entropy(class1_count: usize, class2_count: usize) -> f64 {
    let total = class1_count + class2_count;
    if total == 0 {
        return 0.0;
    }

    let p1 = class1_count as f64 / total as f64;
    let p2 = class2_count as f64 / total as f64;

    let mut entropy = 0.0;
    if p1 > 0.0 {
        entropy -= p1 * p1.ln();
    }
    if p2 > 0.0 {
        entropy -= p2 * p2.ln();
    }

    entropy
}

// ============================================================================
// Robust Statistical Functions
// ============================================================================

/// Calculate median absolute deviation
pub fn calculate_mad<F>(ts: &Array1<F>, median: F) -> Result<F>
where
    F: Float + FromPrimitive,
{
    let n = ts.len();
    if n == 0 {
        return Ok(F::zero());
    }

    let mut deviations: Vec<F> = ts.iter().map(|&x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Ok(if n % 2 == 0 {
        (deviations[n / 2 - 1] + deviations[n / 2]) / F::from(2.0).unwrap()
    } else {
        deviations[n / 2]
    })
}

/// Calculate trimmed mean
pub fn calculate_trimmed_mean<F>(ts: &Array1<F>, trim_fraction: f64) -> Result<F>
where
    F: Float + FromPrimitive,
{
    let n = ts.len();
    if n == 0 {
        return Ok(F::zero());
    }

    let mut sorted = ts.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let trim_count = (n as f64 * trim_fraction).floor() as usize;
    let start = trim_count;
    let end = n - trim_count;

    if start >= end {
        return Ok(sorted[n / 2]); // Return median if too much trimming
    }

    let sum = sorted[start..end].iter().fold(F::zero(), |acc, &x| acc + x);
    let count = F::from(end - start).unwrap();

    Ok(sum / count)
}

/// Calculate winsorized mean
pub fn calculate_winsorized_mean<F>(ts: &Array1<F>, winsor_fraction: f64) -> Result<F>
where
    F: Float + FromPrimitive,
{
    let n = ts.len();
    if n == 0 {
        return Ok(F::zero());
    }

    let mut sorted = ts.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let winsor_count = (n as f64 * winsor_fraction).floor() as usize;

    // Winsorize: replace extreme values
    let lower_bound = sorted[winsor_count];
    let upper_bound = sorted[n - winsor_count - 1];

    let winsorized: Vec<F> = ts
        .iter()
        .map(|&x| {
            if x < lower_bound {
                lower_bound
            } else if x > upper_bound {
                upper_bound
            } else {
                x
            }
        })
        .collect();

    let sum = winsorized.iter().fold(F::zero(), |acc, &x| acc + x);
    let count = F::from(n).unwrap();

    Ok(sum / count)
}

// ============================================================================
// Spectral Analysis Functions
// ============================================================================

/// Compute power spectrum from autocorrelation
pub fn compute_power_spectrum<F>(acf: &Array1<F>) -> Array1<F>
where
    F: Float + FromPrimitive + Clone,
{
    // Simple power spectrum estimation using autocorrelation
    // In practice, this would use FFT of the autocorrelation function
    // For now, we'll approximate by taking the squared magnitude of ACF
    acf.mapv(|x| x * x)
}
