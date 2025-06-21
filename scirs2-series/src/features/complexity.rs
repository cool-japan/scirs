//! Complexity and entropy analysis features for time series.
//!
//! This module provides comprehensive complexity measures including:
//! - Various entropy measures (Shannon, Rényi, Tsallis, permutation, etc.)
//! - Fractal dimension calculations (Higuchi, box counting)
//! - Chaos theory measures (Hurst exponent, DFA exponent, Lyapunov exponents)
//! - Information theory measures (mutual information, transfer entropy)
//! - Lempel-Ziv complexity and other algorithmic complexity measures
//!
//! # Examples
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_series::features::complexity::*;
//!
//! let ts = Array1::from_vec(vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0]);
//! let approx_entropy = calculate_approximate_entropy(&ts, 2, 0.1).unwrap();
//! let perm_entropy = calculate_permutation_entropy(&ts, 3).unwrap();
//! let lz_complexity = calculate_lempel_ziv_complexity(&ts).unwrap();
//! ```

use crate::error::{Result, TimeSeriesError};
use ndarray::{s, Array1};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

// Import functions from utils module
use super::utils::{
    calculate_std_dev, coarse_grain_series, discretize_and_get_probabilities, discretize_value,
    find_min_max, get_ordinal_pattern, linear_fit, refined_coarse_grain_series,
};

/// Calculate approximate entropy
///
/// Approximate entropy (ApEn) measures the regularity and complexity of time series data.
/// It quantifies the likelihood that patterns of observations that are close remain close
/// for incremented template lengths.
///
/// # Arguments
/// * `ts` - Input time series
/// * `m` - Pattern length (typically 2)
/// * `r` - Tolerance for matching patterns (typically 0.1 * std_dev)
///
/// # Returns
/// Approximate entropy value (higher values indicate more irregularity)
#[allow(dead_code)]
pub fn calculate_approximate_entropy<F>(ts: &Array1<F>, m: usize, r: F) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < m + 1 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for approximate entropy calculation".to_string(),
        ));
    }

    let n = ts.len();

    // Create embedding vectors
    let mut phi_m = F::zero();
    let mut phi_m_plus_1 = F::zero();

    // Phi(m)
    for i in 0..=n - m {
        let mut count = F::zero();

        for j in 0..=n - m {
            // Check if vectors are within tolerance r
            let mut is_match = true;

            for k in 0..m {
                let x = *ts.get(i + k).unwrap();
                let y = *ts.get(j + k).unwrap();
                if (x - y).abs() > r {
                    is_match = false;
                    break;
                }
            }

            if is_match {
                count = count + F::one();
            }
        }

        phi_m = phi_m + (count / F::from_usize(n - m + 1).unwrap()).ln();
    }

    phi_m = phi_m / F::from_usize(n - m + 1).unwrap();

    // Phi(m+1)
    for i in 0..=n - m - 1 {
        let mut count = F::zero();

        for j in 0..=n - m - 1 {
            // Check if vectors are within tolerance r
            let mut is_match = true;

            for k in 0..m + 1 {
                let x = *ts.get(i + k).unwrap();
                let y = *ts.get(j + k).unwrap();
                if (x - y).abs() > r {
                    is_match = false;
                    break;
                }
            }

            if is_match {
                count = count + F::one();
            }
        }

        phi_m_plus_1 = phi_m_plus_1 + (count / F::from_usize(n - m).unwrap()).ln();
    }

    phi_m_plus_1 = phi_m_plus_1 / F::from_usize(n - m).unwrap();

    // Approximate entropy is phi_m - phi_(m+1)
    Ok(phi_m - phi_m_plus_1)
}

/// Calculate sample entropy
///
/// Sample entropy (SampEn) is a modification of approximate entropy that eliminates
/// self-matching to provide a more consistent and less biased complexity measure.
///
/// # Arguments
/// * `ts` - Input time series
/// * `m` - Pattern length (typically 2)
/// * `r` - Tolerance for matching patterns
///
/// # Returns
/// Sample entropy value (higher values indicate more irregularity)
#[allow(dead_code)]
pub fn calculate_sample_entropy<F>(ts: &Array1<F>, m: usize, r: F) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < m + 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for sample entropy calculation".to_string(),
        ));
    }

    let n = ts.len();

    // Count matches for m and m+1
    let mut a = F::zero(); // Number of template matches of length m+1
    let mut b = F::zero(); // Number of template matches of length m

    for i in 0..n - m {
        for j in i + 1..n - m {
            // Check match for length m
            let mut is_match_m = true;

            for k in 0..m {
                let x = *ts.get(i + k).unwrap();
                let y = *ts.get(j + k).unwrap();
                if (x - y).abs() > r {
                    is_match_m = false;
                    break;
                }
            }

            if is_match_m {
                b = b + F::one();

                // Check additional element for m+1
                let x = *ts.get(i + m).unwrap();
                let y = *ts.get(j + m).unwrap();
                if (x - y).abs() <= r {
                    a = a + F::one();
                }
            }
        }
    }

    // Calculate sample entropy
    if b == F::zero() {
        // When no matches are found for template length m, it indicates high irregularity
        // Return a high entropy value (e.g., ln(n)) as a reasonable default
        // This is mathematically sound as it represents maximum possible entropy
        return Ok(F::from_f64(n as f64).unwrap().ln());
    }

    if a == F::zero() {
        // This is actually infinity, but we'll return a large value
        return Ok(F::from_f64(100.0).unwrap());
    }

    Ok(-((a / b).ln()))
}

/// Calculate permutation entropy
///
/// Permutation entropy quantifies the complexity of a time series by examining
/// the ordinal patterns in the data. It's robust to noise and works well for
/// non-stationary time series.
///
/// # Arguments
/// * `ts` - Input time series
/// * `order` - Order of the permutation patterns (typically 3-7)
///
/// # Returns
/// Permutation entropy value
#[allow(dead_code)]
pub fn calculate_permutation_entropy<F>(ts: &Array1<F>, order: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    if n < order {
        return Err(TimeSeriesError::InsufficientData {
            message: "Time series too short for permutation entropy".to_string(),
            required: order,
            actual: n,
        });
    }

    let mut pattern_counts = HashMap::new();
    let mut total_patterns = 0;

    // Generate all permutation patterns
    for i in 0..=(n - order) {
        let mut indices: Vec<usize> = (0..order).collect();
        let window: Vec<F> = (0..order).map(|j| ts[i + j]).collect();

        // Sort indices by corresponding values
        indices.sort_by(|&a, &b| {
            window[a]
                .partial_cmp(&window[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Convert to pattern string
        let pattern = indices.iter().map(|&x| x as u8).collect::<Vec<u8>>();
        *pattern_counts.entry(pattern).or_insert(0) += 1;
        total_patterns += 1;
    }

    // Calculate entropy
    let mut entropy = F::zero();
    for &count in pattern_counts.values() {
        if count > 0 {
            let p = F::from(count).unwrap() / F::from(total_patterns).unwrap();
            entropy = entropy - p * p.ln();
        }
    }

    Ok(entropy)
}

/// Calculate Lempel-Ziv complexity
///
/// Lempel-Ziv complexity measures the number of distinct substrings in a sequence,
/// providing a measure of algorithmic complexity.
///
/// # Arguments
/// * `ts` - Input time series
///
/// # Returns
/// Normalized Lempel-Ziv complexity value
#[allow(dead_code)]
pub fn calculate_lempel_ziv_complexity<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    // Convert to binary sequence based on median
    let median = {
        let mut sorted: Vec<F> = ts.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / F::from(2.0).unwrap()
        } else {
            sorted[n / 2]
        }
    };

    let binary_seq: Vec<u8> = ts
        .iter()
        .map(|&x| if x >= median { 1 } else { 0 })
        .collect();

    // Lempel-Ziv complexity calculation
    let mut complexity = 1;
    let mut i = 0;
    let n = binary_seq.len();

    while i < n {
        let mut l = 1;
        let mut found = false;

        while i + l <= n && !found {
            let pattern = &binary_seq[i..i + l];

            // Look for this pattern in previous subsequences
            for j in 0..i {
                if j + l <= i {
                    let prev_pattern = &binary_seq[j..j + l];
                    if pattern == prev_pattern {
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                l += 1;
            }
        }

        if i + l > n {
            l = n - i;
        }

        i += l;
        complexity += 1;
    }

    // Normalize by sequence length
    Ok(F::from(complexity).unwrap() / F::from(n).unwrap())
}

/// Calculate Higuchi fractal dimension
///
/// Higuchi's method estimates the fractal dimension of a time series by
/// examining the curve length at different scales.
///
/// # Arguments
/// * `ts` - Input time series
/// * `k_max` - Maximum scale parameter (typically 8-20)
///
/// # Returns
/// Fractal dimension value
#[allow(dead_code)]
pub fn calculate_higuchi_fractal_dimension<F>(ts: &Array1<F>, k_max: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut log_k_vec = Vec::new();
    let mut log_l_vec = Vec::new();

    for k in 1..=k_max.min(n / 4) {
        let mut l_m = F::zero();

        for m in 1..=k {
            if m > n {
                continue;
            }

            let mut l_mk = F::zero();
            let max_i = (n - m) / k;

            if max_i == 0 {
                continue;
            }

            for i in 1..=max_i {
                let idx1 = m + i * k - 1;
                let idx2 = m + (i - 1) * k - 1;
                if idx1 < n && idx2 < n {
                    l_mk = l_mk + (ts[idx1] - ts[idx2]).abs();
                }
            }

            l_mk = l_mk * F::from(n - 1).unwrap() / (F::from(max_i * k).unwrap());
            l_m = l_m + l_mk;
        }

        l_m = l_m / F::from(k).unwrap();

        if l_m > F::zero() {
            log_k_vec.push(F::from(k).unwrap().ln());
            log_l_vec.push(l_m.ln());
        }
    }

    if log_k_vec.len() < 2 {
        return Ok(F::zero());
    }

    // Linear regression to find slope
    let n_points = log_k_vec.len();
    let sum_x: F = log_k_vec.iter().fold(F::zero(), |acc, &x| acc + x);
    let sum_y: F = log_l_vec.iter().fold(F::zero(), |acc, &x| acc + x);
    let sum_xy: F = log_k_vec
        .iter()
        .zip(log_l_vec.iter())
        .fold(F::zero(), |acc, (&x, &y)| acc + x * y);
    let sum_xx: F = log_k_vec.iter().fold(F::zero(), |acc, &x| acc + x * x);

    let n_f = F::from(n_points).unwrap();
    let slope = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x);

    Ok(-slope) // Negative because we expect negative slope
}

/// Calculate Hurst exponent using R/S analysis
///
/// The Hurst exponent characterizes the long-term memory of time series.
/// Values around 0.5 indicate random behavior, >0.5 indicate persistence,
/// and <0.5 indicate anti-persistence.
///
/// # Arguments
/// * `ts` - Input time series
///
/// # Returns
/// Hurst exponent value (typically between 0 and 1)
#[allow(dead_code)]
pub fn calculate_hurst_exponent<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    if n < 20 {
        return Ok(F::from(0.5).unwrap());
    }

    estimate_hurst_exponent(ts)
}

/// Calculate DFA (Detrended Fluctuation Analysis) exponent
///
/// DFA quantifies the scaling properties of fluctuations in a time series
/// by removing trends at different scales.
///
/// # Arguments
/// * `ts` - Input time series
///
/// # Returns
/// DFA exponent value
#[allow(dead_code)]
pub fn calculate_dfa_exponent<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    if n < 20 {
        return Ok(F::zero());
    }

    let mean = ts.sum() / F::from(n).unwrap();

    // Create integrated series
    let mut integrated = Array1::zeros(n);
    let mut sum = F::zero();
    for i in 0..n {
        sum = sum + (ts[i] - mean);
        integrated[i] = sum;
    }

    let mut log_f_vec = Vec::new();
    let mut log_n_vec = Vec::new();

    // Calculate fluctuation for different window sizes
    let max_window = n / 4;
    for window_size in (4..=max_window).step_by(2) {
        let num_windows = n / window_size;
        if num_windows == 0 {
            continue;
        }

        let mut fluctuation_sum = F::zero();

        for i in 0..num_windows {
            let start = i * window_size;
            let end = start + window_size;

            if end > n {
                break;
            }

            // Linear detrending of the window
            let window = integrated.slice(ndarray::s![start..end]);
            let x_vals: Array1<F> = (0..window_size).map(|j| F::from(j).unwrap()).collect();

            // Linear regression coefficients
            let x_mean = x_vals.sum() / F::from(window_size).unwrap();
            let y_mean = window.sum() / F::from(window_size).unwrap();

            let mut num = F::zero();
            let mut den = F::zero();
            for j in 0..window_size {
                let x_dev = x_vals[j] - x_mean;
                let y_dev = window[j] - y_mean;
                num = num + x_dev * y_dev;
                den = den + x_dev * x_dev;
            }

            let slope = if den > F::zero() {
                num / den
            } else {
                F::zero()
            };
            let intercept = y_mean - slope * x_mean;

            // Calculate detrended fluctuation
            let mut fluctuation = F::zero();
            for j in 0..window_size {
                let trend_val = intercept + slope * x_vals[j];
                let deviation = window[j] - trend_val;
                fluctuation = fluctuation + deviation * deviation;
            }

            fluctuation_sum = fluctuation_sum + fluctuation;
        }

        let avg_fluctuation =
            (fluctuation_sum / F::from(num_windows * window_size).unwrap()).sqrt();

        if avg_fluctuation > F::zero() {
            log_f_vec.push(avg_fluctuation.ln());
            log_n_vec.push(F::from(window_size).unwrap().ln());
        }
    }

    if log_f_vec.len() < 2 {
        return Ok(F::zero());
    }

    // Linear regression to find DFA exponent
    let n_points = log_f_vec.len();
    let sum_x: F = log_n_vec.iter().fold(F::zero(), |acc, &x| acc + x);
    let sum_y: F = log_f_vec.iter().fold(F::zero(), |acc, &x| acc + x);
    let sum_xy: F = log_n_vec
        .iter()
        .zip(log_f_vec.iter())
        .fold(F::zero(), |acc, (&x, &y)| acc + x * y);
    let sum_xx: F = log_n_vec.iter().fold(F::zero(), |acc, &x| acc + x * x);

    let n_f = F::from(n_points).unwrap();
    let dfa_exponent = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x);

    Ok(dfa_exponent)
}

/// Calculate spectral entropy
///
/// Spectral entropy measures the spectral complexity of a signal by treating
/// the power spectrum as a probability distribution.
///
/// # Arguments
/// * `power_spectrum` - Power spectrum of the signal
///
/// # Returns
/// Spectral entropy value
#[allow(dead_code)]
pub fn calculate_spectral_entropy<F>(power_spectrum: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let total_power: F = power_spectrum.sum();
    if total_power == F::zero() {
        return Ok(F::zero());
    }

    let mut entropy = F::zero();
    for &power in power_spectrum.iter() {
        if power > F::zero() {
            let p = power / total_power;
            entropy = entropy - p * p.ln();
        }
    }

    Ok(entropy)
}

// =================================
// Advanced Entropy Measures
// =================================

/// Calculate Shannon entropy for discretized data
#[allow(dead_code)]
pub fn calculate_shannon_entropy<F>(ts: &Array1<F>, n_bins: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let probabilities = discretize_and_get_probabilities(ts, n_bins)?;

    let mut entropy = F::zero();
    for &p in probabilities.iter() {
        if p > F::zero() {
            entropy = entropy - p * p.ln();
        }
    }

    Ok(entropy)
}

/// Calculate Rényi entropy with parameter alpha
#[allow(dead_code)]
pub fn calculate_renyi_entropy<F>(ts: &Array1<F>, n_bins: usize, alpha: f64) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if alpha == 1.0 {
        return calculate_shannon_entropy(ts, n_bins);
    }

    let probabilities = discretize_and_get_probabilities(ts, n_bins)?;
    let alpha_f = F::from(alpha).unwrap();

    let mut sum = F::zero();
    for &p in probabilities.iter() {
        if p > F::zero() {
            sum = sum + p.powf(alpha_f);
        }
    }

    if sum == F::zero() {
        return Ok(F::zero());
    }

    let entropy = (F::one() / (F::one() - alpha_f)) * sum.ln();
    Ok(entropy)
}

/// Calculate Tsallis entropy with parameter q
#[allow(dead_code)]
pub fn calculate_tsallis_entropy<F>(ts: &Array1<F>, n_bins: usize, q: f64) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if q == 1.0 {
        return calculate_shannon_entropy(ts, n_bins);
    }

    let probabilities = discretize_and_get_probabilities(ts, n_bins)?;
    let q_f = F::from(q).unwrap();

    let mut sum = F::zero();
    for &p in probabilities.iter() {
        if p > F::zero() {
            sum = sum + p.powf(q_f);
        }
    }

    let entropy = (F::one() - sum) / (q_f - F::one());
    Ok(entropy)
}

/// Calculate relative entropy (KL divergence from uniform distribution)
#[allow(dead_code)]
pub fn calculate_relative_entropy<F>(ts: &Array1<F>, n_bins: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let probabilities = discretize_and_get_probabilities(ts, n_bins)?;
    let uniform_prob = F::one() / F::from(n_bins).unwrap();

    let mut kl_div = F::zero();
    for &p in probabilities.iter() {
        if p > F::zero() {
            kl_div = kl_div + p * (p / uniform_prob).ln();
        }
    }

    Ok(kl_div)
}

/// Calculate differential entropy (continuous version)
#[allow(dead_code)]
pub fn calculate_differential_entropy<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < 2 {
        return Ok(F::zero());
    }

    // Use kernel density estimation approach
    let std_dev = calculate_std_dev(ts);
    if std_dev == F::zero() {
        return Ok(F::neg_infinity());
    }

    // Gaussian differential entropy approximation: 0.5 * log(2πe * σ²)
    let pi = F::from(std::f64::consts::PI).unwrap();
    let e = F::from(std::f64::consts::E).unwrap();
    let two = F::from(2.0).unwrap();

    let entropy = F::from(0.5).unwrap() * (two * pi * e * std_dev * std_dev).ln();
    Ok(entropy)
}

/// Calculate weighted permutation entropy
#[allow(dead_code)]
pub fn calculate_weighted_permutation_entropy<F>(ts: &Array1<F>, order: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < order + 1 {
        return Ok(F::zero());
    }

    let mut pattern_weights = std::collections::HashMap::new();
    let mut total_weight = F::zero();

    for i in 0..=n - order {
        let window = &ts.slice(s![i..i + order]);

        // Calculate relative variance as weight
        let mean = window.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(order).unwrap();
        let variance = window.iter().fold(F::zero(), |acc, &x| {
            let diff = x - mean;
            acc + diff * diff
        }) / F::from(order).unwrap();

        let weight = variance.sqrt();

        // Get permutation pattern
        let pattern = get_ordinal_pattern(window);

        let entry = pattern_weights.entry(pattern).or_insert(F::zero());
        *entry = *entry + weight;
        total_weight = total_weight + weight;
    }

    if total_weight == F::zero() {
        return Ok(F::zero());
    }

    // Calculate weighted entropy
    let mut entropy = F::zero();
    for (_, &weight) in pattern_weights.iter() {
        let p = weight / total_weight;
        if p > F::zero() {
            entropy = entropy - p * p.ln();
        }
    }

    Ok(entropy)
}

/// Calculate multiscale entropy
#[allow(dead_code)]
pub fn calculate_multiscale_entropy<F>(
    ts: &Array1<F>,
    n_scales: usize,
    m: usize,
    tolerance_fraction: f64,
) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let mut entropies = Vec::new();
    let std_dev = calculate_std_dev(ts);
    let tolerance = F::from(tolerance_fraction).unwrap() * std_dev;

    for scale in 1..=n_scales {
        let coarse_grained = coarse_grain_series(ts, scale)?;
        let entropy = if coarse_grained.len() >= 10 {
            calculate_sample_entropy(&coarse_grained, m, tolerance)?
        } else {
            F::zero()
        };
        entropies.push(entropy);
    }

    Ok(entropies)
}

/// Calculate refined composite multiscale entropy
#[allow(dead_code)]
pub fn calculate_refined_composite_multiscale_entropy<F>(
    ts: &Array1<F>,
    n_scales: usize,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let std_dev = calculate_std_dev(ts);
    let tolerance = F::from(0.15).unwrap() * std_dev;

    let mut all_entropies = Vec::new();

    for scale in 1..=n_scales {
        // Multiple coarse-graining for each scale
        for j in 0..scale {
            let coarse_grained = refined_coarse_grain_series(ts, scale, j)?;
            if coarse_grained.len() >= 10 {
                let entropy = calculate_sample_entropy(&coarse_grained, 2, tolerance)?;
                all_entropies.push(entropy);
            }
        }
    }

    if all_entropies.is_empty() {
        return Ok(F::zero());
    }

    let sum = all_entropies.iter().fold(F::zero(), |acc, &x| acc + x);
    Ok(sum / F::from(all_entropies.len()).unwrap())
}

// =================================
// Advanced Complexity Measures
// =================================

/// Calculate effective complexity
#[allow(dead_code)]
pub fn calculate_effective_complexity<F>(ts: &Array1<F>, n_bins: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let lz_complexity = calculate_lempel_ziv_complexity(ts)?;
    let entropy = calculate_shannon_entropy(ts, n_bins)?;

    // Effective complexity balances order and disorder
    let max_entropy = F::from(n_bins as f64).unwrap().ln();
    let normalized_entropy = if max_entropy > F::zero() {
        entropy / max_entropy
    } else {
        F::zero()
    };

    // Effective complexity peaks at intermediate values between order and chaos
    let complexity = lz_complexity * normalized_entropy * (F::one() - normalized_entropy);
    Ok(complexity)
}

/// Calculate fractal entropy
#[allow(dead_code)]
pub fn calculate_fractal_entropy<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // Simplified fractal dimension-based entropy
    let fractal_dim = estimate_fractal_dimension(ts)?;
    let max_dim = F::from(2.0).unwrap(); // Maximum for time series

    let normalized_dim = fractal_dim / max_dim;
    let entropy = -normalized_dim * normalized_dim.ln()
        - (F::one() - normalized_dim) * (F::one() - normalized_dim).ln();

    Ok(entropy.abs())
}

/// Calculate DFA entropy
#[allow(dead_code)]
pub fn calculate_dfa_entropy<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // Simplified DFA-based entropy
    let dfa_exponent = estimate_dfa_exponent(ts)?;

    // Convert DFA exponent to entropy measure
    let entropy = -dfa_exponent * dfa_exponent.ln();
    Ok(entropy.abs())
}

/// Calculate multifractal entropy width
#[allow(dead_code)]
pub fn calculate_multifractal_entropy_width<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // Simplified multifractal analysis
    let mut entropies = Vec::new();

    for scale in 2..=8 {
        let coarse_grained = coarse_grain_series(ts, scale)?;
        if coarse_grained.len() > 10 {
            let entropy = calculate_shannon_entropy(&coarse_grained, 8)?;
            entropies.push(entropy);
        }
    }

    if entropies.len() < 2 {
        return Ok(F::zero());
    }

    let max_entropy = entropies.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
    let min_entropy = entropies.iter().fold(F::infinity(), |a, &b| a.min(b));

    Ok(max_entropy - min_entropy)
}

/// Calculate Hurst entropy
#[allow(dead_code)]
pub fn calculate_hurst_entropy<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let hurst_exponent = estimate_hurst_exponent(ts)?;

    // Convert Hurst exponent to entropy measure
    // Hurst = 0.5 (random) -> high entropy
    // Hurst != 0.5 (persistent/anti-persistent) -> lower entropy
    let deviation = (hurst_exponent - F::from(0.5).unwrap()).abs();
    let entropy = F::one() - deviation * F::from(2.0).unwrap();

    Ok(entropy.max(F::zero()))
}

// =================================
// Information Theory Measures
// =================================

/// Calculate entropy rate
#[allow(dead_code)]
pub fn calculate_entropy_rate<F>(ts: &Array1<F>, max_lag: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < max_lag + 2 {
        return Ok(F::zero());
    }

    // Simple approximation using conditional entropy
    let n_bins = 10;
    let joint_entropy = calculate_joint_entropy(ts, max_lag, n_bins)?;
    let conditional_entropy = calculate_conditional_entropy(ts, max_lag, n_bins)?;

    Ok(joint_entropy - conditional_entropy)
}

/// Calculate conditional entropy
#[allow(dead_code)]
pub fn calculate_conditional_entropy<F>(ts: &Array1<F>, lag: usize, n_bins: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < lag + 1 {
        return Ok(F::zero());
    }

    // Estimate H(X_{t+lag} | X_t) using discretization
    let mut joint_counts = std::collections::HashMap::new();
    let mut marginal_counts = std::collections::HashMap::new();

    let (min_val, max_val) = find_min_max(ts);

    for i in 0..n - lag {
        let current_bin = discretize_value(ts[i], min_val, max_val, n_bins);
        let future_bin = discretize_value(ts[i + lag], min_val, max_val, n_bins);

        *joint_counts.entry((current_bin, future_bin)).or_insert(0) += 1;
        *marginal_counts.entry(current_bin).or_insert(0) += 1;
    }

    let total = (n - lag) as f64;
    let mut conditional_entropy = F::zero();

    for ((x_bin, _y_bin), &joint_count) in joint_counts.iter() {
        let marginal_count = marginal_counts[x_bin];

        let p_xy = F::from(joint_count as f64 / total).unwrap();
        let p_x = F::from(marginal_count as f64 / total).unwrap();
        let p_y_given_x = p_xy / p_x;

        if p_y_given_x > F::zero() {
            conditional_entropy = conditional_entropy - p_xy * p_y_given_x.ln();
        }
    }

    Ok(conditional_entropy)
}

/// Calculate mutual information between lagged values
#[allow(dead_code)]
pub fn calculate_mutual_information_lag<F>(ts: &Array1<F>, lag: usize, n_bins: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < lag + 1 {
        return Ok(F::zero());
    }

    let current_entropy = calculate_shannon_entropy(&ts.slice(s![0..n - lag]).to_owned(), n_bins)?;
    let future_entropy = calculate_shannon_entropy(&ts.slice(s![lag..]).to_owned(), n_bins)?;
    let joint_entropy = calculate_joint_entropy(ts, lag, n_bins)?;

    Ok(current_entropy + future_entropy - joint_entropy)
}

/// Calculate transfer entropy
#[allow(dead_code)]
pub fn calculate_transfer_entropy<F>(ts: &Array1<F>, lag: usize, n_bins: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // Simplified transfer entropy calculation
    // TE = H(X_{t+1} | X_t) - H(X_{t+1} | X_t, Y_t)
    // For single series, this becomes more complex - using approximation

    let conditional_entropy_single = calculate_conditional_entropy(ts, 1, n_bins)?;
    let conditional_entropy_multi = calculate_conditional_entropy(ts, lag, n_bins)?;

    Ok((conditional_entropy_single - conditional_entropy_multi).abs())
}

/// Calculate excess entropy (stored information)
#[allow(dead_code)]
pub fn calculate_excess_entropy<F>(ts: &Array1<F>, max_lag: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n_bins = 10;
    let mut block_entropies = Vec::new();

    for block_size in 1..=max_lag {
        let entropy = calculate_block_entropy(ts, block_size, n_bins)?;
        block_entropies.push(entropy);
    }

    if block_entropies.len() < 2 {
        return Ok(F::zero());
    }

    // Excess entropy is the limit of block entropy - block_size * entropy_rate
    // Simplified approximation
    let entropy_rate = (block_entropies[block_entropies.len() - 1] - block_entropies[0])
        / F::from(block_entropies.len() - 1).unwrap();
    let excess =
        block_entropies[block_entropies.len() - 1] - F::from(max_lag).unwrap() * entropy_rate;

    Ok(excess.max(F::zero()))
}

// =================================
// Helper Functions for Complexity
// =================================

/// Calculate joint entropy for two variables
fn calculate_joint_entropy<F>(ts: &Array1<F>, lag: usize, n_bins: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < lag + 1 {
        return Ok(F::zero());
    }

    let (min_val, max_val) = find_min_max(ts);
    let mut joint_counts = std::collections::HashMap::new();

    for i in 0..n - lag {
        let current_bin = discretize_value(ts[i], min_val, max_val, n_bins);
        let future_bin = discretize_value(ts[i + lag], min_val, max_val, n_bins);
        *joint_counts.entry((current_bin, future_bin)).or_insert(0) += 1;
    }

    let total = (n - lag) as f64;
    let mut entropy = F::zero();

    for &count in joint_counts.values() {
        if count > 0 {
            let p = F::from(count as f64 / total).unwrap();
            entropy = entropy - p * p.ln();
        }
    }

    Ok(entropy)
}

/// Calculate block entropy
fn calculate_block_entropy<F>(ts: &Array1<F>, block_size: usize, n_bins: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < block_size {
        return Ok(F::zero());
    }

    let mut block_counts = std::collections::HashMap::new();
    let (min_val, max_val) = find_min_max(ts);

    for i in 0..=n - block_size {
        let mut block_pattern = Vec::new();
        for j in 0..block_size {
            block_pattern.push(discretize_value(ts[i + j], min_val, max_val, n_bins));
        }
        *block_counts.entry(block_pattern).or_insert(0) += 1;
    }

    let total = (n - block_size + 1) as f64;
    let mut entropy = F::zero();

    for &count in block_counts.values() {
        if count > 0 {
            let p = F::from(count as f64 / total).unwrap();
            entropy = entropy - p * p.ln();
        }
    }

    Ok(entropy)
}

/// Estimate fractal dimension using box counting
fn estimate_fractal_dimension<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // Simplified fractal dimension estimation
    let n = ts.len();
    if n < 4 {
        return Ok(F::one());
    }

    let mut box_counts = Vec::new();
    let mut scales = Vec::new();

    for scale in 2..=8 {
        let n_boxes = n / scale;
        if n_boxes > 0 {
            scales.push(scale as f64);
            box_counts.push(n_boxes as f64);
        }
    }

    if scales.len() < 2 {
        return Ok(F::one());
    }

    // Linear regression on log-log plot
    let log_scales: Vec<f64> = scales.iter().map(|x| x.ln()).collect();
    let log_counts: Vec<f64> = box_counts.iter().map(|x| x.ln()).collect();

    let n_points = log_scales.len() as f64;
    let sum_x: f64 = log_scales.iter().sum();
    let sum_y: f64 = log_counts.iter().sum();
    let sum_xy: f64 = log_scales
        .iter()
        .zip(log_counts.iter())
        .map(|(x, y)| x * y)
        .sum();
    let sum_x2: f64 = log_scales.iter().map(|x| x * x).sum();

    let slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x * sum_x);

    Ok(F::from(-slope)
        .unwrap()
        .max(F::zero())
        .min(F::from(3.0).unwrap()))
}

/// Estimate DFA exponent
fn estimate_dfa_exponent<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    // Simplified DFA calculation
    let n = ts.len();
    if n < 10 {
        return Ok(F::from(0.5).unwrap());
    }

    // Calculate cumulative sum
    let mean = ts.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n).unwrap();
    let mut cumsum = Vec::with_capacity(n);
    let mut sum = F::zero();

    for &x in ts.iter() {
        sum = sum + (x - mean);
        cumsum.push(sum);
    }

    // Calculate fluctuation for different window sizes
    let mut fluctuations = Vec::new();
    let mut window_sizes = Vec::new();

    for window_size in (4..n / 4).step_by(4) {
        let n_windows = n / window_size;
        let mut mse_sum = F::zero();

        for i in 0..n_windows {
            let start = i * window_size;
            let end = start + window_size;

            // Linear detrending
            let x_vals: Vec<F> = (0..window_size).map(|j| F::from(j).unwrap()).collect();
            let y_vals: Vec<F> = cumsum[start..end].to_vec();

            let (slope, intercept) = linear_fit(&x_vals, &y_vals);

            let mut mse = F::zero();
            for (j, &y_val) in y_vals.iter().enumerate().take(window_size) {
                let predicted = slope * F::from(j).unwrap() + intercept;
                let residual = y_val - predicted;
                mse = mse + residual * residual;
            }
            mse_sum = mse_sum + mse / F::from(window_size).unwrap();
        }

        let fluctuation = (mse_sum / F::from(n_windows).unwrap()).sqrt();
        fluctuations.push(fluctuation);
        window_sizes.push(window_size);
    }

    if fluctuations.len() < 2 {
        return Ok(F::from(0.5).unwrap());
    }

    // Linear regression on log-log plot
    let log_sizes: Vec<f64> = window_sizes.iter().map(|&x| (x as f64).ln()).collect();
    let log_flucts: Vec<f64> = fluctuations
        .iter()
        .map(|x| x.to_f64().unwrap_or(1.0).ln())
        .collect();

    let n_points = log_sizes.len() as f64;
    let sum_x: f64 = log_sizes.iter().sum();
    let sum_y: f64 = log_flucts.iter().sum();
    let sum_xy: f64 = log_sizes
        .iter()
        .zip(log_flucts.iter())
        .map(|(x, y)| x * y)
        .sum();
    let sum_x2: f64 = log_sizes.iter().map(|x| x * x).sum();

    let slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x * sum_x);

    Ok(F::from(slope).unwrap().max(F::zero()).min(F::one()))
}

/// Estimate Hurst exponent using R/S analysis
fn estimate_hurst_exponent<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < 10 {
        return Ok(F::from(0.5).unwrap());
    }

    let _mean = ts.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n).unwrap();

    // Calculate R/S statistic for different window sizes
    let mut rs_values = Vec::new();
    let mut window_sizes = Vec::new();

    for window_size in (10..n / 2).step_by(10) {
        let n_windows = n / window_size;
        let mut rs_sum = F::zero();

        for i in 0..n_windows {
            let start = i * window_size;
            let end = start + window_size;
            let window = &ts.slice(s![start..end]);

            // Calculate cumulative deviations
            let window_mean =
                window.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(window_size).unwrap();
            let mut cumulative_devs = Vec::with_capacity(window_size);
            let mut sum_dev = F::zero();

            for &x in window.iter() {
                sum_dev = sum_dev + (x - window_mean);
                cumulative_devs.push(sum_dev);
            }

            // Calculate range
            let max_dev = cumulative_devs
                .iter()
                .fold(F::neg_infinity(), |a, &b| a.max(b));
            let min_dev = cumulative_devs.iter().fold(F::infinity(), |a, &b| a.min(b));
            let range = max_dev - min_dev;

            // Calculate standard deviation
            let variance = window.iter().fold(F::zero(), |acc, &x| {
                let diff = x - window_mean;
                acc + diff * diff
            }) / F::from(window_size - 1).unwrap();
            let std_dev = variance.sqrt();

            if std_dev > F::zero() {
                rs_sum = rs_sum + range / std_dev;
            }
        }

        if n_windows > 0 {
            rs_values.push(rs_sum / F::from(n_windows).unwrap());
            window_sizes.push(window_size);
        }
    }

    if rs_values.len() < 2 {
        return Ok(F::from(0.5).unwrap());
    }

    // Linear regression on log-log plot
    let log_sizes: Vec<f64> = window_sizes.iter().map(|&x| (x as f64).ln()).collect();
    let log_rs: Vec<f64> = rs_values
        .iter()
        .map(|x| x.to_f64().unwrap_or(1.0).ln())
        .collect();

    let n_points = log_sizes.len() as f64;
    let sum_x: f64 = log_sizes.iter().sum();
    let sum_y: f64 = log_rs.iter().sum();
    let sum_xy: f64 = log_sizes
        .iter()
        .zip(log_rs.iter())
        .map(|(x, y)| x * y)
        .sum();
    let sum_x2: f64 = log_sizes.iter().map(|x| x * x).sum();

    let hurst = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x * sum_x);

    Ok(F::from(hurst).unwrap().max(F::zero()).min(F::one()))
}

// =================================
// Additional Simple Entropy Functions
// =================================

/// Calculate simple permutation entropy implementation
#[allow(dead_code)]
pub fn calculate_permutation_entropy_simple<F>(signal: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    calculate_permutation_entropy(signal, 3)
}

/// Calculate simple sample entropy implementation
#[allow(dead_code)]
pub fn calculate_sample_entropy_simple<F>(signal: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let std_dev = calculate_std_dev(signal);
    let tolerance = F::from(0.2).unwrap() * std_dev;
    calculate_sample_entropy(signal, 2, tolerance)
}

/// Calculate cross entropy between two signals (simplified)
#[allow(dead_code)]
pub fn calculate_cross_entropy_simple<F>(signal1: &Array1<F>, signal2: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if signal1.len() != signal2.len() {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Signals must have the same length for cross entropy".to_string(),
        ));
    }

    // Simple cross entropy approximation using discretized distributions
    let n_bins = 10;
    let prob1 = discretize_and_get_probabilities(signal1, n_bins)?;
    let prob2 = discretize_and_get_probabilities(signal2, n_bins)?;

    let mut cross_entropy = F::zero();
    for (p1, p2) in prob1.iter().zip(prob2.iter()) {
        if *p1 > F::zero() && *p2 > F::zero() {
            cross_entropy = cross_entropy - (*p1) * p2.ln();
        }
    }

    Ok(cross_entropy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_approximate_entropy() {
        let data = Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        let result = calculate_approximate_entropy(&data, 2, 0.1);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 0.0);
    }

    #[test]
    fn test_sample_entropy() {
        let data = Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        let result = calculate_sample_entropy(&data, 2, 0.1);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 0.0);
    }

    #[test]
    fn test_permutation_entropy() {
        let data = Array1::from_vec(vec![1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0]);
        let result = calculate_permutation_entropy(&data, 3);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 0.0);
    }

    #[test]
    fn test_lempel_ziv_complexity() {
        let data = Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let result = calculate_lempel_ziv_complexity(&data);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value >= 0.0);
        assert!(value <= 1.0);
    }

    #[test]
    fn test_higuchi_fractal_dimension() {
        let data = Array1::from_vec(vec![1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.0]);
        let result = calculate_higuchi_fractal_dimension(&data, 5);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 0.0);
    }

    #[test]
    fn test_hurst_exponent() {
        let data = Array1::from_vec(vec![
            1.0, 1.1, 1.2, 1.15, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8,
            1.85, 1.9, 1.95, 2.0,
        ]);
        let result = calculate_hurst_exponent(&data);
        assert!(result.is_ok());
        let hurst = result.unwrap();
        assert!(hurst >= 0.0);
        assert!(hurst <= 1.0);
    }

    #[test]
    fn test_dfa_exponent() {
        let data = Array1::from_vec(vec![
            1.0, 1.1, 1.2, 1.15, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8,
            1.85, 1.9, 1.95, 2.0,
        ]);
        let result = calculate_dfa_exponent(&data);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 0.0);
    }

    #[test]
    fn test_shannon_entropy() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = calculate_shannon_entropy(&data, 8);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 0.0);
    }

    #[test]
    fn test_multiscale_entropy() {
        let data = Array1::from_vec(vec![
            1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
        ]);
        let result = calculate_multiscale_entropy(&data, 3, 2, 0.1);
        assert!(result.is_ok());
        let entropies = result.unwrap();
        assert_eq!(entropies.len(), 3);
        for entropy in entropies {
            assert!(entropy >= 0.0);
        }
    }
}
