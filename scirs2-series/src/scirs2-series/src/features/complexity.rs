//! Complexity and entropy features for time series analysis
//!
//! This module provides comprehensive complexity and entropy feature calculation
//! including approximate entropy, sample entropy, permutation entropy, multiscale
//! entropy, fractal dimensions, and other regularity measures.

use ndarray::Array1;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};
use super::utils::{discretize_and_get_probabilities, get_ordinal_pattern, coarse_grain_series};

/// Complexity-based features for time series
#[derive(Debug, Clone)]
pub struct ComplexityFeatures<F> {
    /// Approximate entropy
    pub approximate_entropy: F,
    /// Sample entropy
    pub sample_entropy: F,
    /// Permutation entropy
    pub permutation_entropy: F,
    /// Lempel-Ziv complexity
    pub lempel_ziv_complexity: F,
    /// Fractal dimension (Higuchi's method)
    pub fractal_dimension: F,
    /// Hurst exponent
    pub hurst_exponent: F,
    /// Detrended fluctuation analysis (DFA) exponent
    pub dfa_exponent: F,
    /// Number of turning points
    pub turning_points: usize,
    /// Longest strike (consecutive increases/decreases)
    pub longest_strike: usize,
}

impl<F> Default for ComplexityFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            approximate_entropy: F::zero(),
            sample_entropy: F::zero(),
            permutation_entropy: F::zero(),
            lempel_ziv_complexity: F::zero(),
            fractal_dimension: F::zero(),
            hurst_exponent: F::from(0.5).unwrap(), // Default for random walk
            dfa_exponent: F::from(0.5).unwrap(),   // Default for random walk
            turning_points: 0,
            longest_strike: 0,
        }
    }
}

/// Comprehensive entropy features for time series analysis
#[derive(Debug, Clone)]
pub struct EntropyFeatures<F> {
    // Classical entropy measures
    /// Shannon entropy
    pub shannon_entropy: F,
    /// Rényi entropy (α=2)
    pub renyi_entropy_2: F,
    /// Rényi entropy (α=0.5)
    pub renyi_entropy_05: F,
    /// Tsallis entropy
    pub tsallis_entropy: F,
    /// Relative entropy (KL divergence from uniform)
    pub relative_entropy: F,

    // Differential entropy measures
    /// Differential entropy
    pub differential_entropy: F,
    /// Approximate entropy
    pub approximate_entropy: F,
    /// Sample entropy
    pub sample_entropy: F,
    /// Permutation entropy
    pub permutation_entropy: F,
    /// Weighted permutation entropy
    pub weighted_permutation_entropy: F,

    // Multiscale entropy measures
    /// Multiscale entropy values
    pub multiscale_entropy: Vec<F>,
    /// Composite multiscale entropy
    pub composite_multiscale_entropy: F,
    /// Refined composite multiscale entropy
    pub refined_composite_multiscale_entropy: F,
    /// Entropy rate
    pub entropy_rate: F,

    // Conditional and joint entropy measures
    /// Conditional entropy
    pub conditional_entropy: F,
    /// Mutual information
    pub mutual_information: F,
    /// Transfer entropy
    pub transfer_entropy: F,
    /// Excess entropy
    pub excess_entropy: F,

    // Spectral entropy measures
    /// Spectral entropy
    pub spectral_entropy: F,
    /// Normalized spectral entropy
    pub normalized_spectral_entropy: F,
    /// Wavelet entropy
    pub wavelet_entropy: F,
    /// Packet wavelet entropy
    pub packet_wavelet_entropy: F,

    // Time-frequency entropy measures
    /// Instantaneous entropy values
    pub instantaneous_entropy: Vec<F>,
    /// Mean instantaneous entropy
    pub mean_instantaneous_entropy: F,
    /// Entropy standard deviation
    pub entropy_std: F,
    /// Entropy trend
    pub entropy_trend: F,

    // Symbolic entropy measures
    /// Binary entropy
    pub binary_entropy: F,
    /// Ternary entropy
    pub ternary_entropy: F,
    /// Multisymbol entropy
    pub multisymbol_entropy: F,
    /// Range entropy
    pub range_entropy: F,

    // Distribution-based entropy measures
    /// Increment entropy
    pub increment_entropy: F,
    /// Relative increment entropy
    pub relative_increment_entropy: F,
    /// Absolute increment entropy
    pub absolute_increment_entropy: F,
    /// Squared increment entropy
    pub squared_increment_entropy: F,
}

impl<F> Default for EntropyFeatures<F>
where
    F: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            // Classical entropy measures
            shannon_entropy: F::zero(),
            renyi_entropy_2: F::zero(),
            renyi_entropy_05: F::zero(),
            tsallis_entropy: F::zero(),
            relative_entropy: F::zero(),

            // Differential entropy measures
            differential_entropy: F::zero(),
            approximate_entropy: F::zero(),
            sample_entropy: F::zero(),
            permutation_entropy: F::zero(),
            weighted_permutation_entropy: F::zero(),

            // Multiscale entropy measures
            multiscale_entropy: Vec::new(),
            composite_multiscale_entropy: F::zero(),
            refined_composite_multiscale_entropy: F::zero(),
            entropy_rate: F::zero(),

            // Conditional and joint entropy measures
            conditional_entropy: F::zero(),
            mutual_information: F::zero(),
            transfer_entropy: F::zero(),
            excess_entropy: F::zero(),

            // Spectral entropy measures
            spectral_entropy: F::zero(),
            normalized_spectral_entropy: F::zero(),
            wavelet_entropy: F::zero(),
            packet_wavelet_entropy: F::zero(),

            // Time-frequency entropy measures
            instantaneous_entropy: Vec::new(),
            mean_instantaneous_entropy: F::zero(),
            entropy_std: F::zero(),
            entropy_trend: F::zero(),

            // Symbolic entropy measures
            binary_entropy: F::zero(),
            ternary_entropy: F::zero(),
            multisymbol_entropy: F::zero(),
            range_entropy: F::zero(),

            // Distribution-based entropy measures
            increment_entropy: F::zero(),
            relative_increment_entropy: F::zero(),
            absolute_increment_entropy: F::zero(),
            squared_increment_entropy: F::zero(),
        }
    }
}

// =============================================================================
// Core Entropy Calculation Functions
// =============================================================================

/// Calculate approximate entropy
pub fn calculate_approximate_entropy<F>(ts: &Array1<F>, m: usize, r: F) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < m + 1 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for approximate entropy calculation".to_string(),
        ));
    }

    fn max_dist<F>(xi: &[F], xj: &[F]) -> F
    where
        F: Float,
    {
        xi.iter()
            .zip(xj.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(F::zero(), |acc, d| acc.max(d))
    }

    fn count_matches<F>(ts: &Array1<F>, m: usize, i: usize, r: F) -> usize
    where
        F: Float + Clone,
    {
        let n = ts.len();
        let mut count = 0;
        let pattern_i = &ts.as_slice().unwrap()[i..i + m];

        for j in 0..=n - m {
            let pattern_j = &ts.as_slice().unwrap()[j..j + m];
            if max_dist(pattern_i, pattern_j) <= r {
                count += 1;
            }
        }
        count
    }

    let mut phi_m = F::zero();
    let mut phi_m_plus_1 = F::zero();

    // Calculate φ(m)
    for i in 0..=n - m {
        let count = count_matches(ts, m, i, r);
        if count > 0 {
            phi_m = phi_m + F::from(count).unwrap().ln();
        }
    }
    phi_m = phi_m / F::from(n - m + 1).unwrap();

    // Calculate φ(m+1)
    for i in 0..=n - m - 1 {
        let count = count_matches(ts, m + 1, i, r);
        if count > 0 {
            phi_m_plus_1 = phi_m_plus_1 + F::from(count).unwrap().ln();
        }
    }
    phi_m_plus_1 = phi_m_plus_1 / F::from(n - m).unwrap();

    Ok(phi_m - phi_m_plus_1)
}

/// Calculate sample entropy
pub fn calculate_sample_entropy<F>(ts: &Array1<F>, m: usize, r: F) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let n = ts.len();
    if n < m + 1 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for sample entropy calculation".to_string(),
        ));
    }

    fn max_dist<F>(xi: &[F], xj: &[F]) -> F
    where
        F: Float,
    {
        xi.iter()
            .zip(xj.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(F::zero(), |acc, d| acc.max(d))
    }

    let mut a = 0; // Number of template matches of length m
    let mut b = 0; // Number of template matches of length m+1

    let ts_slice = ts.as_slice().unwrap();

    for i in 0..n - m {
        for j in i + 1..n - m {
            // Check pattern of length m
            if max_dist(&ts_slice[i..i + m], &ts_slice[j..j + m]) <= r {
                b += 1;
                
                // Check pattern of length m+1
                if i < n - m && j < n - m {
                    if max_dist(&ts_slice[i..i + m + 1], &ts_slice[j..j + m + 1]) <= r {
                        a += 1;
                    }
                }
            }
        }
    }

    if b == 0 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "No template matches found for sample entropy".to_string(),
        ));
    }

    let a_f = F::from(a).unwrap();
    let b_f = F::from(b).unwrap();

    Ok(-((a_f / b_f).ln()))
}

/// Calculate permutation entropy
pub fn calculate_permutation_entropy<F>(ts: &Array1<F>, order: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    if ts.len() < order {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Time series too short for permutation entropy calculation".to_string(),
        ));
    }

    let mut pattern_counts = std::collections::HashMap::new();
    let total_patterns = ts.len() - order + 1;

    for i in 0..total_patterns {
        let window = ts.slice(ndarray::s![i..i + order]);
        let pattern = get_ordinal_pattern(&window);
        *pattern_counts.entry(pattern).or_insert(0) += 1;
    }

    let mut entropy = F::zero();
    let total_f = F::from(total_patterns).unwrap();

    for count in pattern_counts.values() {
        let probability = F::from(*count).unwrap() / total_f;
        if probability > F::zero() {
            entropy = entropy - probability * probability.ln();
        }
    }

    Ok(entropy)
}

/// Calculate Shannon entropy from probability distribution
pub fn calculate_shannon_entropy<F>(probabilities: &[F]) -> F
where
    F: Float + FromPrimitive,
{
    let mut entropy = F::zero();
    for &p in probabilities {
        if p > F::zero() {
            entropy = entropy - p * p.ln();
        }
    }
    entropy
}

/// Calculate Rényi entropy
pub fn calculate_renyi_entropy<F>(probabilities: &[F], alpha: F) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if alpha == F::one() {
        return Ok(calculate_shannon_entropy(probabilities));
    }

    if alpha <= F::zero() {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Alpha must be positive for Rényi entropy".to_string(),
        ));
    }

    let mut sum = F::zero();
    for &p in probabilities {
        if p > F::zero() {
            sum = sum + p.powf(alpha);
        }
    }

    if sum == F::zero() {
        return Ok(F::zero());
    }

    let entropy = sum.ln() / (F::one() - alpha);
    Ok(entropy)
}

/// Calculate multiscale entropy
pub fn calculate_multiscale_entropy<F>(ts: &Array1<F>, max_scale: usize, m: usize, r: F) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let mut mse_values = Vec::with_capacity(max_scale);

    for scale in 1..=max_scale {
        let coarse_grained = coarse_grain_series(ts, scale)?;
        let sample_ent = calculate_sample_entropy(&coarse_grained, m, r)?;
        mse_values.push(sample_ent);
    }

    Ok(mse_values)
}

// =============================================================================
// Complexity Measures
// =============================================================================

/// Calculate Lempel-Ziv complexity
pub fn calculate_lempel_ziv_complexity<F>(ts: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    // Convert to binary string based on median
    let median = super::utils::calculate_median(ts);
    let binary_string: String = ts
        .iter()
        .map(|&x| if x > median { '1' } else { '0' })
        .collect();

    let mut complexity = 0;
    let mut i = 0;
    let chars: Vec<char> = binary_string.chars().collect();
    let n = chars.len();

    while i < n {
        let mut l = 1;
        let mut found = false;

        // Find the longest prefix that exists in the previous part
        while i + l <= n {
            let current_substring = &chars[i..i + l];
            
            // Check if this substring exists in the previous part
            for j in 0..i {
                if j + l <= i {
                    let prev_substring = &chars[j..j + l];
                    if current_substring == prev_substring {
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                break;
            }
            l += 1;
        }

        complexity += 1;
        i += l;
    }

    Ok(F::from(complexity).unwrap())
}

/// Calculate Higuchi fractal dimension
pub fn calculate_higuchi_fractal_dimension<F>(ts: &Array1<F>, k_max: usize) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len();
    let mut log_k = Vec::new();
    let mut log_l = Vec::new();

    for k in 1..=k_max {
        let m = (n - 1) / k;
        let mut l_k = F::zero();

        for i in 1..=k {
            let mut l_m = F::zero();
            for j in 1..=m {
                let idx1 = i + (j - 1) * k - 1;
                let idx2 = i + j * k - 1;
                if idx1 < n && idx2 < n {
                    l_m = l_m + (ts[idx2] - ts[idx1]).abs();
                }
            }
            l_m = l_m * F::from(n - 1).unwrap() / (F::from(m * k).unwrap());
            l_k = l_k + l_m;
        }

        l_k = l_k / F::from(k).unwrap();
        
        if l_k > F::zero() {
            log_k.push(F::from(k).unwrap().ln());
            log_l.push(l_k.ln());
        }
    }

    if log_k.len() < 2 {
        return Err(TimeSeriesError::FeatureExtractionError(
            "Insufficient data for fractal dimension calculation".to_string(),
        ));
    }

    // Linear regression to find slope
    let (slope, _) = super::utils::linear_fit(&log_k, &log_l);
    Ok(-slope) // Fractal dimension is negative slope
}

/// Calculate basic complexity features
pub fn calculate_complexity_features<F>(ts: &Array1<F>) -> Result<ComplexityFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let approximate_entropy = calculate_approximate_entropy(ts, 2, F::from(0.2).unwrap() * super::utils::calculate_std_dev(ts))?;
    let sample_entropy = calculate_sample_entropy(ts, 2, F::from(0.2).unwrap() * super::utils::calculate_std_dev(ts))?;
    let permutation_entropy = calculate_permutation_entropy(ts, 3)?;
    let lempel_ziv_complexity = calculate_lempel_ziv_complexity(ts)?;
    let fractal_dimension = calculate_higuchi_fractal_dimension(ts, 10)?;
    
    // Placeholder values for features requiring more complex implementations
    let hurst_exponent = F::from(0.5).unwrap();
    let dfa_exponent = F::from(0.5).unwrap();
    let turning_points = 0;
    let longest_strike = 0;

    Ok(ComplexityFeatures {
        approximate_entropy,
        sample_entropy,
        permutation_entropy,
        lempel_ziv_complexity,
        fractal_dimension,
        hurst_exponent,
        dfa_exponent,
        turning_points,
        longest_strike,
    })
}

/// Calculate comprehensive entropy features
pub fn calculate_entropy_features<F>(ts: &Array1<F>, config: &super::config::EntropyConfig) -> Result<EntropyFeatures<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let mut features = EntropyFeatures::default();

    if config.calculate_classical_entropy {
        let probabilities = discretize_and_get_probabilities(ts, config.n_bins)?;
        features.shannon_entropy = calculate_shannon_entropy(&probabilities);
        features.renyi_entropy_2 = calculate_renyi_entropy(&probabilities, F::from(2.0).unwrap())?;
        features.renyi_entropy_05 = calculate_renyi_entropy(&probabilities, F::from(0.5).unwrap())?;
    }

    if config.calculate_differential_entropy {
        let std_dev = super::utils::calculate_std_dev(ts);
        let r = F::from(config.tolerance_fraction).unwrap() * std_dev;
        features.approximate_entropy = calculate_approximate_entropy(ts, config.embedding_dimension, r)?;
        features.sample_entropy = calculate_sample_entropy(ts, config.embedding_dimension, r)?;
        features.permutation_entropy = calculate_permutation_entropy(ts, config.permutation_order)?;
    }

    if config.calculate_multiscale_entropy {
        let std_dev = super::utils::calculate_std_dev(ts);
        let r = F::from(config.tolerance_fraction).unwrap() * std_dev;
        features.multiscale_entropy = calculate_multiscale_entropy(ts, config.n_scales, config.embedding_dimension, r)?;
        features.composite_multiscale_entropy = features.multiscale_entropy.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(features.multiscale_entropy.len()).unwrap();
    }

    Ok(features)
}