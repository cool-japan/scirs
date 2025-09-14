//! Statistical sampling
//!
//! This module provides functions for statistical sampling,
//! following SciPy's `stats.sampling` module.

use crate::error::{StatsError, StatsResult};
use crate::random;
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::Float;
use rand::prelude::*;
use rand::SeedableRng;

/// Distribution trait for statistical distributions that can be sampled
pub trait SampleableDistribution<T> {
    /// Generate random samples from the distribution
    fn rvs(&self, size: usize) -> StatsResult<Vec<T>>;
}

/// Sample from a distribution
///
/// # Arguments
///
/// * `dist` - Distribution to sample from
/// * `size` - Number of samples to generate
///
/// # Returns
///
/// * Array of samples
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::{sampling, distributions};
///
/// // Create a normal distribution
/// let normal = distributions::norm(0.0f64, 1.0).unwrap();
///
/// // Sample from it
/// let samples = sampling::sample_distribution(&normal, 100).unwrap();
/// assert_eq!(samples.len(), 100);
/// ```
#[allow(dead_code)]
pub fn sample_distribution<T, D>(dist: &D, size: usize) -> StatsResult<Array1<T>>
where
    T: Float + std::iter::Sum<T> + std::ops::Div<Output = T>,
    D: SampleableDistribution<T>,
{
    if size == 0 {
        return Err(StatsError::InvalidArgument(
            "Size must be positive".to_string(),
        ));
    }

    let samples = dist.rvs(size)?;
    Ok(Array1::from_vec(samples))
}

/// Nonparametric bootstrap
///
/// # Arguments
///
/// * `x` - Input array
/// * `n_resamples` - Number of bootstrap samples to generate
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Bootstrap samples with replacement
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::sampling;
///
/// // Create an array
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// // Generate bootstrap samples
/// let samples = sampling::bootstrap(&data.view(), 10, Some(42)).unwrap();
/// assert_eq!(samples.shape(), &[10, 5]);
/// ```
#[allow(dead_code)]
pub fn bootstrap<T>(
    x: &ArrayView1<T>,
    n_resamples: usize,
    seed: Option<u64>,
) -> StatsResult<Array2<T>>
where
    T: Copy + num_traits::Zero,
{
    random::bootstrap_sample(x, n_resamples, seed)
}

/// Random permutation
///
/// # Arguments
///
/// * `x` - Input array
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Randomly permuted array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::sampling;
///
/// // Create an array
/// let data = array![1, 2, 3, 4, 5];
///
/// // Generate a permutation
/// let perm = sampling::permutation(&data.view(), Some(42)).unwrap();
/// assert_eq!(perm.len(), 5);
/// ```
#[allow(dead_code)]
pub fn permutation<T>(x: &ArrayView1<T>, seed: Option<u64>) -> StatsResult<Array1<T>>
where
    T: Copy,
{
    random::permutation(x, seed)
}

/// Generate stratified random sample
///
/// # Arguments
///
/// * `x` - Input array
/// * `groups` - Group labels for each element in x
/// * `size` - Number of samples per group
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Stratified sample indices
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::sampling;
///
/// // Create an array and group labels
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let groups = array![0, 0, 1, 1, 2, 2];
///
/// // Generate a stratified sample with 1 sample per group
/// let indices = sampling::stratified_sample(&data.view(), &groups.view(), 1, Some(42)).unwrap();
/// assert_eq!(indices.len(), 3);  // 3 groups with 1 sample each
/// ```
#[allow(dead_code)]
pub fn stratified_sample<T, G>(
    x: &ArrayView1<T>,
    groups: &ArrayView1<G>,
    size: usize,
    seed: Option<u64>,
) -> StatsResult<Array1<usize>>
where
    T: Copy,
    G: Copy + Eq + std::hash::Hash,
{
    if x.len() != groups.len() {
        return Err(StatsError::DimensionMismatch(
            "Input array and group array must have the same length".to_string(),
        ));
    }

    if size == 0 {
        return Err(StatsError::InvalidArgument(
            "Size must be positive".to_string(),
        ));
    }

    // Get unique groups
    let mut unique_groups = std::collections::HashSet::new();
    for &g in groups.iter() {
        unique_groups.insert(g);
    }

    let n_groups = unique_groups.len();

    // Create map of group -> indices
    let mut group_indices = std::collections::HashMap::new();
    for (i, &g) in groups.iter().enumerate() {
        group_indices.entry(g).or_insert_with(Vec::new).push(i);
    }

    // Initialize RNG
    let mut rng = match seed {
        Some(seed_value) => rand::rngs::StdRng::seed_from_u64(seed_value),
        None => {
            // Get a seed from the system RNG
            let mut rng = rand::rng();
            let seed = rng.random::<u64>();
            rand::rngs::StdRng::seed_from_u64(seed)
        }
    };

    // Sample from each group
    let mut result = Vec::with_capacity(n_groups * size);

    for (_, indices) in group_indices.iter() {
        if indices.len() < size {
            return Err(StatsError::InvalidArgument(format!(
                "Group size {} is smaller than requested sample size {}",
                indices.len(),
                size
            )));
        }

        // Sample without replacement using Fisher-Yates shuffle
        let mut indices_copy = indices.clone();
        for i in 0..size {
            let j = rng.gen_range(i..indices_copy.len());
            indices_copy.swap(i, j);
            result.push(indices_copy[i]);
        }
    }

    Ok(Array1::from_vec(result))
}

/// Stratified bootstrap sampling
///
/// Performs bootstrap sampling within each stratum (group) separately,
/// maintaining the proportion of each group in the resamples.
///
/// # Arguments
/// * `x` - Input array
/// * `groups` - Group labels for each element in x
/// * `n_resamples` - Number of bootstrap samples to generate
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
/// * Bootstrap samples maintaining group proportions
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::sampling;
///
/// // Create an array and group labels
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let groups = array![0, 0, 1, 1, 2, 2];
///
/// // Generate stratified bootstrap samples
/// let samples = sampling::stratifiedbootstrap(&data.view(), &groups.view(), 5, Some(42)).unwrap();
/// assert_eq!(samples.shape(), &[5, 6]);
/// ```
#[allow(dead_code)]
pub fn stratified_bootstrap<T, G>(
    x: &ArrayView1<T>,
    groups: &ArrayView1<G>,
    n_resamples: usize,
    seed: Option<u64>,
) -> StatsResult<Array2<T>>
where
    T: Copy + num_traits::Zero,
    G: Copy + Eq + std::hash::Hash,
{
    if x.len() != groups.len() {
        return Err(StatsError::DimensionMismatch(
            "Input array and group array must have the same length".to_string(),
        ));
    }

    if n_resamples == 0 {
        return Err(StatsError::InvalidArgument(
            "Number of _resamples must be positive".to_string(),
        ));
    }

    // Create map of group -> indices
    let mut group_indices = std::collections::HashMap::new();
    for (i, &g) in groups.iter().enumerate() {
        group_indices.entry(g).or_insert_with(Vec::new).push(i);
    }

    // Initialize RNG
    let mut rng = match seed {
        Some(seed_value) => rand::rngs::StdRng::seed_from_u64(seed_value),
        None => {
            let mut rng = rand::rng();
            let seed = rng.random::<u64>();
            rand::rngs::StdRng::seed_from_u64(seed)
        }
    };

    let mut samples = Array2::zeros((n_resamples, x.len()));

    for resample_idx in 0..n_resamples {
        let mut sample_idx = 0;

        // Sample from each group proportionally
        for (_, indices) in group_indices.iter() {
            for _ in 0..indices.len() {
                let random_idx = rng.gen_range(0..indices.len());
                let selected_idx = indices[random_idx];
                samples[[resample_idx, sample_idx]] = x[selected_idx];
                sample_idx += 1;
            }
        }
    }

    Ok(samples)
}

/// Block bootstrap sampling for time series data
///
/// Samples contiguous blocks of data to preserve temporal dependencies.
/// Useful for time series and other sequentially dependent data.
///
/// # Arguments
/// * `x` - Input time series array
/// * `blocksize` - Size of each block to sample
/// * `n_resamples` - Number of bootstrap samples to generate
/// * `circular` - Whether to allow wrapping around the end of the series
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
/// * Block bootstrap samples
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::sampling;
///
/// // Create a time series
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
///
/// // Generate block bootstrap samples with block size 3
/// let samples = sampling::blockbootstrap(&data.view(), 3, 5, true, Some(42)).unwrap();
/// assert_eq!(samples.shape(), &[5, 8]);
/// ```
#[allow(dead_code)]
pub fn block_bootstrap<T>(
    x: &ArrayView1<T>,
    blocksize: usize,
    n_resamples: usize,
    circular: bool,
    seed: Option<u64>,
) -> StatsResult<Array2<T>>
where
    T: Copy + num_traits::Zero,
{
    if x.len() == 0 {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    if blocksize == 0 {
        return Err(StatsError::InvalidArgument(
            "Block size must be positive".to_string(),
        ));
    }

    if blocksize > x.len() {
        return Err(StatsError::InvalidArgument(
            "Block size cannot exceed array length".to_string(),
        ));
    }

    if n_resamples == 0 {
        return Err(StatsError::InvalidArgument(
            "Number of _resamples must be positive".to_string(),
        ));
    }

    // Initialize RNG
    let mut rng = match seed {
        Some(seed_value) => rand::rngs::StdRng::seed_from_u64(seed_value),
        None => {
            let mut rng = rand::rng();
            let seed = rng.random::<u64>();
            rand::rngs::StdRng::seed_from_u64(seed)
        }
    };

    let data_len = x.len();
    let max_start_pos = if circular {
        data_len
    } else {
        data_len - blocksize + 1
    };

    let mut samples = Array2::zeros((n_resamples, data_len));

    for resample_idx in 0..n_resamples {
        let mut sample_pos = 0;

        // Fill the resample with blocks
        while sample_pos < data_len {
            // Choose a random starting position for the block
            let start_pos = rng.gen_range(0..max_start_pos);

            // Copy the block (with wrapping if circular)
            for block_offset in 0..blocksize {
                if sample_pos >= data_len {
                    break;
                }

                let data_idx = if circular {
                    (start_pos + block_offset) % data_len
                } else {
                    start_pos + block_offset
                };

                samples[[resample_idx, sample_pos]] = x[data_idx];
                sample_pos += 1;
            }
        }
    }

    Ok(samples)
}

/// Moving block bootstrap for time series data
///
/// A variant of block bootstrap that uses overlapping blocks.
/// Better preserves the temporal structure of the data.
///
/// # Arguments
/// * `x` - Input time series array
/// * `blocksize` - Size of each block to sample
/// * `n_resamples` - Number of bootstrap samples to generate
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
/// * Moving block bootstrap samples
#[allow(dead_code)]
pub fn moving_block_bootstrap<T>(
    x: &ArrayView1<T>,
    blocksize: usize,
    n_resamples: usize,
    seed: Option<u64>,
) -> StatsResult<Array2<T>>
where
    T: Copy + num_traits::Zero,
{
    if x.len() == 0 {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    if blocksize == 0 || blocksize > x.len() {
        return Err(StatsError::InvalidArgument(
            "Block size must be positive and not exceed array length".to_string(),
        ));
    }

    // Generate all possible overlapping blocks
    let mut blocks = Vec::new();
    for i in 0..=(x.len() - blocksize) {
        let mut block = Vec::with_capacity(blocksize);
        for j in i..(i + blocksize) {
            block.push(x[j]);
        }
        blocks.push(block);
    }

    // Initialize RNG
    let mut rng = match seed {
        Some(seed_value) => rand::rngs::StdRng::seed_from_u64(seed_value),
        None => {
            let mut rng = rand::rng();
            let seed = rng.random::<u64>();
            rand::rngs::StdRng::seed_from_u64(seed)
        }
    };

    let data_len = x.len();
    let n_blocks_needed = (data_len + blocksize - 1) / blocksize; // Ceiling division
    let mut samples = Array2::zeros((n_resamples, data_len));

    for resample_idx in 0..n_resamples {
        let mut sample_pos = 0;

        // Sample enough blocks to fill the resample
        for _ in 0..n_blocks_needed {
            if sample_pos >= data_len {
                break;
            }

            // Choose a random block
            let block_idx = rng.gen_range(0..blocks.len());
            let selected_block = &blocks[block_idx];

            // Copy elements from the block
            for &value in selected_block {
                if sample_pos >= data_len {
                    break;
                }
                samples[[resample_idx, sample_pos]] = value;
                sample_pos += 1;
            }
        }
    }

    Ok(samples)
}

/// Stationary bootstrap for time series data
///
/// Uses geometrically distributed block lengths to preserve stationarity.
/// The expected block length is 1/p where p is the probability parameter.
///
/// # Arguments
/// * `x` - Input time series array
/// * `p` - Probability parameter (0 < p < 1), controls expected block length
/// * `n_resamples` - Number of bootstrap samples to generate
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
/// * Stationary bootstrap samples
#[allow(dead_code)]
pub fn stationary_bootstrap<T>(
    x: &ArrayView1<T>,
    p: f64,
    n_resamples: usize,
    seed: Option<u64>,
) -> StatsResult<Array2<T>>
where
    T: Copy + num_traits::Zero,
{
    if x.len() == 0 {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    if p <= 0.0 || p >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "Probability parameter p must be between 0 and 1".to_string(),
        ));
    }

    if n_resamples == 0 {
        return Err(StatsError::InvalidArgument(
            "Number of _resamples must be positive".to_string(),
        ));
    }

    // Initialize RNG
    let mut rng = match seed {
        Some(seed_value) => rand::rngs::StdRng::seed_from_u64(seed_value),
        None => {
            let mut rng = rand::rng();
            let seed = rng.random::<u64>();
            rand::rngs::StdRng::seed_from_u64(seed)
        }
    };

    let data_len = x.len();
    let mut samples = Array2::zeros((n_resamples, data_len));

    for resample_idx in 0..n_resamples {
        let mut sample_pos = 0;

        while sample_pos < data_len {
            // Choose a random starting position
            let start_pos = rng.gen_range(0..data_len);
            let mut current_pos = start_pos;

            // Generate a block with geometric length
            loop {
                samples[[resample_idx, sample_pos]] = x[current_pos];
                sample_pos += 1;

                if sample_pos >= data_len {
                    break;
                }

                // Decide whether to continue the block (with probability 1-p)
                let u: f64 = rng.random();
                if u < p {
                    break; // End the block
                }

                // Continue the block (move to next position with wrapping)
                current_pos = (current_pos + 1) % data_len;
            }
        }
    }

    Ok(samples)
}

/// Double bootstrap for bias correction
///
/// Performs a nested bootstrap procedure to estimate and correct bias
/// in bootstrap statistics. This is useful for improving the accuracy
/// of bootstrap confidence intervals.
///
/// # Arguments
/// * `x` - Input array
/// * `statistic` - Function to compute the statistic of interest
/// * `n_resamples1` - Number of first-level bootstrap samples
/// * `n_resamples2` - Number of second-level bootstrap samples
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
/// * Tuple of (bias-corrected estimate, bootstrap samples, bias estimate)
#[allow(dead_code)]
pub fn double_bootstrap<T, F>(
    x: &ArrayView1<T>,
    statistic: F,
    n_resamples1: usize,
    n_resamples2: usize,
    seed: Option<u64>,
) -> StatsResult<(f64, Array1<f64>, f64)>
where
    T: Copy + num_traits::Zero,
    F: Fn(&ArrayView1<T>) -> StatsResult<f64> + Copy,
{
    if x.len() == 0 {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    if n_resamples1 == 0 || n_resamples2 == 0 {
        return Err(StatsError::InvalidArgument(
            "Number of resamples must be positive".to_string(),
        ));
    }

    // Compute the original statistic
    let original_stat = statistic(x)?;

    // First level bootstrap
    let first_level_samples = bootstrap(x, n_resamples1, seed)?;
    let mut first_level_stats = Array1::zeros(n_resamples1);

    // Prepare RNG for second level
    let mut rng = match seed {
        Some(seed_value) => rand::rngs::StdRng::seed_from_u64(seed_value + 1),
        None => {
            let mut rng = rand::rng();
            let seed = rng.random::<u64>();
            rand::rngs::StdRng::seed_from_u64(seed)
        }
    };

    let mut bias_estimates = Array1::zeros(n_resamples1);

    for i in 0..n_resamples1 {
        let first_sample = first_level_samples.row(i);
        let first_stat = statistic(&first_sample)?;
        first_level_stats[i] = first_stat;

        // Second level bootstrap for this sample
        let second_seed = rng.random::<u64>();
        let second_level_samples = bootstrap(&first_sample, n_resamples2, Some(second_seed))?;

        let mut second_level_stats = Array1::zeros(n_resamples2);
        for j in 0..n_resamples2 {
            let second_sample = second_level_samples.row(j);
            second_level_stats[j] = statistic(&second_sample)?;
        }

        // Estimate bias for this first-level sample
        let second_level_mean = second_level_stats.mean().unwrap();
        bias_estimates[i] = second_level_mean - first_stat;
    }

    // Overall bias estimate
    let overall_bias = bias_estimates.mean().unwrap();

    // Bias-corrected estimate
    let _first_level_mean = first_level_stats.mean().unwrap();
    let bias_corrected = original_stat - overall_bias;

    Ok((bias_corrected, first_level_stats, overall_bias))
}

/// Bootstrap confidence intervals using multiple methods
///
/// Computes confidence intervals using different bootstrap methods:
/// - Percentile method
/// - Bias-corrected (BC) method  
/// - Bias-corrected and accelerated (BCa) method
///
/// # Arguments
/// * `x` - Input array
/// * `statistic` - Function to compute the statistic of interest
/// * `n_resamples` - Number of bootstrap samples
/// * `confidence_level` - Confidence level (0.0 to 1.0)
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
/// * Tuple of (percentile CI, BC CI, BCa CI) where each CI is (lower, upper)
#[allow(dead_code)]
pub fn bootstrap_confidence_intervals<T, F>(
    x: &ArrayView1<T>,
    statistic: F,
    n_resamples: usize,
    confidence_level: f64,
    seed: Option<u64>,
) -> StatsResult<((f64, f64), (f64, f64), (f64, f64))>
where
    T: Copy + num_traits::Zero,
    F: Fn(&ArrayView1<T>) -> StatsResult<f64> + Copy,
{
    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "Confidence _level must be between 0 and 1".to_string(),
        ));
    }

    // Compute original statistic
    let original_stat = statistic(x)?;

    // Generate bootstrap samples
    let bootstrap_samples = bootstrap(x, n_resamples, seed)?;
    let mut bootstrap_stats = Array1::zeros(n_resamples);

    for i in 0..n_resamples {
        let sample = bootstrap_samples.row(i);
        bootstrap_stats[i] = statistic(&sample)?;
    }

    // Sort bootstrap statistics
    let mut sorted_stats = bootstrap_stats.to_vec();
    sorted_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - confidence_level;
    let n = sorted_stats.len() as f64;

    // Percentile method
    let lower_idx = ((alpha / 2.0) * n) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n) as usize;
    let percentile_ci = (
        sorted_stats[lower_idx.min(n_resamples - 1)],
        sorted_stats[upper_idx.min(n_resamples - 1)],
    );

    // Bias-correction
    let below_original = sorted_stats.iter().filter(|&&x| x < original_stat).count() as f64;
    let z0 = if below_original > 0.0 && below_original < n {
        // Inverse normal CDF approximation
        let p = below_original / n;
        // Simple inverse normal approximation
        let z = if p > 0.5 {
            (2.0 * std::f64::consts::PI * p).sqrt()
        } else {
            -(2.0 * std::f64::consts::PI * (1.0 - p)).sqrt()
        };
        z
    } else {
        0.0
    };

    // Acceleration (simplified jackknife estimate)
    let mut jackknife_stats = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let mut jackknife_sample = Vec::with_capacity(x.len() - 1);
        for j in 0..x.len() {
            if i != j {
                jackknife_sample.push(x[j]);
            }
        }
        let jk_array = Array1::from_vec(jackknife_sample);
        jackknife_stats.push(statistic(&jk_array.view())?);
    }

    let jk_mean = jackknife_stats.iter().sum::<f64>() / jackknife_stats.len() as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for &jk_stat in &jackknife_stats {
        let diff = jk_mean - jk_stat;
        numerator += diff.powi(3);
        denominator += diff.powi(2);
    }

    let acceleration = if denominator > 0.0 {
        numerator / (6.0 * denominator.powf(1.5))
    } else {
        0.0
    };

    // BCa confidence intervals
    let z_alpha_2 = 1.96 * alpha / 2.0; // Approximate critical value
    let z_1_alpha_2 = -z_alpha_2;

    let alpha1 = normal_cdf(z0 + (z0 + z_alpha_2) / (1.0 - acceleration * (z0 + z_alpha_2)));
    let alpha2 = normal_cdf(z0 + (z0 + z_1_alpha_2) / (1.0 - acceleration * (z0 + z_1_alpha_2)));

    let bca_lower_idx = (alpha1 * n) as usize;
    let bca_upper_idx = (alpha2 * n) as usize;

    let bc_ci = (
        sorted_stats[bca_lower_idx.min(n_resamples - 1)],
        sorted_stats[bca_upper_idx.min(n_resamples - 1)],
    );

    let bca_ci = (
        sorted_stats[bca_lower_idx.min(n_resamples - 1)],
        sorted_stats[bca_upper_idx.min(n_resamples - 1)],
    );

    Ok((percentile_ci, bc_ci, bca_ci))
}

/// Approximate normal CDF (for BCa intervals)
#[allow(dead_code)]
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation
#[allow(dead_code)]
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}
