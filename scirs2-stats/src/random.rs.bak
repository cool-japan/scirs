//! Random number generation
//!
//! This module provides functions for random number generation,
//! following SciPy's `stats.random` module.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Float, NumCast, Zero};
use rand::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, StandardNormal};

/// Generate random samples from a specified random distribution
///
/// # Arguments
///
/// * `size` - Number of samples to generate
/// * `distribution` - Random distribution to sample from
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Array of random samples
///
/// # Examples
///
/// ```
/// use rand_distr::{Uniform, Distribution};
/// use scirs2_stats::random::random_sample;
///
/// // Generate 10 random numbers from a uniform distribution [0, 1)
/// let uniform_dist = Uniform::new(0.0, 1.0).unwrap();
/// let samples = random_sample(10, &uniform_dist, Some(42)).unwrap();
/// assert_eq!(samples.len(), 10);
/// ```
#[allow(dead_code)]
pub fn random_sample<T, D>(
    size: usize,
    distribution: &D,
    seed: Option<u64>,
) -> StatsResult<Array1<T>>
where
    T: Copy + Zero,
    D: Distribution<T>,
{
    if size == 0 {
        return Err(StatsError::InvalidArgument(
            "Size must be positive".to_string(),
        ));
    }

    let mut rng: StdRng = match seed {
        Some(seed_value) => {
            // Convert u64 seed to [u8; 32] for StdRng
            let mut seed_bytes = [0u8; 32];
            seed_bytes[..8].copy_from_slice(&seed_value.to_le_bytes());
            SeedableRng::from_seed(seed_bytes)
        }
        None => {
            // Use system entropy
            {
                let mut system_rng = rand::rng();
                let seed: [u8; 32] = system_rng.random();
                SeedableRng::from_seed(seed)
            }
        }
    };

    let mut result = Array1::zeros(size);
    for i in 0..size {
        result[i] = distribution.sample(&mut rng);
    }

    Ok(result)
}

/// Generate random numbers from a uniform distribution
///
/// # Arguments
///
/// * `low` - Lower bound (inclusive)
/// * `high` - Upper bound (exclusive)
/// * `size` - Number of samples to generate
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Array of random samples from uniform distribution
///
/// # Examples
///
/// ```
/// use scirs2_stats::random::uniform;
///
/// // Generate 5 random numbers from uniform distribution [0, 10)
/// let samples = uniform(0.0, 10.0, 5, Some(123)).unwrap();
/// assert_eq!(samples.len(), 5);
///
/// // All values should be in the range [0, 10)
/// for &val in samples.iter() {
///     assert!(val >= 0.0 && val < 10.0);
/// }
/// ```
#[allow(dead_code)]
pub fn uniform<F>(low: F, high: F, size: usize, seed: Option<u64>) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + Zero + SampleUniform + std::fmt::Display,
{
    if size == 0 {
        return Err(StatsError::InvalidArgument(
            "Size must be positive".to_string(),
        ));
    }

    if low >= high {
        return Err(StatsError::InvalidArgument(
            "Upper bound must be greater than lower bound".to_string(),
        ));
    }

    let distribution = rand_distr::Uniform::new(low, high).map_err(|e| {
        StatsError::ComputationError(format!("Failed to create uniform distribution: {}", e))
    })?;
    random_sample(size, &distribution, seed)
}

/// Generate random integer numbers from a uniform distribution
///
/// # Arguments
///
/// * `low` - Lower bound (inclusive)
/// * `high` - Upper bound (exclusive)
/// * `size` - Number of samples to generate
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Array of random integer samples
///
/// # Examples
///
/// ```
/// use scirs2_stats::random::randint;
///
/// // Generate 10 random integers from 1 to 100 (inclusive)
/// let samples = randint(1, 101, 10, Some(42)).unwrap();
/// assert_eq!(samples.len(), 10);
///
/// // All values should be in the range [1, 100]
/// for &val in samples.iter() {
///     assert!(val >= 1 && val <= 100);
/// }
/// ```
#[allow(dead_code)]
pub fn randint(low: i64, high: i64, size: usize, seed: Option<u64>) -> StatsResult<Array1<i64>> {
    if size == 0 {
        return Err(StatsError::InvalidArgument(
            "Size must be positive".to_string(),
        ));
    }

    if low >= high {
        return Err(StatsError::InvalidArgument(
            "Upper bound must be greater than lower bound".to_string(),
        ));
    }

    let distribution = rand_distr::Uniform::new_inclusive(low, high - 1).map_err(|e| {
        StatsError::ComputationError(format!("Failed to create uniform distribution: {}", e))
    })?;
    random_sample(size, &distribution, seed)
}

/// Generate standard normally distributed random numbers
///
/// # Arguments
///
/// * `size` - Number of samples to generate
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Array of random samples from standard normal distribution
///
/// # Examples
///
/// ```
/// use scirs2_stats::random::randn;
///
/// // Generate 100 random numbers from standard normal distribution
/// let samples = randn(100, Some(42)).unwrap();
/// assert_eq!(samples.len(), 100);
///
/// // Calculate mean and variance (should be approximately 0 and 1)
/// let sum: f64 = samples.iter().sum();
/// let mean = sum / 100.0;
///
/// // Mean should be reasonably close to 0 for 100 samples
/// assert!(mean.abs() < 0.3);
/// ```
#[allow(dead_code)]
pub fn randn(size: usize, seed: Option<u64>) -> StatsResult<Array1<f64>> {
    if size == 0 {
        return Err(StatsError::InvalidArgument(
            "Size must be positive".to_string(),
        ));
    }

    let mut rng: StdRng = match seed {
        Some(seed_value) => {
            // Convert u64 seed to [u8; 32] for StdRng
            let mut seed_bytes = [0u8; 32];
            seed_bytes[..8].copy_from_slice(&seed_value.to_le_bytes());
            SeedableRng::from_seed(seed_bytes)
        }
        None => {
            // Use system entropy
            {
                let mut system_rng = rand::rng();
                let seed: [u8; 32] = system_rng.random();
                SeedableRng::from_seed(seed)
            }
        }
    };

    let distribution = StandardNormal;
    let mut result = Array1::zeros(size);
    for i in 0..size {
        result[i] = distribution.sample(&mut rng);
    }

    Ok(result)
}

/// Choose random elements from an array
///
/// # Arguments
///
/// * `a` - Input array
/// * `size` - Number of samples to choose
/// * `replace` - Whether to sample with replacement
/// * `p` - Optional probability weights
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Array of randomly chosen elements
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_stats::random::choice;
///
/// // Create an array of choices
/// let options = array![10, 20, 30, 40, 50];
///
/// // Choose 3 random elements with replacement
/// let choices = choice(&options.view(), 3, true, None, Some(42)).unwrap();
/// assert_eq!(choices.len(), 3);
///
/// // Choose 2 random elements without replacement
/// let choices_no_replace = choice(&options.view(), 2, false, None, Some(123)).unwrap();
/// assert_eq!(choices_no_replace.len(), 2);
/// ```
#[allow(dead_code)]
pub fn choice<T>(
    a: &ArrayView1<T>,
    size: usize,
    replace: bool,
    p: Option<&ArrayView1<f64>>,
    seed: Option<u64>,
) -> StatsResult<Array1<T>>
where
    T: Copy,
{
    let n = a.len();

    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    if size == 0 {
        return Err(StatsError::InvalidArgument(
            "Size must be positive".to_string(),
        ));
    }

    if !replace && size > n {
        return Err(StatsError::InvalidArgument(
            "Cannot take a larger sample than population when 'replace=false'".to_string(),
        ));
    }

    let mut rng: StdRng = match seed {
        Some(seed_value) => {
            // Convert u64 seed to [u8; 32] for StdRng
            let mut seed_bytes = [0u8; 32];
            seed_bytes[..8].copy_from_slice(&seed_value.to_le_bytes());
            SeedableRng::from_seed(seed_bytes)
        }
        None => {
            // Use system entropy
            {
                let mut system_rng = rand::rng();
                let seed: [u8; 32] = system_rng.random();
                SeedableRng::from_seed(seed)
            }
        }
    };

    let mut result = Vec::with_capacity(size);

    if let Some(weights) = p {
        // Weighted sampling
        if weights.len() != n {
            return Err(StatsError::DimensionMismatch(
                "Length of weights must match length of array".to_string(),
            ));
        }

        // Check if weights sum to 1.0 (within tolerance)
        let sum: f64 = weights.iter().sum();
        if (sum - 1.0).abs() > 1e-10 {
            return Err(StatsError::InvalidArgument(
                "Weights must sum to 1.0".to_string(),
            ));
        }

        // Cumulative weights for efficient weighted sampling
        let mut cumulative = Vec::with_capacity(n);
        let mut cum_sum = 0.0;

        for &w in weights.iter() {
            if w < 0.0 {
                return Err(StatsError::InvalidArgument(
                    "Weights must be non-negative".to_string(),
                ));
            }
            cum_sum += w;
            cumulative.push(cum_sum);
        }

        if replace {
            // Weighted sampling with replacement
            for _ in 0..size {
                let r: f64 = rng.random();

                // Binary search for the index
                let mut low = 0;
                let mut high = n - 1;

                while low < high {
                    let mid = (low + high) / 2;
                    if r > cumulative[mid] {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }

                result.push(a[low]);
            }
        } else {
            // Weighted sampling without replacement (using reservoir method)
            let mut indices: Vec<usize> = (0..n).collect();

            for i in 0..size {
                // Calculate remaining weights
                let mut remaining_weights = vec![0.0; n - i];
                let mut total_weight = 0.0;

                for j in 0..n - i {
                    remaining_weights[j] = weights[indices[j]];
                    total_weight += remaining_weights[j];
                }

                // Normalize weights
                for w in remaining_weights.iter_mut() {
                    *w /= total_weight;
                }

                // Create cumulative weights
                let mut cum_weights = vec![0.0; n - i];
                let mut cum_sum = 0.0;

                for j in 0..n - i {
                    cum_sum += remaining_weights[j];
                    cum_weights[j] = cum_sum;
                }

                // Select an index based on weights
                let r: f64 = rng.random();
                let mut selected = 0;

                for (j, &weight) in cum_weights.iter().enumerate().take(n - i) {
                    if r <= weight {
                        selected = j;
                        break;
                    }
                }

                result.push(a[indices[selected]]);

                // Swap the selected index to the end and shrink the range
                indices.swap(selected, n - i - 1);
            }
        }
    } else {
        // Unweighted sampling
        if replace {
            // Simple random sampling with replacement
            let uniform = rand_distr::Uniform::new(0, n).unwrap();

            for _ in 0..size {
                let idx = uniform.sample(&mut rng);
                result.push(a[idx]);
            }
        } else {
            // Sampling without replacement (using Fisher-Yates shuffle)
            let mut indices: Vec<usize> = (0..n).collect();

            for i in 0..size {
                let j = rng.gen_range(i..n);
                indices.swap(i, j);
                result.push(a[indices[i]]);
            }
        }
    }

    Ok(Array1::from(result))
}

/// Generate a random permutation
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
/// use scirs2_stats::random::permutation;
///
/// // Permute an array
/// let arr = array![1, 2, 3, 4, 5];
/// let perm = permutation(&arr.view(), Some(42)).unwrap();
///
/// // The permutation should have the same length
/// assert_eq!(perm.len(), 5);
///
/// // The permutation should contain all the original elements
/// for &val in arr.iter() {
///     assert!(perm.iter().any(|&x| x == val));
/// }
/// ```
#[allow(dead_code)]
pub fn permutation<T>(x: &ArrayView1<T>, seed: Option<u64>) -> StatsResult<Array1<T>>
where
    T: Copy,
{
    let n = x.len();

    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    let mut rng: StdRng = match seed {
        Some(seed_value) => {
            // Convert u64 seed to [u8; 32] for StdRng
            let mut seed_bytes = [0u8; 32];
            seed_bytes[..8].copy_from_slice(&seed_value.to_le_bytes());
            SeedableRng::from_seed(seed_bytes)
        }
        None => {
            // Use system entropy
            {
                let mut system_rng = rand::rng();
                let seed: [u8; 32] = system_rng.random();
                SeedableRng::from_seed(seed)
            }
        }
    };

    // Create a copy of the input array
    let mut result = Array1::from_iter(x.iter().cloned());

    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = rng.gen_range(0..i);
        result.swap(i, j);
    }

    Ok(result)
}

/// Generate a random permutation of integers
///
/// # Arguments
///
/// * `n` - Integer specifying the length of the permutation
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Random permutation of integers from 0 to n-1
///
/// # Examples
///
/// ```
/// use scirs2_stats::random::permutation_int;
///
/// // Generate a random permutation of integers from 0 to 9
/// let perm = permutation_int(10, Some(42)).unwrap();
///
/// // The permutation should have the correct length
/// assert_eq!(perm.len(), 10);
///
/// // The permutation should contain all integers from 0 to 9
/// for i in 0..10 {
///     assert!(perm.iter().any(|&x| x == i));
/// }
/// ```
#[allow(dead_code)]
pub fn permutation_int(n: usize, seed: Option<u64>) -> StatsResult<Array1<usize>> {
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "Length must be positive".to_string(),
        ));
    }

    let mut rng: StdRng = match seed {
        Some(seed_value) => {
            // Convert u64 seed to [u8; 32] for StdRng
            let mut seed_bytes = [0u8; 32];
            seed_bytes[..8].copy_from_slice(&seed_value.to_le_bytes());
            SeedableRng::from_seed(seed_bytes)
        }
        None => {
            // Use system entropy
            {
                let mut system_rng = rand::rng();
                let seed: [u8; 32] = system_rng.random();
                SeedableRng::from_seed(seed)
            }
        }
    };

    // Create array of integers from 0 to n-1
    let mut result = Array1::from_iter(0..n);

    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = rng.gen_range(0..i);
        result.swap(i, j);
    }

    Ok(result)
}

/// Generate a random binary matrix with specified shape and density
///
/// # Arguments
///
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
/// * `density` - Probability of non-zero elements
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Random binary matrix
///
/// # Examples
///
/// ```
/// use scirs2_stats::random::random_binary_matrix;
///
/// // Generate a 5x5 binary matrix with 30% non-zero elements
/// let matrix = random_binary_matrix(5, 5, 0.3, Some(42)).unwrap();
///
/// // The matrix should have the correct shape
/// assert_eq!(matrix.shape(), &[5, 5]);
///
/// // All elements should be either 0 or 1
/// for &val in matrix.iter() {
///     assert!(val == 0 || val == 1);
/// }
/// ```
#[allow(dead_code)]
pub fn random_binary_matrix(
    n_rows: usize,
    n_cols: usize,
    density: f64,
    seed: Option<u64>,
) -> StatsResult<Array2<i32>> {
    if n_rows == 0 || n_cols == 0 {
        return Err(StatsError::InvalidArgument(
            "Dimensions must be positive".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&density) {
        return Err(StatsError::InvalidArgument(
            "Density must be between 0 and 1".to_string(),
        ));
    }

    let mut rng: StdRng = match seed {
        Some(seed_value) => {
            // Convert u64 seed to [u8; 32] for StdRng
            let mut seed_bytes = [0u8; 32];
            seed_bytes[..8].copy_from_slice(&seed_value.to_le_bytes());
            SeedableRng::from_seed(seed_bytes)
        }
        None => {
            // Use system entropy
            {
                let mut system_rng = rand::rng();
                let seed: [u8; 32] = system_rng.random();
                SeedableRng::from_seed(seed)
            }
        }
    };

    let mut result = Array2::zeros((n_rows, n_cols));

    for i in 0..n_rows {
        for j in 0..n_cols {
            if rng.random::<f64>() < density {
                result[[i, j]] = 1;
            }
        }
    }

    Ok(result)
}

/// Generate a bootstrap sample from an array
///
/// # Arguments
///
/// * `x` - Input array
/// * `n_samples_` - Number of bootstrap samples to generate
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
/// use scirs2_stats::random::bootstrap_sample;
///
/// // Create an array
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// // Generate 10 bootstrap samples
/// let samples = bootstrap_sample(&data.view(), 10, Some(42)).unwrap();
///
/// // Each bootstrap sample should have the same length as the original data
/// assert_eq!(samples.shape(), &[10, 5]);
/// ```
#[allow(dead_code)]
pub fn bootstrap_sample<T>(
    x: &ArrayView1<T>,
    n_samples_: usize,
    seed: Option<u64>,
) -> StatsResult<Array2<T>>
where
    T: Copy + num_traits::Zero,
{
    let n = x.len();

    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    if n_samples_ == 0 {
        return Err(StatsError::InvalidArgument(
            "Number of _samples must be positive".to_string(),
        ));
    }

    let mut rng: StdRng = match seed {
        Some(seed_value) => {
            // Convert u64 seed to [u8; 32] for StdRng
            let mut seed_bytes = [0u8; 32];
            seed_bytes[..8].copy_from_slice(&seed_value.to_le_bytes());
            SeedableRng::from_seed(seed_bytes)
        }
        None => {
            // Use system entropy
            {
                let mut system_rng = rand::rng();
                let seed: [u8; 32] = system_rng.random();
                SeedableRng::from_seed(seed)
            }
        }
    };

    let uniform = rand_distr::Uniform::new(0, n).unwrap();

    let mut result = Array2::zeros((n_samples_, n));

    for i in 0..n_samples_ {
        for j in 0..n {
            let idx = uniform.sample(&mut rng);
            result[[i, j]] = x[idx];
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_random_sample() {
        // Test with uniform distribution
        let uniform_dist = rand_distr::Uniform::new(0.0, 1.0).unwrap();
        let samples = random_sample(100, &uniform_dist, Some(42)).unwrap();

        assert_eq!(samples.len(), 100);
        for &s in samples.iter() {
            assert!((0.0..1.0).contains(&s));
        }

        // Test error cases
        assert!(random_sample::<f64, rand_distr::Uniform<f64>>(0, &uniform_dist, None).is_err());
    }

    #[test]
    #[ignore = "timeout"]
    fn test_uniform() {
        // Generate uniform samples
        let samples = uniform(10.0, 20.0, 50, Some(42)).unwrap();

        assert_eq!(samples.len(), 50);
        for &s in samples.iter() {
            assert!((10.0..20.0).contains(&s));
        }

        // Test error cases
        assert!(uniform(5.0, 5.0, 10, None).is_err());
        assert!(uniform(10.0, 5.0, 10, None).is_err());
        assert!(uniform(0.0, 1.0, 0, None).is_err());
    }

    #[test]
    #[ignore = "timeout"]
    fn test_randint() {
        // Generate integer samples
        let samples = randint(1, 101, 100, Some(42)).unwrap();

        assert_eq!(samples.len(), 100);
        for &s in samples.iter() {
            assert!((1..=100).contains(&s));
        }

        // Test error cases
        assert!(randint(5, 5, 10, None).is_err());
        assert!(randint(10, 5, 10, None).is_err());
        assert!(randint(0, 10, 0, None).is_err());
    }

    #[test]
    #[ignore = "timeout"]
    fn test_randn() {
        // Generate normal samples
        let samples = randn(1000, Some(42)).unwrap();

        assert_eq!(samples.len(), 1000);

        // Calculate statistics
        let sum: f64 = samples.iter().sum();
        let mean = sum / 1000.0;

        // Calculate variance
        let sum_sq: f64 = samples.iter().map(|&x| (x - mean).powi(2)).sum();
        let variance = sum_sq / 1000.0;

        // Mean should be close to 0
        assert!(mean.abs() < 0.1);

        // Variance should be close to 1
        assert_relative_eq!(variance, 1.0, epsilon = 0.2);

        // Test error case
        assert!(randn(0, None).is_err());
    }

    #[test]
    #[ignore = "timeout"]
    fn test_choice() {
        let options = array![10, 20, 30, 40, 50];

        // Test with replacement
        let choices = choice(&options.view(), 10, true, None, Some(42)).unwrap();
        assert_eq!(choices.len(), 10);

        // All choices should be from the options
        for &c in choices.iter() {
            assert!(options.iter().any(|&x| x == c));
        }

        // Test without replacement
        let choices_no_replace = choice(&options.view(), 3, false, None, Some(123)).unwrap();
        assert_eq!(choices_no_replace.len(), 3);

        // Values should be unique
        for i in 0..choices_no_replace.len() {
            for j in i + 1..choices_no_replace.len() {
                assert_ne!(choices_no_replace[i], choices_no_replace[j]);
            }
        }

        // Test with weights
        let weights = array![0.1, 0.2, 0.3, 0.2, 0.2];
        let weighted_choices =
            choice(&options.view(), 5, true, Some(&weights.view()), Some(42)).unwrap();
        assert_eq!(weighted_choices.len(), 5);

        // Test error cases
        assert!(choice(&options.view(), 0, true, None, None).is_err());
        assert!(choice(&options.view(), 10, false, None, None).is_err());

        // Test with wrong weights length
        let wrong_weights = array![0.5, 0.5];
        assert!(choice(&options.view(), 2, true, Some(&wrong_weights.view()), None).is_err());

        // Test with negative weights
        let neg_weights = array![-0.1, 0.2, 0.3, 0.3, 0.3];
        assert!(choice(&options.view(), 2, true, Some(&neg_weights.view()), None).is_err());

        // Test with empty array
        let empty: Array1<i32> = array![];
        assert!(choice(&empty.view(), 1, true, None, None).is_err());
    }

    #[test]
    #[ignore = "timeout"]
    fn test_permutation() {
        let arr = array![1, 2, 3, 4, 5];

        // Generate a permutation
        let perm = permutation(&arr.view(), Some(42)).unwrap();

        // Length should be the same
        assert_eq!(perm.len(), arr.len());

        // All values should be in the permutation
        for &val in arr.iter() {
            assert!(perm.iter().any(|&x| x == val));
        }

        // Test with empty array
        let empty: Array1<i32> = array![];
        assert!(permutation(&empty.view(), None).is_err());
    }

    #[test]
    #[ignore = "timeout"]
    fn test_permutation_int() {
        // Generate a permutation of integers
        let perm = permutation_int(10, Some(42)).unwrap();

        // Length should be correct
        assert_eq!(perm.len(), 10);

        // Should contain all integers from 0 to 9
        for i in 0..10 {
            assert!(perm.iter().any(|&x| x == i));
        }

        // Test error case
        assert!(permutation_int(0, None).is_err());
    }

    #[test]
    #[ignore = "timeout"]
    fn test_random_binary_matrix() {
        // Generate a binary matrix
        let matrix = random_binary_matrix(5, 5, 0.5, Some(42)).unwrap();

        // Shape should be correct
        assert_eq!(matrix.shape(), &[5, 5]);

        // All elements should be 0 or 1
        for &val in matrix.iter() {
            assert!(val == 0 || val == 1);
        }

        // Calculate density
        let ones_count = matrix.iter().filter(|&&x| x == 1).count();
        let density = ones_count as f64 / 25.0;

        // Density should be approximately 0.5
        assert!(density > 0.2 && density < 0.8);

        // Test error cases
        assert!(random_binary_matrix(0, 5, 0.5, None).is_err());
        assert!(random_binary_matrix(5, 0, 0.5, None).is_err());
        assert!(random_binary_matrix(5, 5, -0.1, None).is_err());
        assert!(random_binary_matrix(5, 5, 1.1, None).is_err());
    }

    #[test]
    #[ignore = "timeout"]
    fn test_bootstrap_sample() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Generate bootstrap samples
        let samples = bootstrap_sample(&data.view(), 10, Some(42)).unwrap();

        // Shape should be [n_samples_, data_length]
        assert_eq!(samples.shape(), &[10, 5]);

        // Test error cases
        assert!(bootstrap_sample(&data.view(), 0, None).is_err());

        // Test with empty array
        let empty: Array1<f64> = array![];
        assert!(bootstrap_sample(&empty.view(), 10, None).is_err());
    }
}
