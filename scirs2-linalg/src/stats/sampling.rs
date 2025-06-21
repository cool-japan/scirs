//! Random sampling from matrix distributions
//!
//! This module provides utilities for generating random samples from various
//! matrix-valued probability distributions, including multivariate normal,
//! Wishart, and other specialized distributions.

use ndarray::{Array2, ArrayView1, ArrayView2};
use num_traits::{Float, One, Zero};

use crate::decomposition::cholesky;
use crate::error::{LinalgError, LinalgResult};
use crate::random::random_normal_matrix;
use crate::stats::distributions::{MatrixNormalParams, WishartParams};

/// Generate samples from a multivariate normal distribution
///
/// # Arguments
///
/// * `mean` - Mean vector
/// * `cov` - Covariance matrix
/// * `n_samples` - Number of samples to generate
/// * `rng_seed` - Optional random seed
///
/// # Returns
///
/// * Matrix with samples as rows
pub fn sample_multivariate_normal<F>(
    mean: &ArrayView1<F>,
    cov: &ArrayView2<F>,
    n_samples: usize,
    rng_seed: Option<u64>,
) -> LinalgResult<Array2<F>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let p = mean.len();

    if cov.nrows() != p || cov.ncols() != p {
        return Err(LinalgError::ShapeError(format!(
            "Covariance matrix must be {}x{}, got {:?}",
            p,
            p,
            cov.shape()
        )));
    }

    // Generate standard normal samples
    let z = random_normal_matrix::<F>((n_samples, p), rng_seed)?;

    // Compute Cholesky factorization of covariance matrix
    let l = cholesky(cov, None)?;

    // Transform samples: X = μ + Z * L^T
    let mut samples = Array2::zeros((n_samples, p));
    for i in 0..n_samples {
        let z_row = z.row(i);
        let transformed = l.t().dot(&z_row);
        for j in 0..p {
            samples[[i, j]] = mean[j] + transformed[j];
        }
    }

    Ok(samples)
}

/// Generate samples from a matrix normal distribution
///
/// # Arguments
///
/// * `params` - Matrix normal distribution parameters
/// * `n_samples` - Number of samples to generate
/// * `rng_seed` - Optional random seed
///
/// # Returns
///
/// * Vector of matrix samples
pub fn sample_matrix_normal_multiple<F>(
    params: &MatrixNormalParams<F>,
    n_samples: usize,
    rng_seed: Option<u64>,
) -> LinalgResult<Vec<Array2<F>>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let mut samples = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let seed = rng_seed.map(|s| s.wrapping_add(i as u64));
        let sample = crate::stats::distributions::sample_matrix_normal(params, seed)?;
        samples.push(sample);
    }

    Ok(samples)
}

/// Generate samples from a Wishart distribution
///
/// # Arguments
///
/// * `params` - Wishart distribution parameters
/// * `n_samples` - Number of samples to generate
/// * `rng_seed` - Optional random seed
///
/// # Returns
///
/// * Vector of positive definite matrix samples
pub fn sample_wishart_multiple<F>(
    params: &WishartParams<F>,
    n_samples: usize,
    rng_seed: Option<u64>,
) -> LinalgResult<Vec<Array2<F>>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let mut samples = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let seed = rng_seed.map(|s| s.wrapping_add(i as u64));
        let sample = crate::stats::distributions::sample_wishart(params, seed)?;
        samples.push(sample);
    }

    Ok(samples)
}

/// Generate samples from an inverse Wishart distribution
///
/// # Arguments
///
/// * `scale` - Scale matrix parameter
/// * `dof` - Degrees of freedom
/// * `n_samples` - Number of samples to generate
/// * `rng_seed` - Optional random seed
///
/// # Returns
///
/// * Vector of positive definite matrix samples
pub fn sample_inverse_wishart<F>(
    scale: &ArrayView2<F>,
    dof: F,
    n_samples: usize,
    rng_seed: Option<u64>,
) -> LinalgResult<Vec<Array2<F>>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + std::fmt::Display
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let _p = scale.nrows();

    if scale.nrows() != scale.ncols() {
        return Err(LinalgError::ShapeError(
            "Scale matrix must be square".to_string(),
        ));
    }

    // For inverse Wishart, we sample from Wishart and then invert
    let scale_inv = crate::basic::inv(scale, None)?;
    let wishart_params = WishartParams::new(scale_inv, dof)?;

    let mut samples = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let seed = rng_seed.map(|s| s.wrapping_add(i as u64));
        let wishart_sample = crate::stats::distributions::sample_wishart(&wishart_params, seed)?;
        let inverse_wishart_sample = crate::basic::inv(&wishart_sample.view(), None)?;
        samples.push(inverse_wishart_sample);
    }

    Ok(samples)
}

/// Sample from a matrix-variate t-distribution
///
/// # Arguments
///
/// * `mean` - Mean matrix
/// * `row_cov` - Row covariance matrix
/// * `col_cov` - Column covariance matrix
/// * `dof` - Degrees of freedom
/// * `n_samples` - Number of samples to generate
/// * `rng_seed` - Optional random seed
///
/// # Returns
///
/// * Vector of matrix samples
pub fn sample_matrix_t<F>(
    mean: &ArrayView2<F>,
    row_cov: &ArrayView2<F>,
    col_cov: &ArrayView2<F>,
    dof: F,
    n_samples: usize,
    rng_seed: Option<u64>,
) -> LinalgResult<Vec<Array2<F>>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + std::fmt::Display
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let (m, n) = mean.dim();

    if row_cov.dim() != (m, m) || col_cov.dim() != (n, n) {
        return Err(LinalgError::ShapeError(format!(
            "Covariance matrices have incompatible dimensions: mean {:?}, row_cov {:?}, col_cov {:?}",
            mean.shape(), row_cov.shape(), col_cov.shape()
        )));
    }

    let mut samples = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let seed = rng_seed.map(|s| s.wrapping_add(i as u64));

        // Sample from matrix normal
        let matrix_normal_params =
            MatrixNormalParams::new(mean.to_owned(), row_cov.to_owned(), col_cov.to_owned())?;
        let normal_sample =
            crate::stats::distributions::sample_matrix_normal(&matrix_normal_params, seed)?;

        // Sample chi-square for scaling (simplified - using normal approximation)
        let chi_approx = random_normal_matrix::<F>((1, 1), seed)?;
        let scale_factor = (dof / (dof + chi_approx[[0, 0]] * chi_approx[[0, 0]])).sqrt();

        // Scale the normal sample
        let t_sample = mean + &((&normal_sample - mean) * scale_factor);
        samples.push(t_sample);
    }

    Ok(samples)
}

/// Generate bootstrap samples from a dataset
///
/// # Arguments
///
/// * `data` - Original dataset matrix (samples as rows)
/// * `n_bootstrap` - Number of bootstrap samples
/// * `sample_size` - Size of each bootstrap sample (if None, use original size)
/// * `rng_seed` - Optional random seed
///
/// # Returns
///
/// * Vector of bootstrap sample matrices
pub fn bootstrap_sample<F>(
    data: &ArrayView2<F>,
    n_bootstrap: usize,
    sample_size: Option<usize>,
    rng_seed: Option<u64>,
) -> LinalgResult<Vec<Array2<F>>>
where
    F: Float + Copy,
{
    let n_original = data.nrows();
    let p = data.ncols();
    let n_sample = sample_size.unwrap_or(n_original);

    let mut samples = Vec::with_capacity(n_bootstrap);

    // Simple pseudorandom number generation for sampling indices
    let mut seed = rng_seed.unwrap_or(42);

    for _ in 0..n_bootstrap {
        let mut bootstrap_sample = Array2::zeros((n_sample, p));

        for i in 0..n_sample {
            // Simple linear congruential generator for index selection
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let index = (seed as usize) % n_original;

            // Copy the selected row
            for j in 0..p {
                bootstrap_sample[[i, j]] = data[[index, j]];
            }
        }

        samples.push(bootstrap_sample);
    }

    Ok(samples)
}

/// Generate permutation samples for permutation tests
///
/// # Arguments
///
/// * `data` - Original dataset matrix (samples as rows)
/// * `n_permutations` - Number of permutations to generate
/// * `rng_seed` - Optional random seed
///
/// # Returns
///
/// * Vector of permuted sample matrices
pub fn permutation_sample<F>(
    data: &ArrayView2<F>,
    n_permutations: usize,
    rng_seed: Option<u64>,
) -> LinalgResult<Vec<Array2<F>>>
where
    F: Float + Copy,
{
    let n = data.nrows();
    let p = data.ncols();

    let mut samples = Vec::with_capacity(n_permutations);
    let mut seed = rng_seed.unwrap_or(42);

    for _ in 0..n_permutations {
        let mut permuted_sample = data.to_owned();

        // Fisher-Yates shuffle algorithm
        for i in (1..n).rev() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let j = (seed as usize) % (i + 1);

            // Swap rows i and j
            for k in 0..p {
                let temp = permuted_sample[[i, k]];
                permuted_sample[[i, k]] = permuted_sample[[j, k]];
                permuted_sample[[j, k]] = temp;
            }
        }

        samples.push(permuted_sample);
    }

    Ok(samples)
}

/// Monte Carlo sampling for complex distributions
///
/// Uses Metropolis-Hastings algorithm for sampling from unnormalized densities.
///
/// # Arguments
///
/// * `log_density` - Log density function to sample from
/// * `initial_value` - Starting point for the chain
/// * `proposal_cov` - Covariance matrix for proposal distribution
/// * `n_samples` - Number of samples to generate
/// * `burn_in` - Number of burn-in samples to discard
/// * `rng_seed` - Optional random seed
///
/// # Returns
///
/// * Vector of samples from the target distribution
pub fn metropolis_hastings_sample<F>(
    log_density: impl Fn(&ArrayView2<F>) -> LinalgResult<F>,
    initial_value: Array2<F>,
    proposal_cov: &ArrayView2<F>,
    n_samples: usize,
    burn_in: usize,
    rng_seed: Option<u64>,
) -> LinalgResult<Vec<Array2<F>>>
where
    F: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + std::iter::Sum,
{
    let (m, n) = initial_value.dim();
    let total_samples = n_samples + burn_in;

    let mut current = initial_value;
    let mut current_log_density = log_density(&current.view())?;
    let mut samples = Vec::with_capacity(n_samples);
    let mut accepted = 0;

    // Cholesky factor for proposal covariance
    let l = cholesky(proposal_cov, None)?;

    for i in 0..total_samples {
        let seed = rng_seed.map(|s| s.wrapping_add(i as u64));

        // Generate proposal
        let noise = random_normal_matrix::<F>((m, n), seed)?;
        let proposal_noise = l.t().dot(&noise);
        let proposal = &current + &proposal_noise;

        // Compute acceptance probability
        let proposal_log_density = match log_density(&proposal.view()) {
            Ok(density) => density,
            Err(_) => F::neg_infinity(), // Reject if density evaluation fails
        };

        let log_alpha = proposal_log_density - current_log_density;

        // Accept or reject
        let uniform_sample = random_normal_matrix::<F>((1, 1), seed)?;
        let uniform = (uniform_sample[[0, 0]].abs() % F::one()).abs(); // Rough uniform approximation

        if log_alpha > uniform.ln() {
            current = proposal;
            current_log_density = proposal_log_density;
            accepted += 1;
        }

        // Collect sample after burn-in
        if i >= burn_in {
            samples.push(current.clone());
        }
    }

    // Optionally log acceptance rate
    let acceptance_rate = accepted as f64 / total_samples as f64;
    if !(0.2..=0.7).contains(&acceptance_rate) {
        eprintln!(
            "Warning: MCMC acceptance rate is {:.3}, consider adjusting proposal covariance",
            acceptance_rate
        );
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_sample_multivariate_normal() {
        let mean = array![0.0, 0.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];

        let samples = sample_multivariate_normal(&mean.view(), &cov.view(), 100, Some(42)).unwrap();

        assert_eq!(samples.dim(), (100, 2));
        assert!(samples.iter().all(|&x| x.is_finite()));

        // Check that sample mean is approximately correct
        let sample_mean = samples.mean_axis(ndarray::Axis(0)).unwrap();
        assert_abs_diff_eq!(sample_mean[0], 0.0, epsilon = 0.5);
        assert_abs_diff_eq!(sample_mean[1], 0.0, epsilon = 0.5);
    }

    #[test]
    fn test_bootstrap_sample() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let bootstrap_samples = bootstrap_sample(&data.view(), 10, Some(2), Some(42)).unwrap();

        assert_eq!(bootstrap_samples.len(), 10);
        for sample in &bootstrap_samples {
            assert_eq!(sample.dim(), (2, 2));
            assert!(sample.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_permutation_sample() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let permuted_samples = permutation_sample(&data.view(), 5, Some(42)).unwrap();

        assert_eq!(permuted_samples.len(), 5);
        for sample in &permuted_samples {
            assert_eq!(sample.dim(), (3, 2));
            assert!(sample.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_sample_matrix_normal_multiple() {
        let mean = array![[0.0, 0.0], [0.0, 0.0]];
        let row_cov = array![[1.0, 0.0], [0.0, 1.0]];
        let col_cov = array![[1.0, 0.0], [0.0, 1.0]];

        let params = MatrixNormalParams::new(mean, row_cov, col_cov).unwrap();
        let samples = sample_matrix_normal_multiple(&params, 5, Some(42)).unwrap();

        assert_eq!(samples.len(), 5);
        for sample in &samples {
            assert_eq!(sample.dim(), (2, 2));
            assert!(sample.iter().all(|&x| x.is_finite()));
        }
    }
}
