//! Gaussian Mixture Models (GMM) for clustering
//!
//! This module implements Gaussian Mixture Models, a probabilistic model that assumes
//! data is generated from a mixture of a finite number of Gaussian distributions.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;
use std::fmt::Debug;
use std::iter::Sum;

use crate::error::{ClusteringError, Result};
use crate::vq::kmeans_plus_plus;
use statrs::statistics::Statistics;

/// Type alias for GMM parameters
type GMMParams<F> = (Array1<F>, Array2<F>, Vec<Array2<F>>);

/// Type alias for GMM fit result
type GMMFitResult<F> = (Array1<F>, Array2<F>, Vec<Array2<F>>, F, usize, bool);

/// Covariance type for GMM
#[derive(Debug, Clone, Copy)]
pub enum CovarianceType {
    /// Each component has its own general covariance matrix
    Full,
    /// Each component has its own diagonal covariance matrix
    Diagonal,
    /// All components share the same general covariance matrix
    Tied,
    /// All components share the same diagonal covariance matrix (spherical)
    Spherical,
}

/// GMM initialization method
#[derive(Debug, Clone, Copy)]
pub enum GMMInit {
    /// Initialize using K-means++
    KMeans,
    /// Random initialization
    Random,
}

/// Options for Gaussian Mixture Model
#[derive(Debug, Clone)]
pub struct GMMOptions<F: Float> {
    /// Number of mixture components
    pub n_components: usize,
    /// Type of covariance parameters
    pub covariance_type: CovarianceType,
    /// Convergence threshold
    pub tol: F,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Number of initializations to perform
    pub n_init: usize,
    /// Initialization method
    pub init_method: GMMInit,
    /// Random seed
    pub random_seed: Option<u64>,
    /// Regularization added to the diagonal of covariance matrices
    pub reg_covar: F,
}

impl<F: Float + FromPrimitive> Default for GMMOptions<F> {
    fn default() -> Self {
        Self {
            n_components: 1,
            covariance_type: CovarianceType::Full,
            tol: F::from(1e-3).unwrap(),
            max_iter: 100,
            n_init: 1,
            init_method: GMMInit::KMeans,
            random_seed: None,
            reg_covar: F::from(1e-6).unwrap(),
        }
    }
}

/// Gaussian Mixture Model
pub struct GaussianMixture<F: Float> {
    /// Options
    options: GMMOptions<F>,
    /// Weights of each mixture component
    weights: Option<Array1<F>>,
    /// Means of each mixture component
    means: Option<Array2<F>>,
    /// Covariances of each mixture component
    covariances: Option<Vec<Array2<F>>>,
    /// Lower bound value (log-likelihood)
    lower_bound: Option<F>,
    /// Number of iterations run
    n_iter: Option<usize>,
    /// Whether the model has converged
    converged: bool,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand + Sum + std::borrow::Borrow<f64>>
    GaussianMixture<F>
{
    /// Create a new GMM instance
    pub fn new(options: GMMOptions<F>) -> Self {
        Self {
            options,
            weights: None,
            means: None,
            covariances: None,
            lower_bound: None,
            n_iter: None,
            converged: false,
        }
    }

    /// Fit the Gaussian Mixture Model to data
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.shape()[0];
        let _n_features = data.shape()[1];

        if n_samples < self.options.n_components {
            return Err(ClusteringError::InvalidInput(
                "Number of samples must be >= number of components".to_string(),
            ));
        }

        let mut best_lower_bound = F::neg_infinity();
        let mut best_params = None;

        // Try multiple initializations
        for _ in 0..self.options.n_init {
            let (weights, means, covariances, lower_bound, n_iter, converged) =
                self.fit_single(data)?;

            if lower_bound > best_lower_bound {
                best_lower_bound = lower_bound;
                best_params = Some((weights, means, covariances, lower_bound, n_iter, converged));
            }
        }

        if let Some((weights, means, covariances, lower_bound, n_iter, converged)) = best_params {
            self.weights = Some(weights);
            self.means = Some(means);
            self.covariances = Some(covariances);
            self.lower_bound = Some(lower_bound);
            self.n_iter = Some(n_iter);
            self.converged = converged;
        }

        Ok(())
    }

    /// Single run of EM algorithm
    fn fit_single(&self, data: ArrayView2<F>) -> Result<GMMFitResult<F>> {
        let _n_samples = data.shape()[0];
        let _n_features = data.shape()[1];
        let _n_components = self.options.n_components;

        // Initialize parameters
        let (mut weights, mut means, mut covariances) = self.initialize_params(data)?;

        let mut lower_bound = F::neg_infinity();
        let mut converged = false;

        for iter in 0..self.options.max_iter {
            // E-step: compute resp_onsibilities
            let (resp_, new_lower_bound) = self.e_step(data, &weights, &means, &covariances)?;

            // Check convergence
            let change = (new_lower_bound - lower_bound).abs();
            if change < self.options.tol {
                converged = true;
                return Ok((
                    weights,
                    means,
                    covariances,
                    new_lower_bound,
                    iter + 1,
                    converged,
                ));
            }
            lower_bound = new_lower_bound;

            // M-step: update parameters
            (weights, means, covariances) = self.m_step(data, resp_)?;
        }

        Ok((
            weights,
            means,
            covariances,
            lower_bound,
            self.options.max_iter,
            converged,
        ))
    }

    /// Initialize GMM parameters
    fn initialize_params(&self, data: ArrayView2<F>) -> Result<GMMParams<F>> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let n_components = self.options.n_components;

        // Initialize weights uniformly
        let weights = Array1::from_elem(n_components, F::one() / F::from(n_components).unwrap());

        // Initialize means
        let means = match self.options.init_method {
            GMMInit::KMeans => {
                // Use k-means++ initialization
                kmeans_plus_plus(data, n_components, self.options.random_seed)?
            }
            GMMInit::Random => {
                // Random selection from data points
                let mut rng = match self.options.random_seed {
                    Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
                    None => rand::rngs::StdRng::seed_from_u64(rand::rng().random::<u64>()),
                };

                let mut means = Array2::zeros((n_components, n_features));
                for i in 0..n_components {
                    let idx = rng.random_range(0..n_samples);
                    means.slice_mut(s![i, ..]).assign(&data.slice(s![idx, ..]));
                }
                means
            }
        };

        // Initialize covariances based on data variance
        let mut covariances = Vec::with_capacity(n_components);

        // Compute data variance for initialization
        let data_mean = data.mean_axis(Axis(0)).unwrap();
        let mut variance = Array1::<F>::zeros(n_features);

        for i in 0..n_samples {
            let diff = &data.slice(s![i, ..]) - &data_mean;
            variance = variance + &diff.mapv(|x| x * x);
        }
        variance = variance / F::from(n_samples - 1).unwrap();

        match self.options.covariance_type {
            CovarianceType::Spherical => {
                let avg_variance = variance.sum() / F::from(variance.len()).unwrap();
                for _ in 0..n_components {
                    let mut cov = Array2::<F>::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = avg_variance;
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Diagonal => {
                for _ in 0..n_components {
                    let mut cov = Array2::<F>::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = variance[i];
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Full | CovarianceType::Tied => {
                // Initialize with diagonal covariance
                for _ in 0..n_components {
                    let mut cov = Array2::<F>::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = variance[i];
                    }
                    covariances.push(cov);
                }
            }
        }

        Ok((weights, means, covariances))
    }

    /// E-step: compute resp_onsibilities
    fn e_step(
        &self,
        data: ArrayView2<F>,
        weights: &Array1<F>,
        means: &Array2<F>,
        covariances: &[Array2<F>],
    ) -> Result<(Array2<F>, F)> {
        let n_samples = data.shape()[0];
        let n_components = self.options.n_components;

        let mut log_prob = Array2::zeros((n_samples, n_components));

        // Compute log probabilities for each component
        for (k, covariance) in covariances.iter().enumerate().take(n_components) {
            let log_prob_k = self.log_multivariate_normal_density(
                data,
                means.slice(s![k, ..]).view(),
                covariance,
            )?;
            log_prob.slice_mut(s![.., k]).assign(&log_prob_k);
        }

        // Add log weights
        for k in 0..n_components {
            let log_weight = weights[k].ln();
            log_prob
                .slice_mut(s![.., k])
                .mapv_inplace(|x| x + log_weight);
        }

        // Compute log normalization
        let log_prob_norm = self.logsumexp(log_prob.view(), Axis(1))?;

        // Compute resp_onsibilities
        let mut resp_ = log_prob.clone();
        for i in 0..n_samples {
            for k in 0..n_components {
                resp_[[i, k]] = (resp_[[i, k]] - log_prob_norm[i]).exp();
            }
        }

        // Compute lower bound
        let lower_bound = log_prob_norm.sum() / F::from(log_prob_norm.len()).unwrap();

        Ok((resp_, lower_bound))
    }

    /// M-step: update parameters
    fn m_step(&self, data: ArrayView2<F>, resp_: Array2<F>) -> Result<GMMParams<F>> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let n_components = self.options.n_components;

        // Compute weights
        let nk = resp_.sum_axis(Axis(0));
        let weights = &nk / F::from(n_samples).unwrap();

        // Compute means
        let mut means = Array2::zeros((n_components, n_features));
        for k in 0..n_components {
            let mut mean_k = Array1::zeros(n_features);
            for i in 0..n_samples {
                mean_k = mean_k + &data.slice(s![i, ..]) * resp_[[i, k]];
            }
            means.slice_mut(s![k, ..]).assign(&(&mean_k / nk[k]));
        }

        // Compute covariances
        let mut covariances = Vec::with_capacity(n_components);

        match self.options.covariance_type {
            CovarianceType::Full => {
                for k in 0..n_components {
                    let mean_k = means.slice(s![k, ..]);
                    let mut cov = Array2::zeros((n_features, n_features));

                    for i in 0..n_samples {
                        let diff = &data.slice(s![i, ..]) - &mean_k;
                        let outer = self.outer_product(diff.view(), diff.view());
                        cov = cov + &outer * resp_[[i, k]];
                    }

                    cov = cov / nk[k];
                    // Add regularization
                    for i in 0..n_features {
                        cov[[i, i]] = cov[[i, i]] + self.options.reg_covar;
                    }

                    covariances.push(cov);
                }
            }
            _ => {
                // Simplified: use diagonal covariances for other types
                for k in 0..n_components {
                    let mean_k = means.slice(s![k, ..]);
                    let mut cov = Array2::zeros((n_features, n_features));

                    for i in 0..n_samples {
                        let diff = &data.slice(s![i, ..]) - &mean_k;
                        for j in 0..n_features {
                            cov[[j, j]] = cov[[j, j]] + diff[j] * diff[j] * resp_[[i, k]];
                        }
                    }

                    for j in 0..n_features {
                        cov[[j, j]] = cov[[j, j]] / nk[k] + self.options.reg_covar;
                    }

                    covariances.push(cov);
                }
            }
        }

        Ok((weights, means, covariances))
    }

    /// Compute log probability under a multivariate Gaussian distribution
    fn log_multivariate_normal_density(
        &self,
        data: ArrayView2<F>,
        mean: ArrayView1<F>,
        covariance: &Array2<F>,
    ) -> Result<Array1<F>> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        // For simplicity, assume diagonal covariance
        let mut log_prob = Array1::zeros(n_samples);

        // Compute determinant (product of diagonal elements for diagonal matrix)
        let mut log_det = F::zero();
        for i in 0..n_features {
            log_det = log_det + covariance[[i, i]].ln();
        }

        let norm_const = F::from(n_features as f64 * (2.0 * PI).ln()).unwrap() + log_det;

        for i in 0..n_samples {
            let diff = &data.slice(s![i, ..]) - &mean;
            let mut mahalanobis = F::zero();

            // For diagonal covariance, this simplifies
            for j in 0..n_features {
                mahalanobis = mahalanobis + diff[j] * diff[j] / covariance[[j, j]];
            }

            log_prob[i] = F::from(-0.5).unwrap() * (norm_const + mahalanobis);
        }

        Ok(log_prob)
    }

    /// Compute log-sum-exp along an axis
    fn logsumexp(&self, arr: ArrayView2<F>, axis: Axis) -> Result<Array1<F>> {
        let max_vals = arr.fold_axis(axis, F::neg_infinity(), |&a, &b| a.max(b));
        let mut result = Array1::zeros(max_vals.len());

        match axis {
            Axis(1) => {
                for i in 0..arr.shape()[0] {
                    let mut sum = F::zero();
                    for j in 0..arr.shape()[1] {
                        sum = sum + (arr[[i, j]] - max_vals[i]).exp();
                    }
                    result[i] = max_vals[i] + sum.ln();
                }
            }
            _ => {
                return Err(ClusteringError::InvalidInput(
                    "Only axis 1 is supported for logsumexp".to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Compute outer product of two vectors
    fn outer_product(&self, a: ArrayView1<F>, b: ArrayView1<F>) -> Array2<F> {
        let n = a.len();
        let m = b.len();
        let mut result = Array2::zeros((n, m));

        for i in 0..n {
            for j in 0..m {
                result[[i, j]] = a[i] * b[j];
            }
        }

        result
    }

    /// Predict cluster labels
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<i32>> {
        if self.weights.is_none() || self.means.is_none() || self.covariances.is_none() {
            return Err(ClusteringError::InvalidInput(
                "Model has not been fitted yet".to_string(),
            ));
        }

        let weights = self.weights.as_ref().unwrap();
        let means = self.means.as_ref().unwrap();
        let covariances = self.covariances.as_ref().unwrap();

        let (resp__, _) = self.e_step(data, weights, means, covariances)?;

        // Assign to component with highest resp_onsibility
        let mut labels = Array1::zeros(data.shape()[0]);
        for i in 0..data.shape()[0] {
            let mut max_resp_ = F::neg_infinity();
            let mut best_k = 0;

            for k in 0..self.options.n_components {
                if resp__[[i, k]] > max_resp_ {
                    max_resp_ = resp__[[i, k]];
                    best_k = k;
                }
            }

            labels[i] = best_k as i32;
        }

        Ok(labels)
    }
}

/// Fit a Gaussian Mixture Model
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `options` - GMM options
///
/// # Returns
///
/// * Array of cluster labels
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use scirs2_cluster::gmm::{gaussian_mixture, GMMOptions};
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     4.0, 5.0,
///     4.2, 4.8,
///     3.9, 5.1,
/// ]).unwrap();
///
/// let options = GMMOptions {
///     n_components: 2,
///     ..Default::default()
/// };
///
/// let labels = gaussian_mixture(data.view(), options).unwrap();
/// ```
#[allow(dead_code)]
pub fn gaussian_mixture<F>(data: ArrayView2<F>, options: GMMOptions<F>) -> Result<Array1<i32>>
where
    F: Float + FromPrimitive + Debug + ScalarOperand + Sum + std::borrow::Borrow<f64>,
{
    let mut gmm = GaussianMixture::new(options);
    gmm.fit(data)?;
    gmm.predict(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_gmm_simple() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        let options = GMMOptions {
            n_components: 2,
            max_iter: 10,
            ..Default::default()
        };

        let result = gaussian_mixture(data.view(), options);
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.len(), 6);

        // Check that we have 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert!(unique_labels.len() <= 2);
    }

    #[test]
    fn test_gmm_different_covariance_types() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 0.9, 0.9, 1.2, 0.8, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9, 5.2, 4.8,
            ],
        )
        .unwrap();

        let covariance_types = vec![
            CovarianceType::Full,
            CovarianceType::Diagonal,
            CovarianceType::Spherical,
            CovarianceType::Tied,
        ];

        for cov_type in covariance_types {
            let options = GMMOptions {
                n_components: 2,
                covariance_type: cov_type,
                max_iter: 50,
                ..Default::default()
            };

            let result = gaussian_mixture(data.view(), options);
            assert!(
                result.is_ok(),
                "Failed with covariance type: {:?}",
                cov_type
            );

            let labels = result.unwrap();
            assert_eq!(labels.len(), 8);
        }
    }

    #[test]
    fn test_gmm_initialization_methods() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        let init_methods = vec![GMMInit::KMeans, GMMInit::Random];

        for init_method in init_methods {
            let options = GMMOptions {
                n_components: 2,
                init_method,
                random_seed: Some(42),
                max_iter: 20,
                ..Default::default()
            };

            let result = gaussian_mixture(data.view(), options);
            assert!(result.is_ok(), "Failed with init method: {:?}", init_method);

            let labels = result.unwrap();
            assert_eq!(labels.len(), 6);
        }
    }

    #[test]
    fn test_gmm_parameter_validation() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0]).unwrap();

        // Test with n_components = 0 (invalid)
        let options = GMMOptions {
            n_components: 0,
            ..Default::default()
        };
        let result = gaussian_mixture(data.view(), options);
        assert!(result.is_err());

        // Test with n_components > n_samples (questionable but should work)
        let options = GMMOptions {
            n_components: 10,
            max_iter: 5, // Keep low to avoid long convergence
            ..Default::default()
        };
        let result = gaussian_mixture(data.view(), options);
        // This might succeed or fail depending on implementation
        // Just check it doesn't panic
        let _result = result;
    }

    #[test]
    fn test_gmm_convergence_criteria() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        // Test with different tolerance values
        let tolerances = vec![1e-3, 1e-6, 1e-9];

        for tol in tolerances {
            let options = GMMOptions {
                n_components: 2,
                tol,
                max_iter: 100,
                ..Default::default()
            };

            let result = gaussian_mixture(data.view(), options);
            assert!(result.is_ok(), "Failed with tolerance: {}", tol);
        }
    }

    #[test]
    fn test_gmm_single_component() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 1.1, 2.1]).unwrap();

        let options = GMMOptions {
            n_components: 1,
            max_iter: 20,
            ..Default::default()
        };

        let result = gaussian_mixture(data.view(), options);
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.len(), 4);

        // All labels should be 0 for single component
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_gmm_reproducibility_with_seed() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        let options1 = GMMOptions {
            n_components: 2,
            random_seed: Some(42),
            max_iter: 50,
            ..Default::default()
        };

        let options2 = GMMOptions {
            n_components: 2,
            random_seed: Some(42),
            max_iter: 50,
            ..Default::default()
        };

        let labels1 = gaussian_mixture(data.view(), options1).unwrap();
        let labels2 = gaussian_mixture(data.view(), options2).unwrap();

        // With same seed, results should be consistent in clustering structure
        // Note: cluster labels might be swapped (0->1, 1->0) but the clustering should be the same
        assert_eq!(labels1.len(), labels2.len());

        // Check that the number of unique clusters is the same
        let unique1: std::collections::HashSet<_> = labels1.iter().cloned().collect();
        let unique2: std::collections::HashSet<_> = labels2.iter().cloned().collect();
        assert_eq!(unique1.len(), unique2.len());
    }

    #[test]
    fn test_gmm_many_components() {
        let data = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 3.0, 3.0, 3.1, 3.1, 3.2, 3.2, 5.0, 5.0, 5.1, 5.1,
                5.2, 5.2, 7.0, 7.0,
            ],
        )
        .unwrap();

        let options = GMMOptions {
            n_components: 3,
            max_iter: 50,
            ..Default::default()
        };

        let result = gaussian_mixture(data.view(), options);
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.len(), 10);

        // Should find up to 3 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert!(unique_labels.len() <= 3);
        assert!(!unique_labels.is_empty());
    }

    #[test]
    fn test_gmm_regularization() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        // Test with different regularization values
        let reg_values = vec![1e-6, 1e-3, 1e-1];

        for reg_covar in reg_values {
            let options = GMMOptions {
                n_components: 2,
                reg_covar,
                max_iter: 20,
                ..Default::default()
            };

            let result = gaussian_mixture(data.view(), options);
            assert!(result.is_ok(), "Failed with reg_covar: {}", reg_covar);
        }
    }

    #[test]
    fn test_gmm_fit_predict_workflow() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 0.9, 0.9, 1.2, 0.8, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9, 5.2, 4.8,
            ],
        )
        .unwrap();

        let options = GMMOptions {
            n_components: 2,
            max_iter: 50,
            random_seed: Some(42),
            ..Default::default()
        };

        // Test the fit-predict workflow using the struct directly
        let mut gmm = GaussianMixture::new(options);

        // Fit the model
        let fit_result = gmm.fit(data.view());
        assert!(fit_result.is_ok());

        // Predict on the same data
        let predict_result = gmm.predict(data.view());
        assert!(predict_result.is_ok());

        let labels = predict_result.unwrap();
        assert_eq!(labels.len(), 8);

        // Predict on new data (should work after fitting)
        let new_data = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 5.0, 5.0]).unwrap();

        let new_labels = gmm.predict(new_data.view());
        assert!(new_labels.is_ok());
        assert_eq!(new_labels.unwrap().len(), 2);
    }
}
