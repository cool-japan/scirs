//! GAN evaluation metrics
//!
//! This module provides comprehensive evaluation metrics for Generative
//! Adversarial Networks including Inception Score, FID, KID, and LPIPS.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use scirs2_core::numeric::Float;
use std::iter::Sum;

use super::results::{InceptionScoreResult, KIDResult};

/// GAN evaluation metrics
pub struct GANEvaluationMetrics<F: Float> {
    /// Number of inception features to use
    pub n_inception_features: usize,
    /// Number of samples for KID estimation
    pub n_kid_samples: usize,
    /// Enable LPIPS computation
    pub enable_lpips: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    _phantom: std::marker::PhantomData<F>,
}

impl<
        F: Float + scirs2_core::numeric::FromPrimitive + Sum + scirs2_core::ndarray::ScalarOperand,
    > Default for GANEvaluationMetrics<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        F: Float + scirs2_core::numeric::FromPrimitive + Sum + scirs2_core::ndarray::ScalarOperand,
    > GANEvaluationMetrics<F>
{
    /// Create new GAN evaluation metrics
    pub fn new() -> Self {
        Self {
            n_inception_features: 2048,
            n_kid_samples: 10000,
            enable_lpips: true,
            random_seed: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set inception features dimension
    pub fn with_inception_features(mut self, n: usize) -> Self {
        self.n_inception_features = n;
        self
    }

    /// Set KID sample size
    pub fn with_kid_samples(mut self, n: usize) -> Self {
        self.n_kid_samples = n;
        self
    }

    /// Enable/disable LPIPS
    pub fn with_lpips(mut self, enable: bool) -> Self {
        self.enable_lpips = enable;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Compute Inception Score (IS)
    pub fn inception_score(
        &self,
        features: &Array2<F>,
        splits: usize,
    ) -> Result<InceptionScoreResult<F>> {
        if features.is_empty() || splits == 0 {
            return Err(MetricsError::InvalidInput(
                "Empty features or zero splits".to_string(),
            ));
        }

        let n_samples = features.nrows();
        let split_size = n_samples / splits;
        let mut scores = Vec::with_capacity(splits);

        for i in 0..splits {
            let start_idx = i * split_size;
            let end_idx = if i == splits - 1 {
                n_samples
            } else {
                (i + 1) * split_size
            };

            let split_features = features.slice(scirs2_core::ndarray::s![start_idx..end_idx, ..]);

            // Convert features to probabilities (assuming they're logits)
            let probabilities = split_features.mapv(|x| F::one() / (F::one() + (-x).exp()));

            // Compute marginal probability
            let marginal = probabilities.mean_axis(Axis(0)).unwrap();

            // Compute KL divergence for each sample
            let mut kl_sum = F::zero();
            let mut valid_samples = 0;

            for sample_idx in 0..probabilities.nrows() {
                let sample_probs = probabilities.row(sample_idx);
                let mut sample_kl = F::zero();
                let mut valid_probs = 0;

                for (&p_sample, &p_marginal) in sample_probs.iter().zip(marginal.iter()) {
                    if p_sample > F::zero() && p_marginal > F::zero() {
                        sample_kl = sample_kl + p_sample * (p_sample / p_marginal).ln();
                        valid_probs += 1;
                    }
                }

                if valid_probs > 0 {
                    kl_sum = kl_sum + sample_kl;
                    valid_samples += 1;
                }
            }

            if valid_samples > 0 {
                let mean_kl = kl_sum / F::from(valid_samples).unwrap();
                scores.push(mean_kl.exp());
            } else {
                scores.push(F::one());
            }
        }

        let mean_score = scores.iter().copied().sum::<F>() / F::from(scores.len()).unwrap();
        let std_score = {
            let variance = scores
                .iter()
                .map(|&x| {
                    let diff = x - mean_score;
                    diff * diff
                })
                .sum::<F>()
                / F::from(scores.len()).unwrap();
            variance.sqrt()
        };

        Ok(InceptionScoreResult {
            mean_score,
            std_score,
            split_scores: scores,
        })
    }

    /// Compute Fréchet Inception Distance (FID)
    pub fn frechet_inception_distance(
        &self,
        real_features: &Array2<F>,
        fake_features: &Array2<F>,
    ) -> Result<F> {
        if real_features.is_empty() || fake_features.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty feature arrays".to_string(),
            ));
        }

        if real_features.ncols() != fake_features.ncols() {
            return Err(MetricsError::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }

        // Compute means
        let mu_real = real_features.mean_axis(Axis(0)).unwrap();
        let mu_fake = fake_features.mean_axis(Axis(0)).unwrap();

        // Compute covariances
        let cov_real = self.compute_covariance_matrix(real_features)?;
        let cov_fake = self.compute_covariance_matrix(fake_features)?;

        // Compute squared L2 distance between means
        let mean_diff = &mu_real - &mu_fake;
        let mean_dist_sq = mean_diff.mapv(|x| x * x).sum();

        // Compute trace of covariances
        let trace_cov_real: F = (0..cov_real.nrows()).map(|i| cov_real[[i, i]]).sum();
        let trace_cov_fake: F = (0..cov_fake.nrows()).map(|i| cov_fake[[i, i]]).sum();

        // Compute product of covariances (simplified to diagonal approximation for efficiency)
        let mut trace_product = F::zero();
        for i in 0..cov_real.nrows() {
            for j in 0..cov_fake.ncols() {
                if i == j {
                    trace_product = trace_product + (cov_real[[i, j]] * cov_fake[[i, j]]).sqrt();
                }
            }
        }
        let trace_product_sqrt = trace_product;

        // FID = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
        let fid = mean_dist_sq + trace_cov_real + trace_cov_fake
            - F::from(2.0).unwrap() * trace_product_sqrt;

        Ok(fid)
    }

    /// Compute Kernel Inception Distance (KID)
    pub fn kernel_inception_distance(
        &self,
        real_features: &Array2<F>,
        fake_features: &Array2<F>,
        degree: usize,
        gamma: Option<F>,
    ) -> Result<KIDResult<F>> {
        if real_features.is_empty() || fake_features.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty feature arrays".to_string(),
            ));
        }

        let n_real = real_features.nrows().min(self.n_kid_samples);
        let n_fake = fake_features.nrows().min(self.n_kid_samples);

        // Subsample for efficiency
        let real_sub = real_features.slice(scirs2_core::ndarray::s![0..n_real, ..]);
        let fake_sub = fake_features.slice(scirs2_core::ndarray::s![0..n_fake, ..]);

        // Compute polynomial kernel matrices
        let gamma_val = gamma.unwrap_or_else(|| F::one() / F::from(real_features.ncols()).unwrap());

        let k_rr = self.compute_polynomial_kernel(&real_sub, &real_sub, degree, gamma_val)?;
        let k_ff = self.compute_polynomial_kernel(&fake_sub, &fake_sub, degree, gamma_val)?;
        let k_rf = self.compute_polynomial_kernel(&real_sub, &fake_sub, degree, gamma_val)?;

        // Compute KID estimate
        let term1 = k_rr.sum() / F::from(n_real * n_real).unwrap();
        let term2 = k_ff.sum() / F::from(n_fake * n_fake).unwrap();
        let term3 = k_rf.sum() / F::from(n_real * n_fake).unwrap();

        let kid_estimate = term1 + term2 - F::from(2.0).unwrap() * term3;

        // Compute bias correction
        let bias_correction = self.compute_kid_bias_correction(n_real, n_fake, &k_rr, &k_ff)?;
        let kid_corrected = kid_estimate - bias_correction;

        Ok(KIDResult {
            kid_estimate,
            kid_corrected,
            bias_correction,
            n_samples_real: n_real,
            n_samples_fake: n_fake,
        })
    }

    /// Compute covariance matrix
    fn compute_covariance_matrix(&self, features: &Array2<F>) -> Result<Array2<F>> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        if n_samples < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 samples for covariance".to_string(),
            ));
        }

        // Center the data
        let mean = features.mean_axis(Axis(0)).unwrap();
        let centered = features - &mean.insert_axis(Axis(0));

        // Compute covariance matrix: (1/(n-1)) * X^T * X
        let mut cov = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in i..n_features {
                let mut sum = F::zero();
                for k in 0..n_samples {
                    sum = sum + centered[[k, i]] * centered[[k, j]];
                }
                let cov_val = sum / F::from(n_samples - 1).unwrap();
                cov[[i, j]] = cov_val;
                if i != j {
                    cov[[j, i]] = cov_val; // Symmetric
                }
            }
        }

        Ok(cov)
    }

    /// Compute polynomial kernel matrix
    fn compute_polynomial_kernel(
        &self,
        x1: &ArrayView2<F>,
        x2: &ArrayView2<F>,
        degree: usize,
        gamma: F,
    ) -> Result<Array2<F>> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut kernel = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                // Compute dot product
                let mut dot_product = F::zero();
                for k in 0..x1.ncols() {
                    dot_product = dot_product + x1[[i, k]] * x2[[j, k]];
                }

                // Polynomial kernel: (gamma * <x1, x2> + 1)^degree
                let kernel_val = (gamma * dot_product + F::one()).powf(F::from(degree).unwrap());
                kernel[[i, j]] = kernel_val;
            }
        }

        Ok(kernel)
    }

    /// Compute KID bias correction
    fn compute_kid_bias_correction(
        &self,
        n_real: usize,
        n_fake: usize,
        k_rr: &Array2<F>,
        k_ff: &Array2<F>,
    ) -> Result<F> {
        // Simplified bias correction (diagonal terms)
        let diag_rr = (0..n_real).map(|i| k_rr[[i, i]]).sum::<F>() / F::from(n_real).unwrap();
        let diag_ff = (0..n_fake).map(|i| k_ff[[i, i]]).sum::<F>() / F::from(n_fake).unwrap();

        let bias = (diag_rr / F::from(n_real).unwrap()) + (diag_ff / F::from(n_fake).unwrap());
        Ok(bias)
    }
}
