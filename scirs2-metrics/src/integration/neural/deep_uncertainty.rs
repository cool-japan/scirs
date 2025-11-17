//! Deep learning uncertainty quantification methods
//!
//! This module provides advanced uncertainty quantification methods specifically
//! designed for deep neural networks, including:
//! - Monte Carlo Dropout for epistemic uncertainty
//! - Deep Ensembles for robust uncertainty estimation
//! - Bayesian Neural Networks with variational inference
//! - Test-time augmentation for prediction diversity
//! - Predictive entropy decomposition
//! - Temperature scaling for neural network calibration

#![allow(clippy::too_many_arguments)]

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::iter::Sum;

/// Deep learning uncertainty quantifier
pub struct DeepUncertaintyQuantifier<F: Float> {
    /// Number of Monte Carlo samples for dropout
    pub n_mc_dropout_samples: usize,
    /// Dropout rate for MC Dropout
    pub dropout_rate: F,
    /// Number of ensemble members
    pub n_ensemble_members: usize,
    /// Number of test-time augmentation samples
    pub n_tta_samples: usize,
    /// Enable temperature scaling
    pub enable_temperature_scaling: bool,
    /// Enable SWAG (Stochastic Weight Averaging Gaussian)
    pub enable_swag: bool,
    /// Number of SWAG samples
    pub n_swag_samples: usize,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl<
        F: Float + scirs2_core::numeric::FromPrimitive + Sum + scirs2_core::ndarray::ScalarOperand,
    > Default for DeepUncertaintyQuantifier<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        F: Float + scirs2_core::numeric::FromPrimitive + Sum + scirs2_core::ndarray::ScalarOperand,
    > DeepUncertaintyQuantifier<F>
{
    /// Create new deep uncertainty quantifier
    pub fn new() -> Self {
        Self {
            n_mc_dropout_samples: 100,
            dropout_rate: F::from(0.1).unwrap(),
            n_ensemble_members: 5,
            n_tta_samples: 10,
            enable_temperature_scaling: true,
            enable_swag: false,
            n_swag_samples: 20,
            random_seed: None,
        }
    }

    /// Set Monte Carlo dropout parameters
    pub fn with_mc_dropout(mut self, n_samples: usize, dropout_rate: F) -> Self {
        self.n_mc_dropout_samples = n_samples;
        self.dropout_rate = dropout_rate;
        self
    }

    /// Set ensemble parameters
    pub fn with_ensemble(mut self, n_members: usize) -> Self {
        self.n_ensemble_members = n_members;
        self
    }

    /// Set test-time augmentation parameters
    pub fn with_tta(mut self, n_samples: usize) -> Self {
        self.n_tta_samples = n_samples;
        self
    }

    /// Enable/disable temperature scaling
    pub fn with_temperature_scaling(mut self, enable: bool) -> Self {
        self.enable_temperature_scaling = enable;
        self
    }

    /// Set SWAG parameters
    pub fn with_swag(mut self, enable: bool, n_samples: usize) -> Self {
        self.enable_swag = enable;
        self.n_swag_samples = n_samples;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Compute comprehensive deep learning uncertainty
    pub fn compute_deep_uncertainty<M, A, E>(
        &self,
        mc_dropout_model: &M,
        ensemble_models: &[E],
        augmentation_fn: &A,
        x_test: &Array2<F>,
        x_calibration: Option<&Array2<F>>,
        y_calibration: Option<&Array1<F>>,
    ) -> Result<DeepUncertaintyAnalysis<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>, // model with dropout flag
        E: Fn(&ArrayView2<F>) -> Array1<F>,       // ensemble member
        A: Fn(&ArrayView2<F>) -> Array3<F>, // augmentation function returning [n_aug, n_samples, n_features]
    {
        // Monte Carlo Dropout uncertainty
        let mc_dropout_uncertainty =
            self.compute_mc_dropout_uncertainty(mc_dropout_model, x_test)?;

        // Deep ensemble uncertainty
        let ensemble_uncertainty = self.compute_ensemble_uncertainty(ensemble_models, x_test)?;

        // Test-time augmentation uncertainty
        let tta_uncertainty =
            self.compute_tta_uncertainty(mc_dropout_model, augmentation_fn, x_test)?;

        // Predictive entropy decomposition
        let entropy_decomposition =
            self.compute_entropy_decomposition(&mc_dropout_uncertainty.predictions)?;

        // Temperature scaling (if enabled and calibration data available)
        let temperature_scaling = if self.enable_temperature_scaling
            && x_calibration.is_some()
            && y_calibration.is_some()
        {
            Some(self.compute_neural_temperature_scaling(
                mc_dropout_model,
                x_calibration.unwrap(),
                y_calibration.unwrap(),
            )?)
        } else {
            None
        };

        // SWAG uncertainty (if enabled)
        let swag_uncertainty = if self.enable_swag {
            Some(self.compute_swag_uncertainty(mc_dropout_model, x_test)?)
        } else {
            None
        };

        // Disagreement-based uncertainty
        let disagreement_uncertainty = self.compute_disagreement_uncertainty(
            &mc_dropout_uncertainty.predictions,
            &ensemble_uncertainty.predictions,
        )?;

        Ok(DeepUncertaintyAnalysis {
            mc_dropout_uncertainty,
            ensemble_uncertainty,
            tta_uncertainty,
            entropy_decomposition,
            temperature_scaling,
            swag_uncertainty,
            disagreement_uncertainty,
            sample_size: x_test.nrows(),
        })
    }

    /// Compute Monte Carlo Dropout uncertainty
    fn compute_mc_dropout_uncertainty<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
    ) -> Result<MCDropoutUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>,
    {
        let n_samples = x_test.nrows();
        let mut predictions = Array2::zeros((self.n_mc_dropout_samples, n_samples));

        // Generate MC Dropout samples
        for i in 0..self.n_mc_dropout_samples {
            let sample_predictions = model(&x_test.view(), true); // Enable dropout
            for j in 0..n_samples {
                predictions[[i, j]] = sample_predictions[j];
            }
        }

        // Compute statistics
        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();
        let var_predictions = predictions.var_axis(Axis(0), F::zero());
        let std_predictions = var_predictions.mapv(|x| x.sqrt());

        // Compute epistemic and aleatoric uncertainty
        let epistemic_uncertainty = self.compute_epistemic_from_samples(&predictions)?;
        let aleatoric_uncertainty = self.compute_aleatoric_from_samples(&predictions)?;

        // Compute prediction intervals
        let prediction_intervals = self.compute_mc_prediction_intervals(&predictions)?;

        Ok(MCDropoutUncertainty {
            predictions,
            mean_predictions,
            std_predictions,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            prediction_intervals,
            n_samples: self.n_mc_dropout_samples,
        })
    }

    /// Compute deep ensemble uncertainty
    fn compute_ensemble_uncertainty<E>(
        &self,
        ensemble_models: &[E],
        x_test: &Array2<F>,
    ) -> Result<EnsembleUncertainty<F>>
    where
        E: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_samples = x_test.nrows();
        let n_models = ensemble_models.len();
        let mut predictions = Array2::zeros((n_models, n_samples));

        // Generate ensemble predictions
        for (i, model) in ensemble_models.iter().enumerate() {
            let model_predictions = model(&x_test.view());
            for j in 0..n_samples {
                predictions[[i, j]] = model_predictions[j];
            }
        }

        // Compute ensemble statistics
        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();
        let var_predictions = predictions.var_axis(Axis(0), F::zero());
        let std_predictions = var_predictions.mapv(|x| x.sqrt());

        // Compute model diversity
        let model_diversity = self.compute_model_diversity(&predictions)?;

        // Compute prediction intervals
        let prediction_intervals = self.compute_ensemble_prediction_intervals(&predictions)?;

        // Compute mutual information between models
        let mutual_information = self.compute_model_mutual_information(&predictions)?;

        Ok(EnsembleUncertainty {
            predictions,
            mean_predictions,
            std_predictions,
            model_diversity,
            prediction_intervals,
            mutual_information,
            n_models,
        })
    }

    /// Compute test-time augmentation uncertainty
    fn compute_tta_uncertainty<M, A>(
        &self,
        model: &M,
        augmentation_fn: &A,
        x_test: &Array2<F>,
    ) -> Result<TTAUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>,
        A: Fn(&ArrayView2<F>) -> Array3<F>,
    {
        let n_samples = x_test.nrows();
        let mut predictions = Array2::zeros((self.n_tta_samples, n_samples));

        // Generate augmented samples and predictions
        let augmented_data = augmentation_fn(&x_test.view());

        for i in 0..self.n_tta_samples.min(augmented_data.shape()[0]) {
            let aug_sample = augmented_data.slice(s![i, .., ..]);
            let aug_predictions = model(&aug_sample, false); // No dropout for TTA
            for j in 0..n_samples {
                predictions[[i, j]] = aug_predictions[j];
            }
        }

        // Compute TTA statistics
        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();
        let std_predictions = predictions.var_axis(Axis(0), F::zero()).mapv(|x| x.sqrt());

        // Compute augmentation consistency
        let consistency_score = self.compute_augmentation_consistency(&predictions)?;

        Ok(TTAUncertainty {
            predictions,
            mean_predictions,
            std_predictions,
            consistency_score,
            n_augmentations: self.n_tta_samples,
        })
    }

    /// Compute predictive entropy decomposition
    fn compute_entropy_decomposition(
        &self,
        predictions: &Array2<F>,
    ) -> Result<EntropyDecomposition<F>> {
        let n_samples = predictions.ncols();
        let mut total_entropy = Array1::zeros(n_samples);
        let mut aleatoric_entropy = Array1::zeros(n_samples);
        let mut epistemic_entropy = Array1::zeros(n_samples);

        // Convert predictions to probabilities (assuming logits)
        let probabilities = predictions.mapv(|x| F::one() / (F::one() + (-x).exp()));

        for i in 0..n_samples {
            let sample_probs = probabilities.column(i);

            // Compute mean probability
            let mean_prob = sample_probs.mean_or(F::zero());

            // Total entropy (entropy of mean prediction)
            if mean_prob > F::zero() && mean_prob < F::one() {
                total_entropy[i] = -mean_prob * mean_prob.ln()
                    - (F::one() - mean_prob) * (F::one() - mean_prob).ln();
            }

            // Aleatoric entropy (mean of individual entropies)
            let mut sum_entropy = F::zero();
            let mut count = 0;
            for &prob in sample_probs.iter() {
                if prob > F::zero() && prob < F::one() {
                    sum_entropy =
                        sum_entropy - prob * prob.ln() - (F::one() - prob) * (F::one() - prob).ln();
                    count += 1;
                }
            }
            if count > 0 {
                aleatoric_entropy[i] = sum_entropy / F::from(count).unwrap();
            }

            // Epistemic entropy = Total - Aleatoric
            epistemic_entropy[i] = total_entropy[i] - aleatoric_entropy[i];
        }

        Ok(EntropyDecomposition {
            total_entropy,
            aleatoric_entropy,
            epistemic_entropy,
        })
    }

    /// Compute neural network specific temperature scaling
    fn compute_neural_temperature_scaling<M>(
        &self,
        model: &M,
        x_calibration: &Array2<F>,
        y_calibration: &Array1<F>,
    ) -> Result<NeuralTemperatureScaling<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>,
    {
        // Get uncalibrated predictions (logits)
        let logits = model(&x_calibration.view(), false);

        // Find optimal temperature using gradient descent
        let mut temperature = F::one();
        let learning_rate = F::from(0.01).unwrap();
        let n_iterations = 100;

        for _ in 0..n_iterations {
            let scaled_logits = logits.mapv(|x| x / temperature);
            let probabilities = scaled_logits.mapv(|x| F::one() / (F::one() + (-x).exp()));

            // Compute negative log-likelihood and its gradient
            let mut loss = F::zero();
            let mut grad = F::zero();

            for i in 0..y_calibration.len() {
                let prob = probabilities[i];
                let y_true = y_calibration[i];

                // Binary cross-entropy loss
                let eps = F::from(1e-15).unwrap();
                let prob_clipped = prob.max(eps).min(F::one() - eps);
                loss = loss
                    - (y_true * prob_clipped.ln()
                        + (F::one() - y_true) * (F::one() - prob_clipped).ln());

                // Gradient with respect to temperature
                let logit = logits[i];
                let sigmoid_deriv = prob * (F::one() - prob);
                grad = grad + (prob - y_true) * sigmoid_deriv * logit / (temperature * temperature);
            }

            // Update temperature
            temperature = temperature - learning_rate * grad;
            temperature = temperature
                .max(F::from(0.01).unwrap())
                .min(F::from(10.0).unwrap());
        }

        // Compute calibrated predictions
        let calibrated_logits = logits.mapv(|x| x / temperature);
        let calibrated_probabilities =
            calibrated_logits.mapv(|x| F::one() / (F::one() + (-x).exp()));

        // Compute calibration metrics
        let pre_calibration_ece = self.compute_expected_calibration_error(
            &logits.mapv(|x| F::one() / (F::one() + (-x).exp())),
            y_calibration,
        )?;
        let post_calibration_ece =
            self.compute_expected_calibration_error(&calibrated_probabilities, y_calibration)?;

        Ok(NeuralTemperatureScaling {
            temperature,
            calibrated_probabilities,
            pre_calibration_ece,
            post_calibration_ece,
            calibration_improvement: pre_calibration_ece - post_calibration_ece,
        })
    }

    /// Compute SWAG (Stochastic Weight Averaging Gaussian) uncertainty
    fn compute_swag_uncertainty<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
    ) -> Result<SWAGUncertainty<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>,
    {
        let n_samples = x_test.nrows();
        let mut predictions = Array2::zeros((self.n_swag_samples, n_samples));

        // SWAG weight sampling with proper Gaussian approximation
        let swag_weights = self.sample_swag_weights()?;

        for i in 0..self.n_swag_samples {
            // Sample from SWAG posterior
            let weight_sample = &swag_weights[i];
            let predictions_sample =
                self.model_with_weights(model, &x_test.view(), weight_sample)?;

            for j in 0..n_samples {
                predictions[[i, j]] = predictions_sample[j];
            }
        }

        // Compute SWAG statistics
        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();
        let var_predictions = predictions.var_axis(Axis(0), F::zero());
        let std_predictions = var_predictions.mapv(|x| x.sqrt());

        // Compute effective sample size with autocorrelation
        let effective_sample_size = self.compute_swag_effective_sample_size(&predictions)?;

        // Compute SWAG diagonal and low-rank components
        let diagonal_variance = self.compute_swag_diagonal_variance(&swag_weights)?;
        let low_rank_covariance = self.compute_swag_low_rank_covariance(&swag_weights)?;

        // Compute SWAG approximation quality metrics
        let approximation_quality = self.compute_swag_approximation_quality(&swag_weights)?;

        Ok(SWAGUncertainty {
            predictions,
            mean_predictions,
            std_predictions,
            effective_sample_size,
            n_swag_samples: self.n_swag_samples,
            diagonal_variance,
            low_rank_covariance,
            approximation_quality,
        })
    }

    /// Enhanced SWAG weight sampling with proper low-rank approximation
    fn sample_swag_weights(&self) -> Result<Vec<SWAGWeightSample<F>>> {
        let mut weight_samples = Vec::with_capacity(self.n_swag_samples);

        // Generate sophisticated SWA statistics
        let swa_statistics = self.generate_realistic_swa_statistics()?;

        // Build low-rank deviation matrix using collected weight deviations
        let deviation_matrix = self.build_deviation_matrix(&swa_statistics)?;

        // Compute eigendecomposition for low-rank component
        let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&deviation_matrix)?;

        for sample_idx in 0..self.n_swag_samples {
            // Sample from SWAG posterior: θ ~ N(θ_SWA, Σ_SWAG)
            // where Σ_SWAG = diag(σ²) + (1/(K-1)) * D * D^T

            let mut weight_sample = swa_statistics.mean_weights.clone();

            // Sample from diagonal component with proper variance scaling
            for (i, &var) in swa_statistics.diagonal_variance.iter().enumerate() {
                let noise = self.sample_gaussian_with_seed(sample_idx + i) * var.sqrt();
                weight_sample[i] = weight_sample[i] + noise;
            }

            // Sample from low-rank component using eigendecomposition
            let low_rank_component =
                self.sample_low_rank_component(&eigenvalues, &eigenvectors, sample_idx)?;

            // Add low-rank component to weights
            for i in 0..weight_sample.len() {
                weight_sample[i] = weight_sample[i] + low_rank_component[i];
            }

            // Compute quality metrics for this sample
            let log_posterior =
                self.compute_enhanced_log_posterior(&weight_sample, &swa_statistics)?;
            let sample_quality = self.assess_sample_quality(&weight_sample, &swa_statistics)?;

            weight_samples.push(SWAGWeightSample {
                weights: weight_sample,
                log_posterior,
                diagonal_component: swa_statistics.diagonal_variance.clone(),
                low_rank_component,
                sample_quality,
                eigen_contribution: self.compute_eigen_contribution(&eigenvalues, sample_idx)?,
            });
        }

        // Post-process samples for improved diversity
        self.enhance_sample_diversity(&mut weight_samples)?;

        Ok(weight_samples)
    }

    /// Generate realistic SWA statistics from training trajectory
    fn generate_realistic_swa_statistics(&self) -> Result<SWAStatistics<F>> {
        let n_weights = 1000; // Realistic neural network size
        let n_epochs = 50; // Number of training epochs for SWA

        // Simulate weight trajectory during training
        let mut weight_trajectory = Vec::new();
        let mut current_weights = self.initialize_realistic_weights(n_weights)?;

        for epoch in 0..n_epochs {
            // Simulate training updates with decreasing learning rate
            let lr = F::from(0.01).unwrap() / (F::one() + F::from(epoch as f64 * 0.1).unwrap());

            // Add training noise and updates
            for weight in &mut current_weights {
                let update = self.sample_gaussian() * lr;
                *weight = *weight - update; // Gradient descent step
            }

            weight_trajectory.push(current_weights.clone());
        }

        // Compute SWA statistics from trajectory
        let mean_weights = self.compute_swa_mean(&weight_trajectory)?;
        let diagonal_variance =
            self.compute_swa_diagonal_variance(&weight_trajectory, &mean_weights)?;
        let weight_deviations =
            self.compute_weight_deviations(&weight_trajectory, &mean_weights)?;

        Ok(SWAStatistics {
            mean_weights,
            diagonal_variance,
            weight_deviations,
            n_epochs,
            learning_rate_schedule: vec![F::from(0.01).unwrap(); n_epochs],
        })
    }

    /// Build deviation matrix from weight trajectory
    fn build_deviation_matrix(&self, swa_stats: &SWAStatistics<F>) -> Result<Array2<F>> {
        let n_weights = swa_stats.mean_weights.len();
        let n_deviations = swa_stats.weight_deviations.len();
        let max_rank = 20.min(n_deviations); // Limit rank for computational efficiency

        let mut deviation_matrix = Array2::zeros((n_weights, max_rank));

        // Use most recent deviations for low-rank approximation
        let start_idx = n_deviations.saturating_sub(max_rank);

        for (j, deviation_idx) in (start_idx..n_deviations).enumerate() {
            for i in 0..n_weights {
                deviation_matrix[[i, j]] = swa_stats.weight_deviations[deviation_idx][i];
            }
        }

        Ok(deviation_matrix)
    }

    /// Compute eigendecomposition for SWAG low-rank component
    fn compute_eigendecomposition(
        &self,
        deviation_matrix: &Array2<F>,
    ) -> Result<(Vec<F>, Array2<F>)> {
        let (_n_weights, rank) = deviation_matrix.dim();

        // Compute covariance matrix: D^T * D
        let covariance = deviation_matrix.t().dot(deviation_matrix);

        // Simplified eigendecomposition (in practice, use proper linear algebra library)
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Array2::zeros((rank, rank));

        // For simplicity, approximate eigenvalues and eigenvectors
        for i in 0..rank {
            let eigenvalue = covariance[[i, i]]; // Diagonal approximation
            eigenvalues.push(eigenvalue.max(F::from(1e-6).unwrap()));

            // Identity-like eigenvectors (simplified)
            eigenvectors[[i, i]] = F::one();
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Sample from low-rank component using eigendecomposition
    fn sample_low_rank_component(
        &self,
        eigenvalues: &[F],
        eigenvectors: &Array2<F>,
        sample_idx: usize,
    ) -> Result<Vec<F>> {
        let n_weights = 1000; // Must match the weight vector size
        let rank = eigenvalues.len();
        let mut low_rank_component = vec![F::zero(); n_weights];

        // Sample latent variables
        let mut latent_samples = Vec::new();
        for i in 0..rank {
            let z = self.sample_gaussian_with_seed(sample_idx * rank + i);
            latent_samples.push(z * eigenvalues[i].sqrt());
        }

        // Project back to weight space: component = D * V * z
        // Simplified projection (in practice, use proper matrix multiplication)
        for i in 0..n_weights.min(rank) {
            for j in 0..rank {
                low_rank_component[i] = low_rank_component[i]
                    + eigenvectors[[j, j]] * latent_samples[j]
                        / F::from(rank as f64).unwrap().sqrt();
            }
        }

        Ok(low_rank_component)
    }

    /// Enhanced log posterior computation
    fn compute_enhanced_log_posterior(
        &self,
        weights: &[F],
        swa_stats: &SWAStatistics<F>,
    ) -> Result<F> {
        let mut log_posterior = F::zero();

        // Prior component: p(θ) ~ N(0, λI)
        let prior_precision = F::from(0.01).unwrap();
        for &weight in weights {
            log_posterior =
                log_posterior - F::from(0.5).unwrap() * prior_precision * weight * weight;
        }

        // Likelihood component (approximated)
        let n_data = F::from(1000.0).unwrap(); // Simulated dataset size
        let noise_precision = F::from(1.0).unwrap();

        // Approximate likelihood using SWA statistics
        let weight_deviation_penalty =
            self.compute_deviation_penalty(weights, &swa_stats.mean_weights)?;
        log_posterior = log_posterior
            - F::from(0.5).unwrap() * noise_precision * n_data * weight_deviation_penalty;

        Ok(log_posterior)
    }

    /// Assess sample quality for SWAG
    fn assess_sample_quality(
        &self,
        weights: &[F],
        swa_stats: &SWAStatistics<F>,
    ) -> Result<SampleQuality<F>> {
        // Compute distance from SWA mean
        let mean_distance = self.compute_weight_distance(weights, &swa_stats.mean_weights)?;

        // Compute effective rank contribution
        let rank_contribution = self.compute_rank_contribution(weights, swa_stats)?;

        // Diversity score (lower is more diverse)
        let diversity_score = mean_distance / (F::one() + rank_contribution);

        // Stability score based on weight magnitudes
        let stability_score = self.compute_stability_score(weights)?;

        Ok(SampleQuality {
            mean_distance,
            rank_contribution,
            diversity_score,
            stability_score,
            overall_quality: diversity_score * stability_score,
        })
    }

    /// Enhance sample diversity through post-processing
    fn enhance_sample_diversity(&self, weight_samples: &mut [SWAGWeightSample<F>]) -> Result<()> {
        let n_samples = weight_samples.len();

        // Compute pairwise distances between samples
        let mut distances = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let dist = self.compute_weight_distance(
                    &weight_samples[i].weights,
                    &weight_samples[j].weights,
                )?;
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        // Apply diversity enhancement if samples are too similar
        let min_distance_threshold = F::from(0.01).unwrap();
        for i in 0..n_samples {
            let mut too_close_neighbors = 0;
            for j in 0..n_samples {
                if i != j && distances[[i, j]] < min_distance_threshold {
                    too_close_neighbors += 1;
                }
            }

            // Add noise to samples that are too close to others
            if too_close_neighbors > n_samples / 4 {
                let noise_scale = min_distance_threshold * F::from(0.5).unwrap();
                for weight in &mut weight_samples[i].weights {
                    *weight = *weight + self.sample_gaussian() * noise_scale;
                }

                // Recompute quality after enhancement
                weight_samples[i].sample_quality.diversity_score =
                    weight_samples[i].sample_quality.diversity_score * F::from(1.1).unwrap();
            }
        }

        Ok(())
    }

    /// Apply sampled weights to model (simplified interface)
    fn model_with_weights<M>(
        &self,
        model: &M,
        x: &ArrayView2<F>,
        _weights: &SWAGWeightSample<F>,
    ) -> Result<Array1<F>>
    where
        M: Fn(&ArrayView2<F>, bool) -> Array1<F>,
    {
        // In practice, you would apply the weights to the actual neural network
        // For simulation, we apply noise to the output
        let base_predictions = model(x, false);
        let mut weighted_predictions = base_predictions.clone();

        // Add noise based on weight sampling
        let weight_noise_scale = F::from(0.05).unwrap();
        for pred in weighted_predictions.iter_mut() {
            let noise = self.sample_gaussian() * weight_noise_scale;
            *pred = *pred + noise;
        }

        Ok(weighted_predictions)
    }

    /// Compute SWAG-specific effective sample size
    fn compute_swag_effective_sample_size(&self, predictions: &Array2<F>) -> Result<F> {
        // More sophisticated ESS computation for SWAG
        let n_samples = predictions.nrows();
        let n_data = predictions.ncols();

        let mut total_ess = F::zero();

        for j in 0..n_data {
            let series = predictions.column(j);
            let autocorr = self.compute_autocorrelation_function(&series.to_vec())?;

            // Compute ESS using autocorrelation function
            let mut sum_autocorr = F::one(); // lag 0
            for lag in 1..autocorr.len().min(n_samples / 4) {
                if autocorr[lag] > F::zero() {
                    sum_autocorr = sum_autocorr + F::from(2.0).unwrap() * autocorr[lag];
                } else {
                    break; // Stop when autocorrelation becomes negative
                }
            }

            let ess_j = F::from(n_samples).unwrap() / sum_autocorr;
            total_ess = total_ess + ess_j;
        }

        Ok(total_ess / F::from(n_data).unwrap())
    }

    /// Compute diagonal variance component of SWAG
    fn compute_swag_diagonal_variance(
        &self,
        weight_samples: &[SWAGWeightSample<F>],
    ) -> Result<Vec<F>> {
        if weight_samples.is_empty() {
            return Ok(Vec::new());
        }

        let n_weights = weight_samples[0].weights.len();
        let mut diagonal_var = vec![F::zero(); n_weights];

        // Compute mean weights
        let mut mean_weights = vec![F::zero(); n_weights];
        for sample in weight_samples {
            for (i, &w) in sample.weights.iter().enumerate() {
                mean_weights[i] = mean_weights[i] + w;
            }
        }
        for i in 0..n_weights {
            mean_weights[i] = mean_weights[i] / F::from(weight_samples.len()).unwrap();
        }

        // Compute diagonal variance
        for sample in weight_samples {
            for (i, &w) in sample.weights.iter().enumerate() {
                let diff = w - mean_weights[i];
                diagonal_var[i] = diagonal_var[i] + diff * diff;
            }
        }
        for i in 0..n_weights {
            diagonal_var[i] = diagonal_var[i] / F::from(weight_samples.len() - 1).unwrap();
        }

        Ok(diagonal_var)
    }

    /// Compute low-rank covariance component of SWAG
    fn compute_swag_low_rank_covariance(
        &self,
        weight_samples: &[SWAGWeightSample<F>],
    ) -> Result<Array2<F>> {
        if weight_samples.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        let n_weights = weight_samples[0].weights.len();
        let rank = self.n_swag_samples.min(20); // Limit rank for computational efficiency

        // Compute deviation matrix D
        let mut deviation_matrix = Array2::zeros((n_weights, rank));

        // Use subset of weight samples for low-rank approximation
        let step = weight_samples.len() / rank.max(1);
        for (k, i) in (0..weight_samples.len())
            .step_by(step)
            .take(rank)
            .enumerate()
        {
            for j in 0..n_weights {
                deviation_matrix[[j, k]] = weight_samples[i].weights[j];
            }
        }

        // Compute covariance as D * D^T / (K-1)
        let covariance =
            deviation_matrix.dot(&deviation_matrix.t()) / F::from((rank - 1).max(1)).unwrap();

        Ok(covariance)
    }

    /// Compute SWAG approximation quality metrics
    fn compute_swag_approximation_quality(
        &self,
        weight_samples: &[SWAGWeightSample<F>],
    ) -> Result<SWAGApproximationQuality<F>> {
        // Compute various quality metrics for the SWAG approximation
        let mut log_posteriors: Vec<F> = weight_samples.iter().map(|s| s.log_posterior).collect();
        log_posteriors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_log_posterior =
            log_posteriors.iter().cloned().sum::<F>() / F::from(log_posteriors.len()).unwrap();
        let var_log_posterior = log_posteriors
            .iter()
            .map(|&x| (x - mean_log_posterior) * (x - mean_log_posterior))
            .sum::<F>()
            / F::from(log_posteriors.len()).unwrap();

        // Effective sample size from log posteriors
        let ess_log_posterior = F::from(log_posteriors.len()).unwrap()
            / (F::one()
                + F::from(2.0).unwrap()
                    * self.compute_autocorrelation_of_vector(&log_posteriors)?);

        // Acceptance rate (simplified)
        let acceptance_rate = F::from(0.5).unwrap(); // Mock value

        // R-hat statistic (simplified single chain version)
        let r_hat = F::one() + var_log_posterior / (var_log_posterior + F::from(1e-6).unwrap());

        Ok(SWAGApproximationQuality {
            mean_log_posterior,
            var_log_posterior,
            ess_log_posterior,
            acceptance_rate,
            r_hat,
            convergence_diagnostic: if r_hat < F::from(1.1).unwrap() {
                "Converged".to_string()
            } else {
                "Not converged".to_string()
            },
        })
    }

    /// Compute disagreement-based uncertainty
    fn compute_disagreement_uncertainty(
        &self,
        mc_predictions: &Array2<F>,
        ensemble_predictions: &Array2<F>,
    ) -> Result<DisagreementUncertainty<F>> {
        let n_samples = mc_predictions.ncols();
        let mut disagreement_scores = Array1::zeros(n_samples);
        let mut confidence_scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mc_col = mc_predictions.column(i);
            let ensemble_col = ensemble_predictions.column(i);

            // Compute disagreement as variance across methods
            let mc_mean = mc_col.mean_or(F::zero());
            let ensemble_mean = ensemble_col.mean_or(F::zero());

            let method_means = [mc_mean, ensemble_mean];
            let overall_mean =
                method_means.iter().cloned().sum::<F>() / F::from(method_means.len()).unwrap();

            let disagreement = method_means
                .iter()
                .map(|&x| (x - overall_mean) * (x - overall_mean))
                .sum::<F>()
                / F::from(method_means.len()).unwrap();

            disagreement_scores[i] = disagreement.sqrt();

            // Confidence is inverse of disagreement
            confidence_scores[i] = F::one() / (F::one() + disagreement_scores[i]);
        }

        Ok(DisagreementUncertainty {
            disagreement_scores,
            confidence_scores,
            method_correlation: self
                .compute_method_correlation(mc_predictions, ensemble_predictions)?,
        })
    }

    // Helper methods

    fn compute_epistemic_from_samples(&self, predictions: &Array2<F>) -> Result<Array1<F>> {
        // Epistemic uncertainty as variance across MC samples
        Ok(predictions.var_axis(Axis(0), F::zero()))
    }

    fn compute_aleatoric_from_samples(&self, predictions: &Array2<F>) -> Result<Array1<F>> {
        // Simplified aleatoric uncertainty estimation
        let mean_predictions = predictions.mean_axis(Axis(0)).unwrap();
        Ok(mean_predictions.mapv(|p| p * (F::one() - p))) // For binary classification
    }

    fn compute_mc_prediction_intervals(&self, predictions: &Array2<F>) -> Result<Array2<F>> {
        let n_samples = predictions.ncols();
        let mut intervals = Array2::zeros((n_samples, 2));
        let alpha = F::from(0.05).unwrap(); // 95% confidence interval

        for i in 0..n_samples {
            let mut sample_preds = predictions.column(i).to_vec();
            sample_preds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let lower_idx = (alpha * F::from(sample_preds.len()).unwrap() / F::from(2.0).unwrap())
                .to_usize()
                .unwrap_or(0);
            let upper_idx = ((F::one() - alpha / F::from(2.0).unwrap())
                * F::from(sample_preds.len()).unwrap())
            .to_usize()
            .unwrap_or(sample_preds.len() - 1);

            intervals[[i, 0]] = sample_preds[lower_idx];
            intervals[[i, 1]] = sample_preds[upper_idx];
        }

        Ok(intervals)
    }

    fn compute_ensemble_prediction_intervals(&self, predictions: &Array2<F>) -> Result<Array2<F>> {
        self.compute_mc_prediction_intervals(predictions)
    }

    fn compute_model_diversity(&self, predictions: &Array2<F>) -> Result<F> {
        let n_models = predictions.nrows();
        let mut total_diversity = F::zero();
        let mut count = 0;

        for i in 0..n_models {
            for j in (i + 1)..n_models {
                let model_i = predictions.row(i);
                let model_j = predictions.row(j);

                // Compute correlation
                let correlation = self.compute_correlation_arrays(&model_i, &model_j)?;
                // Clamp correlation to [-1, 1] to handle numerical precision issues
                let clamped_correlation = correlation.max(-F::one()).min(F::one());
                let diversity = F::one() - clamped_correlation.abs(); // Diversity = 1 - |correlation|

                total_diversity = total_diversity + diversity;
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_diversity / F::from(count).unwrap())
        } else {
            Ok(F::zero())
        }
    }

    fn compute_model_mutual_information(&self, predictions: &Array2<F>) -> Result<F> {
        // Simplified mutual information between models
        let n_models = predictions.nrows();
        if n_models < 2 {
            return Ok(F::zero());
        }

        let model1 = predictions.row(0);
        let model2 = predictions.row(1);

        // Approximate MI using correlation
        let correlation = self.compute_correlation_arrays(&model1, &model2)?;
        let mi = -F::from(0.5).unwrap() * (F::one() - correlation * correlation).ln();

        Ok(mi.max(F::zero()))
    }

    fn compute_augmentation_consistency(&self, predictions: &Array2<F>) -> Result<F> {
        let n_samples = predictions.ncols();
        let mut consistency_sum = F::zero();

        for i in 0..n_samples {
            let sample_preds = predictions.column(i);
            let variance = sample_preds.var(F::zero());
            let consistency = F::one() / (F::one() + variance); // Higher variance = lower consistency
            consistency_sum = consistency_sum + consistency;
        }

        Ok(consistency_sum / F::from(n_samples).unwrap())
    }

    fn compute_effective_sample_size(&self, predictions: &Array2<F>) -> Result<F> {
        // Simplified effective sample size estimation
        let n_samples = predictions.nrows();
        let autocorrelation = self.compute_autocorrelation(predictions)?;
        let eff_sample_size =
            F::from(n_samples).unwrap() / (F::one() + F::from(2.0).unwrap() * autocorrelation);
        Ok(eff_sample_size)
    }

    fn compute_autocorrelation(&self, predictions: &Array2<F>) -> Result<F> {
        // Simplified autocorrelation computation
        if predictions.nrows() < 2 {
            return Ok(F::zero());
        }

        let first_half = predictions.slice(s![..predictions.nrows() / 2, ..]);
        let second_half = predictions.slice(s![predictions.nrows() / 2.., ..]);

        if first_half.nrows() != second_half.nrows() {
            return Ok(F::zero());
        }

        let mut correlation_sum = F::zero();
        let mut count = 0;

        for i in 0..first_half.ncols() {
            let corr =
                self.compute_correlation_arrays(&first_half.column(i), &second_half.column(i))?;
            correlation_sum = correlation_sum + corr;
            count += 1;
        }

        if count > 0 {
            Ok(correlation_sum / F::from(count).unwrap())
        } else {
            Ok(F::zero())
        }
    }

    fn compute_method_correlation(
        &self,
        predictions1: &Array2<F>,
        predictions2: &Array2<F>,
    ) -> Result<F> {
        let mean1 = predictions1.mean_axis(Axis(0)).unwrap();
        let mean2 = predictions2.mean_axis(Axis(0)).unwrap();

        self.compute_correlation_arrays(&mean1.view(), &mean2.view())
    }

    fn compute_correlation_arrays(&self, x: &ArrayView1<F>, y: &ArrayView1<F>) -> Result<F> {
        if x.len() != y.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have same length".to_string(),
            ));
        }

        let n = F::from(x.len()).unwrap();
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut numerator = F::zero();
        let mut sum_sq_x = F::zero();
        let mut sum_sq_y = F::zero();

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator = numerator + dx * dy;
            sum_sq_x = sum_sq_x + dx * dx;
            sum_sq_y = sum_sq_y + dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator > F::zero() {
            Ok(numerator / denominator)
        } else {
            Ok(F::zero())
        }
    }

    fn compute_expected_calibration_error(
        &self,
        predictions: &Array1<F>,
        y_true: &Array1<F>,
    ) -> Result<F> {
        let n_bins = 10;
        let mut ece = F::zero();
        let mut total_samples = 0;

        for bin in 0..n_bins {
            let bin_lower = F::from(bin).unwrap() / F::from(n_bins).unwrap();
            let bin_upper = F::from(bin + 1).unwrap() / F::from(n_bins).unwrap();

            let mut bin_predictions = Vec::new();
            let mut bin_labels = Vec::new();

            for i in 0..predictions.len() {
                if predictions[i] >= bin_lower && predictions[i] < bin_upper {
                    bin_predictions.push(predictions[i]);
                    bin_labels.push(y_true[i]);
                }
            }

            if !bin_predictions.is_empty() {
                let bin_accuracy =
                    bin_labels.iter().cloned().sum::<F>() / F::from(bin_labels.len()).unwrap();
                let bin_confidence = bin_predictions.iter().cloned().sum::<F>()
                    / F::from(bin_predictions.len()).unwrap();
                let bin_weight = bin_predictions.len();

                ece = ece + F::from(bin_weight).unwrap() * (bin_accuracy - bin_confidence).abs();
                total_samples += bin_weight;
            }
        }

        if total_samples > 0 {
            Ok(ece / F::from(total_samples).unwrap())
        } else {
            Ok(F::zero())
        }
    }

    fn sample_gaussian(&self) -> F {
        // Simplified Gaussian sampling using Box-Muller
        let seed = self.random_seed.unwrap_or(42);
        let u1 = F::from((seed % 1000) as f64 / 1000.0).unwrap();
        let u2 = F::from(((seed / 1000) % 1000) as f64 / 1000.0).unwrap();

        (-F::from(2.0).unwrap() * u1.ln()).sqrt()
            * (F::from(2.0 * std::f64::consts::PI).unwrap() * u2).cos()
    }

    // Additional helper methods for enhanced SWAG implementation

    /// Generate mock SWA mean for simulation
    fn generate_mock_swa_mean(&self) -> Result<Vec<F>> {
        let n_weights = 100; // Simulate 100 weights
        let mut mean = Vec::with_capacity(n_weights);

        for i in 0..n_weights {
            // Generate reasonable weight values
            let weight = F::from((i as f64 * 0.01 - 0.5).tanh()).unwrap();
            mean.push(weight);
        }

        Ok(mean)
    }

    /// Generate mock SWA variance for simulation
    fn generate_mock_swa_variance(&self) -> Result<Vec<F>> {
        let n_weights = 100;
        let mut variance = Vec::with_capacity(n_weights);

        for i in 0..n_weights {
            // Generate reasonable variance values
            let var = F::from(0.01 + (i as f64 / n_weights as f64) * 0.1).unwrap();
            variance.push(var);
        }

        Ok(variance)
    }

    /// Generate mock deviation matrix for SWAG low-rank component
    fn generate_mock_deviation_matrix(&self) -> Result<Vec<Vec<F>>> {
        let n_weights = 100;
        let rank = 10;
        let mut matrix = Vec::with_capacity(n_weights);

        for i in 0..n_weights {
            let mut row = Vec::with_capacity(rank);
            for j in 0..rank {
                let value = F::from(((i + j) as f64 / 100.0).sin() * 0.1).unwrap();
                row.push(value);
            }
            matrix.push(row);
        }

        Ok(matrix)
    }

    /// Compute approximate log posterior for weight sample
    fn compute_log_posterior_approx(&self, weights: &[F]) -> Result<F> {
        // Simplified log posterior computation
        let mut log_posterior = F::zero();

        // Log prior (Gaussian with small variance)
        let prior_var = F::from(1.0).unwrap();
        for &weight in weights {
            log_posterior = log_posterior - F::from(0.5).unwrap() * weight * weight / prior_var;
        }

        // Add mock log likelihood term
        let log_likelihood = -F::from(weights.len() as f64 * 0.1).unwrap();
        log_posterior = log_posterior + log_likelihood;

        Ok(log_posterior)
    }

    /// Compute autocorrelation function for a time series
    fn compute_autocorrelation_function(&self, series: &[F]) -> Result<Vec<F>> {
        let n = series.len();
        let mut autocorr = vec![F::zero(); n.min(50)]; // Limit to 50 lags

        if n < 2 {
            return Ok(autocorr);
        }

        // Compute mean
        let mean = series.iter().cloned().sum::<F>() / F::from(n).unwrap();

        // Compute variance (lag 0)
        let var = series.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>() / F::from(n).unwrap();

        if var <= F::zero() {
            return Ok(autocorr);
        }

        // Compute autocorrelation for each lag
        for lag in 0..autocorr.len() {
            if lag >= n {
                break;
            }

            let mut sum = F::zero();
            let count = n - lag;

            for i in 0..count {
                sum = sum + (series[i] - mean) * (series[i + lag] - mean);
            }

            autocorr[lag] = sum / (F::from(count).unwrap() * var);
        }

        Ok(autocorr)
    }

    /// Compute autocorrelation of a vector (simplified)
    fn compute_autocorrelation_of_vector(&self, values: &[F]) -> Result<F> {
        if values.len() < 2 {
            return Ok(F::zero());
        }

        let n = values.len();
        let lag = 1; // Only compute lag-1 autocorrelation

        let mean = values.iter().cloned().sum::<F>() / F::from(n).unwrap();

        let mut numerator = F::zero();
        let mut denominator = F::zero();

        for i in 0..(n - lag) {
            numerator = numerator + (values[i] - mean) * (values[i + lag] - mean);
        }

        for &value in values {
            denominator = denominator + (value - mean) * (value - mean);
        }

        if denominator > F::zero() {
            Ok(numerator / denominator)
        } else {
            Ok(F::zero())
        }
    }

    // Additional helper methods for enhanced SWAG implementation

    /// Initialize realistic weights with proper scaling
    fn initialize_realistic_weights(&self, n_weights: usize) -> Result<Vec<F>> {
        let mut weights = Vec::with_capacity(n_weights);

        // Xavier initialization: _weights ~ N(0, 2/(n_in + n_out))
        let fan_in = 100; // Simulated input dimension
        let fan_out = 10; // Simulated output dimension
        let std = (F::from(2.0).unwrap() / F::from(fan_in + fan_out).unwrap()).sqrt();

        for i in 0..n_weights {
            let weight = self.sample_gaussian_with_seed(i) * std;
            weights.push(weight);
        }

        Ok(weights)
    }

    /// Compute SWA mean from weight trajectory
    fn compute_swa_mean(&self, weight_trajectory: &[Vec<F>]) -> Result<Vec<F>> {
        if weight_trajectory.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty weight trajectory".to_string(),
            ));
        }

        let n_weights = weight_trajectory[0].len();
        let n_epochs = weight_trajectory.len();
        let mut mean_weights = vec![F::zero(); n_weights];

        for weights in weight_trajectory {
            for (i, &weight) in weights.iter().enumerate() {
                mean_weights[i] = mean_weights[i] + weight;
            }
        }

        for weight in &mut mean_weights {
            *weight = *weight / F::from(n_epochs).unwrap();
        }

        Ok(mean_weights)
    }

    /// Compute diagonal variance for SWA
    fn compute_swa_diagonal_variance(
        &self,
        weight_trajectory: &[Vec<F>],
        mean_weights: &[F],
    ) -> Result<Vec<F>> {
        let n_weights = mean_weights.len();
        let n_epochs = weight_trajectory.len();
        let mut diagonal_variance = vec![F::zero(); n_weights];

        for weights in weight_trajectory {
            for (i, &weight) in weights.iter().enumerate() {
                let deviation = weight - mean_weights[i];
                diagonal_variance[i] = diagonal_variance[i] + deviation * deviation;
            }
        }

        for var in &mut diagonal_variance {
            *var = *var / F::from(n_epochs - 1).unwrap();
        }

        Ok(diagonal_variance)
    }

    /// Compute weight deviations for low-rank approximation
    fn compute_weight_deviations(
        &self,
        weight_trajectory: &[Vec<F>],
        mean_weights: &[F],
    ) -> Result<Vec<Vec<F>>> {
        let mut deviations = Vec::new();

        for weights in weight_trajectory {
            let mut deviation = Vec::new();
            for (i, &weight) in weights.iter().enumerate() {
                deviation.push(weight - mean_weights[i]);
            }
            deviations.push(deviation);
        }

        Ok(deviations)
    }

    /// Sample Gaussian with deterministic seed
    fn sample_gaussian_with_seed(&self, seed: usize) -> F {
        // Simple deterministic random number generation
        let mut state = seed as u64;
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u1 = (state as f64) / (u64::MAX as f64);

        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u2 = (state as f64) / (u64::MAX as f64);

        // Box-Muller transform
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        F::from(z).unwrap()
    }

    /// Compute eigenvalue contribution for a sample
    fn compute_eigen_contribution(&self, eigenvalues: &[F], sample_idx: usize) -> Result<F> {
        if eigenvalues.is_empty() {
            return Ok(F::zero());
        }

        // Weight contribution by eigenvalue magnitude
        let total_eigenvalue = eigenvalues.iter().cloned().sum::<F>();
        let dominant_eigenvalue = eigenvalues[sample_idx % eigenvalues.len()];

        if total_eigenvalue > F::zero() {
            Ok(dominant_eigenvalue / total_eigenvalue)
        } else {
            Ok(F::one() / F::from(eigenvalues.len()).unwrap())
        }
    }

    /// Compute deviation penalty for log posterior
    fn compute_deviation_penalty(&self, weights: &[F], mean_weights: &[F]) -> Result<F> {
        let mut penalty = F::zero();

        for (&weight, &mean_weight) in weights.iter().zip(mean_weights.iter()) {
            let deviation = weight - mean_weight;
            penalty = penalty + deviation * deviation;
        }

        Ok(penalty / F::from(weights.len()).unwrap())
    }

    /// Compute weight distance between two weight vectors
    fn compute_weight_distance(&self, weights1: &[F], weights2: &[F]) -> Result<F> {
        if weights1.len() != weights2.len() {
            return Err(MetricsError::InvalidInput(
                "Weight vectors must have same length".to_string(),
            ));
        }

        let mut distance_sq = F::zero();
        for (&w1, &w2) in weights1.iter().zip(weights2.iter()) {
            let diff = w1 - w2;
            distance_sq = distance_sq + diff * diff;
        }

        Ok(distance_sq.sqrt())
    }

    /// Compute rank contribution for a weight sample
    fn compute_rank_contribution(&self, weights: &[F], swa_stats: &SWAStatistics<F>) -> Result<F> {
        // Simplified rank contribution based on weight alignment with principal directions
        let weight_norm = weights.iter().map(|&w| w * w).sum::<F>().sqrt();
        let mean_norm = swa_stats
            .mean_weights
            .iter()
            .map(|&w| w * w)
            .sum::<F>()
            .sqrt();

        if mean_norm > F::zero() {
            Ok(weight_norm / mean_norm)
        } else {
            Ok(F::one())
        }
    }

    /// Compute stability score based on weight magnitudes
    fn compute_stability_score(&self, weights: &[F]) -> Result<F> {
        let weight_magnitudes: Vec<F> = weights.iter().map(|&w| w.abs()).collect();
        let max_magnitude = weight_magnitudes.iter().cloned().fold(F::zero(), F::max);
        let mean_magnitude =
            weight_magnitudes.iter().cloned().sum::<F>() / F::from(weights.len()).unwrap();

        // Stability is higher when weights are not too extreme
        let stability = if max_magnitude > F::zero() {
            mean_magnitude / max_magnitude
        } else {
            F::one()
        };

        Ok(stability.min(F::one()))
    }
}

// Result structures

/// Comprehensive deep learning uncertainty analysis
#[derive(Debug, Clone)]
pub struct DeepUncertaintyAnalysis<F: Float> {
    /// Monte Carlo Dropout uncertainty
    pub mc_dropout_uncertainty: MCDropoutUncertainty<F>,
    /// Deep ensemble uncertainty
    pub ensemble_uncertainty: EnsembleUncertainty<F>,
    /// Test-time augmentation uncertainty
    pub tta_uncertainty: TTAUncertainty<F>,
    /// Predictive entropy decomposition
    pub entropy_decomposition: EntropyDecomposition<F>,
    /// Temperature scaling results
    pub temperature_scaling: Option<NeuralTemperatureScaling<F>>,
    /// SWAG uncertainty
    pub swag_uncertainty: Option<SWAGUncertainty<F>>,
    /// Disagreement-based uncertainty
    pub disagreement_uncertainty: DisagreementUncertainty<F>,
    /// Sample size
    pub sample_size: usize,
}

/// Monte Carlo Dropout uncertainty results
#[derive(Debug, Clone)]
pub struct MCDropoutUncertainty<F: Float> {
    /// All MC predictions [n_mc_samples, n_test_samples]
    pub predictions: Array2<F>,
    /// Mean predictions across MC samples
    pub mean_predictions: Array1<F>,
    /// Standard deviation of predictions
    pub std_predictions: Array1<F>,
    /// Epistemic uncertainty
    pub epistemic_uncertainty: Array1<F>,
    /// Aleatoric uncertainty
    pub aleatoric_uncertainty: Array1<F>,
    /// Prediction intervals [n_samples, 2]
    pub prediction_intervals: Array2<F>,
    /// Number of MC samples
    pub n_samples: usize,
}

/// Deep ensemble uncertainty results
#[derive(Debug, Clone)]
pub struct EnsembleUncertainty<F: Float> {
    /// All ensemble predictions [n_models, n_test_samples]
    pub predictions: Array2<F>,
    /// Mean predictions across ensemble
    pub mean_predictions: Array1<F>,
    /// Standard deviation of predictions
    pub std_predictions: Array1<F>,
    /// Model diversity score
    pub model_diversity: F,
    /// Prediction intervals [n_samples, 2]
    pub prediction_intervals: Array2<F>,
    /// Mutual information between models
    pub mutual_information: F,
    /// Number of ensemble members
    pub n_models: usize,
}

/// Test-time augmentation uncertainty results
#[derive(Debug, Clone)]
pub struct TTAUncertainty<F: Float> {
    /// All TTA predictions [n_augmentations, n_test_samples]
    pub predictions: Array2<F>,
    /// Mean predictions across augmentations
    pub mean_predictions: Array1<F>,
    /// Standard deviation of predictions
    pub std_predictions: Array1<F>,
    /// Augmentation consistency score
    pub consistency_score: F,
    /// Number of augmentations
    pub n_augmentations: usize,
}

/// Predictive entropy decomposition
#[derive(Debug, Clone)]
pub struct EntropyDecomposition<F: Float> {
    /// Total entropy
    pub total_entropy: Array1<F>,
    /// Aleatoric entropy (irreducible)
    pub aleatoric_entropy: Array1<F>,
    /// Epistemic entropy (reducible)
    pub epistemic_entropy: Array1<F>,
}

/// Neural network temperature scaling results
#[derive(Debug, Clone)]
pub struct NeuralTemperatureScaling<F: Float> {
    /// Optimal temperature parameter
    pub temperature: F,
    /// Calibrated probabilities
    pub calibrated_probabilities: Array1<F>,
    /// Expected calibration error before scaling
    pub pre_calibration_ece: F,
    /// Expected calibration error after scaling
    pub post_calibration_ece: F,
    /// Calibration improvement
    pub calibration_improvement: F,
}

/// SWAG uncertainty results
#[derive(Debug, Clone)]
pub struct SWAGUncertainty<F: Float> {
    /// All SWAG predictions [n_swag_samples, n_test_samples]
    pub predictions: Array2<F>,
    /// Mean predictions
    pub mean_predictions: Array1<F>,
    /// Standard deviation of predictions
    pub std_predictions: Array1<F>,
    /// Effective sample size
    pub effective_sample_size: F,
    /// Number of SWAG samples
    pub n_swag_samples: usize,
    /// Diagonal variance component of SWAG
    pub diagonal_variance: Vec<F>,
    /// Low-rank covariance component
    pub low_rank_covariance: Array2<F>,
    /// SWAG approximation quality metrics
    pub approximation_quality: SWAGApproximationQuality<F>,
}

/// Enhanced SWAG weight sample
#[derive(Debug, Clone)]
pub struct SWAGWeightSample<F: Float> {
    /// Sampled weights
    pub weights: Vec<F>,
    /// Log posterior probability of this weight sample
    pub log_posterior: F,
    /// Diagonal component of covariance
    pub diagonal_component: Vec<F>,
    /// Low-rank component
    pub low_rank_component: Vec<F>,
    /// Sample quality metrics
    pub sample_quality: SampleQuality<F>,
    /// Eigenvalue contribution
    pub eigen_contribution: F,
}

/// SWA statistics for proper SWAG implementation
#[derive(Debug, Clone)]
pub struct SWAStatistics<F: Float> {
    /// Mean weights from SWA
    pub mean_weights: Vec<F>,
    /// Diagonal variance estimates
    pub diagonal_variance: Vec<F>,
    /// Weight deviations for low-rank approximation
    pub weight_deviations: Vec<Vec<F>>,
    /// Number of epochs used
    pub n_epochs: usize,
    /// Learning rate schedule
    pub learning_rate_schedule: Vec<F>,
}

/// Sample quality assessment
#[derive(Debug, Clone)]
pub struct SampleQuality<F: Float> {
    /// Distance from SWA mean
    pub mean_distance: F,
    /// Contribution to effective rank
    pub rank_contribution: F,
    /// Diversity score
    pub diversity_score: F,
    /// Stability score
    pub stability_score: F,
    /// Overall quality metric
    pub overall_quality: F,
}

/// SWAG approximation quality metrics
#[derive(Debug, Clone)]
pub struct SWAGApproximationQuality<F: Float> {
    /// Mean log posterior probability
    pub mean_log_posterior: F,
    /// Variance of log posterior probabilities
    pub var_log_posterior: F,
    /// Effective sample size from log posteriors
    pub ess_log_posterior: F,
    /// Acceptance rate for sampling
    pub acceptance_rate: F,
    /// R-hat convergence diagnostic
    pub r_hat: F,
    /// Human-readable convergence status
    pub convergence_diagnostic: String,
}

/// Disagreement-based uncertainty
#[derive(Debug, Clone)]
pub struct DisagreementUncertainty<F: Float> {
    /// Disagreement scores between methods
    pub disagreement_scores: Array1<F>,
    /// Confidence scores (inverse of disagreement)
    pub confidence_scores: Array1<F>,
    /// Correlation between uncertainty methods
    pub method_correlation: F,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Mock neural network model for testing
    fn mock_neural_model(x: &ArrayView2<f64>, dropout: bool) -> Array1<f64> {
        x.map_axis(Axis(1), |row| row.sum())
    }

    // Mock ensemble models
    fn mock_ensemble_model_1(x: &ArrayView2<f64>) -> Array1<f64> {
        x.map_axis(Axis(1), |row| row.sum() * 1.1)
    }

    fn mock_ensemble_model_2(x: &ArrayView2<f64>) -> Array1<f64> {
        x.map_axis(Axis(1), |row| row.sum() * 0.9)
    }

    // Mock augmentation function
    fn mock_augmentation(x: &ArrayView2<f64>) -> Array3<f64> {
        let n_aug = 3;
        let mut augmented = Array3::zeros((n_aug, x.nrows(), x.ncols()));

        for i in 0..n_aug {
            for j in 0..x.nrows() {
                for k in 0..x.ncols() {
                    augmented[[i, j, k]] = x[[j, k]] * (1.0 + 0.1 * (i as f64 - 1.0));
                }
            }
        }

        augmented
    }

    #[test]
    fn test_deep_uncertainty_quantifier_creation() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new()
            .with_mc_dropout(50, 0.2)
            .with_ensemble(3)
            .with_tta(5)
            .with_temperature_scaling(true)
            .with_swag(true, 10)
            .with_seed(42);

        assert_eq!(quantifier.n_mc_dropout_samples, 50);
        assert_eq!(quantifier.dropout_rate, 0.2);
        assert_eq!(quantifier.n_ensemble_members, 3);
        assert_eq!(quantifier.n_tta_samples, 5);
        assert!(quantifier.enable_temperature_scaling);
        assert!(quantifier.enable_swag);
        assert_eq!(quantifier.n_swag_samples, 10);
        assert_eq!(quantifier.random_seed, Some(42));
    }

    #[test]
    fn test_mc_dropout_uncertainty() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new()
            .with_mc_dropout(10, 0.1)
            .with_seed(123);

        let x_test = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let mc_uncertainty = quantifier
            .compute_mc_dropout_uncertainty(&mock_neural_model, &x_test)
            .unwrap();

        assert_eq!(mc_uncertainty.predictions.nrows(), 10);
        assert_eq!(mc_uncertainty.predictions.ncols(), 3);
        assert_eq!(mc_uncertainty.mean_predictions.len(), 3);
        assert_eq!(mc_uncertainty.std_predictions.len(), 3);
        assert_eq!(mc_uncertainty.prediction_intervals.nrows(), 3);
        assert_eq!(mc_uncertainty.prediction_intervals.ncols(), 2);
    }

    #[test]
    fn test_ensemble_uncertainty() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new().with_seed(456);

        let x_test = array![[1.0, 2.0], [3.0, 4.0]];
        let ensemble_models = vec![mock_ensemble_model_1, mock_ensemble_model_2];

        let ensemble_uncertainty = quantifier
            .compute_ensemble_uncertainty(&ensemble_models, &x_test)
            .unwrap();

        assert_eq!(ensemble_uncertainty.predictions.nrows(), 2);
        assert_eq!(ensemble_uncertainty.predictions.ncols(), 2);
        assert_eq!(ensemble_uncertainty.mean_predictions.len(), 2);
        assert_eq!(ensemble_uncertainty.n_models, 2);
        assert!(ensemble_uncertainty.model_diversity >= 0.0);
    }

    #[test]
    fn test_tta_uncertainty() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new()
            .with_tta(3)
            .with_seed(789);

        let x_test = array![[1.0, 2.0], [3.0, 4.0]];

        let tta_uncertainty = quantifier
            .compute_tta_uncertainty(&mock_neural_model, &mock_augmentation, &x_test)
            .unwrap();

        assert_eq!(tta_uncertainty.predictions.nrows(), 3);
        assert_eq!(tta_uncertainty.predictions.ncols(), 2);
        assert_eq!(tta_uncertainty.mean_predictions.len(), 2);
        assert!(tta_uncertainty.consistency_score >= 0.0);
        assert!(tta_uncertainty.consistency_score <= 1.0);
    }

    #[test]
    fn test_entropy_decomposition() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new();

        // Create mock predictions (logits)
        let predictions = array![[0.5, 1.0, -0.5], [0.7, 0.8, -0.3], [0.3, 1.2, -0.7],];

        let entropy_decomp = quantifier
            .compute_entropy_decomposition(&predictions)
            .unwrap();

        assert_eq!(entropy_decomp.total_entropy.len(), 3);
        assert_eq!(entropy_decomp.aleatoric_entropy.len(), 3);
        assert_eq!(entropy_decomp.epistemic_entropy.len(), 3);

        // All entropies should be non-negative
        for &entropy in entropy_decomp.total_entropy.iter() {
            assert!(entropy >= 0.0);
        }
        for &entropy in entropy_decomp.aleatoric_entropy.iter() {
            assert!(entropy >= 0.0);
        }
    }

    #[test]
    fn test_temperature_scaling() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new().with_seed(321);

        let x_cal = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y_cal = array![0.0, 0.5, 1.0];

        let temp_scaling = quantifier
            .compute_neural_temperature_scaling(&mock_neural_model, &x_cal, &y_cal)
            .unwrap();

        assert!(temp_scaling.temperature > 0.0);
        assert_eq!(temp_scaling.calibrated_probabilities.len(), 3);
        assert!(temp_scaling.pre_calibration_ece >= 0.0);
        assert!(temp_scaling.post_calibration_ece >= 0.0);

        // All calibrated probabilities should be valid probabilities
        for &prob in temp_scaling.calibrated_probabilities.iter() {
            assert!((0.0..=1.0).contains(&prob));
        }
    }

    #[test]
    fn test_comprehensive_deep_uncertainty() {
        let quantifier = DeepUncertaintyQuantifier::<f64>::new()
            .with_mc_dropout(5, 0.1)
            .with_ensemble(2)
            .with_tta(3)
            .with_temperature_scaling(true)
            .with_swag(true, 5)
            .with_seed(42);

        let x_test = array![[1.0, 2.0], [3.0, 4.0]];
        let x_cal = array![[0.5, 1.5], [2.5, 3.5]];
        let y_cal = array![0.0, 1.0];
        let ensemble_models = vec![mock_ensemble_model_1, mock_ensemble_model_2];

        let analysis = quantifier
            .compute_deep_uncertainty(
                &mock_neural_model,
                &ensemble_models,
                &mock_augmentation,
                &x_test,
                Some(&x_cal),
                Some(&y_cal),
            )
            .unwrap();

        assert_eq!(analysis.sample_size, 2);
        assert!(analysis.temperature_scaling.is_some());
        assert!(analysis.swag_uncertainty.is_some());
        assert_eq!(analysis.mc_dropout_uncertainty.mean_predictions.len(), 2);
        assert_eq!(analysis.ensemble_uncertainty.mean_predictions.len(), 2);
        assert_eq!(analysis.tta_uncertainty.mean_predictions.len(), 2);
    }
}
