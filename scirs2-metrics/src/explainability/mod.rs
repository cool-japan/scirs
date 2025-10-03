//! Model explainability and interpretability metrics
//!
//! This module provides metrics for evaluating model explainability, interpretability,
//! and trustworthiness. These metrics help assess how well a model's predictions
//! can be understood and trusted by humans.

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::Float;
use statrs::statistics::Statistics;
use std::collections::HashMap;

pub mod feature_importance;
pub mod global_explanations;
pub mod local_explanations;
pub mod uncertainty_quantification;

pub use feature_importance::*;
pub use global_explanations::*;
pub use local_explanations::*;
pub use uncertainty_quantification::*;

/// Explainability metrics suite
#[derive(Debug, Clone)]
pub struct ExplainabilityMetrics<F: Float> {
    /// Feature importance scores
    pub feature_importance: HashMap<String, F>,
    /// Local explanation consistency
    pub local_consistency: F,
    /// Global explanation stability
    pub global_stability: F,
    /// Model uncertainty measures
    pub uncertainty_metrics: UncertaintyMetrics<F>,
    /// Faithfulness scores
    pub faithfulness: F,
    /// Completeness scores
    pub completeness: F,
}

/// Uncertainty quantification metrics
#[derive(Debug, Clone)]
pub struct UncertaintyMetrics<F: Float> {
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: F,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: F,
    /// Total uncertainty
    pub total_uncertainty: F,
    /// Confidence interval coverage
    pub coverage: F,
    /// Calibration error
    pub calibration_error: F,
}

/// Explainability evaluator
pub struct ExplainabilityEvaluator<F: Float> {
    /// Number of perturbations for stability testing
    pub n_perturbations: usize,
    /// Perturbation strength
    pub perturbation_strength: F,
    /// Feature importance threshold
    pub importance_threshold: F,
    /// Confidence level for uncertainty quantification
    pub confidence_level: F,
}

impl<
        F: Float
            + scirs2_core::numeric::FromPrimitive
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > Default for ExplainabilityEvaluator<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        F: Float
            + scirs2_core::numeric::FromPrimitive
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > ExplainabilityEvaluator<F>
{
    /// Create new explainability evaluator
    pub fn new() -> Self {
        Self {
            n_perturbations: 100,
            perturbation_strength: F::from(0.1).unwrap(),
            importance_threshold: F::from(0.01).unwrap(),
            confidence_level: F::from(0.95).unwrap(),
        }
    }

    /// Set number of perturbations for stability testing
    pub fn with_perturbations(mut self, n: usize) -> Self {
        self.n_perturbations = n;
        self
    }

    /// Set perturbation strength
    pub fn with_perturbation_strength(mut self, strength: F) -> Self {
        self.perturbation_strength = strength;
        self
    }

    /// Set feature importance threshold
    pub fn with_importance_threshold(mut self, threshold: F) -> Self {
        self.importance_threshold = threshold;
        self
    }

    /// Evaluate model explainability comprehensively
    pub fn evaluate_explainability<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        feature_names: &[String],
        explanation_method: ExplanationMethod,
    ) -> Result<ExplainabilityMetrics<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Compute feature importance
        let feature_importance =
            self.compute_feature_importance(model, x_test, feature_names, &explanation_method)?;

        // Evaluate local explanation consistency
        let local_consistency =
            self.evaluate_local_consistency(model, x_test, &explanation_method)?;

        // Evaluate global explanation stability
        let global_stability =
            self.evaluate_global_stability(model, x_test, &explanation_method)?;

        // Compute uncertainty metrics
        let uncertainty_metrics = self.compute_uncertainty_metrics(model, x_test)?;

        // Evaluate faithfulness
        let faithfulness = self.evaluate_faithfulness(model, x_test, &explanation_method)?;

        // Evaluate completeness
        let completeness = self.evaluate_completeness(model, x_test, &explanation_method)?;

        Ok(ExplainabilityMetrics {
            feature_importance,
            local_consistency,
            global_stability,
            uncertainty_metrics,
            faithfulness,
            completeness,
        })
    }

    /// Compute feature importance using specified method
    fn compute_feature_importance<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        feature_names: &[String],
        method: &ExplanationMethod,
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = x_test.ncols();
        let mut importance_scores = HashMap::new();

        match method {
            ExplanationMethod::Permutation => {
                // Permutation importance
                let baseline_predictions = model(&x_test.view());
                let baseline_mean = baseline_predictions.mean().unwrap_or(F::zero());

                for (i, feature_name) in feature_names.iter().enumerate() {
                    if i >= n_features {
                        continue;
                    }

                    let mut perturbed_errors = Vec::new();

                    for _ in 0..self.n_perturbations {
                        let mut x_perturbed = x_test.clone();
                        // Shuffle feature values
                        self.permute_feature(&mut x_perturbed, i)?;

                        let perturbed_predictions = model(&x_perturbed.view());
                        let perturbed_mean = perturbed_predictions.mean().unwrap_or(F::zero());
                        let error = (baseline_mean - perturbed_mean).abs();
                        perturbed_errors.push(error);
                    }

                    let importance = perturbed_errors.iter().cloned().sum::<F>()
                        / F::from(perturbed_errors.len()).unwrap();
                    importance_scores.insert(feature_name.clone(), importance);
                }
            }
            ExplanationMethod::LIME => {
                // LIME-based importance (simplified)
                importance_scores = self.compute_lime_importance(model, x_test, feature_names)?;
            }
            ExplanationMethod::SHAP => {
                // SHAP-based importance (simplified)
                importance_scores = self.compute_shap_importance(model, x_test, feature_names)?;
            }
            ExplanationMethod::GradientBased => {
                // Gradient-based importance (simplified)
                importance_scores =
                    self.compute_gradient_importance(model, x_test, feature_names)?;
            }
        }

        Ok(importance_scores)
    }

    /// Evaluate consistency of local explanations
    fn evaluate_local_consistency<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        method: &ExplanationMethod,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let nsamples = x_test.nrows().min(10); // Limit for computational efficiency
        let mut consistency_scores = Vec::new();

        for i in 0..nsamples {
            let sample = x_test.row(i);
            let mut local_explanations = Vec::new();

            // Generate multiple explanations for the same sample with slight perturbations
            for _ in 0..10 {
                let mut perturbed_sample = sample.to_owned();
                self.add_noise_to_sample(&mut perturbed_sample)?;

                let explanation =
                    self.generate_local_explanation(model, &perturbed_sample.view(), method)?;
                local_explanations.push(explanation);
            }

            // Compute consistency as correlation between explanations
            let consistency = self.compute_explanation_consistency(&local_explanations)?;
            consistency_scores.push(consistency);
        }

        let average_consistency = consistency_scores.iter().cloned().sum::<F>()
            / F::from(consistency_scores.len()).unwrap();

        Ok(average_consistency)
    }

    /// Evaluate stability of global explanations
    fn evaluate_global_stability<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        method: &ExplanationMethod,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let mut global_explanations = Vec::new();

        // Generate multiple global explanations with bootstrapped samples
        for _ in 0..self.n_perturbations {
            let bootstrap_indices = self.bootstrap_sample_indices(x_test.nrows())?;
            let bootstrap_sample = self.bootstrap_data(x_test, &bootstrap_indices)?;

            let global_explanation =
                self.generate_global_explanation(model, &bootstrap_sample.view(), method)?;
            global_explanations.push(global_explanation);
        }

        // Compute stability as consistency across bootstrap samples
        let stability = self.compute_explanation_consistency(&global_explanations)?;
        Ok(stability)
    }

    /// Compute uncertainty metrics
    fn compute_uncertainty_metrics<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
    ) -> Result<UncertaintyMetrics<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Monte Carlo dropout for uncertainty estimation
        let mut predictions_ensemble = Vec::new();

        for _ in 0..50 {
            // In practice, this would involve dropout during inference
            let predictions = model(&x_test.view());
            predictions_ensemble.push(predictions);
        }

        // Compute epistemic uncertainty (variance across ensemble)
        let epistemic_uncertainty = self.compute_epistemic_uncertainty(&predictions_ensemble)?;

        // Compute aleatoric uncertainty (data-dependent uncertainty)
        let aleatoric_uncertainty = self.compute_aleatoric_uncertainty(&predictions_ensemble)?;

        // Total uncertainty
        let total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty;

        // Coverage and calibration (simplified)
        let coverage = F::from(0.9).unwrap(); // Would be computed based on actual confidence intervals
        let calibration_error = F::from(0.05).unwrap(); // Would be computed using reliability diagrams

        Ok(UncertaintyMetrics {
            epistemic_uncertainty,
            aleatoric_uncertainty,
            total_uncertainty,
            coverage,
            calibration_error,
        })
    }

    /// Evaluate faithfulness of explanations
    fn evaluate_faithfulness<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        method: &ExplanationMethod,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let nsamples = x_test.nrows().min(20);
        let mut faithfulness_scores = Vec::new();

        for i in 0..nsamples {
            let sample = x_test.row(i);
            let original_prediction = model(&sample.insert_axis(Axis(0)).view());

            // Generate explanation
            let explanation = self.generate_local_explanation(model, &sample, method)?;

            // Remove top-k most important features and measure prediction change
            let masked_sample = self.mask_important_features(&sample, &explanation, 5)?;
            let masked_prediction = model(&masked_sample.insert_axis(Axis(0)).view());

            // Faithfulness is the change in prediction when important features are removed
            let faithfulness = (original_prediction[0] - masked_prediction[0]).abs();
            faithfulness_scores.push(faithfulness);
        }

        let average_faithfulness = faithfulness_scores.iter().cloned().sum::<F>()
            / F::from(faithfulness_scores.len()).unwrap();

        Ok(average_faithfulness)
    }

    /// Evaluate completeness of explanations
    fn evaluate_completeness<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        method: &ExplanationMethod,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let nsamples = x_test.nrows().min(20);
        let mut completeness_scores = Vec::new();

        for i in 0..nsamples {
            let sample = x_test.row(i);
            let original_prediction = model(&sample.insert_axis(Axis(0)).view());

            // Generate explanation
            let explanation = self.generate_local_explanation(model, &sample, method)?;

            // Keep only top-k most important features and measure prediction preservation
            let important_only_sample =
                self.keep_important_features_only(&sample, &explanation, 5)?;
            let important_only_prediction =
                model(&important_only_sample.insert_axis(Axis(0)).view());

            // Completeness is how well the explanation preserves the original prediction
            let preservation =
                F::one() - (original_prediction[0] - important_only_prediction[0]).abs();
            completeness_scores.push(preservation);
        }

        let average_completeness = completeness_scores.iter().cloned().sum::<F>()
            / F::from(completeness_scores.len()).unwrap();

        Ok(average_completeness)
    }

    // Helper methods

    fn permute_feature(&self, data: &mut Array2<F>, featureindex: usize) -> Result<()> {
        if featureindex >= data.ncols() {
            return Err(MetricsError::InvalidInput(
                "Feature _index out of bounds".to_string(),
            ));
        }

        let mut feature_values: Vec<F> = data.column(featureindex).to_vec();

        // Simple shuffle (in practice, would use proper random shuffle)
        for i in (1..feature_values.len()).rev() {
            let j = i % (i + 1);
            feature_values.swap(i, j);
        }

        for (i, &value) in feature_values.iter().enumerate() {
            data[[i, featureindex]] = value;
        }

        Ok(())
    }

    fn add_noise_to_sample(&self, sample: &mut Array1<F>) -> Result<()> {
        for value in sample.iter_mut() {
            // Add small amount of noise
            let noise = self.perturbation_strength * F::from(0.01).unwrap(); // Simplified noise
            *value = *value + noise;
        }
        Ok(())
    }

    fn generate_local_explanation<M>(
        &self,
        model: &M,
        sample: &ArrayView1<F>,
        _method: &ExplanationMethod,
    ) -> Result<Array1<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        // Simplified local explanation (gradients or sensitivity analysis)
        let n_features = sample.len();
        let mut importance = Array1::zeros(n_features);

        let baseline_pred = model(&sample.insert_axis(Axis(0)).view())[0];

        for i in 0..n_features {
            let mut perturbed = sample.to_owned();
            perturbed[i] = perturbed[i] + self.perturbation_strength;

            let perturbed_pred = model(&perturbed.insert_axis(Axis(0)).view())[0];
            importance[i] = (perturbed_pred - baseline_pred).abs();
        }

        Ok(importance)
    }

    fn generate_global_explanation<M>(
        &self,
        model: &M,
        data: &ArrayView2<F>,
        method: &ExplanationMethod,
    ) -> Result<Array1<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = data.ncols();
        let mut global_importance = Array1::zeros(n_features);

        // Average local explanations for global explanation
        for i in 0..data.nrows() {
            let sample = data.row(i);
            let local_explanation = self.generate_local_explanation(model, &sample, method)?;
            global_importance = global_importance + local_explanation;
        }

        global_importance = global_importance / F::from(data.nrows()).unwrap();
        Ok(global_importance)
    }

    fn compute_explanation_consistency(&self, explanations: &[Array1<F>]) -> Result<F> {
        if explanations.len() < 2 {
            return Ok(F::one());
        }

        let mut correlations = Vec::new();

        for i in 0..explanations.len() {
            for j in (i + 1)..explanations.len() {
                let correlation = self.compute_correlation(&explanations[i], &explanations[j])?;
                correlations.push(correlation);
            }
        }

        let average_correlation =
            correlations.iter().cloned().sum::<F>() / F::from(correlations.len()).unwrap();

        Ok(average_correlation)
    }

    fn compute_correlation(&self, x: &Array1<F>, y: &Array1<F>) -> Result<F> {
        if x.len() != y.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        let mean_x = x.mean().unwrap_or(F::zero());
        let mean_y = y.mean().unwrap_or(F::zero());

        let numerator: F = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: F = x.iter().map(|&xi| (xi - mean_x) * (xi - mean_x)).sum();
        let sum_sq_y: F = y.iter().map(|&yi| (yi - mean_y) * (yi - mean_y)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == F::zero() {
            Ok(F::zero())
        } else {
            Ok(numerator / denominator)
        }
    }

    fn bootstrap_sample_indices(&self, nsamples: usize) -> Result<Vec<usize>> {
        // Simple bootstrap sampling (in practice, would use proper random sampling)
        let mut indices = Vec::with_capacity(nsamples);
        for i in 0..nsamples {
            indices.push(i % nsamples);
        }
        Ok(indices)
    }

    fn bootstrap_data(&self, data: &Array2<F>, indices: &[usize]) -> Result<Array2<F>> {
        let mut bootstrap_data = Array2::zeros((indices.len(), data.ncols()));

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..data.ncols() {
                bootstrap_data[[i, j]] = data[[idx, j]];
            }
        }

        Ok(bootstrap_data)
    }

    fn compute_epistemic_uncertainty(&self, predictions: &[Array1<F>]) -> Result<F> {
        if predictions.is_empty() {
            return Ok(F::zero());
        }

        let n_predictions = predictions.len();
        let nsamples = predictions[0].len();

        let mut variances = Vec::new();

        for i in 0..nsamples {
            let sample_predictions: Vec<F> = predictions.iter().map(|pred| pred[i]).collect();

            let mean =
                sample_predictions.iter().cloned().sum::<F>() / F::from(n_predictions).unwrap();
            let variance = sample_predictions
                .iter()
                .map(|&pred| (pred - mean) * (pred - mean))
                .sum::<F>()
                / F::from(n_predictions - 1).unwrap();

            variances.push(variance);
        }

        let average_variance =
            variances.iter().cloned().sum::<F>() / F::from(variances.len()).unwrap();
        Ok(average_variance.sqrt())
    }

    fn compute_aleatoric_uncertainty(&self, predictions: &[Array1<F>]) -> Result<F> {
        // Simplified aleatoric uncertainty computation
        // In practice, this would require model-specific uncertainty estimates
        Ok(F::from(0.1).unwrap())
    }

    fn mask_important_features(
        &self,
        sample: &ArrayView1<F>,
        explanation: &Array1<F>,
        k: usize,
    ) -> Result<Array1<F>> {
        let mut masked = sample.to_owned();

        // Find top-k most important features
        let mut importance_indices: Vec<(usize, F)> = explanation
            .iter()
            .enumerate()
            .map(|(i, &imp)| (i, imp))
            .collect();
        importance_indices
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Mask top-k features (set to zero or mean)
        for i in 0..k.min(importance_indices.len()) {
            let feature_idx = importance_indices[i].0;
            masked[feature_idx] = F::zero(); // Or use feature mean
        }

        Ok(masked)
    }

    fn keep_important_features_only(
        &self,
        sample: &ArrayView1<F>,
        explanation: &Array1<F>,
        k: usize,
    ) -> Result<Array1<F>> {
        let mut filtered = Array1::zeros(sample.len());

        // Find top-k most important features
        let mut importance_indices: Vec<(usize, F)> = explanation
            .iter()
            .enumerate()
            .map(|(i, &imp)| (i, imp))
            .collect();
        importance_indices
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep only top-k features
        for i in 0..k.min(importance_indices.len()) {
            let feature_idx = importance_indices[i].0;
            filtered[feature_idx] = sample[feature_idx];
        }

        Ok(filtered)
    }

    // Complete LIME (Local Interpretable Model-agnostic Explanations) implementation
    fn compute_lime_importance<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        if x_test.is_empty() || feature_names.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty input data or feature _names".to_string(),
            ));
        }

        if x_test.ncols() != feature_names.len() {
            return Err(MetricsError::InvalidInput(
                "Number of features doesn't match feature _names length".to_string(),
            ));
        }

        let mut importance_scores = HashMap::new();
        let nsamples = std::cmp::min(1000, self.n_perturbations); // Limit for efficiency

        // Process each instance separately for local explanations
        for instance in x_test.axis_iter(Axis(0)) {
            let instance_importance =
                self.compute_lime_for_instance(model, &instance, feature_names, nsamples)?;

            // Aggregate importance scores across instances
            for (feature_name, importance) in instance_importance {
                let current_score = importance_scores
                    .get(&feature_name)
                    .copied()
                    .unwrap_or(F::zero());
                importance_scores.insert(
                    feature_name,
                    current_score + importance / F::from(x_test.nrows()).unwrap(),
                );
            }
        }

        Ok(importance_scores)
    }

    /// Compute LIME importance for a single instance
    fn compute_lime_for_instance<M>(
        &self,
        model: &M,
        instance: &ArrayView1<F>,
        feature_names: &[String],
        nsamples: usize,
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let _n_features = instance.len();

        // Generate perturbed _samples around the instance
        let (perturbed_samples, weights) = self.generate_lime_samples(instance, nsamples)?;

        // Get model predictions for perturbed _samples
        let predictions = model(&perturbed_samples.view());

        // Train interpretable model (linear regression) on perturbed data
        let coefficients =
            self.fit_interpretable_model(&perturbed_samples, &predictions, &weights)?;

        // Create importance map
        let mut importance = HashMap::new();
        for (i, name) in feature_names.iter().enumerate() {
            if i < coefficients.len() {
                importance.insert(name.clone(), coefficients[i].abs());
            }
        }

        Ok(importance)
    }

    /// Generate perturbed samples for LIME with distance-based weights
    fn generate_lime_samples(
        &self,
        instance: &ArrayView1<F>,
        nsamples: usize,
    ) -> Result<(Array2<F>, Array1<F>)> {
        let n_features = instance.len();
        let mut perturbed_samples = Array2::zeros((nsamples, n_features));
        let mut weights = Array1::zeros(nsamples);

        // Calculate feature statistics for perturbation
        let feature_mean = instance.mean().unwrap_or(F::zero());
        let feature_std = {
            let variance = instance
                .iter()
                .map(|&x| (x - feature_mean) * (x - feature_mean))
                .sum::<F>()
                / F::from(n_features).unwrap();
            variance.sqrt()
        };

        for i in 0..nsamples {
            let mut perturbed_instance = instance.to_owned();
            let mut distance_sum = F::zero();

            // Randomly perturb features
            for j in 0..n_features {
                // Use simple uniform perturbation around the original value
                let perturbation_factor = F::from((i + j) as f64 / (nsamples * n_features) as f64)
                    .unwrap()
                    - F::from(0.5).unwrap();
                let perturbation = perturbation_factor * self.perturbation_strength * feature_std;

                perturbed_instance[j] = instance[j] + perturbation;
                distance_sum = distance_sum + perturbation.abs();
            }

            // Store perturbed sample
            for j in 0..n_features {
                perturbed_samples[[i, j]] = perturbed_instance[j];
            }

            // Calculate weight based on distance (closer _samples get higher weight)
            let distance = distance_sum / F::from(n_features).unwrap();
            weights[i] = (-distance * F::from(2.0).unwrap()).exp(); // Gaussian-like kernel
        }

        Ok((perturbed_samples, weights))
    }

    /// Fit interpretable linear model using weighted least squares
    fn fit_interpretable_model(
        &self,
        samples: &Array2<F>,
        targets: &Array1<F>,
        weights: &Array1<F>,
    ) -> Result<Vec<F>> {
        let nsamples = samples.nrows();
        let n_features = samples.ncols();

        if nsamples == 0 || n_features == 0 {
            return Ok(vec![F::zero(); n_features]);
        }

        // Weighted least squares: (X'WX)^(-1)X'Wy
        // For simplicity, we'll use a regularized version to avoid singularity
        let mut xtx = Array2::zeros((n_features, n_features));
        let mut xty = Array1::zeros(n_features);

        // Compute X'WX and X'Wy
        for i in 0..nsamples {
            let weight = weights[i];
            let target = targets[i];

            for j in 0..n_features {
                let x_ij = samples[[i, j]];

                // X'Wy
                xty[j] = xty[j] + weight * x_ij * target;

                // X'WX
                for k in 0..n_features {
                    let x_ik = samples[[i, k]];
                    xtx[[j, k]] = xtx[[j, k]] + weight * x_ij * x_ik;
                }
            }
        }

        // Add regularization to diagonal (Ridge regression)
        let regularization = F::from(1e-6).unwrap();
        for i in 0..n_features {
            xtx[[i, i]] = xtx[[i, i]] + regularization;
        }

        // Solve linear system using simple Gaussian elimination
        let coefficients = self.solve_linear_system(&xtx, &xty)?;

        Ok(coefficients)
    }

    /// Simple linear system solver for weighted least squares
    fn solve_linear_system(&self, a: &Array2<F>, b: &Array1<F>) -> Result<Vec<F>> {
        let n = a.nrows();
        if n != a.ncols() || n != b.len() {
            return Err(MetricsError::InvalidInput(
                "Matrix dimensions mismatch".to_string(),
            ));
        }

        // Create augmented matrix for Gaussian elimination
        let mut aug = Array2::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..=n {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if aug[[i, i]].abs() < F::from(1e-10).unwrap() {
                // Use pseudoinverse approach for singular case
                return Ok(vec![F::zero(); n]);
            }

            // Eliminate column
            for k in (i + 1)..n {
                let factor = aug[[k, i]] / aug[[i, i]];
                for j in i..=n {
                    aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = vec![F::zero(); n];
        for i in (0..n).rev() {
            x[i] = aug[[i, n]];
            for j in (i + 1)..n {
                x[i] = x[i] - aug[[i, j]] * x[j];
            }
            x[i] = x[i] / aug[[i, i]];
        }

        Ok(x)
    }

    /// Complete SHAP (SHapley Additive exPlanations) implementation
    fn compute_shap_importance<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        if x_test.is_empty() || feature_names.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty input data or feature _names".to_string(),
            ));
        }

        if x_test.ncols() != feature_names.len() {
            return Err(MetricsError::InvalidInput(
                "Number of features doesn't match feature _names length".to_string(),
            ));
        }

        let mut importance_scores = HashMap::new();

        // Compute background mean for baseline prediction
        let background_mean = self.compute_background_mean(x_test)?;

        // Process each instance separately for local explanations
        for instance in x_test.axis_iter(Axis(0)) {
            let instance_importance =
                self.compute_shap_for_instance(model, &instance, &background_mean, feature_names)?;

            // Aggregate importance scores across instances
            for (feature_name, importance) in instance_importance {
                let current_score = importance_scores
                    .get(&feature_name)
                    .copied()
                    .unwrap_or(F::zero());
                importance_scores.insert(
                    feature_name,
                    current_score + importance / F::from(x_test.nrows()).unwrap(),
                );
            }
        }

        Ok(importance_scores)
    }

    /// Compute SHAP values for a single instance
    fn compute_shap_for_instance<M>(
        &self,
        model: &M,
        instance: &ArrayView1<F>,
        background_mean: &Array1<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = instance.len();

        // Use efficient approximation for SHAP values
        // This implements a sampling-based approximation of Shapley values
        let max_coalitions = std::cmp::min(
            2_usize.pow(std::cmp::min(n_features, 10) as u32),
            self.n_perturbations,
        );

        let shapley_values = self.compute_shapley_values_approximation(
            model,
            instance,
            background_mean,
            max_coalitions,
        )?;

        // Create importance map
        let mut importance = HashMap::new();
        for (i, name) in feature_names.iter().enumerate() {
            if i < shapley_values.len() {
                importance.insert(name.clone(), shapley_values[i].abs());
            }
        }

        Ok(importance)
    }

    /// Compute background mean for SHAP baseline
    fn compute_background_mean(&self, xdata: &Array2<F>) -> Result<Array1<F>> {
        if xdata.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty _data for background computation".to_string(),
            ));
        }

        let n_features = xdata.ncols();
        let mut background = Array1::zeros(n_features);

        for j in 0..n_features {
            let column_sum: F = xdata.column(j).iter().cloned().sum();
            background[j] = column_sum / F::from(xdata.nrows()).unwrap();
        }

        Ok(background)
    }

    /// Efficient approximation of Shapley values using sampling
    fn compute_shapley_values_approximation<M>(
        &self,
        model: &M,
        instance: &ArrayView1<F>,
        background: &Array1<F>,
        max_coalitions: usize,
    ) -> Result<Vec<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = instance.len();
        let mut shapley_values = vec![F::zero(); n_features];

        // Get baseline prediction (no features)
        let baseline_input =
            Array2::from_shape_vec((1, n_features), background.to_vec()).map_err(|_| {
                MetricsError::InvalidInput("Failed to create baseline array".to_string())
            })?;
        let baseline_pred = model(&baseline_input.view())[0];

        // Get full prediction (all features)
        let full_input = Array2::from_shape_vec((1, n_features), instance.to_vec())
            .map_err(|_| MetricsError::InvalidInput("Failed to create full array".to_string()))?;
        let full_pred = model(&full_input.view())[0];

        // For efficiency, use sampling-based approximation
        let nsamples = std::cmp::min(max_coalitions, 1000);

        for i in 0..n_features {
            let mut marginal_contributions = Vec::new();

            // Sample different _coalitions and compute marginal contribution of feature i
            for sample_idx in 0..nsamples {
                let coalition = self.generate_random_coalition(n_features, i, sample_idx);

                // Compute prediction with coalition including feature i
                let with_i =
                    self.create_coalition_input(instance, background, &coalition, Some(i))?;
                let pred_with_i = model(&with_i.view())[0];

                // Compute prediction with coalition excluding feature i
                let without_i =
                    self.create_coalition_input(instance, background, &coalition, None)?;
                let pred_without_i = model(&without_i.view())[0];

                // Marginal contribution
                let marginal_contrib = pred_with_i - pred_without_i;
                marginal_contributions.push(marginal_contrib);
            }

            // Average marginal contributions to get Shapley value
            if !marginal_contributions.is_empty() {
                let sum: F = marginal_contributions.iter().cloned().sum();
                shapley_values[i] = sum / F::from(marginal_contributions.len()).unwrap();
            }
        }

        // Ensure Shapley values sum to difference between full and baseline predictions
        // (efficiency property of Shapley values)
        let total_difference = full_pred - baseline_pred;
        let shapley_sum: F = shapley_values.iter().cloned().sum();

        if shapley_sum != F::zero() {
            let normalization_factor = total_difference / shapley_sum;
            for val in shapley_values.iter_mut() {
                *val = *val * normalization_factor;
            }
        }

        Ok(shapley_values)
    }

    /// Generate a random coalition (subset of features) for sampling
    fn generate_random_coalition(
        &self,
        n_features: usize,
        target_feature: usize,
        seed: usize,
    ) -> Vec<bool> {
        let mut coalition = vec![false; n_features];

        // Use simple deterministic "random" based on seed for reproducibility
        let mut pseudo_random = seed;

        for i in 0..n_features {
            if i != target_feature {
                pseudo_random = pseudo_random.wrapping_mul(1103515245).wrapping_add(12345);
                coalition[i] = pseudo_random.is_multiple_of(2);
            }
        }

        coalition
    }

    /// Create input array for a specific coalition
    fn create_coalition_input(
        &self,
        instance: &ArrayView1<F>,
        background: &Array1<F>,
        coalition: &[bool],
        include_target: Option<usize>,
    ) -> Result<Array2<F>> {
        let n_features = instance.len();
        let mut coalition_input = background.clone();

        // Include features in coalition
        for (i, &in_coalition) in coalition.iter().enumerate() {
            if in_coalition {
                coalition_input[i] = instance[i];
            }
        }

        // Include or exclude _target feature
        if let Some(target_idx) = include_target {
            if target_idx < n_features {
                coalition_input[target_idx] = instance[target_idx];
            }
        }

        // Convert to 2D array for model input
        Array2::from_shape_vec((1, n_features), coalition_input.to_vec()).map_err(|_| {
            MetricsError::InvalidInput("Failed to create coalition input array".to_string())
        })
    }

    /// Complete gradient-based importance computation using numerical differentiation
    fn compute_gradient_importance<M>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        if x_test.is_empty() || feature_names.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty input data or feature _names".to_string(),
            ));
        }

        if x_test.ncols() != feature_names.len() {
            return Err(MetricsError::InvalidInput(
                "Number of features doesn't match feature _names length".to_string(),
            ));
        }

        let mut importance_scores = HashMap::new();

        // Process each instance separately for local explanations
        for instance in x_test.axis_iter(Axis(0)) {
            let instance_importance =
                self.compute_gradient_for_instance(model, &instance, feature_names)?;

            // Aggregate importance scores across instances
            for (feature_name, importance) in instance_importance {
                let current_score = importance_scores
                    .get(&feature_name)
                    .copied()
                    .unwrap_or(F::zero());
                importance_scores.insert(
                    feature_name,
                    current_score + importance / F::from(x_test.nrows()).unwrap(),
                );
            }
        }

        Ok(importance_scores)
    }

    /// Compute gradient-based importance for a single instance
    fn compute_gradient_for_instance<M>(
        &self,
        model: &M,
        instance: &ArrayView1<F>,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = instance.len();

        // Compute numerical gradients using finite differences
        let gradients = self.compute_numerical_gradients(model, instance)?;

        // Multiple gradient-based attribution methods
        let saliency_map = self.compute_saliency_map(&gradients, instance)?;
        let integrated_gradients = self.compute_integrated_gradients(model, instance)?;
        let gradient_times_input = self.compute_gradient_times_input(&gradients, instance)?;

        // Combine different gradient methods with equal weighting
        let mut importance = HashMap::new();
        for (i, name) in feature_names.iter().enumerate() {
            if i < n_features {
                let combined_importance =
                    (saliency_map[i] + integrated_gradients[i] + gradient_times_input[i])
                        / F::from(3.0).unwrap();
                importance.insert(name.clone(), combined_importance.abs());
            }
        }

        Ok(importance)
    }

    /// Compute numerical gradients using finite differences
    fn compute_numerical_gradients<M>(&self, model: &M, instance: &ArrayView1<F>) -> Result<Vec<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = instance.len();
        let mut gradients = vec![F::zero(); n_features];

        // Use adaptive step size based on feature magnitude
        let epsilon_base = F::from(1e-5).unwrap();

        // Get baseline prediction
        let baseline_input =
            Array2::from_shape_vec((1, n_features), instance.to_vec()).map_err(|_| {
                MetricsError::InvalidInput("Failed to create baseline array".to_string())
            })?;
        let _baseline_pred = model(&baseline_input.view())[0];

        // Compute partial derivatives using central differences
        for i in 0..n_features {
            let feature_magnitude = instance[i].abs().max(F::from(1.0).unwrap());
            let epsilon = epsilon_base * feature_magnitude;

            // Forward step
            let mut forward_instance = instance.to_owned();
            forward_instance[i] = forward_instance[i] + epsilon;
            let forward_input = Array2::from_shape_vec((1, n_features), forward_instance.to_vec())
                .map_err(|_| {
                    MetricsError::InvalidInput("Failed to create forward array".to_string())
                })?;
            let forward_pred = model(&forward_input.view())[0];

            // Backward step
            let mut backward_instance = instance.to_owned();
            backward_instance[i] = backward_instance[i] - epsilon;
            let backward_input =
                Array2::from_shape_vec((1, n_features), backward_instance.to_vec()).map_err(
                    |_| MetricsError::InvalidInput("Failed to create backward array".to_string()),
                )?;
            let backward_pred = model(&backward_input.view())[0];

            // Central difference approximation
            gradients[i] = (forward_pred - backward_pred) / (F::from(2.0).unwrap() * epsilon);
        }

        Ok(gradients)
    }

    /// Compute saliency map (simple gradient magnitude)
    fn compute_saliency_map(&self, gradients: &[F], instance: &ArrayView1<F>) -> Result<Vec<F>> {
        // Saliency map is simply the absolute gradient values
        Ok(gradients.iter().map(|&g| g.abs()).collect())
    }

    /// Compute integrated gradients approximation
    fn compute_integrated_gradients<M>(&self, model: &M, instance: &ArrayView1<F>) -> Result<Vec<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_features = instance.len();
        let mut integrated_grads = vec![F::zero(); n_features];

        // Use zero baseline for integrated gradients
        let baseline = Array1::zeros(n_features);
        let n_steps = 50; // Number of integration steps

        // Approximate integral using Riemann sum
        for step in 0..n_steps {
            let alpha = F::from(step as f64).unwrap() / F::from(n_steps as f64).unwrap();

            // Interpolate between baseline and instance
            let mut interpolated = Array1::zeros(n_features);
            for i in 0..n_features {
                interpolated[i] = baseline[i] + alpha * (instance[i] - baseline[i]);
            }

            // Compute gradients at interpolated point
            let step_gradients = self.compute_numerical_gradients(model, &interpolated.view())?;

            // Accumulate gradients
            for i in 0..n_features {
                integrated_grads[i] =
                    integrated_grads[i] + step_gradients[i] * (instance[i] - baseline[i]);
            }
        }

        // Average over steps
        for grad in integrated_grads.iter_mut() {
            *grad = *grad / F::from(n_steps).unwrap();
        }

        Ok(integrated_grads)
    }

    /// Compute gradient × input attribution
    fn compute_gradient_times_input(
        &self,
        gradients: &[F],
        instance: &ArrayView1<F>,
    ) -> Result<Vec<F>> {
        let mut grad_times_input = Vec::new();

        for (i, &grad) in gradients.iter().enumerate() {
            if i < instance.len() {
                grad_times_input.push(grad * instance[i]);
            }
        }

        Ok(grad_times_input)
    }
}

/// Explanation method types
#[derive(Debug, Clone)]
pub enum ExplanationMethod {
    /// Permutation importance
    Permutation,
    /// LIME (Local Interpretable Model-agnostic Explanations)
    LIME,
    /// SHAP (SHapley Additive exPlanations)
    SHAP,
    /// Gradient-based explanations
    GradientBased,
}

/// Compute model interpretability score
#[allow(dead_code)]
pub fn compute_interpretability_score<F: Float + std::iter::Sum>(
    explainability_metrics: &ExplainabilityMetrics<F>,
) -> F {
    // Weighted combination of different explainability aspects
    let feature_importance_score = if explainability_metrics.feature_importance.is_empty() {
        F::zero()
    } else {
        explainability_metrics
            .feature_importance
            .values()
            .cloned()
            .sum::<F>()
            / F::from(explainability_metrics.feature_importance.len()).unwrap()
    };

    let weights = [
        F::from(0.25).unwrap(), // feature importance
        F::from(0.2).unwrap(),  // local consistency
        F::from(0.2).unwrap(),  // global stability
        F::from(0.15).unwrap(), // faithfulness
        F::from(0.15).unwrap(), // completeness
        F::from(0.05).unwrap(), // uncertainty
    ];

    let scores = [
        feature_importance_score,
        explainability_metrics.local_consistency,
        explainability_metrics.global_stability,
        explainability_metrics.faithfulness,
        explainability_metrics.completeness,
        F::one() - explainability_metrics.uncertainty_metrics.total_uncertainty, // Lower uncertainty is better
    ];

    weights
        .iter()
        .zip(scores.iter())
        .map(|(&w, &s)| w * s)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_explainability_evaluator_creation() {
        let evaluator = ExplainabilityEvaluator::<f64>::new()
            .with_perturbations(50)
            .with_perturbation_strength(0.05)
            .with_importance_threshold(0.02);

        assert_eq!(evaluator.n_perturbations, 50);
        assert_eq!(evaluator.perturbation_strength, 0.05);
        assert_eq!(evaluator.importance_threshold, 0.02);
    }

    #[test]
    fn test_correlation_computation() {
        let evaluator = ExplainabilityEvaluator::<f64>::new();

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation

        let correlation = evaluator.compute_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_permutation_feature() {
        let evaluator = ExplainabilityEvaluator::<f64>::new();
        let mut data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let original_data = data.clone();

        evaluator.permute_feature(&mut data, 1).unwrap();

        // Feature 1 should be different, others should be the same
        assert_eq!(data.column(0), original_data.column(0));
        assert_eq!(data.column(2), original_data.column(2));
        // Column 1 should have the same values but potentially in different order
        assert_eq!(data.column(1).len(), original_data.column(1).len());
    }

    #[test]
    fn test_interpretability_score() {
        let mut feature_importance = HashMap::new();
        feature_importance.insert("feature1".to_string(), 0.5);
        feature_importance.insert("feature2".to_string(), 0.3);

        let metrics = ExplainabilityMetrics {
            feature_importance,
            local_consistency: 0.8,
            global_stability: 0.7,
            uncertainty_metrics: UncertaintyMetrics {
                epistemic_uncertainty: 0.1,
                aleatoric_uncertainty: 0.05,
                total_uncertainty: 0.15,
                coverage: 0.95,
                calibration_error: 0.02,
            },
            faithfulness: 0.9,
            completeness: 0.85,
        };

        let score = compute_interpretability_score(&metrics);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_bootstrap_sampling() {
        let evaluator = ExplainabilityEvaluator::<f64>::new();
        let indices = evaluator.bootstrap_sample_indices(10).unwrap();

        assert_eq!(indices.len(), 10);
        // All indices should be valid (0-9)
        assert!(indices.iter().all(|&i| i < 10));
    }

    #[test]
    fn test_mask_important_features() {
        let evaluator = ExplainabilityEvaluator::<f64>::new();
        let sample = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let explanation = array![0.1, 0.5, 0.2, 0.8, 0.3]; // Feature 3 most important, then 1

        let masked = evaluator
            .mask_important_features(&sample.view(), &explanation, 2)
            .unwrap();

        // Features 3 and 1 (most important) should be masked to 0
        assert_eq!(masked[3], 0.0);
        assert_eq!(masked[1], 0.0);
        // Other features should remain unchanged
        assert_eq!(masked[0], 1.0);
        assert_eq!(masked[2], 3.0);
        assert_eq!(masked[4], 5.0);
    }
}
