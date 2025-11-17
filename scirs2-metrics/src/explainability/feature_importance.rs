//! Feature importance analysis for model explainability
//!
//! This module provides comprehensive feature importance calculation methods including:
//! - Permutation importance with proper random shuffling
//! - SHAP (SHapley Additive exPlanations) values
//! - LIME (Local Interpretable Model-agnostic Explanations)
//! - Integrated Gradients
//! - Gain-based importance for tree models
//! - Improved mutual information estimation

#![allow(clippy::too_many_arguments)]

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::iter::Sum;

/// Feature importance calculator with advanced methods
pub struct FeatureImportanceCalculator<F: Float> {
    /// Number of permutations for permutation importance
    pub n_permutations: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Number of SHAP background samples
    pub n_shap_background: usize,
    /// Use proper random number generation
    pub use_proper_rng: bool,
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum> Default
    for FeatureImportanceCalculator<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum>
    FeatureImportanceCalculator<F>
{
    /// Create new feature importance calculator
    pub fn new() -> Self {
        Self {
            n_permutations: 100,
            random_seed: None,
            n_shap_background: 100,
            use_proper_rng: true,
            enable_simd: true,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set number of permutations
    pub fn with_permutations(mut self, n: usize) -> Self {
        self.n_permutations = n;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set number of SHAP background samples
    pub fn with_shap_background(mut self, n: usize) -> Self {
        self.n_shap_background = n;
        self
    }

    /// Enable/disable proper random number generation
    pub fn with_proper_rng(mut self, enable: bool) -> Self {
        self.use_proper_rng = enable;
        self
    }

    /// Enable/disable SIMD acceleration
    pub fn with_simd(mut self, enable: bool) -> Self {
        self.enable_simd = enable;
        self
    }

    /// Compute permutation importance
    pub fn permutation_importance<M, S>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        y_test: &Array1<F>,
        score_fn: S,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
        S: Fn(&Array1<F>, &Array1<F>) -> F,
    {
        // Baseline score
        let baseline_predictions = model(&x_test.view());
        let baseline_score = score_fn(y_test, &baseline_predictions);

        let mut importance_scores = HashMap::new();
        let n_features = x_test.ncols();

        for (feature_idx, feature_name) in feature_names.iter().enumerate() {
            if feature_idx >= n_features {
                continue;
            }

            let mut permutation_scores = Vec::new();

            for _ in 0..self.n_permutations {
                let mut x_permuted = x_test.clone();
                self.permute_column(&mut x_permuted, feature_idx)?;

                let permuted_predictions = model(&x_permuted.view());
                let permuted_score = score_fn(y_test, &permuted_predictions);

                let importance = baseline_score - permuted_score;
                permutation_scores.push(importance);
            }

            let mean_importance = permutation_scores.iter().cloned().sum::<F>()
                / F::from(permutation_scores.len()).unwrap();

            importance_scores.insert(feature_name.clone(), mean_importance);
        }

        Ok(importance_scores)
    }

    /// Compute feature importance using drop column method
    pub fn drop_column_importance<M, S>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        y_test: &Array1<F>,
        score_fn: S,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
        S: Fn(&Array1<F>, &Array1<F>) -> F,
    {
        // Baseline score with all features
        let baseline_predictions = model(&x_test.view());
        let baseline_score = score_fn(y_test, &baseline_predictions);

        let mut importance_scores = HashMap::new();
        let n_features = x_test.ncols();

        for (feature_idx, feature_name) in feature_names.iter().enumerate() {
            if feature_idx >= n_features {
                continue;
            }

            // Create dataset without this feature
            let _x_without_feature = self.drop_column(x_test, feature_idx)?;

            // Note: In practice, you'd need a model that can handle different input sizes
            // For this example, we'll set the dropped feature to zero instead
            let mut x_zeroed = x_test.clone();
            for i in 0..x_zeroed.nrows() {
                x_zeroed[[i, feature_idx]] = F::zero();
            }

            let reduced_predictions = model(&x_zeroed.view());
            let reduced_score = score_fn(y_test, &reduced_predictions);

            let importance = baseline_score - reduced_score;
            importance_scores.insert(feature_name.clone(), importance);
        }

        Ok(importance_scores)
    }

    /// Compute feature importance statistics
    pub fn compute_importance_statistics(
        &self,
        importance_scores: &HashMap<String, F>,
    ) -> FeatureImportanceStats<F> {
        let values: Vec<F> = importance_scores.values().cloned().collect();

        if values.is_empty() {
            return FeatureImportanceStats::default();
        }

        let mean = values.iter().cloned().sum::<F>() / F::from(values.len()).unwrap();

        let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>()
            / F::from(values.len()).unwrap();

        let std_dev = variance.sqrt();

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted_values.len().is_multiple_of(2) {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / F::from(2).unwrap()
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];

        FeatureImportanceStats {
            mean,
            std_dev,
            median,
            min,
            max,
            n_features: values.len(),
        }
    }

    /// Get top-k most important features
    pub fn get_top_features(
        &self,
        importance_scores: &HashMap<String, F>,
        k: usize,
    ) -> Vec<(String, F)> {
        let mut sorted_features: Vec<(String, F)> = importance_scores
            .iter()
            .map(|(name, &score)| (name.clone(), score))
            .collect();

        sorted_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_features.truncate(k);
        sorted_features
    }

    /// Filter features by importance threshold
    pub fn filter_by_threshold(
        &self,
        importance_scores: &HashMap<String, F>,
        threshold: F,
    ) -> HashMap<String, F> {
        importance_scores
            .iter()
            .filter(|(_, &score)| score >= threshold)
            .map(|(name, &score)| (name.clone(), score))
            .collect()
    }

    /// Compute SHAP values for feature importance
    pub fn shap_importance<M, S>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        x_background: &Array2<F>,
        score_fn: S,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
        S: Fn(&Array1<F>, &Array1<F>) -> F,
    {
        let mut shap_values = HashMap::new();
        let n_features = x_test.ncols();

        for (feature_idx, feature_name) in feature_names.iter().enumerate() {
            if feature_idx >= n_features {
                continue;
            }

            let shap_value = self.compute_single_shap_value(
                model,
                x_test,
                x_background,
                &score_fn,
                feature_idx,
            )?;

            shap_values.insert(feature_name.clone(), shap_value);
        }

        Ok(shap_values)
    }

    /// Compute gain-based importance (for tree-like models)
    pub fn gain_importance<M>(
        &self,
        model: &M,
        _test: &Array2<F>,
        feature_names: &[String],
        tree_splits: &[TreeSplit<F>],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let mut gain_scores = HashMap::new();

        for (feature_idx, feature_name) in feature_names.iter().enumerate() {
            let total_gain = tree_splits
                .iter()
                .filter(|split| split.feature_idx == feature_idx)
                .map(|split| split.gain)
                .sum();

            gain_scores.insert(feature_name.clone(), total_gain);
        }

        Ok(gain_scores)
    }

    /// Compute LIME-style local importance
    pub fn lime_importance<M, S>(
        &self,
        model: &M,
        x_instance: &ArrayView1<F>,
        x_background: &Array2<F>,
        _score_fn: S,
        feature_names: &[String],
        n_samples: usize,
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
        S: Fn(&Array1<F>, &Array1<F>) -> F,
    {
        let mut lime_scores = HashMap::new();
        let _n_features = x_instance.len();

        // Generate perturbed _samples around the _instance
        let perturbed_samples = self.generate_lime_samples(x_instance, x_background, n_samples)?;

        // Compute model predictions for perturbed _samples
        let predictions = model(&perturbed_samples.view());

        // Fit linear model to approximate local behavior
        let feature_weights = self.fit_linear_approximation(&perturbed_samples, &predictions)?;

        for (feature_idx, feature_name) in feature_names.iter().enumerate() {
            if feature_idx < feature_weights.len() {
                lime_scores.insert(feature_name.clone(), feature_weights[feature_idx]);
            }
        }

        Ok(lime_scores)
    }

    /// Compute integrated gradients importance
    pub fn integrated_gradients_importance<M>(
        &self,
        model: &M,
        x_instance: &ArrayView1<F>,
        x_baseline: &ArrayView1<F>,
        feature_names: &[String],
        n_steps: usize,
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let mut ig_scores = HashMap::new();

        // Generate interpolated path from _baseline to _instance
        let interpolated_samples =
            self.generate_interpolated_path(x_baseline, x_instance, n_steps)?;

        // Compute gradients along the path (simplified numerical gradients)
        let gradients = self.compute_numerical_gradients(model, &interpolated_samples)?;

        // Integrate gradients
        let integrated_grads = self.integrate_gradients(&gradients, x_instance, x_baseline)?;

        for (feature_idx, feature_name) in feature_names.iter().enumerate() {
            if feature_idx < integrated_grads.len() {
                ig_scores.insert(feature_name.clone(), integrated_grads[feature_idx]);
            }
        }

        Ok(ig_scores)
    }

    // Helper methods

    fn permute_column(&self, data: &mut Array2<F>, columnidx: usize) -> Result<()> {
        if columnidx >= data.ncols() {
            return Err(MetricsError::InvalidInput(
                "Column index out of bounds".to_string(),
            ));
        }

        let mut column_values: Vec<F> = data.column(columnidx).to_vec();

        if self.use_proper_rng {
            // Use Fisher-Yates shuffle for proper randomization
            self.fisher_yates_shuffle(&mut column_values)?;
        } else {
            // Fallback to simple permutation
            for i in (1..column_values.len()).rev() {
                let j = match self.random_seed {
                    Some(seed) => (seed as usize + i) % (i + 1),
                    None => i % (i + 1),
                };
                column_values.swap(i, j);
            }
        }

        for (i, &value) in column_values.iter().enumerate() {
            data[[i, columnidx]] = value;
        }

        Ok(())
    }

    /// Proper Fisher-Yates shuffle implementation
    fn fisher_yates_shuffle(&self, values: &mut [F]) -> Result<()> {
        let seed = self.random_seed.unwrap_or(42);
        let mut rng_state = seed;

        for i in (1..values.len()).rev() {
            // Linear congruential generator for reproducible randomness
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (rng_state as usize) % (i + 1);
            values.swap(i, j);
        }

        Ok(())
    }

    /// Compute single SHAP value for a feature
    fn compute_single_shap_value<M, S>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        x_background: &Array2<F>,
        score_fn: &S,
        feature_idx: usize,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
        S: Fn(&Array1<F>, &Array1<F>) -> F,
    {
        let n_features = x_test.ncols();
        let mut total_contribution = F::zero();
        let mut sample_count = 0;

        // Sample coalition subsets
        for _ in 0..self.n_shap_background {
            let coalition = self.sample_coalition(n_features, feature_idx)?;

            // Compute marginal contribution
            let contribution = self.compute_marginal_contribution(
                model,
                x_test,
                x_background,
                score_fn,
                &coalition,
                feature_idx,
            )?;

            total_contribution = total_contribution + contribution;
            sample_count += 1;
        }

        Ok(total_contribution / F::from(sample_count).unwrap())
    }

    /// Sample a random coalition of features
    fn sample_coalition(&self, n_features: usize, targetfeature: usize) -> Result<Vec<bool>> {
        let mut coalition = vec![false; n_features];
        let seed = self.random_seed.unwrap_or(42);
        let mut rng_state = seed;

        for i in 0..n_features {
            if i != targetfeature {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                coalition[i] = rng_state.is_multiple_of(2);
            }
        }

        Ok(coalition)
    }

    /// Compute marginal contribution of a feature to a coalition
    fn compute_marginal_contribution<M, S>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        x_background: &Array2<F>,
        _score_fn: &S,
        coalition: &[bool],
        feature_idx: usize,
    ) -> Result<F>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
        S: Fn(&Array1<F>, &Array1<F>) -> F,
    {
        // Create coalition with feature
        let mut coalition_with = coalition.to_vec();
        coalition_with[feature_idx] = true;

        // Create coalition without feature
        let coalition_without = coalition.to_vec();

        // Generate samples for both coalitions
        let x_with = self.generate_coalition_sample(x_test, x_background, &coalition_with)?;
        let x_without = self.generate_coalition_sample(x_test, x_background, &coalition_without)?;

        // Compute model outputs
        let pred_with = model(&x_with.view());
        let pred_without = model(&x_without.view());

        // Compute marginal contribution
        let contribution = if let (Some(&first_with), Some(&first_without)) =
            (pred_with.first(), pred_without.first())
        {
            first_with - first_without
        } else {
            F::zero()
        };

        Ok(contribution)
    }

    /// Generate sample for a specific coalition
    fn generate_coalition_sample(
        &self,
        x_test: &Array2<F>,
        x_background: &Array2<F>,
        coalition: &[bool],
    ) -> Result<Array2<F>> {
        let mut sample = x_test.clone();

        for (feature_idx, &in_coalition) in coalition.iter().enumerate() {
            if !in_coalition && feature_idx < x_background.ncols() {
                // Replace with _background value
                let bg_mean = x_background.column(feature_idx).mean_or(F::zero());
                for row_idx in 0..sample.nrows() {
                    sample[[row_idx, feature_idx]] = bg_mean;
                }
            }
        }

        Ok(sample)
    }

    /// Generate LIME samples around an instance
    fn generate_lime_samples(
        &self,
        x_instance: &ArrayView1<F>,
        x_background: &Array2<F>,
        n_samples: usize,
    ) -> Result<Array2<F>> {
        let n_features = x_instance.len();
        let mut samples = Array2::zeros((n_samples, n_features));

        let seed = self.random_seed.unwrap_or(42);
        let mut rng_state = seed;

        for i in 0..n_samples {
            for j in 0..n_features {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);

                if rng_state.is_multiple_of(2) {
                    // Use original feature value
                    samples[[i, j]] = x_instance[j];
                } else {
                    // Use _background value
                    let bg_mean = x_background.column(j).mean_or(F::zero());
                    samples[[i, j]] = bg_mean;
                }
            }
        }

        Ok(samples)
    }

    /// Fit linear approximation for LIME
    fn fit_linear_approximation(
        &self,
        xsamples: &Array2<F>,
        y_predictions: &Array1<F>,
    ) -> Result<Vec<F>> {
        let n_features = xsamples.ncols();
        let mut weights = vec![F::zero(); n_features];

        // Simplified linear regression using normal equations
        // In practice, this would use proper linear algebra
        for feature_idx in 0..n_features {
            let feature_values: Vec<F> = xsamples.column(feature_idx).to_vec();
            let correlation =
                self.compute_correlation_vec(&feature_values, &y_predictions.to_vec())?;
            weights[feature_idx] = correlation;
        }

        Ok(weights)
    }

    /// Generate interpolated path for integrated gradients
    fn generate_interpolated_path(
        &self,
        x_baseline: &ArrayView1<F>,
        x_instance: &ArrayView1<F>,
        n_steps: usize,
    ) -> Result<Array2<F>> {
        let n_features = x_baseline.len();
        let mut path = Array2::zeros((n_steps, n_features));

        for step in 0..n_steps {
            let alpha = F::from(step).unwrap() / F::from(n_steps - 1).unwrap();

            for feature_idx in 0..n_features {
                path[[step, feature_idx]] = x_baseline[feature_idx]
                    + alpha * (x_instance[feature_idx] - x_baseline[feature_idx]);
            }
        }

        Ok(path)
    }

    /// Compute numerical gradients
    fn compute_numerical_gradients<M>(&self, model: &M, xsamples: &Array2<F>) -> Result<Array2<F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
    {
        let n_samples = xsamples.nrows();
        let n_features = xsamples.ncols();
        let mut gradients = Array2::zeros((n_samples, n_features));

        let epsilon = F::from(1e-6).unwrap();

        for sample_idx in 0..n_samples {
            for feature_idx in 0..n_features {
                let mut x_plus = xsamples.row(sample_idx).to_owned();
                let mut x_minus = xsamples.row(sample_idx).to_owned();

                x_plus[feature_idx] = x_plus[feature_idx] + epsilon;
                x_minus[feature_idx] = x_minus[feature_idx] - epsilon;

                let pred_plus = model(&x_plus.view().insert_axis(Axis(0)));
                let pred_minus = model(&x_minus.view().insert_axis(Axis(0)));

                let gradient = (pred_plus[0] - pred_minus[0]) / (F::from(2.0).unwrap() * epsilon);
                gradients[[sample_idx, feature_idx]] = gradient;
            }
        }

        Ok(gradients)
    }

    /// Integrate gradients for integrated gradients method
    fn integrate_gradients(
        &self,
        gradients: &Array2<F>,
        x_instance: &ArrayView1<F>,
        x_baseline: &ArrayView1<F>,
    ) -> Result<Vec<F>> {
        let n_features = x_instance.len();
        let mut integrated = vec![F::zero(); n_features];

        for feature_idx in 0..n_features {
            let feature_diff = x_instance[feature_idx] - x_baseline[feature_idx];
            let avg_gradient = gradients.column(feature_idx).mean_or(F::zero());
            integrated[feature_idx] = feature_diff * avg_gradient;
        }

        Ok(integrated)
    }

    /// Compute correlation between two vectors
    fn compute_correlation_vec(&self, x: &[F], y: &[F]) -> Result<F> {
        if x.len() != y.len() {
            return Err(MetricsError::InvalidInput(
                "Vectors must have the same length".to_string(),
            ));
        }

        let n = F::from(x.len()).unwrap();
        let mean_x = x.iter().cloned().sum::<F>() / n;
        let mean_y = y.iter().cloned().sum::<F>() / n;

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

    fn drop_column(&self, data: &Array2<F>, columnidx: usize) -> Result<Array2<F>> {
        if columnidx >= data.ncols() {
            return Err(MetricsError::InvalidInput(
                "Column index out of bounds".to_string(),
            ));
        }

        let n_rows = data.nrows();
        let n_cols = data.ncols() - 1;
        let mut result = Array2::zeros((n_rows, n_cols));

        for i in 0..n_rows {
            let mut result_col = 0;
            for j in 0..data.ncols() {
                if j != columnidx {
                    result[[i, result_col]] = data[[i, j]];
                    result_col += 1;
                }
            }
        }

        Ok(result)
    }
}

/// Feature importance statistics
#[derive(Debug, Clone)]
pub struct FeatureImportanceStats<F: Float> {
    pub mean: F,
    pub std_dev: F,
    pub median: F,
    pub min: F,
    pub max: F,
    pub n_features: usize,
}

impl<F: Float> Default for FeatureImportanceStats<F> {
    fn default() -> Self {
        Self {
            mean: F::zero(),
            std_dev: F::zero(),
            median: F::zero(),
            min: F::zero(),
            max: F::zero(),
            n_features: 0,
        }
    }
}

/// Tree split information for gain-based importance
#[derive(Debug, Clone)]
pub struct TreeSplit<F: Float> {
    pub feature_idx: usize,
    pub threshold: F,
    pub gain: F,
    pub n_samples: usize,
}

/// Advanced feature importance results
#[derive(Debug, Clone)]
pub struct AdvancedImportanceResults<F: Float> {
    pub permutation_importance: HashMap<String, F>,
    pub shap_values: Option<HashMap<String, F>>,
    pub lime_importance: Option<HashMap<String, F>>,
    pub integrated_gradients: Option<HashMap<String, F>>,
    pub gain_importance: Option<HashMap<String, F>>,
    pub mutual_information: HashMap<String, F>,
}

/// Compute comprehensive feature importance using multiple methods
#[allow(dead_code)]
pub fn comprehensive_feature_importance<F, M, S>(
    calculator: &FeatureImportanceCalculator<F>,
    model: &M,
    x_test: &Array2<F>,
    y_test: &Array1<F>,
    x_background: Option<&Array2<F>>,
    score_fn: S,
    feature_names: &[String],
    tree_splits: Option<&[TreeSplit<F>]>,
) -> Result<AdvancedImportanceResults<F>>
where
    F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    M: Fn(&ArrayView2<F>) -> Array1<F>,
    S: Fn(&Array1<F>, &Array1<F>) -> F + Copy,
{
    // Permutation importance (always computed)
    let permutation_importance =
        calculator.permutation_importance(model, x_test, y_test, score_fn, feature_names)?;

    // SHAP values (if _background data provided)
    let shap_values = if let Some(bg_data) = x_background {
        Some(calculator.shap_importance(model, x_test, bg_data, score_fn, feature_names)?)
    } else {
        None
    };

    // LIME importance (using first instance if _background provided)
    let lime_importance = if let Some(bg_data) = x_background {
        if x_test.nrows() > 0 {
            Some(calculator.lime_importance(
                model,
                &x_test.row(0),
                bg_data,
                score_fn,
                feature_names,
                100,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    // Integrated gradients (using first instance if _background provided)
    let integrated_gradients = if let Some(bg_data) = x_background {
        if x_test.nrows() > 0 && bg_data.nrows() > 0 {
            Some(calculator.integrated_gradients_importance(
                model,
                &x_test.row(0),
                &bg_data.row(0),
                feature_names,
                50,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    // Gain-based importance (if tree _splits provided)
    let gain_importance = if let Some(splits) = tree_splits {
        Some(calculator.gain_importance(model, x_test, feature_names, splits)?)
    } else {
        None
    };

    // Mutual information (always computed)
    let mutual_information = mutual_information_importance(x_test, y_test, feature_names)?;

    Ok(AdvancedImportanceResults {
        permutation_importance,
        shap_values,
        lime_importance,
        integrated_gradients,
        gain_importance,
        mutual_information,
    })
}

/// Compute mutual information based feature importance with improved estimation
#[allow(dead_code)]
pub fn mutual_information_importance<
    F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
>(
    x: &Array2<F>,
    y: &Array1<F>,
    feature_names: &[String],
) -> Result<HashMap<String, F>> {
    let mut importance_scores = HashMap::new();

    for (i, feature_name) in feature_names.iter().enumerate() {
        if i >= x.ncols() {
            continue;
        }

        let feature_column = x.column(i);
        let mi_score = compute_mutual_information_improved(&feature_column, y)?;
        importance_scores.insert(feature_name.clone(), mi_score);
    }

    Ok(importance_scores)
}

/// Compute mutual information between two variables (improved with binning)
#[allow(dead_code)]
fn compute_mutual_information_improved<
    F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
>(
    x: &scirs2_core::ndarray::ArrayView1<F>,
    y: &Array1<F>,
) -> Result<F> {
    if x.len() != y.len() {
        return Err(MetricsError::InvalidInput(
            "Variables must have the same length".to_string(),
        ));
    }

    // Improved mutual information with histogram-based estimation
    let n_bins = 10.min(x.len() / 5).max(2); // Adaptive number of bins

    // Create bins for both variables
    let x_bins = create_bins(x, n_bins)?;
    let y_bins = create_bins(&y.view(), n_bins)?;

    // Compute joint and marginal histograms
    let joint_hist = compute_joint_histogram::<F>(&x_bins, &y_bins, n_bins)?;
    let x_hist: Vec<F> = compute_marginal_histogram(&x_bins, n_bins)?;
    let y_hist: Vec<F> = compute_marginal_histogram(&y_bins, n_bins)?;

    // Compute mutual information
    let mut mi = F::zero();
    let n_total = F::from(x.len()).unwrap();

    for i in 0..n_bins {
        for j in 0..n_bins {
            let p_xy: F = joint_hist[(i, j)] / n_total;
            let p_x = x_hist[i] / n_total;
            let p_y = y_hist[j] / n_total;

            if p_xy > F::zero() && p_x > F::zero() && p_y > F::zero() {
                let ratio: F = p_xy / (p_x * p_y);
                mi = mi + p_xy * ratio.ln();
            }
        }
    }

    Ok(mi.max(F::zero()))
}

/// Create bins for a variable
#[allow(dead_code)]
fn create_bins<F: Float + scirs2_core::numeric::FromPrimitive>(
    values: &scirs2_core::ndarray::ArrayView1<F>,
    n_bins: usize,
) -> Result<Vec<usize>> {
    let min_val = values.iter().cloned().fold(F::infinity(), F::min);
    let max_val = values.iter().cloned().fold(F::neg_infinity(), F::max);

    if min_val == max_val {
        return Ok(vec![0; values.len()]);
    }

    let bin_width = (max_val - min_val) / F::from(n_bins).unwrap();

    let _bins: Vec<usize> = values
        .iter()
        .map(|&val| {
            let bin_idx = ((val - min_val) / bin_width).to_usize().unwrap_or(0);
            bin_idx.min(n_bins - 1)
        })
        .collect();

    Ok(_bins)
}

/// Compute joint histogram
#[allow(dead_code)]
fn compute_joint_histogram<F: Float + scirs2_core::numeric::FromPrimitive>(
    x_bins: &[usize],
    y_bins: &[usize],
    n_bins: usize,
) -> Result<scirs2_core::ndarray::Array2<F>> {
    let mut hist = scirs2_core::ndarray::Array2::zeros((n_bins, n_bins));

    for (&x_bin, &y_bin) in x_bins.iter().zip(y_bins.iter()) {
        hist[(x_bin, y_bin)] = hist[(x_bin, y_bin)] + F::one();
    }

    Ok(hist)
}

/// Compute marginal histogram
#[allow(dead_code)]
fn compute_marginal_histogram<F: Float + scirs2_core::numeric::FromPrimitive>(
    bins: &[usize],
    n_bins: usize,
) -> Result<Vec<F>> {
    let mut hist = vec![F::zero(); n_bins];

    for &bin_idx in bins {
        hist[bin_idx] = hist[bin_idx] + F::one();
    }

    Ok(hist)
}

/// Compute correlation coefficient
#[allow(dead_code)]
fn compute_correlation<F: Float + std::iter::Sum>(
    x: &scirs2_core::ndarray::ArrayView1<F>,
    y: &Array1<F>,
) -> Result<F> {
    let n = F::from(x.len()).unwrap();

    let mean_x = x.iter().cloned().sum::<F>() / n;
    let mean_y = y.iter().cloned().sum::<F>() / n;

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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_feature_importance_calculator() {
        let calculator = FeatureImportanceCalculator::<f64>::new()
            .with_permutations(10)
            .with_seed(42);

        assert_eq!(calculator.n_permutations, 10);
        assert_eq!(calculator.random_seed, Some(42));
    }

    #[test]
    fn test_drop_column() {
        let calculator = FeatureImportanceCalculator::<f64>::new();
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let result = calculator.drop_column(&data, 1).unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 3.0);
        assert_eq!(result[[1, 0]], 4.0);
        assert_eq!(result[[1, 1]], 6.0);
    }

    #[test]
    fn test_importance_statistics() {
        let calculator = FeatureImportanceCalculator::<f64>::new();
        let mut scores = HashMap::new();
        scores.insert("feature1".to_string(), 0.5);
        scores.insert("feature2".to_string(), 0.3);
        scores.insert("feature3".to_string(), 0.8);
        scores.insert("feature4".to_string(), 0.1);

        let stats = calculator.compute_importance_statistics(&scores);

        assert_eq!(stats.n_features, 4);
        assert!((stats.mean - 0.425).abs() < 1e-10);
        assert_eq!(stats.min, 0.1);
        assert_eq!(stats.max, 0.8);
    }

    #[test]
    fn test_top_features() {
        let calculator = FeatureImportanceCalculator::<f64>::new();
        let mut scores = HashMap::new();
        scores.insert("feature1".to_string(), 0.5);
        scores.insert("feature2".to_string(), 0.3);
        scores.insert("feature3".to_string(), 0.8);
        scores.insert("feature4".to_string(), 0.1);

        let top_features = calculator.get_top_features(&scores, 2);

        assert_eq!(top_features.len(), 2);
        assert_eq!(top_features[0].0, "feature3");
        assert_eq!(top_features[0].1, 0.8);
        assert_eq!(top_features[1].0, "feature1");
        assert_eq!(top_features[1].1, 0.5);
    }

    #[test]
    fn test_threshold_filtering() {
        let calculator = FeatureImportanceCalculator::<f64>::new();
        let mut scores = HashMap::new();
        scores.insert("feature1".to_string(), 0.5);
        scores.insert("feature2".to_string(), 0.3);
        scores.insert("feature3".to_string(), 0.8);
        scores.insert("feature4".to_string(), 0.1);

        let filtered = calculator.filter_by_threshold(&scores, 0.4);

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("feature1"));
        assert!(filtered.contains_key("feature3"));
    }

    #[test]
    fn test_correlation_computation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let correlation = compute_correlation(&x.view(), &y).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_importance() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];

        let importance = mutual_information_importance(&x, &y, &feature_names).unwrap();

        assert_eq!(importance.len(), 2);
        assert!(importance.contains_key("feature1"));
        assert!(importance.contains_key("feature2"));
    }

    #[test]
    fn test_feature_importance_calculator_with_new_options() {
        let calculator = FeatureImportanceCalculator::<f64>::new()
            .with_permutations(50)
            .with_seed(123)
            .with_shap_background(200)
            .with_proper_rng(true)
            .with_simd(true);

        assert_eq!(calculator.n_permutations, 50);
        assert_eq!(calculator.random_seed, Some(123));
        assert_eq!(calculator.n_shap_background, 200);
        assert!(calculator.use_proper_rng);
        assert!(calculator.enable_simd);
    }

    #[test]
    fn test_fisher_yates_shuffle() {
        let calculator = FeatureImportanceCalculator::<f64>::new().with_seed(42);
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let original = values.clone();

        calculator.fisher_yates_shuffle(&mut values).unwrap();

        // Values should be different order (with high probability)
        // but contain the same elements
        let mut sorted_original = original;
        sorted_original.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut sorted_shuffled = values;
        sorted_shuffled.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(sorted_original, sorted_shuffled);
    }

    #[test]
    fn test_shap_importance() {
        let calculator = FeatureImportanceCalculator::<f64>::new()
            .with_shap_background(10)
            .with_seed(42);

        let x_test = array![[1.0, 2.0], [3.0, 4.0]];
        let x_background = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];

        // Mock model that returns sum of features
        let model = |x: &ArrayView2<f64>| x.map_axis(Axis(1), |row| row.sum());
        let score_fn = |y_true: &Array1<f64>, y_pred: &Array1<f64>| {
            y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(t, p)| (t - p).abs())
                .sum::<f64>()
                / y_true.len() as f64
        };

        let shap_values = calculator
            .shap_importance(&model, &x_test, &x_background, score_fn, &feature_names)
            .unwrap();

        assert_eq!(shap_values.len(), 2);
        assert!(shap_values.contains_key("feature1"));
        assert!(shap_values.contains_key("feature2"));
    }

    #[test]
    fn test_lime_importance() {
        let calculator = FeatureImportanceCalculator::<f64>::new().with_seed(42);
        let x_instance = array![1.0, 2.0];
        let x_background = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];

        let model = |x: &ArrayView2<f64>| x.map_axis(Axis(1), |row| row.sum());
        let score_fn = |y_true: &Array1<f64>, y_pred: &Array1<f64>| {
            y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(t, p)| (t - p).abs())
                .sum::<f64>()
                / y_true.len() as f64
        };

        let lime_scores = calculator
            .lime_importance(
                &model,
                &x_instance.view(),
                &x_background,
                score_fn,
                &feature_names,
                20,
            )
            .unwrap();

        assert_eq!(lime_scores.len(), 2);
        assert!(lime_scores.contains_key("feature1"));
        assert!(lime_scores.contains_key("feature2"));
    }

    #[test]
    fn test_integrated_gradients() {
        let calculator = FeatureImportanceCalculator::<f64>::new().with_seed(42);
        let x_instance = array![1.0, 2.0];
        let x_baseline = array![0.0, 0.0];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];

        let model = |x: &ArrayView2<f64>| x.map_axis(Axis(1), |row| row.sum());

        let ig_scores = calculator
            .integrated_gradients_importance(
                &model,
                &x_instance.view(),
                &x_baseline.view(),
                &feature_names,
                10,
            )
            .unwrap();

        assert_eq!(ig_scores.len(), 2);
        assert!(ig_scores.contains_key("feature1"));
        assert!(ig_scores.contains_key("feature2"));
    }

    #[test]
    fn test_gain_importance() {
        let calculator = FeatureImportanceCalculator::<f64>::new();
        let x_test = array![[1.0, 2.0], [3.0, 4.0]];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];

        let tree_splits = vec![
            TreeSplit {
                feature_idx: 0,
                threshold: 2.0,
                gain: 0.5,
                n_samples: 100,
            },
            TreeSplit {
                feature_idx: 1,
                threshold: 3.0,
                gain: 0.3,
                n_samples: 50,
            },
            TreeSplit {
                feature_idx: 0,
                threshold: 1.5,
                gain: 0.2,
                n_samples: 75,
            },
        ];

        let model = |x: &ArrayView2<f64>| x.map_axis(Axis(1), |row| row.sum());

        let gain_scores = calculator
            .gain_importance(&model, &x_test, &feature_names, &tree_splits)
            .unwrap();

        assert_eq!(gain_scores.len(), 2);
        assert_eq!(gain_scores["feature1"], 0.7); // 0.5 + 0.2
        assert_eq!(gain_scores["feature2"], 0.3);
    }

    #[test]
    #[allow(clippy::too_many_arguments)]
    fn test_comprehensive_feature_importance() {
        let calculator = FeatureImportanceCalculator::<f64>::new()
            .with_permutations(5)
            .with_shap_background(5)
            .with_seed(42);

        let x_test = array![[1.0, 2.0], [3.0, 4.0]];
        let y_test = array![3.0, 7.0];
        let x_background = array![[0.0, 0.0], [1.0, 1.0]];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];

        let model = |x: &ArrayView2<f64>| x.map_axis(Axis(1), |row| row.sum());
        let score_fn = |y_true: &Array1<f64>, y_pred: &Array1<f64>| {
            y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(t, p)| (t - p).abs())
                .sum::<f64>()
                / y_true.len() as f64
        };

        let results = comprehensive_feature_importance(
            &calculator,
            &model,
            &x_test,
            &y_test,
            Some(&x_background),
            score_fn,
            &feature_names,
            None,
        )
        .unwrap();

        assert_eq!(results.permutation_importance.len(), 2);
        assert!(results.shap_values.is_some());
        assert!(results.lime_importance.is_some());
        assert!(results.integrated_gradients.is_some());
        assert!(results.gain_importance.is_none());
        assert_eq!(results.mutual_information.len(), 2);
    }

    #[test]
    fn test_improved_mutual_information() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let mi = compute_mutual_information_improved(&x.view(), &y).unwrap();

        // For perfectly correlated data, MI should be high
        assert!(mi > 0.0);
    }

    #[test]
    fn test_bin_creation() {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let bins = create_bins(&values.view(), 3).unwrap();

        assert_eq!(bins.len(), 5);
        // All bin indices should be within range
        assert!(bins.iter().all(|&bin| bin < 3));
    }
}
